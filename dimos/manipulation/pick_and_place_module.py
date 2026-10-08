# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capability-composed pick-and-place workflow."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import time
from typing import Any, Literal

import numpy as np
from pydantic import Field

from dimos.agents.annotation import skill
from dimos.agents.capabilities import CAP_MOVEMENT
from dimos.agents.skill_result import SkillResult
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.manipulation.grasp_verification import (
    GraspVerificationConfig,
    GripperSettle,
    await_gripper_settle,
    grasp_failure,
    open_failure,
)
from dimos.manipulation.grasping.grasp_gen_spec import GraspGenSpec
from dimos.manipulation.manipulation_spec import (
    ExecutionResult,
    ManipulationSpec,
    PlanningGroupInfo,
    PlanResult,
)
from dimos.manipulation.planning.spec.models import PlanningGroupID
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.manipulation_msgs.GraspCandidateArray import GraspCandidateArray
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.vision_msgs.Detection3DArray import Detection3DArray
from dimos.perception.experimental.object_scene_registration_spec import ObjectSceneRegistrationSpec

# The one box of mapped cells a pick asks the planner to ignore.
VOXEL_MAP_EXCLUSION_ID = "pick-and-place/target"


class PickAndPlaceModuleConfig(ModuleConfig):
    planning_frame: str = "base_link"
    pregrasp_offset: float = Field(default=0.10, gt=0.0)
    # Margin (metres) added around the target's bounding box. Mapped cells in
    # that box are not obstacles during the pick, and the same box rides with
    # the gripper while the object is held. Too small, and a cell on the
    # object's edge or the table under it stops the plan; too large, and a
    # neighbour's cells are ignored too.
    target_clearance: float = Field(default=0.03, gt=0.0)
    # A learned provider returns a ranked spread whose best-scoring pose is not
    # always kinematically reachable; a single-candidate provider is unaffected.
    max_grasp_attempts: int = Field(default=5, gt=0)
    yaw_policy: Literal["generated", "preserve_current"] = "generated"
    grasp_verification: GraspVerificationConfig = Field(default_factory=GraspVerificationConfig)
    # A target still seen within this distance (metres) of where it was scanned
    # never left the table, whatever the jaws report.
    grasp_displacement_tolerance: float = Field(default=0.03, gt=0.0)
    # How far (metres, horizontally) a released object may rest from the requested point.
    place_tolerance: float = Field(default=0.07, gt=0.0)
    # How long (seconds) a pick or place waits for the manipulation module to list
    # its planning groups and receive the arm's first joint state.
    robot_ready_timeout: float = Field(default=10.0, gt=0.0)
    robot_ready_poll_interval: float = Field(default=0.25, gt=0.0)


@dataclass(frozen=True)
class ScannedObject:
    """One object from the latest scan: its id, label and centre in the planning frame."""

    object_id: str
    name: str
    position: Vector3


@dataclass(frozen=True)
class TargetBox:
    """Axis-aligned bounds of a target's point cloud, planning frame, metres."""

    center: Vector3
    size: Vector3


@dataclass(frozen=True)
class HeldObject:
    """The object a pick left in the gripper, and where it was taken from."""

    object_id: str
    name: str
    grasp: PoseStamped
    picked_from: Vector3


def _status(result: PlanResult | ExecutionResult) -> str:
    """Status name and message of a planner or execution result, e.g. 'FAILED: no path'."""
    return f"{result.status.name}: {result.message}" if result.message else result.status.name


def _gripper_reading(settle: GripperSettle) -> str:
    """One sentence saying where the jaws stopped after a gripper command."""
    if settle.position is None:
        return "No gripper position readback."
    if not settle.settled:
        return (
            f"Gripper position {settle.position:.2f} had not settled after {settle.elapsed:.1f} s."
        )
    return f"Final gripper position {settle.position:.2f}."


class PickAndPlaceModule(Module):
    """Coordinate scene registration, grasp generation, and manipulation execution."""

    config: PickAndPlaceModuleConfig
    _scene: ObjectSceneRegistrationSpec
    _grasp_generator: GraspGenSpec
    _manipulation: ManipulationSpec

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._objects: dict[str, ScannedObject] = {}
        self._grasp_candidates = GraspCandidateArray()
        self._held: HeldObject | None = None

    @skill
    def scan_objects(self, prompts: list[str]) -> SkillResult:
        """Scan the latest RGB-D frame for prompted objects.

        Args:
            prompts: Object descriptions to detect, such as ["red cup", "bottle"].
        """
        prompts = [prompt.strip() for prompt in prompts if prompt.strip()]
        if not prompts:
            raise ValueError("At least one object prompt is required")
        if self._held is None:
            self._clear_proposals()
        self._objects = {}
        detections = self._scene.scan_scene(text=prompts)
        self._objects = {
            str(detection.id): ScannedObject(
                object_id=str(detection.id),
                name=str(detection.results[0].hypothesis.class_id),
                position=_center(detection),
            )
            for detection in detections.detections
            if detection.id and detection.results
        }
        return SkillResult(
            f"Detected {detections.detections_length} object(s)",
            metadata={
                "prompts": prompts,
                "objects": [
                    {"object_id": obj.object_id, "name": obj.name} for obj in self._objects.values()
                ],
            },
        )

    @rpc
    def get_object(self, object_id: str) -> dict[str, Any] | None:
        obj = self._objects.get(object_id)
        return None if obj is None else {"object_id": obj.object_id, "name": obj.name}

    @rpc
    def get_held_object(self) -> dict[str, Any] | None:
        """The object the gripper holds after a verified pick, or None."""
        held = self._held
        return None if held is None else {"object_id": held.object_id, "name": held.name}

    @skill(uses=[CAP_MOVEMENT])
    def pick_object(
        self, object_id: str, planning_group: PlanningGroupID | None = None
    ) -> SkillResult:
        """Grasp one object from the latest scan and lift it clear of the surface.

        "Pick complete" means the camera confirmed it: after the lift the object is
        no longer seen where it was scanned. Any other answer says why the object
        is not held; re-scan and try again.

        Args:
            object_id: Exact object ID returned by the latest scan_objects call.
            planning_group: Gripper-capable pose group; omitted only when unambiguous.
        """
        if self._held is not None:
            return SkillResult(
                f"Still holding {self._held.name} (object {self._held.object_id}); "
                f"did not start a pick of object {object_id}. Use place_at to put it down first."
            )
        self._clear_proposals()
        target = self._objects.get(object_id)
        if target is None:
            scanned = ", ".join(self._objects) or "none"
            return SkillResult(
                f"No object with id {object_id} in the latest scan. Scanned ids: {scanned}. "
                "Use scan_objects to refresh the list."
            )
        group = self._await_group(planning_group)
        pointcloud = self._scene.get_object_pointcloud_by_object_id(object_id)
        if pointcloud is None:
            return SkillResult(
                f"Object {object_id} has no point cloud in the latest scan. Use scan_objects again."
            )
        candidates = self._grasp_generator.propose_grasps(pointcloud)
        self._grasp_candidates = candidates
        self._manipulation.show_grasp_proposals(candidates)
        if candidates.header.frame_id != self.config.planning_frame:
            raise RuntimeError(
                f"Grasp candidates are in frame {candidates.header.frame_id!r}; "
                f"the planning frame is {self.config.planning_frame!r}"
            )
        if not candidates.candidates:
            return SkillResult(f"Generated 0 grasp candidates for object {object_id}.")
        box = _target_box(pointcloud)
        # The target is mapped geometry: with its cells as obstacles the planner
        # refuses the approach into it, and the held object blocks every move after.
        self._exclude_target(box)
        try:
            return self._pick_from(candidates, target, box, group)
        finally:
            if self._held is None:
                self._manipulation.clear_voxel_map_exclusion(VOXEL_MAP_EXCLUSION_ID)

    def _pick_from(
        self,
        candidates: GraspCandidateArray,
        target: ScannedObject,
        box: TargetBox | None,
        group: PlanningGroupID,
    ) -> SkillResult:
        object_id = target.object_id
        if not_open := self._open_gripper(group, "before grasping"):
            return not_open

        last_plan = ""
        for rank, candidate in enumerate(candidates.candidates[: self.config.max_grasp_attempts]):
            grasp = self._apply_yaw_policy(
                PoseStamped(
                    ts=candidates.header.timestamp,
                    frame_id=candidates.header.frame_id,
                    position=candidate.pose.position,
                    orientation=candidate.pose.orientation,
                ),
                group,
            )
            pregrasp = self._offset_pose(grasp, self.config.pregrasp_offset)
            blocked = self._move(pregrasp, group) or self._servo(pregrasp, grasp, group)
            if isinstance(blocked, PlanResult):
                # The planner found no path to this candidate; the next one may
                # differ. A motion that stopped part-way would stop the same way
                # for every candidate, so that is not retried.
                last_plan = _status(blocked)
                continue
            if blocked is not None:
                return self._stopped(
                    f"Move to grasp candidate {rank} for object {object_id}", blocked
                )
            if not_held := self._close_and_verify(group, object_id):
                # Back out while the target's cells are still ignored: left at
                # the grasp pose, the arm would stand inside the obstacle the
                # target becomes again, and no later plan could start.
                self._servo(grasp, pregrasp, group)
                return not_held

            # The jaws stopped on something; the camera has the last word after the lift.
            held = HeldObject(object_id, target.name, grasp, target.position)
            self._held = held
            self._carry_target(box, grasp, group)
            if blocked := self._servo(grasp, pregrasp, group):
                return self._stopped(f"Retract after grasping object {object_id}", blocked)
            if not_lifted := self._verify_lift(held, group):
                return not_lifted
            return SkillResult(
                "Pick complete",
                metadata={
                    "object_id": object_id,
                    "name": target.name,
                    "rank": rank,
                    "score": candidate.score,
                    "candidates": len(candidates.candidates),
                },
            )
        attempted = min(len(candidates.candidates), self.config.max_grasp_attempts)
        return SkillResult(
            f"The planner found no path to any of the {attempted} grasp candidate(s) tried "
            f"for object {object_id}; last planner result {last_plan}."
        )

    @rpc
    def get_grasp_candidates(self) -> GraspCandidateArray:
        return self._grasp_candidates

    @skill(uses=[CAP_MOVEMENT])
    def place_at(
        self,
        x: float,
        y: float,
        z: float,
        planning_group: PlanningGroupID | None = None,
    ) -> SkillResult:
        """Lower the held object to a planning-frame position and release it.

        "Place complete" means the camera found the released object within the
        place tolerance of the requested point; otherwise the answer says how far
        from it the object ended.

        Args:
            x: Planning-frame X coordinate in meters.
            y: Planning-frame Y coordinate in meters.
            z: Planning-frame Z coordinate in meters.
            planning_group: Gripper-capable pose group; omitted only when unambiguous.
        """
        target = f"({x:.2f}, {y:.2f}, {z:.2f})"
        held = self._held
        if held is None:
            return SkillResult(
                f"Not holding any object; nothing was placed at {target}. Use pick_object first."
            )
        group = self._await_group(planning_group)
        if lost := self._lost_in_transit(held, group, target):
            return lost
        place = PoseStamped(
            frame_id=self.config.planning_frame,
            position=Vector3(x, y, z),
            orientation=held.grasp.orientation,
        )
        preplace = self._offset_pose(place, self.config.pregrasp_offset)
        if blocked := self._move(preplace, group):
            return self._stopped(f"Move to the pre-place pose above {target}", blocked)
        if blocked := self._servo(preplace, place, group):
            return self._stopped(f"Move down to the place pose {target}", blocked)
        if not_open := self._open_gripper(group, "to release the object"):
            return SkillResult(f"{not_open.message} The arm stayed at the place pose.")
        self._held = None
        self._clear_proposals()
        self._manipulation.clear_voxel_map_exclusion(VOXEL_MAP_EXCLUSION_ID)
        tip = self._tip_position(group)
        released_at = place.position if tip is None else tip
        if blocked := self._servo(place, preplace, group):
            return self._stopped(f"Retract from {target} after releasing", blocked)
        return self._verify_place(held, place.position, released_at)

    def _verify_lift(self, held: HeldObject, planning_group: PlanningGroupID) -> SkillResult | None:
        """Confirm the lifted object came along; None when it did, else what happened.

        A close that stops early on an edge reads as a hold, so the jaws are only a first
        filter. The wrist camera then looks down at where the object was: if the target
        is still detected there, the grasp missed it. Raises RuntimeError when the scan
        fails.
        """
        jaws = self._gripper_position(planning_group)
        if jaws is not None and jaws <= self.config.grasp_verification.held_low:
            return self._grasp_failed(
                planning_group,
                f"{held.name} slipped out of the gripper during the lift "
                f"(gripper position {jaws:.3f})",
            )
        try:
            detections = self._scene.scan_scene(text=[held.name])
        except RuntimeError as exc:
            raise RuntimeError(
                f"Lifted {held.name} but could not check whether it came along: {exc}"
            ) from exc
        tolerance = self.config.grasp_displacement_tolerance
        if any(center.distance(held.picked_from) <= tolerance for center in _centers(detections)):
            return self._grasp_failed(
                planning_group,
                f"{held.name} is still on the table at its original position after the lift",
            )
        return None

    def _grasp_failed(self, planning_group: PlanningGroupID, what: str) -> SkillResult:
        """Drop the hold on a pick that did not take the object along, and say so."""
        self._held = None
        self._clear_proposals()
        reopened = self._open_gripper(planning_group, "after the failed grasp")
        reopen = "Reopened the gripper." if reopened is None else reopened.message
        return SkillResult(f"Pick not complete: {what}. {reopen} Re-scan and try again.")

    def _lost_in_transit(
        self, held: HeldObject, planning_group: PlanningGroupID, target: str
    ) -> SkillResult | None:
        """None while the jaws still hold something, else a statement that the object is gone."""
        jaws = self._gripper_position(planning_group)
        if jaws is None or jaws > self.config.grasp_verification.held_low:
            return None
        self._held = None
        self._clear_proposals()
        self._manipulation.clear_voxel_map_exclusion(VOXEL_MAP_EXCLUSION_ID)
        reopened = self._open_gripper(planning_group, "after finding it empty")
        reopen = "Reopened the gripper." if reopened is None else reopened.message
        return SkillResult(
            f"{held.name} is no longer in the gripper (gripper position {jaws:.3f}); "
            f"nothing was placed at {target}. {reopen} Re-scan and pick it again."
        )

    def _verify_place(self, held: HeldObject, target: Vector3, released_at: Vector3) -> SkillResult:
        """Report where the released object rests relative to the requested point.

        Raises RuntimeError when the scan fails.
        """
        try:
            detections = self._scene.scan_scene(text=[held.name])
        except RuntimeError as exc:
            raise RuntimeError(
                f"Released {held.name} but could not check where it landed: {exc}"
            ) from exc
        centers = _centers(detections)
        if centers:
            rest = min(centers, key=lambda center: _horizontal_distance(center, target))
            verified_by = "detection"
        else:
            # Right under the gripper the object can hide behind the fingers; where
            # the jaws opened is then the best estimate of where it rests.
            rest, verified_by = released_at, "release_pose"
        offset = _horizontal_distance(rest, target)
        metadata = {
            "object_id": held.object_id,
            "name": held.name,
            "offset_m": round(offset, 3),
            "verified_by": verified_by,
        }
        if offset <= self.config.place_tolerance:
            return SkillResult("Place complete", metadata=metadata)
        return SkillResult(
            f"Released {held.name}, which ended about {offset * 100:.0f} cm from the target, "
            f"farther than the {self.config.place_tolerance * 100:.0f} cm place tolerance.",
            metadata=metadata,
        )

    def _clear_proposals(self) -> None:
        self._grasp_candidates = GraspCandidateArray()
        self._manipulation.show_grasp_proposals(GraspCandidateArray())

    def _exclude_target(self, box: TargetBox | None) -> None:
        """Keep the target's own cells, and the table right under it, out of the planner."""
        if box is None:
            return
        self._manipulation.set_voxel_map_exclusion(
            VOXEL_MAP_EXCLUSION_ID, box.center, self._padded(box.size)
        )

    def _carry_target(
        self, box: TargetBox | None, grasp: PoseStamped, group: PlanningGroupID
    ) -> None:
        """Move the exclusion from the table to the gripper, where the object now is."""
        if box is None:
            return
        # The box centre relative to the tool tip, in the tip's own frame.
        offset = grasp.orientation.inverse().rotate_vector(box.center - grasp.position)
        self._manipulation.set_voxel_map_exclusion(
            VOXEL_MAP_EXCLUSION_ID, offset, self._padded(box.size), planning_group=group
        )

    def _padded(self, size: Vector3) -> Vector3:
        margin = 2.0 * self.config.target_clearance
        return Vector3(size.x + margin, size.y + margin, size.z + margin)

    def _await_group(self, planning_group: PlanningGroupID | None) -> PlanningGroupID:
        """Pick the gripper group once the manipulation module and the arm are up.

        Modules start in parallel, so an early call can land before the manipulation
        module has loaded its robot model or received the arm's first joint state.

        Args:
            planning_group: Group ID to use, or None to use the only gripper-capable group.

        Raises ValueError when the ID names no gripper-capable group, or when it is
        omitted and there is not exactly one. Raises TimeoutError when the groups or
        the arm's joint state do not show up within ``robot_ready_timeout``.
        """
        deadline = time.monotonic() + self.config.robot_ready_timeout
        while True:
            groups = self._manipulation.list_planning_groups()
            if groups:
                group = self._gripper_group(groups, planning_group)
                state = self._manipulation.get_state().groups.get(group)
                if state is not None and state.joints is not None:
                    return group
                waiting_for = "the arm's first joint state"
            else:
                waiting_for = "the manipulation module to list its planning groups"
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Waited {self.config.robot_ready_timeout:.0f} s for {waiting_for}; "
                    "is the arm driver running?"
                )
            time.sleep(self.config.robot_ready_poll_interval)

    @staticmethod
    def _gripper_group(
        groups: Sequence[PlanningGroupInfo], planning_group: PlanningGroupID | None
    ) -> PlanningGroupID:
        """Pick the planning group that has both a gripper and a tool frame.

        Args:
            groups: The manipulation module's planning groups.
            planning_group: Group ID to use, or None to use the only such group.

        Raises ValueError when the ID names no such group, or when it is omitted
        and there is not exactly one.
        """
        gripper_groups = [
            group.id for group in groups if group.has_gripper and group.tip_frame is not None
        ]
        if planning_group is None:
            if len(gripper_groups) == 1:
                return gripper_groups[0]
            raise ValueError(
                "Expected exactly one gripper-capable planning group when planning_group "
                f"is omitted; found {gripper_groups}"
            )
        if planning_group not in gripper_groups:
            raise ValueError(
                f"planning_group {planning_group!r} is not gripper-capable; "
                f"gripper-capable groups: {gripper_groups}"
            )
        return planning_group

    @staticmethod
    def _stopped(step: str, result: PlanResult | ExecutionResult) -> SkillResult:
        """Report a motion step that did not finish, and what stopped it.

        Args:
            step: The motion that was attempted, as the start of a sentence.
            result: The planner or execution result that did not succeed.
        """
        source = "planner" if isinstance(result, PlanResult) else "execution"
        return SkillResult(f"{step} did not complete; {source} returned {_status(result)}")

    def _apply_yaw_policy(self, pose: PoseStamped, group: PlanningGroupID) -> PoseStamped:
        if self.config.yaw_policy == "generated":
            return pose
        current = self._manipulation.get_state().groups[group].end_effector_pose
        if current is None:
            return pose
        euler = pose.orientation.to_euler()
        current_euler = current.orientation.to_euler()
        return PoseStamped(
            ts=pose.ts,
            frame_id=pose.frame_id,
            position=pose.position,
            orientation=Quaternion.from_euler(Vector3(euler.x, euler.y, current_euler.z)),
        )

    @staticmethod
    def _offset_pose(pose: PoseStamped, offset: float) -> PoseStamped:
        return PoseStamped(
            ts=pose.ts,
            frame_id=pose.frame_id,
            position=pose.position + pose.orientation.rotate_vector(Vector3(0.0, 0.0, -offset)),
            orientation=pose.orientation,
        )

    def _servo(
        self, start: PoseStamped, end: PoseStamped, planning_group: PlanningGroupID
    ) -> PlanResult | ExecutionResult | None:
        """Drive the last leg as a straight line with collision checking off.

        The object being grasped is itself mapped geometry once a voxel map feeds
        the planner, so a collision-checked plan into it can only ever be
        rejected. This leg is short, straight, and deliberately ends in contact.

        Returns the planner or execution result that stopped the leg, or None
        when the arm arrived.
        """
        result = self._manipulation.move_linear(
            end.position.x - start.position.x,
            end.position.y - start.position.y,
            end.position.z - start.position.z,
            planning_group,
            check_collision=False,
        )
        if not result.plan.succeeded:
            return result.plan
        if result.execution is None:
            raise RuntimeError("Linear move was planned but never executed")
        return None if result.execution.succeeded else result.execution

    def _move(
        self, pose: PoseStamped, planning_group: PlanningGroupID
    ) -> PlanResult | ExecutionResult | None:
        """Plan a collision-checked path to ``pose`` and run it.

        Returns the planner or execution result that stopped the leg, or None
        when the arm arrived.
        """
        plan = self._manipulation.plan_to_poses({planning_group: pose})
        if not plan.succeeded:
            return plan
        execution = self._manipulation.execute(blocking=True)
        return None if execution.succeeded else execution

    def _command_and_settle(
        self,
        position: float,
        planning_group: PlanningGroupID,
        arrival_tolerance: float | None = None,
    ) -> GripperSettle:
        """Command the gripper to ``position`` and wait for its readback to stop moving.

        Args:
            position: Target opening, 0.0 fully closed to 1.0 fully open.
            planning_group: Group whose gripper to command.
            arrival_tolerance: How close to ``position`` counts as arrived, in the
                same 0.0 to 1.0 units; None uses the settle tolerance.

        Raises RuntimeError when the gripper command is not accepted.
        """
        result = self._manipulation.set_gripper_position(position, planning_group)
        if not result.succeeded:
            raise RuntimeError(
                f"Gripper command to {position:.2f} on {planning_group} was not accepted: "
                f"{result.message}"
            )
        return await_gripper_settle(
            lambda: self._gripper_position(planning_group),
            position,
            self.config.grasp_verification,
            arrival_tolerance=arrival_tolerance,
        )

    def _open_gripper(self, planning_group: PlanningGroupID, step: str) -> SkillResult | None:
        """Open the gripper and wait for it to settle.

        Args:
            planning_group: Group whose gripper to open.
            step: Why it is being opened, completing "Commanded the gripper open ...".

        Returns None when the jaws reached open (or there is no readback), else a
        statement of where they stopped.
        """
        # Jaws resting against the open stop never move and never reach the
        # commanded extreme; open_tolerance is the band that already decides
        # whether where they stopped counts as open.
        settle = self._command_and_settle(
            self.config.grasp_verification.open_position,
            planning_group,
            arrival_tolerance=self.config.grasp_verification.open_tolerance,
        )
        if settle.position is None or open_failure(settle, self.config.grasp_verification) is None:
            return None
        return SkillResult(f"Commanded the gripper open {step}. {_gripper_reading(settle)}")

    def _close_and_verify(
        self, planning_group: PlanningGroupID, object_id: str
    ) -> SkillResult | None:
        """Close the gripper on the object and check the jaws stopped on something.

        Args:
            planning_group: Group whose gripper to close.
            object_id: ID of the object being grasped, for the report.

        Returns None when an object is held, else a statement of where the jaws
        stopped. Raises RuntimeError when verification is on but the gripper
        gives no position readback.
        """
        config = self.config.grasp_verification
        settle = self._command_and_settle(config.closed_position, planning_group)
        if not config.enabled:
            return None
        if settle.position is None:
            raise RuntimeError(
                f"No gripper position readback from {planning_group} "
                f"after closing on object {object_id}"
            )
        failure = grasp_failure(settle, config)
        if failure is None:
            return None
        report = f"Closed the gripper on object {object_id}. {_gripper_reading(settle)}"
        if "nothing in the jaws" in failure:
            reopened = self._open_gripper(planning_group, "after closing on nothing")
            report = f"{report} {'Reopened the gripper.' if reopened is None else reopened.message}"
        return SkillResult(report)

    def _gripper_position(self, planning_group: PlanningGroupID) -> float | None:
        state = self._manipulation.get_state().groups.get(planning_group)
        return state.gripper_position if state is not None else None

    def _tip_position(self, planning_group: PlanningGroupID) -> Vector3 | None:
        state = self._manipulation.get_state().groups.get(planning_group)
        if state is None or state.end_effector_pose is None:
            return None
        return state.end_effector_pose.position


def _target_box(pointcloud: PointCloud2) -> TargetBox | None:
    """The axis-aligned bounds of an object's points, or None for an empty cloud."""
    points = np.asarray(pointcloud.points_f32(), dtype=np.float64).reshape((-1, 3))
    if not len(points) or not np.isfinite(points).all():
        return None
    low, high = points.min(axis=0), points.max(axis=0)
    return TargetBox(Vector3(*((low + high) / 2.0)), Vector3(*(high - low)))


def _center(detection: Any) -> Vector3:
    position = detection.bbox.center.position
    return Vector3(position.x, position.y, position.z)


def _centers(detections: Detection3DArray) -> list[Vector3]:
    return [_center(detection) for detection in detections.detections]


def _horizontal_distance(a: Vector3, b: Vector3) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)
