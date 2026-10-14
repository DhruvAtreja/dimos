# Copyright 2026 Dimensional Inc.
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

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import ConfigDict, with_config
from typing_extensions import NotRequired, TypedDict

# Under pytest (which exports PYTEST_VERSION, as dimos's .env rule uses) a field a response
# doesn't declare fails the test; a running gateway passes it through rather than drop it.
CONFIG = ConfigDict(extra="forbid" if "PYTEST_VERSION" in os.environ else "allow")


@with_config(CONFIG)
class Body(TypedDict):
    pass


class Info(Body):
    """The dimos checkout this gateway drives."""

    dir: str
    found: bool
    installed: bool
    version: str | None


class ServerInfo(Body):
    exe: str
    exeModified: int | None
    kind: Literal["dimos"]
    startedAt: int
    zenohNamespace: str | None


class Paths(Body):
    """Where dimos keeps its runs, logs and recordings, and which process serves this API."""

    dimosDir: str
    runsDir: str
    logsDirs: list[str]
    recordingsDir: str
    server: ServerInfo


class Stopping(Body):
    stopping: bool


class Python(Body):
    """The python that runs this dimos checkout."""

    python: str


class BlueprintEntry(Body):
    name: str
    kind: Literal["builtin", "external"]
    importable: bool | None
    import_error: str | None
    missing_module: str | None


class BlueprintList(Body):
    """Every blueprint dimos can run, and whether the discovery scan could import it."""

    blueprints: list[BlueprintEntry]


class Param(Body):
    name: str
    type: str | None
    default: str | None


class Method(Body):
    name: str
    params: list[Param]
    return_type: str | None
    doc: str


class Stream(Body):
    name: str
    type: str
    direction: str
    topic: str | None


BlueprintModule = TypedDict(
    "BlueprintModule",
    {
        "name": str,
        "class": str,
        "doc": str,
        "summary": str,
        "streams": list[Stream],
        "rpcs": list[Method],
        "skills": list[Method],
        "file": str | None,
        "line": int | None,
    },
)
with_config(CONFIG)(BlueprintModule)


class Blueprint(Body):
    """A blueprint's modules: their streams (topic, type, direction), RPCs and skills."""

    name: str
    file: str | None
    line: int | None
    modules: list[BlueprintModule]


class ConfigArg(Body):
    name: str
    type: str | None
    default: Any
    description: str | None
    required: bool
    base: bool
    json_compatible: bool
    secret: bool
    choices: NotRequired[list[Any]]
    schema: NotRequired[dict[str, Any]]
    value: NotRequired[Any]


ModuleConfig = TypedDict(
    "ModuleConfig",
    {"module": str, "class": str, "args": list[ConfigArg], "error": NotRequired[str]},
)
with_config(CONFIG)(ModuleConfig)


class BlueprintConfig(Body):
    """A blueprint's configurable args per module, and the values Desktop saved for it."""

    name: str
    modules: list[ModuleConfig]
    overrides: dict[str, dict[str, Any]]


class GlobalConfig(Body):
    """dimos's GlobalConfig: its JSON schema, defaults, the saved overrides and which keys are secret."""

    schema: dict[str, Any]
    defaults: dict[str, Any]
    overrides: dict[str, Any]
    secrets: list[str]


class Robot(Body):
    name: str
    description: str
    type: str | None
    manufacturer: NotRequired[str | None]
    dirs: NotRequired[list[str]]
    recommended: list[str]
    blueprints: dict[str, RobotBlueprint]


class RobotBlueprint(Body):
    title: str
    description: str
    robot: str
    recommended_config: list[dict[str, Any]]
    starter: int | None
    hidden: bool
    registered: bool


class Robots(Body):
    """Every robot dimos supports and its blueprints, with each one's recommended settings."""

    about: None
    types: dict[str, dict[str, Any]]
    robots: dict[str, Robot]


ScannedModule = TypedDict(
    "ScannedModule", {"name": str, "class": str, "module": str, "streams": list[Stream]}
)
with_config(CONFIG)(ScannedModule)


class DiscoveredBlueprint(Body):
    name: str
    ref: str | None
    builtin: bool | None
    importable: bool
    optional_dependency: bool
    import_error: str | None
    missing_module: str | None
    modules: list[ScannedModule]
    robot: str | None
    suggested_extras: list[str]


class Discovered(Body):
    """What the discovery scan found for each blueprint, and the extras that would fix one that won't import."""

    stale: bool
    blueprints: list[DiscoveredBlueprint]


class DiscoveryStatus(Body):
    """The discovery scan's progress."""

    state: Literal["idle", "scanning", "done", "failed"]
    reason: str | None
    key: str | None
    stale: bool
    started_at: str | None
    finished_at: str | None
    current: str | None
    blueprints_total: int
    blueprints_done: int
    importable: int
    errors: list[str]


class CatalogBlueprint(Body):
    name: str
    ref: str | None
    robot: str | None
    modules: list[str]
    doc: str


class CatalogSkill(Body):
    name: str
    doc: str
    params: list[Param]
    module: str
    robots: list[str]


class PortType(Body):
    name: str
    type: str


CatalogModule = TypedDict(
    "CatalogModule",
    {
        "name": str,
        "class": str,
        "doc": str,
        "inputs": list[PortType],
        "outputs": list[PortType],
        "robots": list[str],
        "skills": list[str],
    },
)
with_config(CONFIG)(CatalogModule)


class Catalog(Body):
    """The importable built-in blueprints, every module and every skill, for browsing."""

    blueprints: list[CatalogBlueprint]
    modules: list[CatalogModule]
    skills: list[CatalogSkill]
    errors: list[str]


class Extra(Body):
    name: str
    installed: bool
    applicable: bool
    requires: list[str]
    includes: list[str]
    missing: list[str]
    download_bytes: int | None


class Extras(Body):
    """dimos's optional extras, and which this python has installed."""

    mode: Literal["checkout", "library"]
    python: str
    extras: list[Extra]


class JobStarted(Body):
    """The job running the install: follow it at /dimos/jobs/{job}/log."""

    shell: None
    job: str
    command: list[str]


class JobLog(Body):
    """A job's output from line `after` on, and whether it finished."""

    job: str
    title: str
    kind: str
    done: bool
    ok: bool | None
    started_at: str
    finished_at: str | None
    command: list[str]
    lines: list[str]
    next: int
    error: str | None
    failure: list[str]
    code: str | None


class GuidePage(Body):
    title: str
    source_path: str
    url: str | None


class CustomRobotGuide(Body):
    """dimos's guide to adding a robot, as markdown and HTML, with links made absolute."""

    title: str
    markdown: str
    html: str | None
    source_path: str
    url: str | None
    others: list[GuidePage]


class RunEntry(Body):
    run_id: str
    pid: int
    blueprint: str
    started_at: str
    log_dir: str


OneOff = TypedDict(
    "OneOff",
    {
        "global": dict[str, Any],
        "modules": dict[str, dict[str, Any]],
        "secrets": NotRequired[list[str]],
    },
)
with_config(CONFIG)(OneOff)


class Launch(Body):
    """The blueprint this gateway launched: its phase, output so far and the config it ran with."""

    blueprint: str
    phase: Literal["starting", "running", "stopping", "stopped", "failed"]
    startedAt: str
    pid: int
    output: str
    runId: str | None
    logDir: str | None
    error: str | None
    overrides: dict[str, Any]
    modules: dict[str, dict[str, Any]]
    oneOff: OneOff
    args: NotRequired[list[str]]


class Runs(Body):
    """The running blueprints, and the launch this gateway started."""

    runs: list[RunEntry]
    launch: Launch | None


class Stopped(Body):
    """Which run was stopped."""

    output: str


class LogRecord(Body):
    timestamp: str
    level: str
    logger: str
    event: str
    extra: dict[str, Any]
    raw: str


class RunLog(Body):
    """A run's structured log records, from offset `after` on."""

    runId: str
    records: list[LogRecord]
    offset: int
    loggers: list[str]


class Account(Body):
    """Whether this machine is logged in to Dimensional cloud, and as whom."""

    loggedIn: bool
    email: str | None
    scopes: str | list[str] | None
    source: Literal["env", "stored"] | None
    cloudUrl: str
    error: str | None


class Login(Body):
    """The device login: the URL and code to open, until it's done, expired or cancelled."""

    state: str
    url: str | None
    urlComplete: str | None
    code: str | None
    expiresAt: int | None
    email: str | None
    error: str | None


class Upload(Body):
    """One upload in the queue: its progress, and its link once done."""

    id: str
    path: str
    name: str
    size: int
    robotId: str | None
    kind: str | None
    state: str
    phase: str | None
    bytesDone: int
    bytesTotal: int
    rateBps: float | None
    etaSeconds: float | None
    uploadId: str | None
    skipped: bool
    notice: str | None
    error: str | None
    errorCode: str | None
    log: str | None
    createdAt: int
    startedAt: int | None
    finishedAt: int | None
    link: str | None


class Uploads(Body):
    """The upload queue, and whether it waits for a login."""

    uploads: list[Upload]
    waitingForLogin: bool


class Uploaded(Body):
    """A file uploaded before, and whether it changed since."""

    path: str
    uploadId: str
    size: int
    mtimeMs: int
    uploadedAt: int
    link: str | None
    changed: bool


class UploadedByPath(Body):
    """Every file uploaded before, by path."""

    byPath: dict[str, Uploaded]


class Ok(Body):
    ok: bool


class SkillRun(Body):
    runId: str | None
    blueprint: str | None


class Skill(Body):
    name: str
    module: str
    description: str
    params: dict[str, Any]
    required: list[str]
    lifecycle: str
    uses: list[str]
    runId: str | None
    blueprint: str | None


class SkillError(Body):
    module: str
    error: str


class Skills(Body):
    """The running blueprint's skills, with each one's JSON schema for its args."""

    skills: list[Skill]
    run: SkillRun | None
    errors: list[SkillError]


class SkillResult(Body):
    """What the skill answered, as text and as MCP content blocks."""

    skill: str
    module: str
    runId: str | None
    blueprint: str | None
    via: Literal["mcp", "rpc"]
    ok: bool
    text: str
    content: list[dict[str, Any]]
