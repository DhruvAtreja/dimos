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

"""Setuptools source packages from explicit, local message inputs.

Preparation of upstream sources is a separate action. No hook fetches sources,
installs tools, or invokes CMake/Cargo. The source distribution carries inputs.
"""

from __future__ import annotations

from importlib.util import find_spec
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any

from setuptools import find_packages
from setuptools.command.sdist import sdist

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from .definitions import parse_message
from .generate import generate
from .ownership import Dependency


def _local(project: Path, value: str) -> Path:
    path = (project / value).resolve()
    if not path.is_relative_to(project.resolve()):
        raise ValueError(f"Message input must be inside its source project: {value}")
    return path


def configuration(project: Path) -> dict[str, Any]:
    data = tomllib.loads((project / "pyproject.toml").read_text())
    settings = data["tool"]["dimos"]["messages"]
    source = _local(project, settings.get("source", "interfaces"))
    if not source.is_dir():
        raise ValueError(
            f"Message source is not prepared: {source}; prepare it explicitly before building"
        )
    module = settings.get("module", data["project"]["name"].replace("-", "_"))
    resources = [_local(project, value) for value in settings.get("resources", [])]
    names = settings.get("roots")
    if "catalog" in settings:
        if names is not None:
            raise ValueError("Select roots or catalog, not both")
        catalog_path = _local(project, settings["catalog"])
        catalog = json.loads(catalog_path.read_text())
        if catalog.get("format") != 1 or not catalog.get("roots"):
            raise ValueError("A nonempty explicit catalog is required")
        if any(
            not isinstance(reason, str) or not reason.strip()
            for reason in catalog["roots"].values()
        ):
            raise ValueError("Each catalog root requires a reason")
        names = list(catalog["roots"])
        resources.append(catalog_path)
    if names is None:
        names = [parse_message(path).name for path in sorted(source.rglob("*.msg"))]
    if not names:
        raise ValueError("A message package requires at least one root")
    dependencies = []
    for name, version in sorted(settings.get("dependencies", {}).items()):
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"Invalid message dependency module: {name}")
        spec = find_spec(name + "_schemas")
        if spec is None or spec.origin is None:
            raise ValueError(f"Message build dependency is not installed: {name}")
        dependency = Dependency.load(Path(spec.origin).parent / "package")
        if dependency.module != name or dependency.version != version:
            raise ValueError(f"Message dependency identity/version mismatch: {name}=={version}")
        dependencies.append(dependency)
    for resource in resources:
        if not resource.is_file():
            raise ValueError(f"Message resource is not prepared: {resource}")
    if len({resource.name for resource in resources}) != len(resources):
        raise ValueError("Message resource basenames must be unique")
    return {
        "module": module,
        "version": data["project"]["version"],
        "source": source,
        "roots": names,
        "resources": resources,
        "dependencies": tuple(dependencies),
    }


def prepare(project: Path, output: Path | None = None) -> Path:
    config = configuration(project)
    output = output or project / "build/message-source"
    generate(
        [config["source"]],
        output,
        config["roots"],
        config["module"],
        version=config["version"],
        dependencies=config["dependencies"],
        shared=True,
        bundled=False,
    )
    resources = output / "python" / (config["module"] + "_schemas") / "package"
    for path in config["resources"]:
        shutil.copyfile(path, resources / path.name)
    return output / "python"


class SourceDistribution(sdist):
    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        project = Path.cwd()
        config = configuration(project)
        super().make_release_tree(
            base_dir, [name for name in files if not name.startswith("build/")]
        )
        destination = Path(base_dir)
        source = config["source"]
        target = destination / source.relative_to(project)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        for path in config["resources"]:
            target = destination / path.relative_to(project)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def setup_arguments(project: Path) -> dict[str, Any]:
    project = project.resolve()
    source = prepare(project)
    module = configuration(project)["module"]
    return {
        "packages": find_packages(str(source)),
        "package_dir": {"": str(source.relative_to(project))},
        "package_data": {
            module: ["py.typed", "*.pyi", "**/*.pyi"],
            module + "_schemas": ["schemas.json", "schemas/**/*", "package/**/*"],
        },
        "entry_points": {"dimos.messages": [module + "=" + module + "_schemas.provider"]},
        "cmdclass": {"sdist": SourceDistribution},
    }
