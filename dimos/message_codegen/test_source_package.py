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

"""Generic source-package builds preserve owners and work offline outside a checkout."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest

from dimos.message_codegen.source_package import configuration


def project(root, name, package, definition, dependency=False):
    root.mkdir()
    source = root / "interfaces" / package / "msg" / ("Reading.msg" if dependency else "Stamp.msg")
    source.parent.mkdir(parents=True)
    source.write_text(definition)
    requires = '["setuptools>=70", "wheel", "dimos-message-build==0.1.0"'
    if dependency:
        requires += ', "base-messages==0.1.0"'
    (root / "pyproject.toml").write_text(
        "[build-system]\nrequires = " + requires + ']\nbuild-backend = "setuptools.build_meta"\n'
        f'[project]\nname = "{name}"\nversion = "0.1.0"\n'
        'dependencies = ["dimos-message-build==0.1.0", "numpy>=1.26.4", "rosbags==0.11.0"'
        + (', "base-messages==0.1.0"' if dependency else "")
        + "]\n"
        '[tool.dimos.messages]\nsource = "interfaces"\nresources = ["pins.json"]\n'
        + ('[tool.dimos.messages.dependencies]\nbase_messages = "0.1.0"\n' if dependency else "")
    )
    (root / "pins.json").write_text('{"revision":"local-test-input"}\n')
    (root / "setup.py").write_text(
        "from pathlib import Path\nfrom setuptools import setup\n"
        "from dimos_message_build.source_package import setup_arguments\n"
        "setup(**setup_arguments(Path(__file__).resolve().parent))\n"
    )
    (root / "MANIFEST.in").write_text("include pyproject.toml setup.py\nprune build\n")


def test_unprepared_inputs_and_paths_outside_project_fail(tmp_path):
    root = tmp_path / "message-project"
    project(root, "base-messages", "shared_msgs", "int32 tick\nstring frame\n")
    shutil.rmtree(root / "interfaces")
    with pytest.raises(ValueError, match="not prepared"):
        configuration(root)
    metadata = root / "pyproject.toml"
    metadata.write_text(
        metadata.read_text().replace('source = "interfaces"', 'source = "../outside"')
    )
    with pytest.raises(ValueError, match="inside its source project"):
        configuration(root)


def test_generic_cold_source_wheel_sdist_editable_and_dependency_identity(tmp_path):
    available = os.environ.get("DIMOS_MESSAGE_WHEELHOUSE")
    if not available:
        if os.environ.get("DIMOS_MESSAGE_INSTALL_REQUIRED") == "1":
            pytest.fail("Prepare DIMOS_MESSAGE_WHEELHOUSE for offline package acceptance")
        pytest.skip("Offline package acceptance requires a prepared dependency wheelhouse")
    root = Path(__file__).resolve().parents[2]
    wheels = tmp_path / "wheels"
    shutil.copytree(available, wheels)
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment.update(
        PIP_NO_INDEX="1", PIP_FIND_LINKS=str(wheels), PIP_DISABLE_PIP_VERSION_CHECK="1"
    )

    def run(command, cwd=tmp_path):
        result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr

    backend_source = tmp_path / "backend"
    shutil.copytree(
        root / "dimos/message_codegen",
        backend_source,
        ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__"),
    )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheels),
            str(backend_source),
        ]
    )
    backend = next(wheels.glob("dimos_message_build-*.whl"))
    with zipfile.ZipFile(backend) as archive:
        assert not any(name.endswith(".msg") for name in archive.namelist())
    base, application = tmp_path / "base", tmp_path / "application"
    project(base, "base-messages", "shared_msgs", "int32 tick\nstring frame\n")
    project(
        application, "app-messages", "app_msgs", "shared_msgs/Stamp stamp\nfloat64 value\n", True
    )
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    for tool in ["cc", "c++", "cmake", "cargo", "rustc", "rustup"]:
        path = blocked / tool
        path.write_text("#!/bin/sh\nexit 97\n")
        path.chmod(0o755)
    environment.update(
        PATH=str(blocked) + os.pathsep + environment["PATH"],
        CC=str(blocked / "cc"),
        CXX=str(blocked / "c++"),
    )
    run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(wheels), str(base)])
    venv = tmp_path / "venv"
    run([sys.executable, "-m", "venv", str(venv)])
    python = str(venv / "bin/python")
    run([python, "-m", "pip", "install", str(application)])
    check = """
import pickle
from base_messages.shared_msgs.msg import Stamp
from app_messages.app_msgs.msg import Reading
from dimos_message_build.registry import encode, decode, schema
from app_messages_schemas.provider import package_root
from pathlib import Path
m = Reading(Stamp(17, 'map'), 21.5)
assert type(m.stamp) is Stamp
assert type(decode(encode(m), Reading).stamp) is Stamp
assert type(pickle.loads(pickle.dumps(m)).stamp) is Stamp
assert 'MSG: shared_msgs/Stamp' in schema(m.__msgtype__)
assert (Path(package_root())/'pins.json').is_file()
"""
    run([python, "-I", "-c", check])
    run([python, "setup.py", "sdist", "--dist-dir", str(tmp_path / "sdists")], cwd=application)
    archive = next((tmp_path / "sdists").glob("*.tar.gz"))
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
        assert any(name.endswith("interfaces/app_msgs/msg/Reading.msg") for name in names)
        assert any(name.endswith("pins.json") for name in names)
        assert not any("/build/" in name for name in names)
    run([python, "-m", "pip", "install", "--force-reinstall", str(archive)])
    run([python, "-I", "-c", check])
    run(
        [
            python,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(tmp_path / "output"),
            str(archive),
        ]
    )
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            str(next((tmp_path / "output").glob("app_messages-*.whl"))),
        ]
    )
    run([python, "-I", "-c", check])
    run([python, "-m", "pip", "install", "-e", str(application)])
    run([python, "-I", "-c", check])
    source = application / "interfaces/app_msgs/msg/Reading.msg"
    source.write_text("shared_msgs/Stamp stamp\nfloat32 value\n")
    run([python, "-m", "pip", "install", "-e", str(application)])
    run([python, "-I", "-c", check + "\nassert 'float32 value' in schema(m.__msgtype__)\n"])
