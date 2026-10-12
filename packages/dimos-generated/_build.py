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

"""Generate package sources locally, without fetching or compiling native code."""

from pathlib import Path
import shutil

from dimos_message_build.builtin import generate_builtin
from setuptools.command.sdist import sdist


def canonical(project: Path) -> Path:
    source = project / "canonical"
    if not source.is_dir():
        source = project.parents[1] / "dimos/message_codegen/schemas"
    if not source.is_dir():
        raise RuntimeError("Canonical message sources are missing from this source distribution")
    return source


def prepare(project: Path, version: str) -> Path:
    output = project / "build/message-source"
    generate_builtin(output, version, canonical(project))
    return output / "python"


class SourceDistribution(sdist):
    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        # Generated outputs belong in wheels, not in the source input inventory.
        files = [name for name in files if not name.startswith("build/")]
        super().make_release_tree(base_dir, files)
        project = Path.cwd()
        destination = Path(base_dir) / "canonical"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(canonical(project), destination)
        provenance = canonical(project).parent / "sources.json"
        if provenance.is_file():
            shutil.copyfile(provenance, destination / "sources.json")
        metadata = Path(base_dir) / "pyproject.toml"
        text = metadata.read_text()
        marker = "# Checkout-only uv source."
        if marker in text:
            metadata.write_text(text.split(marker, 1)[0].rstrip() + "\n")
