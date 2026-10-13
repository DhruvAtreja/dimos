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

"""Product adapter for generic source-package hooks."""

from pathlib import Path

from dimos_message_build.source_package import SourceDistribution, setup_arguments
from setuptools import setup


class BuiltinSourceDistribution(SourceDistribution):
    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        super().make_release_tree(base_dir, files)
        metadata = Path(base_dir) / "pyproject.toml"
        marker = "# Checkout-only uv source."
        text = metadata.read_text()
        if marker in text:
            metadata.write_text(text.split(marker, 1)[0].rstrip() + "\n")


arguments = setup_arguments(Path(__file__).resolve().parent)
arguments["cmdclass"] = {"sdist": BuiltinSourceDistribution}
setup(**arguments)
