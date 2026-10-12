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

"""PEP 517/660 uses the same explicit message source build as release packaging."""

from pathlib import Path
import sys

from setuptools import find_packages, setup

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _build import SourceDistribution, prepare

project = Path(__file__).resolve().parent
version = tomllib.loads((project / "pyproject.toml").read_text())["project"]["version"]
source = prepare(project, version)
setup(
    packages=find_packages(str(source)),
    package_dir={"": str(source.relative_to(project))},
    cmdclass={"sdist": SourceDistribution},
)
