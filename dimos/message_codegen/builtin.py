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

"""The explicit built-in catalog shared by package and native source builds."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from .definitions import BUNDLED_SCHEMAS, Definitions
from .generate import generate


def supported_names(source: Path = BUNDLED_SCHEMAS) -> tuple[str, ...]:
    data = json.loads((source / "catalog.json").read_text())
    if data.get("format") != 1 or not data.get("roots"):
        raise ValueError("A nonempty explicit built-in catalog is required")
    if any(not isinstance(reason, str) or not reason.strip() for reason in data["roots"].values()):
        raise ValueError("Each built-in root requires a support reason")
    return tuple(
        message.name for message in Definitions([source], bundled=False).resolve(data["roots"])
    )


def generate_builtin(output: Path, version: str, source: Path = BUNDLED_SCHEMAS) -> tuple[str, ...]:
    names = generate(
        [source],
        output,
        list(supported_names(source)),
        "dimos_generated",
        version=version,
        shared=True,
        bundled=False,
    )
    resources = output / "python/dimos_generated_schemas/package"
    shutil.copyfile(source / "catalog.json", resources / "catalog.json")
    provenance = source / "sources.json"
    if not provenance.is_file():
        provenance = source.parent / "sources.json"
    if not provenance.is_file():
        raise ValueError("Canonical schema provenance is missing")
    shutil.copyfile(provenance, resources / "sources.json")
    return names
