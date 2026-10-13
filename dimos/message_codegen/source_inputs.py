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

"""Explicit preparation of pinned data archives; never called by pip hooks."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from tempfile import TemporaryDirectory

import requests


def _relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe source input path: {value}")
    return path


def prepare_sources(lock: Path, output: Path, cache: Path, *, offline: bool = False) -> None:
    data = json.loads(lock.read_text())
    if data.get("format") != 1:
        raise ValueError("Unsupported source lock format")
    cache.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary)
        for source in data["sources"]:
            cache_name = sha256(source["url"].encode()).hexdigest() + ".tar.gz"
            archive = cache / cache_name
            if archive.is_file():
                payload = archive.read_bytes()
            else:
                if offline:
                    raise FileNotFoundError(f"Pinned source archive is not cached: {source['url']}")
                with requests.get(source["url"], timeout=90) as response:
                    response.raise_for_status()
                    payload = response.content
            if sha256(payload).hexdigest() != source["sha256"]:
                raise ValueError(f"Source archive SHA256 mismatch: {source['url']}")
            prefix = str(_relative(source["prefix"])).rstrip("/") + "/"
            with tarfile.open(fileobj=BytesIO(payload)) as bundle:
                for destination, expected in source["files"].items():
                    target = stage / _relative(destination)
                    member = bundle.getmember(prefix + str(_relative(expected["path"])))
                    if not member.isfile():
                        raise ValueError(f"Source input must be a regular file: {member.name}")
                    stream = bundle.extractfile(member)
                    assert stream is not None
                    with stream:
                        contents = stream.read()
                    if sha256(contents).hexdigest() != expected["sha256"]:
                        raise ValueError(f"Source file SHA256 mismatch: {member.name}")
                    if target.exists() and target.read_bytes() != contents:
                        raise ValueError(f"Conflicting source input: {destination}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(contents)
            if not archive.exists():
                archive.write_bytes(payload)
        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(stage, output)
