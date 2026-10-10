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

"""Save disk in the GraspGenX image by pointing at the main venv's copy of a package.

GraspGenX runs in its own Python environment, which wants the same torch, CUDA and triton
wheels (about 6 GB) that the image's main venv at /opt/venv already holds. This script
makes the second environment use those files through symlinks instead of a second copy.

    python gx_link.py prelink MAIN_SITE TARGET_SITE UV_LOCK
        Before `uv sync`: for every package whose locked version in UV_LOCK equals the
        version installed in MAIN_SITE, put that package into TARGET_SITE as symlinks to
        MAIN_SITE's files, so uv sees it installed and does not download it.

    python gx_link.py dedupe MAIN_SITE DIR [DIR ...]
        After uv has run: in every package folder under each DIR (a site-packages or an
        unpacked wheel in the uv cache) whose name and version match a package installed
        in MAIN_SITE, replace each file that is byte-identical to MAIN_SITE's copy with a
        symlink to it.

Only packages of exactly the same version are linked, so a package never mixes files of
two versions. MAIN_SITE is a site-packages folder, e.g. /opt/venv/lib/python3.12/site-packages.
"""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
import re
import shutil
import sys

import tomllib


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def dists(site: Path) -> dict[tuple[str, str], Path]:
    """(normalized name, version) -> dist-info folder, for every package in a folder."""
    found = {}
    for info in site.glob("*.dist-info"):
        name, _, version = info.name[: -len(".dist-info")].rpartition("-")
        if name and version:
            found[(norm(name), version)] = info
    return found


def record_files(info: Path) -> list[str]:
    """Paths (relative to the site folder) of the files a package installed, without
    its dist-info and without anything outside the site folder (scripts in bin/)."""
    record = info / "RECORD"
    if not record.is_file():
        return []
    paths = []
    with record.open(newline="") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            path = row[0]
            if path.startswith("..") or path.startswith("/") or path.startswith(info.name + "/"):
                continue
            paths.append(path)
    return paths


def prelink(main_site: Path, target_site: Path, uv_lock: Path) -> None:
    lock = tomllib.loads(uv_lock.read_text())
    wanted = {(norm(p["name"]), p["version"]) for p in lock["package"] if "version" in p}
    installed = dists(main_site)
    linked = 0
    for key in sorted(wanted & installed.keys()):
        info = installed[key]
        for rel in record_files(info):
            src, dst = main_site / rel, target_site / rel
            if not src.exists() or dst.exists() or dst.is_symlink():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src)
        # The dist-info itself is copied, so uv can read and rewrite it like any other.
        shutil.copytree(info, target_site / info.name, symlinks=True, dirs_exist_ok=True)
        linked += 1
    print(f"prelink: {linked} of {len(wanted)} locked packages linked from {main_site}")


def digest(path: Path) -> str:
    h = hashlib.blake2b(digest_size=20)
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def dedupe(main_site: Path, roots: list[Path]) -> None:
    installed = dists(main_site)
    saved = files = 0
    for root in roots:
        for info in root.rglob("*.dist-info"):
            if not info.is_dir() or info.is_symlink():
                continue
            name, _, version = info.name[: -len(".dist-info")].rpartition("-")
            main_info = installed.get((norm(name), version))
            if main_info is None or info.parent.resolve() == main_site.resolve():
                continue
            for rel in record_files(info):
                here, there = info.parent / rel, main_site / rel
                if here.is_symlink() or not here.is_file() or not there.is_file():
                    continue
                size = here.stat().st_size
                if size == 0 or size != there.stat().st_size or digest(here) != digest(there):
                    continue
                tmp = here.with_name(here.name + ".gxlink")
                tmp.symlink_to(there.resolve())
                os.replace(tmp, here)
                saved += size
                files += 1
    print(f"dedupe: {files} files ({saved / 1e9:.2f} GB) now point into {main_site}")


def main(argv: list[str]) -> None:
    if len(argv) >= 4 and argv[0] == "prelink":
        prelink(Path(argv[1]), Path(argv[2]), Path(argv[3]))
    elif len(argv) >= 3 and argv[0] == "dedupe":
        dedupe(Path(argv[1]), [Path(a) for a in argv[2:]])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
