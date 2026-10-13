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

"""Prepare pinned upstream schemas for conformance tests, outside package hooks."""

import argparse
from pathlib import Path

from dimos.message_codegen.source_inputs import prepare_sources

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "build/message-codegen/test-inputs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "build/message-codegen/source-cache")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    prepare_sources(
        Path(__file__).with_name("message-test-sources.json"),
        OUTPUT,
        args.cache,
        offline=args.offline,
    )


if __name__ == "__main__":
    main()
