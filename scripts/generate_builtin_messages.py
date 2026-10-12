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

"""Regenerate build artifacts; --check rejects drift without rewriting them."""

import argparse
from pathlib import Path
import shutil
import sys
import tempfile

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from dimos.message_codegen.builtin import generate_builtin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "packages/dimos-generated/pyproject.toml").read_text())[
        "project"
    ]
    version = project["version"]
    target = root / "packages/dimos-generated/build/message-source/python"
    with tempfile.TemporaryDirectory(prefix="dimos-source-") as temporary:
        output = Path(temporary)
        generate_builtin(output, version)
        source = output / "python"
        packages = [source / "dimos_generated", source / "dimos_generated_schemas"]
        expected = {
            str(path.relative_to(source)): path.read_bytes()
            for package in packages
            for path in package.rglob("*")
            if path.is_file()
        }
        actual = {
            str(path.relative_to(target)): path.read_bytes()
            for path in target.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
            and not any(part.endswith(".egg-info") for part in path.parts)
        }
        if args.check:
            if actual != expected:
                missing = sorted(expected.keys() - actual.keys())
                stale = sorted(actual.keys() - expected.keys())
                changed = sorted(
                    name
                    for name in actual.keys() & expected.keys()
                    if actual[name] != expected[name]
                )
                parser.error(
                    f"Generated source drift: missing={missing}, stale={stale}, changed={changed}. Run python -m scripts.generate_builtin_messages"
                )
            print("Built-in generated Python/C++/Rust sources match canonical definitions")
            return
        target.mkdir(parents=True, exist_ok=True)
        for package in packages:
            destination = target / package.name
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(package, destination)
        print(f"Generated built-in source in {target}")


if __name__ == "__main__":
    main()
