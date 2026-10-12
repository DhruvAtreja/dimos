#!/usr/bin/env bash
# Compiler-free installation acceptance for checked-in built-in message sources.
set -euo pipefail
cd "$(dirname "$0")/.."
output="$PWD/build/message-codegen/builtin-source"
uv build --python .venv/bin/python --wheel --out-dir "$output/backend" dimos/message_codegen
uv build --python .venv/bin/python --no-sources --find-links "$output/backend" --out-dir "$output/dist" packages/dimos-generated
uv venv --python .venv/bin/python "$output/venv"
uv pip install --python "$output/venv/bin/python" "$output"/backend/*.whl "$output"/dist/*.whl
uv pip install --python "$output/venv/bin/python" setuptools wheel
CC=/bin/false CXX=/bin/false uv pip install --no-build-isolation \
  --python "$output/venv/bin/python" packages/dimos-generated
CC=/bin/false CXX=/bin/false uv pip install --no-build-isolation \
  --python "$output/venv/bin/python" -e packages/dimos-generated
# Exercise every installed built-in type. Only the documented bounded type's
# two endian cases are strict xfails; new unsupported schemas fail normally.
uv pip install --python "$output/venv/bin/python" pytest
(cd /tmp && env -u PYTHONPATH "$output/venv/bin/python" -I -m pytest \
  "$OLDPWD/dimos/message_codegen/test_native.py" --noconftest -o addopts='' -q -ra)

# The dependency download is explicit preparation. The acceptance itself is offline.
"$output/venv/bin/python" -m pip download --only-binary=:all: --dest "$output/dependencies" \
  setuptools wheel "rosbags==0.11.0" "empy==4.2" "lark==1.2.2" "numpy>=1.26.4" "tomli>=2"
DIMOS_MESSAGE_WHEELHOUSE="$output/dependencies" DIMOS_MESSAGE_INSTALL_REQUIRED=1 \
  "${DIMOS_CODEGEN_PYTHON:-$PWD/.venv/bin/python}" -m pytest --noconftest -o addopts='' -q \
  dimos/message_codegen/test_builtin_install.py dimos/message_codegen/test_builtin_catalog.py
