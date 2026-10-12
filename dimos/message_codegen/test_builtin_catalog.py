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

"""An explicit catalog preserves dependencies without exporting unrelated sources."""

import json
import shutil

import pytest

from dimos.message_codegen.builtin import generate_builtin, supported_names
from dimos.message_codegen.definitions import BUNDLED_SCHEMAS


def test_unlisted_sources_do_not_expand_the_public_catalog(tmp_path):
    source = tmp_path / "schemas"
    shutil.copytree(BUNDLED_SCHEMAS, source)
    shutil.copyfile(BUNDLED_SCHEMAS.parent / "sources.json", tmp_path / "sources.json")
    extra = source / "extra_msgs/msg/NotExported.msg"
    extra.parent.mkdir(parents=True)
    extra.write_text("std_msgs/Header header\n")
    output = tmp_path / "output"
    names = generate_builtin(output, "0.1.0", source)
    assert "extra_msgs/msg/NotExported" not in names
    assert not (output / "schemas/extra_msgs/msg/NotExported.msg").exists()
    assert "builtin_interfaces/msg/Time" in names
    assert len(names) == 73


def test_declared_root_requires_a_reason_and_a_resolvable_closure(tmp_path):
    source = tmp_path / "schemas"
    shutil.copytree(BUNDLED_SCHEMAS, source)
    catalog = source / "catalog.json"
    catalog.write_text(json.dumps({"format": 1, "roots": {"std_msgs/msg/Header": ""}}))
    with pytest.raises(ValueError, match="support reason"):
        supported_names(source)
    catalog.write_text(json.dumps({"format": 1, "roots": {"std_msgs/msg/Header": "Public API"}}))
    assert set(supported_names(source)) == {"std_msgs/msg/Header", "builtin_interfaces/msg/Time"}
    (source / "builtin_interfaces/msg/Time.msg").unlink()
    with pytest.raises(ValueError, match="unresolved message dependency"):
        supported_names(source)
