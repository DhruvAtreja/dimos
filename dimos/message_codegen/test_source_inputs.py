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

"""Pinned source preparation is explicit, hash checked, and transactional."""

from hashlib import sha256
from io import BytesIO
import json
import tarfile

import pytest

from dimos.message_codegen.source_inputs import prepare_sources


@pytest.fixture
def prepared_archive(tmp_path, monkeypatch):
    content = b"int32 value\n"
    stream = BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as bundle:
        member = tarfile.TarInfo("upstream-revision/msg/Value.msg")
        member.size = len(content)
        bundle.addfile(member, BytesIO(content))
    payload = stream.getvalue()
    source = {
        "url": "https://example.invalid/pinned.tar.gz",
        "sha256": sha256(payload).hexdigest(),
        "prefix": "upstream-revision",
        "files": {
            "sample_msgs/msg/Value.msg": {
                "path": "msg/Value.msg",
                "sha256": sha256(content).hexdigest(),
            }
        },
    }
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"format": 1, "sources": [source]}))
    cache = tmp_path / "cache"
    cache.mkdir()
    archive = cache / (sha256(source["url"].encode()).hexdigest() + ".tar.gz")
    archive.write_bytes(payload)

    def no_network(*args, **kwargs):
        raise AssertionError("Offline source preparation attempted network access")

    monkeypatch.setattr("dimos.message_codegen.source_inputs.requests.get", no_network)
    return lock, cache, archive


def test_offline_preparation_checks_archive_and_file_hashes(prepared_archive, tmp_path):
    lock, cache, archive = prepared_archive
    output = tmp_path / "inputs"
    prepare_sources(lock, output, cache, offline=True)
    assert (output / "sample_msgs/msg/Value.msg").read_bytes() == b"int32 value\n"
    archive.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="archive SHA256"):
        prepare_sources(lock, output, cache, offline=True)
    assert (output / "sample_msgs/msg/Value.msg").read_bytes() == b"int32 value\n"


def test_wrong_file_hash_and_missing_cache_preserve_inputs(prepared_archive, tmp_path):
    lock, cache, archive = prepared_archive
    output = tmp_path / "inputs"
    output.mkdir()
    (output / "previous").write_text("retained")
    data = json.loads(lock.read_text())
    data["sources"][0]["files"]["sample_msgs/msg/Value.msg"]["sha256"] = "0" * 64
    lock.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="file SHA256"):
        prepare_sources(lock, output, cache, offline=True)
    archive.unlink()
    with pytest.raises(FileNotFoundError, match="not cached"):
        prepare_sources(lock, output, cache, offline=True)
    assert (output / "previous").read_text() == "retained"


def test_source_destination_cannot_escape_output(prepared_archive, tmp_path):
    lock, cache, _ = prepared_archive
    data = json.loads(lock.read_text())
    file = data["sources"][0]["files"].pop("sample_msgs/msg/Value.msg")
    data["sources"][0]["files"]["../escape"] = file
    lock.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unsafe source input path"):
        prepare_sources(lock, tmp_path / "inputs", cache, offline=True)
    assert not (tmp_path / "escape").exists()
