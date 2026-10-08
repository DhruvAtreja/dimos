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

"""Regression tests for safe refreshes of Git-backed robot assets."""

from pathlib import Path
import shutil

from git import Repo
from git.remote import Remote
import pytest
from pytest_mock import MockerFixture

from dimos.robot.assets import git_cache
from dimos.robot.assets.git_cache import GitAssetCache, GitAssetCacheWarning


def _make_remote(tmp_path: Path) -> tuple[Path, Repo]:
    remote_path = tmp_path / "remote.git"
    Repo.init(remote_path, bare=True)
    author = Repo.clone_from(remote_path, tmp_path / "author")
    author.git.checkout("-b", "main")
    author.config_writer().set_value("user", "name", "Test Author").set_value(
        "user", "email", "author@example.com"
    ).release()
    (Path(author.working_tree_dir or "") / "model.urdf").write_text("initial")
    author.index.add(["model.urdf"])
    author.index.commit("initial")
    author.git.push("--set-upstream", "origin", "main")
    return remote_path, author


def test_refresh_preserves_local_only_commits(tmp_path: Path) -> None:
    remote_path, _ = _make_remote(tmp_path)
    cache = GitAssetCache(tmp_path / "cache")
    checkout = cache.resolve(str(remote_path), "main")
    repo = Repo(checkout)
    repo.config_writer().set_value("user", "name", "Local User").set_value(
        "user", "email", "local@example.com"
    ).release()
    (checkout / "local.txt").write_text("keep me")
    repo.index.add(["local.txt"])
    local_commit = repo.index.commit("local-only").hexsha

    with pytest.warns(GitAssetCacheWarning, match="has local-only commits"):
        refreshed = cache.resolve(str(remote_path), "main")

    assert refreshed == checkout
    assert Repo(checkout).head.commit.hexsha == local_commit
    assert (checkout / "local.txt").read_text() == "keep me"


def test_refresh_updates_clean_checkout(tmp_path: Path) -> None:
    remote_path, author = _make_remote(tmp_path)
    cache = GitAssetCache(tmp_path / "cache")
    checkout = cache.resolve(str(remote_path), "main")
    original_commit = Repo(checkout).head.commit.hexsha

    author_path = Path(author.working_tree_dir or "")
    (author_path / "model.urdf").write_text("updated")
    author.index.add(["model.urdf"])
    updated_commit = author.index.commit("update").hexsha
    author.git.push("origin", "main")

    assert cache.resolve(str(remote_path), "main") == checkout
    assert original_commit != updated_commit
    assert Repo(checkout).head.commit.hexsha == updated_commit
    assert (checkout / "model.urdf").read_text() == "updated"


def test_pinned_commit_checkout_is_used_without_fetching(
    tmp_path: Path, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote_path, author = _make_remote(tmp_path)
    commit = author.head.commit.hexsha
    cache = GitAssetCache(tmp_path / "cache")
    checkout = cache.resolve(str(remote_path), commit)
    # Forget this process's earlier answer, as a fresh process would.
    monkeypatch.setattr(git_cache, "_pinned_checkouts", {})
    fetch = mocker.patch.object(Remote, "fetch")

    assert GitAssetCache(tmp_path / "cache").resolve(str(remote_path), commit) == checkout
    fetch.assert_not_called()
    assert Repo(checkout).head.commit.hexsha == commit
    assert (checkout / "model.urdf").read_text() == "initial"


def test_pinned_commit_resolves_once_per_process(tmp_path: Path, mocker: MockerFixture) -> None:
    remote_path, author = _make_remote(tmp_path)
    commit = author.head.commit.hexsha
    checkout = GitAssetCache(tmp_path / "cache").resolve(str(remote_path), commit)
    refresh = mocker.spy(GitAssetCache, "_refresh_cached")

    # A new cache object per call, like every unpickled robot description path.
    paths = {GitAssetCache(tmp_path / "cache").resolve(str(remote_path), commit) for _ in range(5)}

    assert paths == {checkout}
    refresh.assert_not_called()


def test_pinned_commit_is_cloned_again_after_cache_removal(tmp_path: Path) -> None:
    remote_path, author = _make_remote(tmp_path)
    commit = author.head.commit.hexsha
    cache = GitAssetCache(tmp_path / "cache")
    checkout = cache.resolve(str(remote_path), commit)
    shutil.rmtree(checkout)

    assert cache.resolve(str(remote_path), commit) == checkout
    assert Repo(checkout).head.commit.hexsha == commit
