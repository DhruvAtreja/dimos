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

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from experimental.gateway import responses
from experimental.gateway.state import ApiError, FreshQuery, GatewayState

router = APIRouter()


class UploadRequest(BaseModel):
    path: str
    robotId: str | None = None
    kind: str | None = None


@router.get("/dimos/cloud/account", response_model=responses.Account)
async def account(state: GatewayState, fresh: FreshQuery = False) -> dict[str, Any]:
    return await state.uploads.account(fresh)


@router.get("/dimos/cloud/login", response_model=responses.Login)
async def login(state: GatewayState) -> dict[str, Any]:
    return state.uploads.login_state()


@router.post("/dimos/cloud/login", response_model=responses.Login)
async def start_login(state: GatewayState) -> dict[str, Any]:
    """Start a device login: open the answer's url and enter its code."""
    return await state.uploads.start_login()


@router.delete("/dimos/cloud/login", response_model=responses.Login)
async def cancel_login(state: GatewayState) -> dict[str, Any]:
    return state.uploads.cancel_login()


@router.post("/dimos/cloud/logout", response_model=responses.Account)
async def logout(state: GatewayState) -> dict[str, Any]:
    return await state.uploads.logout()


@router.get("/dimos/uploads", response_model=responses.Uploads)
async def upload_list(state: GatewayState) -> dict[str, Any]:
    return state.uploads.listing()


@router.post("/dimos/uploads", response_model=responses.Upload)
async def enqueue(state: GatewayState, request: UploadRequest) -> dict[str, Any]:
    try:
        return state.uploads.enqueue(request.path, request.robotId, request.kind)
    except ValueError as error:
        raise ApiError(400, str(error))


@router.delete("/dimos/uploads", response_model=responses.Uploads)
async def clear_finished(state: GatewayState) -> dict[str, Any]:
    return state.uploads.clear_finished()


@router.get(
    "/dimos/uploads/uploaded", response_model=responses.UploadedByPath | responses.Uploaded | None
)
async def uploaded(state: GatewayState, path: str | None = None) -> Any:
    """Every file uploaded before, or with `path` that one file (null if it never was)."""
    return state.uploads.uploaded() if path is None else state.uploads.uploaded_one(path)


@router.delete("/dimos/uploads/{id}", response_model=responses.Ok)
async def cancel(state: GatewayState, id: str) -> dict[str, Any]:
    try:
        state.uploads.cancel(id)
    except KeyError as error:
        raise ApiError(404, error.args[0])
    return {"ok": True}


@router.post("/dimos/uploads/{id}/retry", response_model=responses.Upload)
async def retry(state: GatewayState, id: str) -> dict[str, Any]:
    """Queue a finished upload again."""
    try:
        return state.uploads.retry(id)
    except KeyError as error:
        raise ApiError(404, error.args[0])
    except ValueError as error:
        raise ApiError(409, str(error))
