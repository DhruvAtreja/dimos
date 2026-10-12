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

"""JSON-RPC 2.0 wire models for DimOS's ``args``/``kwargs`` parameter format.

Use ``model_validate_json`` to validate incoming payloads. These models check
message structure; the RPC backend owns routing, method binding and matching
response IDs to pending requests. Batch handling is outside these models.
"""

import math
from typing import Annotated, Literal, TypeAlias

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, JsonValue

# DimOS uses string or integer IDs; explicit null is distinct from an absent ID.
JsonRpcId: TypeAlias = str | int | None

JsonRpcValue: TypeAlias = Annotated[
    JsonValue,
    AfterValidator(
        lambda v: None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v
    ),
]


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class JsonRpcParams(_WireModel):
    """JSON-compatible arguments to pass to ``f(*args, **kwargs)``."""

    args: list[JsonRpcValue] = Field(default_factory=list)
    kwargs: dict[str, JsonRpcValue] = Field(default_factory=dict)


class JsonRpcRequest(_WireModel):
    """A call requiring a response, including when its ID is explicitly null."""

    jsonrpc: Literal["2.0"]
    id: JsonRpcId
    method: str = Field(min_length=1)
    params: JsonRpcParams = Field(default_factory=JsonRpcParams)


class JsonRpcNotification(_WireModel):
    """A call without an ID; no response should be sent."""

    jsonrpc: Literal["2.0"]
    method: str = Field(min_length=1)
    params: JsonRpcParams = Field(default_factory=JsonRpcParams)


class JsonRpcError(_WireModel):
    """A protocol or application error with optional JSON-compatible details."""

    code: int
    message: str
    data: JsonRpcValue = None


class JsonRpcSuccessResponse(_WireModel):
    """A successful response; ``result`` is required even when it is null."""

    jsonrpc: Literal["2.0"]
    id: JsonRpcId
    result: JsonRpcValue


class JsonRpcErrorResponse(_WireModel):
    """An error response; an unknown request ID must be explicitly null."""

    jsonrpc: Literal["2.0"]
    id: JsonRpcId
    error: JsonRpcError


JsonRpcCall: TypeAlias = JsonRpcRequest | JsonRpcNotification
JsonRpcResponse: TypeAlias = JsonRpcSuccessResponse | JsonRpcErrorResponse
