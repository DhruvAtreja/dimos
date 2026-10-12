# Zenoh RPC 与 JSON-RPC 方案

本次修改的是 **DimOS 模块间调用 `@rpc` 方法的 RPC 通信层**，也就是 `RPCClient / RpcCall → RPCSpec → ZenohRPC` 这条链路。Spec 注入的模块引用通过它调用远端方法，模块生命周期和 Coordinator 管理接口也使用这套 RPC。首期只迁移适合 JSON 的 Python 方法。

这里不修改 worker 管理使用的 **Actor IPC**，不修改供 Agent 调用工具的 **MCP 协议**，也不修改 `In/Out/IO` 的数据流传输。

保留 Zenoh 负责监听、按 key 分发请求和传输响应；由 DimOS 使用 Python 标准库 `json` 完成编解码，并检查 JSON-RPC 消息格式。先做 Python，原有调用方式不变；无法用 JSON 表达的对象继续走现有协议。

以下是设计方案，尚未修改运行时代码。

## 1. 现有 Zenoh RPC 流程

以 `self.calc.compute1(2, 3)` 返回 `5` 为例。

启动时，模块将带 `@rpc` 的方法注册到 Zenoh。Blueprint 根据 Spec 找到目标模块，将 `RPCClient` 代理注入 `self.calc`。

现有 RPC 的 Zenoh key 格式是 `dimos/rpc/<模块实例名>/<方法名>`。例如，`Calculator` 的两个方法分别注册在 `dimos/rpc/Calculator/compute1` 和 `dimos/rpc/Calculator/compute2`。客户端查询对应 key，服务端在同一个 key 上回复，不另外创建响应 topic。

| 步骤        | 发生什么                                                                                 |
| ----------- | ---------------------------------------------------------------------------------------- |
| 1. 获取方法 | `self.calc.compute1` 返回 `RpcCall` 对象                                                 |
| 2. 发起调用 | `RpcCall` 调用 `call_sync()`，目标为 `Calculator/compute1`，参数为 `((2, 3), {})`        |
| 3. 发送请求 | `ZenohRPC` 用 pickle 编码参数，通过 `session.get()` 发往 `dimos/rpc/Calculator/compute1` |
| 4. 远端执行 | Zenoh Queryable 收到请求，在工作线程中解码参数并执行 `compute1(2, 3)`                    |
| 5. 返回结果 | 服务端用 pickle 编码 `5`，通过 `query.reply()` 回传                                      |
| 6. 完成调用 | 客户端回调解码结果，唤醒 `call_sync()`，业务代码得到 `5`                                 |

当前协议有三个特点：

- 方法名放在 Zenoh key 中，请求体只有 `(args, kwargs)`。
- `call_id` 只用于客户端管理回调，不会随请求发送。
- 调用看起来是同步的，底层通过回调接收响应。

因此，当前是 **Zenoh + pickle RPC**，还不是 JSON-RPC。

相关代码：[客户端代理](/dimos/core/rpc_client.py)、[RPC 接口](/dimos/protocol/rpc/spec.py)、[Zenoh 后端](/dimos/protocol/rpc/zenohrpc.py)。

## 2. JSON-RPC 方案

业务代码仍然调用 `self.calc.compute1(2, 3)`，只在 RPC 后端增加 JSON-RPC 支持。

| 组件 | 职责 |
| --- | --- |
| 现有 `RPCClient` / `RpcCall` | 保留业务调用入口 |
| 新增 `ZenohJsonRPC` | 实现现有 `RPCSpec` 接口，完成 JSON 编解码、消息与参数校验、执行已绑定方法和异常转换 |
| Zenoh | 通过 Queryable 监听方法 key，将请求路由到对应回调，并传输响应 |

方法仍通过 `session.declare_queryable()` 注册到 Zenoh。Zenoh 回调收到请求后，将处理任务交给工作线程；DimOS 完成解码与校验，再执行原模块方法。

建议给新协议单独使用 `dimos/jsonrpc/python-args/<模块实例名>/<方法名>`，避免与现有 pickle 请求混用。仍然为每个方法注册一个 Zenoh key，保持现有的路由方式。请求体中的 `method` 用于和 key 做一致性校验，不用于二次分发。

| 协议 | Calculator 示例的 Zenoh key | 如何选择方法 |
| --- | --- | --- |
| 现有 pickle RPC | `dimos/rpc/Calculator/compute1` | Zenoh 按 key 路由到对应方法回调 |
| 拟新增 JSON-RPC | `dimos/jsonrpc/python-args/Calculator/compute1` | 同样由 Zenoh 按 key 路由，回调内完成校验与执行 |

同一次计算的新请求包含：

| 字段      | 值                               |
| --------- | -------------------------------- |
| `jsonrpc` | `"2.0"`                          |
| `id`      | `"client1:1"`                    |
| `method`  | `"Calculator/compute1"`          |
| `params`  | `{"args": [2, 3], "kwargs": {}}` |

成功响应为 `{"jsonrpc": "2.0", "id": "client1:1", "result": 5}`。

这里的 `args` 和 `kwargs` 是 DimOS 自己约定的参数结构。DimOS 适配层校验消息和参数后，执行已绑定的原方法 `f(*args, **kwargs)`，从而保留 Python 的位置参数、关键字参数和混合调用。它符合 JSON-RPC 的命名参数形式，但其他语言接入时也需要遵守这个约定。[JSON-RPC 2.0 规范](https://www.jsonrpc.org/specification)

新流程就是：**原调用入口 → JSON-RPC 编码 → Zenoh 按 key 分发 → DimOS 校验并执行原方法 → Zenoh 返回 JSON-RPC 响应**。客户端收到响应后，也需校验消息结构和请求 id，再返回结果或抛出异常。

## 3. 错误处理

现有 `ZenohRPC` 在方法抛出异常时，将异常类型、参数和 traceback 序列化为 pickle，通过 `query.reply_err()` 返回，encoding 为 `dimos/rpc-exception`。客户端还原异常，交给 `call_sync()` 抛出：内置异常尽量保留原类型，自定义异常转为 `RemoteError`，并保留远端调用栈。

新方案由 DimOS 适配层将解析、校验和执行错误转换为 JSON-RPC 响应的 `error` 字段，通过 Zenoh 的普通 `query.reply()` 返回；`result` 和 `error` 不同时出现。例如：

`{"jsonrpc": "2.0", "id": "client1:1", "error": {"code": -32000, "message": "参数不能为负数"}}`

| 错误          | code                   | 含义                       |
| ------------- | ---------------------- | -------------------------- |
| JSON 解析失败 | `-32700`               | 请求不是合法 JSON          |
| 请求格式错误  | `-32600`               | JSON-RPC 请求结构不正确    |
| 方法不存在    | `-32601`               | 处理端已收到请求，但确认目标方法不可用 |
| 参数格式错误  | `-32602`               | `params` 不符合接口要求    |
| 内部错误      | `-32603`               | RPC 框架处理请求失败       |
| 业务异常      | `-32000`（本方案约定） | 原模块方法执行时抛出异常   |

标准错误定义见 [JSON-RPC 错误对象](https://www.jsonrpc.org/specification#error_object)。为兼容现有 Python 代码，业务异常还应在 `error.data` 中携带 `type_name`、`type_module`、`args` 和 `traceback`，由客户端适配回原来的异常处理逻辑。方法体抛出的 `TypeError` 等异常不能一律当作协议参数错误；无法无损编码异常的方法继续使用旧协议。

由于按方法 key 监听，如果目标 key 没有 Queryable，就没有处理端返回 `-32601`，客户端会等待超时。超时仍抛出 `TimeoutError`，不表示远端没有执行，也不触发自动重发。`stop()` 等不等待响应的调用只在服务端记录异常，不向客户端返回错误。

## 4. 如何兼容现有代码

**先让新旧协议共存，按方法迁移，不一次性替换 pickle。**

1. **默认保留旧协议。** 只有双方都支持 JSON-RPC，且方法的参数、返回值和异常可以无损转换时，才启用新通道。第一阶段可以显式配置这些方法，不必先做复杂的自动协商。
2. **复杂对象继续走旧通道。** `set_module_ref()` 传递模块代理，`set_transport()` 传递 Transport，`get_skills()` 返回 `SkillInfo`，这些都不能直接当普通 JSON 使用。后续再逐个定义转换方式。
3. **保留原调用行为。** 同步、回调、异步接口、超时配置和异常类型都要兼容；`stop()` 仍不等待响应。新后端继续调用已有的异步 `@rpc` 包装器，代理跨 worker 后也要保留协议选择。
4. **发送前决定协议。** JSON 请求发出后，即使超时或结果解码失败，也不能自动改用旧协议再调用一次，避免重复执行动作。
5. **Rust/C++ 暂不改动。** 现有 `NativeModule` 的子进程启动、stdin 配置，以及 typed streams 的 topic 和消息编码保持不变。后续再为 Rust/C++ 增加遵守相同 JSON 契约的 RPC 接口。

兼容性涉及的主要代码：[模块接口](/dimos/core/module.py)、[异常还原](/dimos/protocol/rpc/rpc_utils.py)、[NativeModule](/dimos/core/native_module.py)。Actor 的进程管理和属性访问仍走现有 IPC，不属于这次替换范围。

实现时分别验证 JSON-RPC 消息校验、Zenoh 的方法路由，以及 DimOS 的参数绑定和异常还原。

兼容验收标准是：**现有代码无需修改即可继续运行，已迁移方法保持相同的结果类型和调用行为，现有 Rust/C++ 数据流不受影响。**
