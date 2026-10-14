# Dimos gateway: how do I...?

Snippets run in an app page served by Desktop at `/apps/<name>/`, so `../../dimos/` is the gateway. An app's
`dimos.yaml` lists each route it calls, relative to `/dimos`:

```yaml
uses:
    "@dimos-gateway":
        - GET /blueprints
        - POST /runs
```

## How do I list blueprints?

```js
const { blueprints } = await (await fetch("../../dimos/blueprints")).json() // [{ name, kind, importable }]
```

## How do I launch a blueprint?

```js
await fetch("../../dimos/runs", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ blueprint: "unitree-go2", replay: true, overrides: { global: { n_workers: 4 } } }),
}) // answers at once, phase "starting"
```

## How do I see what's running?

```js
const { runs, launch } = await (await fetch("../../dimos/runs")).json() // runs: [{ run_id, blueprint, pid }], launch.phase
getZenoh().subscribeDimos("launch", (launch) => show(launch.phase)) // live phase changes; getZenoh from ./dim-app/mod.js
```

## How do I stop a blueprint?

```js
await fetch("../../dimos/runs/stop", { method: "POST" }) // or body { runId } for a specific run
```

## How do I relaunch?

```js
await fetch("../../dimos/runs/restart", { method: "POST" }) // the last launch again, with the config saved now
```

## How do I read a run's logs?

```js
const page = await (await fetch("../../dimos/runs/latest/log?level=warning&limit=200")).json() // { records, offset }
const newer = await (await fetch(`../../dimos/runs/latest/log?after=${page.offset}`)).json() // tail
```

## How do I list a blueprint's modules, their streams and RPCs?

```js
const { modules } = await (await fetch("../../dimos/blueprints/unitree-go2")).json()
// modules[i]: { name, rpcs, skills, streams: [{ name, type, direction, topic }] }
```

## How do I list every module's inputs and outputs?

```js
const { modules } = await (await fetch("../../dimos/catalog")).json() // [{ name, inputs, outputs, skills }]; slow the first time
```

## How do I list skills?

```js
const { skills } = await (await fetch("../../dimos/skills")).json() // the running blueprint's: [{ name, module, params }]
```

## How do I call a skill?

```js
const result = await (await fetch("../../dimos/skills/call", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ skill: "execute_sport_command", args: { command_name: "FrontJump" } }),
})).json() // { ok, text }; this acts on the robot, so call it only from a user's action
```

## How do I decode dimos messages in a page?

```js
import { DimApp } from "./dim-app/mod.js"

const app = new DimApp({ msgDecodeEndpoint: "../../dimos/msgs.js" })
app.subscribe("odom", (odom) => console.log(odom.pose.pose.position)) // decoded
const twist = app.msgs.geometry_msgs.Twist.encode({ linear: { x: 0.3 } }) // bytes
```

## How do I list robots and their recommended blueprints?

```js
const { robots } = await (await fetch("../../dimos/robots")).json()
robots.go2.recommended // ["unitree-go2-basic", ...]
robots.go2.blueprints["unitree-go2"] // { title, description, starter, recommended_config }
```

## How do I get dimos's python?

```js
const { python } = await (await fetch("../../dimos/python")).json() // run `${python} -c 'import dimos'`
```

## How do I read a blueprint's config?

```js
const { modules } = await (await fetch("../../dimos/blueprints/unitree-go2/config")).json()
// modules[i]: { module, args: [{ name, type, default, value }] }
```

## How do I save a blueprint's config?

```js
await fetch("../../dimos/blueprints/unitree-go2/config", {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ overrides: { go2connection: { lidar: false } } }), // replaces the saved config; null drops a field
}) // applies at the next launch
```
