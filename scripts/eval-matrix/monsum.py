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

"""Peak and mean resource use from a monitor.sh file: python3 monsum.py FILE [FROM TO]."""

from collections import defaultdict
import re
import sys

lines = open(sys.argv[1]).read().splitlines()
lo, hi = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("00:00:00", "99:99:99")
host, gpu, load, nrun = [], [], [], []
cpu, mem = defaultdict(list), defaultdict(list)
per_ts = defaultdict(int)
for l in lines:
    ts = l[:8]
    if not (lo <= ts <= hi):
        continue
    m = re.search(r"host=(\d+)MB used.*load=([\d.]+) gpu_mem_MB,util%=(\d+),(\d+)", l)
    if m:
        host.append(int(m[1]))
        load.append(float(m[2]))
        gpu.append((int(m[3]), int(m[4])))
        continue
    m = re.search(r"(\S+) cpu=([\d.]+)% mem=([\d.]+)(GiB|MiB)", l)
    if m:
        v = float(m[3]) * (1 if m[4] == "GiB" else 1 / 1024)
        cpu[m[1]].append(float(m[2]))
        mem[m[1]].append(v)
        per_ts[ts] += 1
full = max(per_ts.values()) if per_ts else 0
print(f"host RAM used peak {max(host) / 1024:.1f} GB, mean {sum(host) / len(host) / 1024:.1f} GB")
print(
    f"GPU mem peak {max(g[0] for g in gpu) / 1024:.1f} GB, util mean {sum(g[1] for g in gpu) / len(gpu):.0f}%"
)
print(f"load avg peak {max(load):.1f}, mean {sum(load) / len(load):.1f} (8 vCPU)")
for c in sorted(cpu):
    print(
        f"  {c}: RAM peak {max(mem[c]):.1f} GB, CPU mean {sum(cpu[c]) / len(cpu[c]):.0f}% peak {max(cpu[c]):.0f}%"
    )
# Total container CPU at moments when all containers were up.
tot = defaultdict(float)
for l in lines:
    m = re.search(r"^(\S+)\s+(\S+) cpu=([\d.]+)%", l)
    if m and lo <= m[1] <= hi and per_ts[m[1]] == full:
        tot[m[1]] += float(m[3])
if tot:
    v = sorted(tot.values())
    print(
        f"all {full} running: total CPU mean {sum(v) / len(v):.0f}% p90 {v[int(len(v) * 0.9)]:.0f}% of 800%"
    )
