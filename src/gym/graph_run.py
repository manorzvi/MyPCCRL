# Copyright 2019 Nathan Jay and Noga Rotman
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
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import re
from pprint import pprint

if (not (len(sys.argv) == 3)) or (sys.argv[1] == "-h") or (sys.argv[1] == "--help"):
    print("usage: python3 graph_run.py <pcc_env_log_filename.json>")
    exit(0)

filename = sys.argv[1]
stdout_and_stderr_filename = sys.argv[2]
fig_filename = os.path.splitext(filename)[0] + '.pdf'

last_lines = []
with open(stdout_and_stderr_filename, 'r') as f:
    for line in f:
        if len(last_lines) == 3:
            last_lines.pop(0)
        last_lines.append(line)
        if re.search(filename, line):
            link_params_str = last_lines[0]
            break

bw      = float(re.search(r'bw=(\d+\.\d+)', link_params_str).group(1))
lat     = float(re.search(r'lat=(\d+\.\d+)', link_params_str).group(1))
queue   = float(re.search(r'queue= *(\d+)', link_params_str).group(1))
loss    = float(re.search(r'loss=(\d+\.\d+)', link_params_str).group(1))

print(
    f'bandwidth={1500.0 * 8.0 * bw}[b\s]\n'
    f'latency={lat}\n'
    f'queue_size={queue}\n'
    f'max_queue_letancy={queue/bw}\n'
    f'loss_rate={loss}\n'
    f'min_loss_rate={loss + (1-loss)*loss}'
)

data = {}
with open(filename, 'r') as f:
    data = json.load(f)

time_data = [float(event["Time"]) for event in data["Events"][1:]]
rew_data = [float(event["Reward"]) for event in data["Events"][1:]]
send_data = [float(event["Send Rate"]) for event in data["Events"][1:]]
thpt_data = [float(event["Throughput"]) for event in data["Events"][1:]]
latency_data = [float(event["Latency"]) for event in data["Events"][1:]]
loss_data = [float(event["Loss Rate"]) for event in data["Events"][1:]]

max_bw_data = [1500.0 * 8.0 * bw]*len(time_data)  # 1500 B in packet X 8 b in B X packets/s
min_latency = [lat * 2.0]*len(time_data)  # 2 links X each link latency. Min Latency => no link queue delay
max_latency = [(lat+queue/bw) * 2.0]*len(time_data)  # 2 links X each link latency + max queue delay
min_loss_rate = [loss + (1-loss)*loss]*len(time_data)  # Random loss rate. 2 links. A packet can drop on the 1st link with probability <loss>, or pass the 1st link with probability 1-<loss> and then drop on the 2nd link
fig, axes = plt.subplots(5, figsize=(10, 12))
rew_axis = axes[0]
send_axis = axes[1]
thpt_axis = axes[2]
latency_axis = axes[3]
loss_axis = axes[4]

rew_axis.plot(time_data, rew_data, label='Reward')
rew_axis.legend()

send_axis.plot(time_data, send_data, label='Send Rate [b/s]')
send_axis.plot(time_data, max_bw_data, label='Max BW [b/s]')
send_axis.legend()

thpt_axis.plot(time_data, thpt_data, label='Throughput [b/s]')
thpt_axis.plot(time_data, max_bw_data, label='Max BW [b/s]')
thpt_axis.legend()

latency_axis.plot(time_data, latency_data, label='Latency [s]')
latency_axis.plot(time_data, min_latency, label='Min Latency [s]')
latency_axis.plot(time_data, max_latency, label='Max Latency [s]')

latency_axis.legend()

loss_axis.plot(time_data, loss_data, label='Loss Rate')
loss_axis.plot(time_data, min_loss_rate, label='Min Loss Rate')
loss_axis.legend()
loss_axis.set_xlabel("Monitor Interval")

fig.suptitle("Summary Graph for %s" % sys.argv[1])
fig.savefig(fig_filename)
