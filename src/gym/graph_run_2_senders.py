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
        if len(last_lines) == 4:
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
with open(filename) as f:
    data = json.load(f)

time_data       = [float(event["Time"]) for event in data["Events"][1:]]
rew_data        = [float(event["Reward"]) for event in data["Events"][1:]]

send_data0      = [float(event["Send Rate 0"]) for event in data["Events"][1:]]
send_data1      = [float(event["Send Rate 1"]) for event in data["Events"][1:]]

thpt_data0      = [float(event["Throughput 0"]) for event in data["Events"][1:]]
thpt_data1      = [float(event["Throughput 1"]) for event in data["Events"][1:]]

latency_data0   = [float(event["Latency 0"]) for event in data["Events"][1:]]
latency_data1   = [float(event["Latency 1"]) for event in data["Events"][1:]]

loss_data0      = [float(event["Loss Rate 0"]) for event in data["Events"][1:]]
loss_data1      = [float(event["Loss Rate 1"]) for event in data["Events"][1:]]

max_bw_data = [1500.0 * 8.0 * bw]*len(time_data)  # 1500 B in packet X 8 b in B X packets/s
min_latency = [lat * 2.0]*len(time_data)  # 2 links X each link latency. Min Latency => no link queue delay
max_latency = [(lat+queue/bw) * 2.0]*len(time_data)  # 2 links X each link latency + max queue delay

fig, axes       = plt.subplots(5, figsize=(10, 12))
rew_axis        = axes[0]
send_axis       = axes[1]
thpt_axis       = axes[2]
latency_axis    = axes[3]
loss_axis       = axes[4]

rew_axis.plot(time_data, rew_data, label='Reward')
rew_axis.legend()

send_axis.plot(time_data, send_data0, label='Send Rate Sender0 [b/s]')
send_axis.plot(time_data, send_data1, label='Send Rate Sender1 [b/s]')
send_axis.plot(time_data, max_bw_data, label='Max BW [b/s]')
send_axis.legend()

thpt_axis.plot(time_data, thpt_data0, label='Throughput Sender0 [b/s]')
thpt_axis.plot(time_data, thpt_data1, label='Throughput Sender1 [b/s]')
thpt_axis.plot(time_data, max_bw_data, label='Max BW [b/s]')
thpt_axis.legend()

latency_axis.plot(time_data, latency_data0, label='Latency Sender0 [s]')
latency_axis.plot(time_data, latency_data1, label='Latency Sender1 [s]')
latency_axis.plot(time_data, min_latency, label='Min Latency [s]')
latency_axis.plot(time_data, max_latency, label='Max Latency [s]')
latency_axis.legend()

loss_axis.plot(time_data, loss_data0, label='Loss Rate Sender0')
loss_axis.plot(time_data, loss_data1, label='Loss Rate Sender1')
loss_axis.legend()
loss_axis.set_xlabel("Monitor Interval")

fig.suptitle("Summary Graph for %s" % sys.argv[1])
fig.savefig(fig_filename)
