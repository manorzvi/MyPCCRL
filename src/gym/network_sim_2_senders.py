from abc import ABC

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
import heapq
import time
import random
import json
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from common import sender_obs, config
from common.simple_arg_parse import arg_or_default
from loguru import logger

MAX_RATE = 1000
MIN_RATE = 40

REWARD_SCALE = 0.001

MAX_STEPS = 400

EVENT_TYPE_SEND = 'S'
EVENT_TYPE_ACK = 'A'

BYTES_PER_PACKET = 1500

LATENCY_PENALTY = 1.0
LOSS_PENALTY = 1.0


class Link:

    def __init__(self, bandwidth, delay, queue_size, loss_rate):
        self.bw = float(bandwidth)
        self.dl = delay
        self.lr = loss_rate
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0
        self.max_queue_delay = queue_size / self.bw

        # logger.info(f'bandwidth={bandwidth}|delay={delay}|max_queue_delay={self.max_queue_delay}|loss_rate={loss_rate}')

    def get_cur_queue_delay(self, event_time):
        return max(0.0, self.queue_delay - (event_time - self.queue_delay_update_time))

    def get_cur_latency(self, event_time):
        return self.dl + self.get_cur_queue_delay(event_time)

    def packet_enters_link(self, event_time):
        if (random.random() < self.lr):
            return False
        self.queue_delay = self.get_cur_queue_delay(event_time)
        self.queue_delay_update_time = event_time
        extra_delay = 1.0 / self.bw
        # print("Extra delay: %f, Current delay: %f, Max delay: %f" % (extra_delay, self.queue_delay, self.max_queue_delay))
        if extra_delay + self.queue_delay > self.max_queue_delay:
            # print("\tDrop!")
            return False
        self.queue_delay += extra_delay
        # print("\tNew delay = %f" % self.queue_delay)
        return True

    def print_debug(self):
        print("Link:")
        print("Bandwidth: %f" % self.bw)
        print("Delay: %f" % self.dl)
        print("Queue Delay: %f" % self.queue_delay)
        print("Max Queue Delay: %f" % self.max_queue_delay)
        print("One Packet Queue Delay: %f" % (1.0 / self.bw))

    def reset(self):
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0


class Network:

    def __init__(self, senders, links, throughput_coef: float, latency_coef: float, loss_coef: float, special_loss):
        self.q = []
        self.cur_time = 0.0
        self.senders = senders
        self.links = links

        self.throughput_coef = throughput_coef
        self.latency_coef = latency_coef
        self.loss_coef = loss_coef
        self.special_loss = special_loss

        self.queue_initial_packets()

    def queue_initial_packets(self):
        for sender in self.senders:
            sender.register_network(self)
            sender.reset_obs()
            heapq.heappush(self.q, (1.0 / sender.rate, sender, EVENT_TYPE_SEND, 0, 0.0, False))

    def run_for_dur(self, dur):

        for sender in self.senders:
            sender.reset_obs()

        end_time = self.cur_time + dur

        while self.cur_time < end_time:

            event_time, sender, event_type, next_hop, cur_latency, dropped = heapq.heappop(self.q)

            self.cur_time = event_time
            new_event_time = event_time
            new_event_type = event_type
            new_next_hop = next_hop
            new_latency = cur_latency
            new_dropped = dropped

            push_new_event = False

            if event_type == EVENT_TYPE_ACK:
                if next_hop == len(sender.path):
                    if dropped:
                        sender.on_packet_lost()
                    else:
                        sender.on_packet_acked(cur_latency)
                else:
                    new_next_hop = next_hop + 1
                    link_latency = sender.path[next_hop].get_cur_latency(self.cur_time)
                    new_latency += link_latency
                    new_event_time += link_latency
                    push_new_event = True
            if event_type == EVENT_TYPE_SEND:
                if next_hop == 0:
                    # print("Packet sent at time %f" % self.cur_time)
                    sender.on_packet_sent()
                    push_new_event = True
                    heapq.heappush(self.q,
                                   (self.cur_time + (1.0 / sender.rate), sender, EVENT_TYPE_SEND, 0, 0.0, False))
                else:
                    push_new_event = True

                if next_hop == sender.dest:
                    new_event_type = EVENT_TYPE_ACK
                new_next_hop = next_hop + 1

                link_latency = sender.path[next_hop].get_cur_latency(self.cur_time)
                new_latency += link_latency
                new_event_time += link_latency
                new_dropped = not sender.path[next_hop].packet_enters_link(self.cur_time)

            if push_new_event:
                heapq.heappush(self.q, (new_event_time, sender, new_event_type, new_next_hop, new_latency, new_dropped))

        throughputs = []
        latencys = []
        losses = []

        for sender in self.senders:
            sender_mi = sender.get_run_data()
            throughputs.append(sender_mi.get("recv rate"))
            latencys.append(sender_mi.get("avg latency"))
            losses.append(sender_mi.get("loss ratio"))

        if self.special_loss is None:
            reward = self.throughput_coef * sum(throughputs) / (8 * BYTES_PER_PACKET) + \
                     self.latency_coef * sum(latencys) + \
                     self.loss_coef * sum(losses)
        elif self.special_loss == 'fairness1':
            higher_throughput = throughputs[0] if throughputs[0] > throughputs[1] else throughputs[1]
            lower_throughput = throughputs[0] if throughputs[0] < throughputs[1] else throughputs[1]
            reward = self.throughput_coef * (sum(throughputs) - (higher_throughput-lower_throughput)) / (8 * BYTES_PER_PACKET) + \
                     self.latency_coef * sum(latencys) + \
                     self.loss_coef * sum(losses)
        elif self.special_loss == 'fairness2':
            higher_throughput = throughputs[0] if throughputs[0] > throughputs[1] else throughputs[1]
            lower_throughput = throughputs[0] if throughputs[0] < throughputs[1] else throughputs[1]
            reward = self.throughput_coef * ((higher_throughput + lower_throughput) /
                                             (higher_throughput - lower_throughput)) / (8 * BYTES_PER_PACKET) + \
                     self.latency_coef * sum(latencys) + \
                     self.loss_coef * sum(losses)

        return reward * REWARD_SCALE


class Sender:

    def __init__(self, rate, path, dest, features, history_len=10):

        self.id = Sender._get_next_id()
        self.starting_rate = rate
        self.rate = rate
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.bytes_in_flight = 0
        self.min_latency = None
        self.rtt_samples = []
        self.sample_time = []
        self.net = None
        self.path = path
        self.dest = dest
        self.history_len = history_len
        self.features = features
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)

        logger.info(f'Sender id={self.id}|starting_rate={self.starting_rate}|path={path}|dest={dest}')

    _next_id = 0

    @staticmethod
    def _get_next_id():
        id = Sender._next_id
        Sender._next_id += 1
        return id

    def apply_rate_delta(self, delta):
        delta *= config.DELTA_SCALE
        if delta >= 0.0:
            self.set_rate(self.rate * (1.0 + delta))
        else:
            self.set_rate(self.rate / (1.0 - delta))

    def register_network(self, net):
        self.net = net

    def on_packet_sent(self):
        self.sent += 1
        self.bytes_in_flight += BYTES_PER_PACKET

    def on_packet_acked(self, rtt):
        self.acked += 1
        self.rtt_samples.append(rtt)
        if (self.min_latency is None) or (rtt < self.min_latency):
            self.min_latency = rtt
        self.bytes_in_flight -= BYTES_PER_PACKET

    def on_packet_lost(self):
        self.lost += 1
        self.bytes_in_flight -= BYTES_PER_PACKET

    def set_rate(self, new_rate):
        self.rate = new_rate
        if self.rate > MAX_RATE:
            self.rate = MAX_RATE
        if self.rate < MIN_RATE:
            self.rate = MIN_RATE

    def record_run(self):
        smi = self.get_run_data()
        self.history.step(smi)

    def get_obs(self):
        return self.history.as_array()

    def get_run_data(self):
        obs_end_time = self.net.cur_time

        return sender_obs.SenderMonitorInterval(
            self.id,
            bytes_sent=self.sent * BYTES_PER_PACKET,
            bytes_acked=self.acked * BYTES_PER_PACKET,
            bytes_lost=self.lost * BYTES_PER_PACKET,
            send_start=self.obs_start_time,
            send_end=obs_end_time,
            recv_start=self.obs_start_time,
            recv_end=obs_end_time,
            rtt_samples=self.rtt_samples,
            packet_size=BYTES_PER_PACKET
        )

    def reset_obs(self):
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.rtt_samples = []
        self.obs_start_time = self.net.cur_time

    def print_debug(self):
        print("Sender:")
        print("Obs: %s" % str(self.get_obs()))
        print("Rate: %f" % self.rate)
        print("Sent: %d" % self.sent)
        print("Acked: %d" % self.acked)
        print("Lost: %d" % self.lost)
        print("Min Latency: %s" % str(self.min_latency))

    def reset(self):
        print(f"Resetting sender {self.id}!")
        self.rate = self.starting_rate
        self.bytes_in_flight = 0
        self.min_latency = None
        self.reset_obs()
        self.history = sender_obs.SenderHistory(self.history_len, self.features, self.id)

    def __gt__(self, sender2):
        return False


class SimulatedNetworkEnv(gym.Env, ABC):

    def __init__(
            self,
            bw=None, min_bw=100, max_bw=500,
            latency=None, min_latency=0.05, max_latency=0.5,
            queue=None, min_queue=0, max_queue=8,
            loss=None, min_loss=0.0, max_loss=0.05,
            min_send_rate_factor=0.3, max_send_rate_factor=1.5,
            throughput_coef=10.0, latency_coef= -1e3, loss_coef= -2e3,
            mi_len=None, episode_len=MAX_STEPS,
            special_loss=None,
            history_len=arg_or_default(
                "--history-len", default=10
            ),
            features=arg_or_default(
                "--input-features",
                default="sent latency inflation,latency ratio,send ratio"
            )
    ):

        if bw is None:
            self.min_bw, self.max_bw = min_bw, max_bw
        else:
            self.min_bw, self.max_bw = bw, bw
        if latency is None:
            self.min_lat, self.max_lat = min_latency, max_latency
        else:
            self.min_lat, self.max_lat = latency, latency
        if queue is None:
            self.min_queue, self.max_queue = min_queue, max_queue
        else:
            self.min_queue, self.max_queue = queue, queue
        if loss is None:
            self.min_loss, self.max_loss = min_loss, max_loss
        else:
            self.min_loss, self.max_loss = loss, loss

        self.min_send_rate_factor   = min_send_rate_factor
        self.max_send_rate_factor   = max_send_rate_factor

        self.throughput_coef        = throughput_coef
        self.latency_coef           = latency_coef
        self.loss_coef              = loss_coef

        self.special_loss           = special_loss

        self.mi_len                 = mi_len
        self.episode_len            = episode_len

        self.reward_sum = 0.0
        self.reward_ewma = 0.0
        self.episodes_run = -1
        self.done = True

        self.history_len = history_len
        self.features = features.split(",")

        self.action_space = spaces.Box(np.array([-1e12, -1e12]), np.array([1e12, 1e12]), dtype=np.float32)

        single_obs_min_vec = sender_obs.get_min_obs_vector(self.features)
        single_obs_max_vec = sender_obs.get_max_obs_vector(self.features)
        self.observation_space = spaces.Box(
            np.tile(single_obs_min_vec, self.history_len * 2),  # 2 for two senders
            np.tile(single_obs_max_vec, self.history_len * 2),  # 2 for two senders
            dtype=np.float32
        )

    def _get_all_sender_obs(self):

        sender_obs = []
        for i in range(0, 2):
            sender_obs.append(self.senders[i].get_obs())
        sender_obs = np.stack(sender_obs)
        sender_obs = np.array(sender_obs).reshape(-1, )

        return sender_obs

    def step(self, action):

        for i in range(0, 2):
            self.senders[i].apply_rate_delta(action[i])

        reward = self.net.run_for_dur(self.mi_len)
        for sender in self.senders:
            sender.record_run()
        self.steps_taken += 1

        event = {"Name": "Step", "Time": self.steps_taken, "Reward": reward}

        for i, sender in enumerate(self.senders):
            sender_mi = sender.get_run_data()

            event[f"Send Rate {i}"] = sender_mi.get("send rate")
            event[f"Throughput {i}"] = sender_mi.get("recv rate")
            event[f"Latency {i}"] = sender_mi.get("avg latency")
            event[f"Loss Rate {i}"] = sender_mi.get("loss ratio")
            event[f"Latency Inflation {i}"] = sender_mi.get("sent latency inflation")
            event[f"Latency Ratio {i}"] = sender_mi.get("latency ratio")
            event[f"Send Ratio {i}"] = sender_mi.get("send ratio")

        self.event_record["Events"].append(event)

        self.reward_sum += reward

        sender_obs = self._get_all_sender_obs()

        self.done = self.steps_taken >= self.episode_len

        return sender_obs, reward, self.done, {}

    def print_debug(self):
        print("---Link Debug---")
        for link in self.links:
            link.print_debug()
        print("---Sender Debug---")
        for sender in self.senders:
            sender.print_debug()

    def create_new_links_and_senders(self):
        bw = random.uniform(self.min_bw, self.max_bw)
        lat = random.uniform(self.min_lat, self.max_lat)
        queue = 1 + int(np.exp(random.uniform(self.min_queue, self.max_queue)))
        loss = random.uniform(self.min_loss, self.max_loss)

        logger.info(f'bw={bw}|lat={lat}|queue={queue}|loss={loss}')

        self.links = [
            Link(bw, lat, queue, loss),
            Link(bw, lat, queue, loss),
        ]
        self.senders = [
            Sender(random.uniform(self.min_send_rate_factor, self.max_send_rate_factor) * bw,
                   [self.links[0], self.links[1]], 0, self.features, history_len=self.history_len),
            Sender(random.uniform(self.min_send_rate_factor, self.max_send_rate_factor) * bw,
                   [self.links[0], self.links[1]], 0, self.features, history_len=self.history_len)
        ]

        if self.mi_len is None:
            self.mi_len = 3 * lat

    def reset(self):

        # if not self.done:
        #     raise ValueError('Agent called reset before the environment has done')  # TODO: Due to the agent existing a training iteration without finishing the episode
        self.done = False

        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links, throughput_coef=self.throughput_coef, latency_coef=self.latency_coef, loss_coef=self.loss_coef, special_loss=self.special_loss)

        self.steps_taken   = 0
        self.episodes_run += 1

        if self.episodes_run > 0 and self.episodes_run % 100 == 0:
            self.dump_events_to_file("pcc_env_log_run_%d.json" % self.episodes_run)

        self.event_record = {"Events": []}

        self.net.run_for_dur(self.mi_len)

        self.reward_ewma *= 0.99
        self.reward_ewma += 0.01 * self.reward_sum
        logger.info("Reward: %0.2f, Ewma Reward: %0.2f" % (self.reward_sum, self.reward_ewma))
        self.reward_sum = 0.0

        return self._get_all_sender_obs()

    def dump_events_to_file(self, filename):
        logger.info(f'dump events to file: {filename}')
        with open(filename, 'w') as f:
            json.dump(self.event_record, f, indent=4)


# Default mode (legacy)
register(id='PccNs-v10', entry_point='network_sim_2_senders:SimulatedNetworkEnv')
# Zero random loss set-up
register(id='PccNs-v11', entry_point='network_sim_2_senders:SimulatedNetworkEnv', kwargs={'loss': 0.0})
# Zero congestive loss set-up (very large queue)
register(id='PccNs-v12', entry_point='network_sim_2_senders:SimulatedNetworkEnv', kwargs={'queue': 8})
# Starting sending rate always equal to half the link bw
register(id='PccNs-v13', entry_point='network_sim_2_senders:SimulatedNetworkEnv', kwargs={'min_send_rate_factor': 0.5,
                                                                                          'max_send_rate_factor': 0.5})
# Same as PccNs-v13, but with no random loss at all
register(id='PccNs-v14', entry_point='network_sim_2_senders:SimulatedNetworkEnv', kwargs={'loss': 0.0,
                                                                                          'min_send_rate_factor': 0.5,
                                                                                          'max_send_rate_factor': 0.5})
# Same as PccNs-v14, but specifically with low Latency reward.
register(id='PccNs-v15', entry_point='network_sim_2_senders:SimulatedNetworkEnv', kwargs={'loss': 0.0,
                                                                                          'min_send_rate_factor': 0.5,
                                                                                          'max_send_rate_factor': 0.5,
                                                                                          'throughput_coef': 2.0,
                                                                                          'latency_coef': -1e3,
                                                                                          'loss_coef': -2e3})
# Same as PccNs-v14, but with long Monitor Intervals of 3[s] (regular mode is 3 * link latency)
register(id='PccNs-v16', entry_point='network_sim_2_senders:SimulatedNetworkEnv', kwargs={'loss': 0.0,
                                                                                          'min_send_rate_factor': 0.5,
                                                                                          'max_send_rate_factor': 0.5,
                                                                                          'mi_len': 3.0})
# Same as PccNs-v13, but with long episodes (800 MIs)
register(id='PccNs-v17', entry_point='network_sim_2_senders:SimulatedNetworkEnv', kwargs={'loss': 0.0,
                                                                                          'min_send_rate_factor': 0.5,
                                                                                          'max_send_rate_factor': 0.5,
                                                                                          'episode_len': 800})
# Same as PccNs-v11, but specifically with low Latency reward.
register(id='PccNs-v18', entry_point='network_sim_2_senders:SimulatedNetworkEnv', kwargs={'loss': 0.0,
                                                                                          'throughput_coef': 2.0,
                                                                                          'latency_coef': -1e3,
                                                                                          'loss_coef': -2e3})
# Same as PccNs-v11, but with special loss to enforce fairness between agents. The idea is to panelize the reward on the difference between the throughputs.
register(id='PccNs-v19', entry_point='network_sim_2_senders:SimulatedNetworkEnv', kwargs={'loss': 0.0,
                                                                                          'special_loss': 'fairness1'})
# Same as PccNs-v18, but with special loss to enforce fairness between agents
register(id='PccNs-v20', entry_point='network_sim_2_senders:SimulatedNetworkEnv', kwargs={'loss': 0.0,
                                                                                          'throughput_coef': 2.0,
                                                                                          'latency_coef': -1e3,
                                                                                          'loss_coef': -2e3,
                                                                                          'special_loss': 'fairness1'})
# Same as PccNs-v19, but with another special loss, designed to keep x-y in scale with x+y
register(id='PccNs-v21', entry_point='network_sim_2_senders:SimulatedNetworkEnv', kwargs={'loss': 0.0,
                                                                                          'special_loss': 'fairness2'})
# Same as PccNs-v20, but with another special loss, designed to keep x-y in scale with x+y
register(id='PccNs-v22', entry_point='network_sim_2_senders:SimulatedNetworkEnv', kwargs={'loss': 0.0,
                                                                                          'throughput_coef': 2.0,
                                                                                          'latency_coef': -1e3,
                                                                                          'loss_coef': -2e3,
                                                                                          'special_loss': 'fairness2'})
