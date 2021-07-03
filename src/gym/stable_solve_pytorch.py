import os
import sys
import inspect
from argparse import ArgumentParser, Namespace
import gym
from pprint import pprint
from stable_baselines3 import PPO

import network_sim


def common_args() -> Namespace:
    parser = ArgumentParser()
    args = parser.parse_args()
    return args


def make_args(args: Namespace) -> Namespace:
    args.currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    args.parentdir  = os.path.dirname(args.currentdir)

    sys.path.insert(0, args.parentdir)
    from common.simple_arg_parse import arg_or_default
    args.arch_str = arg_or_default("--arch", default="32,16")

    return args


if __name__ == '__main__':
    args = common_args()
    args = make_args(args)

    pprint(args.__dict__)

    env = gym.make('PccNs-v0')

    model = PPO('MlpPolicy', env, verbose=2)
    print(model)

    for i in range(0, 6):
        model.learn(total_timesteps=(1600 * 410))
        model.save(f"./pcc_model_{i}.ckpt")
