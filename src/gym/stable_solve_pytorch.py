from typing import List
import random
import numpy as np
from argparse import ArgumentParser, Namespace
import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from loguru import logger
import network_sim
import network_sim_2_senders


def common_args() -> Namespace:

    parser = ArgumentParser()
    parser.add_argument('--arch', default=[32, 16], type=List[int])
    parser.add_argument('--env', default='PccNs-v0', type=str)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    args.eval_env = args.env.split('-')[0] + '_eval-' + args.env.split('-')[1]
    return args


def set_seed(args: Namespace):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


if __name__ == '__main__':
    args = common_args()

    logger.info(args.__dict__)

    set_seed(args)

    env         = gym.make(args.env)
    eval_env    = gym.make(args.eval_env)

    policy_kwargs = dict(net_arch=[dict(pi=args.arch, vf=args.arch)])

    model = PPO('MlpPolicy', env, verbose=2, policy_kwargs=policy_kwargs)

    logger.info(f"Evaluate model: ./pcc_model_{0}.ckpt ... ")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    logger.info(f"... Done. Evaluation: mean_reward={mean_reward:.2f} +/- {std_reward}")

    logger.info(f"save model: ./pcc_model_{0}.ckpt ... ")
    model.save(f"./pcc_model_{0}.ckpt")
    logger.info(f"... Done.")

    for i in range(1, 11):
    # for i in range(1, 2):
        model.learn(total_timesteps=(1600 * 410))
        # model.learn(total_timesteps=(2 * 410))

        logger.info(f"Evaluate model: ./pcc_model_{i}.ckpt ... ")
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
        logger.info(f"... Done. Evaluation: mean_reward={mean_reward:.2f} +/- {std_reward}")

        logger.info(f"save model: ./pcc_model_{i}.ckpt ... ")
        model.save(f"./pcc_model_{i}.ckpt")
        logger.info(f"... Done.")

    del model

    logger.info(f"load model: ./pcc_model_{i}.ckpt ... ")
    model = PPO.load(f"./pcc_model_{i}.ckpt")
    logger.info(f"... Done.")
