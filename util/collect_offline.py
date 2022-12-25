# coding: utf-8

import argparse
import sys
import os
import numpy as np
from PIL import Image
import pickle
import gym

sys.path.append('./')
import policies
import util_policy




def main(): 

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='shooting')
    parser.add_argument('--env', type=str, default='vizdoom')
    parser.add_argument('--buffer', type=str, default='source')
    parser.add_argument('--optp', type=bool, default=False) # わざわざFalse指定するとTrueになる．指定自体がTF
    parser.add_argument('--lowp', type=bool, default=False) # わざわざFalse指定するとTrueになる．指定自体がTF
    args = parser.parse_args()
    print(args.env, args.task)

    env = util_policy.Environment(envtype=args.env, task=args.task, optp=args.optp, lowp=args.lowp)

    # prepare buffer dir
    dump_dir = '../dump/'+args.task
    print(dump_dir)
    try:
        os.mkdir(dump_dir)
    except:
        print("except")

    try:
        os.mkdir(dump_dir+'/'+args.buffer)
    except:
        raise Exception

    if args.task == 'shooting':
        if args.buffer == 'source':
            max_files = 50
        elif args.buffer == 'offline':
            max_files = 10
        elif args.buffer == 'test_b':
            max_files = 5
        elif args.buffer == 'test_o':
            max_files = 5

    elif args.task == 'kuka_grasp':
        if args.buffer == 'source':
            max_files = 50
        elif args.buffer == 'offline':
            max_files = 10
        elif 'test' in args.buffer:
            max_files = 5

    elif 'halfcheetah' in args.task:
        if args.buffer == 'source' or args.buffer == 'offline':
            max_files = 100
        elif 'test' in args.buffer:
            max_files = 20

    else:
        raise NotImplementedError

    buffer = util_policy.Buffer(dump_dir=dump_dir+'/'+args.buffer, max_files=max_files)


    if args.task == 'shooting':
        if args.buffer == 'test_o':
            policy = policies.collect_wrapper(policies.hand_crafted_shooting)
        else:
            policy = policies.behaviour_shooting()

    if args.task == 'kuka_grasp':
        if args.buffer == 'test_o':
            policy = policies.collect_wrapper(policies.hand_crafted_kuka_grasp)
        else:
            policy = policies.behaviour_kuka_grasp()

    if 'halfcheetah' in args.task:
        policy = env.env



    for i in range(10000):
        policy.new_eps()
            
        print(i)
        cont = util_policy.play_episode(env, policy)
        flg = buffer.push(cont)
        if not flg:
            break




if __name__ == '__main__':
    main()