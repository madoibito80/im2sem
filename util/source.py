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


def run_policy(env, policy, dump_dir, test=False, n_repeat=5):

    lcurve = {"reward":[]}
    buffer = util_policy.Buffer(dump_dir=dump_dir)


    for phase in range(1000):
        if not test:
            policy.ask()

        results = []

        for pop in range(policy.popsize):
            print(phase, pop)

            if test: # always exec mean of CMA-ES
                pop = -1

            r = 0.
            for i in range(n_repeat):

                cont = util_policy.play_episode(env, lambda x: policy.policy(x, pop))
                buffer.push(cont)
                p_reward = cont[4]
                r += float(np.sum(p_reward)) / n_repeat


            results.append(r)
            print("reward: ", results[-1])
            

        if not test:
            # only for checking performans of MEAN so we dont update CMA-ES
            policy.tell(results)

        ## dump
        lcurve["reward"].append(results)
        with open(dump_dir+"/policy/cma-"+str(20*(int(phase/20)+1)), "wb") as f:
            pickle.dump(policy, f)

        with open(dump_dir+"/policy/lcurve", "wb") as f:
            pickle.dump(lcurve, f)





def main(): 

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='vizdoom')
    parser.add_argument('--task', type=str, default='shooting')
    parser.add_argument('--trial', type=str, default='0')
    args = parser.parse_args()
    print(args.env, args.task)

    env = util_policy.Environment(envtype=args.env, task=args.task)
    policy = policies.CMAES(dim_in=env.dim_in, dim_out=env.dim_out, popsize=64)

    # prepare buffer dir
    dump_dir = '../dump/'+args.task+'/'+args.trial
    print(dump_dir)
    try:
        os.mkdir(dump_dir)
        os.mkdir(dump_dir+'/buffer')
    except:
        print("except")

    try:
        os.mkdir(dump_dir+'/policy')
        os.mkdir(dump_dir+'/buffer/source')
    except:
        raise Exception

    run_policy(env, policy, dump_dir+'/buffer/source', test=False)

    # train_policy and save buffer
    # data structure
    # - 1000 steps per file
    # - each file has a list with size 7
    # - elements are [state, state_next, acton, is_end, reward, sem_state, sem_state_next]
    # - numpy array with shape (1000, d)
    




if __name__ == '__main__':
    main()