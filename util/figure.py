# coding: utf-8
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np
import cupy as cp
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import sys
sys.path.append(".")
import util


def collect():
    import util_policy
    import policies

    env = util_policy.Environment(envtype='pybullet', task='kuka_grasp', optp=True, lowp=False)
    s = 128
    env.img_size = s
    env.env._width = s
    env.env._height = s

    policy = policies.collect_wrapper(policies.hand_crafted_kuka_grasp)
    policy.new_eps()
    cont1 = util_policy.play_episode(env, policy)

    policy = policies.behaviour_kuka_grasp()
    policy.new_eps()
    cont2 = util_policy.play_episode(env, policy)

    f = open("./figure_episode.pickle", "wb")
    pickle.dump(cont1, f)
    pickle.dump(cont2, f)
    f.close()


#collect()
#exit()

def ponch():

    task = 'kuka_grasp'
    dump_file = './figure_episode.pickle'
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


    IMG_SIZE = 128
    n_ch = 3
    chs = [1]

    space1 = 60
    dst = Image.new('RGBA', ((IMG_SIZE+space1)*5-35, IMG_SIZE*2+30),0)

    f = open('./figure_episode.pickle', 'rb')
    for i in range(1):
        (q_state, q_state_next, _, q_is_end, _, q_sem_state, _) = pickle.load(f)
        
        for n_sample in range(0,5):
            t = n_sample * 8
            if n_sample == 4:
                t = 39
            for ch in chs:
                state = (q_state_next[t,ch*3:(ch+1)*3]).astype(np.uint8).transpose((1,2,0))
                state = Image.fromarray(state)
                x = n_sample*(IMG_SIZE+space1) + i*((IMG_SIZE+space1)*5+100) + 10
                y = ch*(IMG_SIZE)+20
                dst.paste(state, (x, y))
                draw = ImageDraw.Draw(dst)

            ex = round(q_sem_state[t,0],2)
            ey = round(q_sem_state[t,1],2)
            ez = round(q_sem_state[t,2],2)
            text = "["+str(ex)+", "+str(ey)+", ...]"
            font = ImageFont.truetype(font_path, 15)
            draw.text((x+5,10),str(text),fill=(0,0,0), font=font)

    f.close()
    dst.save("./buffer.png")


ponch()
exit()



def ponch2():

    tasks = ['shooting', 'kuka_grasp', 'halfcheetah_rand']

    c = 0

    IMG_SIZE = 64
    space1 = 5
    dst = Image.new('RGB', ((IMG_SIZE+space1)*7-space1, IMG_SIZE),(255,255,255))

    chs = [1,3,3]
    inxs = [0,0,0]

    for i in range(len(tasks)):
        task = tasks[i]
        dump_file = '../dump/'+task+'/offline/0'
        with open(dump_file, 'rb') as f:
            (q_state, q_state_next, _, q_is_end, _, q_sem_state, _) = pickle.load(f)
        n_ch = int(q_state.shape[1]/3)

        inx = inxs[i]
        for ch in range(chs[i]):
            state = (q_state[inx,ch*3:(ch+1)*3]).astype(np.uint8).transpose((1,2,0))
            state = Image.fromarray(state)
            x = c*(IMG_SIZE+space1)
            y = 0
            dst.paste(state, (x, y))
            draw = ImageDraw.Draw(dst)

            text = ["Shooting", 'KUKA', 'HalfCheetah'][i]
            #draw.text((x,1),str(text),fill=(0,0,0))
            #draw.text((x,12),"cam-"+str(ch+1),fill=(0,0,0))

            c += 1


    dst.save("./buffer.png")




ponch2()
exit()