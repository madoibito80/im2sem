# coding: utf-8
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np
import cupy as cp
from PIL import Image
from PIL import ImageDraw

import sys
sys.path.append(".")
import util


def buffer_visualize(dump_file, task):

    print(dump_file)
    with open(dump_file, "rb") as f:
            (q_state, _, _, q_is_end, _, q_sem_state, _) = pickle.load(f)
        

    
    print('shape: ',q_state.shape)
    """
    mi = np.min(q_sem_state, axis=0)
    ma = np.max(q_sem_state, axis=0)
    mi2 = np.min(q_sem_state_next, axis=0)
    ma2 = np.max(q_sem_state_next, axis=0)
    #mean = np.mean(q_sem_state, axis=0)
    #std = np.std(q_sem_state, axis=0)

    print(mi)
    print(ma)
    print(mi2)
    print(ma2)
    """

    IMG_SIZE = q_state.shape[2]
    print(IMG_SIZE)
    n_ch = int(q_state.shape[1]/3)
    dst = Image.new('RGB', ((IMG_SIZE+2)*10, (IMG_SIZE+2)*int(q_state.shape[0]/10)),(0,0,128))

    pos = 0
    for n_sample in range(700,q_state.shape[0]):
        for ch in range(n_ch):
            state = (q_state[n_sample,ch*3:(ch+1)*3]).astype(np.uint8).transpose((1,2,0))
            state = Image.fromarray(state)
            x = (pos%10)*(IMG_SIZE+2)
            y = int(pos/10)*(IMG_SIZE+2)
            dst.paste(state, (x, y))
            draw = ImageDraw.Draw(dst)
            #draw.text((x,y),str(round(rs[n_sample], 3)))
            #draw.text((x,y),str(round(states_sensor[n_sample,0], 3))) # 撃破したときに敵の位置を出力しているからnanになる
            draw.text((x,y),str(n_sample),fill=(0,0,0))
            if q_is_end[n_sample] == 1.:
                draw.text((x,y+10),"E",fill=(255,0,0))
            pos += 1

    dst.save("./buffer.png")



def buffer_visualize2(dump_file):

    print(dump_file)

    IMG_SIZE = 64
    dst = Image.new('RGB', ((IMG_SIZE+2)*10, (IMG_SIZE+2)*int(1000/10)),(0,0,128))

    
    pos = 0
    f = open(dump_file, "rb")
    for n_sample in range(1000):
        states = pickle.load(f)
        states = states.reshape(states.shape[1:])
        n_ch = int(states.shape[0]/3)
        for ch in range(n_ch):
            state = (states[ch*3:(ch+1)*3]).copy().astype(np.uint8).transpose((1,2,0))
            print(state.shape)
            #print(np.min(state))
            #print(np.max(state))
            #print(type(state))
            #exit()
            state = Image.fromarray(state)
            x = (pos%10)*(IMG_SIZE+2)
            y = int(pos/10)*(IMG_SIZE+2)
            dst.paste(state, (x, y))
            draw = ImageDraw.Draw(dst)
            #draw.text((x,y),str(round(rs[n_sample], 3)))
            #draw.text((x,y),str(round(states_sensor[n_sample,0], 3))) # 撃破したときに敵の位置を出力しているからnanになる
            draw.text((x,y),str(n_sample))
            pos += 1

    dst.save("./buffer.png")



def source_policy_lcurve(dump_file):

	with open(dump_file,"rb") as f:
		lcurve = pickle.load(f)

	y = np.mean(lcurve["reward"], axis=1)
	plt.plot(y)
	plt.savefig("./lcurve.png")



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='buffer')
    parser.add_argument('--env', type=str, default='vizdoom')
    parser.add_argument('--task', type=str, default='shooting')
    parser.add_argument('--trial', type=str, default='0')
    parser.add_argument('--nfile', type=str, default='0')
    args = parser.parse_args()

    if args.mode == 'buffer':
        dump_file = '../dump/'+args.task+'/offline/'+args.nfile
        buffer_visualize(dump_file, args.task)

    if args.mode == 'minmax':
        dirn = '../dump/'+args.task+'/'+args.trial+'/buffer/source'
        min_max(dirn)
    
    if args.mode == 'policy':
        dump_file = '../dump/'+args.task+'/'+args.trial+'/policy/lcurve'
        source_policy_lcurve(dump_file)



    if args.mode == 'policy_buffer':
        dump_file = '../dump/'+args.task+'/policy/traj'
        buffer_visualize2(dump_file)