# coding: utf-8

import subprocess
import time



tasks = ['shooting', 'kuka_grasp', 'halfcheetah_rand', 'halfcheetah_low']
task = 'halfcheetah_opt'

#time.sleep(60*120)

#DONE: 0, 999, 50, 100

for i in [0,4]:
    trial = str(i)
    gpu = str(i%2)

    c1 = 'python al_select.py --task='+task+' --trial='+trial+' --gpu='+gpu
    c2 = 'python train_f.py --task='+task+' --trial='+trial+' --gpu='+gpu


    c3 = 'python train_model.py --task='+task+' --trial='+trial+' --gpu='+gpu+' --mode=tran'
    c4 = 'python train_zhang.py --task='+task+' --trial='+trial+' --gpu='+gpu


    c3 = 'python al_select.py --mode=add_noise --task='+task+' --trial='+trial+' --gpu='+gpu+' --alpha_anno=0.04 --alpha_tran=0.01'
    c4 = 'python al_select.py --mode=add_noise --task='+task+' --trial='+trial+' --gpu='+gpu+' --alpha_anno=0.15 --alpha_tran=0.04'
    c5 = 'python al_select.py --mode=add_noise --task='+task+' --trial='+trial+' --gpu='+gpu+' --alpha_anno=0.3 --alpha_tran=0.1'

    c6 = 'python train_f.py --mode=noisy-train --task='+task+' --trial='+trial+' --gpu='+gpu+' --alpha_anno=0.04 --alpha_tran=0.01'
    c7 = 'python train_f.py --mode=noisy-train --task='+task+' --trial='+trial+' --gpu='+gpu+' --alpha_anno=0.15 --alpha_tran=0.04 --aug_steps=50'
    

    c1 = 'python train_model.py --task='+task+' --trial='+trial+' --gpu='+gpu
    c2 = 'python train_f.py --mode=train --task='+task+' --trial='+trial+' --gpu='+gpu+' --star=1'

    cs = [c1,c2]

    for c in cs:
        print("@@@ command: ", c)
        result = subprocess.call(c.split(" "))
        print(result)
        time.sleep(30)

