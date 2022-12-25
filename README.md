# Few-Shot Image-to-Semantics Translation for Policy Transfer in Reinforcement Learning
- This repository is the original (author's) implementation for the paper 'Few-Shot Image-to-Semantics Translation for Policy Transfer in Reinforcement Learning'.


### Contact
Rei Sato: reisato[at]bbo.cs.tsukuba.ac.jp



# Requirements
- See Dockerfile
- Specification of versions are indicated below:
	- ~~Chainer=7.8.0, CUDA=11.4, Cupy=9.4.0, cuDNN=8.2~~ (it cannot load previous pickle data)
	- Chainer=7.8.0, CUDA=11.2, Cupy=8.6.0, cuDNN=8.1
	- stable-baselines3==1.0, sb3-contrib==1.0
	- python 3.8(.5)






# Usage

Please replace [SSH-USER], [SSH-HOST] as some appropriate one.

### Preparation

- clone repository

` $ rsync -rv ~/desktop/fpt [SSH-USER]@[SSH-HOST]:~/`

- build docker image from Dockerfile

` $ docker build -t fpt_im_01 .`

- run docker

` $ docker run --gpus all -v $HOME:$HOME --workdir /home/[SSH-USER]/fpt/util -i -t fpt_im_01 /bin/bash `


### Provide augmented pairs
- collect offline dataset using behaviour policy

` $ python collect_offline.py --env=vizdoom --task=shooting --buffer=source `

(env,task) = {(vizdoom,shooting),(pybullet,kuka_grasp),(pybullet,halfcheetah)}

- train VAE and transition using offline dataset

` $ python train_model.py --mode=train --task=shooting --trial=0 `

- select annotation data, make pair data @ output: /dump/task/trial/f

` $ python al_select.py `


### For Noisy-pairs

- add noise to clean pairs with some alpha (random_point_anno, random_batch_anno, random_batch_tran)

` $ python al_select.py --mode=add_noise --task=shooting --trial=0 --gpu=0 --alpha_anno=0.005 --alpha_tran=0.001`


- show noisy-pairs

` $ python al_select.py --mode=show_pp --task=shooting --trial=0 --gpu=0 --alpha_anno=0.005 --alpha_tran=0.001`


- download scatter images

` $ scp [SSH-USER]@[SSH-HOST]:~/fpt/util/scatter_random_batch_FalseTrue.png ~/desktop`

` $ scp [SSH-USER]@[SSH-HOST]:~/fpt/util/scatter_random_batch_TrueFalse.png ~/desktop`

` $ scp [SSH-USER]@[SSH-HOST]:~/fpt/util/scatter_random_point_TrueFalse.png ~/desktop`




### Train hat-F

- train f using NOISY paired dataset

` $ python train_f.py --mode=noisy-train --task=shooting --trial=0 --gpu=0 --alpha_anno=0.005 --alpha_tran=0.001`



- augmentation-steps: train f using NOISY paired dataset

` $ python train_f.py --mode=noisy-train --task=halfcheetah_rand --trial=0 --gpu=0 --alpha_anno=0 --alpha_tran=0.04 --aug_steps=50`


- you can use multi-trial set script

` $ python job1.py` 



### Check hat-F

- check performance of hat F

` $ python check_f.py --task=shooting --noisy=True`

- with non-max aug_steps , check performance of hat F

` $ python check_f.py --task=halfcheetah_rand --noisy=True --aug_steps=0`



- plot table for w/o noise

` $ python plot.py --mode=table --task=shooting`


- plot table for with noise

` $ python plot.py --mode=table_noisy --task=shooting`




### Train F-star

F-star is a hat F using full offline data as paired data.

- train f-star 

` $ python train_f.py --mode=train --task=halfcheetah_opt --trial=0 --gpu=0 --star=1`


