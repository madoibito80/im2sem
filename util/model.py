# coding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L
import cupy as xp
import numpy as np


######################################################################


def vae_lossfunc(x, y, mean, logvar):

        #return F.mean_squared_error(x,y)


        # https://github.com/hardmaru/WorldModelsExperiments/blob/master/doomrnn/doomrnn.py

        rec_loss = F.square(x - y)
        rec_loss = F.mean(rec_loss, axis=0)
        rec_loss = F.sum(rec_loss)

        #kl = F.gaussian_kl_divergence(self.mean, self.logvar, reduce='no')
        kl = 1. + logvar - F.square(mean) - F.exp(logvar)
        kl *= -0.5
        kl = F.sum(kl, axis=1)
        kl = F.mean(kl, axis=0)
        """
        kl_loss = - 0.5 * F.sum((1 + self.logvar - F.square(self.mean) - F.exp(self.logvar)),axis=(1))

        kl_tolerance = 0.5

        kl_loss = F.relu(kl_loss-kl_tolerance*self.z_dim)+kl_tolerance*self.z_dim
        kl_loss = F.mean(kl_loss)
        """

        # gamma=5はshootingでちょっと敵が消えていた

        gamma = 50. #1000だとlatentがおわってる
        loss = gamma*rec_loss + 1.*kl

        return loss





class VAE(chainer.Chain):
# Recurrent World Models Facilitate Policy Evolution

    def __init__(self, z_dim=64, n_ch=1, img_size=64):
        super(VAE, self).__init__()
        with self.init_scope():

            self.st = int(img_size/64)
            self.z_dim = z_dim

            nc = [16,32,64]*2

            self.el1 = L.Convolution2D(3*n_ch,      nc[0]*n_ch,ksize=3,stride=2,groups=n_ch)
            self.el2 = L.Convolution2D(nc[0]*n_ch,  nc[1]*n_ch,ksize=3,stride=2,groups=n_ch)
            self.el3 = L.Convolution2D(nc[1]*n_ch,  nc[1]*n_ch,ksize=3,stride=1,groups=n_ch)
            self.el4 = L.Convolution2D(nc[1]*n_ch,  nc[1]*n_ch,ksize=3,stride=2,groups=n_ch)
            self.el5 = L.Convolution2D(nc[1]*n_ch,  nc[1]*n_ch,ksize=3,stride=1,groups=n_ch)
            self.el6 = L.Convolution2D(nc[1]*n_ch,  nc[2]*n_ch,ksize=3,stride=2,groups=n_ch)
            #self.el7 = L.Linear(nc[2]*n_ch,64)
            self.elm = L.Convolution2D(nc[2]*n_ch,  z_dim,ksize=1,stride=1,groups=n_ch)
            self.els = L.Convolution2D(nc[2]*n_ch,  z_dim,ksize=1,stride=1,groups=n_ch)

            #self.elm = L.Linear(64,self.z_dim)
            #self.els = L.Linear(64,self.z_dim)

            #self.dl1 = L.Linear(self.z_dim,64*n_ch)
            self.dl2 = L.Deconvolution2D(z_dim,   nc[2]*n_ch,ksize=4,stride=1,groups=n_ch)
            self.dl3 = L.Deconvolution2D(nc[2]*n_ch,    nc[1]*n_ch,ksize=4,stride=2,groups=n_ch)
            self.dl4 = L.Deconvolution2D(nc[1]*n_ch,    nc[1]*n_ch,ksize=4,stride=1,groups=n_ch)
            self.dl5 = L.Deconvolution2D(nc[1]*n_ch,    nc[1]*n_ch,ksize=4,stride=2,groups=n_ch)
            self.dl6 = L.Deconvolution2D(nc[1]*n_ch,    nc[0]*n_ch,ksize=4,stride=1,groups=n_ch)
            self.dl7 = L.Deconvolution2D(nc[0]*n_ch,    3*n_ch,ksize=4,stride=2,groups=n_ch)

            self.mean = None
            self.logvar = None




    def encode(self, x, use_gpu=True):

        h = self.el1(x)
        h = F.relu(h)
        h = self.el2(h)
        h = F.relu(h)
        h = self.el3(h)
        h = F.relu(h)
        h = self.el4(h)
        h = F.relu(h)
        h = self.el5(h)
        h = F.relu(h)
        h = self.el6(h)
        h = F.relu(h)
        #h = self.el7(h)
        #h = F.relu(h)
        #self.mean = self.el7(h)
        #return self.mean

        self.mean = self.elm(h)
        self.logvar = self.els(h)

        self.mean = F.reshape(self.mean, (x.shape[0],-1))
        self.logvar = F.reshape(self.logvar, (x.shape[0],-1))

        sigma = F.exp(self.logvar / 2.)

        batch_size = x.shape[0]

        ep = xp.random.randn(self.z_dim*batch_size).astype(xp.float32).reshape((batch_size,self.z_dim))
        ep = chainer.Variable(ep)

        if not use_gpu:
            ep.to_cpu()

        return ep*sigma + self.mean
        #return self.mean


    def decode(self, x):
        #h = self.dl1(x)
        #h = F.relu(h)
        h = F.reshape(x,(x.shape[0],x.shape[1],1,1))
        h = self.dl2(h)
        h = F.relu(h)
        h = self.dl3(h)
        h = F.relu(h)
        h = self.dl4(h)
        h = F.relu(h)
        h = self.dl5(h)
        h = F.relu(h)
        h = self.dl6(h)
        h = F.relu(h)
        h = self.dl7(h)
        h = F.sigmoid(h)
        return h

    def lossfunc(self, x, y):

        return vae_lossfunc(x, y, self.mean, self.logvar)












class Transition_Continuous(chainer.Chain):

    def __init__(self, s_d, a_d):
        super(Transition_Continuous, self).__init__()
        with self.init_scope():

            h_d = 1024
            self.fc1 = L.Linear(s_d+a_d, h_d)
            self.fc2 = L.Linear(h_d, h_d)
            self.fc3 = L.Linear(h_d, s_d)



    def forward(self, x, a):

        h = F.concat((x, a), axis=1)
        
        #h = F.clip(h, 0., 1.)
        h = self.fc1(h)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.relu(h)
        h = self.fc3(h)
        #h += x
        #h = F.sigmoid(h)

        return h


    
    def lossfunc(self, y, s_tp1):

        #x2 = F.sum(F.square(y - s_tp1), axis=1)
        #loss = F.sum(F.sqrt(x2))

        loss = F.mean_squared_error(y, s_tp1)


        #if xp.isnan(loss.data):
        #    print(x2, y, s_tp1)
        #    exit()
        return loss







class Phi(chainer.Chain):

    def __init__(self, dim_out, n_ch):
        super(Phi, self).__init__()
        with self.init_scope():

            #self.task = task
            self.el1 = L.Convolution2D(3*n_ch,32,ksize=4,stride=2)
            self.el2 = L.Convolution2D(32,64,ksize=4,stride=2)
            self.el3 = L.Convolution2D(64,128,ksize=4,stride=2)
            self.el4 = L.Convolution2D(128,256,ksize=4,stride=2)
            self.el5 = L.Linear(1024,1024)
            self.el6 = L.Linear(1024,dim_out)

            #self.bn1 = L.BatchNormalization(256)


    def forward(self, x):

        h = self.el1(x)
        h = F.relu(h)
        h = self.el2(h)
        h = F.relu(h)
        h = self.el3(h)
        h = F.relu(h)
        h = self.el4(h)
        h = F.relu(h)
        h = F.dropout(h,0.3)
        h = self.el5(h)
        h = F.relu(h)
        h = self.el6(h)
        #h = F.sigmoid(h)
        """
        if self.task == 'kuka_grasp':
            dum1 = xp.zeros((x.shape[0],4)).astype(xp.float32)
            dum2 = xp.zeros((x.shape[0],1)).astype(xp.float32)

            h1 = h[:,:2]
            h2 = h[:,2:]
            h = F.concat((h1,dum1),axis=1)
            h = F.concat((h,h2),axis=1)
            h = F.concat((h,dum2),axis=1)
        """
        return h

    def lossfunc(self, y, t):
        #l = F.sum(F.square(y - t), axis=1)
        #loss = F.sum(F.sqrt(l))
        #if self.task == 'kuka_grasp':
        #    y = F.concat((y[:,:2], y[:,6:8]),axis=1)
        #    t = F.concat((t[:,:2], t[:,6:8]),axis=1)
            
        loss = F.mean_squared_error(y,t)
        return loss





class Phi2(chainer.Chain):

    def __init__(self, dim_in, dim_out, encoder):
        super(Phi2, self).__init__()
        with self.init_scope():

            dim_h = 1024
            self.l1 = L.Linear(dim_in, dim_h)
            self.l2 = L.Linear(dim_h, dim_h)
            self.l3 = L.Linear(dim_h, dim_out)
            self.encoder = encoder

    def forward(self, x):

        h = self.encoder.encode(x)
        h = chainer.Variable(h.data)

        h = self.l1(h)
        h = F.relu(h)
        h = self.l2(h)
        h = F.relu(h)
        h = self.l3(h)
        h = F.sigmoid(h)
        return h

    def lossfunc(self, y, t):
        loss = F.mean_squared_error(y,t)
        return loss



class Phi3(chainer.Chain):

    def __init__(self, dim_in, dim_out, encoder):
        super(Phi3, self).__init__()
        with self.init_scope():

            self.dim_out = dim_out
            self.ksize = 64
            self.dim_h = self.dim_out*self.ksize
            self.l1 = L.Linear(dim_in, self.dim_h)
            self.l2 = L.Convolution2D(in_channels=self.dim_h, out_channels=self.dim_h, groups=dim_out, ksize=(1,1))
            self.l3 = L.Convolution2D(in_channels=self.dim_h, out_channels=self.dim_h, groups=dim_out, ksize=(1,1))
            self.l4 = L.Convolution2D(in_channels=self.dim_h, out_channels=dim_out, groups=dim_out, ksize=(1,1))
            self.encoder = encoder
            self.heaviside = 0

    def forward(self, x):

        bsize = x.shape[0]
        h = self.encoder.encode(x)
        h = chainer.Variable(h.data)

        h = self.l1(h)
        h = F.relu(h)
        h = F.reshape(h, (bsize, self.dim_h, 1, 1))
        h = self.l2(h)
        h = F.relu(h)
        h = self.l3(h)
        h = F.relu(h)
        h = self.l4(h)
        h = F.reshape(h, (bsize, self.dim_out))
        #h = F.sigmoid(h)

        try:
            #if self.heaviside > 0:
            #    h.data[:,-self.heaviside:] = (h.data[:,-self.heaviside:] > 0.5)
            None
        except:
            None

        return h

    def lossfunc(self, y, t):
        loss = F.mean_squared_error(y,t)
        return loss




class Discriminator(chainer.Chain):

    def __init__(self, dim_in):
        super(Discriminator, self).__init__()
        with self.init_scope():

            self.dim_in = dim_in
            dim_h = 512
            self.l1 = L.Linear(dim_in, dim_h)
            self.l2 = L.Linear(dim_h, dim_h)
            self.l3 = L.Linear(dim_h, 1)
            #self.bn1 = L.BatchNormalization(dim_h)


    def forward(self, z):

        h = self.l1(z)
        h = F.relu(h)
        h = self.l2(h)
        h = F.relu(h)
        h = self.l3(h)
        return h

    def _lossfunc(self, x, t):
        return F.sigmoid_cross_entropy(x, t)


    def lossfunc(self, x, label):

        t = label.data.reshape((x.shape[0],1)).copy()
        t = chainer.Variable(t.astype(xp.float32))

        #x = F.sigmoid(x)
        loss = F.mean_squared_error(t, x)
        return loss



######################################################################




class AE(chainer.Chain):
# Recurrent World Models Facilitate Policy Evolution

    def __init__(self, dim_out, task):
        super(AE, self).__init__()
        with self.init_scope():

            z_dim = 32

            self.el1 = L.Convolution2D(3,32,ksize=4,stride=2)
            self.el2 = L.Convolution2D(32,64,ksize=4,stride=2)
            self.el3 = L.Convolution2D(64,128,ksize=4,stride=2)
            self.el4 = L.Convolution2D(128,256,ksize=4,stride=2)
            self.el5 = L.Linear(1024,z_dim)

            dim_h = 256
            self.phi1 = L.Linear(z_dim, dim_h)
            self.phi2 = L.Linear(dim_h, dim_h)
            self.phi3 = L.Linear(dim_h, dim_out)

            self.dl1 = L.Linear(z_dim,1024)
            self.dl2 = L.Deconvolution2D(1024,128,ksize=5)
            self.dl3 = L.Deconvolution2D(128,64,ksize=5,stride=2)
            self.dl4 = L.Deconvolution2D(64,32,ksize=6,stride=2)
            self.dl5 = L.Deconvolution2D(32,3,ksize=6,stride=2)


    def forward(self, x):
        self.x = x
        h = self.el1(x)
        h = F.relu(h)
        h = self.el2(h)
        h = F.relu(h)
        h = self.el3(h)
        h = F.relu(h)
        h = self.el4(h)
        h = F.relu(h)
        self.z = self.el5(h)

        h = self.phi1(chainer.Variable(self.z.data))
        h = F.relu(h)
        h = self.phi2(h)
        h = F.relu(h)
        h = self.phi3(h)
        h = F.sigmoid(h)

        return h

    def decode(self, z):
        h = self.dl1(z)
        h = F.relu(h)
        h = F.reshape(h,(z.shape[0],1024,1,1))
        h = self.dl2(h)
        h = F.relu(h)
        h = self.dl3(h)
        h = F.relu(h)
        h = self.dl4(h)
        h = F.relu(h)
        h = self.dl5(h)
        h = F.sigmoid(h)

        return h



    def lossfunc(self):
        y = self.decode(self.z)
        loss = F.mean_squared_error(y, self.x)

        return loss