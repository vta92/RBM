#an implementation of rbm using pytorch

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.optim as optimizer
import torch.nn as nn
import torch.nn.parallel as parallel
import torch.utils.data

#these files have no headers
movies=pd.read_csv('ml-1m/movies.dat', sep='::', header=None, encoding='latin-1', engine='python')
ratings=pd.read_csv('ml-1m/ratings.dat',sep='::', header=None, encoding='latin-1', engine='python')
users=pd.read_csv('ml-1m/users.dat', sep='::', header=None, encoding='latin-1', engine='python')

movies.columns=['Movie_ID','Name','Tags']
ratings.columns=['User_ID','Movie_ID','Rating','Timestamp']
users.columns=['User_ID','Gender','Age','Job_ID','Zip_Code']

train_set1 = pd.read_csv('ml-100k/u1.base',delimiter='\t',header=None, encoding='latin-1', engine='python')
test_set1 = pd.read_csv('ml-100k/u1.test', delimiter='\t', header=None, encoding='latin-1', engine='python')

train_set1.columns=['User_ID','Movie_ID','Rating','Timestamp']
test_set1.columns=['User_ID','Movie_ID','Rating','Timestamp']

#have to convert from pd to np array for processing later
train_set1 = np.array(train_set1,dtype='int')
test_set1 = np.array(test_set1,dtype='int')

#dropping the timestamps
train_set1= train_set1[:,:-1]
test_set1 = test_set1[:,:-1]

#assuming the max user id = total number of user in a given dataset, we have:
total_users = max(train_set1[:,0].max(), test_set1[:,0].max())
total_movies = max(train_set1[:,1].max(), test_set1[:,1].max())

#we have a 943x1682 matrix from the 2 variables above, with each cell contains a rating
#pytorch expects a list of list
def matrix_rep(data, total_users, total_movies):
    matrix=[[0.0 for i in range(total_movies)] for i in range(total_users)]
    
    for row in range(total_users):
        for col in range(total_movies):
            user_id, movie_id, rating = data[row][0], data[row][1], data[row][2]
            matrix[user_id-1][movie_id-1] = float(rating)
    return matrix

#unit test for matrix population
train_set1 = matrix_rep(train_set1, total_users, total_movies)
test_set1 = matrix_rep(test_set1, total_users, total_movies)


train_set1 = torch.FloatTensor(train_set1) #tensor input is list of list
test_set1 = torch.FloatTensor(test_set1)

#ratings need to be binary in result.
#torch tensor can't do >= or <= filtering

def filtering(tensor):
    
    for row in range(len(tensor)):
        for col in range(len(tensor[row])):
            
            if tensor[row][col] == 0:
                tensor[row][col] = -1
            elif tensor[row][col] <= 2:
                tensor[row][col] = 0
            else:
                tensor[row][col] = 1
    return tensor

#ratings need to be binary in result.
train_set1[train_set1 == 0] = -1
test_set1[test_set1 == 0] = -1

train_set1[train_set1 == 1] = 0
train_set1[train_set1 == 2] = 0
test_set1[test_set1 == 1] = 0
test_set1[test_set1 == 2] = 0

train_set1[train_set1 > 2] = 1
test_set1[test_set1 > 2] = 1


train_set1 = filtering(train_set1)
test_set1 = filtering(test_set1)


#implementation of RBM
class RBM(object):
    def __init__(self, num_vis, num_hid): #number of visible and hidden nodes
        self.weight = torch.randn(num_hid,num_vis) #mean,variance = (0,1) with said dimensions
        self.bias_vis = torch.randn(1, num_vis)
        self.bias_hid = torch.randn(1, num_hid) #fake dimension for torch to go through with biases/batches as num_hid.
        #one bias for each hidden/vis nodes.
        
    def sample_hid(self,vis): #vis = visible neuron v in the prob. of p(v) given h
        activation_hid = torch.mm(vis, self.weight.t()) + self.bias_hid.expand_as(torch.mm(vis, self.weight.t()))
        prob_v_from_hid = torch.sigmoid(activation_hid) #sigmoid is used for our prob calculation
        return prob_v_from_hid, torch.bernoulli(prob_v_from_hid) #bernoulie result of 0/1 for our probability of our 1d batch prob tensor

        
    #based off gibb's sampling. using sigmoid with W*vis+bias
    def sample_vis(self,hid): #vis = visible neuron v in the prob. of p(h) given v
        activation_vis = torch.mm(hid, self.weight) + self.bias_vis.expand_as(torch.mm(hid, self.weight))
        #make sure we expand our bias so that each of the element in our batch has a bias
        prob_h_from_vis = torch.sigmoid(activation_vis) #sigmoid is used for our prob calculation
        return prob_h_from_vis, torch.bernoulli(prob_h_from_vis) #bernoulie result of 0/1 for our probability of our 1d batch prob tensor
        
        
    #contrastive divergence with k-sampling. Measure errors
    #minimize the energy = maximizing the log likelihood
    #refer to the attached paper for the training algorithm of contrastive div
    def training(self, vis0,vis_k,prob_h_vis0, prob_h_vis_k):#visible node, vis after k sampling
        self.weight +=torch.mm(vis0.t(),prob_h_vis0) - torch.mm(vis_k.t(), prob_h_vis_k) #transposing vis's, and differences in prob between sampling
        self.bias_vis += torch.sum((vis0-vis_k),0) #the 0 to keep the format as a 2d tensor        
        self.bias_hid += torch.sum((prob_h_vis0 - prob_h_vis_k),0)


nv = len(train_set1[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, total_users - batch_size, batch_size):
        vk = train_set1[id_user:id_user+batch_size]
        v0 = train_set1[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_hid(v0)
        for k in range(10):
            _,hk = rbm.sample_hid(vk)
            _,vk = rbm.sample_vis(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_hid(vk)
        rbm.training(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))



























