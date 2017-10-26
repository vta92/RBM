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

def matrix_rep(data, total_users, total_movies):
    matrix=[[0 for i in range(total_movies)] for i in range(total_users)]
    
    for row in range(total_users):
        for col in range(total_movies):
            user_id, movie_id, rating = data[row][0], data[row][1], data[row][2]
            matrix[user_id-1][movie_id-1] = rating
    return matrix

#unit test for matrix population
#a = matrix_rep(train_set1, total_users, total_movies)
