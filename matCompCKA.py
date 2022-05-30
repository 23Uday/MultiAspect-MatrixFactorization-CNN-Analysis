import os
import torch
import torch.nn as nn
import torch.nn.functional as FU
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pandas as pd
from skimage import io, transform
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import matplotlib
#matplotlib.use('agg')
#matplotlib.pyplot.switch_backend('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils
import argparse
import glob
import PIL
from PIL import Image
import pdb
# from calsSC import cals
from cnmf import cnmf
from numpy import dot, zeros, array, eye, kron, prod
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
import math
from sklearn.preprocessing import normalize
#import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import pickle
from torch.utils.data.sampler import SequentialSampler, RandomSampler, SubsetRandomSampler
from collections import Counter
import random
import pdb
from PIL import Image, ImageDraw, ImageFont, ImageColor
# from parafac2ALS import als
from scipy.optimize import curve_fit
import shutil
# Ignore warnings
import warnings
# from torch._six import inf
from bisect import bisect_right
from functools import partial
from scipy import signal,misc
import copy
import operator
import networkx as nx
from networkx.algorithms import bipartite
from collections import Counter
from itertools import combinations
from tools import analyzeFKNN, analyzeFKNNPlots1, neuralEvalAnalysis, analyzeFKNNPlots2

parser = argparse.ArgumentParser() # argparser object
parser.add_argument("matDir1", help = "Enter the path of folder which containts the first matrix :",type = str)
parser.add_argument("matDir2", help = "Enter the path of folder which containts the second matrix :",type = str)
parser.add_argument("matDir3", help = "Enter the path of folder which containts the second matrix :",type = str)
parser.add_argument("matDir4", help = "Enter the path of folder which containts the second matrix :",type = str)
# parser.add_argument("outputFolderName", help = "Enter the name(Path) of the Output Folder :", type = str)
args = parser.parse_args()

matDir1 = args.matDir1
matDir2 = args.matDir2
matDir3 = args.matDir3
matDir4 = args.matDir4

def HSIC(K,L):
	r,c = K.shape
	H = np.eye(r) - 1/c * np.ones((r,c))
	return np.trace(H@K@H@H@L@H)/(r-1)**2

def HSICNC(K,L):
	r,c = K.shape
	H = np.eye(r) - 1/c * np.ones((r,c))
	return np.trace(K@H@L@H)/(r-1)**2

def CKA(K,L):
	return HSIC(K,L)/np.sqrt(HSIC(K,K)*HSIC(L,L))

### loading matrices ###
F1 = np.load(matDir1)
F2 = np.load(matDir2)
F3 = np.load(matDir3)
F4 = np.load(matDir4)
F = [F1.T@F1, F2.T@F2, F3.T@F3, F4.T@F4]
L = range(0,len(F))
pairwiseIndex = list(combinations(L,2))

C = np.zeros((4,4))

for index in L:
	C[index,index] = CKA(F[index], F[index])

for pair in pairwiseIndex:
	C[pair[0],pair[1]] = CKA(F[pair[0]],F[pair[1]])
	C[pair[1],pair[0]] = CKA(F[pair[1]],F[pair[0]])
print(C)
pdb.set_trace()
# array([[1.        , 0.96090907, 0.96129191, 0.95527595],
# 	   [0.96090907, 1.        , 0.96356171, 0.95881191],
# 	   [0.96129191, 0.96356171, 1.        , 0.9627601 ],
# 	   [0.95527595, 0.95881191, 0.9627601 , 1.        ]])