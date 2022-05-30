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
from tools import analyzeFKNN, analyzeFKNNPlots1, neuralEvalAnalysis, analyzeFKNNPlots2
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, FullGrad, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser() # argparser object
parser.add_argument("algo", help = "Enter the name of algorithm to run :",type = str)
parser.add_argument("rootDir", help = "Enter the name of root folder which containts the subfolders, networks and dicts :",type = str)
parser.add_argument("inputImagePath", help = "Enter the path to the image that is to be intepreted :",type = str)
parser.add_argument("networkFile", help = "Enter the path to the Network :",type = str)
parser.add_argument("classDictFile", help = "Enter the path to the file which has classes :",type = str)
parser.add_argument("outputFolderName", help = "Enter the name(Path) of the Output Folder where the interpreted image must be saved:", type = str)
args = parser.parse_args()


algo = args.algo
rootDir = args.rootDir
inputImagePath = args.inputImagePath
networkFile = args.networkFile
classDictFile = args.classDictFile
outputFolderName = args.outputFolderName


try:
	os.makedirs(outputFolderName)
except:
	pass

test_batch_size = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Resnets
class BasicBlock(nn.Module):
	"""Basic Block for resnet 18 and resnet 34
	"""

	#BasicBlock and BottleNeck block 
	#have different output size
	#we use class attribute expansion
	#to distinct
	expansion = 1

	def __init__(self, in_channels, out_channels, stride=1):
		super().__init__()

		#residual function
		self.residual_function = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels * BasicBlock.expansion)
		)

		#shortcut
		self.shortcut = nn.Sequential()

		#the shortcut output dimension is not the same with residual function
		#use 1*1 convolution to match the dimension
		if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_channels * BasicBlock.expansion)
			)
		
	def forward(self, x):
		return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
	"""Residual block for resnet over 50 layers
	"""
	expansion = 4
	def __init__(self, in_channels, out_channels, stride=1):
		super().__init__()
		self.residual_function = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
			nn.BatchNorm2d(out_channels * BottleNeck.expansion),
		)

		self.shortcut = nn.Sequential()

		if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
				nn.BatchNorm2d(out_channels * BottleNeck.expansion)
			)
		
	def forward(self, x):
		return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
	
class ResNet(nn.Module):

	def __init__(self, block, num_block, num_classes=100):
		super().__init__()

		self.in_channels = 64

		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True))
		#we use a different inputsize than the original paper
		#so conv2_x's stride is 1
		self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
		self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
		self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
		self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
		self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512 * block.expansion, num_classes)

	def _make_layer(self, block, out_channels, num_blocks, stride):
		"""make resnet layers(by layer i didnt mean this 'layer' was the 
		same as a neuron netowork layer, ex. conv layer), one layer may 
		contain more than one residual block 
		Args:
			block: block type, basic block or bottle neck block
			out_channels: output depth channel number of this layer
			num_blocks: how many blocks per layer
			stride: the stride of the first block of this layer
		
		Return:
			return a resnet layer
		"""

		# we have num_block blocks per layer, the first block 
		# could be 1 or 2, other blocks would always be 1
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_channels, out_channels, stride))
			self.in_channels = out_channels * block.expansion
		
		return nn.Sequential(*layers)

	def forward(self, x):
		output = self.conv1(x)
		output = self.conv2_x(output)
		output = self.conv3_x(output)
		output = x1 = self.conv4_x(output)
		output = x2 = self.conv5_x(output)
		output = x3 = self.avg_pool(output)
		output = output.view(output.size(0), -1)
		output = self.fc(output)

		return FU.log_softmax(output)#,(x1,x2,x3) 

	def countPosLayers(self):
		return 3


# Helper Functions

def get_image(path):
	with open(os.path.abspath(path), 'rb') as f:
		with Image.open(f) as img:
			return img.convert('RGB') 

fname = inputImagePath.split('/')[-1]
img = get_image(inputImagePath)
classOfInput = inputImagePath.split('/')[-1].split('_')[2] # This is only valid for CIFAR-10.
# This will be changed later
# plt.imshow(img)
# pdb.set_trace()
# resize and take the center part of image to what our model expects
def get_input_transform():  
	transform_test = transforms.Compose([
		transforms.Resize((32, 32),interpolation=Image.NEAREST),
		transforms.ToTensor(),
		# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		# transforms.Normalize(tuple([float(x) for x in testMean]),tuple([float(x) for x in testVar]))
	])  

	return transform_test

def get_input_tensors(img):
	transf = get_input_transform()
	# unsqeeze converts single image to batch of 1
	return transf(img).unsqueeze(0)

print("*** Creating Cam Algo Dict ***")

methods = {"gradcam": GradCAM,
		 "scorecam": ScoreCAM,
		 "gradcam++": GradCAMPlusPlus,
		 "ablationcam": AblationCAM,
		 "xgradcam": XGradCAM,
		 "eigencam": EigenCAM,
		 "eigengradcam": EigenGradCAM,
		 "layercam": LayerCAM,
		 "fullgrad": FullGrad}

	

print("*** Creating Dict ***")

classIndexToClassNameDict = dict()


with open(os.path.join(rootDir,classDictFile),'r') as fH:
	lines = fH.readlines()
	for i,line in enumerate(lines):
		classIndexToClassNameDict.update({i:line.strip()})

classNameToClassIndexDict = {v: k for k, v in classIndexToClassNameDict.items()}

# pdb.set_trace()
print("*** Class Name and Class Index ***")
classIndexOfInput = classNameToClassIndexDict[classOfInput]
print(classOfInput,classIndexOfInput)

# pdb.set_trace()
print("*** Loading Network ***")
model = torch.load(os.path.join(rootDir,networkFile))
model = model.to(device)

# for name, m in model.named_modules():
#     print(name)
print("*** Loaded Network ***")

print("*** Layer to be analyzed ***")
target_layers = [model.conv5_x[-1]]

# pdb.set_trace()
### Convert loaded image to tensor
print("*** image to tensor creation loading ***")
inputTensor = get_input_tensors(img)
inputTensor = inputTensor.to(device)
print("*** image to tensor creation done ***")

print("*** Feeding the input image to the network ***")
out = model(inputTensor)


# pdb.set_trace()
print("*** Loading CAM - %s***"%algo)
CAM = methods[algo]
camObject = CAM(model=model, target_layers=target_layers, use_cuda=True)
targets = [ClassifierOutputTarget(classIndexOfInput)]
grayscale_cam = camObject(input_tensor=inputTensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]
print("*** Generating Explanation ***")
visualization = show_cam_on_image(np.float32(img)/255, grayscale_cam, use_rgb=True)

outputImg = Image.fromarray(visualization)
outputImg.save(os.path.join(outputFolderName,fname))
print("*** Done ***")

# pdb.set_trace()

