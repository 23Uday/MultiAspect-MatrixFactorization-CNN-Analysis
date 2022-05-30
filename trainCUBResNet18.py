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

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser() # argparser object
parser.add_argument("rootDir", help = "Enter the name of root folder which containts the data subfolders :",type = str)
parser.add_argument("rootDirTest", help = "Enter the name of root folder which containts the test data subfolders :",type = str)
# parser.add_argument("rootDirAdv", help = "Enter the name of root folder which containts the original adv data subfolders :",type = str)
# parser.add_argument("networkFile", help = "Enter the name of root folder which containts the Network :",type = str)
parser.add_argument("outputFolderName", help = "Enter the name(Path) of the Output Folder :", type = str)
parser.add_argument("NetworkName", help = "Enter the name(Path) of the network file :", type = str)
# parser.add_argument("Rank1", help = "Enter Rank 1 :", type = int)
# # parser.add_argument("Rank2", help = "Enter Rank 2 :", type = int)
# # parser.add_argument("Rank3", help = "Enter Rank 2 :", type = int)
# parser.add_argument("numGroups", help = "Enter the number of groups: ", type = int)
# parser.add_argument("maxIters", help = "Enter Max no. of iterations: ", type = int)
# # parser.add_argument("lmbdaSR", help = "Enter lambda S or R: ", type = float)
# # parser.add_argument("lmbdaSimplex", help = "Enter lambda Simplex :", type = float)
# parser.add_argument("lmbdaF", help = "Enter lambda F: ", type = float)
# parser.add_argument("lmbdaTV", help = "Enter lambda TV: ", type = float)
# parser.add_argument("lmbdaOrtho", help = "Enter orthogonality penalty: ", type = float)
parser.add_argument("samplingFactor", help = "Enter the ratio of dataset to be used: ", type = float)
parser.add_argument("samplingFactorTest", help = "Enter the ratio of dataset to be used: ", type = float)
parser.add_argument("lr", help = "learning rate for SGD to be used: ", type = float)
parser.add_argument("wd", help = "Enter the weight decay: ", type = float)
parser.add_argument("numEpochs", help = "Enter the number of epochs: ", type = int)
# parser.add_argument('opt',help = "Enter the optimization algorithm", type=str, default='sgd', choices=('sgd', 'adam', 'rmsprop'))
parser.add_argument("imgSize", help = "Enter the min dimension img size for the experiment", type = int, default = 32, choices = (32,64,128,192,224,256))
# parser.add_argument('P_init',help = "Enter the initialization for P", type=str, default='random', choices=('random', 'ID', 'NNDSVD'))
# parser.add_argument("classBased", help = "Enter if class based training is required ", type = str, default = 'False',choices = ('True','False'))
# # parser.add_argument("classListTrain", help = "Enter the list of classes, eg [1,2,3,4]: ",type = str,default = "['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']")
# # parser.add_argument("classListTest", help = "Enter the list of classes, eg [1,2,3,4]: ",type = str,default = "['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']")
# parser.add_argument("actOnlyMode", help = "Enter true if Data matrices have to be ignored", type = str, default = 'False',choices = ('True','False'))
# parser.add_argument("groupSparseF", help = "Enter true if F matrix is to have group sparsity", type = str, default = 'False',choices = ('True','False'))


args = parser.parse_args()


rootDir = args.rootDir
rootDirTest = args.rootDirTest

outputFolderName = args.outputFolderName
if outputFolderName == 'pwd':
	outputFolderName = os.getcwd()
NetworkName = args.NetworkName

samplingFactor = args.samplingFactor
samplingFactorTest = args.samplingFactorTest
if samplingFactor > 1:
	samplingFactor = 1.0
elif samplingFactor < 0:
	samplingFactor = 1.0


if samplingFactorTest > 1:
	samplingFactorTest = 1.0
elif samplingFactorTest < 0:
	samplingFactorTest = 1.0


lr = args.lr

wd = args.wd
numEpochs = args.numEpochs
# opt = args.opt
imgSize = args.imgSize

samplingAdd = 'samplingFactor'+str(samplingFactor)

outputFolderName =  outputFolderName+'samplingFactor'+str(samplingFactor)
outputFolderName = outputFolderName+"lr:%s-wd:%s"%(lr,wd)

try:
	os.makedirs(outputFolderName)
except:
	pass

cwd = os.getcwd()

train_batch_size = 32
test_batch_size = 1


### For testing data loader
# dataSet = '/home/usain001/research/DataSets/CifarLocal/train/'

class CUBDataset(Dataset):
	""" CIFAR 10 dataset """
	def __init__(self,root_dir, classToNum = {}, all_image_paths = [], all_labels = [],transform =None, sampleSize = 1.0):
		"""
		root_dir : The testing/training data set which contains subfolders.
		sampleSize needs to be between 0 and 1
		"""

		#Raw Data
		self.root_dir = root_dir
		self.transform = transform
		self.classToNum = classToNum
		# self.superClassToClassNum = {}
		self.classToSuperClassMap = {}
		self.superClassSet = {}
		self.all_labels = []
		self.all_Superlabels = []
		self.all_image_paths = glob.glob(root_dir+"/**/*.jpg", recursive=True)
		self.all_Classlabels = [fadd.split('/')[-2] for fadd in self.all_image_paths ]
		self.all_SuperClasslabels = [fadd.split('/')[-3] for fadd in self.all_image_paths ]
		# pdb.set_trace()
		for fadd in self.all_image_paths:
			if fadd.split('/')[-2] not in self.classToNum:
				self.classToNum.update({fadd.split('/')[-2]: len(self.classToNum)})
				# self.superClassToClassNum.update({fadd.split('/')[-3]: len(self.classToNum)})
				if fadd.split('/')[-3] not in self.superClassSet:
					self.superClassSet.update({fadd.split('/')[-3]:len(self.superClassSet)})

				### mapping the class to super class
				self.classToSuperClassMap.update({self.classToNum[fadd.split('/')[-2]]:self.superClassSet[fadd.split('/')[-3]] })


				self.all_labels.append(self.classToNum[fadd.split('/')[-2]])
				self.all_Superlabels.append(self.superClassSet[fadd.split('/')[-3]])
			else:
				self.all_labels.append(self.classToNum[fadd.split('/')[-2]])


				if fadd.split('/')[-3] not in self.superClassSet:
					self.superClassSet.update({fadd.split('/')[-3]:len(self.superClassSet)})


				self.all_Superlabels.append(self.superClassSet[fadd.split('/')[-3]])


		self.numToClass = {v: k for k, v in self.classToNum.items()}
		self.superClassSetReverse = {v: k for k, v in self.superClassSet.items()}
		self.classes = set(self.all_labels)

		# preprocessing
		self.counter = Counter(self.all_labels)
		self.sampleSize = sampleSize


		self.LabelToIndex = self.generateLabelsToIndex()
		self.sampledLabelToIndex = self.generateSampledLabelsToIndex()


		# Sampled Data
		self.all_sampled_idx = self.generateSampledIdx() 
		self.all_sampled_labels = [self.all_labels[i] for i in self.all_sampled_idx] # to be used for all labels
		self.all_sampled_super_labels = [self.all_Superlabels[i] for i in self.all_sampled_idx] # to be used for all labels
		self.sampled_counter = Counter(self.all_sampled_labels)
		self.all_sampled_image_paths = [self.all_image_paths[i] for i in self.all_sampled_idx]






	def __len__(self):
		""" returns the total number of files"""
		return len(self.all_sampled_image_paths)


	def __getitem__(self,idx):
		""" return image for the given index"""
		imgAdd = self.all_sampled_image_paths[idx]
		img = Image.open(imgAdd).convert('RGB')
		if self.transform == None:
			img = transforms.ToTensor()(img)
		else:

			img = self.transform(img)
		# pdb.set_trace(0)
		return (img,self.all_sampled_labels[idx],self.all_sampled_super_labels[idx],self.all_sampled_image_paths[idx])


	# Helper Functions	

	def getLabels(self):
		return self.all_labels

	def generateLabelsToIndex(self):
		LabelToIndex = {}
		for idx,label in enumerate(self.all_labels):
			if label not in LabelToIndex:
				LabelToIndex.update({label:[idx]})
			else:
				LabelToIndex[label].append(idx)
		return LabelToIndex


	def getLabelsToIndex(self):
		return self.LabelToIndex

	def generateSampledLabelsToIndex(self):
		sampledLabelToIndex = {}
		for label, idx_list in self.LabelToIndex.items():
			indices = random.sample(range(len(idx_list)), int(len(idx_list)*self.sampleSize))
			sampledLabelToIndex.update({label: [idx_list[i] for i in sorted(indices)]})

		return sampledLabelToIndex


	def generateSampledIdx(self):
		all_sampled_idx = []
		for label,idx_list in self.sampledLabelToIndex.items():
			all_sampled_idx += idx_list
		return all_sampled_idx

	def getInputChannels(self):
		return 3


def get_mean_and_std(dataset):
	'''Compute the mean and std value of dataset.'''
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
	mean = torch.zeros(3)
	std = torch.zeros(3)
	print('==> Computing mean and std..')
	for inputs, targets, st, ip in dataloader:
		for i in range(3):
			mean[i] += inputs[:,i,:,:].mean()
			std[i] += inputs[:,i,:,:].std()
	mean.div_(len(dataset))
	std.div_(len(dataset))
	return mean, std


# transform_train_stat = transforms.Compose([
# 		transforms.Resize(64),
# 		transforms.RandomCrop(64, padding=4),
# 		transforms.RandomHorizontalFlip(),
# 		transforms.RandomRotation(15),
# 		transforms.ToTensor(),
# 	])


# transform_test_stat = transforms.Compose([
# 		transforms.Resize(64),
# 		transforms.CenterCrop(64),
# 		transforms.ToTensor(),
# 	])

# trainMean,TrainVar = get_mean_and_std(CUBDataset(root_dir = rootDir, transform = transform_train_stat))
# testMean, testVar = get_mean_and_std(CUBDataset(root_dir = rootDirTest, transform = transform_test_stat))

# print("Training Data Mean : {mean}\t Training Data Variance : {var}\n".format(mean = trainMean, var = TrainVar))
# print("Testing Data Mean : {mean}\t Testing Data Variance : {var}\n".format(mean = testMean, var = testVar))


transform_train = transforms.Compose([
		transforms.Resize(imgSize),
		transforms.RandomCrop(imgSize, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.RandomRotation(15),
		transforms.ToTensor(),
		# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		# transforms.Normalize(tuple([float(x) for x in trainMean]),tuple([float(x) for x in TrainVar]))
	])


transform_test = transforms.Compose([
		transforms.Resize(imgSize),
		transforms.CenterCrop(imgSize),
		transforms.ToTensor(),
		# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		# transforms.Normalize(tuple([float(x) for x in testMean]),tuple([float(x) for x in testVar]))
	])

print("Training Data: Loading sampling based loader")
CUBtrain = CUBDataset(root_dir = rootDir,sampleSize = samplingFactor, transform = transform_train)
lenTrain = len(CUBtrain)

print("Probing Data: Loading sampling based loader")
CUBval1 = CUBDataset(root_dir = rootDirTest,sampleSize = samplingFactorTest, transform = transform_test)
lenVal1 = len(CUBval1)
numClasses = len(CUBtrain.classToNum)
print("Length Training Data = {lT}\tLength Testing Data = {lTe}\tNumber of Training Classes = {nC}".format(lT=lenTrain, lTe= lenVal1, nC = numClasses))


train_loader = torch.utils.data.DataLoader(dataset=CUBtrain,
										   batch_size=train_batch_size,
										   shuffle=True)

val1_loader = torch.utils.data.DataLoader(dataset=CUBval1,
										  batch_size=test_batch_size,
										  shuffle=False)

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

		return FU.log_softmax(output),(x1,x2,x3) 

	def countPosLayers(self):
		return 3


model = ResNet(BasicBlock, [2, 2, 2, 2], numClasses)
model = model.cuda()
print("*** Initialized Network ***")
	# model = AlexNet()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay = wd)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120,150,180], gamma=0.2)

def train(epoch):
	model.train()
	total_loss = 0
	for batch_idx, (data, target, superTarget, datapath) in enumerate(train_loader):
		# pdb.set_trace()
		# data, target = Variable(data), Variable(target)
		data, target = Variable(data).cuda(), Variable(target).cuda() # GPU
		optimizer.zero_grad()
		output,act = model(data)
		# pdb.set_trace()
		# output = model(data)
		loss = FU.nll_loss(output, target)
		total_loss += loss*len(data)
		loss.backward()
		optimizer.step()
		if batch_idx % 10 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

	print('Avg. Loss over the epoch: %f'%(total_loss/len(train_loader.dataset)))
	return total_loss/len(train_loader.dataset)




def testval1(): # add X-val later
	model.eval()
	val1_loss = 0
	correct = 0
	targetList = []
	supertargetList = []
	predList = []
	for data, target, superTarget, datapath in val1_loader:
		targetList.append(int(target))
		supertargetList.append(int(superTarget))
		# data, target = Variable(data, volatile=True), Variable(target)
		data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # gpu
		output,act = model(data)
		# output = model(data)
		# sum up batch loss
		# pdb.set_trace()
		val1_loss += FU.nll_loss(output, target).item()
		# get the index of the max log-probability
		pred = output.data.max(1, keepdim=True)[1]
		predList.append(int(pred.cpu().numpy()[0][0]))
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()

	val1_loss /= len(val1_loader.dataset)
	print('\nVal set 1: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		val1_loss, correct, len(val1_loader.dataset),
		100. * correct / len(val1_loader.dataset)))
	return val1_loss,100. * correct / len(val1_loader.dataset),predList


def numToClassText(numToClassDict,saveTo):
	with open(saveTo,'w') as fH:
		for i in range(len(numToClassDict)):
			fH.write(numToClassDict[i]+'\n')


netLossTraining = []
netLossVal1 = []
netAccVal1 = []
E = []


for epoch in range(0,numEpochs):
	epochFolder = 'Epoch Num %d'%epoch
	try:
		os.makedirs(os.path.join(outputFolderName,epochFolder))
	except:
		pass

	plt.figure()
	fig,axes = plt.subplots(1,1,figsize=(7,7))
	E.append(epoch) 
	trainLoss = train(epoch)
	scheduler.step()
	netLossTraining.append(float(trainLoss))
	# testval1()
	if CUBtrain.classToNum == CUBval1.classToNum: #classListTrain == classListTest or classBased=='False':
		testval1Loss, testval1Acc , predList = testval1()
		netLossVal1.append(float(testval1Loss))
		netAccVal1.append(float(testval1Acc))
	# else if classBased=='False':
	# 	testval1Loss, testval1Acc = testval1()
	# 	netLossVal1.append(float(testval1Loss))
	# 	netAccVal1.append(float(testval1Acc))


	axes.set_ylabel("NLL Loss")
	axes.set_xlabel("Epochs")
	axes.set_title("Loss vs Epochs")
	# axes[0].set_xlim([-1,1])
	# axes[0].set_ylim([-0.5,4.5])
	axes.semilogy(E, netLossTraining, color = 'r', linewidth=1, label = "Training Loss")
	if CUBtrain.classToNum == CUBval1.classToNum: #classListTrain == classListTest or classBased=='False':
		axes.semilogy(E, netLossVal1, color = 'g', linewidth=1, label = "Validation Loss") ## For testing validation, set the dictionary of val set the same as test set
	axes.legend(loc = "upper right")
	axes.autoscale()

	plt.savefig(os.path.join(outputFolderName,epochFolder,'LossVsEpochs'))
	plt.clf()


	
	# #Accuracy Plots
	if CUBtrain.classToNum == CUBval1.classToNum:#classListTrain == classListTest or classBased=='False':
		plt.figure()
		fig,axes = plt.subplots(1,1,figsize=(7,7))
		axes.set_ylabel("Validation Accuracy")
		axes.set_xlabel("Epochs")
		axes.set_title("Accuracy vs Epochs")
		# axes[0].set_xlim([-1,1])
		# axes[0].set_ylim([-0.5,4.5])
		axes.plot(E, netAccVal1, color = 'r', linewidth=1, label = "Validation Accuracy")
		# axes[1].semilogy(E, netLossVal1, color = 'g', linewidth=1, label = "Validation Loss")
		axes.legend(loc = "upper right")
		axes.autoscale()
		plt.savefig(os.path.join(outputFolderName,epochFolder,'AccuracyVsEpochs'))
		plt.clf()


	torch.save(model,os.path.join(outputFolderName,epochFolder,NetworkName+"_Epoch-%d"%epoch))
	numToClassText(CUBtrain.numToClass,os.path.join(outputFolderName,epochFolder,"numToClass.txt"))