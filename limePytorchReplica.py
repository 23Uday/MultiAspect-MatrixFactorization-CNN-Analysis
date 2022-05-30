import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json
from lime import lime_image
from skimage.segmentation import mark_boundaries
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as FU
import argparse

parser = argparse.ArgumentParser() # argparser object
parser.add_argument("netPath", help = "Enter the name of pretrained network file :",type = str)
parser.add_argument("imgPath", help = "Enter the name of image file to be analyzed :",type = str)
parser.add_argument("imgName", help = "Enter the name of analyzed image file :",type = str)
parser.add_argument("outPath", help = "Enter the name of folder where the analyzed image will be stored :",type = str)
args = parser.parse_args()
netPath = args.netPath
imgPath = args.imgPath
imgName = args.imgName
outPath = args.outPath

try:
	os.makedirs(outPath)
except:
	pass



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


def get_image(path):
	with open(os.path.abspath(path), 'rb') as f:
		with Image.open(f) as img:
			return img.convert('RGB') 
		
img = get_image(imgPath)
# plt.imshow(img)

# resize and take the center part of image to what our model expects
def get_input_transform():  
	transf = transforms.Compose([
		transforms.ToTensor()
	])    

	return transf

def get_input_tensors(img):
	transf = get_input_transform()
	# unsqeeze converts single image to batch of 1
	return transf(img).unsqueeze(0)


model = torch.load(netPath)
model = model.cuda()
print("*** Loaded Network ***")


def get_pil_transform(): 
	transf = transforms.Compose([
		transforms.Resize((128, 128)),
		transforms.CenterCrop(128)
	])    

	return transf

def get_preprocess_transform():
	# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									# std=[0.229, 0.224, 0.225])     
	transf = transforms.Compose([
		transforms.ToTensor()
	])    

	return transf    

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()


def batch_predict(images):
	model.eval()
	batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	batch = batch.to(device)
	
	# logits = model(batch)
	# probs = FU.softmax(logits, dim=1)
	out,act = model(batch)
	probs = torch.exp(out)
	return probs.detach().cpu().numpy()

test_pred = batch_predict([pill_transf(img)])
test_pred.squeeze().argmax()

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(np.array(pill_transf(img)), 
										 batch_predict, # classification function
										 top_labels=5, 
										 hide_color=0, 
										 num_samples=1000) # number of images that will be sent to classification function

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=True)
img_boundry1 = mark_boundaries(temp/255.0, mask)
# plt.imshow(img_boundry1)
plt.imsave(os.path.join(outPath,imgName),img_boundry1)

# temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
# img_boundry2 = mark_boundaries(temp/255.0, mask)
# plt.imshow(img_boundry2)