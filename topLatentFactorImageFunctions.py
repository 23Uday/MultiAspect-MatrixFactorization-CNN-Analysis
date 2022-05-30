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





def topImagesPerLatentFactor(vF,iF,indexToImagePath,saveTo):
	top = min(200,int(iF.shape[0]/10))
	for LFNUM in range(iF.shape[1]):
		if not os.path.exists(os.path.join(saveTo,'LatentFactor-%d'%LFNUM)):
			os.makedirs(os.path.join(saveTo,'LatentFactor-%d'%LFNUM))
		sumFactor = sum(vF[:,LFNUM])
		weightCollected = sum(vF[:top,LFNUM])
		for imageNum,imageIndex in enumerate(iF[:top,LFNUM]):
			imagePath = indexToImagePath[imageIndex]
			# pdb.set_trace()
			fname = imagePath.split('/')[-3].strip()+'*'+imagePath.split('/')[-2].strip()+'*'+imagePath.split('/')[-1].strip().split('.')[0]
			fname = str(imageNum+1)+'_'+fname+'_'+'Weight : %s'%str(vF[imageNum,LFNUM])+'_'+'Per : %s'%str(100*vF[imageNum,LFNUM]/sumFactor)+'_'+'Top_Weight : %s'+str(weightCollected)
			shutil.copy2(imagePath,os.path.join(saveTo,'LatentFactor-%d'%LFNUM,fname+'.png'))



def topMaskedImagesPerLatentFactor(vF,iF,P,indexToImagePath,saveTo,imgSize = 32):
	top = min(200,int(iF.shape[0]/10))
	numChannels = len(P)
	for LFNUM in range(iF.shape[1]):
		if not os.path.exists(os.path.join(saveTo,'LatentFactor-%d'%LFNUM)):
			os.makedirs(os.path.join(saveTo,'LatentFactor-%d'%LFNUM))
		sumFactor = sum(vF[:,LFNUM])
		weightCollected = sum(vF[:top,LFNUM])
		for imageNum,imageIndex in enumerate(iF[:top,LFNUM]):
			imagePath = indexToImagePath[imageIndex]
			img = Image.open(imagePath).convert('RGB')
			img = img.resize((imgSize,imgSize))
			# imgArr = np.array(img)
			imgc0 = np.array(img.getchannel(0))
			imgc1 = np.array(img.getchannel(1))
			imgc2 = np.array(img.getchannel(2))

			fname = imagePath.split('/')[-3].strip()+'*'+imagePath.split('/')[-2].strip()+'*'+imagePath.split('/')[-1].strip().split('.')[0]
			fname = str(imageNum+1)+'_'+fname+'_'+'Weight : %s'%str(vF[imageNum,LFNUM])+'_'+'Per : %s'%str(100*vF[imageNum,LFNUM]/sumFactor)+'_'+'Top_Weight : %s'+str(weightCollected)
			
			latentImage = []
			maskedImage = []
			for c in range(numChannels):
				imgC = P[c][:,LFNUM].reshape(imgSize,imgSize)
				imgC.__imul__(255/imgC.max())
				latentImage.append(np.uint8(imgC))

				# binaryImgC[]

			if numChannels != 1:
				latentImage = tuple(latentImage)
				M = np.dstack(latentImage)
				latentFactorImg = PIL.Image.fromarray(M)
				latentFactorImg = latentFactorImg.convert('RGB')
				latentFactorImgArr = np.array(latentFactorImg)
				# latentFactorImgArrMean = latentFactorImgArr.mean()
				# latentFactorImgArr[latentFactorImgArr > 10] = 1
				# pdb.set_trace()
				# latentFactorImgArr[latentFactorImgArr <= 10] = 0
				# imgc0[latentFactorImgArr == 0] = 0
				# imgc1[latentFactorImgArr == 0] = 0
				# imgc2[latentFactorImgArr == 0] = 0
				imgc0[latentFactorImgArr[:,:,0] <= np.median(latentFactorImgArr[:,:,0])] = 0
				imgc1[latentFactorImgArr[:,:,1] <= np.median(latentFactorImgArr[:,:,1])] = 0
				imgc2[latentFactorImgArr[:,:,2] <= np.median(latentFactorImgArr[:,:,2])] = 0

			elif numChannels == 1:
				M = latentImage[l-1]
				latentFactorImg = PIL.Image.fromarray(M)
				latentFactorImg = latentFactorImg.convert('L')
				latentFactorImgArr = np.array(latentFactorImg)
				# latentFactorImgArrMean = latentFactorImgArr.mean()
				# latentFactorImgArr[latentFactorImgArr > 10] = 1
				latentFactorImgArr[latentFactorImgArr <= 10] = 0

			# targetImage = np.multiply(imgArr,M)

			targetImagec0 = imgc0
			targetImagec1 = imgc1
			targetImagec2 = imgc2

			# pdb.set_trace()


			# targetImagec0 = np.multiply(imgc0,latentFactorImgArr)
			# targetImagec1 = np.multiply(imgc1,latentFactorImgArr)
			# targetImagec2 = np.multiply(imgc2,latentFactorImgArr)


			imgMasked = Image.fromarray(np.uint8(np.dstack((targetImagec0,targetImagec1,targetImagec2))))

			imgMasked.save(os.path.join(saveTo,'LatentFactor-%d'%LFNUM,fname+'.png'))




def topMaskedImagesPerLatentFactorAggPercent(vF,iF,P,indexToImagePath,saveTo,imgSize = 32, aggMode = True, fracMode = True, percent = 0.2):
	top = min(200,int(iF.shape[0]/10))
	numChannels = len(P)
	for LFNUM in range(iF.shape[1]):
		if not os.path.exists(os.path.join(saveTo,'LatentFactor-%d'%LFNUM)):
			os.makedirs(os.path.join(saveTo,'LatentFactor-%d'%LFNUM))
		sumFactor = sum(vF[:,LFNUM])
		weightCollected = sum(vF[:top,LFNUM])
		# pdb.set_trace()
		for imageNum,imageIndex in enumerate(iF[:top,LFNUM]):
			imagePath = indexToImagePath[imageIndex]
			img = Image.open(imagePath).convert('RGB')
			img = img.resize((imgSize,imgSize))
			# imgArr = np.array(img)
			imgc0 = np.array(img.getchannel(0))
			imgc1 = np.array(img.getchannel(1))
			imgc2 = np.array(img.getchannel(2))

			fname = imagePath.split('/')[-3].strip()+'_'+imagePath.split('/')[-2].strip()+'_'+imagePath.split('/')[-1].strip().split('.')[0]
			fname = str(imageNum+1)+'_'+fname+'_'+'Weight : %s'%str(vF[imageNum,LFNUM])+'_'+'Per : %s'%str(100*vF[imageNum,LFNUM]/sumFactor)+'_'+'Top_Weight : %s'+str(weightCollected)
			
			latentImage = []
			maskedImage = []

			aggMaskedImage = []

			if aggMode:
				# For now percentage mode is used in aggregated masked image
				# This is only for the purposes of DSAA 2022
				imgC = np.zeros((imgSize,imgSize))
				imgCMasked = np.zeros((imgSize,imgSize))
				# imgV = np.zeros(imgSize*imgSize)
				# pdb.set_trace()
				for c in range(numChannels):
					# pdb.set_trace()
					imgC += P[c][:,LFNUM].reshape(imgSize,imgSize)				
				

				imgCMasked = imgC
				imgCMasked[imgCMasked < np.percentile(imgCMasked,int((1-percent)*100))] = 0
				# pdb.set_trace()
				imgC.__imul__(255/imgC.max())
				imgCMasked.__imul__(255/imgCMasked.max())
				latentImage.append(np.uint8(imgC))
				aggMaskedImage.append(np.uint8(imgCMasked))

				# pdb.set_trace()


			else:
				for c in range(numChannels):
					pdb.set_trace()
					imgC = P[c][:,LFNUM].reshape(imgSize,imgSize)
					imgC.__imul__(255/imgC.max())
					latentImage.append(np.uint8(imgC))

					# binaryImgC[]

			if numChannels != 1:

				if aggMode:
					## This code is not well written, made for DSAA (Revisit this function later)
					# pass
					# pdb.set_trace()
					# aggMaskedImage = tuple(aggMaskedImage)
					# M = np.dstack(latentImage)
					latentFactorImg = PIL.Image.fromarray(aggMaskedImage[0])
					latentFactorImg = latentFactorImg.convert('L')
					latentFactorImgArr = np.array(latentFactorImg)
					# pdb.set_trace()

				else:
					# pdb.set_trace()
					latentImage = tuple(latentImage)
					M = np.dstack(latentImage)
					latentFactorImg = PIL.Image.fromarray(M)
					latentFactorImg = latentFactorImg.convert('RGB')
					latentFactorImgArr = np.array(latentFactorImg)
					# latentFactorImgArrMean = latentFactorImgArr.mean()
					# latentFactorImgArr[latentFactorImgArr > 10] = 1
					# pdb.set_trace()
					# latentFactorImgArr[latentFactorImgArr <= 10] = 0

				if fracMode:
					imgc0[latentFactorImgArr == 0] = 0
					imgc1[latentFactorImgArr == 0] = 0
					imgc2[latentFactorImgArr == 0] = 0
					# pdb.set_trace()
				else:
					imgc0[latentFactorImgArr[:,:,0] <= np.median(latentFactorImgArr[:,:,0])] = 0
					imgc1[latentFactorImgArr[:,:,1] <= np.median(latentFactorImgArr[:,:,1])] = 0
					imgc2[latentFactorImgArr[:,:,2] <= np.median(latentFactorImgArr[:,:,2])] = 0

				# pdb.set_trace()

			elif numChannels == 1:
				M = latentImage[l-1]
				latentFactorImg = PIL.Image.fromarray(M)
				latentFactorImg = latentFactorImg.convert('L')
				latentFactorImgArr = np.array(latentFactorImg)
				# latentFactorImgArrMean = latentFactorImgArr.mean()
				# latentFactorImgArr[latentFactorImgArr > 10] = 1
				latentFactorImgArr[latentFactorImgArr <= 10] = 0

			# targetImage = np.multiply(imgArr,M)

			targetImagec0 = imgc0
			targetImagec1 = imgc1
			targetImagec2 = imgc2

			# pdb.set_trace()


			# targetImagec0 = np.multiply(imgc0,latentFactorImgArr)
			# targetImagec1 = np.multiply(imgc1,latentFactorImgArr)
			# targetImagec2 = np.multiply(imgc2,latentFactorImgArr)


			imgMasked = Image.fromarray(np.uint8(np.dstack((targetImagec0,targetImagec1,targetImagec2))))

			imgMasked.save(os.path.join(saveTo,'LatentFactor-%d'%LFNUM,fname+'.png'))






def topMaskedImagesPerLatentFactor(vF,iF,P,indexToImagePath,saveTo,imgSize = 32):
	top = min(200,int(iF.shape[0]/10))
	numChannels = len(P)
	for LFNUM in range(iF.shape[1]):
		if not os.path.exists(os.path.join(saveTo,'LatentFactor-%d'%LFNUM)):
			os.makedirs(os.path.join(saveTo,'LatentFactor-%d'%LFNUM))
		sumFactor = sum(vF[:,LFNUM])
		weightCollected = sum(vF[:top,LFNUM])
		for imageNum,imageIndex in enumerate(iF[:top,LFNUM]):
			imagePath = indexToImagePath[imageIndex]
			img = Image.open(imagePath).convert('RGB')
			img = img.resize((imgSize,imgSize))
			# imgArr = np.array(img)
			imgc0 = np.array(img.getchannel(0))
			imgc1 = np.array(img.getchannel(1))
			imgc2 = np.array(img.getchannel(2))

			fname = imagePath.split('/')[-3].strip()+'_'+imagePath.split('/')[-2].strip()+'_'+imagePath.split('/')[-1].strip().split('.')[0]
			fname = str(imageNum+1)+'_'+fname+'_'+'Weight : %s'%str(vF[imageNum,LFNUM])+'_'+'Per : %s'%str(100*vF[imageNum,LFNUM]/sumFactor)+'_'+'Top_Weight : %s'+str(weightCollected)
			
			latentImage = []
			maskedImage = []
			for c in range(numChannels):
				imgC = P[c][:,LFNUM].reshape(imgSize,imgSize)
				imgC.__imul__(255/imgC.max())
				latentImage.append(np.uint8(imgC))

				# binaryImgC[]

			if numChannels != 1:
				latentImage = tuple(latentImage)
				M = np.dstack(latentImage)
				latentFactorImg = PIL.Image.fromarray(M)
				latentFactorImg = latentFactorImg.convert('RGB')
				latentFactorImgArr = np.array(latentFactorImg)
				# latentFactorImgArrMean = latentFactorImgArr.mean()
				# latentFactorImgArr[latentFactorImgArr > 10] = 1
				# pdb.set_trace()
				# latentFactorImgArr[latentFactorImgArr <= 10] = 0
				# imgc0[latentFactorImgArr == 0] = 0
				# imgc1[latentFactorImgArr == 0] = 0
				# imgc2[latentFactorImgArr == 0] = 0
				imgc0[latentFactorImgArr[:,:,0] <= np.median(latentFactorImgArr[:,:,0])] = 0
				imgc1[latentFactorImgArr[:,:,1] <= np.median(latentFactorImgArr[:,:,1])] = 0
				imgc2[latentFactorImgArr[:,:,2] <= np.median(latentFactorImgArr[:,:,2])] = 0

			elif numChannels == 1:
				M = latentImage[l-1]
				latentFactorImg = PIL.Image.fromarray(M)
				latentFactorImg = latentFactorImg.convert('L')
				latentFactorImgArr = np.array(latentFactorImg)
				# latentFactorImgArrMean = latentFactorImgArr.mean()
				# latentFactorImgArr[latentFactorImgArr > 10] = 1
				latentFactorImgArr[latentFactorImgArr <= 10] = 0

			# targetImage = np.multiply(imgArr,M)

			targetImagec0 = imgc0
			targetImagec1 = imgc1
			targetImagec2 = imgc2

			# pdb.set_trace()


			# targetImagec0 = np.multiply(imgc0,latentFactorImgArr)
			# targetImagec1 = np.multiply(imgc1,latentFactorImgArr)
			# targetImagec2 = np.multiply(imgc2,latentFactorImgArr)


			imgMasked = Image.fromarray(np.uint8(np.dstack((targetImagec0,targetImagec1,targetImagec2))))

			imgMasked.save(os.path.join(saveTo,'LatentFactor-%d'%LFNUM,fname+'.png'))

