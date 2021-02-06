# NUtrainnet is a CNN designed to generate accurate segmentation masks from 1280x1024 images for soccerballs/goals/fieldlines/horizons
# Written solely by Bryce Tuppurainen for the NUbots team, all rights reserved, please email c3286917@uon.edu.au if you have any questions

# IMPORT THE APPROPRIATE LIBRARIES TO PROCESS THE DATA EFFICIENTLY
import pandas
import numpy
from skimage.io import imread
import torch
import matplotlib.pyplot as plt
import torchvision
import tqdm

# IMPORT ALL THE TRAINING IMAGES INTO A GREYSCALE ARRAY TO SIMPLIFY
train = []
for filename in range(1, 16):
	path = 'Dataset/Datasets/testA/' + str(filename)+'.jpg'
	img = imread(path, as_gray=True)
	img /= 255.0
	img = img.astype('float32')
	train.append(img)

def convertImage(imgData):
	livePixels = []
	for row in imgData:
		for data in row:
			for pixel in data:
				livePixels.append(data)
		break # This causes the for loop which is switching through each in train to break on the first image
	return torch.FloatTensor(livePixels)

# IMPORT MODULES FOR NETWORK BUILDING
import torch.nn as nn
import torch.nn.functional as functional

# DEFINING THE LAYERS AND NETWORK (Two hidden layer, )
class Network(nn.Module):
	def __init__(self):
		super().__init__()
		self.inl = nn.Linear(1105920, 128)
		self.hl1 = nn.Linear(128, 128) # HIDDEN LAYER (Fully connected 128 sigmoid nueron nodes)
		self.hl2 = nn.Linear(128, 128) # HIDDEN LAYER (Fully connected 128 sigmoid nueron nodes)
		self.oul = nn.Linear(128, 1105920) # Output layer (the goal is to provide a greyscale segmentation mask of a soccer ball)


	# Defining neuron activation to create a feed forward for data through the network
	def forward(self, datain):
		print("Passing data... ", end="")
		datain = functional.relu(self.inl(datain)) # relu is defining the activation function of these neurons as rectified linear in each hidden layer
		datain = functional.relu(self.hl1(datain))
		datain = functional.relu(self.hl2(datain))
		datain = self.oul(datain)
		return functional.log_softmax(datain, dim=1) # This will be the segmentation mask of the location of the soccer ball

liveNetwork = Network()


# DEBUGGING! PASS ONE IMAGE THROUGH THE NETWORK TO CONFIRM THAT THE FORWARD PASS IS RUNNING CORRECTLY

print("[Debugging message] Live network is running!")
liveNetwork.zero_grad()
output = liveNetwork(convertImage(train))
print(output)