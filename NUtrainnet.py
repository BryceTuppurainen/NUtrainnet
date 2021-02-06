# NUtrainnet is a CNN designed to generate accurate segmentation masks from 1280x1024 images for soccerballs/goals/fieldlines/horizons
# Written by Bryce Tuppurainen (some code has been pulled from open-source sites) for the NUbots team, all rights reserved, please email c3286917@uon.edu.au if you have any questions

# IMPORT THE APPROPRIATE LIBRARIES TO PROCESS THE DATA EFFICIENTLY
import pandas
import numpy
from skimage.io import imread
import torch
import matplotlib.pyplot as plt
import torchvision
import tqdm
import torch.nn as nn
import torch.nn.functional as functional
import time

TRAIN_PATH = "./Dataset/Datasets/testA"

TRANSFORM_IMAGE = torchvision.transforms.Compose([
	torchvision.transforms.Grayscale(num_output_channels=1),
	torchvision.transforms.ToTensor()
	])

trainData = torchvision.datasets.ImageFolder(root=TRAIN_PATH, transform=TRANSFORM_IMAGE)
trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size=1, shuffle=True,  num_workers=4)

# DEFINING THE LAYERS AND NETWORK
class Network(nn.Module):
	def __init__(self):
		super().__init__()
		self.inl = nn.Linear(1310720, 128)
		self.hl1 = nn.Linear(128, 128) # HIDDEN LAYER (Fully connected 128 sigmoid nueron nodes)
		self.hl2 = nn.Linear(128, 128) # HIDDEN LAYER (Fully connected 128 sigmoid nueron nodes)
		self.oul = nn.Linear(128, 1105920) # Output layer (the goal is to provide a greyscale segmentation mask of a soccer ball)


	# Defining neuron activation to create a feed forward for data through the network
	def forward(self, datain):
		datain = functional.relu(self.inl(datain)) # relu is defining the activation function of these neurons as rectified linear in each hidden layer
		datain = functional.relu(self.hl1(datain))
		datain = functional.relu(self.hl2(datain))
		datain = self.oul(datain)
		return functional.log_softmax(datain, dim=1) # This will be the segmentation mask of the location of the soccer ball

liveNetwork = Network()


# DEBUGGING! PASS ONE IMAGE THROUGH THE NETWORK TO CONFIRM THAT THE FORWARD PASS IS RUNNING CORRECTLY

print("[Debug] Live network is up!")
START_TIME = time.time()
epochs = int(input("[Input] Number of epochs to run? > "))
debug = input("[Input] Show debug messages? (y/n) > ")
print("[Time] Started at time "+str(time.ctime(START_TIME))+"!")
for epoch in range(epochs):
	i = 0
	
	if epoch == 1:
		EPOCH_TIME = time.time() - START_TIME
		print("[Time] Approximate time remaining: "+str(int(EPOCH_TIME*(epochs-epoch)))+" seconds...")
	if epoch > 1:
		print("[Time] Approximate time remaining: "+str(int(EPOCH_TIME*(epochs-epoch)))+" seconds...")
	
	for data in enumerate(trainDataLoader):
		i+=1
		if debug == "y":
			print("[Debug] Feeding network image number "+str(i)+" of "+str(len(trainDataLoader))+" in epoch "+str(epoch+1)+" of "+str(epochs)+"...")
		liveNetwork.zero_grad()
		output = liveNetwork(data[1][0].view(-1, 1310720))

print("[Info] NUtrainnet has completed!")