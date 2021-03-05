# IMPORT MODULES FOR CONFIG
# Import pyTorch, matplotlib and torchvision (note that the install will be dependant on the CUDA of the GPU in your system usually)
import matplotlib.pyplot as pyplot
import torch
import torchvision
from torchvision import transforms, datasets
import tqdm

def viewImage(inputArray):
	pyplot.imshow(inputArray.view(1024, 1280))
	pyplot.show()

# CONFIGURE AND DEBUG DATASET

TEST_PATH = "./datasets/labels/"
TRAIN_PATH = "./datasets/train/"

TRANSFORM_IMAGE = torchvision.transforms.Compose([
	torchvision.transforms.Grayscale(num_output_channels=1),
	torchvision.transforms.ToTensor()
	])

train = torchvision.datasets.ImageFolder(root=TRAIN_PATH, transform=TRANSFORM_IMAGE)
trainset = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True,  num_workers=4)

test = torchvision.datasets.ImageFolder(root=TEST_PATH, transform=TRANSFORM_IMAGE)
testset = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True,  num_workers=4)

for data in trainset:
	viewImage(data[0][0]) # Use the library matplotlib to show the first image to the user for debugging (note that it was randomised earlier)
	break

for data in testset:
	viewImage(data[0][0]) # Use the library matplotlib to show the first image to the user for debugging (note that it was randomised earlier)
	break

# IMPORT MODULES FOR NETWORK BUILDING
import torch.nn as nn
import torch.nn.functional as functional

# DEFINING THE LAYERS AND NETWORK
class Network(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(1310720, 64)
		self.fc2 = nn.Linear(64, 64) # HIDDEN LAYER
		self.fc3 = nn.Linear(64, 64) # HIDDEN LAYER
		self.fc4 = nn.Linear(64, 1310720) # This is the output layer

	# Defining neuron activation to create a feed forward for data through the network
	def forward(self, datain):
		datain = functional.relu(self.fc1(datain)) # relu is defining the activation function of these neurons as rectified linear in each hidden layer
		datain = functional.relu(self.fc2(datain))
		datain = functional.relu(self.fc3(datain))
		datain = self.fc4(datain)
		return functional.log_softmax(datain, dim=1)

liveNetwork = Network()

# SETTING UP THE OPTIMIZER TO TRAIN THE NETWORK
import torch.optim as optim

optimizer = optim.Adam(liveNetwork.parameters(), lr=0.001) # setting up a learning rate of 0.001

epochs = int(input("Number of epochs to run? > ")) # Number of times passing through the whole dataset
print("Beggining training of the dataset through "+str(epochs)+" epochs!")
for epoch in range(epochs):
	for data in trainset:
		pixels, target = data # Unpacking training data into the greyscale values of the pixels as tensors and scalar target values
		liveNetwork.zero_grad()
		output = liveNetwork(pixels.view(-1, 1310720))
		loss = functional.nll_loss(output, target) # Compare the scalar value provided as the output of the current network to the real value in order to determine loss at this point in time
		loss.backward() # The magic backpropogation method that pytorch has included in the Adam optimizer, documentation is online
		optimizer.step() # Adjust the bias and weights of the network based on the backpropogation from the previous line
	print("\nEpoch: "+str(epoch+1)+" of "+str(epochs)+" Current loss: "+str(loss.item()))
print("\nFinal loss after "+str(epochs)+" epochs was: "+str(loss.item())+"\n")

# EVALUATION OF NETWORK ACCURACY
#valid = 0
#total = 0


#print("Beginning evaluation of network...\n")

with torch.no_grad():
	for data in trainset:
		pixels, target = data
		output = liveNetwork(pixels.view(-1, 1310720))
		viewImage(output)
		break
#		for idx, i in output:
#			if i == target[idx]:
#				valid += 1
#			total += 1


#print("After evaluating against "+str(total)+" images the final accuracy was: "+str(100*round(valid/total, 10))+"%\n")