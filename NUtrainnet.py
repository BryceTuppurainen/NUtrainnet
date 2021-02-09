# NUtrainnet is a CNN designed to generate accurate segmentation masks from 1280x1024 images for soccerballs/goals/fieldlines/horizons
# Written by Bryce Tuppurainen for the NUbots team, all rights reserved, please email c3286917@uon.edu.au if you have any questions

# IMPORT THE APPROPRIATE LIBRARIES TO PROCESS THE DATA EFFICIENTLY
import numpy as np
from skimage.io import imread
import torch
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as functional
import time

TRAIN_PATH = "./dataset/train/"
