import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional

torch.set_default_dtype(torch.float32)
import torch.optim as optim
import pickle
import math

###############################################################################
# Define the loss functions for MRI imaging
###############################################################################
def Loss_kspace_diff(sigma):
	def func(y_true, y_pred):
		return torch.mean(torch.abs(y_pred - y_true), (1, 2, 3)) / sigma
	return func

def Loss_kspace_diff2(sigma):
	def func(y_true, y_pred):
		return torch.mean((y_pred - y_true)**2, (1, 2, 3)) / (sigma)**2
	return func

def Loss_l1(y_pred):
	# image prior - sparsity loss
	return torch.mean(torch.abs(y_pred), (-1, -2))

def Loss_TSV(y_pred):
	# image prior - total squared variation loss
	return torch.mean((y_pred[:, 1::, :] - y_pred[:, 0:-1, :])**2, (-1, -2)) + torch.mean((y_pred[:, :, 1::] - y_pred[:, :, 0:-1])**2, (-1, -2))

def Loss_TV(y_pred):
	# image prior - total variation loss
	return torch.mean(torch.abs(y_pred[:, 1::, :] - y_pred[:, 0:-1, :]), (-1, -2)) + torch.mean(torch.abs(y_pred[:, :, 1::] - y_pred[:, :, 0:-1]), (-1, -2))

# def Loss_TV(y_pred):
# 	# image prior - total variation loss
# 	eps = 1e-24
# 	return torch.mean(torch.sqrt((y_pred[:, 1::, :]-y_pred[:, 0:-1, :])**2+eps), (-1, -2)) + torch.mean(torch.sqrt((y_pred[:, :, 1::]-y_pred[:, :, 0:-1])**2+eps), (-1, -2))
