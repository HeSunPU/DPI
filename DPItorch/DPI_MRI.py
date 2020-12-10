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
import cv2


from generative_model import glow_model
from generative_model import realnvpfc_model

from MRI_helpers import *


import argparse

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description="Deep Probabilistic Imaging Trainer for MRI")
parser.add_argument("--impath", default='../dataset/fastmri_sample/mri/scan_1.pkl', type=str, help="MRI image scan")
parser.add_argument("--maskpath", default='../dataset/fastmri_sample/mask/mask8.npy', type=str, help="MRI image scan mask")
parser.add_argument("--save_path", default='./save_path_mri', type=str, help="file save path")
parser.add_argument("--npix", default=64, type=int, help="image shape (pixels)")
parser.add_argument("--ratio", default=1/8, type=float, help="MRI compression ratio")
parser.add_argument("--model_form", default='realnvp', type=str, help="form of the deep generative model")
parser.add_argument("--n_flow", default=16, type=int, help="number of flows in RealNVP or Glow")
parser.add_argument("--n_block", default=4, type=int, help="number of blocks in Glow")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--n_epoch", default=3000, type=int, help="number of epochs for training RealNVP")
parser.add_argument("--logdet", default=10.0, type=float, help="logdet weight")
# parser.add_argument("--l1", default=1e4, type=float, help="l1 prior weight")
parser.add_argument("--l1", default=0.0, type=float, help="l1 prior weight")
parser.add_argument("--tsv", default=1e4, type=float, help="tsv prior weight")
parser.add_argument("--flux", default=0.1, type=float, help="flux prior weight")
parser.add_argument("--clip", default=1e-3, type=float, help="gradient clip for neural network training")


def readMRIdata(filepath):
	with open(filepath, 'rb') as f:
		obj = pickle.load(f)
		kspace = obj['kspace']
		target_image = obj['target']
		attributes = obj['attributes']
	return kspace, target_image, attributes

def fft2c(data):
	# data = np.fft.ifftshift(data)
	data = np.fft.fft2(data, norm="ortho")
	# data = np.fft.fftshift(data)
	return np.stack((data.real, data.imag), axis=-1)


def fft2c_torch(img):
	x = img.unsqueeze(-1)
	x = torch.cat([x, torch.zeros_like(x)], -1)
	kspace_pred = torch.fft(x, 2, normalized=True)
	return kspace_pred


class Img_logscale(nn.Module):
	""" Custom Linear layer but mimics a standard linear layer """
	def __init__(self, scale=1):
		super().__init__()
		log_scale = torch.Tensor(np.log(scale)*np.ones(1))
		self.log_scale = nn.Parameter(log_scale)

	def forward(self):
		return self.log_scale


if __name__ == "__main__":
	args = parser.parse_args()
	impath = args.impath
	npix = args.npix
	ratio = args.ratio
	save_path = args.save_path

	save_path = args.save_path
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	sigma = 1/4 * 2e-6

	_, img_true, _ = readMRIdata(impath)
	img_true = cv2.resize(img_true, (npix, npix), interpolation=cv2.INTER_AREA)
	kspace = fft2c(img_true)
	kspace = kspace + np.random.normal(size=kspace.shape) * sigma
	mask = np.load(args.maskpath)
	mask[24:40, 24:40] = 1
	mask = np.fft.fftshift(mask)
	mask = np.stack((mask, mask), axis=-1)

	if args.model_form == 'realnvp':
		n_flow = args.n_flow
		affine = True
		img_generator = realnvpfc_model.RealNVP(npix*npix, n_flow, affine=affine).to(device)
	elif args.model_form == 'glow':
		n_channel = 1
		n_flow = args.n_flow
		n_block = args.n_block
		affine = True
		no_lu = False#True
		z_shapes = glow_model.calc_z_shapes(n_channel, npix, n_flow, n_block)
		img_generator = glow_model.Glow(n_channel, n_flow, n_block, affine=affine, conv_lu=not no_lu).to(device)

	logscale_factor = Img_logscale(scale=args.flux/(0.8*npix*npix)).to(device)


	# define the losses and weights for MRI
	# Loss_kspace_img = Loss_kspace_diff(sigma)
	Loss_kspace_img = Loss_kspace_diff2(sigma)

	kspace_weight = 1.0
	imgl1_weight = args.l1
	imgtsv_weight = args.tsv * npix#args.tsv * npix*npix#100*npix*npix
	logdet_weight = args.logdet / (npix*npix)#1.0 / (npix*npix) #2.0 * 1.0 / len(obs.camp['camp'])

	kspace_true = torch.Tensor(mask * kspace).to(device=device)

	# optimize both scale and image generator
	lr = args.lr
	optimizer = optim.Adam(list(img_generator.parameters())+list(logscale_factor.parameters()), lr = lr)


	n_epoch = args.n_epoch#30000#10000#100000#50000#100#
	loss_list = []
	loss_prior_list = []
	loss_kspace_list = []
	logdet_list = []
	loss_tsv_list = []
	loss_l1_list = []


	n_batch = 32#8
	for k in range(n_epoch):
		if args.model_form == 'realnvp':
			z_sample = torch.randn(n_batch, npix*npix).to(device=device)
		elif args.model_form == 'glow':
			z_sample = []
			for z in z_shapes:
				z_new = torch.randn(n_batch, *z)
				z_sample.append(z_new.to(device))

		# generate image samples
		img_samp, logdet = img_generator.reverse(z_sample)
		img_samp = img_samp.reshape((-1, npix, npix))

		# apply scale factor and sigmoid/softplus layer for positivity constraint
		scale_factor = torch.exp(logscale_factor.forward())
		img = torch.nn.Softplus()(img_samp) * scale_factor
		det_softplus = torch.sum(img_samp - torch.nn.Softplus()(img_samp), (1, 2))
		logdet = logdet + det_softplus

		kspace_pred = fft2c_torch(img)
		loss_data = Loss_kspace_img(kspace_true, kspace_pred * torch.Tensor(mask).to(device)) / np.mean(mask)

		loss_l1 = Loss_l1(img) if imgl1_weight>0 else 0
		# loss_tsv = Loss_TSV(img) if imgtsv_weight>0 else 0
		loss_tsv = Loss_TV(img) if imgtsv_weight>0 else 0

		loss_prior = imgtsv_weight * loss_tsv + imgl1_weight * loss_l1

		loss = torch.mean(loss_data) + torch.mean(loss_prior) - logdet_weight*torch.mean(logdet)

		optimizer.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(list(img_generator.parameters())+ list(logscale_factor.parameters()), args.clip)
		optimizer.step()

		loss_list.append(loss.detach().cpu().numpy())
		loss_kspace_list.append(torch.mean(loss_data).detach().cpu().numpy())
		loss_prior_list.append(torch.mean(loss_prior).detach().cpu().numpy())
		logdet_list.append(-torch.mean(logdet).detach().cpu().numpy() / (npix*npix))
		loss_tsv_list.append(imgtsv_weight * torch.mean(loss_tsv).detach().cpu().numpy() if imgtsv_weight>0 else 0)
		loss_l1_list.append(imgl1_weight * torch.mean(loss_l1).detach().cpu().numpy() if imgl1_weight>0 else 0)


		print(f"epoch: {k:}, loss: {loss_list[-1]:.5f}, loss kspace: {loss_kspace_list[-1]:.5f}, logdet: {logdet_list[-1]:.5f}")
		print(f"loss tsv: {loss_tsv_list[-1]:.5f}, loss l1: {loss_l1_list[-1]:.5f}")


	torch.save(img_generator.state_dict(), save_path+'/generativemodel_'+args.model_form+'ratio{}_res{}flow{}logdet{}_tv'.format(args.ratio, npix, n_flow, args.logdet))
	np.save(save_path+'/generativeimage_'+args.model_form+'ratio{}_res{}flow{}logdet{}_tv.npy'.format(args.ratio, npix, n_flow, args.logdet), img.cpu().detach().numpy().squeeze())

	loss_all = {}
	loss_all['total'] = np.array(loss_list)
	loss_all['kspace'] = np.array(loss_kspace_list)
	loss_all['logdet'] = np.array(logdet_list)
	loss_all['tsv'] = np.array(loss_tsv_list)

	loss_all['l1'] = np.array(loss_l1_list)
	np.save(save_path+'/loss_'+args.model_form+'ratio{}_res{}flow{}logdet{}_tv.npy'.format(args.ratio, npix, n_flow, args.logdet), loss_all)

