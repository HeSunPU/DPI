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

from torchkbnufft import KbNufft, AdjKbNufft
from torchkbnufft.mri.dcomp_calc import calculate_radial_dcomp_pytorch
from torchkbnufft.math import absolute

# from loupe import models_vlbi, layers_vlbi # loupe package
import ehtim as eh # eht imaging package

from ehtim.observing.obs_helpers import *
import ehtim.const_def as ehc

from astropy.io import fits
from pynfft.nfft import NFFT



def torch_complex_mul(x, y):
	# complex multiplication in pytorch
	xy_real = x[:, :, 0:1] * y[0:1] - x[:, :, 1::] * y[1::]
	xy_imag = x[:, :, 0:1] * y[1::] + x[:, :, 1::] * y[0:1]
	return torch.cat([xy_real, xy_imag], -2)

def torch_complex_matmul(x, F):
	Fx_real = torch.matmul(x, F[:, :, 0])
	Fx_imag = torch.matmul(x, F[:, :, 1])
	return torch.cat([Fx_real.unsqueeze(1), Fx_imag.unsqueeze(1)], -2)

###############################################################################
# extracte interferometry observation parameters from obs and simim files
###############################################################################
def Obs_params_torch(obs, simim, snrcut=0.0, ttype='nfft'):
	###############################################################################
	# generate the discrete Fourier transform matrices or nfft variables for complex visibilities
	###############################################################################
	obs_data = obs.unpack(['u', 'v', 'vis', 'sigma'])
	uv = np.hstack((obs_data['u'].reshape(-1,1), obs_data['v'].reshape(-1,1)))
	vu = np.hstack((obs_data['v'].reshape(-1,1), obs_data['u'].reshape(-1,1)))

	fft_pad_factor = ehc.FFT_PAD_DEFAULT
	p_rad = ehc.GRIDDER_P_RAD_DEFAULT
	npad = int(fft_pad_factor * np.max((simim.xdim, simim.ydim)))
	nfft_info_vis = NFFTInfo(simim.xdim, simim.ydim, simim.psize, simim.pulse, npad, p_rad, uv)
	pulsefac_vis = nfft_info_vis.pulsefac

	vu_scaled =  np.array(vu * simim.psize * 2 * np.pi)
	ktraj_vis = torch.tensor(vu_scaled.T).unsqueeze(0)
	pulsefac_vis_torch = torch.tensor(np.concatenate([np.expand_dims(pulsefac_vis.real, 0), 
													np.expand_dims(pulsefac_vis.imag, 0)], 0))
	if ttype == 'direct':
		dft_mat = ftmatrix(simim.psize, simim.xdim, simim.ydim, uv, pulse=simim.pulse)
		dft_mat = np.expand_dims(dft_mat.T, -1)
		dft_mat = np.concatenate([dft_mat.real, dft_mat.imag], -1)
		dft_mat = torch.tensor(dft_mat, dtype=torch.float32)
	else:
		dft_mat = None

	###############################################################################
	# generate the discrete Fourier transform matrices for closure phases
	###############################################################################
	# if snrcut > 0:
	# 	obs.add_cphase(count='min', snrcut=snrcut)
	# else:
	# 	obs.add_cphase(count='min')

	# if snrcut > 0:
	# 	obs.add_cphase(count='max', snrcut=snrcut)
	# else:
	# 	obs.add_cphase(count='max')

	if snrcut > 0:
		obs.add_cphase(count='min-cut0bl', uv_min=.1e9, snrcut=snrcut)
	else:
		obs.add_cphase(count='min-cut0bl', uv_min=.1e9)



	tc1 = obs.cphase['t1']
	tc2 = obs.cphase['t2']
	tc3 = obs.cphase['t3']

	cphase_map = np.zeros((len(obs.cphase['time']), 3))

	zero_symbol = 100000
	for k1 in range(cphase_map.shape[0]):
		for k2 in list(np.where(obs.data['time']==obs.cphase['time'][k1])[0]):
			if obs.data['t1'][k2] == obs.cphase['t1'][k1] and obs.data['t2'][k2] == obs.cphase['t2'][k1]:
				cphase_map[k1, 0] = k2
				if k2 == 0:
					cphase_map[k1, 0] = zero_symbol
			elif obs.data['t2'][k2] == obs.cphase['t1'][k1] and obs.data['t1'][k2] == obs.cphase['t2'][k1]:
				cphase_map[k1, 0] = -k2
				if k2 == 0:
					cphase_map[k1, 0] = -zero_symbol
			elif obs.data['t1'][k2] == obs.cphase['t2'][k1] and obs.data['t2'][k2] == obs.cphase['t3'][k1]:
				cphase_map[k1, 1] = k2
				if k2 == 0:
					cphase_map[k1, 1] = zero_symbol
			elif obs.data['t2'][k2] == obs.cphase['t2'][k1] and obs.data['t1'][k2] == obs.cphase['t3'][k1]:
				cphase_map[k1, 1] = -k2
				if k2 == 0:
					cphase_map[k1, 1] = -zero_symbol
			elif obs.data['t1'][k2] == obs.cphase['t3'][k1] and obs.data['t2'][k2] == obs.cphase['t1'][k1]:
				cphase_map[k1, 2] = k2
				if k2 == 0:
					cphase_map[k1, 2] = zero_symbol
			elif obs.data['t2'][k2] == obs.cphase['t3'][k1] and obs.data['t1'][k2] == obs.cphase['t1'][k1]:
				cphase_map[k1, 2] = -k2
				if k2 == 0:
					cphase_map[k1, 2] = -zero_symbol

	cphase_ind1 = np.abs(cphase_map[:, 0]).astype(np.int)
	cphase_ind1[cphase_ind1==zero_symbol] = 0
	cphase_ind2 = np.abs(cphase_map[:, 1]).astype(np.int)
	cphase_ind2[cphase_ind2==zero_symbol] = 0
	cphase_ind3 = np.abs(cphase_map[:, 2]).astype(np.int)
	cphase_ind3[cphase_ind3==zero_symbol] = 0
	cphase_sign1 = np.sign(cphase_map[:, 0])
	cphase_sign2 = np.sign(cphase_map[:, 1])
	cphase_sign3 = np.sign(cphase_map[:, 2])


	cphase_ind_list = [torch.tensor(cphase_ind1), torch.tensor(cphase_ind2), torch.tensor(cphase_ind3)]
	cphase_sign_list = [torch.tensor(cphase_sign1), torch.tensor(cphase_sign2), torch.tensor(cphase_sign3)]

	###############################################################################
	# generate the discrete Fourier transform matrices for closure amp
	###############################################################################
	if snrcut > 0:
		obs.add_camp(debias=True, count='min', snrcut=snrcut)
		obs.add_logcamp(debias=True, count='min', snrcut=snrcut)
	else:
		obs.add_camp(debias=True, count='min')
		obs.add_logcamp(debias=True, count='min')

	# if snrcut > 0:
	# 	obs.add_camp(debias=True, count='max', snrcut=snrcut)
	# 	obs.add_logcamp(debias=True, count='max', snrcut=snrcut)
	# else:
	# 	obs.add_camp(debias=True, count='max')
	# 	obs.add_logcamp(debias=True, count='max')

	# obs.add_camp(count='max')
	tca1 = obs.camp['t1']
	tca2 = obs.camp['t2']
	tca3 = obs.camp['t3']
	tca4 = obs.camp['t4']

	camp_map = np.zeros((len(obs.camp['time']), 6))

	zero_symbol = 10000
	for k1 in range(camp_map.shape[0]):
		for k2 in list(np.where(obs.data['time']==obs.camp['time'][k1])[0]):
			if obs.data['t1'][k2] == obs.camp['t1'][k1] and obs.data['t2'][k2] == obs.camp['t2'][k1]:
				camp_map[k1, 0] = k2
				if k2 == 0:
					camp_map[k1, 0] = zero_symbol
			elif obs.data['t2'][k2] == obs.camp['t1'][k1] and obs.data['t1'][k2] == obs.camp['t2'][k1]:
				camp_map[k1, 0] = -k2
				if k2 == 0:
					camp_map[k1, 0] = -zero_symbol
			elif obs.data['t1'][k2] == obs.camp['t1'][k1] and obs.data['t2'][k2] == obs.camp['t3'][k1]:
				camp_map[k1, 1] = k2
				if k2 == 0:
					camp_map[k1, 1] = zero_symbol
			elif obs.data['t2'][k2] == obs.camp['t1'][k1] and obs.data['t1'][k2] == obs.camp['t3'][k1]:
				camp_map[k1, 1] = -k2
				if k2 == 0:
					camp_map[k1, 1] = -zero_symbol
			elif obs.data['t1'][k2] == obs.camp['t1'][k1] and obs.data['t2'][k2] == obs.camp['t4'][k1]:
				camp_map[k1, 2] = k2
				if k2 == 0:
					camp_map[k1, 2] = zero_symbol
			elif obs.data['t2'][k2] == obs.camp['t1'][k1] and obs.data['t1'][k2] == obs.camp['t4'][k1]:
				camp_map[k1, 2] = -k2
				if k2 == 0:
					camp_map[k1, 2] = -zero_symbol
			elif obs.data['t1'][k2] == obs.camp['t2'][k1] and obs.data['t2'][k2] == obs.camp['t3'][k1]:
				camp_map[k1, 3] = k2
				if k2 == 0:
					camp_map[k1, 3] = zero_symbol
			elif obs.data['t2'][k2] == obs.camp['t2'][k1] and obs.data['t1'][k2] == obs.camp['t3'][k1]:
				camp_map[k1, 3] = -k2
				if k2 == 0:
					camp_map[k1, 3] = -zero_symbol
			elif obs.data['t1'][k2] == obs.camp['t2'][k1] and obs.data['t2'][k2] == obs.camp['t4'][k1]:
				camp_map[k1, 4] = k2
				if k2 == 0:
					camp_map[k1, 4] = zero_symbol
			elif obs.data['t2'][k2] == obs.camp['t2'][k1] and obs.data['t1'][k2] == obs.camp['t4'][k1]:
				camp_map[k1, 4] = -k2
				if k2 == 0:
					camp_map[k1, 4] = -zero_symbol
			elif obs.data['t1'][k2] == obs.camp['t3'][k1] and obs.data['t2'][k2] == obs.camp['t4'][k1]:
				camp_map[k1, 5] = k2
				if k2 == 0:
					camp_map[k1, 5] = zero_symbol
			elif obs.data['t2'][k2] == obs.camp['t3'][k1] and obs.data['t1'][k2] == obs.camp['t4'][k1]:
				camp_map[k1, 5] = -k2
				if k2 == 0:
					camp_map[k1, 5] = -zero_symbol

	camp_ind1 = np.abs(camp_map[:, 0]).astype(np.int)
	camp_ind1[camp_ind1==zero_symbol] = 0
	camp_ind2 = np.abs(camp_map[:, 5]).astype(np.int)
	camp_ind2[camp_ind2==zero_symbol] = 0
	camp_ind3 = np.abs(camp_map[:, 2]).astype(np.int)
	camp_ind3[camp_ind3==zero_symbol] = 0
	camp_ind4 = np.abs(camp_map[:, 3]).astype(np.int)
	camp_ind4[camp_ind4==zero_symbol] = 0
	# camp_sign1 = np.sign(camp_map[:, 0])
	# camp_sign2 = np.sign(camp_map[:, 5])
	# camp_sign3 = np.sign(camp_map[:, 2])
	# camp_sign4 = np.sign(camp_map[:, 3])

	camp_ind_list = [torch.tensor(camp_ind1), torch.tensor(camp_ind2), torch.tensor(camp_ind3), torch.tensor(camp_ind4)]
	# camp_sign_list = [torch.tensor(camp_sign1), torch.tensor(camp_sign2), torch.tensor(camp_sign3), torch.tensor(camp_sign4)]
	return dft_mat, ktraj_vis, pulsefac_vis_torch, cphase_ind_list, cphase_sign_list, camp_ind_list

###############################################################################
# Define the interferometry observation function
###############################################################################
def eht_observation_pytorch(npix, nufft_ob, dft_mat, ktraj_vis, pulsefac_vis_torch, cphase_ind_list, cphase_sign_list, camp_ind_list, device, ttype='nfft'):
	eps = 1e-16
	nufft_ob = nufft_ob.to(device=device)
	ktraj_vis = ktraj_vis.to(device=device)
	pulsefac_vis_torch = pulsefac_vis_torch.to(device=device)

	cphase_ind1 = cphase_ind_list[0].to(device=device)
	cphase_ind2 = cphase_ind_list[1].to(device=device)
	cphase_ind3 = cphase_ind_list[2].to(device=device)

	cphase_sign1 = cphase_sign_list[0].to(device=device)
	cphase_sign2 = cphase_sign_list[1].to(device=device)
	cphase_sign3 = cphase_sign_list[2].to(device=device)

	camp_ind1 = camp_ind_list[0].to(device=device)
	camp_ind2 = camp_ind_list[1].to(device=device)
	camp_ind3 = camp_ind_list[2].to(device=device)
	camp_ind4 = camp_ind_list[3].to(device=device)

	if ttype == 'direct':
		F = dft_mat.to(device=device)
	
	def func(x):
		if ttype == 'direct':
			x = torch.reshape(x, (-1, npix*npix)).type(torch.float32).to(device=device)
			vis_torch = torch_complex_matmul(x, F)
		elif ttype == 'nfft':
			x = torch.reshape(x, (-1, npix, npix)).type(torch.float32).to(device=device).unsqueeze(1)
			x = torch.cat([x, torch.zeros_like(x)], 1)
			x = x.unsqueeze(0)

			kdata = nufft_ob(x, ktraj_vis)
			vis_torch = torch_complex_mul(kdata, pulsefac_vis_torch).squeeze(0)
		vis_amp = torch.sqrt((vis_torch[:, 0, :])**2 + (vis_torch[:, 1, :])**2 + eps)

		vis1_torch = torch.index_select(vis_torch, -1, cphase_ind1)
		vis2_torch = torch.index_select(vis_torch, -1, cphase_ind2)
		vis3_torch = torch.index_select(vis_torch, -1, cphase_ind3)

		ang1 = torch.atan2(vis1_torch[:, 1, :], vis1_torch[:, 0, :])
		ang2 = torch.atan2(vis2_torch[:, 1, :], vis2_torch[:, 0, :])
		ang3 = torch.atan2(vis3_torch[:, 1, :], vis3_torch[:, 0, :])
		cphase = (cphase_sign1*ang1 + cphase_sign2*ang2 + cphase_sign3*ang3) * 180 / np.pi


		vis12_torch = torch.index_select(vis_torch, -1, camp_ind1)
		vis12_amp = torch.sqrt((vis12_torch[:, 0, :])**2 + (vis12_torch[:, 1, :])**2 + eps)
		vis34_torch = torch.index_select(vis_torch, -1, camp_ind2)
		vis34_amp = torch.sqrt((vis34_torch[:, 0, :])**2 + (vis34_torch[:, 1, :])**2 + eps)
		vis14_torch = torch.index_select(vis_torch, -1, camp_ind3)
		vis14_amp = torch.sqrt((vis14_torch[:, 0, :])**2 + (vis14_torch[:, 1, :])**2 + eps)
		vis23_torch = torch.index_select(vis_torch, -1, camp_ind4)
		vis23_amp = torch.sqrt((vis23_torch[:, 0, :])**2 + (vis23_torch[:, 1, :])**2 + eps)

		logcamp = torch.log(vis12_amp) + torch.log(vis34_amp) - torch.log(vis14_amp) - torch.log(vis23_amp)

		return vis_torch, vis_amp, cphase, logcamp
	return func


###############################################################################
# Define the loss functions for interferometry imaging
###############################################################################

def Loss_angle_diff(sigma, device):
	# closure phase difference loss
	sigma = torch.Tensor(sigma).type(torch.float32).to(device=device)
	def func(y_true, y_pred):
		angle_true = y_true * np.pi / 180
		angle_pred = y_pred * np.pi / 180
		# return K.mean(1 - K.cos(angle_true - angle_pred))
		return 2.0*torch.mean((1 - torch.cos(angle_true - angle_pred))/(sigma*np.pi/180)**2, 1)
	return func


def Loss_logca_diff(sigma, device):
	# log closure amp difference loss (computed based on closure amp and its std)
	sigma = torch.Tensor(sigma).type(torch.float32).to(device=device)
	def func(y_true, y_pred):
		return torch.mean((y_true/sigma)**2 * (torch.log(y_true)-torch.log(y_pred))**2, 1)
	return func


def Loss_logca_diff2(sigma, device):
	# log closure amp difference loss (computed based on log closure amp and its std)
	sigma = torch.Tensor(sigma).type(torch.float32).to(device=device)
	def func(y_true, y_pred):
		return torch.mean((y_true - y_pred)**2/(sigma)**2, 1)
	return func


def Loss_vis_diff(sigma, device):
	# visibility difference loss
	sigma = torch.Tensor(sigma).type(torch.float32).to(device=device)
	def func(y_true, y_pred):
		return torch.mean(((y_true[0]-y_pred[:, 0])**2+(y_true[1]-y_pred[:, 1])**2)/(sigma)**2, 1)
	return func


def Loss_logamp_diff(sigma, device):
	# log amp difference loss
	sigma = torch.Tensor(sigma).type(torch.float32).to(device=device)
	def func(y_true, y_pred):
		return torch.mean((y_true)**2/(sigma)**2 * (torch.log(y_true)-torch.log(y_pred))**2, 1)
	return func


def Loss_visamp_diff(sigma, device):
	# log amp difference loss
	sigma = torch.Tensor(sigma).type(torch.float32).to(device=device)
	def func(y_true, y_pred):
		return torch.mean((y_true-y_pred)**2/(sigma)**2, 1)
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


def Loss_flux(flux):
	# image prior - flux loss
	def func(y_pred):
		return (torch.sum(y_pred, (-1, -2)) - flux)**2
	return func

def Loss_center(device, center=15.5, dim=32):
	# image prior - centering loss
	X = np.concatenate([np.arange(dim).reshape((1, dim))] * dim, 0)
	Y = np.concatenate([np.arange(dim).reshape((dim, 1))] * dim, 1)
	X = torch.Tensor(X).type(torch.float32).to(device = device)
	Y = torch.Tensor(Y).type(torch.float32).to(device = device)
	def func(y_pred):
		y_pred_flux = torch.mean(y_pred, (-1, -2))
		xc_pred_norm = torch.mean(y_pred * X, (-1, -2)) / y_pred_flux
		yc_pred_norm = torch.mean(y_pred * Y, (-1, -2)) / y_pred_flux

		loss = 0.5 * ((xc_pred_norm-center)**2 + (yc_pred_norm-center)**2)
		return loss
	return func

def Loss_cross_entropy(y_true, y_pred):
	# image prior - cross entropy loss (measuring difference between recon and reference images)
	loss = torch.mean(y_pred * (torch.log(y_pred + 1e-12) - torch.log(y_true + 1e-12)), (-1, -2))
	return loss