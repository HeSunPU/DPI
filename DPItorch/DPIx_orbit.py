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

from generative_model import realnvpfc_model
from orbit_helpers import *

import astropy.units as u
import astropy.constants as consts
import warnings

import corner
import argparse

plt.ion()


import time
import pandas as pd


class Params2orbits(nn.Module):
	def __init__(self, sma_range=[10.0, 1000.0], ecc_range=[0.0, 1.0],
				inc_range=[0.0, np.pi], aop_range=[0.0, 2*np.pi],
				pan_range=[0.0, 2*np.pi], tau_range=[0.0, 1.0],
				plx_range=[56.95-3*0.26, 56.95+3*0.26], mtot_range=[1.22-3*0.08, 1.22+3*0.08]):
		super().__init__()
		self.sma_range = sma_range
		self.ecc_range = ecc_range
		self.inc_range = inc_range
		self.aop_range = aop_range
		self.pan_range = pan_range
		self.tau_range = tau_range
		self.plx_range = plx_range
		self.mtot_range = mtot_range



	def forward(self, params):
		log_sma = np.log(self.sma_range[0]) + params[:, 0] * (np.log(self.sma_range[1])-np.log(self.sma_range[0]))
		sma = torch.exp(log_sma)
		# sma = self.sma_range[0] + params[:, 0] * (self.sma_range[1]-self.sma_range[0])
		# log_ecc = np.log(np.max([self.ecc_range[0], 1e-8])) + params[:, 1] * (np.log(self.ecc_range[1])-np.log(np.max([self.ecc_range[0], 1e-8])))
		# ecc = torch.exp(log_ecc)
		ecc = self.ecc_range[0] + params[:, 1] * (self.ecc_range[1]-self.ecc_range[0])
		inc = torch.acos(np.cos(self.inc_range[1]) + params[:, 2] * (np.cos(self.inc_range[0])-np.cos(self.inc_range[1])))
		aop = self.aop_range[0] + params[:, 3] * (self.aop_range[1]-self.aop_range[0])
		pan = self.pan_range[0] + params[:, 4] * (self.pan_range[1]-self.pan_range[0])
		tau = self.tau_range[0] + params[:, 5] * (self.tau_range[1]-self.tau_range[0])
		plx = self.plx_range[0] + params[:, 6] * (self.plx_range[1]-self.plx_range[0])
		mtot = self.mtot_range[0] + params[:, 7] * (self.mtot_range[1]-self.mtot_range[0])

		return sma, ecc, inc, aop%(2*np.pi), pan%(2*np.pi), tau%1, plx, mtot
		# return sma, ecc, inc, aop, pan, tau, plx, mtot
	def reverse(self, sma, ecc, inc, aop, pan, tau, plx, mtot):
		log_sma = torch.log(sma)
		params0 = (log_sma - np.log(self.sma_range[0])) / (np.log(self.sma_range[1])-np.log(self.sma_range[0]))
		# params0 = (sma - self.sma_range[0]) / (self.sma_range[1]-self.sma_range[0])
		# log_ecc = torch.log(ecc)
		# params1 = (log_ecc - np.log(np.max([self.ecc_range[0], 1e-8]))) / (np.log(self.ecc_range[1])-np.log(np.max([self.ecc_range[0], 1e-8])))
		params1 = (ecc - self.ecc_range[0]) / (self.ecc_range[1]-self.ecc_range[0])
		params2 = (torch.cos(inc) - np.cos(self.inc_range[1]))/(np.cos(self.inc_range[0])-np.cos(self.inc_range[1]))
		params3 = (aop - self.aop_range[0]) / (self.aop_range[1]-self.aop_range[0])
		params4 = (pan - self.pan_range[0]) / (self.pan_range[1]-self.pan_range[0])
		params5 = (tau - self.tau_range[0]) / (self.tau_range[1]-self.tau_range[0])
		params6 = (plx - self.plx_range[0]) / (self.plx_range[1]-self.plx_range[0])
		params7 = (mtot - self.mtot_range[0]) / (self.mtot_range[1]-self.mtot_range[0])

		return torch.cat([params0.unsqueeze(-1), params1.unsqueeze(-1), params2.unsqueeze(-1), params3.unsqueeze(-1),
						params4.unsqueeze(-1), params5.unsqueeze(-1), params6.unsqueeze(-1), params7.unsqueeze(-1)], -1)


parser = argparse.ArgumentParser(description="Deep Probabilistic Imaging Trainer for orbit fitting")

parser.add_argument("--divergence_type", default='alpha', type=str, help="KL or alpha, type of objective divergence used for variational inference")
parser.add_argument("--alpha_divergence", default=1.0, type=float, help="hyperparameters for alpha divergence")
parser.add_argument("--save_path", default='./checkpoint/orbit_beta_pic_b/cartesian/alpha1', type=str, help="path to save normalizing flow models")
# parser.add_argument("--alpha_divergence", default=0.5, type=float, help="hyperparameters for alpha divergence")
# parser.add_argument("--save_path", default='./checkpoint/orbit_beta_pic_b/all/randomtest', type=str, help="path to save normalizing flow models")

parser.add_argument("--coordinate_type", default='cartesian', type=str, help="coordinate type")
parser.add_argument("--target", default='betapic', type=str, help="target exoplanet")

parser.add_argument("--data_weight", default=1.0, type=float, help="final data weight for training, between 0-1")
parser.add_argument("--start_order", default=4, type=float, help="start order")
parser.add_argument("--decay_rate", default=3000, type=float, help="decay rate")
parser.add_argument("--n_epoch", default=24000, type=int, help="number of epochs for training RealNVP")

parser.add_argument("--n_flow", default=16, type=int, help="number of affine coupling layers in RealNVP")


if torch.cuda.is_available():
	device = torch.device('cuda:{}'.format(0))

if __name__ == "__main__":
	args = parser.parse_args()

	save_path = args.save_path#'./checkpoint/GJ504'
	# save_path = './checkpoint/orbit_beta_pic_b/all_simulated_annealing_alphadiv'#'./checkpoint/orbit_beta_pic_b/all_simulated_annealing'#'./checkpoint/orbit_beta_pic_b/all_simulated_annealing'#'./checkpoint/orbit_beta_pic_b_cartesian'#'./checkpoint/orbit_GJ504'#
	if not os.path.exists(save_path):
		os.makedirs(save_path)



	n_flow = args.n_flow#16#4#8#16#32#16#4#32#
	affine = True
	nparams = 8 

	base_distribution = 'gaussian'#'gmm'#'gmm_only'#


	params_generator = realnvpfc_model.RealNVP(nparams, n_flow, affine=affine, seqfrac=1/16, batch_norm=True).to(device)
	# params_generator = realnvpfc_model.RealNVP(nparams, n_flow, affine=affine, seqfrac=1/64, batch_norm=True).to(device)
	# params_generator = realnvpfc_model.RealNVP(nparams, n_flow, affine=affine, seqfrac=1/128, batch_norm=False).to(device)
	# params_generator = realnvpfc_model.RealNVP(nparams, n_flow, affine=affine, seqfrac=1/64, batch_norm=False).to(device)
	# params_generator = realnvpfc_model.RealNVP(nparams, n_flow, affine=affine, seqfrac=1/32, batch_norm=False).to(device)


	target = args.target#'GJ504'#'betapic'#
	if target == 'betapic':
		astrometry_data = pd.read_csv('../dataset/orbital_fit/betapic_astrometry.csv')
		# astrometry_data['raoff'] = astrometry_data['sep']  * np.sin(astrometry_data['pa'] * np.pi / 180)
		# astrometry_data['decoff'] = astrometry_data['sep']  * np.cos(astrometry_data['pa'] * np.pi / 180)
		# astrometry_data['raoff_err'] = astrometry_data['sep_err']
		# astrometry_data['decoff_err'] = astrometry_data['sep_err']
		# astrometry_data['sep'] = astrometry_data['sep'] - 0.5*astrometry_data['sep_err']**2/astrometry_data['sep']
		# astrometry_data = astrometry_data[0:17]

		cartesian_indices = np.where(np.logical_not(np.isnan(np.array(astrometry_data['raoff']))))[0]
		polar_indices = np.where(np.logical_not(np.isnan(np.array(astrometry_data['pa']))))[0]
		# polar_indices = np.arange(18)



		polar_exclude_cartesian_indices = np.where(np.logical_and(np.isnan(np.array(astrometry_data['raoff'])), np.logical_not(np.isnan(np.array(astrometry_data['pa'])))))[0]
		all_indices = np.concatenate([cartesian_indices, polar_exclude_cartesian_indices])



		epochs = torch.tensor(np.array(astrometry_data['epoch']), dtype=torch.float32).to(device)
		sep = torch.tensor(np.array(astrometry_data['sep'][polar_indices]), dtype=torch.float32).to(device)
		sep_err = torch.tensor(np.array(astrometry_data['sep_err'][polar_indices]), dtype=torch.float32).to(device)
		pa_values = np.array(astrometry_data['pa'][polar_indices])
		pa_values[pa_values>180] = pa_values[pa_values>180] - 360
		pa = np.pi / 180 * torch.tensor(pa_values, dtype=torch.float32).to(device)
		pa_err = np.pi / 180 * torch.tensor(np.array(astrometry_data['pa_err'][polar_indices]), dtype=torch.float32).to(device)

		sep_err = sep_err * 3
		pa_err = pa_err * 3

		raoff = torch.tensor(np.array(astrometry_data['raoff'][cartesian_indices]), dtype=torch.float32).to(device)
		raoff_err = torch.tensor(np.array(astrometry_data['raoff_err'][cartesian_indices]), dtype=torch.float32).to(device)
		decoff = torch.tensor(np.array(astrometry_data['decoff'][cartesian_indices]), dtype=torch.float32).to(device)
		decoff_err = torch.tensor(np.array(astrometry_data['decoff_err'][cartesian_indices]), dtype=torch.float32).to(device)

		raoff_convert = sep * torch.sin(pa)
		decoff_convert = sep * torch.cos(pa)

		eps = 1e-3
		orbit_converter = Params2orbits(sma_range=[4, 40], ecc_range=[1e-5, 0.99],
										inc_range=[81/180*np.pi, 99/180*np.pi], aop_range=[0.0-eps, 2.0*np.pi+eps],
										pan_range=[25/180*np.pi, 85/180*np.pi], tau_range=[0.0-eps, 1.0+eps],
										plx_range=[51.44-5*0.12, 51.44+5*0.12], mtot_range=[1.75-5*0.05, 1.75+5*0.05]).to(device)

		
		# orbit_converter = Params2orbits(sma_range=[4, 40], ecc_range=[1e-8, 0.99],
		# 								inc_range=[81/180*np.pi, 99/180*np.pi], aop_range=[-2.0*np.pi, 2.0*np.pi],
		# 								pan_range=[25/180*np.pi, 85/180*np.pi], tau_range=[-1.0, 1.0],
		# 								plx_range=[51.44-5*0.12, 51.44+5*0.12], mtot_range=[1.75-5*0.05, 1.75+5*0.05])




		coordinate_type = args.coordinate_type#'all'#'cartesian'#'polar'#
		if coordinate_type == 'cartesian':
			epochs = epochs[cartesian_indices]
		elif coordinate_type == 'polar':
			epochs = epochs[polar_indices]
		elif coordinate_type == 'all' or coordinate_type == 'all_cartesian':
			epochs = epochs[all_indices]


		if coordinate_type == 'cartesian':
			scale_factor = 1.0 / len(cartesian_indices)
		elif coordinate_type == 'polar':
			scale_factor = 1.0 / len(polar_indices) 
		elif coordinate_type == 'all' or coordinate_type == 'all_cartesian':
			scale_factor = 1.0 / len(all_indices)

		


	elif target == 'GJ504':

		epochs = torch.tensor([55645.95, 55702.89, 55785.015, 55787.935, 55985.19400184, 56029.11400323, 56072.30200459], dtype=torch.float32).to(device)
		sep = torch.tensor([2479, 2483, 2481, 2448, 2483, 2487, 2499], dtype=torch.float32).to(device)
		sep_err = torch.tensor([16, 8, 33, 24, 15, 8, 26], dtype=torch.float32).to(device)
		pa_values = np.array([327.94, 327.45, 326.84, 325.82, 326.46, 326.54, 326.14])
		pa_values[pa_values>180] = pa_values[pa_values>180] - 360
		pa = np.pi / 180 * torch.tensor(pa_values, dtype=torch.float32).to(device)
		pa_err = np.pi / 180 * torch.tensor([0.39, 0.19, 0.94, 0.66, 0.36, 0.18, 0.61], dtype=torch.float32).to(device)


		coordinate_type = 'polar'

		sep_weight = 1.0
		pa_weight = 1.0
		logdet_weight = 2.0 / len(epochs)
		prior_weight = 1.0  / len(epochs)

		eps = 1e-3
		orbit_converter = Params2orbits(sma_range=[1e1, 1e4], ecc_range=[1e-8, 0.99],
										inc_range=[0.0+eps, np.pi-eps], aop_range=[0.0-eps, 2.0*np.pi+eps],
										pan_range=[0.0-eps, 2.0*np.pi+eps], tau_range=[0.0-eps, 1.0+eps],
										plx_range=[56.95-5*0.26, 56.95+5*0.26], mtot_range=[1.22-5*0.08, 1.22+5*0.08]).to(device)

		# orbit_converter = Params2orbits(sma_range=[1e1, 1e4], ecc_range=[1e-8, 0.99],
		# 								inc_range=[0.0+3e-4, np.pi-3e-4], aop_range=[-2.0*np.pi, 2.0*np.pi],
		# 								pan_range=[-2.0*np.pi, 2.0*np.pi], tau_range=[-1.0, 1.0],
		# 								plx_range=[56.95-5*0.26, 56.95+5*0.26], mtot_range=[1.22-5*0.08, 1.22+5*0.08])

		scale_factor = 1.0 / len(epochs)


	n_batch = 8192#256#2048#512#4096#256#128#64#32#8
	n_smooth = 10
	loss_best = 1e8


	loss_list = []
	loss_prior_list = []
	loss_sep_list = []
	loss_pa_list = []
	loss_raoff_list = []
	loss_decoff_list = []
	loss_raoff_convert_list = []
	loss_decoff_convert_list = []
	logdet_list = []



	# optimize both scale and image generator
	lr = 2e-4#3e-4#1e-4#1e-4#3e-4#1e-4#3e-4#1e-3#3e-4#1e-3#3e-3#1e-2#1e-4#1e-5#2e-4#1e-4#1e-6#1e-5#1e-3#args.lr#1e-5#
	clip = 1e-4#1e-4#1e-5#1e-4#1e-4#1e-3#1e-4#1e-3#1e-5#1e-4#3e-4#1e-3#1e-4#1e-5#2e-4#1e-4#1e-6#1e-5#3e-5#1e-3#1#1e2#1e-1#


	optimizer = optim.Adam(params_generator.parameters(), lr = lr, amsgrad=True)
	# optimizer = optim.Adam(params_generator.parameters(), lr = lr)

	start_order = args.start_order#5#6#5#4#
	n_epoch = args.n_epoch#30000#30000#21000#100000#30000#3000#
	decay_rate = args.decay_rate#3000#10000#5000#3000
	start_time = time.time()
	alpha_divergence = args.alpha_divergence#1.0#0.99#0.6#0.8#0.95#0.999#0.9#

	divergence_type = args.divergence_type#'alpha'#'KL'#

	final_data_weight = args.data_weight

	for k in range(n_epoch):
		data_weight = min(10**(-start_order+k/decay_rate), final_data_weight)

		z_sample = torch.randn((n_batch, nparams)).to(device=device)


		# generate image samples
		params_samp, logdet = params_generator.reverse(z_sample)
		params = torch.sigmoid(params_samp)
		# compute log determinant
		det_sigmoid = torch.sum(-params_samp-2*torch.nn.Softplus()(-params_samp), -1)
		logdet = logdet + det_sigmoid


		# params_samp = params_generator.forward()
		# params = torch.sigmoid(params_samp)

		sma, ecc, inc, aop, pan, tau, plx, mtot = orbit_converter.forward(params)

		if target == 'betapic':
			raoff_torch, deoff_torch, vz_torch = calc_orbit_torch(epochs, sma, ecc, inc, aop, pan, tau, plx, mtot, max_iter=10, tolerance=1e-8, tau_ref_epoch=50000)
		else:
			raoff_torch, deoff_torch, vz_torch = calc_orbit_torch(epochs, sma, ecc, inc, aop, pan, tau, plx, mtot, max_iter=10, tolerance=1e-8)

		sep_torch = torch.transpose(torch.sqrt(raoff_torch**2 + deoff_torch**2), 0, 1)
		pa_torch = torch.transpose(torch.atan2(raoff_torch, deoff_torch), 0, 1)

		raoff_torch = torch.transpose(raoff_torch, 0, 1)
		deoff_torch = torch.transpose(deoff_torch, 0, 1)

		if coordinate_type == 'polar':
			# loss_sep = (sep_torch - sep)**2 / sep_err**2
			# loss_pa = (torch.atan2(torch.sin(pa-pa_torch), torch.cos(pa-pa_torch)))**2 / pa_err**2
			# loss_sep = (torch.log(sep_torch) - torch.log(sep))**2 / (sep_err/sep)**2
			loss_sep = (torch.log(sep_torch) - torch.log(sep) + 0.5*sep_err**2/sep**2)**2 / (sep_err/sep)**2
			loss_pa = 2.0 * (1 - torch.cos(pa-pa_torch)) / pa_err**2
		elif coordinate_type == 'cartesian':
			loss_raoff = (raoff_torch - raoff)**2 / raoff_err**2
			loss_decoff = (deoff_torch - decoff)**2 / decoff_err**2
		elif coordinate_type == 'all':
			# loss_sep = (sep_torch[:, polar_exclude_cartesian_indices] - sep[polar_exclude_cartesian_indices])**2 / sep_err[polar_exclude_cartesian_indices]**2
			loss_sep = (torch.log(sep_torch[:, polar_exclude_cartesian_indices]) - torch.log(sep[polar_exclude_cartesian_indices]))**2 / (sep_err[polar_exclude_cartesian_indices]/sep[polar_exclude_cartesian_indices])**2
			# loss_pa  = (torch.atan2(torch.sin(pa_torch[:, polar_exclude_cartesian_indices]-pa[polar_exclude_cartesian_indices]), torch.cos(pa_torch[:, polar_exclude_cartesian_indices]-pa[polar_exclude_cartesian_indices])))**2 / pa_err[polar_exclude_cartesian_indices]**2
			loss_pa = 2.0 * (1 - torch.cos(pa_torch[:, polar_exclude_cartesian_indices] - pa[polar_exclude_cartesian_indices])) / pa_err[polar_exclude_cartesian_indices]**2

			loss_raoff = (raoff_torch[:, cartesian_indices] - raoff[cartesian_indices])**2 / raoff_err[cartesian_indices]**2
			loss_decoff = (deoff_torch[:, cartesian_indices] - decoff[cartesian_indices])**2 / decoff_err[cartesian_indices]**2
		elif coordinate_type ==  'all_cartesian':
			loss_raoff_convert = (raoff_torch[:, polar_exclude_cartesian_indices] - raoff_convert[polar_exclude_cartesian_indices])**2 / sep_err[polar_exclude_cartesian_indices]**2
			loss_decoff_convert = (deoff_torch[:, polar_exclude_cartesian_indices] - decoff_convert[polar_exclude_cartesian_indices])**2 / sep_err[polar_exclude_cartesian_indices]**2

			loss_raoff = (raoff_torch[:, cartesian_indices] - raoff[cartesian_indices])**2 / raoff_err[cartesian_indices]**2
			loss_decoff = (deoff_torch[:, cartesian_indices] - decoff[cartesian_indices])**2 / decoff_err[cartesian_indices]**2

		if target == 'betapic':
			loss_prior = (plx - 51.44)**2 / 0.12**2 + (mtot - 1.75)**2 / 0.05**2
		elif target == 'GJ504':
			loss_prior = (plx - 56.95)**2 / 0.26**2 + (mtot - 1.22)**2 / 0.08**2


		logprob = -logdet - 0.5*torch.sum(z_sample**2, 1)

		# Define the divergence loss - annealed loss
		if coordinate_type == 'polar':
			loss = data_weight * (0.5* torch.sum(loss_sep, -1) + 0.5 * torch.sum(loss_pa, -1) + 0.5 * loss_prior) + logprob	
		elif coordinate_type == 'cartesian':
			loss = data_weight * (0.5* torch.sum(loss_raoff, -1) + 0.5 * torch.sum(loss_decoff, -1) + 0.5 * loss_prior) + logprob
		elif coordinate_type == 'all':
			loss = data_weight * (0.5* torch.sum(loss_sep, -1) + 0.5 * torch.sum(loss_pa, -1) + \
								0.5* torch.sum(loss_raoff, -1) + 0.5 * torch.sum(loss_decoff, -1) + 0.5 * loss_prior) + logprob		
		elif coordinate_type == 'all_cartesian':
			loss = data_weight * (0.5* torch.sum(loss_raoff_convert, -1) + 0.5 * torch.sum(loss_decoff_convert, -1) + \
								0.5* torch.sum(loss_raoff, -1) + 0.5 * torch.sum(loss_decoff, -1) + 0.5 * loss_prior) + logprob
		

		if divergence_type == 'KL' or alpha_divergence == 1:
			loss = torch.mean(scale_factor * loss)
		elif divergence_type == 'alpha':
			rej_weights = nn.Softmax(dim=0)(-(1-alpha_divergence)*loss).detach()
			loss = torch.sum(rej_weights * scale_factor * loss)

		
		# Define the divergence loss - original loss
		if coordinate_type == 'polar':
			loss_orig = (0.5* torch.sum(loss_sep, -1) + 0.5 * torch.sum(loss_pa, -1) + 0.5 * loss_prior) + logprob
		elif coordinate_type == 'cartesian':
			loss_orig = (0.5* torch.sum(loss_raoff, -1) + 0.5 * torch.sum(loss_decoff, -1) + 0.5 * loss_prior) + logprob
		elif coordinate_type == 'all':
			loss_orig = (0.5* torch.sum(loss_sep, -1) + 0.5 * torch.sum(loss_pa, -1) + \
								0.5* torch.sum(loss_raoff, -1) + 0.5 * torch.sum(loss_decoff, -1) + 0.5 * loss_prior) + logprob
		elif coordinate_type == 'all_cartesian':
			loss_orig = (0.5* torch.sum(loss_raoff_convert, -1) + 0.5 * torch.sum(loss_decoff_convert, -1) + \
								0.5* torch.sum(loss_raoff, -1) + 0.5 * torch.sum(loss_decoff, -1) + 0.5 * loss_prior) + logprob

		# Define the divergence loss - original loss
		if divergence_type == 'KL' or alpha_divergence == 1:
			loss_orig = torch.mean(scale_factor * loss_orig)
		elif divergence_type == 'alpha':
			loss_orig = scale_factor * torch.log(torch.mean(torch.exp(-(1-alpha_divergence)*loss_orig)))/(alpha_divergence-1)


		optimizer.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(params_generator.parameters(), clip)
		optimizer.step()

		loss_list.append(loss_orig.detach().cpu().numpy())
		loss_prior_list.append(torch.mean(loss_prior).detach().cpu().numpy())
		if coordinate_type == 'polar':
			loss_sep_list.append(torch.mean(loss_sep).detach().cpu().numpy())
			loss_pa_list.append(torch.mean(loss_pa).detach().cpu().numpy())
		elif coordinate_type == 'cartesian':
			loss_raoff_list.append(torch.mean(loss_raoff).detach().cpu().numpy())
			loss_decoff_list.append(torch.mean(loss_decoff).detach().cpu().numpy())
		elif coordinate_type == 'all':
			loss_sep_list.append(torch.mean(loss_sep).detach().cpu().numpy())
			loss_pa_list.append(torch.mean(loss_pa).detach().cpu().numpy())
			loss_raoff_list.append(torch.mean(loss_raoff).detach().cpu().numpy())
			loss_decoff_list.append(torch.mean(loss_decoff).detach().cpu().numpy())
		elif coordinate_type == 'all_cartesian':
			loss_raoff_convert_list.append(torch.mean(loss_raoff_convert).detach().cpu().numpy())
			loss_decoff_convert_list.append(torch.mean(loss_decoff_convert).detach().cpu().numpy())
			loss_raoff_list.append(torch.mean(loss_raoff).detach().cpu().numpy())
			loss_decoff_list.append(torch.mean(loss_decoff).detach().cpu().numpy())
		logdet_list.append(-torch.mean(logdet).detach().cpu().numpy()/nparams)


		if coordinate_type == 'polar':
			print(f"epoch: {(k):}, loss: {loss_list[-1]:.5f}, loss sep: {loss_sep_list[-1]:.5f}, loss pa: {loss_pa_list[-1]:.5f}, loss prior: {loss_prior_list[-1]:.5f}, logdet: {logdet_list[-1]:.5f}")
		elif coordinate_type == 'cartesian':
			print(f"epoch: {(k):}, loss: {loss_list[-1]:.5f}, loss raoff: {loss_raoff_list[-1]:.5f}, loss decoff: {loss_decoff_list[-1]:.5f}, loss prior: {loss_prior_list[-1]:.5f}, logdet: {logdet_list[-1]:.5f}")
		elif coordinate_type == 'all':
			print(f"epoch: {(k):}, loss: {loss_list[-1]:.5f}, loss sep: {loss_sep_list[-1]:.5f}, loss pa: {loss_pa_list[-1]:.5f}, loss raoff: {loss_raoff_list[-1]:.5f}, loss decoff: {loss_decoff_list[-1]:.5f}, loss prior: {loss_prior_list[-1]:.5f}, logdet: {logdet_list[-1]:.5f}")
		elif coordinate_type == 'all_cartesian':
			print(f"epoch: {(k):}, loss: {loss_list[-1]:.5f}, loss raoff convert: {loss_raoff_convert_list[-1]:.5f}, loss decoff convert: {loss_decoff_convert_list[-1]:.5f}, loss raoff: {loss_raoff_list[-1]:.5f}, loss decoff: {loss_decoff_list[-1]:.5f}, loss prior: {loss_prior_list[-1]:.5f}, logdet: {logdet_list[-1]:.5f}")
		# if k > n_smooth and data_weight==1:
		if k > n_smooth + 1:
			loss_now = np.mean(loss_list[-n_smooth::])
			if loss_now <= loss_best:
				loss_best = loss_now
				print('################{}###############'.format(loss_best))
				
				torch.save(params_generator.state_dict(), save_path+'/generativemodelbest_'+coordinate_type+'_'+'RealNVP'+'_flow{}'.format(n_flow))

		if k == 0 or (k+1)%decay_rate == 0:
			torch.save(params_generator.state_dict(), save_path+'/generativemodel_loop{}_'.format((k+1)//decay_rate)+coordinate_type+'_'+'RealNVP'+'_flow{}'.format(n_flow))
	end_time = time.time()


	torch.save(params_generator.state_dict(), save_path+'/generativemodel_'+coordinate_type+'_'+'RealNVP'+'_flow{}'.format(n_flow))

	loss_all = {}
	loss_all['total'] = np.array(loss_list)
	loss_all['prior'] = np.array(loss_prior_list)
	if coordinate_type == 'polar':
		loss_all['sep'] = np.array(loss_sep_list)
		loss_all['pa'] = np.array(loss_pa_list)
	elif coordinate_type == 'cartesian':
		loss_all['raoff'] = np.array(loss_raoff_list)
		loss_all['decoff'] = np.array(loss_decoff_list)
	elif coordinate_type == 'all':
		loss_all['raoff'] = np.array(loss_raoff_list)
		loss_all['decoff'] = np.array(loss_decoff_list)
		loss_all['sep'] = np.array(loss_sep_list)
		loss_all['pa'] = np.array(loss_pa_list)
	elif coordinate_type == 'all_cartesian':
		loss_all['raoff'] = np.array(loss_raoff_list)
		loss_all['decoff'] = np.array(loss_decoff_list)
		oss_all['raoff_convert'] = np.array(loss_raoff_convert_list)
		loss_all['decoff_convert'] = np.array(loss_decoff_convert_list)

	loss_all['logdet'] = np.array(logdet_list)
	loss_all['time'] = end_time - start_time
	np.save(save_path+'/loss_'+coordinate_type+'_'+'RealNVP'+'_flow{}'.format(n_flow), loss_all)


	#####################################Visualization##################################################
	# params_generator.load_state_dict(torch.load(save_path+'/generativemodel_'+coordinate_type+'_'+'RealNVP'+'_flow{}'.format(n_flow)))

	# # save_path = './checkpoint/orbit_beta_pic_b/cartesian2/alpha09'
	# save_path = './checkpoint/orbit_beta_pic_b/all2/KL'

	# alpha_divergence = 0.99#0.8
	# coordinate_type = 'all'#'cartesian'
	# 

	# def Gen_samples(params_generator, rejsamp_flag=True, n_concat=10, alpha_divergence=1.0, coordinate_type='cartesian'):

	# 	for k in range(n_concat):

	# 		z_sample = torch.randn((n_batch, nparams)).to(device=device)
	# 		# generate image samples
	# 		params_samp, logdet = params_generator.reverse(z_sample)
	# 		params = torch.sigmoid(params_samp)

	# 		sma, ecc, inc, aop, pan, tau, plx, mtot = orbit_converter.forward(params)

	# 		if rejsamp_flag:

	# 			det_sigmoid = torch.sum(-params_samp-2*torch.nn.Softplus()(-params_samp), -1)
	# 			logdet = logdet + det_sigmoid
	# 			logprob = -logdet - 0.5*torch.sum(z_sample**2, 1)

	# 			if target == 'betapic':
	# 				raoff_torch, deoff_torch, vz_torch = calc_orbit_torch(epochs, sma, ecc, inc, aop, pan, tau, plx, mtot, max_iter=10, tolerance=1e-8, tau_ref_epoch=50000)
	# 			else:
	# 				raoff_torch, deoff_torch, vz_torch = calc_orbit_torch(epochs, sma, ecc, inc, aop, pan, tau, plx, mtot, max_iter=10, tolerance=1e-8)

	# 			sep_torch = torch.transpose(torch.sqrt(raoff_torch**2 + deoff_torch**2), 0, 1)
	# 			pa_torch = torch.transpose(torch.atan2(raoff_torch, deoff_torch), 0, 1)

	# 			raoff_torch = torch.transpose(raoff_torch, 0, 1)
	# 			deoff_torch = torch.transpose(deoff_torch, 0, 1)

	# 			if coordinate_type == 'polar':
	# 				# loss_sep = (sep_torch - sep)**2 / sep_err**2
	# 				# loss_sep = (torch.log(sep_torch) - torch.log(sep))**2 / (sep_err/sep)**2
	# 				loss_sep = (torch.log(sep_torch) - torch.log(sep) + 0.5*sep_err**2/sep**2)**2 / (sep_err/sep)**2
	# 				loss_pa  = (torch.atan2(torch.sin(pa-pa_torch), torch.cos(pa-pa_torch)))**2 / pa_err**2
	# 			elif coordinate_type == 'cartesian':
	# 				loss_raoff = (raoff_torch - raoff)**2 / raoff_err**2
	# 				loss_decoff = (deoff_torch - decoff)**2 / decoff_err**2
	# 			elif coordinate_type == 'all':
	# 				# loss_sep = (sep_torch[:, polar_exclude_cartesian_indices] - sep[polar_exclude_cartesian_indices])**2 / sep_err[polar_exclude_cartesian_indices]**2
	# 				loss_sep = (torch.log(sep_torch[:, polar_exclude_cartesian_indices]) - torch.log(sep[polar_exclude_cartesian_indices]))**2 / (sep_err[polar_exclude_cartesian_indices]/sep[polar_exclude_cartesian_indices])**2
	# 				loss_pa  = (torch.atan2(torch.sin(pa_torch[:, polar_exclude_cartesian_indices]-pa[polar_exclude_cartesian_indices]), torch.cos(pa_torch[:, polar_exclude_cartesian_indices]-pa[polar_exclude_cartesian_indices])))**2 / pa_err[polar_exclude_cartesian_indices]**2

	# 				loss_raoff = (raoff_torch[:, cartesian_indices] - raoff[cartesian_indices])**2 / raoff_err[cartesian_indices]**2
	# 				loss_decoff = (deoff_torch[:, cartesian_indices] - decoff[cartesian_indices])**2 / decoff_err[cartesian_indices]**2

	# 			if target == 'betapic':
	# 				loss_prior = (plx - 51.44)**2 / 0.12**2 + (mtot - 1.75)**2 / 0.05**2
	# 			elif target == 'GJ504':
	# 				loss_prior = (plx - 56.95)**2 / 0.26**2 + (mtot - 1.22)**2 / 0.08**2

	# 			# # Loss function for generative model
	# 			# if coordinate_type == 'polar':
	# 			# 	loss_data = sep_weight * torch.mean(loss_sep, 1) + pa_weight * torch.mean(loss_pa, 1) + prior_weight * torch.mean(loss_prior)
	# 			# elif coordinate_type == 'cartesian':
	# 			# 	loss_data = raoff_weight * torch.mean(loss_raoff, 1) + decoff_weight * torch.mean(loss_decoff, 1) + prior_weight * torch.mean(loss_prior)
	# 			# elif coordinate_type == 'all':
	# 			# 	loss_data = sep_weight * torch.mean(loss_sep, 1) + pa_weight * torch.mean(loss_pa, 1) + raoff_weight * torch.mean(loss_raoff, 1) + decoff_weight * torch.mean(loss_decoff, 1) + \
	# 			# 						prior_weight * torch.mean(loss_prior)


	# 			# Loss function for generative model
	# 			if coordinate_type == 'polar':
	# 				loss_orig = (0.5* torch.sum(loss_sep, -1) + 0.5 * torch.sum(loss_pa, -1) + 0.5 * loss_prior) + logprob
	# 			elif coordinate_type == 'cartesian':
	# 				loss_orig = (0.5* torch.sum(loss_raoff, -1) + 0.5 * torch.sum(loss_decoff, -1) + 0.5 * loss_prior) + logprob
	# 			elif coordinate_type == 'all':
	# 				loss_orig = (0.5* torch.sum(loss_sep, -1) + 0.5 * torch.sum(loss_pa, -1) + \
	# 									0.5* torch.sum(loss_raoff, -1) + 0.5 * torch.sum(loss_decoff, -1) + 0.5 * loss_prior) + logprob
	# 			# rej_prob = n_batch * nn.Softmax(dim=0)(-(1-alpha_divergence)*loss_orig)
	# 			rej_prob = n_batch * nn.Softmax(dim=0)(-loss_orig)

	# 			# rej_prob = rej_prob / torch.max(rej_prob)

	# 			rej_M = torch.sort(rej_prob)[0][int(0.99*n_batch)]
	# 			rej_prob = rej_prob / rej_M

	# 			U = torch.rand((n_batch, )).to(device=device)

	# 			# logU = torch.log(U)
	# 			# ind = torch.where((-loss_data - logprob - logM) > logU)[0]

	# 			ind = torch.where(rej_prob > U)[0]

	# 		else:
	# 			ind = np.arange(n_batch)
	# 			# ind = torch.sort(sep_weight * torch.mean(loss_pa, -1) + pa_weight * torch.mean(loss_sep, -1) + prior_weight * loss_prior)[1][0:int(0.99*n_batch)].detach().cpu().numpy()

	# 			# ind = torch.sort(raoff_weight * torch.mean(loss_raoff, -1) + decoff_weight * torch.mean(loss_decoff, -1) + prior_weight * loss_prior)[1][0:int(0.99*n_batch)].detach().cpu().numpy()

			

	# 		# aop[aop>np.pi] = aop[aop>np.pi] - 2 * np.pi
	# 		# tau[tau>0.5] = tau[tau>0.5] - 1.0
	# 		orbit_params1 = np.concatenate([sma[ind].unsqueeze(-1).detach().cpu().numpy(), (tau[ind].unsqueeze(-1).detach().cpu().numpy()), 
	# 			(180/np.pi*aop[ind].unsqueeze(-1).detach().cpu().numpy()), 
	# 			180/np.pi*pan[ind].unsqueeze(-1).detach().cpu().numpy(), 180/np.pi*inc[ind].unsqueeze(-1).detach().cpu().numpy(), 
	# 			ecc[ind].unsqueeze(-1).detach().cpu().numpy(),  mtot[ind].unsqueeze(-1).detach().cpu().numpy()], -1)



	# 		orbit_params2 = np.concatenate([sma[ind].unsqueeze(-1).detach().cpu().numpy(), ecc[ind].unsqueeze(-1).detach().cpu().numpy(),  
	# 			180/np.pi*inc[ind].unsqueeze(-1).detach().cpu().numpy(), 
	# 			(180/np.pi*aop[ind].unsqueeze(-1).detach().cpu().numpy()), 
	# 			180/np.pi*pan[ind].unsqueeze(-1).detach().cpu().numpy(), (tau[ind].unsqueeze(-1).detach().cpu().numpy()), 
	# 			plx[ind].unsqueeze(-1).detach().cpu().numpy(), mtot[ind].unsqueeze(-1).detach().cpu().numpy()], -1)

	# 		if k == 0:
	# 			orbit_params_all1 = np.array(orbit_params1)
	# 			orbit_params_all2 = np.array(orbit_params2)

	# 		else:
	# 			orbit_params_all1 = np.concatenate([orbit_params_all1, orbit_params1], 0)
	# 			orbit_params_all2 = np.concatenate([orbit_params_all2, orbit_params2], 0)

	# 	return orbit_params_all2


	def Gen_samples2(params_generator, rejsamp_flag=True, n_concat=10, alpha_divergence=1.0, coordinate_type='cartesian'):

		for k in range(n_concat):

			z_sample = torch.randn((n_batch, nparams)).to(device=device)
			# generate image samples
			params_samp, logdet = params_generator.reverse(z_sample)
			params = torch.sigmoid(params_samp)

			sma, ecc, inc, aop, pan, tau, plx, mtot = orbit_converter.forward(params)

			if rejsamp_flag:

				det_sigmoid = torch.sum(-params_samp-2*torch.nn.Softplus()(-params_samp), -1)
				logdet = logdet + det_sigmoid
				logprob = -logdet - 0.5*torch.sum(z_sample**2, 1)

				if target == 'betapic':
					raoff_torch, deoff_torch, vz_torch = calc_orbit_torch(epochs, sma, ecc, inc, aop, pan, tau, plx, mtot, max_iter=10, tolerance=1e-8, tau_ref_epoch=50000)
				else:
					raoff_torch, deoff_torch, vz_torch = calc_orbit_torch(epochs, sma, ecc, inc, aop, pan, tau, plx, mtot, max_iter=10, tolerance=1e-8)

				sep_torch = torch.transpose(torch.sqrt(raoff_torch**2 + deoff_torch**2), 0, 1)
				pa_torch = torch.transpose(torch.atan2(raoff_torch, deoff_torch), 0, 1)

				raoff_torch = torch.transpose(raoff_torch, 0, 1)
				deoff_torch = torch.transpose(deoff_torch, 0, 1)

				if coordinate_type == 'polar':
					loss_sep = (torch.log(sep_torch) - torch.log(sep) + 0.5*sep_err**2/sep**2)**2 / (sep_err/sep)**2
					loss_pa  = (torch.atan2(torch.sin(pa-pa_torch), torch.cos(pa-pa_torch)))**2 / pa_err**2
				elif coordinate_type == 'cartesian':
					loss_raoff = (raoff_torch - raoff)**2 / raoff_err**2
					loss_decoff = (deoff_torch - decoff)**2 / decoff_err**2
				elif coordinate_type == 'all':
					loss_sep = (torch.log(sep_torch[:, polar_exclude_cartesian_indices]) - torch.log(sep[polar_exclude_cartesian_indices]))**2 / (sep_err[polar_exclude_cartesian_indices]/sep[polar_exclude_cartesian_indices])**2
					loss_pa  = (torch.atan2(torch.sin(pa_torch[:, polar_exclude_cartesian_indices]-pa[polar_exclude_cartesian_indices]), torch.cos(pa_torch[:, polar_exclude_cartesian_indices]-pa[polar_exclude_cartesian_indices])))**2 / pa_err[polar_exclude_cartesian_indices]**2

					loss_raoff = (raoff_torch[:, cartesian_indices] - raoff[cartesian_indices])**2 / raoff_err[cartesian_indices]**2
					loss_decoff = (deoff_torch[:, cartesian_indices] - decoff[cartesian_indices])**2 / decoff_err[cartesian_indices]**2

				if target == 'betapic':
					loss_prior = (plx - 51.44)**2 / 0.12**2 + (mtot - 1.75)**2 / 0.05**2
				elif target == 'GJ504':
					loss_prior = (plx - 56.95)**2 / 0.26**2 + (mtot - 1.22)**2 / 0.08**2


				# Loss function for generative model
				if coordinate_type == 'polar':
					loss_orig = (0.5* torch.sum(loss_sep, -1) + 0.5 * torch.sum(loss_pa, -1) + 0.5 * loss_prior) + logprob
				elif coordinate_type == 'cartesian':
					loss_orig = (0.5* torch.sum(loss_raoff, -1) + 0.5 * torch.sum(loss_decoff, -1) + 0.5 * loss_prior) + logprob
				elif coordinate_type == 'all':
					loss_orig = (0.5* torch.sum(loss_sep, -1) + 0.5 * torch.sum(loss_pa, -1) + \
										0.5* torch.sum(loss_raoff, -1) + 0.5 * torch.sum(loss_decoff, -1) + 0.5 * loss_prior) + logprob
				# rej_prob = n_batch * nn.Softmax(dim=0)(-(1-alpha_divergence)*loss_orig)
				

				# importance_weights = nn.Softmax(dim=0)(-loss_orig)
				# importance_weights = torch.exp(-loss_orig)
				importance_weights = -loss_orig

				ind = np.arange(n_batch)

			else:
				ind = np.arange(n_batch)



			orbit_params2 = np.concatenate([sma[ind].unsqueeze(-1).detach().cpu().numpy(), ecc[ind].unsqueeze(-1).detach().cpu().numpy(),  
				180/np.pi*inc[ind].unsqueeze(-1).detach().cpu().numpy(), 
				(180/np.pi*aop[ind].unsqueeze(-1).detach().cpu().numpy()), 
				180/np.pi*pan[ind].unsqueeze(-1).detach().cpu().numpy(), (tau[ind].unsqueeze(-1).detach().cpu().numpy()), 
				plx[ind].unsqueeze(-1).detach().cpu().numpy(), mtot[ind].unsqueeze(-1).detach().cpu().numpy()], -1)

			if rejsamp_flag:
				orbit_params2 = np.concatenate([orbit_params2, importance_weights[ind].detach().cpu().numpy().reshape((-1, 1))], 1)
			if k == 0:
				orbit_params_all2 = np.array(orbit_params2)

			else:
				orbit_params_all2 = np.concatenate([orbit_params_all2, orbit_params2], 0)

		return orbit_params_all2
	# corner.corner(orbit_params_all1, labels=['sma', 'tau', 'aop', 'pan', 'inc', 'ecc', 'mtot'], bins=50, quantiles=[0.16, 0.84])



	params_generator.load_state_dict(torch.load(save_path+'/generativemodelbest_'+coordinate_type+'_'+'RealNVP'+'_flow{}'.format(n_flow)))
	params_generator.eval()

	orbit_params_all2 = Gen_samples2(params_generator, rejsamp_flag=False, n_concat=100, alpha_divergence=alpha_divergence, coordinate_type=coordinate_type)
	# # corner.corner(orbit_params_all2, labels=['sma', 'ecc', 'inc', 'aop', 'pan', 'tau', 'plx', 'mtot'], bins=50, quantiles=[0.16, 0.84])

	# corner.corner(orbit_params_all2, labels=['sma', 'ecc', 'inc', 'aop', 'pan', 'tau', 'plx', 'mtot'], bins=50, quantiles=[0.16, 0.84],
	# 			range=[[8, 40], [0.0, 0.9], [85.5, 93.0], [0, 360], [29.6, 33.0], [0.0, 1.0], [50.8, 52.1], [1.5, 2.0]])

	np.save(save_path+'/'+target+'_postsamples_norej_alpha{}.npy'.format(alpha_divergence), orbit_params_all2)



	orbit_params_all2 = Gen_samples2(params_generator, rejsamp_flag=True, n_concat=100, alpha_divergence=alpha_divergence, coordinate_type=coordinate_type)
	# corner.corner(orbit_params_all2, labels=['sma', 'ecc', 'inc', 'aop', 'pan', 'tau', 'plx', 'mtot'], bins=50, quantiles=[0.16, 0.84],
	# 			range=[[8, 40], [0.0, 0.9], [85.5, 93.0], [0, 360], [29.6, 33.0], [0.0, 1.0], [50.8, 52.1], [1.5, 2.0]])

	# np.save(save_path+'/'+target+'_postsamples_rej_alpha{}.npy'.format(alpha_divergence), orbit_params_all2)
	# np.save(save_path+'/'+target+'_postsamples_rej2_alpha{}.npy'.format(alpha_divergence), orbit_params_all2)
	np.save(save_path+'/'+target+'_postsamples_importance_alpha{}.npy'.format(alpha_divergence), orbit_params_all2)


	# corner.corner(samples, labels=['sma', 'ecc', 'inc', 'aop', 'pan', 'tau', 'plx', 'mtot'], bins=50, quantiles=[0.16, 0.84],
	# 			range=[[8, 40], [0.0, 0.9], [85.5, 93.0], [0, 360], [29.6, 33.0], [0.0, 1.0], [50.8, 52.1], [1.5, 2.0]])

	# corner.corner(orbit_params_all2, labels=['sma', 'ecc', 'inc', 'aop', 'pan', 'tau', 'plx', 'mtot'], bins=50, quantiles=[0.16, 0.84])

	# corner.corner(gmm_sample.detach().cpu().numpy(), quantiles=[0.16, 0.84])
	# corner.corner(z_sample.detach().cpu().numpy(), quantiles=[0.16, 0.84])


	for i in range(1, (n_epoch//decay_rate)+1):

		params_generator.load_state_dict(torch.load(save_path+'/generativemodel_loop{}_'.format(i)+coordinate_type+'_'+'RealNVP'+'_flow{}'.format(n_flow)))
		params_generator.eval() 

		orbit_params_all2 = Gen_samples2(params_generator, rejsamp_flag=False, n_concat=100, alpha_divergence=alpha_divergence, coordinate_type=coordinate_type)
		np.save(save_path+'/'+target+'_postsamples_norej_alpha{}_loop{}.npy'.format(alpha_divergence, i), orbit_params_all2)

		orbit_params_all2 = Gen_samples2(params_generator, rejsamp_flag=True, n_concat=100, alpha_divergence=alpha_divergence, coordinate_type=coordinate_type)
		# np.save(save_path+'/'+target+'_postsamples_rej_alpha{}_loop{}.npy'.format(alpha_divergence, i), orbit_params_all2)
		# np.save(save_path+'/'+target+'_postsamples_rej2_alpha{}_loop{}.npy'.format(alpha_divergence, i), orbit_params_all2)
		np.save(save_path+'/'+target+'_postsamples_importance_alpha{}_loop{}.npy'.format(alpha_divergence, i), orbit_params_all2)
