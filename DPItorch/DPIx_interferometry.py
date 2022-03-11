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
from scipy.ndimage import gaussian_filter
import skimage.transform
# import helpers as hp
import csv
import sys
import datetime
import warnings
import copy

import gc
import cv2

from astropy.io import fits
from pynfft.nfft import NFFT

from generative_model import glow_model
from generative_model import realnvpfc_model

from interferometry_helpers import *
from geometric_model import *


import argparse
from ehtim.imaging import starwarps as sw


import time


import ehtim.scattering.stochastic_optics as so
sm = so.ScatteringModel()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description="Deep Probabilistic Imaging Trainer for Interferometry")
parser.add_argument("--cuda", default=0, type=int, help="cuda index in use")

# parser.add_argument("--obspath", default='../dataset/interferometry6/mring_test/mring2.uvfits', type=str, help="EHT observation file path")
# parser.add_argument("--obspath", default='../dataset/interferometry6/mring_test/mring_scatt.uvfits', type=str, help="EHT observation file path")

# parser.add_argument("--obspath", default='../dataset/interferometry1/obs.uvfits', type=str, help="EHT observation file path")
# parser.add_argument("--obspath", default='../dataset/interferometry_m87/lo/hops_3601_M87+netcal.uvfits', type=str, help="EHT observation file path")
# parser.add_argument("--obspath", default='../dataset/interferometry_m87/lo/hops_3598_M87+netcal.uvfits', type=str, help="EHT observation file path")
parser.add_argument("--obspath", default='../dataset/interferometry_m87/synthetic_crescentfloorgaussian2/obs_mring_synthdata_allnoise_scanavg_sysnoise2.uvfits', type=str, help="EHT observation file path")

# parser.add_argument("--obspath", default='../dataset/sgra_standardized_uvfits_hops/uvfits/3599_besttime/hops/hops_3599_SGRA_LO_netcal_LMTcal_normalized_10s_regionIII.uvfits', type=str, help="EHT observation file path")
# parser.add_argument("--obspath", default='../dataset/SGRA_ER6_partitioned/hops_3599_SGRA_lo_V0_both_scan_netcal_LMTcal_10s_region_data/hops_3599_SGRA_lo_V0_both_scan_netcal_LMTcal_10s_regionIII.uvfits', type=str, help="EHT observation file path")
# parser.add_argument("--obspath", default='../dataset/SGRA_ER6_partitioned/hops_3598_SGRA_lo_V0_both_scan_netcal_LMTcal_10s_region_data/hops_3598_SGRA_lo_V0_both_scan_netcal_LMTcal_10s_regionIII.uvfits', type=str, help="EHT observation file path")

# parser.add_argument("--obspath", default='../dataset/sgra_standardized_uvfits_hops/uvfits/3599_besttime/hops/hops_3599_SGRA_HI_netcal_LMTcal_normalized_10s_regionIII.uvfits', type=str, help="EHT observation file path")
# parser.add_argument("--obspath", default='../dataset/grmhd_data_besttime/uvfits/mad_a0.94_i30_netcal_LMTcal.uvfits', type=str, help="EHT observation file path")
# parser.add_argument("--obspath", default='../dataset/grmhd_data_besttime/uvfits/mad_a0.94_i30_semistatic_pa1_netcal_LMTcal.uvfits', type=str, help="EHT observation file path")
# parser.add_argument("--obspath", default='../dataset/interferometry7/obs_singleframe0_scattering.uvfits', type=str, help="EHT observation file path")
# parser.add_argument("--obspath", default='../dataset/grmhd_data_besttime/uvfits/crescent0_allnoise.uvfits', type=str, help="EHT observation file path")
# parser.add_argument("--obspath", default='../dataset/grmhd_data_besttime/uvfits/crescent0_thermalnoise.uvfits', type=str, help="EHT observation file path")
# parser.add_argument("--obspath", default='../dataset/grmhd_data_besttime/uvfits/uniform_ring_thermalnoise.uvfits', type=str, help="EHT observation file path")
# parser.add_argument("--obspath", default='../dataset/interferometry3/obs_data_m87_day101.uvfits', type=str, help="EHT observation file path")
# parser.add_argument("--obspath", default='../dataset/interferometry1/obs.uvfits', type=str, help="EHT observation file path")
# parser.add_argument("--obspath", default='../dataset/interferometry1/obs_thermal.uvfits', type=str, help="EHT observation file path")


parser.add_argument("--impath", default='../dataset/interferometry_m87/synthetic_crescentfloorgaussian2/groundtruth.fits', type=str, help="groud-truth EHT image file path")

# parser.add_argument("--impath", default='../dataset/interferometry1/gt.fits', type=str, help="groud-truth EHT image file path")

# parser.add_argument("--save_path", default='./checkpoint/interferometry_sanity_snapshot_mring/trial5', type=str, help="file save path")
# parser.add_argument("--save_path", default='./checkpoint/interferometry_sanity_snapshot_mring_scatt/trial1', type=str, help="file save path")
# parser.add_argument("--save_path", default='./checkpoint/interferometry_besttime_grmhd_snapshot_mring_mad_a0.94_i30_netcalLMTcal', type=str, help="file save path")
# parser.add_argument("--save_path", default='./checkpoint/interferometry_sgra_snapshot_mring_LO_sanitycheck/trial1', type=str, help="file save path")
# parser.add_argument("--save_path", default='./checkpoint/interferometry_besttime_grmhd_crescent', type=str, help="file save path")
# parser.add_argument("--save_path", default='./checkpoint/interferometry_crescent0_crescent_sanitycheck/trial1', type=str, help="file save path")
# parser.add_argument("--save_path", default='./checkpoint/interferometry_crescent_m87snapshot/trial101', type=str, help="file save path")
# parser.add_argument("--save_path", default='./checkpoint/interferometry1_thermal', type=str, help="file save path")
# parser.add_argument("--save_path", default='./checkpoint/interferometry_sgra_snapshot_mringdecay/mring4/3599', type=str, help="file save path")
# parser.add_argument("--save_path", default='./checkpoint/interferometry_m87_mcfe/lo/3601/crescentfloornuissance/alpha1', type=str, help="file save path")
# parser.add_argument("--save_path", default='./checkpoint/interferometry_m87_mcfe/lo/3598/crescentfloornuissance/alpha1', type=str, help="file save path")
parser.add_argument("--save_path", default='./checkpoint/interferometry_m87_mcfe/synthetic/3598_processedsys2/crescentfloornuissance/beta1closure', type=str, help="file save path")


parser.add_argument("--npix", default=64, type=int, help="image shape (pixels)")
# parser.add_argument("--geometric_model", default='simple_crescent', type=str, help="geometric model used for fitting black hole")
# parser.add_argument("--geometric_model", default='simple_crescent2', type=str, help="geometric model used for fitting black hole")
# parser.add_argument("--geometric_model", default='simple_crescent_nuisance', type=str, help="geometric model used for fitting black hole")
# parser.add_argument("--geometric_model", default='mring', type=str, help="geometric model used for fitting black hole")
# parser.add_argument("--geometric_model", default='mring_nuisance', type=str, help="geometric model used for fitting black hole")
# parser.add_argument("--geometric_model", default='mring_spline_width', type=str, help="geometric model used for fitting black hole")
# parser.add_argument("--geometric_model", default='spline_brightness', type=str, help="geometric model used for fitting black hole")
# parser.add_argument("--geometric_model", default='spline_brightness_width', type=str, help="geometric model used for fitting black hole")
parser.add_argument("--geometric_model", default='simple_crescent_floor_nuisance', type=str, help="geometric model used for fitting black hole")
parser.add_argument("--n_gaussian", default=2, type=int, help="number of additional nuissance gaussian")


parser.add_argument("--mring_order", default=4, type=int, help="Order of Fourier series of ring brightness")
parser.add_argument("--spline_order", default=6, type=int, help="Order of Fourier series of ring brightness")


# parser.add_argument("--fov", default=160, type=float, help="field of view of the image in micro-arcsecond")
parser.add_argument("--fov", default=120, type=float, help="field of view of the image in micro-arcsecond")

parser.add_argument("--sys_noise", default=0.02, type=float, help="additive systematic noise")
# parser.add_argument("--sys_noise", default=0.05, type=float, help="additive systematic noise")

parser.add_argument("--prior_fwhm", default=80, type=float, help="fwhm of image prior in micro-arcsecond")
parser.add_argument("--model_form", default='realnvp', type=str, help="form of the deep generative model")
parser.add_argument("--n_flow", default=16, type=int, help="number of flows in RealNVP or Glow")
parser.add_argument("--logdet", default=1.0, type=float, help="logdet weight")

parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
# parser.add_argument("--clip", default=1e-5, type=float, help="gradient clip for neural network training")
parser.add_argument("--clip", default=1e-4, type=float, help="gradient clip for neural network training")

# parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
# parser.add_argument("--n_epoch", default=10000, type=int, help="number of epochs for training RealNVP")
# parser.add_argument("--clip", default=1e-4, type=float, help="gradient clip for neural network training")


# parser.add_argument("--lr", default=3e-5, type=float, help="learning rate")
# parser.add_argument("--n_epoch", default=10000, type=int, help="number of epochs for training RealNVP")
# parser.add_argument("--clip", default=3e-5, type=float, help="gradient clip for neural network training")

# parser.add_argument("--data_product", default='vis', type=str, help="data product used for reconstruction")
parser.add_argument("--data_product", default='cphase_logcamp', type=str, help="data product used for reconstruction")
# parser.add_argument("--data_product", default='cphase_amp', type=str, help="data product used for reconstruction")
# parser.add_argument("--data_product", default='cphase_logcamp_amp', type=str, help="data product used for reconstruction")

parser.add_argument("--divergence_type", default='alpha', type=str, help="KL or alpha, type of objective divergence used for variational inference")
parser.add_argument("--alpha_divergence", default=1.0, type=float, help="hyperparameters for alpha divergence")
parser.add_argument("--start_order", default=4, type=float, help="start order")
parser.add_argument("--decay_rate", default=2000, type=float, help="decay rate")
parser.add_argument("--n_epoch", default=10000, type=int, help="number of epochs for training RealNVP")

parser.add_argument("--beta", default=0.0, type=float, help="hyperparameters for alpha divergence")
# parser.add_argument("--beta", default=1.0, type=float, help="hyperparameters for alpha divergence")

parser.add_argument("--ttype", default='nfft', type=str, help="fourier transform computation method")



if __name__ == "__main__":
	args = parser.parse_args()
	obs_path = args.obspath
	# gt_path = args.impath
	npix = args.npix
	# nparams = args.nparams
	geometric_model = args.geometric_model

	if torch.cuda.is_available():
		device = torch.device('cuda:{}'.format(args.cuda))
	else:
		device = torch.device('cpu')


	obs = eh.obsdata.load_uvfits(obs_path)


	# # Add systematic noisef
	# sys_noise = args.sys_noise#0.05
	# obs = obs.add_fractional_noise(sys_noise)


	# define the prior image for MEM regularizer
	flux_const = np.median(obs.unpack_bl('AP', 'AA', 'amp')['amp'])#
	# flux_const = np.median(obs.unpack_bl('APEX', 'ALMA', 'amp')['amp'])
	prior_fwhm = args.prior_fwhm*eh.RADPERUAS#60*eh.RADPERUAS#
	fov = args.fov*eh.RADPERUAS
	zbl = flux_const#2.0#0.8#

	prior = eh.image.make_square(obs, npix, fov)
	prior = prior.add_tophat(zbl, prior_fwhm/2.0)
	prior = prior.blur_circ(obs.res())

	simim = prior.copy()

	# simim = eh.image.load_fits(gt_path)
	# simim = simim.regrid_image(fov, npix)
	simim.ra = obs.ra
	simim.dec = obs.dec
	simim.rf = obs.rf

	save_path = args.save_path
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	# define the eht observation function
	ttype = args.ttype
	nufft_ob = KbNufft(im_size=(npix, npix), numpoints=3)
	dft_mat, ktraj_vis, pulsefac_vis_torch, cphase_ind_list, cphase_sign_list, camp_ind_list = Obs_params_torch(obs, simim, snrcut=0.0, ttype=ttype)
	eht_obs_torch = eht_observation_pytorch(npix, nufft_ob, dft_mat, ktraj_vis, pulsefac_vis_torch, cphase_ind_list, cphase_sign_list, camp_ind_list, device, ttype=ttype)


	if args.data_product == 'cphase_logcamp':
		flux_flag = False
	else:
		flux_flag = True


	flux_range = [0.8*flux_const, 1.2*flux_const]
	r_range = [10.0, 40.0]#[10.0, 50.0]
	if geometric_model == 'simple_crescent':
		img_converter = SimpleCrescent_Param2Img(npix, r_range=r_range, fov=args.fov, flux_flag=flux_flag, flux_range=flux_range).to(device=device)
	elif geometric_model == 'simple_crescent2':
		img_converter = SimpleCrescent2_Param2Img(npix, r_range=r_range, fov=args.fov, flux_flag=flux_flag, flux_range=flux_range).to(device=device)
	elif geometric_model == 'simple_crescent_nuisance':
		img_converter = SimpleCrescentNuisance_Param2Img(npix, r_range=r_range, fov=args.fov, n_gaussian=args.n_gaussian, flux_flag=flux_flag, flux_range=flux_range).to(device=device)
	elif geometric_model == 'simple_crescent_floor_nuisance':
		img_converter = SimpleCrescentNuisanceFloor_Param2Img(npix, r_range=r_range, fov=args.fov, n_gaussian=args.n_gaussian, flux_flag=flux_flag, flux_range=flux_range).to(device=device)
	elif geometric_model == 'mring':
		img_converter = MringPhase_Param2Img(npix, r_range=r_range, fov=args.fov, n_order=args.mring_order, flux_flag=flux_flag, flux_range=flux_range).to(device=device)
	elif geometric_model == 'mring2':
		img_converter = MringPhase2_Param2Img(npix, r_range=r_range, fov=args.fov, n_order=args.mring_order, flux_flag=flux_flag, flux_range=flux_range).to(device=device)
	elif geometric_model == 'mring_floor':
		img_converter = MringPhaseFloor_Param2Img(npix, r_range=r_range, floor_range=[0.0, 1.0], fov=args.fov, n_order=args.mring_order, flux_flag=flux_flag, flux_range=flux_range).to(device=device)
	elif geometric_model == 'mring_nuisance':
		img_converter = MringPhaseNuisance_Param2Img(npix, r_range=r_range, fov=args.fov, n_order=args.mring_order, n_gaussian=2,
													flux_flag=flux_flag, flux_range=flux_range).to(device=device)
	elif geometric_model == 'spline_brightness':
		img_converter = SplineBrightnessRing_Param2Img(npix, fov=args.fov, n_pieces=args.spline_order, flux_flag=flux_flag, flux_range=flux_range).to(device=device)
	elif geometric_model == 'spline_brightness_width':
		img_converter = SplineBrightnessWidthRing_Param2Img(npix, fov=args.fov, n_pieces=args.spline_order, flux_flag=flux_flag, flux_range=flux_range).to(device=device)
	elif geometric_model == 'mring_spline_width':
		img_converter = MringSplineWidthRing_Param2Img(npix, fov=args.fov, n_order=args.mring_order, n_pieces=args.spline_order, flux_flag=flux_flag, flux_range=flux_range).to(device=device)

	n_flow = args.n_flow
	affine = True
	nparams = img_converter.nparams

	params_generator = realnvpfc_model.RealNVP(nparams, n_flow, affine=affine, seqfrac=1/16, permute='random', batch_norm=True).to(device)

	# define the losses and weights for very long baseline interferometric imaging
	Loss_vis_img = Loss_vis_diff(obs.data['sigma'], device)
	Loss_cphase_img = Loss_angle_diff(obs.cphase['sigmacp'], device)
	# Loss_visamp_img = Loss_logamp_diff(obs.data['sigma'], device)
	Loss_visamp_img = Loss_visamp_diff(obs.data['sigma'], device)
	# Loss_logca_img = Loss_logca_diff(obs.camp['sigmaca'], device)
	Loss_logca_img2 = Loss_logca_diff2(obs.logcamp['sigmaca'], device)


	if args.data_product == 'vis':
		vis_weight = 1.0
		camp_weight = 0.0
		cphase_weight = 0.0#10.0#
		visamp_weight = 0.0#1e-3#1e-4#1e-5#1.0#
		logdet_weight = 2.0 * args.logdet / len(obs.data['vis'])#2.0 * args.logdet / len(obs.data['vis'])#1.0 / (npix*npix) #
		scale_factor = 1.0 / len(obs.data['vis'])
	elif args.data_product == 'cphase_logcamp':
		vis_weight = 0.0
		camp_weight = 1.0
		cphase_weight = len(obs.cphase['cphase'])/len(obs.camp['camp'])#1.0#10.0#
		visamp_weight = 0.0#1e-3#1e-4#1e-5#1.0#
		logdet_weight = 2.0 * args.logdet / len(obs.camp['camp'])#1.0 / (npix*npix) #
		scale_factor = 1.0 / len(obs.camp['camp'])
	elif args.data_product == 'cphase_amp':
		vis_weight = 0.0
		camp_weight = 0.0
		cphase_weight = len(obs.cphase['cphase']) / len(obs.data['vis'])#1.0#10.0#
		visamp_weight = 1.0#0.0#1e-3#1e-4#1e-5#1.0#
		logdet_weight = 2.0 * args.logdet / len(obs.data['vis'])#1.0 / (npix*npix) #
		scale_factor = 1.0 / len(obs.data['vis'])
	elif args.data_product == 'cphase_logcamp_amp':
		vis_weight = 0.0
		camp_weight = len(obs.camp['camp']) / len(obs.data['vis'])
		cphase_weight = len(obs.cphase['cphase']) / len(obs.data['vis'])#1.0#10.0#
		visamp_weight = 0.1#1.0#0.0#1e-3#1e-4#1e-5#1.0#
		logdet_weight = 2.0 * args.logdet / len(obs.data['vis'])#1.0 / (npix*npix) #
		scale_factor = 1.0 / len(obs.data['vis'])


	vis_true = torch.Tensor(np.concatenate([np.expand_dims(obs.data['vis'].real, 0), 
							np.expand_dims(obs.data['vis'].imag, 0)], 0)).to(device=device)
	data_arr = obs.unpack(['amp'], debias=True)
	visamp_true = torch.Tensor(np.array(data_arr['amp'])).to(device=device)
	# visamp_true = torch.Tensor(np.array(np.abs(obs.data['vis'])) - 0.5*np.array(obs.data['sigma'])**2/np.array(np.abs(obs.data['vis']))).to(device=device)
	cphase_true = torch.Tensor(np.array(obs.cphase['cphase'])).to(device=device)
	camp_true = torch.Tensor(np.array(obs.camp['camp'])).to(device=device)
	logcamp_true = torch.Tensor(np.array(obs.logcamp['camp'])).to(device=device)
	prior_im = torch.Tensor(np.array(prior.imvec.reshape((npix, npix)))).to(device=device)

	# optimize both scale and image generator
	lr = args.lr
	optimizer = optim.Adam(params_generator.parameters(), lr = lr)


	n_epoch = args.n_epoch#30000#10000#100000#50000#100#
	loss_list = []
	loss_prior_list = []
	loss_cphase_list = []
	loss_logca_list = []
	loss_visamp_list = []
	loss_vis_list = []
	logdet_list = []

	n_batch = 2048#1024#256#64#512#64#32#8
	n_smooth = 10
	loss_best = 1e5

	decay_rate = args.decay_rate#2000
	starting_order = args.start_order#3#2#
	divergence_type = args.divergence_type
	beta = args.beta
	if beta == 0:
		alpha_divergence = args.alpha_divergence
	else:
		alpha_divergence = 1-beta*scale_factor

	t_start = time.time()
	for k in range(n_epoch):

		data_weight = min(10**(-starting_order+k/decay_rate), 1.0)


		z_sample = torch.randn((n_batch, nparams)).to(device=device)


		# generate image samples
		params_samp, logdet = params_generator.reverse(z_sample)
		params = torch.sigmoid(params_samp)

		img = img_converter.forward(params)

		# compute log determinant
		det_sigmoid = torch.sum(-params_samp-2*torch.nn.Softplus()(-params_samp), -1)
		logdet = logdet + det_sigmoid


		vis, visamp, cphase, logcamp = eht_obs_torch(img)
		if vis_weight == 0:
			loss_visamp = Loss_visamp_img(visamp_true, visamp) if visamp_weight>0 else 0
			loss_cphase = Loss_cphase_img(cphase_true, cphase) if cphase_weight>0 else 0
			loss_camp = Loss_logca_img2(logcamp_true, logcamp) if camp_weight>0 else 0
			loss_data = camp_weight * loss_camp + cphase_weight * loss_cphase + visamp_weight * loss_visamp
			loss_data = 0.5 * loss_data / scale_factor
		else:
			loss_vis = Loss_vis_img(vis_true, vis)
			loss_data = vis_weight * loss_vis
			loss_data = 0.5 * loss_data / scale_factor

		logprob = -logdet - 0.5*torch.sum(z_sample**2, 1)

		loss = data_weight * loss_data + logprob
		if divergence_type == 'KL' or alpha_divergence == 1:
			loss = torch.mean(scale_factor * loss)
		else:
			if beta == 0:
				rej_weights = nn.Softmax(dim=0)(-(1-alpha_divergence)*loss).detach()
			else:
				rej_weights = nn.Softmax(dim=0)(-beta * scale_factor * loss).detach()
			loss = torch.sum(rej_weights * scale_factor * loss)


		loss_orig = loss_data + logprob
		if divergence_type == 'KL' or alpha_divergence == 1:
			loss_orig = torch.mean(scale_factor * loss_orig)
		else:
			if beta == 0:
				loss_orig = scale_factor * torch.log(torch.mean(torch.exp(-(1-alpha_divergence)*loss_orig)))/(alpha_divergence-1)
			else:
				loss_orig = scale_factor * torch.log(torch.mean(torch.exp(-beta * scale_factor * loss_orig)))/(alpha_divergence-1)

		
		optimizer.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(params_generator.parameters(), args.clip)
		optimizer.step()

		if vis_weight == 0:
			# loss_list.append(loss.detach().cpu().numpy())
			loss_list.append(loss_orig.detach().cpu().numpy())
			loss_cphase_list.append(torch.mean(loss_cphase).detach().cpu().numpy() if cphase_weight>0 else 0)
			loss_logca_list.append(torch.mean(loss_camp).detach().cpu().numpy() if camp_weight>0 else 0)
			loss_visamp_list.append(torch.mean(loss_visamp).detach().cpu().numpy() if visamp_weight>0 else 0)
			logdet_list.append(-torch.mean(logdet).detach().cpu().numpy()/nparams)
			print(f"epoch: {k:}, loss: {loss_list[-1]:.5f}, loss cphase: {loss_cphase_list[-1]:.5f}, loss camp: {loss_logca_list[-1]:.5f}, loss visamp: {loss_visamp_list[-1]:.5f}, logdet: {logdet_list[-1]:.5f}")
		else:
			# loss_list.append(loss.detach().cpu().numpy())
			loss_list.append(loss_orig.detach().cpu().numpy())
			loss_vis_list.append(torch.mean(loss_vis).detach().cpu().numpy() if vis_weight>0 else 0)
			logdet_list.append(-torch.mean(logdet).detach().cpu().numpy()/nparams)
			print(f"epoch: {k:}, loss: {loss_list[-1]:.5f}, loss vis: {loss_vis_list[-1]:.5f}, logdet: {logdet_list[-1]:.5f}")

		if k > n_smooth + 1:
			loss_now = np.mean(loss_list[-n_smooth::])
			if loss_now <= loss_best:
				loss_best = loss_now
				print('################{}###############'.format(loss_best))
				torch.save(params_generator.state_dict(), save_path+'/generativemodelbest_'+geometric_model+'_res{}flow{}logdet{}'.format(npix, n_flow, args.logdet)+'_'+args.data_product)

		if (k == 0) or (k+1)%decay_rate == 0:
			torch.save(params_generator.state_dict(), save_path+'/generativemodel_decay{}_'.format((k+1)//decay_rate)+geometric_model+'_res{}flow{}logdet{}'.format(npix, n_flow, args.logdet)+'_'+args.data_product)

		# print(f"epoch: {((n_epoch//n_blur)*k_blur+k):}, loss: {loss_list[-1]:.5f}, loss cphase: {loss_cphase_list[-1]:.5f}, loss camp: {loss_logca_list[-1]:.5f}, loss visamp: {loss_visamp_list[-1]:.5f}, loss prior: {loss_prior_list[-1]:.5f}")
	t_end = time.time()

	torch.save(params_generator.state_dict(), save_path+'/generativemodel_'+geometric_model+'_res{}flow{}logdet{}'.format(npix, n_flow, args.logdet)+'_'+args.data_product)
	np.save(save_path+'/generativeimage_'+geometric_model+'_res{}flow{}logdet{}'.format(npix, n_flow, args.logdet)+'_'+args.data_product+'.npy', img.cpu().detach().numpy().squeeze())

	loss_all = {}
	loss_all['total'] = np.array(loss_list)
	if vis_weight == 0:
		loss_all['cphase'] = np.array(loss_cphase_list)
		loss_all['logca'] = np.array(loss_logca_list)
		loss_all['visamp'] = np.array(loss_visamp_list)
	else:
		loss_all['vis'] = np.array(loss_vis_list)
	loss_all['logdet'] = np.array(logdet_list)
	loss_all['time'] = t_end - t_start
	np.save(save_path+'/loss_'+geometric_model+'_res{}flow{}logdet{}'.format(npix, n_flow, args.logdet)+'_'+args.data_product+'.npy', loss_all)
	# np.save(save_path+'/positionangle{}_'.format(ind)+args.model_form+'_res{}flow{}logdet{}_closure_fluxcentermemtsv.npy'.format(npix, n_flow, args.logdet), bias.detach().cpu().numpy()*np.ones(1))


	def Gen_samples(params_generator, rejsamp_flag=True, n_concat=10, alpha_divergence=1.0, coordinate_type='cartesian'):

		for k in range(n_concat):

			data_weight = min(10**(-starting_order+k/decay_rate), 1.0)

			z_sample = torch.randn((n_batch, nparams)).to(device=device)

			# generate image samples
			params_samp, logdet = params_generator.reverse(z_sample)
			params = torch.sigmoid(params_samp)

			if rejsamp_flag:
				img = img_converter.forward(params)

				# compute log determinant
				det_sigmoid = torch.sum(-params_samp-2*torch.nn.Softplus()(-params_samp), -1)
				logdet = logdet + det_sigmoid

				vis, visamp, cphase, logcamp = eht_obs_torch(img)
				if vis_weight == 0:
					loss_visamp = Loss_visamp_img(visamp_true, visamp) if visamp_weight>0 else 0
					loss_cphase = Loss_cphase_img(cphase_true, cphase) if cphase_weight>0 else 0
					loss_camp = Loss_logca_img2(logcamp_true, logcamp) if camp_weight>0 else 0
					loss_data = camp_weight * loss_camp + cphase_weight * loss_cphase + visamp_weight * loss_visamp
				else:
					loss_vis = Loss_vis_img(vis_true, vis)
					loss_data = vis_weight * loss_vis

				logprob = -logdet - 0.5*torch.sum(z_sample**2, 1)

				loss_orig = 0.5 * loss_data + logprob

				# rej_prob = n_batch * nn.Softmax(dim=0)(-(1-alpha_divergence)*loss_orig)
				rej_prob = n_batch * nn.Softmax(dim=0)(-loss_orig)
				rej_prob = rej_prob / torch.max(rej_prob)
				U = torch.rand((n_batch, )).to(device=device)

				ind = torch.where(rej_prob > U)[0]

				img_map = img[torch.argmin(loss_data)]
				img_mean = torch.mean(img, 0)
				img_std = torch.std(img, 0)

			else:
				ind = np.arange(n_batch)
				# ind = torch.sort(sep_weight * torch.mean(loss_pa, -1) + pa_weight * torch.mean(loss_sep, -1) + prior_weight * loss_prior)[1][0:int(0.99*n_batch)].detach().cpu().numpy()

				# ind = torch.sort(raoff_weight * torch.mean(loss_raoff, -1) + decoff_weight * torch.mean(loss_decoff, -1) + prior_weight * loss_prior)[1][0:int(0.99*n_batch)].detach().cpu().numpy()

			
			if geometric_model == 'simple_crescent_floor_nuisance':
				if img_converter.flux_flag:
					r, sigma, s, eta, flux, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x, sigma_y, rho = img_converter.compute_features(params)

					crescent_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
										2 * 0.5 * fov / eh.RADPERUAS * sigma[ind].reshape((-1, 1)).detach().cpu().numpy(),
										s[ind].reshape((-1, 1)).detach().cpu().numpy(),
										180 / np.pi * eta[ind].reshape((-1, 1)).detach().cpu().numpy(),
										crescent_flux[ind].reshape((-1, 1)).detach().cpu().numpy(),
										floor[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)

					if args.n_gaussian > 0:
						gaussian_params = np.concatenate([np.concatenate([nuisance_scale[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * nuisance_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * nuisance_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * sigma_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * sigma_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											180 / np.pi * rho[i][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for i in range(img_converter.n_gaussian)], -1)
						
						model_params = np.concatenate([crescent_params, gaussian_params, flux[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)
					else:
						model_params = np.concatenate([crescent_params, flux[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)

							
				else:
					r, sigma, s, eta, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x, sigma_y, rho = img_converter.compute_features(params)

					crescent_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
										2 * 0.5 * fov / eh.RADPERUAS * sigma[ind].reshape((-1, 1)).detach().cpu().numpy(),
										s[ind].reshape((-1, 1)).detach().cpu().numpy(),
										180 / np.pi * eta[ind].reshape((-1, 1)).detach().cpu().numpy(),
										crescent_flux[ind].reshape((-1, 1)).detach().cpu().numpy(),
										floor[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)

					if args.n_gaussian > 0:
						gaussian_params = np.concatenate([np.concatenate([nuisance_scale[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * nuisance_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * nuisance_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * sigma_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * sigma_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											180 / np.pi * rho[i][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for i in range(img_converter.n_gaussian)], -1)
						
						model_params = np.concatenate([crescent_params, gaussian_params], -1)
					else:
						model_params = np.array(crescent_params)

			elif geometric_model == 'floor_nuisance':
				if img_converter.flux_flag:
					r, flux, crescent_flux, nuisance_scale, nuisance_x, nuisance_y, sigma_x, sigma_y, rho = img_converter.compute_features(params)

					crescent_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
										crescent_flux[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)

					if args.n_gaussian > 0:
						gaussian_params = np.concatenate([np.concatenate([nuisance_scale[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * nuisance_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * nuisance_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * sigma_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * sigma_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											180 / np.pi * rho[i][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for i in range(img_converter.n_gaussian)], -1)
						
						model_params = np.concatenate([crescent_params, gaussian_params, flux[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)
					else:
						model_params = np.concatenate([crescent_params, flux[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)

							
				else:
					r, crescent_flux, nuisance_scale, nuisance_x, nuisance_y, sigma_x, sigma_y, rho = img_converter.compute_features(params)

					crescent_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
										crescent_flux[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)

					if args.n_gaussian > 0:
						gaussian_params = np.concatenate([np.concatenate([nuisance_scale[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * nuisance_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * nuisance_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * sigma_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * sigma_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											180 / np.pi * rho[i][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for i in range(img_converter.n_gaussian)], -1)
						
						model_params = np.concatenate([crescent_params, gaussian_params], -1)
					else:
						model_params = np.array(crescent_params)

			elif geometric_model == 'nuisance_gaussian':
				if img_converter.flux_flag:
					flux, nuisance_scale, nuisance_x, nuisance_y, sigma_x, sigma_y, rho = img_converter.compute_features(params)
					# flux, nuisance_x, nuisance_y, sigma_x, sigma_y, rho = img_converter.compute_features(params)
					gaussian_params = np.concatenate([np.concatenate([nuisance_scale[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
										0.5 * fov  / eh.RADPERUAS * nuisance_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
										0.5 * fov  / eh.RADPERUAS * nuisance_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
										0.5 * fov  / eh.RADPERUAS * sigma_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
										0.5 * fov  / eh.RADPERUAS * sigma_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
										180 / np.pi * rho[i][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for i in range(img_converter.n_gaussian)], -1)
					# gaussian_params = np.concatenate([np.concatenate([0.5 * fov  / eh.RADPERUAS * nuisance_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
					# 					0.5 * fov  / eh.RADPERUAS * nuisance_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
					# 					0.5 * fov / eh.RADPERUAS * sigma_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
					# 					0.5 * fov / eh.RADPERUAS * sigma_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
					# 					180 / np.pi * rho[i][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for i in range(img_converter.n_gaussian)], -1)
					model_params = np.concatenate([gaussian_params, flux[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)
				else:
					nuisance_scale, nuisance_x, nuisance_y, sigma_x, sigma_y, rho = img_converter.compute_features(params)
					# nuisance_x, nuisance_y, sigma_x, sigma_y, rho = img_converter.compute_features(params)
					gaussian_params = np.concatenate([np.concatenate([nuisance_scale[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
										0.5 * fov / eh.RADPERUAS * nuisance_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
										0.5 * fov / eh.RADPERUAS * nuisance_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
										0.5 * fov / eh.RADPERUAS * sigma_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
										0.5 * fov / eh.RADPERUAS * sigma_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
										180 / np.pi * rho[i][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for i in range(img_converter.n_gaussian)], -1)
					# gaussian_params = np.concatenate([np.concatenate([0.5 * fov / eh.RADPERUAS * nuisance_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
					# 					0.5 * fov / eh.RADPERUAS * nuisance_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
					# 					0.5 * fov / eh.RADPERUAS * sigma_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
					# 					0.5 * fov / eh.RADPERUAS * sigma_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
					# 					180 / np.pi * rho[i][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for i in range(img_converter.n_gaussian)], -1)

					model_params = np.array(gaussian_params)

			elif geometric_model == 'mring':
				if img_converter.flux_flag:
					r, sigma, s_list, eta_list, flux = img_converter.compute_features(params)
					model_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
												2 * 0.5 * fov / eh.RADPERUAS * sigma[ind].reshape((-1, 1)).detach().cpu().numpy()] + \
												[np.concatenate([s_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy(),
												180/np.pi * eta_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for k_order in range(img_converter.n_order)] + \
												[flux[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)
												
				else:
					r, sigma, s_list, eta_list = img_converter.compute_features(params)
					model_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
												2 * 0.5 * fov / eh.RADPERUAS * sigma[ind].reshape((-1, 1)).detach().cpu().numpy()] + \
												[np.concatenate([s_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy(),
												180/np.pi * eta_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for k_order in range(img_converter.n_order)], -1)
			
			elif geometric_model == 'mring_floor':
				if img_converter.flux_flag:
					r, sigma, s_list, eta_list, floor, flux = img_converter.compute_features(params)
					model_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
												2 * 0.5 * fov / eh.RADPERUAS * sigma[ind].reshape((-1, 1)).detach().cpu().numpy()] + \
												[np.concatenate([s_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy(),
												180/np.pi * eta_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for k_order in range(img_converter.n_order)] + \
												[floor[ind].reshape((-1, 1)).detach().cpu().numpy(),
												flux[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)
												
				else:
					r, sigma, s_list, eta_list, floor = img_converter.compute_features(params)
					model_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
												2 * 0.5 * fov / eh.RADPERUAS * sigma[ind].reshape((-1, 1)).detach().cpu().numpy()] + \
												[np.concatenate([s_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy(),
												180/np.pi * eta_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for k_order in range(img_converter.n_order)] + \
												[floor[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)
			elif geometric_model == 'mring_gaussianfloor':
				if img_converter.flux_flag:
					r, sigma, s_list, eta_list, floor, floor_r, flux = img_converter.compute_features(params)
					model_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
												2 * 0.5 * fov / eh.RADPERUAS * sigma[ind].reshape((-1, 1)).detach().cpu().numpy()] + \
												[np.concatenate([s_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy(),
												180/np.pi * eta_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for k_order in range(img_converter.n_order)] + \
												[floor[ind].reshape((-1, 1)).detach().cpu().numpy(),
												2 * 0.5 * fov / eh.RADPERUAS * floor_r[ind].reshape((-1, 1)).detach().cpu().numpy(),
												flux[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)
												
				else:
					r, sigma, s_list, eta_list, floor, floor_r = img_converter.compute_features(params)
					model_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
												2 * 0.5 * fov / eh.RADPERUAS * sigma[ind].reshape((-1, 1)).detach().cpu().numpy()] + \
												[np.concatenate([s_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy(),
												180/np.pi * eta_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for k_order in range(img_converter.n_order)] + \
												[floor[ind].reshape((-1, 1)).detach().cpu().numpy(),
												2 * 0.5 * fov / eh.RADPERUAS * floor_r[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)
			

			if k == 0:
				model_params_all = np.array(model_params)
			else:
				model_params_all = np.concatenate([model_params_all, model_params], 0)
		if rejsamp_flag:
			return model_params_all, img_map.squeeze().detach().cpu().numpy(), img_mean.squeeze().detach().cpu().numpy(), img_std.squeeze().detach().cpu().numpy()
		else:
			return model_params_all


	def Gen_samples2(params_generator, rejsamp_flag=True, n_concat=10, alpha_divergence=1.0, coordinate_type='cartesian'):

		for k in range(n_concat):

			data_weight = min(10**(-starting_order+k/decay_rate), 1.0)

			z_sample = torch.randn((n_batch, nparams)).to(device=device)

			# generate image samples
			params_samp, logdet = params_generator.reverse(z_sample)
			params = torch.sigmoid(params_samp)

			if rejsamp_flag:
				img = img_converter.forward(params)

				# compute log determinant
				det_sigmoid = torch.sum(-params_samp-2*torch.nn.Softplus()(-params_samp), -1)
				logdet = logdet + det_sigmoid

				vis, visamp, cphase, logcamp = eht_obs_torch(img)
				if vis_weight == 0:
					loss_visamp = Loss_visamp_img(visamp_true, visamp) if visamp_weight>0 else 0
					loss_cphase = Loss_cphase_img(cphase_true, cphase) if cphase_weight>0 else 0
					loss_camp = Loss_logca_img2(logcamp_true, logcamp) if camp_weight>0 else 0
					loss_data = camp_weight * loss_camp + cphase_weight * loss_cphase + visamp_weight * loss_visamp
				else:
					loss_vis = Loss_vis_img(vis_true, vis)
					loss_data = vis_weight * loss_vis

				logprob = -logdet - 0.5*torch.sum(z_sample**2, 1)

				loss_orig = 0.5 * loss_data + logprob


				importance_weights2 = nn.Softmax(dim=0)(-loss_orig)
				# importance_weights = torch.exp(-loss_orig)
				importance_weights = -loss_orig#importance_weights2


				img_map = img[torch.argmin(loss_data)]

				img_mean = torch.sum(importance_weights2.reshape((-1, 1, 1))*img, 0)
				img_std = torch.sqrt(torch.sum(importance_weights2.reshape((-1, 1, 1))*(img-img_mean)**2, 0))

				ind = np.arange(n_batch)

			else:
				ind = np.arange(n_batch)
				# ind = torch.sort(sep_weight * torch.mean(loss_pa, -1) + pa_weight * torch.mean(loss_sep, -1) + prior_weight * loss_prior)[1][0:int(0.99*n_batch)].detach().cpu().numpy()

				# ind = torch.sort(raoff_weight * torch.mean(loss_raoff, -1) + decoff_weight * torch.mean(loss_decoff, -1) + prior_weight * loss_prior)[1][0:int(0.99*n_batch)].detach().cpu().numpy()

			
			if geometric_model == 'simple_crescent_floor_nuisance':
				if img_converter.flux_flag:
					r, sigma, s, eta, flux, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x, sigma_y, rho = img_converter.compute_features(params)

					crescent_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
										2 * 0.5 * fov / eh.RADPERUAS * sigma[ind].reshape((-1, 1)).detach().cpu().numpy(),
										s[ind].reshape((-1, 1)).detach().cpu().numpy(),
										180 / np.pi * eta[ind].reshape((-1, 1)).detach().cpu().numpy(),
										crescent_flux[ind].reshape((-1, 1)).detach().cpu().numpy(),
										floor[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)
					
					if args.n_gaussian > 0:
						gaussian_params = np.concatenate([np.concatenate([nuisance_scale[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * nuisance_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * nuisance_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * sigma_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * sigma_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											180 / np.pi * rho[i][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for i in range(img_converter.n_gaussian)], -1)
						
						model_params = np.concatenate([crescent_params, gaussian_params, flux[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)
					else:
						model_params = np.concatenate([crescent_params, flux[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)
							
				else:
					r, sigma, s, eta, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x, sigma_y, rho = img_converter.compute_features(params)

					crescent_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
										2 * 0.5 * fov / eh.RADPERUAS * sigma[ind].reshape((-1, 1)).detach().cpu().numpy(),
										s[ind].reshape((-1, 1)).detach().cpu().numpy(),
										180 / np.pi * eta[ind].reshape((-1, 1)).detach().cpu().numpy(),
										crescent_flux[ind].reshape((-1, 1)).detach().cpu().numpy(),
										floor[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)

					if args.n_gaussian > 0:
						gaussian_params = np.concatenate([np.concatenate([nuisance_scale[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * nuisance_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * nuisance_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * sigma_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * sigma_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											180 / np.pi * rho[i][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for i in range(img_converter.n_gaussian)], -1)
						
						model_params = np.concatenate([crescent_params, gaussian_params], -1)
					else:
						model_params = np.array(crescent_params)

			elif geometric_model == 'floor_nuisance':
				if img_converter.flux_flag:
					r, flux, crescent_flux, nuisance_scale, nuisance_x, nuisance_y, sigma_x, sigma_y, rho = img_converter.compute_features(params)

					crescent_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
										crescent_flux[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)
					
					if args.n_gaussian > 0:
						gaussian_params = np.concatenate([np.concatenate([nuisance_scale[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * nuisance_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * nuisance_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * sigma_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * sigma_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											180 / np.pi * rho[i][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for i in range(img_converter.n_gaussian)], -1)
						
						model_params = np.concatenate([crescent_params, gaussian_params, flux[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)
					else:
						model_params = np.concatenate([crescent_params, flux[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)
							
				else:
					r, crescent_flux, nuisance_scale, nuisance_x, nuisance_y, sigma_x, sigma_y, rho = img_converter.compute_features(params)

					crescent_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
										crescent_flux[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)

					if args.n_gaussian > 0:
						gaussian_params = np.concatenate([np.concatenate([nuisance_scale[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * nuisance_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * nuisance_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * sigma_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											0.5 * fov / eh.RADPERUAS * sigma_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
											180 / np.pi * rho[i][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for i in range(img_converter.n_gaussian)], -1)
						
						model_params = np.concatenate([crescent_params, gaussian_params], -1)
					else:
						model_params = np.array(crescent_params)

			elif geometric_model == 'nuisance_gaussian':
				if img_converter.flux_flag:
					flux, nuisance_scale, nuisance_x, nuisance_y, sigma_x, sigma_y, rho = img_converter.compute_features(params)
					# flux, nuisance_x, nuisance_y, sigma_x, sigma_y, rho = img_converter.compute_features(params)
					gaussian_params = np.concatenate([np.concatenate([nuisance_scale[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
										0.5 * fov / eh.RADPERUAS * nuisance_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
										0.5 * fov / eh.RADPERUAS * nuisance_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
										0.5 * fov / eh.RADPERUAS * sigma_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
										0.5 * fov / eh.RADPERUAS * sigma_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
										180 / np.pi * rho[i][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for i in range(img_converter.n_gaussian)], -1)
					# gaussian_params = np.concatenate([np.concatenate([0.5 * fov / eh.RADPERUAS * nuisance_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
					# 					0.5 * fov / eh.RADPERUAS * nuisance_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
					# 					0.5 * fov / eh.RADPERUAS * sigma_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
					# 					0.5 * fov / eh.RADPERUAS * sigma_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
					# 					180 / np.pi * rho[i][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for i in range(img_converter.n_gaussian)], -1)
					model_params = np.concatenate([gaussian_params, flux[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)
				else:
					nuisance_scale, nuisance_x, nuisance_y, sigma_x, sigma_y, rho = img_converter.compute_features(params)
					# nuisance_x, nuisance_y, sigma_x, sigma_y, rho = img_converter.compute_features(params)
					gaussian_params = np.concatenate([np.concatenate([nuisance_scale[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
										0.5 * fov / eh.RADPERUAS * nuisance_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
										0.5 * fov / eh.RADPERUAS * nuisance_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
										0.5 * fov / eh.RADPERUAS * sigma_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
										0.5 * fov / eh.RADPERUAS * sigma_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
										180 / np.pi * rho[i][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for i in range(img_converter.n_gaussian)], -1)
					# gaussian_params = np.concatenate([np.concatenate([0.5 * fov / eh.RADPERUAS * nuisance_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
					# 					0.5 * fov / eh.RADPERUAS * nuisance_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
					# 					0.5 * fov / eh.RADPERUAS * sigma_x[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
					# 					0.5 * fov / eh.RADPERUAS * sigma_y[i][ind].reshape((-1, 1)).detach().cpu().numpy(),
					# 					180 / np.pi * rho[i][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for i in range(img_converter.n_gaussian)], -1)

					model_params = np.array(gaussian_params)

			elif geometric_model == 'mring':
				if img_converter.flux_flag:
					r, sigma, s_list, eta_list, flux = img_converter.compute_features(params)
					model_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
												2 * 0.5 * fov / eh.RADPERUAS * sigma[ind].reshape((-1, 1)).detach().cpu().numpy()] + \
												[np.concatenate([s_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy(),
												180/np.pi * eta_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for k_order in range(img_converter.n_order)] + \
												[flux[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)
												
				else:
					r, sigma, s_list, eta_list = img_converter.compute_features(params)
					model_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
												2 * 0.5 * fov / eh.RADPERUAS * sigma[ind].reshape((-1, 1)).detach().cpu().numpy()] + \
												[np.concatenate([s_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy(),
												180/np.pi * eta_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for k_order in range(img_converter.n_order)], -1)
			
			elif geometric_model == 'mring_floor':
				if img_converter.flux_flag:
					r, sigma, s_list, eta_list, floor, flux = img_converter.compute_features(params)
					model_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
												2 * 0.5 * fov / eh.RADPERUAS * sigma[ind].reshape((-1, 1)).detach().cpu().numpy()] + \
												[np.concatenate([s_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy(),
												180/np.pi * eta_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for k_order in range(img_converter.n_order)] + \
												[floor[ind].reshape((-1, 1)).detach().cpu().numpy(),
												flux[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)
												
				else:
					r, sigma, s_list, eta_list, floor = img_converter.compute_features(params)
					model_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
												2 * 0.5 * fov / eh.RADPERUAS * sigma[ind].reshape((-1, 1)).detach().cpu().numpy()] + \
												[np.concatenate([s_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy(),
												180/np.pi * eta_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for k_order in range(img_converter.n_order)] + \
												[floor[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)
			elif geometric_model == 'mring_gaussianfloor':
				if img_converter.flux_flag:
					r, sigma, s_list, eta_list, floor, floor_r, flux = img_converter.compute_features(params)
					model_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
												2 * 0.5 * fov / eh.RADPERUAS * sigma[ind].reshape((-1, 1)).detach().cpu().numpy()] + \
												[np.concatenate([s_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy(),
												180/np.pi * eta_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for k_order in range(img_converter.n_order)] + \
												[floor[ind].reshape((-1, 1)).detach().cpu().numpy(),
												2 * 0.5 * fov / eh.RADPERUAS * floor_r[ind].reshape((-1, 1)).detach().cpu().numpy(),
												flux[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)
												
				else:
					r, sigma, s_list, eta_list, floor, floor_r = img_converter.compute_features(params)
					model_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
												2 * 0.5 * fov / eh.RADPERUAS * sigma[ind].reshape((-1, 1)).detach().cpu().numpy()] + \
												[np.concatenate([s_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy(),
												180/np.pi * eta_list[k_order][ind].reshape((-1, 1)).detach().cpu().numpy()], -1) for k_order in range(img_converter.n_order)] + \
												[floor[ind].reshape((-1, 1)).detach().cpu().numpy(),
												2 * 0.5 * fov / eh.RADPERUAS * floor_r[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)
			
			if rejsamp_flag:
				model_params = np.concatenate([model_params, importance_weights[ind].detach().cpu().numpy().reshape((-1, 1))], 1)
			
			if k == 0:
				model_params_all = np.array(model_params)
			else:
				model_params_all = np.concatenate([model_params_all, model_params], 0)
		if rejsamp_flag:
			return model_params_all, img_map.squeeze().detach().cpu().numpy(), img_mean.squeeze().detach().cpu().numpy(), img_std.squeeze().detach().cpu().numpy()
		else:
			return model_params_all

	params_generator.load_state_dict(torch.load(save_path+'/generativemodelbest_'+geometric_model+'_res{}flow{}logdet{}'.format(npix, n_flow, args.logdet)+'_'+args.data_product))
	params_generator.eval()

	model_params_all = Gen_samples(params_generator, rejsamp_flag=False, n_concat=100, alpha_divergence=alpha_divergence)
	np.save(save_path+'/'+'postsamples_norej_alpha{}.npy'.format(alpha_divergence), model_params_all)

	model_params_all, img_map, img_mean, img_std = Gen_samples2(params_generator, rejsamp_flag=True, n_concat=100, alpha_divergence=alpha_divergence)
	np.save(save_path+'/'+'postsamples_rej_alpha{}.npy'.format(alpha_divergence), model_params_all)
	np.save(save_path+'/'+'img_map_alpha{}.npy'.format(alpha_divergence), img_map)
	np.save(save_path+'/'+'img_mean_alpha{}.npy'.format(alpha_divergence), img_mean)
	np.save(save_path+'/'+'img_std_alpha{}.npy'.format(alpha_divergence), img_std)

	from scipy.special import softmax

	postsamples = model_params_all
	importance_weights = softmax(postsamples[:, -1].squeeze())
	sample_indices = np.random.choice(np.arange(postsamples.shape[0]), size=postsamples.shape[0], replace=True, p=importance_weights)
	postsamples_rs = postsamples[sample_indices, 0:-1]
	np.save(save_path+'/'+'postsamples_rs_alpha{}.npy'.format(alpha_divergence), postsamples_rs)

	for i in range(0, int(n_epoch//decay_rate)+1):
		params_generator.load_state_dict(torch.load(save_path+'/generativemodel_decay{}_'.format(i)+geometric_model+'_res{}flow{}logdet{}'.format(npix, n_flow, args.logdet)+'_'+args.data_product))
		params_generator.eval()

		model_params_all = Gen_samples(params_generator, rejsamp_flag=False, n_concat=10, alpha_divergence=alpha_divergence)
		np.save(save_path+'/'+'postsamples_norej_alpha{}_loop{}.npy'.format(alpha_divergence, i), model_params_all)

		model_params_all, img_map, img_mean, img_std = Gen_samples2(params_generator, rejsamp_flag=True, n_concat=10, alpha_divergence=alpha_divergence)
		np.save(save_path+'/'+'postsamples_rej_alpha{}_loop{}.npy'.format(alpha_divergence, i), model_params_all)
		np.save(save_path+'/'+'img_map_alpha{}_loop{}.npy'.format(alpha_divergence, i), img_map)
		np.save(save_path+'/'+'img_mean_alpha{}_loop{}.npy'.format(alpha_divergence, i), img_mean)
		np.save(save_path+'/'+'img_std_alpha{}_loop{}.npy'.format(alpha_divergence, i), img_std)


# ######################################################################################################################
# import corner
# plt.ion()

# save_path = './checkpoint/interferometry_m87_mcfe/lo/3601/crescentfloornuissance/alpha1'

# loss = np.load(save_path + '/loss{}_'.format(args.slice)+geometric_model+'_res{}flow{}logdet{}'.format(npix, n_flow, args.logdet)+'_'+args.data_product+'.npy', allow_pickle=True)

# n_batch = 8196
# for k_decay in range(8):
# 	params_generator.load_state_dict(torch.load(save_path+'/generativemodel{}_decay{}_'.format(args.slice, k_decay)+geometric_model+'_res{}flow{}logdet{}'.format(npix, n_flow, args.logdet)+'_'+args.data_product))
# 	params_generator.eval()

# 	n_group = 10
# 	mring_params_list = []
# 	for k in range(n_group):
# 		z_sample = torch.randn((n_batch, nparams)).to(device=device)

# 		# generate image samples
# 		params_samp, logdet = params_generator.reverse(z_sample)
# 		params = torch.sigmoid(params_samp)

# 		ind = np.arange(n_batch)
# 		# r, sigma, s, eta, floor, nuisance_scale, nuisance_x, nuisance_y, nuisance_covinv1, nuisance_covinv2, nuisance_covinv12 = img_converter.compute_features(params)
# 		r, sigma, s, eta, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x, sigma_y, rho = img_converter.compute_features2(params)
# 		eta = (180/np.pi * eta) % 360
# 		eta[eta>180] = eta[eta>180] - 360




# 		mring_params = np.concatenate([nuisance_scale[0][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 										nuisance_x[0][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 										nuisance_y[0][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 										sigma_x[0][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 										sigma_y[0][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 										rho[0][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 										nuisance_scale[1][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 										nuisance_x[1][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 										nuisance_y[1][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 										sigma_x[1][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 										sigma_y[1][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 										rho[1][ind].reshape((-1, 1)).detach().cpu().numpy()], -1)


# 		mring_params2 = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r.reshape((-1, 1)).detach().cpu().numpy(),
# 									2 * 0.5 * fov / eh.RADPERUAS * sigma.reshape((-1, 1)).detach().cpu().numpy(),
# 									s.reshape((-1, 1)).detach().cpu().numpy(),
# 									eta.reshape((-1, 1)).detach().cpu().numpy(),
# 									floor.reshape((-1, 1)).detach().cpu().numpy(),
# 									nuisance_scale[0].reshape((-1, 1)).detach().cpu().numpy(),
# 									nuisance_x[0].reshape((-1, 1)).detach().cpu().numpy(),
# 									nuisance_y[0].reshape((-1, 1)).detach().cpu().numpy(),
# 									sigma_x[0].reshape((-1, 1)).detach().cpu().numpy(),
# 									sigma_y[0].reshape((-1, 1)).detach().cpu().numpy(),
# 									rho[0].reshape((-1, 1)).detach().cpu().numpy(),
# 									nuisance_scale[1].reshape((-1, 1)).detach().cpu().numpy(),
# 									nuisance_x[1].reshape((-1, 1)).detach().cpu().numpy(),
# 									nuisance_y[1].reshape((-1, 1)).detach().cpu().numpy(),
# 									sigma_x[1].reshape((-1, 1)).detach().cpu().numpy(),
# 									sigma_y[1].reshape((-1, 1)).detach().cpu().numpy(),
# 									rho[1].reshape((-1, 1)).detach().cpu().numpy()], -1)

# 	mring_params_list.append(mring_params)
# 	mring_params = np.concatenate(mring_params_list, 0)
# 	corner.corner(mring_params, bins=50, quantiles=[0.16, 0.84])


# # save_path = './checkpoint/interferometry_m87_mcfe/lo/3601new'
# params_generator.load_state_dict(torch.load(save_path+'/generativemodelbest{}_'.format(args.slice)+geometric_model+'_res{}flow{}logdet{}'.format(npix, n_flow, args.logdet)+'_'+args.data_product))
# params_generator.eval()

# n_batch = 8196
# n_group = 10
# mring_params_list = []
# for k in range(n_group):
# 	z_sample = torch.randn((n_batch, nparams)).to(device=device)

# 	# generate image samples
# 	params_samp, logdet = params_generator.reverse(z_sample)
# 	params = torch.sigmoid(params_samp)

# 	# r, sigma, s, eta, floor, nuisance_scale, nuisance_x, nuisance_y, nuisance_covinv1, nuisance_covinv2, nuisance_covinv12 = img_converter.compute_features(params)
# 	r, sigma, s, eta, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x, sigma_y, rho = img_converter.compute_features2(params)
# 	eta = (180/np.pi * eta) % 360
# 	eta[eta>180] = eta[eta>180] - 360

# 	mring_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r.reshape((-1, 1)).detach().cpu().numpy(),
# 									2 * 0.5 * fov / eh.RADPERUAS * sigma.reshape((-1, 1)).detach().cpu().numpy(),
# 									s.reshape((-1, 1)).detach().cpu().numpy(),
# 									eta.reshape((-1, 1)).detach().cpu().numpy(),
# 									floor.reshape((-1, 1)).detach().cpu().numpy(),
# 									nuisance_scale[0].reshape((-1, 1)).detach().cpu().numpy(),
# 									nuisance_x[0].reshape((-1, 1)).detach().cpu().numpy(),
# 									nuisance_y[0].reshape((-1, 1)).detach().cpu().numpy(),
# 									sigma_x[0].reshape((-1, 1)).detach().cpu().numpy(),
# 									sigma_y[0].reshape((-1, 1)).detach().cpu().numpy(),
# 									rho[0].reshape((-1, 1)).detach().cpu().numpy(),
# 									nuisance_scale[1].reshape((-1, 1)).detach().cpu().numpy(),
# 									nuisance_x[1].reshape((-1, 1)).detach().cpu().numpy(),
# 									nuisance_y[1].reshape((-1, 1)).detach().cpu().numpy(),
# 									sigma_x[1].reshape((-1, 1)).detach().cpu().numpy(),
# 									sigma_y[1].reshape((-1, 1)).detach().cpu().numpy(),
# 									rho[1].reshape((-1, 1)).detach().cpu().numpy()], -1)
# 	mring_params_list.append(mring_params)

# mring_params = np.concatenate(mring_params_list, 0)
# corner.corner(mring_params, bins=50, quantiles=[0.16, 0.84])



# ######################################################################################################################











# r, sigma, s_list, eta_list, nuisance_scale, nuisance_x, nuisance_y, nuisance_covinv1, nuisance_covinv2, nuisance_covinv12 = img_converter.compute_features(params)
# import corner
# bh_params = np.concatenate([nuisance_y[0].detach().cpu().numpy().reshape((-1, 1)),
# 							nuisance_y[1].detach().cpu().numpy().reshape((-1, 1))], -1)

# corner.corner(bh_params)

# import corner

# params_generator.load_state_dict(torch.load(save_path+'/generativemodelbest{}_'.format(args.slice)+geometric_model+'_res{}flow{}logdet{}'.format(npix, n_flow, args.logdet)+'_'+args.data_product))
# params_generator.eval()
# z_sample = torch.randn((n_batch, nparams)).to(device=device)

# # generate image samples
# params_samp, logdet = params_generator.reverse(z_sample)
# params = torch.sigmoid(params_samp)

# img = img_converter.forward(params)

# # compute log determinant
# det_sigmoid = torch.sum(-params_samp-2*torch.nn.Softplus()(-params_samp), -1)
# logdet = logdet + det_sigmoid


# vis, visamp, cphase, logcamp = eht_obs_torch(img)
# if vis_weight == 0:
# 	loss_visamp = Loss_visamp_img(visamp_true, visamp) if visamp_weight>0 else 0
# 	loss_cphase = Loss_cphase_img(cphase_true, cphase) if cphase_weight>0 else 0
# 	loss_camp = Loss_logca_img2(logcamp_true, logcamp) if camp_weight>0 else 0
# 	loss_data = camp_weight * loss_camp + cphase_weight * loss_cphase + visamp_weight * loss_visamp
# else:
# 	loss_vis = Loss_vis_img(vis_true, vis)
# 	loss_data = vis_weight * loss_vis

# portion = 0.90#0.99
# ind = torch.sort(loss_data)[1][0:int(portion*n_batch)]

# ind = np.arange(n_batch)

# r, sigma, s_list, eta_list, flux = img_converter.compute_features(params)
# mring_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
# 							2 * 0.5 * fov / eh.RADPERUAS * sigma[ind].reshape((-1, 1)).detach().cpu().numpy(),
# 							s_list[0][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 							180/np.pi * eta_list[0][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 							s_list[1][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 							180/np.pi * eta_list[1][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 							flux[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)

# ind = np.arange(2048)
# r, sigma, s_list, eta_list = img_converter.compute_features(params)
# eta_list[0] = (180/np.pi * eta_list[0]) % 360
# eta_list[0][eta_list[0]>180] = eta_list[0][eta_list[0]>180] - 360
# eta_list[1] = (180/np.pi * eta_list[1]) % 360
# eta_list[1][eta_list[1]>180] = eta_list[1][eta_list[1]>180] - 360
# mring_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								2 * 0.5 * fov / eh.RADPERUAS * sigma[ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								s_list[0][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								eta_list[0][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								s_list[1][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								eta_list[1][ind].reshape((-1, 1)).detach().cpu().numpy()], -1)

# r, sigma, s, eta = img_converter.compute_features(params)
# eta = (180/np.pi * eta) % 360
# eta[eta>180] = eta[eta>180] - 360
# mring_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								2 * 0.5 * fov / eh.RADPERUAS * sigma[ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								s[ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								eta[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)


# ind = np.arange(2048)
# r, sigma, s_list, eta_list = img_converter.compute_features(params)
# n_order = 4
# for k in range(n_order):
# 	eta_list[k] = (180/np.pi * eta_list[k]) % 360
# 	eta_list[k][eta_list[k]>180] = eta_list[k][eta_list[k]>180] - 360
	
# mring_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								2 * 0.5 * fov / eh.RADPERUAS * sigma[ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								s_list[0][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								eta_list[0][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								s_list[1][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								eta_list[1][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								s_list[2][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								eta_list[2][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								s_list[3][ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								eta_list[3][ind].reshape((-1, 1)).detach().cpu().numpy()], -1)

# corner.corner(mring_params)




# r, sigma, s, eta, flux = img_converter.compute_features(params)
# mring_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								2 * 0.5 * fov / eh.RADPERUAS * sigma[ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								s[ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								180/np.pi * eta[ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								flux[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)


# r, sigma, s, eta, floor, nuisance_scale, nuisance_x, nuisance_y, nuisance_covinv1, nuisance_covinv2, nuisance_covinv12 = img_converter.compute_features(params)
# mring_params = np.concatenate([2 * 0.5 * fov / eh.RADPERUAS * r[ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								2 * 0.5 * fov / eh.RADPERUAS * sigma[ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								s[ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								180/np.pi * eta[ind].reshape((-1, 1)).detach().cpu().numpy(),
# 								floor[ind].reshape((-1, 1)).detach().cpu().numpy()], -1)


# corner.corner(mring_params, labels=['diameter', 'width', 'asymmetry1', 'pa1', 'floor'])


# corner.corner(mring_params, labels=['diameter', 'width', 'asymmetry1', 'pa1', 'flux'])


# corner.corner(mring_params, labels=['diameter', 'width', 'asymmetry1', 'pa1', 'asymmetry2', 'pa2', 'flux'])


# GAIN_OFFSET = {'AA': 0.15, 'AP': 0.15, 'AZ': 0.15, 'LM': 0.6, 'PV': 0.15, 'SM': 0.15, 'JC': 0.15, 'SP': 0.15,
#                'SR': 0.0}
# GAINP = {'AA': 0.05, 'AP': 0.05, 'AZ': 0.05, 'LM': 0.5, 'PV': 0.05, 'SM': 0.05, 'JC': 0.05, 'SP': 0.15,
#      'SR': 0.0}

# gain_offset = GAIN_OFFSET
# gainp = GAINP
# for deg in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
# 	r = 25.0/(0.5*img_converter.fov)
# 	sigma = 8.0/(0.5*img_converter.fov)
# 	s = 0.2
# 	eta = deg/180 * np.pi
# 	ring = torch.exp(- 0.5 * (img_converter.grid_r - r)**2 / (sigma)**2)
# 	S = 1 + s * torch.cos(img_converter.grid_theta - eta)
# 	crescent = S * ring
# 	crescent = crescent / torch.sum(crescent)

# 	simim.imvec = crescent.cpu().numpy().reshape((-1, ))

# 	obs_path_old = '../dataset/sgra_standardized_uvfits_hops/uvfits/3599_besttime/hops/hops_3599_SGRA_LO_netcal_LMTcal_normalized_10s_regionIII.uvfits'
# 	obs_orig = eh.obsdata.load_uvfits(obs_path_old)

# 	obs_orig.add_scans()
# 	obs = obs_orig.copy()
# 	simim.rf = obs.rf
# 	# obs = simim.observe_same(obs, add_th_noise=True, ampcal=False, phasecal=False, stabilize_scan_phase=True, stabilize_scan_amp=True, gain_offset=gain_offset, gainp=gainp)
# 	obs = simim.observe_same(obs, add_th_noise=True, ampcal=True, phasecal=True, stabilize_scan_phase=True, stabilize_scan_amp=True, gain_offset=gain_offset, gainp=gainp)

# 	obs.save_uvfits('../dataset/grmhd_data_besttime/uvfits/crescent{}_thermalnoise.uvfits'.format(deg))
# 	# obs.save_uvfits('../dataset/grmhd_data_besttime/uvfits/crescent{}_allnoise.uvfits'.format(deg))

# 	simim.display()


# r = 25.0/(0.5*img_converter.fov)
# sigma = 5.0/(0.5*img_converter.fov)
# s_list = [0.360555, 0.15620]
# eta_list = [33.69/180 * np.pi, 39.805/180 * np.pi]
# ring = torch.exp(- 0.5 * (img_converter.grid_r - r)**2 / (sigma)**2)
# S = 1 + s_list[0] * torch.cos(img_converter.grid_theta - eta_list[0])
# S = S + s_list[1] * torch.cos(2*img_converter.grid_theta - eta_list[1])
# crescent = S * ring
# crescent = crescent / torch.sum(crescent)

# simim.imvec = crescent.cpu().numpy().reshape((-1, ))

# obs_path_old = '../dataset/sgra_standardized_uvfits_hops/uvfits/3599_besttime/hops/hops_3599_SGRA_LO_netcal_LMTcal_normalized_10s_regionIII.uvfits'
# obs_orig = eh.obsdata.load_uvfits(obs_path_old)

# obs_orig.add_scans()
# obs = obs_orig.copy()
# simim.rf = obs.rf
# # obs = simim.observe_same(obs, add_th_noise=True, ampcal=False, phasecal=False, stabilize_scan_phase=True, stabilize_scan_amp=True, gain_offset=gain_offset, gainp=gainp)
# obs = simim.observe_same(obs, add_th_noise=True, ampcal=True, phasecal=True, stabilize_scan_phase=True, stabilize_scan_amp=True, gain_offset=gain_offset, gainp=gainp)

# obs.save_uvfits('../dataset/interferometry6/mring_test/mring2.uvfits')
# # obs.save_uvfits('../dataset/grmhd_data_besttime/uvfits/crescent{}_allnoise.uvfits'.format(deg))

# simim.display()


# for deg in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
# 	obs_path = '../dataset/grmhd_data_besttime/uvfits/crescent{}_allnoise.uvfits'.format(deg)
# 	obs = eh.obsdata.load_uvfits(obs_path)
# 	flux_const = 1.0#np.median(obs.unpack_bl('AP', 'AA', 'amp')['amp'])
# 	obs_nc = eh.netcal(obs, flux_const, gain_tol=1.0)
# 	obs = preimcal.preim_pipeline(obs_nc,
# 			    is_normalized=False,#True,#
# 			    is_deblurred=False,
# 			    lcarr=None,
# 			    nproc=-1,
# 			    do_LMTcal=True,#False,
# 			    LMTcal_fwhm=60.,
# 			    do_JCMTcal=True,#False,
# 			    tint=60.,
# 			    syserr=0.02,
# 			    do_ref=False,#True,#
# 			    ref_optype="dime",
# 			    ref_scale=1.,
# 			    do_deblurr=False,#True,#
# 			    do_psd_noise=False,
# 			    a=0.03,
# 			    u0=1.,
# 			    b=2.,
# 			    c=0.)
# 	obs.save_uvfits('../dataset/grmhd_data_besttime/uvfits/crescent{}_allnoise_processed_nodeblur.uvfits'.format(deg))


# plt.figure()
# for deg in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
# # for deg in [180, 210, 240, 270]:
# 	obs_path = '../dataset/grmhd_data_besttime/uvfits/crescent{}_allnoise_processed_nodeblur.uvfits'.format(deg)
# 	obs = eh.obsdata.load_uvfits(obs_path)
# 	# uvdist = np.sqrt(obs.data['u']**2 + obs.data['v']**2)
# 	# amp = np.abs(obs.data['vis'])
# 	# err = obs.data['sigma']
# 	# plt.errorbar(uvdist, amp, yerr=err, linestyle='')
# 	obs.plotall('uvdist', 'amp')

# deg = 180
# obs_path = '../dataset/grmhd_data_besttime/uvfits/crescent{}_allnoise_processed_nodeblur.uvfits'.format(deg)
# obs = eh.obsdata.load_uvfits(obs_path)
# uvdist = np.sqrt(obs.data['u']**2 + obs.data['v']**2)
# amp = np.abs(obs.data['vis'])
# err = obs.data['sigma']
# plt.errorbar(uvdist, amp, yerr=err, linestyle='')


# video_file = '../dataset/grmhd_data_besttime/groundtruthmovie/ring_static_scattering.hdf5'
# obs_path_old = '../dataset/sgra_standardized_uvfits_hops/uvfits/3599_besttime/hops/hops_3599_SGRA_LO_netcal_LMTcal_normalized_10s_regionIII.uvfits'
# obs_orig = eh.obsdata.load_uvfits(obs_path_old)
# mov = eh.movie.load_hdf5(video_file)
# obs_orig.add_scans()
# obs = obs_orig.copy()
# mov.rf = obs.rf
# obs = mov.observe_same(obs, add_th_noise=True, ampcal=True, phasecal=True, stabilize_scan_phase=True, stabilize_scan_amp=True, gain_offset=gain_offset, gainp=gainp)
# obs.save_uvfits('../dataset/grmhd_data_besttime/uvfits/uniform_ring_allnoise.uvfits')


# flux_const = np.median(obs.unpack_bl('AP', 'AA', 'amp')['amp'])
# obs_nc = eh.netcal(obs, flux_const, gain_tol=1.0)
# obs = preimcal.preim_pipeline(obs_nc,
# 		    is_normalized=False,#True,#
# 		    is_deblurred=False,
# 		    lcarr=None,
# 		    nproc=-1,
# 		    do_LMTcal=True,#False,
# 		    LMTcal_fwhm=60.,
# 		    do_JCMTcal=True,#False,
# 		    tint=60.,
# 		    syserr=0.02,
# 		    do_ref=True,#False,#
# 		    ref_optype="dime",
# 		    ref_scale=1.,
# 		    do_deblurr=True,#False,#
# 		    do_psd_noise=False,_ge
# 		    a=0.03,
# 		    u0=1.,
# 		    b=2.,
# 		    c=0.)
# obs.save_uvfits('../dataset/grmhd_data_besttime/uvfits/uniform_ring_allnoise_processed.uvfits')


# for deg in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
# 	obs_path = '../dataset/grmhd_data_besttime/uvfits/crescent{}_allnoise.uvfits'.format(deg)
# 	obs = eh.obsdata.load_uvfits(obs_path)

# 	obs = preimcal.preim_pipeline(obs,
# 	    is_normalized=False,
# 	    is_deblurred=False,
# 	    lcarr=None,
# 	    nproc=-1,
# 	    do_LMTcal=True,#False,
# 	    LMTcal_fwhm=60.,
# 	    do_JCMTcal=True,#False,
# 	    tint=60.,
# 	    syserr=0.02,
# 	    do_ref=False,#True,#
# 	    ref_optype="dime",
# 	    ref_scale=1.,
# 	    do_deblurr=False,#True,#
# 	    do_psd_noise=False,
# 	    a=0.03,
# 	    u0=1.,
# 	    b=2.,
# 	    c=0.)

# 	obs.save_uvfits('../dataset/grmhd_data_besttime/uvfits/crescent{}_allnoise_processed_nodeblur.uvfits'.format(deg))

# obs_path = '../dataset/grmhd_data_besttime/uvfits/uniform_ring_allnoise.uvfits'
# obs = eh.obsdata.load_uvfits(obs_path)

# obs = preimcal.preim_pipeline(obs,
#     is_normalized=False,
#     is_deblurred=False,
#     lcarr=None,
#     nproc=-1,
#     do_LMTcal=True,#False,#
#     LMTcal_fwhm=60.,
#     do_JCMTcal=True,#False,#
#     tint=60.,
#     syserr=0.02,
#     do_ref=True,#False,#
#     ref_optype="dime",
#     ref_scale=1.,
#     do_deblurr=True,#False,#
#     do_psd_noise=False,
#     a=0.03,
#     u0=1.,
#     b=2.,
#     c=0.)
# obs.save_uvfits('../dataset/grmhd_data_besttime/uvfits/uniform_ring_allnoise_processed.uvfits')

# import corner
# import pandas as pd

# chain = pd.read_csv('~/BoumanLab/Lab_meeting/20210419/paul_fit/mringF02T10Avg1Gains/Results/Chain/nested_chain_scan-6.csv')
# mring_params = np.concatenate([2 * np.array(chain['rad']).reshape((-1, 1)),
# 							2 * np.array(chain['σ']).reshape((-1, 1)),
# 							np.sqrt(np.array(chain['α1'])**2+np.array(chain['β1'])**2).reshape((-1, 1)),
# 							180/np.pi*np.arctan2(np.array(chain['α1']), np.array(chain['β1'])).reshape((-1, 1)),
# 							np.sqrt(np.array(chain['α2'])**2+np.array(chain['β2'])**2).reshape((-1, 1)),
# 							180/np.pi*np.arctan2(np.array(chain['α2']), np.array(chain['β2'])).reshape((-1, 1)),
# 							np.array(chain['f']).reshape((-1, 1))], -1)

# corner.corner(mring_params, labels=['diameter', 'width', 'asymmetry1', 'pa1', 'asymmetry2', 'pa2', 'flux'])

# ###########################################################################################
# ind = -1
# # save_path = './checkpoint/interferometry_besttime_grmhd_singleframe_crescent4'
# save_path = './checkpoint/interferometry_sgra_snapshot_mring_spline_width'

# loss_all = np.load(save_path+'/loss{}_'.format(ind)+args.model_form+'_res{}flow{}logdet{}_closure_fluxcentermemtsv.npy'.format(npix, n_flow, args.logdet), allow_pickle=True)
# params_generator.load_state_dict(torch.load(save_path+'/generativemodelbest{}_'.format(ind)+args.model_form+'_res{}flow{}logdet{}_closure_fluxcentermemtsv'.format(npix, n_flow, args.logdet)))
# params_generator.eval()
# n_batch = 1024
# z_sample = torch.randn((n_batch, nparams)).to(device=device)
# # generate image samples
# params_samp, logdet = params_generator.reverse(z_sample)
# params = torch.sigmoid(params_samp)
# img = img_converter.forward(params)

# # r, sigma, s, eta, nuisance_scale, nuisance_x, nuisance_y, nuisance_covinv1, nuisance_covinv2, nuisance_covinv12 = img_converter.compute_features(params)
# r, sigma, s, eta = img_converter.compute_features(params)
# r = r.detach().cpu().numpy().squeeze() * 0.5 * fov / eh.RADPERUAS
# sigma = sigma.detach().cpu().numpy().squeeze() * 0.5 * fov / eh.RADPERUAS
# eta = (eta.detach().cpu().numpy().squeeze() * 180 / np.pi) % 360


# import seaborn as sns
# from scipy.stats.kde import gaussian_kde
# from scipy.stats import norm


# data = np.array(sigma)
# fill = 'y'
# n_points = 360#180
# plt.figure()
# # xx = np.linspace(np.min(data), np.max(data), n_points)
# xx = np.linspace(np.sort(data)[int(0.005*n_batch)], np.sort(data)[-int(0.005*n_batch)], n_points)
# pdf = gaussian_kde(data)
# curve = pdf(xx)
# plt.fill_between(xx, np.zeros(n_points), curve, color=fill)
# plt.plot(xx, curve, c='k')
# # plt.plot(xx[np.argmax(curve)]*np.ones(50), np.linspace(0, np.max(curve), 50), 'r--')
# plt.plot(np.median(data)*np.ones(50), np.linspace(0, np.max(curve), 50), 'r--')
# # plt.title('PDF of ring diameter (map: {:0.2f}, std: {:0.2e})'.format(xx[np.argmax(curve)], np.std(data)))
# plt.title('PDF of width (median: {:0.2f}, std: {:0.2e})'.format(np.median(data), np.std(data)))
# plt.xlabel('micro arcsecond')


# data = np.array(r*2)
# fill = 'y'
# n_points = 360#180
# plt.figure()
# # xx = np.linspace(np.min(data), np.max(data), n_points)
# xx = np.linspace(np.sort(data)[int(0.005*n_batch)], np.sort(data)[-int(0.005*n_batch)], n_points)
# pdf = gaussian_kde(data)
# curve = pdf(xx)
# plt.fill_between(xx, np.zeros(n_points), curve, color=fill)
# plt.plot(xx, curve, c='k')
# # plt.plot(xx[np.argmax(curve)]*np.ones(50), np.linspace(0, np.max(curve), 50), 'r--')
# plt.plot(np.median(data)*np.ones(50), np.linspace(0, np.max(curve), 50), 'r--')


# plt.title('PDF of diameter (median: {:0.2f}, std: {:0.2e})'.format(np.median(data), np.std(data)))
# plt.xlabel('micro arcsecond')



# data = np.array(eta)
# fill = 'y'
# n_points = 360#180
# plt.figure()
# # xx = np.linspace(np.min(data), np.max(data), n_points)
# xx = np.linspace(np.sort(data)[int(0.005*n_batch)], np.sort(data)[-int(0.005*n_batch)], n_points)
# pdf = gaussian_kde(data)
# curve = pdf(xx)
# plt.fill_between(xx, np.zeros(n_points), curve, color=fill)
# plt.plot(xx, curve, c='k')
# # plt.plot(xx[np.argmax(curve)]*np.ones(50), np.linspace(0, np.max(curve), 50), 'r--')
# plt.plot(np.median(data)*np.ones(50), np.linspace(0, np.max(curve), 50), 'r--')
# # plt.title('PDF of ring diameter (map: {:0.2f}, std: {:0.2e})'.format(xx[np.argmax(curve)], np.std(data)))
# plt.title('PDF of position angle (median: {:0.2f}, std: {:0.2e})'.format(np.median(data), np.std(data)))
# plt.xlabel('degree')

# ##########################################################################################
# save_path = './checkpoint/interferometry_besttime_grmhd_singleframe_crescent4'
# save_path = './checkpoint/interferometry_sgra_snapshot_mring_spline_width'
# save_path = './checkpoint/3599_besttime_crescent_HI'
# save_path = './checkpoint/3599_besttime_crescent_LO'
# save_path = './checkpoint/interferometry_sgra_snapshot_mring_spline_width_LO'
# save_path = './checkpoint/interferometry_sgra_snapshot_mring_spline_width_HI'
# save_path = './checkpoint/interferometry_sgra_recon_nonoise_snapshot_crescent'
# save_path = './checkpoint/interferometry_sgra_recon_allnoise_snapshot_crescent'
# save_path = './checkpoint/interferometry_sgra_recon_thermalnoise_snapshot_crescent'

# save_path = './checkpoint/interferometry_sgra_recon_allnoise_snapshot_crescent'

# save_path = './checkpoint/interferometry_besttime_grmhd_crescent_orig3'
# save_path = './checkpoint/interferometry_sgra_3599_snapshot_crescent_old'#'./checkpoint/interferometry_staticring_snapshot_crescent'#


# n_batch = 2048#512#10240#256#128#64#32#8#256#

# from scipy.stats.kde import gaussian_kde
# from scipy.stats import norm


# def ridgeline(data, time_list, amp=1, fill=True, modec='r', labels=None, n_points=150):
#     """
#     Creates a standard ridgeline plot.

#     data, list of lists.
#     overlap, overlap between distributions. 1 max overlap, 0 no overlap.
#     fill, matplotlib color to fill the distributions.
#     n_points, number of points to evaluate each distribution function.
#     labels, values to place on the y axis to describe the distributions.
#     """
#     # if overlap > 1 or overlap < 0:
#     #     raise ValueError('overlap must be in [0 1]')
#     xx = np.linspace(np.min(np.concatenate(data)),
#                      np.max(np.concatenate(data)), n_points)
#     curves = []
#     ys = []
#     for i, d in enumerate(data):
#         pdf = gaussian_kde(d)
#         y = time_list[i]	
#         ys.append(y)
#         curve = pdf(xx)
#         curve *= amp
#         if fill:
#             plt.fill_between(xx, np.ones(n_points)*y, 
#                              curve+y, zorder=len(data)-i+1, color=fill, alpha=0.5)
#         plt.plot(xx, curve+y, c='k', zorder=len(data)-i+1)

#         # peak = xx[np.argmax(curve)]
#         # plt.plot(peak, np.max(curve)+y, modec+'o', zorder=len(data), markersize=10)

#     if labels:
#         plt.yticks(ys, labels)


# save_path = './checkpoint/interferometry_crescent_m87snapshot/trial95'#'./checkpoint/interferometry_crescent_sanitycheck/trial330'#'./checkpoint/interferometry_sgra_3599_snapshot_crescent'#'./checkpoint/interferometry_uniformring_sanitycheck/trial_allnoise'#'./checkpoint/interferometry_uniformring_sanitycheck/trial_thermalnoise'
# data_product = 'cphase_logcamp'#'vis'#

# d_samples = []
# sigma_samples = []
# eta_samples = []
# s_samples = []
# beta1real_samples = []
# beta1imag_samples = []
# beta2real_samples = []
# beta2imag_samples = []

# image_samples = []
# loss_cphase_list = []
# loss_logca_list = []
# loss_vis_list = []
# logdet_list = []

# eta2_samples = []

# r_median_list = []
# r_upper_list = []
# r_lower_list = []

# sigma_median_list = []
# sigma_upper_list = []
# sigma_lower_list = []

# diameter_mode_list = []
# sigma_mode_list = []
# eta_mode_list = []


# n_time = 18#61#54#58#62#20#18#30#5#14#
# # ind_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
# # 			21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
# # 			41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
# for ind in range(n_time):
# # for ind in ind_list:
# 	if ind != 1:
# 		# params_generator.load_state_dict(torch.load(save_path+'/generativemodelbest{}_'.format(ind)+args.model_form+'_res{}flow{}logdet{}_closure_fluxcentermemtsv'.format(npix, n_flow, args.logdet), map_location=device))
# 		params_generator.load_state_dict(torch.load(save_path+'/generativemodelbest{}_'.format(ind)+geometric_model+'_res{}flow{}logdet{}'.format(npix, n_flow, args.logdet)+'_'+data_product, map_location=device))
# 		# params_generator.load_state_dict(torch.load(save_path+'/generativemodel{}_'.format(ind)+geometric_model+'_res{}flow{}logdet{}'.format(npix, n_flow, args.logdet)+'_'+data_product, map_location=device))
# 		# loss_all = np.load(save_path+'/loss{}_'.format(ind)+args.model_form+'_res{}flow{}logdet{}_closure_fluxcentermemtsv.npy'.format(npix, n_flow, args.logdet), allow_pickle=True)
		
# 		loss_all = np.load(save_path+'/loss{}_'.format(ind)+geometric_model+'_res{}flow{}logdet{}'.format(npix, n_flow, args.logdet)+'_'+args.data_product+'.npy', allow_pickle=True)
# 		if data_product == 'cphase_logcamp':
# 			loss_cphase_list.append(loss_all.item()['cphase'][np.argmin(loss_all.item()['total'])])
# 			loss_logca_list.append(loss_all.item()['logca'][np.argmin(loss_all.item()['total'])])	
# 		elif data_product == 'vis':
# 			loss_vis_list.append(loss_all.item()['vis'][np.argmin(loss_all.item()['total'])])

# 		logdet_list.append(loss_all.item()['logdet'][np.argmin(loss_all.item()['total'])])

# 		params_generator.eval()
# 		z_sample = torch.randn(n_batch, nparams).to(device=device)
# 		# generate image samples
# 		params_samp, logdet = params_generator.reverse(z_sample)
# 		params = torch.sigmoid(params_samp)

# 		# compute statistics
# 		# r, sigma, s, eta = img_converter.compute_features(params)
# 		if args.geometric_model == 'simple_crescent':
# 			if data_product == 'cphase_logcamp':
# 				r, sigma, s, eta = img_converter.compute_features(params)
# 			elif data_product == 'vis':
# 				r, sigma, s, eta, flux = img_converter.compute_features(params)
# 			r = r.detach().cpu().numpy().squeeze() * 0.5 * fov / eh.RADPERUAS
# 			sigma = sigma.detach().cpu().numpy().squeeze() * 0.5 * fov / eh.RADPERUAS
# 			eta = (eta.detach().cpu().numpy().squeeze() * 180 / np.pi - 90) % 360
# 			s = s.detach().cpu().numpy().squeeze()
# 		if args.geometric_model == 'mring_spline_width':
# 			r, sigma, s_list, eta_list, yw = img_converter.compute_features(params)
# 			r = r.detach().cpu().numpy().squeeze() * 0.5 * fov / eh.RADPERUAS
# 			sigma = sigma.detach().cpu().numpy().squeeze() * 0.5 * fov / eh.RADPERUAS
# 			eta = (eta_list[0].detach().cpu().numpy().squeeze() * 180 / np.pi - 90) % 360
# 		if args.geometric_model == 'mring':
# 			r, sigma, s_list, eta_list = img_converter.compute_features(params)
# 			r = r.detach().cpu().numpy().squeeze() * 0.5 * fov / eh.RADPERUAS
# 			sigma = sigma.detach().cpu().numpy().squeeze() * 0.5 * fov / eh.RADPERUAS
# 			eta = (eta_list[0].detach().cpu().numpy().squeeze() * 180 / np.pi - 90) % 360

# 		# d = r#eta
# 		# xx = np.linspace(np.min(d), np.max(d), 360)
# 		# pdf = gaussian_kde(d)
# 		# d_curve = pdf(xx)
# 		# ind = np.argmin(np.abs(d-xx[np.argmax(d_curve)]))

# 		d = r#eta
# 		xx = np.linspace(np.min(d), np.max(d), 360)
# 		pdf = gaussian_kde(d)
# 		d_curve = pdf(xx)
# 		ind_r = np.argsort(np.abs(d-xx[np.argmax(d_curve)]))
# 		diameter_mode_list.append(xx[np.argmax(d_curve)] * 2)

# 		d = sigma#eta
# 		xx = np.linspace(np.min(d), np.max(d), 360)
# 		pdf = gaussian_kde(d)
# 		d_curve = pdf(xx)
# 		ind_sigma = np.argsort(np.abs(d-xx[np.argmax(d_curve)]))
# 		sigma_mode_list.append(xx[np.argmax(d_curve)])

# 		d = eta
# 		xx = np.linspace(np.min(d), np.max(d), 360)
# 		pdf = gaussian_kde(d)
# 		d_curve = pdf(xx)
# 		ind_eta = np.argsort(np.abs(d-xx[np.argmax(d_curve)]))
# 		eta_mode_list.append(xx[np.argmax(d_curve)])

# 		dic = {}
# 		for k in range(len(ind_r)):
# 			dic[ind_r[k]] = k
# 		for k in range(len(ind_eta)):
# 			dic[ind_eta[k]] = np.max([k, dic[ind_eta[k]]])

# 		for k in range(len(ind_sigma)):
# 			dic[ind_sigma[k]] = np.max([k, dic[ind_sigma[k]]])

# 		ind = 0
# 		ind_max = n_batch
# 		for k in range(len(ind_eta)):
# 			if dic[k] < ind_max:
# 				ind_max = dic[k]
# 				ind = k

# 		# ind = np.argmax(logdet.detach().cpu().numpy())


# 		# img = img_converter.forward(params)
# 		# image_samples.append(img[ind].squeeze())

# 		# img = img_converter.forward(params)
# 		# image_samples.append(img[np.argsort(sigma)[n_batch//2]].squeeze())

# 		# d_samples.append(2*r)
# 		# sigma_samples.append(sigma)
# 		# eta_samples.append(eta)
# 		d_samples.append(np.sort(2*r)[int(0.005*n_batch):-int(0.005*n_batch)])
# 		sigma_samples.append(np.sort(2*sigma)[int(0.005*n_batch):-int(0.005*n_batch)])

# 		eta_samples.append(np.sort(eta)[int(0.005*n_batch):-int(0.005*n_batch)])
# 		s_samples.append(np.sort(s)[int(0.005*n_batch):-int(0.005*n_batch)])
# 		# eta_samples.append((eta_value_list[0]+180)%360)
# 		# eta2_samples.append((eta_value_clist[1]+180)%360)

# # data = [norm.rvs(loc=i, scale=2, size=50) for i in range(8)]
# # ridgeline(eta_samples[0:20], obs.tstart + t_slice * index_time[1:21], overlap=.85, fill='y')

# # t_slice = 1/60#1/30#1/120#(obs.tstop - obs.tstart)/100
# # index_time = np.unique((obs.data['time'] - obs.tstart)//t_slice)

# obs = eh.obsdata.load_uvfits(obs_path)

# tint = 60
# obs.add_scans()
# obs = obs.avg_coherent(tint, scan_avg=False)
# obs_list = obs.split_obs()

# t_list = []
# for k in range(n_time):
# 	t_list.append(obs_list[k].data['time'][0])

# baseline_list = []
# cphase_list = []
# logcamp_list = []
# for k in range(n_time):
# 	if k !=1:
# 		obs_split = obs_list[k].copy()
# 		obs_split.add_cphase(count='min')
# 		obs_split.add_logcamp(count='min')
# 		baseline_list.append(len(obs_split.data['vis']))
# 		cphase_list.append(len(obs_split.cphase['cphase']))
# 		logcamp_list.append(len(obs_split.logcamp['camp']))

# fill_color = 'y'#'b'#'g'#
# # plt.figure(), ridgeline(eta_samples[0:n_time], t_list, amp=0.5, fill=fill_color, n_points=360)
# plt.figure(), ridgeline(eta_samples[0:n_time], t_list, amp=3.0, fill=fill_color, n_points=360)
# plt.xlabel('position angle (degree)')
# plt.ylabel('time (hour)')

# # plt.xlim([20, 70])

# # plt.figure(), ridgeline(d_samples[0:n_time], t_list, amp=0.1, fill=fill_color, n_points=360)
# plt.figure(), ridgeline(d_samples[0:n_time], t_list, amp=0.6, fill=fill_color, n_points=360)
# plt.xlabel('diameter (muas)')
# plt.ylabel('time (hour)')
# # plt.xlim([30, 70])


# # plt.figure(), ridgeline(sigma_samples[0:n_time], t_list, amp=0.05, fill=fill_color, n_points=360)
# plt.figure(), ridgeline(sigma_samples[0:n_time], t_list, amp=0.3, fill=fill_color, n_points=360)
# plt.xlabel('width (muas)')
# plt.ylabel('time (hour)')
# # plt.xlim([3, 17])


# plt.figure(), ridgeline(s_samples[0:n_time], t_list, amp=0.01, fill=fill_color, n_points=360)
# plt.xlabel('aymmetry')
# plt.ylabel('time (hour)')


# plt.figure(), plt.plot(loss_cphase_list, t_list, 'ro')
# plt.xlabel('cphase chi2')
# plt.ylabel('time (hour)')

# plt.figure(), plt.plot(loss_logca_list, t_list, 'ro')
# plt.xlabel('logcamp chi2')
# plt.ylabel('time (hour)')

# plt.figure(), plt.plot(logdet_list, t_list, 'ro')
# plt.xlabel('logdet')
# plt.ylabel('time (hour)')


# # plt.figure(), plt.plot(loss_vis_list, t_list, 'ro')
# # plt.xlabel('vis chi2')
# # plt.ylabel('time (hour)')


# # plt.figure(), plt.plot(baseline_list, t_list, 'ro')
# # plt.xlabel('number of baselines')
# # plt.ylabel('time (hour)')





# eta_mean_list = []
# eta_err_list = []

# d_mean_list = []
# d_err_list = []

# sigma_mean_list = []
# sigma_err_list = []

# s_mean_list = []
# s_err_list = []

# for ind in range(n_time):
# 	eta_mean_list.append(np.mean(eta_samples[ind]))
# 	eta_err_list.append(np.std(eta_samples[ind]))

# 	d_mean_list.append(np.mean(d_samples[ind]))
# 	d_err_list.append(np.std(d_samples[ind]))

# 	sigma_mean_list.append(np.mean(sigma_samples[ind]))
# 	sigma_err_list.append(np.std(sigma_samples[ind]))

# 	s_mean_list.append(np.mean(s_samples[ind]))
# 	s_err_list.append(np.std(s_samples[ind]))

# plt.figure(), plt.errorbar(t_list, eta_mean_list, yerr=eta_err_list,
# 							linestyle='', ecolor='red', elinewidth=None, 
# 							marker='^', mfc='red', mec='red', ms=2, mew=4,zorder=200)
# plt.ylabel('position angle (degree)')
# plt.xlabel('time (hour)')


# plt.figure(), plt.errorbar(t_list, d_mean_list, yerr=d_err_list,
# 							linestyle='', ecolor='red', elinewidth=None, 
# 							marker='^', mfc='red', mec='red', ms=2, mew=4,zorder=200)
# plt.ylabel('diameter (muas)')
# plt.xlabel('time (hour)')



# plt.figure(), plt.errorbar(t_list, sigma_mean_list, yerr=sigma_err_list,
# 							linestyle='', ecolor='red', elinewidth=None, 
# 							marker='^', mfc='red', mec='red', ms=2, mew=4,zorder=200)
# plt.ylabel('width (muas)')
# plt.xlabel('time (hour)')


# plt.figure(), plt.errorbar(t_list, s_mean_list, yerr=s_err_list,
# 							linestyle='', ecolor='red', elinewidth=None, 
# 							marker='^', mfc='red', mec='red', ms=2, mew=4,zorder=200)
# plt.ylabel('aymmetry')
# plt.xlabel('time (hour)')





# fill_color = 'g'#'b'#'g'#
# modec = 'b'
# # plt.figure(), ridgeline(eta_samples[0:n_time], t_list, amp=0.5, fill=fill_color, n_points=360)
# plt.figure(1), ridgeline(eta_samples[0:n_time], t_list, modec=modec, amp=3.0, fill=fill_color, n_points=360)
# plt.xlabel('position angle (degree)')
# plt.ylabel('time (hour)')

# # plt.xlim([20, 70])

# # plt.figure(), ridgeline(d_samples[0:n_time], t_list, amp=0.1, fill=fill_color, n_points=360)
# plt.figure(2), ridgeline(d_samples[0:n_time], t_list, modec=modec, amp=0.6, fill=fill_color, n_points=360)
# plt.xlabel('diameter (muas)')
# plt.ylabel('time (hour)')
# # plt.xlim([30, 70])


# # plt.figure(), ridgeline(sigma_samples[0:n_time], t_list, amp=0.05, fill=fill_color, n_points=360)
# plt.figure(3), ridgeline(sigma_samples[0:n_time], t_list, modec=modec, amp=0.3, fill=fill_color, n_points=360)
# plt.xlabel('width (muas)')
# plt.ylabel('time (hour)')
# # plt.xlim([3, 17])


# plt.figure(4), plt.plot(loss_cphase_list, t_list, 'bo')
# plt.xlabel('cphase chi2')
# plt.ylabel('time (hour)')

# plt.figure(5), plt.plot(loss_logca_list, t_list, 'bo')
# plt.xlabel('logcamp chi2')
# plt.ylabel('time (hour)')

# plt.figure(6), plt.plot(logdet_list, t_list, 'bo')
# plt.xlabel('logdet')
# plt.ylabel('time (hour)')


# ind = 12
# crescent_params = np.concatenate([np.expand_dims(eta_samples[ind], -1),
# 								np.expand_dims(d_samples[ind], -1),
# 								np.expand_dims(sigma_samples[ind], -1),
# 								np.expand_dims(s_samples[ind], -1)], -1)


# corner.corner(crescent_params, labels=['angle', 'diameter', 'width', 'amp'])

# import pandas as pd
# # csv_path = '/home/groot/BoumanLab/Lab_meeting/20210327/DAR2/SgrALOF05T10NoGainsM1Dblr/Results/Chain'
# # csv_path = '/home/groot/BoumanLab/Lab_meeting/20210327/DAR2/SgrALOF02T10GainsM1Dblr/'
# # csv_path = '/home/groot/BoumanLab/Lab_meeting/20210327/DAR2_GRMHD/mad_a0.94_i30_F02T10GainsM1Dblr/'
# csv_path = '/home/groot/BoumanLab/Lab_meeting/20210327/DAR2_GRMHD/mad_a+0.94_i30_variable_pa1_NoGainsM1Dblr/'



# d_samples = []
# sigma_samples = []
# eta_samples = []

# # k = 0
# n_time = 61#60#
# for k in range(n_time):
# 	params = pd.read_csv(csv_path+'Results/Chain/nested_chain_scan-{}.csv'.format(k+1))
# 	r = np.array(params['rad'])
# 	sigma = np.array(params['σ'])
# 	eta = (np.arctan2(np.array(params['α1']), np.array(params['β1'])) * 180 / np.pi - 90) % 360
# 	# eta = (np.arctan2(np.array(params['β1']), np.array(params['α1'])) * 180 / np.pi) % 360

# 	d_samples.append(np.sort(2*r)[int(0.005*n_batch):-int(0.005*n_batch)])
# 	sigma_samples.append(np.sort(sigma)[int(0.005*n_batch):-int(0.005*n_batch)])

# 	eta_samples.append(np.sort(eta)[int(0.005*n_batch):-int(0.005*n_batch)])

# stats = pd.read_csv(csv_path+'Results/Stats/summary_stats.csv')

# loss_cphase_list = list(np.array(stats['mcpchi2']))
# loss_amp_list = list(np.array(stats['mampchi2']))
# # # # plt.figure(), plt.plot(cphase_list, t_list, 'ro')
# # # # plt.xlabel('number of cphases')
# # # # plt.ylabel('time (hour)')

# # # # plt.figure(), plt.plot(logcamp_list, t_list, 'ro')
# # # # plt.xlabel('number of logcamp')
# # # # plt.ylabel('time (hour)')


# # for k in range(n_time):
# # 	simim2 = simim.copy()
# # 	simim.imvec = image_samples[k].detach().cpu().numpy().reshape((-1, ))
# # 	simim.display()
# # 	plt.title('time: {}'.format(t_list[k]))
# # 	# plt.savefig('/home/groot/BoumanLab/Lab_meeting/20210326/average_coherent60_crescent_singleframe_snapshot/snapshot{}.jpg'.format(k))
# # 	plt.savefig('/home/groot/BoumanLab/Lab_meeting/20210327/SgrAnonoise_LO_logcampcphase_crescent/snapshot{}.jpg'.format(k))



# # import imageio
# # # filenames = ['/home/groot/BoumanLab/Lab_meeting/20210326/average_coherent60_crescent_singleframe_snapshot/snapshot{}.jpg'.format(k) for k in range(n_time)]
# # filenames = ['/home/groot/BoumanLab/Lab_meeting/20210327/SgrAnonoise_LO_logcampcphase_crescent/snapshot{}.jpg'.format(k) for k in range(n_time)]

# # images = []
# # for filename in filenames:
# #     images.append(imageio.imread(filename))
# # # imageio.mimsave('/home/groot/BoumanLab/Lab_meeting/20210326/average_coherent60_crescent_singleframe_snapshot/movie.gif', images)
# # imageio.mimsave('/home/groot/BoumanLab/Lab_meeting/20210327/SgrAnonoise_LO_logcampcphase_crescent/movie.gif', images, duration=0.5)

# # # # plt.figure(), plt.plot(t_list, loss_cphase_list)
# # # # plt.ylabel('cphase chi2')
# # # # plt.xlabel('time (hour)')

# # # # plt.figure(), plt.plot(t_list, loss_logca_list)
# # # # plt.ylabel('logcamp chi2')
# # # # plt.xlabel('time (hour)')

# # # # plt.figure(), plt.plot(t_list, logdet_list)
# # # # plt.ylabel('logdet')
# # # # plt.xlabel('time (hour)')


# # plt.figure(11), plt.plot(t_list, eta_mode_list, 'b-x')
# # plt.ylabel('position angle (degree)')
# # plt.xlabel('time (hour)')

# # plt.figure(12), plt.plot(t_list, diameter_mode_list, 'b-x')
# # plt.ylabel('diameter (muas)')
# # plt.xlabel('time (hour)')

# # plt.figure(13), plt.plot(t_list, sigma_mode_list, 'b-x')
# # plt.ylabel('width (muas)')
# # plt.xlabel('time (hour)')



# # plt.figure(11), plt.plot(t_list, eta_mode_list, 'r-o')
# # plt.ylabel('position angle (degree)')
# # plt.xlabel('time (hour)')

# # plt.figure(12), plt.plot(t_list, diameter_mode_list, 'r-o')
# # plt.ylabel('diameter (muas)')
# # plt.xlabel('time (hour)')

# # plt.figure(13), plt.plot(t_list, sigma_mode_list, 'r-o')
# # plt.ylabel('width (muas)')
# # plt.xlabel('time (hour)')


# # ind_select = np.where(np.logical_and(np.logical_and(diameter_mode_HI>=40, diameter_mode_LO>=40), np.logical_and(diameter_mode_HI<=70, diameter_mode_LO<=70)))
# # ind_select = np.where(np.logical_and(np.logical_and(diameter_mode_HI>=40, diameter_mode_LO>=40), np.logical_and(diameter_mode_HI<=70, diameter_mode_LO<=70)))
# # ind_select = np.where(np.logical_and(diameter_mode_LO>=40, diameter_mode_LO<=70))


# plt.figure(11), plt.plot(np.array(t_list)[ind_select[0]], np.array(eta_mode_list)[ind_select[0]], 'b-x')
# plt.ylabel('position angle (degree)')
# plt.xlabel('time (hour)')

# plt.figure(12), plt.plot(np.array(t_list)[ind_select[0]], np.array(diameter_mode_list)[ind_select[0]], 'b-x')
# plt.ylabel('diameter (muas)')
# plt.xlabel('time (hour)')

# plt.figure(13), plt.plot(np.array(t_list)[ind_select[0]], np.array(sigma_mode_list)[ind_select[0]], 'b-x')
# plt.ylabel('width (muas)')
# plt.xlabel('time (hour)')


# plt.figure(11), plt.plot(np.array(t_list)[ind_select[0]], np.array(eta_mode_list)[ind_select[0]], 'r-o')
# plt.ylabel('position angle (degree)')
# plt.xlabel('time (hour)')

# plt.figure(12), plt.plot(np.array(t_list)[ind_select[0]], np.array(diameter_mode_list)[ind_select[0]], 'r-o')
# plt.ylabel('diameter (muas)')
# plt.xlabel('time (hour)')

# plt.figure(13), plt.plot(np.array(t_list)[ind_select[0]], np.array(sigma_mode_list)[ind_select[0]], 'r-o')
# plt.ylabel('width (muas)')
# plt.xlabel('time (hour)')







# fill_color = 'g'#'b'#'g'#
# modec = 'b'
# # plt.figure(), ridgeline(eta_samples[0:n_time], t_list, amp=0.5, fill=fill_color, n_points=360)
# plt.figure(31), ridgeline([eta_samples[k] for k in list(ind_select[0])], [t_list[k] for k in list(ind_select[0])], modec=modec, amp=1.5, fill=fill_color, n_points=360)
# plt.xlabel('position angle (degree)')
# plt.ylabel('time (hour)')

# # plt.xlim([20, 70])

# # plt.figure(), ridgeline(d_samples[0:n_time], t_list, amp=0.1, fill=fill_color, n_points=360)
# plt.figure(32), ridgeline([d_samples[k] for k in list(ind_select[0])], [t_list[k] for k in list(ind_select[0])], modec=modec, amp=0.6, fill=fill_color, n_points=360)
# plt.xlabel('diameter (muas)')
# plt.ylabel('time (hour)')
# # plt.xlim([30, 70])


# # plt.figure(), ridgeline(sigma_samples[0:n_time], t_list, amp=0.05, fill=fill_color, n_points=360)
# plt.figure(33), ridgeline([sigma_samples[k] for k in list(ind_select[0])], [t_list[k] for k in list(ind_select[0])], modec=modec, amp=0.3, fill=fill_color, n_points=360)
# plt.xlabel('width (muas)')
# plt.ylabel('time (hour)')


# fill_color = 'y'#'b'#'g'#
# modec = 'r'
# # plt.figure(), ridgeline(eta_samples[0:n_time], t_list, amp=0.5, fill=fill_color, n_points=360)
# plt.figure(31), ridgeline([eta_samples[k] for k in list(ind_select[0])], [t_list[k] for k in list(ind_select[0])], modec=modec, amp=1.5, fill=fill_color, n_points=360)
# plt.xlabel('position angle (degree)')
# plt.ylabel('time (hour)')

# # plt.xlim([20, 70])

# # plt.figure(), ridgeline(d_samples[0:n_time], t_list, amp=0.1, fill=fill_color, n_points=360)
# plt.figure(32), ridgeline([d_samples[k] for k in list(ind_select[0])], [t_list[k] for k in list(ind_select[0])], modec=modec, amp=0.6, fill=fill_color, n_points=360)
# plt.xlabel('diameter (muas)')
# plt.ylabel('time (hour)')
# # plt.xlim([30, 70])


# # plt.figure(), ridgeline(sigma_samples[0:n_time], t_list, amp=0.05, fill=fill_color, n_points=360)
# plt.figure(33), ridgeline([sigma_samples[k] for k in list(ind_select[0])], [t_list[k] for k in list(ind_select[0])], modec=modec, amp=0.3, fill=fill_color, n_points=360)
# plt.xlabel('width (muas)')
# plt.ylabel('time (hour)')


# # PA_imaging = [120.44873691, 116.89059474, 117.26279424, 112.21742794,
# #         115.96121594, 118.10534754, 121.76725818, 120.38167922,
# #         119.30075298, 115.97183328, 116.34454218, 116.42269609,
# #         117.34008198, 118.4975996 , 118.94903574, 115.24464982,
# #         115.94937432, 112.69440235, 112.09638083,  74.56638459,
# #          56.39083515,  40.74477016,  21.36192897,   6.50924088,
# #          -1.42816478, -16.54262857, -31.63068098, -22.60053162,
# #         -15.79790209, -19.96257764, -26.8969503 , -37.89702762,
# #         -39.64500902, -47.70897203, -48.61292652, -53.40778558,
# #         -54.93870361, -52.79678922, -45.6824989 , -36.93266489,
# #         -33.60799549, -36.11289628, -31.62801463, -34.29765601,
# #         -34.39735719, -35.32886539, -32.82709657, -32.89492632,
# #         -35.62194149, -30.98306609, -26.8059143 , -27.86000538,
# #         -32.31388861, -31.98384639, -30.9014533 , -33.35624359,
# #         -38.19128389, -46.75856224, -40.17282911, -42.45028647,
# #         -46.37997463]

# # std_imaging = [57.21628871, 57.38401769, 56.66683051, 59.20066369, 57.38279173,
# #         56.91012812, 55.78725856, 55.85662564, 55.68895806, 58.00352744,
# #         57.43312967, 58.22247565, 57.9438194 , 57.10989265, 58.39031867,
# #         58.46574481, 58.18461756, 58.321894  , 56.13603728, 57.97026583,
# #         57.03145205, 52.32992337, 51.63820166, 51.0830175 , 53.11625219,
# #         53.2805527 , 52.38257797, 55.26137197, 57.04247862, 62.32120453,
# #         68.37227659, 71.78881195, 69.92766893, 67.57510609, 62.77645238,
# #         58.58139012, 53.80719432, 56.10041684, 60.04412062, 61.96287357,
# #         59.72867939, 58.88807578, 56.59894269, 55.25671464, 55.74842471,
# #         55.66980248, 54.4342137 , 54.43935484, 55.68859471, 56.27915515,
# #         57.47622096, 57.78976087, 58.60099174, 58.42791241, 57.3612622 ,
# #         55.36501752, 51.72757904, 47.5832291 , 48.69393516, 46.63293819,
# #         44.43302565]

# # time_imaging = [12.45833333, 12.625     , 12.64166667, 12.65833333, 12.675     , 12.69166667,
# #      12.70833333, 12.725     , 12.74166667, 12.75833333, 12.775     , 12.84166667,
# #      12.85833333, 12.875     , 12.89166667, 12.90833333, 12.925     , 12.94166667,
# #      12.95833333, 12.975     , 12.99166667, 13.15833333, 13.175     , 13.19166667,
# #      13.20833333, 13.225     , 13.24166667, 13.25833333, 13.275     , 13.29166667,
# #      13.44166667, 13.45833333, 13.475     , 13.49166667, 13.50833333, 13.525     ,
# #      13.54166667, 13.55833333, 13.725     , 13.74166667, 13.75833333, 13.775     ,
# #      13.79166667, 13.80833333, 13.825     , 13.84166667, 13.85833333, 13.875     ,
# #      13.94166667, 13.95833333, 13.975     , 13.99166667, 14.00833333, 14.025     ,
# #      14.04166667, 14.05833333, 14.125     , 14.14166667, 14.15833333, 14.175     ,
# #      14.19166667]


# # # plt.figure(21), plt.plot((np.array(PA_imaging)[ind_select[0]])%360, np.array(time_imaging)[ind_select[0]], markersize=10, c='c', marker='o')
# # plt.figure(21), plt.plot((np.array(PA_imaging)[ind_select[0]])%360, np.array(time_imaging)[ind_select[0]], 'ro', markersize=8, zorder=100)


# # plt.figure(21), plt.fill_betweenx(np.array(time_imaging), np.maximum((np.array(PA_imaging))%360 - np.array(std_imaging), 0), 
# #                              np.minimum((np.array(PA_imaging))%360 + np.array(std_imaging), 360), zorder=101, color='red', alpha=0.3)

# # plt.figure(21), plt.fill_betweenx(np.array(time_imaging), np.zeros_like(np.array(time_imaging)), 
# #                              1.0 * (((np.array(PA_imaging))%360 + np.array(std_imaging))>360) * ((np.array(PA_imaging))%360 + np.array(std_imaging))%360, zorder=101, color='red', alpha=0.3)



# # plt.figure(21), plt.fill_betweenx(np.array(time_imaging), 1.0*(((np.array(PA_imaging))%360 - np.array(std_imaging))<0) * ((np.array(PA_imaging))%360 - np.array(std_imaging))%360 + 360.0*(((np.array(PA_imaging))%360 - np.array(std_imaging))>0), 
# #                              360 * np.ones_like(np.array(time_imaging)), zorder=101, color='red', alpha=0.3)

# # # # plt.figure(), ridgeline(beta1real_samples[0:n_time], obs.tstart + t_slice * index_time[0:n_time], amp=0.01, fill=fill_color, n_points=360)
# # # # plt.xlabel('diameter (muas)')
# # # # plt.ylabel('time (hour)')


# # # # plt.figure(), ridgeline(beta1imag_samples[0:n_time], obs.tstart + t_slice * index_time[0:n_time], amp=0.01, fill=fill_color, n_points=360)
# # # # plt.xlabel('diameter (muas)')
# # # # plt.ylabel('time (hour)')


# # # # r_lowerror_list = [r_median_list[k]-r_lower_list[k] for k in range(len(r_upper_list))]
# # # # r_uperror_list = [r_upper_list[k]-r_median_list[k] for k in range(len(r_upper_list))]
# # # # plt.figure(), plt.errorbar(obs.tstart + t_slice * index_time[0:n_time], r_median_list, yerr=[r_lowerror_list, r_uperror_list],
# # # #             fmt='o', ecolor='g', capthick=2)
# # # # plt.xlabel('time (utc)')
# # # # plt.ylabel('diameter (uas)')
# # # # plt.ylim([40, 60])


# # # # sigma_lowerror_list = [sigma_median_list[k]-sigma_lower_list[k] for k in range(len(sigma_upper_list))]
# # # # sigma_uperror_list = [sigma_upper_list[k]-sigma_median_list[k] for k in range(len(sigma_upper_list))]
# # # # plt.figure(), plt.errorbar(obs.tstart + t_slice * index_time[0:n_time], sigma_median_list, yerr=[sigma_lowerror_list, sigma_uperror_list],
# # # #             fmt='o', ecolor='g', capthick=2)
# # # # plt.xlabel('time (utc)')
# # # # plt.ylabel('width (uas)')
# # # # plt.ylim([2.5, 17.5])


# # # # beta1real_median_list = [np.median(eta_samples[k]) for k in range(n_time)]
# # # # beta1real_upper_list = [np.sort(eta_samples[k])[int(0.16*n_batch)] for k in range(n_time)]
# # # # beta1real_lower_list = [np.sort(eta_samples[k])[-int(0.16*n_batch)] for k in range(n_time)]

# # # # beta1real_lowerror_list = [beta1real_median_list[k]-beta1real_lower_list[k] for k in range(len(beta1real_upper_list))]
# # # # beta1real_uperror_list = [beta1real_upper_list[k]-beta1real_median_list[k] for k in range(len(beta1real_upper_list))]
# # # # plt.figure(), plt.errorbar(obs.tstart + t_slice * index_time[0:n_time], beta1real_median_list, yerr=[beta1real_lowerror_list, beta1real_uperror_list],
# # # #             fmt='o', ecolor='g', capthick=2)
# # # # plt.xlabel('time (utc)')
# # # # plt.ylabel('position angle1 (degree)')
# # # # plt.ylim([20, 90])

# # # # beta1real_median_list = [np.median(180/np.pi*np.arctan2(beta1real_samples[k], beta1imag_samples[k])) for k in range(n_time)]
# # # # beta1real_upper_list = [np.sort(180/np.pi*np.arctan2(beta1real_samples[k], beta1imag_samples[k]))[int(0.16*n_batch)] for k in range(n_time)]
# # # # beta1real_lower_list = [np.sort(180/np.pi*np.arctan2(beta1real_samples[k], beta1imag_samples[k]))[-int(0.16*n_batch)] for k in range(n_time)]

# # # # beta1real_lowerror_list = [beta1real_median_list[k]-beta1real_lower_list[k] for k in range(len(beta1real_upper_list))]
# # # # beta1real_uperror_list = [beta1real_upper_list[k]-beta1real_median_list[k] for k in range(len(beta1real_upper_list))]
# # # # plt.figure(), plt.errorbar(obs.tstart + t_slice * index_time[0:n_time], beta1real_median_list, yerr=[beta1real_lowerror_list, beta1real_uperror_list],
# # # #             fmt='o', ecolor='g', capthick=2)
# # # # plt.xlabel('time (utc)')
# # # # plt.ylabel('position angle1 (degree)')
# # # # plt.ylim([0, 60])


# # # # beta1real_median_list = [np.median(180/np.pi*np.arctan2(beta2imag_samples[k], beta2real_samples[k])) for k in range(n_time)]
# # # # beta1real_upper_list = [np.sort(180/np.pi*np.arctan2(beta2imag_samples[k], beta2real_samples[k]))[int(0.16*n_batch)] for k in range(n_time)]
# # # # beta1real_lower_list = [np.sort(180/np.pi*np.arctan2(beta2imag_samples[k], beta2real_samples[k]))[-int(0.16*n_batch)] for k in range(n_time)]

# # # # beta1real_lowerror_list = [beta1real_median_list[k]-beta1real_lower_list[k] for k in range(len(beta1real_upper_list))]
# # # # beta1real_uperror_list = [beta1real_upper_list[k]-beta1real_median_list[k] for k in range(len(beta1real_upper_list))]
# # # # plt.figure(), plt.errorbar(obs.tstart + t_slice * index_time[0:n_time], beta1real_median_list, yerr=[beta1real_lowerror_list, beta1real_uperror_list],
# # # #             fmt='o', ecolor='g', capthick=2)
# # # # plt.xlabel('time (utc)')
# # # # plt.ylabel('width (uas)')
# # # # plt.ylim([2.5, 17.5])


# # # # plt.figure(),  plt.plot(obs.tstart + t_slice * index_time[0:n_time], loss_cphase_list, 'r^')
# # # # plt.plot(obs.tstart + t_slice * index_time[0:n_time], loss_logca_list, 'gx')
# # # # plt.legend(['cphase chisq', 'logcamp chisq'])
# # # # plt.xlabel('time (utc)')