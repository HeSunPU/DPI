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
# from interferometry_loss import *
# from interferometry_obs import *
from interferometry_helpers import *

import argparse

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description="Deep Probabilistic Imaging Trainer for Interferometry")
parser.add_argument("--cuda", default=0, type=int, help="cuda index in use")
parser.add_argument("--obspath", default='../dataset/interferometry1/obs.uvfits', type=str, help="EHT observation file path")
parser.add_argument("--impath", default='../dataset/interferometry1/gt.fits', type=str, help="groud-truth EHT image file path")
parser.add_argument("--save_path", default='./save_path', type=str, help="file save path")
parser.add_argument("--npix", default=32, type=int, help="image shape (pixels)")
parser.add_argument("--fov", default=160, type=float, help="field of view of the image in micro-arcsecond")
parser.add_argument("--prior_fwhm", default=50, type=float, help="fwhm of image prior in micro-arcsecond")
parser.add_argument("--model_form", default='realnvp', type=str, help="form of the deep generative model")
parser.add_argument("--n_flow", default=16, type=int, help="number of flows in RealNVP or Glow")
parser.add_argument("--n_block", default=4, type=int, help="number of blocks in Glow")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--n_epoch", default=30000, type=int, help="number of epochs for training RealNVP")
parser.add_argument("--logdet", default=1.0, type=float, help="logdet weight")
parser.add_argument("--l1", default=1.0, type=float, help="l1 prior weight")
parser.add_argument("--tsv", default=100.0, type=float, help="tsv prior weight")
parser.add_argument("--flux", default=1000.0, type=float, help="flux prior weight")
parser.add_argument("--center", default=1.0, type=float, help="centering prior weight")
parser.add_argument("--mem", default=1024.0, type=float, help="mem prior weight")
parser.add_argument("--clip", default=0.1, type=float, help="gradient clip for neural network training")

parser.add_argument("--ttype", default='nfft', type=str, help="fourier transform computation method")


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
	obs_path = args.obspath
	gt_path = args.impath
	npix = args.npix

	if torch.cuda.is_available():
		device = torch.device('cuda:{}'.format(args.cuda))

	obs = eh.obsdata.load_uvfits(obs_path)

	# define the prior image for MEM regularizer
	flux_const = np.median(obs.unpack_bl('APEX', 'ALMA', 'amp')['amp'])
	prior_fwhm = args.prior_fwhm*eh.RADPERUAS#60*eh.RADPERUAS#
	fov = args.fov*eh.RADPERUAS
	zbl = flux_const#2.0#0.8#
	prior = eh.image.make_square(obs, npix, fov).add_gauss(zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))
	prior = prior.add_gauss(zbl*1e-6, (prior_fwhm, prior_fwhm, 0, prior_fwhm, prior_fwhm))

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

	
	if args.model_form == 'realnvp':
		n_flow = args.n_flow
		affine = True
		img_generator = realnvpfc_model.RealNVP(npix*npix, n_flow, affine=affine).to(device)
		# img_generator.load_state_dict(torch.load(save_path+'/generativemodel_'+args.model_form+'_res{}flow{}logdet{}_closure_fluxcentermemtsv'.format(npix, n_flow, args.logdet)))
	elif args.model_form == 'glow':
		n_channel = 1
		n_flow = args.n_flow
		n_block = args.n_block
		affine = True
		no_lu = False#True
		z_shapes = glow_model.calc_z_shapes(n_channel, npix, n_flow, n_block)
		img_generator = glow_model.Glow(n_channel, n_flow, n_block, affine=affine, conv_lu=not no_lu).to(device)

	logscale_factor = Img_logscale(scale=flux_const/(0.8*npix*npix)).to(device)
	# logscale_factor.load_state_dict(torch.load(save_path+'/generativescale_'+args.model_form+'_res{}flow{}logdet{}_closure_fluxcentermemtsv'.format(npix, n_flow, args.logdet)))

	# define the losses and weights for very long baseline interferometric imaging
	Loss_center_img = Loss_center(device, center=npix/2-0.5, dim=npix)
	Loss_flux_img = Loss_flux(flux_const)
	Loss_vis_img = Loss_vis_diff(obs.data['sigma'], device)
	Loss_cphase_img = Loss_angle_diff(obs.cphase['sigmacp'], device)
	Loss_logamp_img = Loss_logamp_diff(obs.data['sigma'], device)
	# Loss_logca_img = Loss_logca_diff(obs.camp['sigmaca'], device)
	Loss_logca_img2 = Loss_logca_diff2(obs.logcamp['sigmaca'], device)


	camp_weight = 1.0
	cphase_weight = len(obs.cphase['cphase'])/len(obs.camp['camp'])#1.0#10.0#
	visamp_weight = 0.0#1e-3#1.0#1e-5#
	imgl1_weight = args.l1 * npix*npix/flux_const#npix*npix/flux_const#1.0
	imgtsv_weight = args.tsv * npix*npix#100*npix*npix
	imgflux_weight = args.flux#1024#0.0#npix*npix#1.0#10
	imgcenter_weight = args.center*1e5/(npix*npix)#1e5/(npix*npix)#100#0.0#1.0#npix*npix#10#
	imgcrossentropy_weight = args.mem#1024#10*npix*npix
	logdet_weight = 2.0 * args.logdet / len(obs.camp['camp'])#args.logdet / (npix*npix)#1.0 / (npix*npix) #


	vis_true = torch.Tensor(np.concatenate([np.expand_dims(obs.data['vis'].real, 0), 
							np.expand_dims(obs.data['vis'].imag, 0)], 0)).to(device=device)
	visamp_true = torch.Tensor(np.array(np.abs(obs.data['vis']))).to(device=device)
	cphase_true = torch.Tensor(np.array(obs.cphase['cphase'])).to(device=device)
	camp_true = torch.Tensor(np.array(obs.camp['camp'])).to(device=device)
	logcamp_true = torch.Tensor(np.array(obs.logcamp['camp'])).to(device=device)
	prior_im = torch.Tensor(np.array(prior.imvec.reshape((npix, npix)))).to(device=device)

	# optimize both scale and image generator
	lr = args.lr
	optimizer = optim.Adam(list(img_generator.parameters())+list(logscale_factor.parameters()), lr = lr)
	# optimizer = optim.Adam(img_generator.parameters(), lr = lr)


	n_epoch = args.n_epoch#30000#10000#100000#50000#100#
	loss_list = []
	loss_prior_list = []
	loss_cphase_list = []
	loss_logca_list = []
	loss_visamp_list = []
	# loss_vis_list = []
	logdet_list = []
	loss_center_list = []
	loss_tsv_list = []
	loss_flux_list = []
	loss_cross_entropy_list = []
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
		logscale_factor_value = logscale_factor.forward()
		scale_factor = torch.exp(logscale_factor_value)
		img = torch.nn.Softplus()(img_samp) * scale_factor
		det_softplus = torch.sum(img_samp - torch.nn.Softplus()(img_samp), (1, 2))
		# logdet = logdet + det_softplus
		det_scale = logscale_factor_value * npix * npix
		logdet = logdet + det_softplus + det_scale

		vis, visamp, cphase, logcamp = eht_obs_torch(img)
		loss_center = Loss_center_img(img) if imgcenter_weight>0 else 0
		loss_l1 = Loss_l1(img) if imgl1_weight>0 else 0
		loss_tsv = Loss_TSV(img) if imgtsv_weight>0 else 0
		loss_cross_entropy = Loss_cross_entropy(prior_im, img) if imgcrossentropy_weight>0 else 0
		loss_flux = Loss_flux_img(img) if imgflux_weight>0 else 0
		# loss_vis = Loss_vis_img(vis_true, vis)
		loss_visamp = Loss_logamp_img(visamp_true, visamp) if visamp_weight>0 else 0
		loss_cphase = Loss_cphase_img(cphase_true, cphase) if cphase_weight>0 else 0
		loss_camp = Loss_logca_img2(logcamp_true, logcamp) if camp_weight>0 else 0

		loss_data = camp_weight * loss_camp + cphase_weight * loss_cphase + visamp_weight * loss_visamp
		# loss_prior = imgflux_weight * loss_flux + imgl1_weight * loss_l1 + imgcenter_weight * loss_center
		loss_prior = imgcrossentropy_weight*loss_cross_entropy + imgflux_weight * loss_flux + \
					imgtsv_weight * loss_tsv + imgcenter_weight * loss_center + imgl1_weight * loss_l1

		loss = torch.mean(loss_data) + torch.mean(loss_prior) - logdet_weight*torch.mean(logdet)

		optimizer.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(list(img_generator.parameters())+ list(logscale_factor.parameters()), args.clip)
		# nn.utils.clip_grad_norm_(img_generator.parameters(), args.clip)
		optimizer.step()

		loss_list.append(loss.detach().cpu().numpy())
		loss_cphase_list.append(torch.mean(loss_cphase).detach().cpu().numpy() if cphase_weight>0 else 0)
		loss_logca_list.append(torch.mean(loss_camp).detach().cpu().numpy() if camp_weight>0 else 0)
		loss_visamp_list.append(torch.mean(loss_visamp).detach().cpu().numpy() if visamp_weight>0 else 0)
		loss_prior_list.append(torch.mean(loss_prior).detach().cpu().numpy())
		logdet_list.append(-torch.mean(logdet).detach().cpu().numpy() / (npix*npix))
		loss_flux_list.append(torch.mean(loss_flux).detach().cpu().numpy() if imgflux_weight>0 else 0)
		loss_tsv_list.append(torch.mean(loss_tsv).detach().cpu().numpy() if imgtsv_weight>0 else 0)
		loss_center_list.append(torch.mean(loss_center).detach().cpu().numpy() if imgcenter_weight>0 else 0)
		loss_cross_entropy_list.append(torch.mean(loss_cross_entropy).detach().cpu().numpy() if imgcrossentropy_weight>0 else 0)
		loss_l1_list.append(torch.mean(loss_l1).detach().cpu().numpy() if imgl1_weight>0 else 0)


		print(f"epoch: {k:}, loss: {loss_list[-1]:.5f}, loss cphase: {loss_cphase_list[-1]:.5f}, loss camp: {loss_logca_list[-1]:.5f}, loss visamp: {loss_visamp_list[-1]:.5f}, logdet: {logdet_list[-1]:.5f}")
		print(f"loss cross entropy: {loss_cross_entropy_list[-1]:.5f}, loss tsv: {loss_tsv_list[-1]:.5f}, loss l1: {loss_l1_list[-1]:.5f}, loss center: {loss_center_list[-1]:.5f}, loss flux: {loss_flux_list[-1]:.5f}")


		# print(f"epoch: {((n_epoch//n_blur)*k_blur+k):}, loss: {loss_list[-1]:.5f}, loss cphase: {loss_cphase_list[-1]:.5f}, loss camp: {loss_logca_list[-1]:.5f}, loss visamp: {loss_visamp_list[-1]:.5f}, loss prior: {loss_prior_list[-1]:.5f}")

	torch.save(img_generator.state_dict(), save_path+'/generativemodel_'+args.model_form+'_res{}flow{}logdet{}_closure_fluxcentermemtsv'.format(npix, n_flow, args.logdet))
	torch.save(logscale_factor.state_dict(), save_path+'/generativescale_'+args.model_form+'_res{}flow{}logdet{}_closure_fluxcentermemtsv'.format(npix, n_flow, args.logdet))
	np.save(save_path+'/generativeimage_'+args.model_form+'_res{}flow{}logdet{}_closure_fluxcentermemtsv.npy'.format(npix, n_flow, args.logdet), img.cpu().detach().numpy().squeeze())

	loss_all = {}
	loss_all['total'] = np.array(loss_list)
	loss_all['cphase'] = np.array(loss_cphase_list)
	loss_all['logca'] = np.array(loss_logca_list)
	loss_all['visamp'] = np.array(loss_visamp_list)
	loss_all['logdet'] = np.array(logdet_list)
	loss_all['flux'] = np.array(loss_flux_list)
	loss_all['tsv'] = np.array(loss_tsv_list)
	loss_all['center'] = np.array(loss_center_list)
	loss_all['mem'] = np.array(loss_cross_entropy_list)
	loss_all['l1'] = np.array(loss_l1_list)
	np.save(save_path+'/loss_'+args.model_form+'_res{}flow{}logdet{}_closure_fluxcentermemtsv.npy'.format(npix, n_flow, args.logdet), loss_all)

