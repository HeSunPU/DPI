import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional

torch.set_default_dtype(torch.float32)


class SimpleCrescent_Param2Img(nn.Module):
	def __init__(self, npix, fov=160, r_range=[10.0, 40.0], width_range=[1.0, 40.0],
				flux_range=[0.8, 1.2], flux_flag=False):
		super().__init__()
		self.fov = fov
		self.r_range = r_range
		self.width_range = width_range
		self.flux_range = flux_range
		self.flux_flag = flux_flag
		if self.flux_flag:
			self.nparams = 5
		else:
			self.nparams = 4
		self.eps = 1e-4
		self.gap = 1.0 / npix
		xs = torch.arange(-1+self.gap, 1, 2*self.gap)
		# grid_x, grid_y = torch.meshgrid(xs, xs)
		grid_y, grid_x = torch.meshgrid(-xs, xs)
		self.grid_x = grid_x
		self.grid_y = grid_y
		self.grid_r = torch.sqrt(grid_x**2 + grid_y**2)
		self.grid_theta = torch.atan2(grid_y, grid_x)

	def compute_features(self, params):
		r = self.r_range[0]/(0.5*self.fov) + params[:, 0].unsqueeze(-1).unsqueeze(-1) * (self.r_range[1]-self.r_range[0])/(0.5*self.fov)
		sigma = self.width_range[0]/(0.5*self.fov) + params[:, 1].unsqueeze(-1).unsqueeze(-1) * (self.width_range[1]-self.width_range[0])/(0.5*self.fov)
		s = params[:, 2].unsqueeze(-1).unsqueeze(-1)
		eta = 181/180 * np.pi * (2.0 * params[:, 3].unsqueeze(-1).unsqueeze(-1) - 1.0)
		if self.flux_flag:
			flux = self.flux_range[0] + (self.flux_range[1]-self.flux_range[0]) * params[:, 4].unsqueeze(-1).unsqueeze(-1)
			return r, sigma, s, eta, flux
		else:
			return r, sigma, s, eta

	def forward(self, params):
		if self.flux_flag:
			r, sigma, s, eta, flux = self.compute_features(params)
		else:
			r, sigma, s, eta = self.compute_features(params)
		ring = torch.exp(- 0.5 * (self.grid_r - r)**2 / (sigma)**2)
		S = 1 + s * torch.cos(self.grid_theta - eta)
		crescent = S * ring

		if self.flux_flag:
			crescent = flux * crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		else:
			crescent = crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		
		return crescent

	def to(self, device):
		self.grid_x = self.grid_x.to(device)
		self.grid_y = self.grid_y.to(device)
		self.grid_r = self.grid_r.to(device)
		self.grid_theta = self.grid_theta.to(device)
		return self


class SimpleCrescent2_Param2Img(nn.Module):
	def __init__(self, npix, fov=160, r_range=[10.0, 40.0], width_range=[1.0, 40.0],
				flux_range=[0.8, 1.2], flux_flag=False):
		super().__init__()
		self.fov = fov
		self.r_range = r_range
		self.width_range = width_range
		self.flux_range = flux_range
		self.flux_flag = flux_flag
		if self.flux_flag:
			self.nparams = 5
		else:
			self.nparams = 4
		self.eps = 1e-4
		self.gap = 1.0 / npix
		xs = torch.arange(-1+self.gap, 1, 2*self.gap)
		# grid_x, grid_y = torch.meshgrid(xs, xs)
		grid_y, grid_x = torch.meshgrid(-xs, xs)
		self.grid_x = grid_x
		self.grid_y = grid_y
		self.grid_r = torch.sqrt(grid_x**2 + grid_y**2)
		self.grid_theta = torch.atan2(grid_y, grid_x)

		xs_kern = 2*self.gap * torch.arange(-5, 6, 1)
		grid_x_kern, grid_y_kern = torch.meshgrid(xs_kern, xs_kern)
		self.grid_r_kern = torch.sqrt(grid_x_kern**2 + grid_y_kern**2)


	def compute_features(self, params):
		r = self.r_range[0]/(0.5*self.fov) + params[:, 0].unsqueeze(-1).unsqueeze(-1) * (self.r_range[1]-self.r_range[0])/(0.5*self.fov)
		sigma = self.width_range[0]/(0.5*self.fov) + params[:, 1].unsqueeze(-1).unsqueeze(-1) * (self.width_range[1]-self.width_range[0])/(0.5*self.fov)
		s = params[:, 2].unsqueeze(-1).unsqueeze(-1)
		eta = 181/180 * np.pi * (2.0 * params[:, 3].unsqueeze(-1).unsqueeze(-1) - 1.0)
		if self.flux_flag:
			flux = self.flux_range[0] + (self.flux_range[1]-self.flux_range[0]) * params[:, 4].unsqueeze(-1).unsqueeze(-1)
			return r, sigma, s, eta, flux
		else:
			return r, sigma, s, eta

	def forward(self, params):
		if self.flux_flag:
			r, sigma, s, eta, flux = self.compute_features(params)
		else:
			r, sigma, s, eta = self.compute_features(params)
		ring = 1 / np.sqrt(2 * self.gap) * torch.exp(- 0.5 * (self.grid_r - r)**2 / (2 * self.gap+self.eps)**2)
		
		S = 1 + s * torch.cos(self.grid_theta - eta)
		crescent = S * ring

		kernel = torch.exp(- 0.5 * (self.grid_r_kern / sigma)**2)
		kernel = kernel / torch.sum(kernel, (1, 2)).unsqueeze(-1).unsqueeze(-1)
		crescent_reshape = crescent.unsqueeze(0)
		crescent = torch.nn.functional.conv2d(crescent_reshape, kernel.unsqueeze(1), groups=kernel.shape[0], padding=5).permute(1, 0, 2, 3)
		crescent = crescent.squeeze(1)
		if self.flux_flag:
			crescent = flux * crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		else:
			crescent = crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		
		return crescent

	def to(self, device):
		self.grid_x = self.grid_x.to(device)
		self.grid_y = self.grid_y.to(device)
		self.grid_r = self.grid_r.to(device)
		self.grid_theta = self.grid_theta.to(device)
		self.grid_r_kern = self.grid_r_kern.to(device)
		return self

class SimpleCrescentNuisance_Param2Img(nn.Module):
	def __init__(self, npix, n_gaussian=1, fov=160, r_range=[10.0, 40.0], 
				width_range=[1.0, 40.0], flux_range=[0.8, 1.2], flux_flag=False):
		super().__init__()
		self.n_gaussian = n_gaussian
		self.fov = fov
		self.r_range = r_range
		self.width_range = width_range
		self.flux_range = flux_range
		self.flux_flag = flux_flag
		if self.flux_flag:
			self.nparams = 5 + 6 * n_gaussian
		else:
			self.nparams = 4 + 6 * n_gaussian
		self.eps = 1e-4
		self.gap = 1.0 / npix
		xs = torch.arange(-1+self.gap, 1, 2*self.gap)
		# grid_x, grid_y = torch.meshgrid(xs, xs)
		grid_y, grid_x = torch.meshgrid(-xs, xs)
		self.grid_x = grid_x
		self.grid_y = grid_y
		self.grid_r = torch.sqrt(grid_x**2 + grid_y**2)
		self.grid_theta = torch.atan2(grid_y, grid_x)


	def compute_features(self, params):
		r = self.r_range[0]/(0.5*self.fov) + params[:, 0].unsqueeze(-1).unsqueeze(-1) * (self.r_range[1]-self.r_range[0])/(0.5*self.fov)
		sigma = self.width_range[0]/(0.5*self.fov) + params[:, 1].unsqueeze(-1).unsqueeze(-1) * (self.width_range[1]-self.width_range[0])/(0.5*self.fov)
		s = params[:, 2].unsqueeze(-1).unsqueeze(-1)
		eta = 181/180 * np.pi * (2.0 * params[:, 3].unsqueeze(-1).unsqueeze(-1) - 1.0)
		
		nuisance_scale = []
		nuisance_covinv1 = []
		nuisance_covinv2 = []
		nuisance_covinv12 = []
		nuisance_x = []
		nuisance_y = []
		for k in range(self.n_gaussian):
			x_shift = (2 * params[:, 4+k*6] - 1).unsqueeze(-1).unsqueeze(-1)
			y_shift = (2 * params[:, 5+k*6] - 1).unsqueeze(-1).unsqueeze(-1)
			scale = params[:, 6+k*6].unsqueeze(-1).unsqueeze(-1)
			# sigma_x = params[:, 7+k*6].unsqueeze(-1).unsqueeze(-1) * 0.5
			# sigma_y = params[:, 8+k*6].unsqueeze(-1).unsqueeze(-1) * 0.5
			sigma_x = params[:, 7+k*6].unsqueeze(-1).unsqueeze(-1)
			sigma_y = params[:, 8+k*6].unsqueeze(-1).unsqueeze(-1)
			rho = 2 * 0.99 * (params[:, 9+k*6].unsqueeze(-1).unsqueeze(-1) - 0.5)
			sigma_xy = rho * sigma_x * sigma_y
			factor = self.eps**2 + self.eps * sigma_x**2 + self.eps * sigma_y**2 + (1-rho**2) * sigma_x**2 * sigma_y**2
			covinv1 = sigma_y**2 / factor
			covinv2 = sigma_x**2 / factor
			covinv12 = sigma_xy / factor

			nuisance_x.append(x_shift)
			nuisance_y.append(y_shift)
			nuisance_scale.append(scale)
			nuisance_covinv1.append(covinv1)
			nuisance_covinv2.append(covinv2)
			nuisance_covinv12.append(covinv12)
			
		if self.flux_flag:
			flux = self.flux_range[0] + (self.flux_range[1]-self.flux_range[0]) * params[:, 4+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
			return r, sigma, s, eta, flux, nuisance_scale, nuisance_x, nuisance_y, nuisance_covinv1, nuisance_covinv2, nuisance_covinv12
		else:
			return r, sigma, s, eta, nuisance_scale, nuisance_x, nuisance_y, nuisance_covinv1, nuisance_covinv2, nuisance_covinv12


	def forward(self, params):
		if self.flux_flag:
			r, sigma, s, eta, flux, nuisance_scale, nuisance_x, nuisance_y, nuisance_covinv1, nuisance_covinv2, nuisance_covinv12 = self.compute_features(params)
		else:
			r, sigma, s, eta, nuisance_scale, nuisance_x, nuisance_y, nuisance_covinv1, nuisance_covinv2, nuisance_covinv12 = self.compute_features(params)
		
		ring = torch.exp(- 0.5 * (self.grid_r - r)**2 / (sigma)**2)
		S = 1 + s * torch.cos(self.grid_theta - eta)
		crescent = S * ring

		for k in range(self.n_gaussian):
			x_c = self.grid_x - nuisance_x[k]
			y_c = self.grid_y - nuisance_y[k]
			delta = 0.5 * (nuisance_covinv1[k] * x_c**2 + nuisance_covinv2[k] * y_c**2 - 2*nuisance_covinv12[k]*x_c*y_c)
			nuisance_now = torch.exp(-delta) * nuisance_scale[k]
			crescent += nuisance_now

		if self.flux_flag:
			crescent = flux * crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		else:
			crescent = crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		
		return crescent

	def to(self, device):
		self.grid_x = self.grid_x.to(device)
		self.grid_y = self.grid_y.to(device)
		self.grid_r = self.grid_r.to(device)
		self.grid_theta = self.grid_theta.to(device)
		return self


# Almost similar to paper VI version
class SimpleCrescentNuisanceFloor_Param2Img(nn.Module):
	def __init__(self, npix, n_gaussian=1, fov=160, r_range=[10.0, 40.0], asym_range=[1e-3, 0.99],
				width_range=[1.0, 40.0], floor_range=[0.0, 1.0], flux_range=[0.8, 1.2], crescent_flux_range=[1e-3, 2.0], 
				shift_range=[-200.0, 200.0], sigma_range=[1.0, 100.0], gaussian_scale_range=[1e-3, 2.0], flux_flag=False):
		super().__init__()
		self.n_gaussian = n_gaussian
		self.fov = fov
		self.r_range = r_range
		self.asym_range = asym_range
		self.width_range = width_range
		self.floor_range = floor_range
		self.flux_range = flux_range
		self.crescent_flux_range = crescent_flux_range
		self.shift_range = shift_range
		self.sigma_range = sigma_range
		self.gaussian_scale_range = gaussian_scale_range
		self.flux_flag = flux_flag
		if self.flux_flag:
			self.nparams = 5 + 6 * n_gaussian + 2
		else:
			self.nparams = 4 + 6 * n_gaussian + 2

		# if self.flux_flag:
		# 	self.nparams = 6 + 6 * n_gaussian
		# else:
		# 	self.nparams = 5 + 6 * n_gaussian

		self.eps = 1e-4
		self.gap = 1.0 / npix
		xs = torch.arange(-1+self.gap, 1, 2*self.gap)
		# grid_x, grid_y = torch.meshgrid(xs, xs)
		grid_y, grid_x = torch.meshgrid(-xs, xs)
		self.grid_x = grid_x
		self.grid_y = grid_y
		self.grid_r = torch.sqrt(grid_x**2 + grid_y**2)
		self.grid_theta = torch.atan2(grid_y, grid_x)
		self.npix = npix

	def compute_features(self, params):
		r = self.r_range[0]/(0.5*self.fov) + params[:, 0].unsqueeze(-1).unsqueeze(-1) * (self.r_range[1]-self.r_range[0])/(0.5*self.fov)
		sigma = self.width_range[0]/(0.5*self.fov) + params[:, 1].unsqueeze(-1).unsqueeze(-1) * (self.width_range[1]-self.width_range[0])/(0.5*self.fov)
		s = self.asym_range[0] + params[:, 2].unsqueeze(-1).unsqueeze(-1) * (self.asym_range[1]-self.asym_range[0])
		# eta = 2 * np.pi * (2.0 * params[:, 3].unsqueeze(-1).unsqueeze(-1) - 1.0)
		eta = 181/180 * np.pi * (2.0 * params[:, 3].unsqueeze(-1).unsqueeze(-1) - 1.0)
		
		nuisance_scale = []
		sigma_x_list = []
		sigma_y_list = []
		theta_list = []
		nuisance_x = []
		nuisance_y = []
		for k in range(self.n_gaussian):
			x_shift = self.shift_range[0]/(0.5*self.fov) + params[:, 4+k*6].unsqueeze(-1).unsqueeze(-1) * (self.shift_range[1] - self.shift_range[0])/(0.5*self.fov)
			y_shift = self.shift_range[0]/(0.5*self.fov) + params[:, 5+k*6].unsqueeze(-1).unsqueeze(-1) * (self.shift_range[1] - self.shift_range[0])/(0.5*self.fov)
			scale = self.gaussian_scale_range[0] + params[:, 6+k*6].unsqueeze(-1).unsqueeze(-1) * (self.gaussian_scale_range[1] - self.gaussian_scale_range[0])
			sigma_x = self.sigma_range[0]/(0.5*self.fov) + params[:, 7+k*6].unsqueeze(-1).unsqueeze(-1) * (self.sigma_range[1] - self.sigma_range[0])/(0.5*self.fov)
			sigma_y = self.sigma_range[0]/(0.5*self.fov) + params[:, 8+k*6].unsqueeze(-1).unsqueeze(-1) * (self.sigma_range[1] - self.sigma_range[0])/(0.5*self.fov)
			theta = 181/180 * 0.5 * np.pi * params[:, 9+k*6].unsqueeze(-1).unsqueeze(-1)

			# x_shift = 4 * (params[:, 4+k*6] - 0.5).unsqueeze(-1).unsqueeze(-1)
			# y_shift = 4 * (params[:, 5+k*6] - 0.5).unsqueeze(-1).unsqueeze(-1)
			# scale = params[:, 6+k*6].unsqueeze(-1).unsqueeze(-1)
			# # sigma_x = params[:, 7+k*6].unsqueeze(-1).unsqueeze(-1) * 0.5
			# # sigma_y = params[:, 8+k*6].unsqueeze(-1).unsqueeze(-1) * 0.5
			# sigma_x = 2 * params[:, 7+k*6].unsqueeze(-1).unsqueeze(-1) + self.gap
			# sigma_y = 2 * params[:, 8+k*6].unsqueeze(-1).unsqueeze(-1) + self.gap
			# theta = 181/180 * 0.5 * np.pi * (params[:, 9+k*6].unsqueeze(-1).unsqueeze(-1) - 0.5)
			

			nuisance_x.append(x_shift)
			nuisance_y.append(y_shift)
			nuisance_scale.append(scale)
			sigma_x_list.append(sigma_x)
			sigma_y_list.append(sigma_y)
			theta_list.append(theta)
			
		if self.flux_flag:
			total_flux = self.flux_range[0] + (self.flux_range[1]-self.flux_range[0]) * params[:, 4+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
			floor = self.floor_range[0] + (self.floor_range[1]-self.floor_range[0]) * params[:, 5+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
			crescent_flux = self.crescent_flux_range[0] + (self.crescent_flux_range[1]-self.crescent_flux_range[0]) * params[:, 6+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
			return r, sigma, s, eta, total_flux, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list
		else:
			floor = self.floor_range[0] + (self.floor_range[1]-self.floor_range[0]) * params[:, 4+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
			crescent_flux = self.crescent_flux_range[0] + (self.crescent_flux_range[1]-self.crescent_flux_range[0]) * params[:, 5+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
			return r, sigma, s, eta, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list

		# if self.flux_flag:
		# 	floor = self.floor_range[0] + (self.floor_range[1]-self.floor_range[0]) * params[:, 4+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
		# 	crescent_flux = self.crescent_flux_range[0] + (self.crescent_flux_range[1]-self.crescent_flux_range[0]) * params[:, 5+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
		# 	return r, sigma, s, eta, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list
		# else:
		# 	floor = self.floor_range[0] + (self.floor_range[1]-self.floor_range[0]) * params[:, 4+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
		# 	crescent_flux = 1
		# 	return r, sigma, s, eta, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list


	def forward(self, params):
		if self.flux_flag:
			r, sigma, s, eta, flux, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list = self.compute_features(params)
		else:
			r, sigma, s, eta, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list = self.compute_features(params)
		
		# r, sigma, s, eta, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list = self.compute_features(params)

		ring = torch.exp(- 0.5 * (self.grid_r - r)**2 / (sigma)**2)
		S = 1 + s * torch.cos(self.grid_theta - eta)
		crescent = S * ring
		disk = 0.5 * (1 + torch.erf((r - self.grid_r)/(np.sqrt(2)*sigma)))
		
		crescent = crescent / (torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)+self.eps)	
		disk = disk / (torch.sum(disk, (-1, -2)).unsqueeze(-1).unsqueeze(-1)+self.eps)
		crescent = crescent_flux * ((1-floor) * crescent + floor * disk)
		# crescent = crescent_flux * (crescent + floor * disk)

		# crescent = crescent + disk * floor * (1-s)
		# crescent = crescent_flux * crescent / (torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)+self.eps)


		for k in range(self.n_gaussian):
			x_c = self.grid_x - nuisance_x[k]
			y_c = self.grid_y - nuisance_y[k]
			x_rot = x_c * torch.cos(theta_list[k]) + y_c * torch.sin(theta_list[k])
			y_rot = -x_c * torch.sin(theta_list[k]) + y_c * torch.cos(theta_list[k])
			delta = 0.5 * (x_rot**2 / sigma_x_list[k]**2 + y_rot**2 / sigma_y_list[k]**2)
			# nuisance_now = nn.Softmax(dim=1)(-delta.reshape((-1, self.npix**2))).reshape((-1, self.npix, self.npix))#torch.exp(-delta)
			nuisance_now = 1 / (2 * np.pi * sigma_x_list[k] * sigma_y_list[k]) * torch.exp(-delta)
			nuisance_now = nuisance_now / (torch.sum(nuisance_now, (-1, -2)).unsqueeze(-1).unsqueeze(-1)+self.eps)
			# nuisance_now = (2*self.gap)**2 * nuisance_now
			nuisance_now = nuisance_scale[k] * nuisance_now# / torch.sum(nuisance_now, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
			crescent += nuisance_now

		

		if self.flux_flag:
			crescent = flux * crescent / (torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)+self.eps)
		else:
			crescent = crescent / (torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)+self.eps)


		# if self.flux_flag:
		# 	crescent = crescent
		# else:
		# 	crescent = crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)

		return crescent
		# return torch.clamp(crescent, min=0.0)

	def to(self, device):
		self.grid_x = self.grid_x.to(device)
		self.grid_y = self.grid_y.to(device)
		self.grid_r = self.grid_r.to(device)
		self.grid_theta = self.grid_theta.to(device)
		return self


# Tapper disk + gaussian ellipse
class NuisanceFloor_Param2Img(nn.Module):
	def __init__(self, npix, n_gaussian=1, fov=160, r_range=[10.0, 40.0],
				flux_range=[0.8, 1.2], crescent_flux_range=[1e-3, 2.0], 
				shift_range=[-200.0, 200.0], sigma_range=[1.0, 100.0], gaussian_scale_range=[1e-3, 2.0], flux_flag=False):
		super().__init__()
		self.n_gaussian = n_gaussian
		self.fov = fov
		self.r_range = r_range
		self.flux_range = flux_range
		self.crescent_flux_range = crescent_flux_range
		self.shift_range = shift_range
		self.sigma_range = sigma_range
		self.gaussian_scale_range = gaussian_scale_range
		self.flux_flag = flux_flag
		if self.flux_flag:
			self.nparams = 1 + 6 * n_gaussian + 2
		else:
			self.nparams = 6 * n_gaussian + 2

		# if self.flux_flag:
		# 	self.nparams = 6 + 6 * n_gaussian
		# else:
		# 	self.nparams = 5 + 6 * n_gaussian

		self.eps = 1e-4
		self.gap = 1.0 / npix
		xs = torch.arange(-1+self.gap, 1, 2*self.gap)
		# grid_x, grid_y = torch.meshgrid(xs, xs)
		grid_y, grid_x = torch.meshgrid(-xs, xs)
		self.grid_x = grid_x
		self.grid_y = grid_y
		self.grid_r = torch.sqrt(grid_x**2 + grid_y**2)
		self.grid_theta = torch.atan2(grid_y, grid_x)
		self.npix = npix

	def compute_features(self, params):
		r = self.r_range[0]/(0.5*self.fov) + params[:, 0].unsqueeze(-1).unsqueeze(-1) * (self.r_range[1]-self.r_range[0])/(0.5*self.fov)

		nuisance_scale = []
		sigma_x_list = []
		sigma_y_list = []
		theta_list = []
		nuisance_x = []
		nuisance_y = []
		for k in range(self.n_gaussian):
			x_shift = self.shift_range[0]/(0.5*self.fov) + params[:, 1+k*6].unsqueeze(-1).unsqueeze(-1) * (self.shift_range[1] - self.shift_range[0])/(0.5*self.fov)
			y_shift = self.shift_range[0]/(0.5*self.fov) + params[:, 2+k*6].unsqueeze(-1).unsqueeze(-1) * (self.shift_range[1] - self.shift_range[0])/(0.5*self.fov)
			scale = self.gaussian_scale_range[0] + params[:, 3+k*6].unsqueeze(-1).unsqueeze(-1) * (self.gaussian_scale_range[1] - self.gaussian_scale_range[0])
			sigma_x = self.sigma_range[0]/(0.5*self.fov) + params[:, 4+k*6].unsqueeze(-1).unsqueeze(-1) * (self.sigma_range[1] - self.sigma_range[0])/(0.5*self.fov)
			sigma_y = self.sigma_range[0]/(0.5*self.fov) + params[:, 5+k*6].unsqueeze(-1).unsqueeze(-1) * (self.sigma_range[1] - self.sigma_range[0])/(0.5*self.fov)
			theta = 181/180 * 0.5 * np.pi * params[:, 6+k*6].unsqueeze(-1).unsqueeze(-1)

			# x_shift = 4 * (params[:, 4+k*6] - 0.5).unsqueeze(-1).unsqueeze(-1)
			# y_shift = 4 * (params[:, 5+k*6] - 0.5).unsqueeze(-1).unsqueeze(-1)
			# scale = params[:, 6+k*6].unsqueeze(-1).unsqueeze(-1)
			# # sigma_x = params[:, 7+k*6].unsqueeze(-1).unsqueeze(-1) * 0.5
			# # sigma_y = params[:, 8+k*6].unsqueeze(-1).unsqueeze(-1) * 0.5
			# sigma_x = 2 * params[:, 7+k*6].unsqueeze(-1).unsqueeze(-1) + self.gap
			# sigma_y = 2 * params[:, 8+k*6].unsqueeze(-1).unsqueeze(-1) + self.gap
			# theta = 181/180 * 0.5 * np.pi * (params[:, 9+k*6].unsqueeze(-1).unsqueeze(-1) - 0.5)
			

			nuisance_x.append(x_shift)
			nuisance_y.append(y_shift)
			nuisance_scale.append(scale)
			sigma_x_list.append(sigma_x)
			sigma_y_list.append(sigma_y)
			theta_list.append(theta)
			
		if self.flux_flag:
			total_flux = self.flux_range[0] + (self.flux_range[1]-self.flux_range[0]) * params[:, 1+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
			crescent_flux = self.crescent_flux_range[0] + (self.crescent_flux_range[1]-self.crescent_flux_range[0]) * params[:, 2+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
			return r, total_flux, crescent_flux, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list
		else:
			crescent_flux = self.crescent_flux_range[0] + (self.crescent_flux_range[1]-self.crescent_flux_range[0]) * params[:, 1+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
			return r, crescent_flux, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list

		# if self.flux_flag:
		# 	floor = self.floor_range[0] + (self.floor_range[1]-self.floor_range[0]) * params[:, 4+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
		# 	crescent_flux = self.crescent_flux_range[0] + (self.crescent_flux_range[1]-self.crescent_flux_range[0]) * params[:, 5+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
		# 	return r, sigma, s, eta, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list
		# else:
		# 	floor = self.floor_range[0] + (self.floor_range[1]-self.floor_range[0]) * params[:, 4+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
		# 	crescent_flux = 1
		# 	return r, sigma, s, eta, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list


	def forward(self, params):
		if self.flux_flag:
			r, flux, crescent_flux, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list = self.compute_features(params)
		else:
			r, crescent_flux, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list = self.compute_features(params)
		
		# r, sigma, s, eta, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list = self.compute_features(params)

		sigma = 10.0/(0.5*self.fov)

		disk = 0.5 * (1 + torch.erf((r - self.grid_r)/(np.sqrt(2)*sigma)))
		
		disk = disk / (torch.sum(disk, (-1, -2)).unsqueeze(-1).unsqueeze(-1)+self.eps)
		crescent = crescent_flux * disk
		# crescent = crescent_flux * (crescent + floor * disk)

		# crescent = crescent + disk * floor * (1-s)
		# crescent = crescent_flux * crescent / (torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)+self.eps)


		for k in range(self.n_gaussian):
			x_c = self.grid_x - nuisance_x[k]
			y_c = self.grid_y - nuisance_y[k]
			x_rot = x_c * torch.cos(theta_list[k]) + y_c * torch.sin(theta_list[k])
			y_rot = -x_c * torch.sin(theta_list[k]) + y_c * torch.cos(theta_list[k])
			delta = 0.5 * (x_rot**2 / sigma_x_list[k]**2 + y_rot**2 / sigma_y_list[k]**2)
			# nuisance_now = nn.Softmax(dim=1)(-delta.reshape((-1, self.npix**2))).reshape((-1, self.npix, self.npix))#torch.exp(-delta)
			nuisance_now = 1 / (2 * np.pi * sigma_x_list[k] * sigma_y_list[k]) * torch.exp(-delta)
			nuisance_now = nuisance_now / (torch.sum(nuisance_now, (-1, -2)).unsqueeze(-1).unsqueeze(-1)+self.eps)
			# nuisance_now = (2*self.gap)**2 * nuisance_now
			nuisance_now = nuisance_scale[k] * nuisance_now# / torch.sum(nuisance_now, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
			crescent += nuisance_now

		

		if self.flux_flag:
			crescent = flux * crescent / (torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)+self.eps)
		else:
			crescent = crescent / (torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)+self.eps)


		# if self.flux_flag:
		# 	crescent = crescent
		# else:
		# 	crescent = crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)

		return crescent
		# return torch.clamp(crescent, min=0.0)

	def to(self, device):
		self.grid_x = self.grid_x.to(device)
		self.grid_y = self.grid_y.to(device)
		self.grid_r = self.grid_r.to(device)
		self.grid_theta = self.grid_theta.to(device)
		return self

# # Almost similar to paper VI version
# class SimpleCrescentNuisanceFloor_Param2Img(nn.Module):
# 	def __init__(self, npix, n_gaussian=1, fov=160, r_range=[10.0, 40.0], 
# 				width_range=[1.0, 40.0], floor_range=[0.0, 1.0], flux_range=[0.8, 1.2], crescent_flux_range=[1e-3, 2.0], 
# 				shift_range=[-200.0, 200.0], sigma_range=[1.0, 100.0], gaussian_scale_range=[1e-3, 2.0], flux_flag=False):
# 		super().__init__()
# 		self.n_gaussian = n_gaussian
# 		self.fov = fov
# 		self.r_range = r_range
# 		self.width_range = width_range
# 		self.floor_range = floor_range
# 		self.flux_range = flux_range
# 		self.crescent_flux_range = crescent_flux_range
# 		self.shift_range = shift_range
# 		self.sigma_range = sigma_range
# 		self.gaussian_scale_range = gaussian_scale_range
# 		self.flux_flag = flux_flag
# 		if self.flux_flag:
# 			self.nparams = 5 + 6 * n_gaussian + 2
# 		else:
# 			self.nparams = 4 + 6 * n_gaussian + 2

# 		# if self.flux_flag:
# 		# 	self.nparams = 6 + 6 * n_gaussian
# 		# else:
# 		# 	self.nparams = 5 + 6 * n_gaussian

# 		self.eps = 1e-4
# 		self.gap = 1.0 / npix
# 		xs = torch.arange(-1+self.gap, 1, 2*self.gap)
# 		# grid_x, grid_y = torch.meshgrid(xs, xs)
# 		grid_y, grid_x = torch.meshgrid(-xs, xs)
# 		self.grid_x = grid_x
# 		self.grid_y = grid_y
# 		self.grid_r = torch.sqrt(grid_x**2 + grid_y**2)
# 		self.grid_theta = torch.atan2(grid_y, grid_x)
# 		self.npix = npix

# 	def compute_features(self, params):
# 		r = self.r_range[0]/(0.5*self.fov) + params[:, 0].unsqueeze(-1).unsqueeze(-1) * (self.r_range[1]-self.r_range[0])/(0.5*self.fov)
# 		sigma = self.width_range[0]/(0.5*self.fov) + params[:, 1].unsqueeze(-1).unsqueeze(-1) * (self.width_range[1]-self.width_range[0])/(0.5*self.fov)
# 		s = params[:, 2].unsqueeze(-1).unsqueeze(-1)
# 		# eta = 2 * np.pi * (2.0 * params[:, 3].unsqueeze(-1).unsqueeze(-1) - 1.0)
# 		eta = 181/180 * np.pi * (2.0 * params[:, 3].unsqueeze(-1).unsqueeze(-1) - 1.0)
		
# 		nuisance_scale = []
# 		sigma_x_list = []
# 		sigma_y_list = []
# 		theta_list = []
# 		nuisance_x = []
# 		nuisance_y = []
# 		for k in range(self.n_gaussian):
# 			x_shift = self.shift_range[0]/(0.5*self.fov) + params[:, 4+k*6].unsqueeze(-1).unsqueeze(-1) * (self.shift_range[1] - self.shift_range[0])/(0.5*self.fov)
# 			y_shift = self.shift_range[0]/(0.5*self.fov) + params[:, 5+k*6].unsqueeze(-1).unsqueeze(-1) * (self.shift_range[1] - self.shift_range[0])/(0.5*self.fov)
# 			scale = self.gaussian_scale_range[0] + params[:, 6+k*6].unsqueeze(-1).unsqueeze(-1) * (self.gaussian_scale_range[1] - self.gaussian_scale_range[0])
# 			sigma_x = self.sigma_range[0]/(0.5*self.fov) + params[:, 7+k*6].unsqueeze(-1).unsqueeze(-1) * (self.sigma_range[1] - self.sigma_range[0])/(0.5*self.fov)
# 			sigma_y = self.sigma_range[0]/(0.5*self.fov) + 0.99 * params[:, 8+k*6].unsqueeze(-1).unsqueeze(-1) * params[:, 7+k*6].unsqueeze(-1).unsqueeze(-1) * (self.sigma_range[1] - self.sigma_range[0])/(0.5*self.fov)
# 			theta = 181/180 * np.pi * params[:, 9+k*6].unsqueeze(-1).unsqueeze(-1)

# 			# x_shift = 4 * (params[:, 4+k*6] - 0.5).unsqueeze(-1).unsqueeze(-1)
# 			# y_shift = 4 * (params[:, 5+k*6] - 0.5).unsqueeze(-1).unsqueeze(-1)
# 			# scale = params[:, 6+k*6].unsqueeze(-1).unsqueeze(-1)
# 			# # sigma_x = params[:, 7+k*6].unsqueeze(-1).unsqueeze(-1) * 0.5
# 			# # sigma_y = params[:, 8+k*6].unsqueeze(-1).unsqueeze(-1) * 0.5
# 			# sigma_x = 2 * params[:, 7+k*6].unsqueeze(-1).unsqueeze(-1) + self.gap
# 			# sigma_y = 2 * params[:, 8+k*6].unsqueeze(-1).unsqueeze(-1) + self.gap
# 			# theta = 181/180 * 0.5 * np.pi * (params[:, 9+k*6].unsqueeze(-1).unsqueeze(-1) - 0.5)
			

# 			nuisance_x.append(x_shift)
# 			nuisance_y.append(y_shift)
# 			nuisance_scale.append(scale)
# 			sigma_x_list.append(sigma_x)
# 			sigma_y_list.append(sigma_y)
# 			theta_list.append(theta)
			
# 		if self.flux_flag:
# 			total_flux = self.flux_range[0] + (self.flux_range[1]-self.flux_range[0]) * params[:, 4+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
# 			floor = self.floor_range[0] + (self.floor_range[1]-self.floor_range[0]) * params[:, 5+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
# 			crescent_flux = self.crescent_flux_range[0] + (self.crescent_flux_range[1]-self.crescent_flux_range[0]) * params[:, 6+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
# 			return r, sigma, s, eta, total_flux, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list
# 		else:
# 			floor = self.floor_range[0] + (self.floor_range[1]-self.floor_range[0]) * params[:, 4+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
# 			crescent_flux = self.crescent_flux_range[0] + (self.crescent_flux_range[1]-self.crescent_flux_range[0]) * params[:, 5+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
# 			return r, sigma, s, eta, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list

# 		# if self.flux_flag:
# 		# 	floor = self.floor_range[0] + (self.floor_range[1]-self.floor_range[0]) * params[:, 4+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
# 		# 	crescent_flux = self.crescent_flux_range[0] + (self.crescent_flux_range[1]-self.crescent_flux_range[0]) * params[:, 5+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
# 		# 	return r, sigma, s, eta, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list
# 		# else:
# 		# 	floor = self.floor_range[0] + (self.floor_range[1]-self.floor_range[0]) * params[:, 4+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
# 		# 	crescent_flux = 1
# 		# 	return r, sigma, s, eta, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list


# 	def forward(self, params):
# 		if self.flux_flag:
# 			r, sigma, s, eta, flux, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list = self.compute_features(params)
# 		else:
# 			r, sigma, s, eta, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list = self.compute_features(params)
		
# 		# r, sigma, s, eta, crescent_flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list = self.compute_features(params)

# 		ring = torch.exp(- 0.5 * (self.grid_r - r)**2 / (sigma)**2)
# 		S = 1 + s * torch.cos(self.grid_theta - eta)
# 		crescent = S * ring
# 		crescent = crescent / (torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)+self.eps)

# 		disk = 0.5 * (1 + torch.erf((r - self.grid_r)/(np.sqrt(2)*sigma)))
# 		disk = disk / (torch.sum(disk, (-1, -2)).unsqueeze(-1).unsqueeze(-1)+self.eps)
# 		crescent = crescent_flux * ((1-floor) * crescent + floor * disk)
# 		# crescent = crescent_flux * (crescent + floor * disk)

# 		for k in range(self.n_gaussian):
# 			x_c = self.grid_x - nuisance_x[k]
# 			y_c = self.grid_y - nuisance_y[k]
# 			x_rot = x_c * torch.cos(theta_list[k]) + y_c * torch.sin(theta_list[k])
# 			y_rot = -x_c * torch.sin(theta_list[k]) + y_c * torch.cos(theta_list[k])
# 			delta = 0.5 * (x_rot**2 / sigma_x_list[k]**2 + y_rot**2 / sigma_y_list[k]**2)
# 			# nuisance_now = nn.Softmax(dim=1)(-delta.reshape((-1, self.npix**2))).reshape((-1, self.npix, self.npix))#torch.exp(-delta)
# 			nuisance_now = 1 / (2 * np.pi * sigma_x_list[k] * sigma_y_list[k]) * torch.exp(-delta)
# 			nuisance_now = nuisance_now / (torch.sum(nuisance_now, (-1, -2)).unsqueeze(-1).unsqueeze(-1)+self.eps)
# 			# nuisance_now = (2*self.gap)**2 * nuisance_now
# 			nuisance_now = nuisance_scale[k] * nuisance_now# / torch.sum(nuisance_now, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
# 			crescent += nuisance_now

		

# 		if self.flux_flag:
# 			crescent = flux * crescent / (torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)+self.eps)
# 		else:
# 			crescent = crescent / (torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)+self.eps)


# 		# if self.flux_flag:
# 		# 	crescent = crescent
# 		# else:
# 		# 	crescent = crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)

# 		return crescent
# 		# return torch.clamp(crescent, min=0.0)

# 	def to(self, device):
# 		self.grid_x = self.grid_x.to(device)
# 		self.grid_y = self.grid_y.to(device)
# 		self.grid_r = self.grid_r.to(device)
# 		self.grid_theta = self.grid_theta.to(device)
# 		return self


# Almost similar to paper VI version
class NuisanceGaussian_Param2Img(nn.Module):
	def __init__(self, npix, n_gaussian=1, fov=160, flux_range=[0.8, 1.2], gaussian_scale_range=[1e-3, 2.0], 
				shift_range=[-200.0, 200.0], sigma_range=[1.0, 100.0], flux_flag=False):
		super().__init__()
		self.n_gaussian = n_gaussian
		self.fov = fov
		self.flux_range = flux_range
		self.shift_range = shift_range
		self.sigma_range = sigma_range
		self.gaussian_scale_range = gaussian_scale_range
		self.flux_flag = flux_flag
		if self.flux_flag:
			self.nparams = 6 * n_gaussian + 1
		else:
			self.nparams = 6 * n_gaussian

		# if self.flux_flag:
		# 	self.nparams = 6 + 6 * n_gaussian
		# else:
		# 	self.nparams = 5 + 6 * n_gaussian

		self.eps = 1e-4
		self.gap = 1.0 / npix
		xs = torch.arange(-1+self.gap, 1, 2*self.gap)
		# grid_x, grid_y = torch.meshgrid(xs, xs)
		grid_y, grid_x = torch.meshgrid(-xs, xs)
		self.grid_x = grid_x
		self.grid_y = grid_y
		self.grid_r = torch.sqrt(grid_x**2 + grid_y**2)
		self.grid_theta = torch.atan2(grid_y, grid_x)
		self.npix = npix

	def compute_features(self, params):

		nuisance_scale = []
		sigma_x_list = []
		sigma_y_list = []
		theta_list = []
		nuisance_x = []
		nuisance_y = []
		for k in range(self.n_gaussian):
			x_shift = self.shift_range[0]/(0.5*self.fov) + params[:, 0+k*6].unsqueeze(-1).unsqueeze(-1) * (self.shift_range[1] - self.shift_range[0])/(0.5*self.fov)
			y_shift = self.shift_range[0]/(0.5*self.fov) + params[:, 1+k*6].unsqueeze(-1).unsqueeze(-1) * (self.shift_range[1] - self.shift_range[0])/(0.5*self.fov)
			scale = self.gaussian_scale_range[0] + params[:, 2+k*6].unsqueeze(-1).unsqueeze(-1) * (self.gaussian_scale_range[1] - self.gaussian_scale_range[0])
			sigma_x = self.sigma_range[0]/(0.5*self.fov) + params[:, 3+k*6].unsqueeze(-1).unsqueeze(-1) * (self.sigma_range[1] - self.sigma_range[0])/(0.5*self.fov)
			sigma_y = self.sigma_range[0]/(0.5*self.fov) + params[:, 4+k*6].unsqueeze(-1).unsqueeze(-1) * (self.sigma_range[1] - self.sigma_range[0])/(0.5*self.fov)
			theta = 181/180 * 0.5 * np.pi * params[:, 5+k*6].unsqueeze(-1).unsqueeze(-1)

			nuisance_x.append(x_shift)
			nuisance_y.append(y_shift)
			nuisance_scale.append(scale)
			sigma_x_list.append(sigma_x)
			sigma_y_list.append(sigma_y)
			theta_list.append(theta)
			
		if self.flux_flag:
			total_flux = self.flux_range[0] + (self.flux_range[1]-self.flux_range[0]) * params[:, self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
			return total_flux, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list
		else:		
			return nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list


	def forward(self, params):
		if self.flux_flag:
			total_flux, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list = self.compute_features(params)
		else:
			nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, theta_list = self.compute_features(params)
		
		sumofgauss = 0
		for k in range(self.n_gaussian):
			x_c = self.grid_x - nuisance_x[k]
			y_c = self.grid_y - nuisance_y[k]
			x_rot = x_c * torch.cos(theta_list[k]) + y_c * torch.sin(theta_list[k])
			y_rot = -x_c * torch.sin(theta_list[k]) + y_c * torch.cos(theta_list[k])
			delta = 0.5 * (x_rot**2 / sigma_x_list[k]**2 + y_rot**2 / sigma_y_list[k]**2)
			nuisance_now = nn.Softmax(dim=1)(-delta.reshape((-1, self.npix**2))).reshape((-1, self.npix, self.npix))#torch.exp(-delta)
			nuisance_now = nuisance_scale[k] * nuisance_now# / torch.sum(nuisance_now, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
			sumofgauss += nuisance_now

		

		if self.flux_flag:
			sumofgauss = total_flux * sumofgauss / torch.sum(sumofgauss, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		else:
			sumofgauss = sumofgauss / torch.sum(sumofgauss, (-1, -2)).unsqueeze(-1).unsqueeze(-1)

		return sumofgauss

	def to(self, device):
		self.grid_x = self.grid_x.to(device)
		self.grid_y = self.grid_y.to(device)
		self.grid_r = self.grid_r.to(device)
		self.grid_theta = self.grid_theta.to(device)
		return self



# class SimpleCrescentNuisanceFloor_Param2Img(nn.Module):
# 	def __init__(self, npix, n_gaussian=1, fov=160, r_range=[10.0, 40.0], 
# 				width_range=[1.0, 40.0], floor_range=[0.0, 0.5], flux_range=[0.8, 1.2], flux_flag=False):
# 		super().__init__()
# 		self.n_gaussian = n_gaussian
# 		self.fov = fov
# 		self.r_range = r_range
# 		self.width_range = width_range
# 		self.floor_range = floor_range
# 		self.flux_range = flux_range
# 		self.flux_flag = flux_flag
# 		if self.flux_flag:
# 			self.nparams = 5 + 6 * n_gaussian + 1
# 		else:
# 			self.nparams = 4 + 6 * n_gaussian + 1
# 		self.eps = 1e-4
# 		self.gap = 1.0 / npix
# 		xs = torch.arange(-1+self.gap, 1, 2*self.gap)
# 		# grid_x, grid_y = torch.meshgrid(xs, xs)
# 		grid_y, grid_x = torch.meshgrid(-xs, xs)
# 		self.grid_x = grid_x
# 		self.grid_y = grid_y
# 		self.grid_r = torch.sqrt(grid_x**2 + grid_y**2)
# 		self.grid_theta = torch.atan2(grid_y, grid_x)


# 	def compute_features(self, params):
# 		r = self.r_range[0]/(0.5*self.fov) + params[:, 0].unsqueeze(-1).unsqueeze(-1) * (self.r_range[1]-self.r_range[0])/(0.5*self.fov)
# 		sigma = self.width_range[0]/(0.5*self.fov) + params[:, 1].unsqueeze(-1).unsqueeze(-1) * (self.width_range[1]-self.width_range[0])/(0.5*self.fov)
# 		s = params[:, 2].unsqueeze(-1).unsqueeze(-1)
# 		eta = 181/180 * np.pi * (2.0 * params[:, 3].unsqueeze(-1).unsqueeze(-1) - 1.0)
		
# 		nuisance_scale = []
# 		nuisance_covinv1 = []
# 		nuisance_covinv2 = []
# 		nuisance_covinv12 = []
# 		nuisance_x = []
# 		nuisance_y = []
# 		for k in range(self.n_gaussian):
# 			x_shift = (2 * params[:, 4+k*6] - 1).unsqueeze(-1).unsqueeze(-1)
# 			y_shift = (2 * params[:, 5+k*6] - 1).unsqueeze(-1).unsqueeze(-1)
# 			scale = params[:, 6+k*6].unsqueeze(-1).unsqueeze(-1)
# 			# sigma_x = params[:, 7+k*6].unsqueeze(-1).unsqueeze(-1) * 0.5
# 			# sigma_y = params[:, 8+k*6].unsqueeze(-1).unsqueeze(-1) * 0.5
# 			sigma_x = params[:, 7+k*6].unsqueeze(-1).unsqueeze(-1)
# 			sigma_y = params[:, 8+k*6].unsqueeze(-1).unsqueeze(-1)
# 			rho = 2 * 0.99 * (params[:, 9+k*6].unsqueeze(-1).unsqueeze(-1) - 0.5)
# 			sigma_xy = rho * sigma_x * sigma_y
# 			factor = self.eps**2 + self.eps * sigma_x**2 + self.eps * sigma_y**2 + (1-rho**2) * sigma_x**2 * sigma_y**2
# 			covinv1 = sigma_y**2 / factor
# 			covinv2 = sigma_x**2 / factor
# 			covinv12 = sigma_xy / factor

# 			nuisance_x.append(x_shift)
# 			nuisance_y.append(y_shift)
# 			nuisance_scale.append(scale)
# 			nuisance_covinv1.append(covinv1)
# 			nuisance_covinv2.append(covinv2)
# 			nuisance_covinv12.append(covinv12)
			
# 		if self.flux_flag:
# 			flux = self.flux_range[0] + (self.flux_range[1]-self.flux_range[0]) * params[:, 4+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
# 			floor = self.floor_range[0] + (self.floor_range[1]-self.floor_range[0]) * params[:, 5+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
# 			return r, sigma, s, eta, flux, floor, nuisance_scale, nuisance_x, nuisance_y, nuisance_covinv1, nuisance_covinv2, nuisance_covinv12
# 		else:
# 			floor = self.floor_range[0] + (self.floor_range[1]-self.floor_range[0]) * params[:, 4+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
# 			return r, sigma, s, eta, floor, nuisance_scale, nuisance_x, nuisance_y, nuisance_covinv1, nuisance_covinv2, nuisance_covinv12

# 	def compute_features2(self, params):
# 		r = self.r_range[0]/(0.5*self.fov) + params[:, 0].unsqueeze(-1).unsqueeze(-1) * (self.r_range[1]-self.r_range[0])/(0.5*self.fov)
# 		sigma = self.width_range[0]/(0.5*self.fov) + params[:, 1].unsqueeze(-1).unsqueeze(-1) * (self.width_range[1]-self.width_range[0])/(0.5*self.fov)
# 		s = params[:, 2].unsqueeze(-1).unsqueeze(-1)
# 		# eta = 2 * np.pi * (2.0 * params[:, 3].unsqueeze(-1).unsqueeze(-1) - 1.0)
# 		eta = 181/180 * np.pi * (2.0 * params[:, 3].unsqueeze(-1).unsqueeze(-1) - 1.0)
		
# 		nuisance_scale = []
# 		sigma_x_list = []
# 		sigma_y_list = []
# 		rho_list = []
# 		nuisance_x = []
# 		nuisance_y = []
# 		for k in range(self.n_gaussian):
# 			x_shift = (2 * params[:, 4+k*6] - 1).unsqueeze(-1).unsqueeze(-1)
# 			y_shift = (2 * params[:, 5+k*6] - 1).unsqueeze(-1).unsqueeze(-1)
# 			scale = params[:, 6+k*6].unsqueeze(-1).unsqueeze(-1)
# 			# sigma_x = params[:, 7+k*6].unsqueeze(-1).unsqueeze(-1) * 0.5
# 			# sigma_y = params[:, 8+k*6].unsqueeze(-1).unsqueeze(-1) * 0.5
# 			sigma_x = params[:, 7+k*6].unsqueeze(-1).unsqueeze(-1)
# 			sigma_y = params[:, 8+k*6].unsqueeze(-1).unsqueeze(-1)
# 			rho = 2 * 0.99 * (params[:, 9+k*6].unsqueeze(-1).unsqueeze(-1) - 0.5)
			

# 			nuisance_x.append(x_shift)
# 			nuisance_y.append(y_shift)
# 			nuisance_scale.append(scale)
# 			sigma_x_list.append(sigma_x)
# 			sigma_y_list.append(sigma_y)
# 			rho_list.append(rho)
			
# 		if self.flux_flag:
# 			flux = self.flux_range[0] + (self.flux_range[1]-self.flux_range[0]) * params[:, 4+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
# 			floor = self.floor_range[0] + (self.floor_range[1]-self.floor_range[0]) * params[:, 5+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
# 			return r, sigma, s, eta, flux, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, rho_list
# 		else:
# 			floor = self.floor_range[0] + (self.floor_range[1]-self.floor_range[0]) * params[:, 4+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
# 			return r, sigma, s, eta, floor, nuisance_scale, nuisance_x, nuisance_y, sigma_x_list, sigma_y_list, rho_list


# 	def forward(self, params):
# 		if self.flux_flag:
# 			r, sigma, s, eta, flux, floor, nuisance_scale, nuisance_x, nuisance_y, nuisance_covinv1, nuisance_covinv2, nuisance_covinv12 = self.compute_features(params)
# 		else:
# 			r, sigma, s, eta, floor, nuisance_scale, nuisance_x, nuisance_y, nuisance_covinv1, nuisance_covinv2, nuisance_covinv12 = self.compute_features(params)
		
# 		ring = torch.exp(- 0.5 * (self.grid_r - r)**2 / (sigma)**2)
# 		S = 1 + s * torch.cos(self.grid_theta - eta)
# 		crescent = S * ring


# 		# disk = floor * torch.sigmoid(20.0*(r - self.grid_r))
# 		# disk = floor * torch.sigmoid(0.5*(r - self.grid_r)*torch.abs(r - self.grid_r)/sigma**2)
# 		disk = floor * 0.5 * (1 + torch.erf((r - self.grid_r)/(np.sqrt(2)*sigma)))
# 		# disk = floor * 0.5 * (1 + torch.erf((r - self.grid_r)/sigma))

# 		# r_orig = 0.5 * r + 0.5 * torch.sqrt(r**2 + sigma**2)
# 		# disk = floor * 0.5 * (1 + torch.erf((r_orig - self.grid_r)/(np.sqrt(2)*sigma)))
# 		# # crescent = torch.where(crescent > disk, crescent, disk)
# 		crescent = crescent + disk

# 		# disk = 0.5 * (1 + torch.erf((r - self.grid_r)/(np.sqrt(2)*sigma)))
# 		# crescent = crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
# 		# disk = disk / torch.sum(disk, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
# 		# crescent = (1-floor) * crescent + floor *disk

# 		for k in range(self.n_gaussian):
# 			x_c = self.grid_x - nuisance_x[k]
# 			y_c = self.grid_y - nuisance_y[k]
# 			delta = 0.5 * (nuisance_covinv1[k] * x_c**2 + nuisance_covinv2[k] * y_c**2 - 2*nuisance_covinv12[k]*x_c*y_c)
# 			nuisance_now = torch.exp(-delta) * nuisance_scale[k]
# 			crescent += nuisance_now

		

# 		if self.flux_flag:
# 			crescent = flux * crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
# 		else:
# 			crescent = crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		
# 		return crescent

# 	def to(self, device):
# 		self.grid_x = self.grid_x.to(device)
# 		self.grid_y = self.grid_y.to(device)
# 		self.grid_r = self.grid_r.to(device)
# 		self.grid_theta = self.grid_theta.to(device)
# 		return self



class MringPhase_Param2Img(nn.Module):
	def __init__(self, npix, n_order=1, fov=160, r_range=[10.0, 40.0], 
				width_range=[1.0, 40.0], flux_range=[0.8, 1.2], flux_flag=False):
		super().__init__()
		self.fov = fov
		self.r_range = r_range
		self.width_range = width_range
		self.flux_range = flux_range
		self.flux_flag = flux_flag
		if self.flux_flag:
			self.nparams = 3 + 2 * n_order
		else:
			self.nparams = 2 + 2 * n_order

		self.eps = 1e-4
		self.gap = 1.0 / npix
		xs = torch.arange(-1+self.gap, 1, 2*self.gap)
		# grid_x, grid_y = torch.meshgrid(xs, xs)
		grid_y, grid_x = torch.meshgrid(-xs, xs)
		self.grid_x = grid_x
		self.grid_y = grid_y
		self.grid_r = torch.sqrt(grid_x**2 + grid_y**2)
		self.grid_theta = torch.atan2(grid_y, grid_x)
		self.n_order = n_order

	def compute_features(self, params):
		r = self.r_range[0]/(0.5*self.fov) + params[:, 0].unsqueeze(-1).unsqueeze(-1) * (self.r_range[1]-self.r_range[0])/(0.5*self.fov)
		sigma = self.width_range[0]/(0.5*self.fov) + params[:, 1].unsqueeze(-1).unsqueeze(-1) * (self.width_range[1]-self.width_range[0])/(0.5*self.fov)
		s_list = []
		eta_list = []
		for k in range(self.n_order):
			s = params[:, 2+2*k].unsqueeze(-1).unsqueeze(-1)
			eta = 181/180 * np.pi * (2.0 * params[:, 3+2*k].unsqueeze(-1).unsqueeze(-1) - 1.0)
			s_list.append(s)
			eta_list.append(eta)

		if self.flux_flag:
			flux = self.flux_range[0] + (self.flux_range[1]-self.flux_range[0]) * params[:, 2+2*self.n_order].unsqueeze(-1).unsqueeze(-1)
			return r, sigma, s_list, eta_list, flux
		else:
			return r, sigma, s_list, eta_list


	def forward(self, params):
		if self.flux_flag:
			r, sigma, s_list, eta_list, flux = self.compute_features(params)
		else:
			r, sigma, s_list, eta_list = self.compute_features(params)
		ring = torch.exp(- 0.5 * (self.grid_r - r)**2 / (sigma)**2)

		for k in range(self.n_order):
			s = s_list[k]
			eta = eta_list[k]
			if k == 0:
				S = 1 + s * torch.cos(self.grid_theta - eta)
			else:
				S = S + s * torch.cos((k+1) * self.grid_theta - eta)

		# crescent = S * ring
		if self.n_order > 0:
			crescent = S * ring
		else:
			crescent = ring

		if self.flux_flag:
			crescent = flux * crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		else:
			crescent = crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		
		return crescent

	def to(self, device):
		self.grid_x = self.grid_x.to(device)
		self.grid_y = self.grid_y.to(device)
		self.grid_r = self.grid_r.to(device)
		self.grid_theta = self.grid_theta.to(device)
		return self



# class MringPhase2_Param2Img(nn.Module):
# 	def __init__(self, npix, n_order=1, fov=160, r_range=[10.0, 40.0], 
# 				width_range=[1.0, 40.0], flux_range=[0.8, 1.2], flux_flag=False):
# 		super().__init__()
# 		self.fov = fov
# 		self.r_range = r_range
# 		self.width_range = width_range
# 		self.flux_range = flux_range
# 		self.flux_flag = flux_flag
# 		if self.flux_flag:
# 			self.nparams = 3 + 2 * n_order
# 		else:
# 			self.nparams = 2 + 2 * n_order

# 		self.eps = 1e-4
# 		self.gap = 1.0 / npix
# 		xs = torch.arange(-1+self.gap, 1, 2*self.gap)
# 		# grid_x, grid_y = torch.meshgrid(xs, xs)
# 		grid_y, grid_x = torch.meshgrid(-xs, xs)
# 		self.grid_x = grid_x
# 		self.grid_y = grid_y
# 		self.grid_r = torch.sqrt(grid_x**2 + grid_y**2)
# 		self.grid_theta = torch.atan2(grid_y, grid_x)
# 		self.n_order = n_order

# 		xs_kern = 2*self.gap * torch.arange(-5, 6, 1)
# 		grid_x_kern, grid_y_kern = torch.meshgrid(xs_kern, xs_kern)
# 		self.grid_r_kern = torch.sqrt(grid_x_kern**2 + grid_y_kern**2)

# 	def compute_features(self, params):
# 		r = self.r_range[0]/(0.5*self.fov) + params[:, 0].unsqueeze(-1).unsqueeze(-1) * (self.r_range[1]-self.r_range[0])/(0.5*self.fov)
# 		sigma = self.width_range[0]/(0.5*self.fov) + params[:, 1].unsqueeze(-1).unsqueeze(-1) * (self.width_range[1]-self.width_range[0])/(0.5*self.fov)
# 		s_list = []
# 		eta_list = []
# 		for k in range(self.n_order):
# 			s = params[:, 2+2*k].unsqueeze(-1).unsqueeze(-1)
# 			eta = 181/180 * np.pi * (2.0 * params[:, 3+2*k].unsqueeze(-1).unsqueeze(-1) - 1.0)
# 			s_list.append(s)
# 			eta_list.append(eta)

# 		if self.flux_flag:
# 			flux = self.flux_range[0] + (self.flux_range[1]-self.flux_range[0]) * params[:, 2+2*self.n_order].unsqueeze(-1).unsqueeze(-1)
# 			return r, sigma, s_list, eta_list, flux
# 		else:
# 			return r, sigma, s_list, eta_list


# 	def forward(self, params):
# 		if self.flux_flag:
# 			r, sigma, s_list, eta_list, flux = self.compute_features(params)
# 		else:
# 			r, sigma, s_list, eta_list = self.compute_features(params)

# 		ring = 1 / np.sqrt(2 * self.gap) * torch.exp(- 0.5 * (self.grid_r - r)**2 / (2 * self.gap+self.eps)**2)

# 		for k in range(self.n_order):
# 			s = s_list[k]
# 			eta = eta_list[k]
# 			if k == 0:
# 				S = 1 + s * torch.cos(self.grid_theta - eta)
# 			else:
# 				S = S + s * torch.cos((k+1) * self.grid_theta - eta)

		

# 		kernel = torch.exp(- 0.5 * (self.grid_r_kern / sigma)**2)
# 		kernel = kernel / torch.sum(kernel, (1, 2)).unsqueeze(-1).unsqueeze(-1)
# 		ring_reshape = ring.unsqueeze(0)
# 		ring = torch.nn.functional.conv2d(ring_reshape, kernel.unsqueeze(1), groups=kernel.shape[0], padding=5).permute(1, 0, 2, 3)
# 		ring = ring.squeeze(1)

# 		crescent = S * ring

# 		if self.flux_flag:
# 			crescent = flux * crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		
# 		return crescent

# 	def to(self, device):
# 		self.grid_x = self.grid_x.to(device)
# 		self.grid_y = self.grid_y.to(device)
# 		self.grid_r = self.grid_r.to(device)
# 		self.grid_theta = self.grid_theta.to(device)
# 		self.grid_r_kern = self.grid_r_kern.to(device)
# 		return self


class MringPhase2_Param2Img(nn.Module):
	def __init__(self, npix, n_order=1, fov=160, r_range=[10.0, 40.0], 
				width_range=[1.0, 40.0], flux_range=[0.8, 1.2], flux_flag=False):
		super().__init__()
		self.fov = fov
		self.r_range = r_range
		self.width_range = width_range
		self.flux_range = flux_range
		self.flux_flag = flux_flag
		if self.flux_flag:
			self.nparams = 3 + 2 * n_order
		else:
			self.nparams = 2 + 2 * n_order

		self.eps = 1e-4
		self.gap = 1.0 / npix
		xs = torch.arange(-1+self.gap, 1, 2*self.gap)
		# grid_x, grid_y = torch.meshgrid(xs, xs)
		grid_y, grid_x = torch.meshgrid(-xs, xs)
		self.grid_x = grid_x
		self.grid_y = grid_y
		self.grid_r = torch.sqrt(grid_x**2 + grid_y**2)
		self.grid_theta = torch.atan2(grid_y, grid_x)
		self.n_order = n_order

		xs_kern = 2*self.gap * torch.arange(-5, 6, 1)
		grid_x_kern, grid_y_kern = torch.meshgrid(xs_kern, xs_kern)
		self.grid_r_kern = torch.sqrt(grid_x_kern**2 + grid_y_kern**2)

	def compute_features(self, params):
		r = self.r_range[0]/(0.5*self.fov) + params[:, 0].unsqueeze(-1).unsqueeze(-1) * (self.r_range[1]-self.r_range[0])/(0.5*self.fov)
		sigma = self.width_range[0]/(0.5*self.fov) + params[:, 1].unsqueeze(-1).unsqueeze(-1) * (self.width_range[1]-self.width_range[0])/(0.5*self.fov)
		s_list = []
		eta_list = []
		for k in range(self.n_order):
			s = params[:, 2+2*k].unsqueeze(-1).unsqueeze(-1)
			eta = 181/180 * np.pi * (2.0 * params[:, 3+2*k].unsqueeze(-1).unsqueeze(-1) - 1.0)
			s_list.append(s)
			eta_list.append(eta)

		if self.flux_flag:
			flux = self.flux_range[0] + (self.flux_range[1]-self.flux_range[0]) * params[:, 2+2*self.n_order].unsqueeze(-1).unsqueeze(-1)
			return r, sigma, s_list, eta_list, flux
		else:
			return r, sigma, s_list, eta_list


	def forward(self, params):
		if self.flux_flag:
			r, sigma, s_list, eta_list, flux = self.compute_features(params)
		else:
			r, sigma, s_list, eta_list = self.compute_features(params)

		ring = 1 / np.sqrt(2 * self.gap) * torch.exp(- 0.5 * (self.grid_r - r)**2 / (2 * self.gap+self.eps)**2)

		for k in range(self.n_order):
			s = s_list[k]
			eta = eta_list[k]
			if k == 0:
				S = 1 + s * torch.cos(self.grid_theta - eta)
			else:
				S = S + s * torch.cos((k+1) * self.grid_theta - eta)

		crescent = S * ring

		kernel = torch.exp(- 0.5 * (self.grid_r_kern / sigma)**2)
		kernel = kernel / torch.sum(kernel, (1, 2)).unsqueeze(-1).unsqueeze(-1)
		crescent_reshape = crescent.unsqueeze(0)
		crescent = torch.nn.functional.conv2d(crescent_reshape, kernel.unsqueeze(1), groups=kernel.shape[0], padding=5).permute(1, 0, 2, 3)
		crescent = crescent.squeeze(1)

		if self.flux_flag:
			crescent = flux * crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		else:
			crescent = crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		
		return crescent

	def to(self, device):
		self.grid_x = self.grid_x.to(device)
		self.grid_y = self.grid_y.to(device)
		self.grid_r = self.grid_r.to(device)
		self.grid_theta = self.grid_theta.to(device)
		self.grid_r_kern = self.grid_r_kern.to(device)
		return self



class MringPhaseFloor_Param2Img(nn.Module):
	def __init__(self, npix, n_order=1, fov=160, r_range=[10.0, 40.0], 
				width_range=[1.0, 40.0], floor_range=[0.0, 1.0], flux_range=[0.8, 1.2], flux_flag=False):
		super().__init__()
		self.fov = fov
		self.r_range = r_range
		self.width_range = width_range
		self.floor_range = floor_range
		self.flux_range = flux_range
		self.flux_flag = flux_flag
		if self.flux_flag:
			self.nparams = 4 + 2 * n_order
		else:
			self.nparams = 3 + 2 * n_order

		self.eps = 1e-4
		self.gap = 1.0 / npix
		xs = torch.arange(-1+self.gap, 1, 2*self.gap)
		# grid_x, grid_y = torch.meshgrid(xs, xs)
		grid_y, grid_x = torch.meshgrid(-xs, xs)
		self.grid_x = grid_x
		self.grid_y = grid_y
		self.grid_r = torch.sqrt(grid_x**2 + grid_y**2)
		self.grid_theta = torch.atan2(grid_y, grid_x)
		self.n_order = n_order

	def compute_features(self, params):
		r = self.r_range[0]/(0.5*self.fov) + params[:, 0].unsqueeze(-1).unsqueeze(-1) * (self.r_range[1]-self.r_range[0])/(0.5*self.fov)
		sigma = self.width_range[0]/(0.5*self.fov) + params[:, 1].unsqueeze(-1).unsqueeze(-1) * (self.width_range[1]-self.width_range[0])/(0.5*self.fov)
		s_list = []
		eta_list = []
		for k in range(self.n_order):
			s = params[:, 2+2*k].unsqueeze(-1).unsqueeze(-1)
			eta = 181/180 * np.pi * (2.0 * params[:, 3+2*k].unsqueeze(-1).unsqueeze(-1) - 1.0)
			s_list.append(s)
			eta_list.append(eta)

		
		if self.flux_flag:
			flux = self.flux_range[0] + (self.flux_range[1]-self.flux_range[0]) * params[:, 2+2*self.n_order].unsqueeze(-1).unsqueeze(-1)
			floor = self.floor_range[0] + (self.floor_range[1]-self.floor_range[0]) * params[:, 3+2*self.n_order].unsqueeze(-1).unsqueeze(-1)
			return r, sigma, s_list, eta_list, floor, flux
		else:
			floor = self.floor_range[0] + (self.floor_range[1]-self.floor_range[0]) * params[:, 2+2*self.n_order].unsqueeze(-1).unsqueeze(-1)
			return r, sigma, s_list, eta_list, floor


	def forward(self, params):
		if self.flux_flag:
			r, sigma, s_list, eta_list, floor, flux = self.compute_features(params)
		else:
			r, sigma, s_list, eta_list, floor = self.compute_features(params)
		ring = torch.exp(- 0.5 * (self.grid_r - r)**2 / (sigma)**2)

		for k in range(self.n_order):
			s = s_list[k]
			eta = eta_list[k]
			if k == 0:
				S = 1 + s * torch.cos(self.grid_theta - eta)
			else:
				S = S + s * torch.cos((k+1) * self.grid_theta - eta)

		# crescent = torch.nn.ReLU(inplace=True)(S) * ring
		if self.n_order > 0:
			crescent = S * ring
		else:
			crescent = ring

		# disk = floor * torch.sigmoid(20.0*(r - self.grid_r))
		# disk = floor * torch.sigmoid(0.5*(r - self.grid_r)*torch.abs(r - self.grid_r)/sigma**2)
		# disk = floor * 0.5 * (1 + torch.erf((r - self.grid_r)/sigma))

		# r_orig = 0.5 * r + 0.5 * torch.sqrt(r**2 + sigma**2)
		# disk = floor * 0.5 * (1 + torch.erf((r_orig - self.grid_r)/(np.sqrt(2)*sigma)))

		# crescent = torch.where(crescent > disk, crescent, disk)


		# disk = floor * 0.5 * (1 + torch.erf((r - self.grid_r)/(np.sqrt(2)*sigma)))
		# crescent = crescent + disk

		disk = 0.5 * (1 + torch.erf((r - self.grid_r)/(np.sqrt(2)*sigma)))
		crescent = crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		disk = disk / torch.sum(disk, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		crescent = (1-floor) * crescent + floor *disk

		if self.flux_flag:
			crescent = flux * crescent# / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		else:
			crescent = crescent# / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		
		# return crescent
		return torch.clamp(crescent, min=0.0)

	def to(self, device):
		self.grid_x = self.grid_x.to(device)
		self.grid_y = self.grid_y.to(device)
		self.grid_r = self.grid_r.to(device)
		self.grid_theta = self.grid_theta.to(device)
		return self


class MringPhaseGaussianFloor_Param2Img(nn.Module):
	def __init__(self, npix, n_order=1, fov=160, r_range=[10.0, 40.0], floor_r_range=[10.0, 40.0],
				width_range=[1.0, 40.0], floor_range=[0.0, 1.0], flux_range=[0.8, 1.2], flux_flag=False):
		super().__init__()
		self.fov = fov
		self.r_range = r_range
		self.floor_r_range = floor_r_range
		self.width_range = width_range
		self.floor_range = floor_range
		self.flux_range = flux_range
		self.flux_flag = flux_flag
		if self.flux_flag:
			self.nparams = 5 + 2 * n_order#4 + 2 * n_order
		else:
			self.nparams = 4 + 2 * n_order#3 + 2 * n_order

		self.eps = 1e-4
		self.gap = 1.0 / npix
		xs = torch.arange(-1+self.gap, 1, 2*self.gap)
		# grid_x, grid_y = torch.meshgrid(xs, xs)
		grid_y, grid_x = torch.meshgrid(-xs, xs)
		self.grid_x = grid_x
		self.grid_y = grid_y
		self.grid_r = torch.sqrt(grid_x**2 + grid_y**2)
		self.grid_theta = torch.atan2(grid_y, grid_x)
		self.n_order = n_order

	def compute_features(self, params):
		r = self.r_range[0]/(0.5*self.fov) + params[:, 0].unsqueeze(-1).unsqueeze(-1) * (self.r_range[1]-self.r_range[0])/(0.5*self.fov)
		sigma = self.width_range[0]/(0.5*self.fov) + params[:, 1].unsqueeze(-1).unsqueeze(-1) * (self.width_range[1]-self.width_range[0])/(0.5*self.fov)
		s_list = []
		eta_list = []
		for k in range(self.n_order):
			s = params[:, 2+2*k].unsqueeze(-1).unsqueeze(-1)
			eta = 181/180 * np.pi * (2.0 * params[:, 3+2*k].unsqueeze(-1).unsqueeze(-1) - 1.0)
			s_list.append(s)
			eta_list.append(eta)

		
		if self.flux_flag:
			flux = self.flux_range[0] + (self.flux_range[1]-self.flux_range[0]) * params[:, 2+2*self.n_order].unsqueeze(-1).unsqueeze(-1)
			floor = self.floor_range[0] + (self.floor_range[1]-self.floor_range[0]) * params[:, 3+2*self.n_order].unsqueeze(-1).unsqueeze(-1)
			floor_r = self.floor_r_range[0]/(0.5*self.fov) + params[:, 4+2*self.n_order].unsqueeze(-1).unsqueeze(-1) * (self.floor_r_range[1]-self.floor_r_range[0])/(0.5*self.fov)
			return r, sigma, s_list, eta_list, floor, floor_r, flux
		else:
			floor = self.floor_range[0] + (self.floor_range[1]-self.floor_range[0]) * params[:, 2+2*self.n_order].unsqueeze(-1).unsqueeze(-1)
			floor_r = self.floor_r_range[0]/(0.5*self.fov) + params[:, 3+2*self.n_order].unsqueeze(-1).unsqueeze(-1) * (self.floor_r_range[1]-self.floor_r_range[0])/(0.5*self.fov)
			return r, sigma, s_list, eta_list, floor, floor_r


	def forward(self, params):
		if self.flux_flag:
			r, sigma, s_list, eta_list, floor, floor_r, flux = self.compute_features(params)
		else:
			r, sigma, s_list, eta_list, floor, floor_r = self.compute_features(params)
		ring = torch.exp(- 0.5 * (self.grid_r - r)**2 / (sigma)**2)

		for k in range(self.n_order):
			s = s_list[k]
			eta = eta_list[k]
			if k == 0:
				S = 1 + s * torch.cos(self.grid_theta - eta)
			else:
				S = S + s * torch.cos((k+1) * self.grid_theta - eta)

		# crescent = torch.nn.ReLU(inplace=True)(S) * ring
		if self.n_order > 0:
			crescent = S * ring
		else:
			crescent = ring

		# disk = floor * torch.sigmoid(20.0*(r - self.grid_r))
		# disk = floor * torch.sigmoid(0.5*(r - self.grid_r)*torch.abs(r - self.grid_r)/sigma**2)
		# disk = floor * 0.5 * (1 + torch.erf((r - self.grid_r)/sigma))

		# r_orig = 0.5 * r + 0.5 * torch.sqrt(r**2 + sigma**2)
		# disk = floor * 0.5 * (1 + torch.erf((r_orig - self.grid_r)/(np.sqrt(2)*sigma)))

		# crescent = torch.where(crescent > disk, crescent, disk)


		# disk = floor * 0.5 * (1 + torch.erf((r - self.grid_r)/(np.sqrt(2)*sigma)))
		# crescent = crescent + disk

		# disk = 0.5 * (1 + torch.erf((r - self.grid_r)/(np.sqrt(2)*sigma)))
		disk = 1/(floor_r)**2 * torch.exp(- 0.5 * (self.grid_r)**2 / (floor_r)**2)
		crescent = crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		disk = disk / torch.sum(disk, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		crescent = (1-floor) * crescent + floor *disk

		if self.flux_flag:
			crescent = flux * crescent# / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		else:
			crescent = crescent# / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		
		# return crescent
		return torch.clamp(crescent, min=0.0)

	def to(self, device):
		self.grid_x = self.grid_x.to(device)
		self.grid_y = self.grid_y.to(device)
		self.grid_r = self.grid_r.to(device)
		self.grid_theta = self.grid_theta.to(device)
		return self

# class MringPhaseFloor_Param2Img(nn.Module):
# 	def __init__(self, npix, n_order=1, fov=160, r_range=[10.0, 40.0], 
# 				width_range=[1.0, 40.0], floor_range=[0.0, 1.0], flux_range=[0.8, 1.2], flux_flag=False):
# 		super().__init__()
# 		self.fov = fov
# 		self.r_range = r_range
# 		self.width_range = width_range
# 		self.floor_range = floor_range
# 		self.flux_range = flux_range
# 		self.flux_flag = flux_flag
# 		if self.flux_flag:
# 			self.nparams = 4 + 2 * n_order
# 		else:
# 			self.nparams = 3 + 2 * n_order

# 		self.eps = 1e-4
# 		self.gap = 1.0 / npix
# 		xs = torch.arange(-1+self.gap, 1, 2*self.gap)
# 		# grid_x, grid_y = torch.meshgrid(xs, xs)
# 		grid_y, grid_x = torch.meshgrid(-xs, xs)
# 		self.grid_x = grid_x
# 		self.grid_y = grid_y
# 		self.grid_r = torch.sqrt(grid_x**2 + grid_y**2)
# 		self.grid_theta = torch.atan2(grid_y, grid_x)
# 		self.n_order = n_order

# 	def compute_features(self, params):
# 		r = self.r_range[0]/(0.5*self.fov) + params[:, 0].unsqueeze(-1).unsqueeze(-1) * (self.r_range[1]-self.r_range[0])/(0.5*self.fov)
# 		sigma = self.width_range[0]/(0.5*self.fov) + params[:, 1].unsqueeze(-1).unsqueeze(-1) * (self.width_range[1]-self.width_range[0])/(0.5*self.fov)
# 		s_list = []
# 		eta_list = []
# 		for k in range(self.n_order):
# 			s = params[:, 2+2*k].unsqueeze(-1).unsqueeze(-1)
# 			eta = 181/180 * np.pi * (2.0 * params[:, 3+2*k].unsqueeze(-1).unsqueeze(-1) - 1.0)
# 			s_list.append(s)
# 			eta_list.append(eta)

		
# 		if self.flux_flag:
# 			flux = self.flux_range[0] + (self.flux_range[1]-self.flux_range[0]) * params[:, 2+2*self.n_order].unsqueeze(-1).unsqueeze(-1)
# 			floor = self.floor_range[0] + (self.floor_range[1]-self.floor_range[0]) * params[:, 3+2*self.n_order].unsqueeze(-1).unsqueeze(-1)
# 			return r, sigma, s_list, eta_list, floor, flux
# 		else:
# 			floor = self.floor_range[0] + (self.floor_range[1]-self.floor_range[0]) * params[:, 2+2*self.n_order].unsqueeze(-1).unsqueeze(-1)
# 			return r, sigma, s_list, eta_list, floor


# 	def forward(self, params):
# 		if self.flux_flag:
# 			r, sigma, s_list, eta_list, floor, flux = self.compute_features(params)
# 		else:
# 			r, sigma, s_list, eta_list, floor = self.compute_features(params)
# 		ring = torch.exp(- 0.5 * (self.grid_r - r)**2 / (sigma)**2)

# 		for k in range(self.n_order):
# 			s = s_list[k]
# 			eta = eta_list[k]
# 			if k == 0:
# 				S = 1 + s * torch.cos(self.grid_theta - eta)
# 			else:
# 				S = S + s * torch.cos((k+1) * self.grid_theta - eta)

# 		crescent = S * ring

# 		# disk = floor * torch.sigmoid(20.0*(r - self.grid_r))
# 		# disk = floor * torch.sigmoid(0.5*(r - self.grid_r)*torch.abs(r - self.grid_r)/sigma**2)
# 		disk = floor * 0.5 * (1 + torch.erf((r - self.grid_r)/(np.sqrt(2)*sigma)))
# 		# disk = floor * 0.5 * (1 + torch.erf((r - self.grid_r)/sigma))

# 		# r_orig = 0.5 * r + 0.5 * torch.sqrt(r**2 + sigma**2)
# 		# disk = floor * 0.5 * (1 + torch.erf((r_orig - self.grid_r)/(np.sqrt(2)*sigma)))

# 		# crescent = torch.where(crescent > disk, crescent, disk)
# 		crescent = crescent + disk


# 		if self.flux_flag:
# 			crescent = flux * crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
# 		else:
# 			crescent = crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		
# 		return crescent

# 	def to(self, device):
# 		self.grid_x = self.grid_x.to(device)
# 		self.grid_y = self.grid_y.to(device)
# 		self.grid_r = self.grid_r.to(device)
# 		self.grid_theta = self.grid_theta.to(device)
# 		return self



class MringPhaseNuisance_Param2Img(nn.Module):
	def __init__(self, npix, n_order=1, n_gaussian=1, fov=160, r_range=[10.0, 40.0], 
				width_range=[1.0, 40.0], flux_range=[0.8, 1.2], flux_flag=False):
		super().__init__()
		self.fov = fov
		self.r_range = r_range
		self.width_range = width_range
		self.flux_range = flux_range
		self.flux_flag = flux_flag
		if self.flux_flag:
			self.nparams = 3 + 2 * n_order + 6 * n_gaussian
		else:
			self.nparams = 2 + 2 * n_order + 6 * n_gaussian

		self.eps = 1e-4
		self.gap = 1.0 / npix
		xs = torch.arange(-1+self.gap, 1, 2*self.gap)
		# grid_x, grid_y = torch.meshgrid(xs, xs)
		grid_y, grid_x = torch.meshgrid(-xs, xs)
		self.grid_x = grid_x
		self.grid_y = grid_y
		self.grid_r = torch.sqrt(grid_x**2 + grid_y**2)
		self.grid_theta = torch.atan2(grid_y, grid_x)
		self.n_order = n_order
		self.n_gaussian = n_gaussian

	def compute_features(self, params):
		r = self.r_range[0]/(0.5*self.fov) + params[:, 0].unsqueeze(-1).unsqueeze(-1) * (self.r_range[1]-self.r_range[0])/(0.5*self.fov)
		sigma = self.width_range[0]/(0.5*self.fov) + params[:, 1].unsqueeze(-1).unsqueeze(-1) * (self.width_range[1]-self.width_range[0])/(0.5*self.fov)
		s_list = []
		eta_list = []
		for k in range(self.n_order):
			s = params[:, 2+2*k].unsqueeze(-1).unsqueeze(-1)
			eta = 181/180 * np.pi * (2.0 * params[:, 3+2*k].unsqueeze(-1).unsqueeze(-1) - 1.0)
			s_list.append(s)
			eta_list.append(eta)
		
		nuisance_scale = []
		nuisance_covinv1 = []
		nuisance_covinv2 = []
		nuisance_covinv12 = []
		nuisance_x = []
		nuisance_y = []
		for k in range(self.n_gaussian):
			x_shift = (2 * params[:, 2+2*self.n_order+k*6] - 1).unsqueeze(-1).unsqueeze(-1)
			y_shift = (2 * params[:, 3+2*self.n_order+k*6] - 1).unsqueeze(-1).unsqueeze(-1)
			scale = params[:, 4+2*self.n_order+k*6].unsqueeze(-1).unsqueeze(-1)
			# sigma_x = params[:, 7+k*6].unsqueeze(-1).unsqueeze(-1) * 0.5
			# sigma_y = params[:, 8+k*6].unsqueeze(-1).unsqueeze(-1) * 0.5
			sigma_x = params[:, 5+2*self.n_order+k*6].unsqueeze(-1).unsqueeze(-1)
			sigma_y = params[:, 6+2*self.n_order+k*6].unsqueeze(-1).unsqueeze(-1)
			rho = 2 * 0.99 * (params[:, 7+2*self.n_order+k*6].unsqueeze(-1).unsqueeze(-1) - 0.5)
			sigma_xy = rho * sigma_x * sigma_y
			factor = self.eps**2 + (1-rho**2) * sigma_x**2 * sigma_y**2
			covinv1 = sigma_y**2 / factor
			covinv2 = sigma_x**2 / factor
			covinv12 = sigma_xy / factor

			nuisance_x.append(x_shift)
			nuisance_y.append(y_shift)
			nuisance_scale.append(scale)
			nuisance_covinv1.append(covinv1)
			nuisance_covinv2.append(covinv2)
			nuisance_covinv12.append(covinv12)
			
		if self.flux_flag:
			flux = self.flux_range[0] + (self.flux_range[1]-self.flux_range[0]) * params[:, 2+2*self.n_order+self.n_gaussian*6].unsqueeze(-1).unsqueeze(-1)
			return r, sigma, s_list, eta_list, flux, nuisance_scale, nuisance_x, nuisance_y, nuisance_covinv1, nuisance_covinv2, nuisance_covinv12
		else:
			return r, sigma, s_list, eta_list, nuisance_scale, nuisance_x, nuisance_y, nuisance_covinv1, nuisance_covinv2, nuisance_covinv12


	def forward(self, params):
		if self.flux_flag:
			r, sigma, s_list, eta_list, flux, nuisance_scale, nuisance_x, nuisance_y, nuisance_covinv1, nuisance_covinv2, nuisance_covinv12 = self.compute_features(params)
		else:
			r, sigma, s_list, eta_list, nuisance_scale, nuisance_x, nuisance_y, nuisance_covinv1, nuisance_covinv2, nuisance_covinv12 = self.compute_features(params)
		ring = torch.exp(- 0.5 * (self.grid_r - r)**2 / (sigma)**2)

		for k in range(self.n_order):
			s = s_list[k]
			eta = eta_list[k]
			if k == 0:
				S = 1 + s * torch.cos(self.grid_theta - eta)
			else:
				S = S + s * torch.cos((k+1) * self.grid_theta - eta)

		crescent = S * ring

		for k in range(self.n_gaussian):
			x_c = self.grid_x - nuisance_x[k]
			y_c = self.grid_y - nuisance_y[k]
			delta = 0.5 * (nuisance_covinv1[k] * x_c**2 + nuisance_covinv2[k] * y_c**2 - 2*nuisance_covinv12[k]*x_c*y_c)
			nuisance_now = torch.exp(-delta) * nuisance_scale[k]
			crescent += nuisance_now

		if self.flux_flag:
			crescent = flux * crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		else:
			crescent = crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)


		return crescent

	def to(self, device):
		self.grid_x = self.grid_x.to(device)
		self.grid_y = self.grid_y.to(device)
		self.grid_r = self.grid_r.to(device)
		self.grid_theta = self.grid_theta.to(device)
		return self


class PeriodicalCubicSpline(nn.Module):
	def __init__(self, n_pieces=8, delta_h=1.0, x0=0.0):
		super().__init__()
		self.n_pieces = n_pieces
		self.delta_h = delta_h
		self.x0 = x0
		self.x_left = torch.arange(self.n_pieces) * self.delta_h
		self.A = torch.zeros((3*self.n_pieces, 3*self.n_pieces))
		for k in range(self.n_pieces):
			self.A[k, 3*k] = delta_h
			self.A[k, 3*k+1] = delta_h**2
			self.A[k, 3*k+2] = delta_h**3
			
			self.A[k+self.n_pieces, 3*k] = 1
			self.A[k+self.n_pieces, 3*k+1] = 2*delta_h
			self.A[k+self.n_pieces, 3*k+2] = 3*delta_h**2
			self.A[k+self.n_pieces, (3*k+3)%(3*self.n_pieces)] = -1

			
			self.A[k+2*self.n_pieces, 3*k+1] = 2
			self.A[k+2*self.n_pieces, 3*k+2] = 6*delta_h
			self.A[k+2*self.n_pieces, (3*k+4)%(3*self.n_pieces)] = -2

		self.Ainv = torch.inverse(self.A)


	def fit(self, y):
		a = y.unsqueeze(-1)
		bcd = torch.matmul(y[:, np.arange(1, self.n_pieces+1)%self.n_pieces]-y, self.Ainv[:, 0:self.n_pieces].T)
		b = bcd[:, 0::3].unsqueeze(-1)
		c = bcd[:, 1::3].unsqueeze(-1)
		d = bcd[:, 2::3].unsqueeze(-1)
		return a, b, c, d

	def forward(self, y, x):
		a, b, c, d = self.fit(y)
		x_left = self.x0 + self.x_left.unsqueeze(-1)
		x_diff = x - x_left
		x_diff_clip = torch.where((x_diff<=self.delta_h*torch.ones_like(x_diff)) * (x_diff>=torch.zeros_like(x_diff)), x_diff, torch.zeros_like(x_diff))
		x_mask = torch.where((x_diff<=self.delta_h*torch.ones_like(x_diff)) * (x_diff>=torch.zeros_like(x_diff)), torch.ones_like(x_diff), torch.zeros_like(x_diff))
		y_interpolate = a * x_mask + b * x_diff_clip + c * x_diff_clip**2 + d * x_diff_clip**3
		y_interpolate = torch.sum(y_interpolate, 1)
		return y_interpolate


	def to(self, device):
		self.x_left = self.x_left.to(device)
		self.A = self.A.to(device)
		self.Ainv = self.Ainv.to(device)
		return self



class SplineBrightnessRing_Param2Img(nn.Module):
	def __init__(self, npix, n_pieces=4, fov=160, r_range=[10.0, 40.0], 
				width_range=[1.0, 40.0], flux_range=[0.8, 1.2], flux_flag=False):
		super().__init__()
		self.npix = npix
		self.n_pieces = n_pieces
		self.fov = fov
		self.r_range = r_range
		self.width_range = width_range
		self.flux_range = flux_range
		self.flux_flag = flux_flag
		if self.flux_flag:
			self.nparams = 3 + n_pieces
		else:
			self.nparams = 2 + n_pieces
		self.eps = 1e-4
		self.gap = 1.0 / npix
		xs = torch.arange(-1+self.gap, 1, 2*self.gap)
		# grid_x, grid_y = torch.meshgrid(xs, xs)
		grid_y, grid_x = torch.meshgrid(-xs, xs)
		self.grid_x = grid_x
		self.grid_y = grid_y
		self.grid_r = torch.sqrt(grid_x**2 + grid_y**2)
		self.grid_theta = torch.atan2(grid_y, grid_x)
		self.spline_brightness = PeriodicalCubicSpline(n_pieces=self.n_pieces, delta_h=2*np.pi/self.n_pieces, x0=-np.pi)

	def compute_features(self, params):
		r = self.r_range[0]/(0.5*self.fov) + params[:, 0].unsqueeze(-1).unsqueeze(-1) * (self.r_range[1]-self.r_range[0])/(0.5*self.fov)
		sigma = self.width_range[0]/(0.5*self.fov) + params[:, 1].unsqueeze(-1).unsqueeze(-1) * (self.width_range[1]-self.width_range[0])/(0.5*self.fov)
		y = 0.9 * (2 * params[:, 2:2+self.n_pieces] -1)
		if self.flux_flag:
			flux = self.flux_range[0] + (self.flux_range[1]-self.flux_range[0]) * params[:, 2+self.n_pieces].unsqueeze(-1).unsqueeze(-1)
			return r, sigma, y, flux
		else:
			return r, sigma, y


	def forward(self, params):
		if self.flux_flag:
			r, sigma, y, flux = self.compute_features(params)
		else:
			r, sigma, y = self.compute_features(params)
		ring = torch.exp(- 0.5 * (self.grid_r - r)**2 / (sigma)**2)
		y_interpolate = self.spline_brightness.forward(y, self.grid_theta.reshape((-1, )))
		S = 1.0 + y_interpolate.reshape((-1, self.npix, self.npix))
		crescent = S * ring

		if self.flux_flag:
			crescent = flux * crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		else:
			crescent = crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)


		return crescent

	def to(self, device):
		self.grid_x = self.grid_x.to(device)
		self.grid_y = self.grid_y.to(device)
		self.grid_r = self.grid_r.to(device)
		self.grid_theta = self.grid_theta.to(device)
		self.spline_brightness = self.spline_brightness.to(device)
		return self


class MringSplineWidthRing_Param2Img(nn.Module):
	def __init__(self, npix, n_order=1, n_pieces=4, fov=160, r_range=[10.0, 40.0], 
				width_range=[1.0, 40.0], flux_range=[0.8, 1.2], flux_flag=False):
		super().__init__()
		self.npix = npix
		self.n_pieces = n_pieces
		self.n_order = n_order
		self.fov = fov
		self.r_range = r_range
		self.width_range = width_range
		self.flux_range = flux_range
		self.flux_flag = flux_flag
		if self.flux_flag:
			self.nparams = 3 + n_pieces + 2*n_order
		else:
			self.nparams = 2 + n_pieces + 2*n_order
		self.eps = 1e-4
		self.gap = 1.0 / npix
		xs = torch.arange(-1+self.gap, 1, 2*self.gap)
		# grid_x, grid_y = torch.meshgrid(xs, xs)
		grid_y, grid_x = torch.meshgrid(-xs, xs)
		self.grid_x = grid_x
		self.grid_y = grid_y
		self.grid_r = torch.sqrt(grid_x**2 + grid_y**2)
		self.grid_theta = torch.atan2(grid_y, grid_x)
		self.spline = PeriodicalCubicSpline(n_pieces=self.n_pieces, delta_h=2*np.pi/self.n_pieces, x0=-np.pi)

	def compute_features(self, params):
		r = self.r_range[0]/(0.5*self.fov) + params[:, 0].unsqueeze(-1).unsqueeze(-1) * (self.r_range[1]-self.r_range[0])/(0.5*self.fov)
		sigma = self.width_range[0]/(0.5*self.fov) + params[:, 1].unsqueeze(-1).unsqueeze(-1) * (self.width_range[1]-self.width_range[0])/(0.5*self.fov)
		s_list = []
		eta_list = []
		for k in range(self.n_order):
			s = params[:, 2+2*k].unsqueeze(-1).unsqueeze(-1)
			eta = 181/180 * np.pi * (2.0 * params[:, 3+2*k].unsqueeze(-1).unsqueeze(-1) - 1.0)
			s_list.append(s)
			eta_list.append(eta)
		yw = 0.2 * (2 * params[:, (2+2*self.n_order):(2+2*self.n_order+self.n_pieces)] -1)

		if self.flux_flag:
			flux = self.flux_range[0] + (self.flux_range[1]-self.flux_range[0]) * params[:, (2+2*self.n_order+self.n_pieces)].unsqueeze(-1).unsqueeze(-1)
			return r, sigma, s_list, eta_list, yw, flux
		else:
			return r, sigma, s_list, eta_list, yw


	def forward(self, params):
		if self.flux_flag:
			r, sigma, s_list, eta_list, yw, flux = self.compute_features(params)
		else:
			r, sigma, s_list, eta_list, yw = self.compute_features(params)
		
		width_interpolate = self.spline.forward(yw, self.grid_theta.reshape((-1, )))
		sigma_varying = (1 + width_interpolate.reshape((-1, self.npix, self.npix))) * sigma
		ring = torch.exp(- 0.5 * (self.grid_r - r)**2 / (sigma_varying)**2)

		for k in range(self.n_order):
			s = s_list[k]
			eta = eta_list[k]
			if k == 0:
				S = 1 + s * torch.cos(self.grid_theta - eta)
			else:
				S = S + s * torch.cos((k+1) * self.grid_theta - eta)

		crescent = S * ring

		if self.flux_flag:
			crescent = flux * crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		else:
			crescent = crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		
		return crescent

	def to(self, device):
		self.grid_x = self.grid_x.to(device)
		self.grid_y = self.grid_y.to(device)
		self.grid_r = self.grid_r.to(device)
		self.grid_theta = self.grid_theta.to(device)
		self.spline = self.spline.to(device)
		return self




class SplineBrightnessWidthRing_Param2Img(nn.Module):
	def __init__(self, npix, n_pieces=4, fov=160, r_range=[10.0, 40.0], 
				width_range=[1.0, 40.0], flux_range=[0.8, 1.2], flux_flag=False):
		super().__init__()
		self.npix = npix
		self.n_pieces = n_pieces
		self.fov = fov
		self.r_range = r_range
		self.width_range = width_range
		self.flux_range = flux_range
		self.flux_flag = flux_flag
		if self.flux_flag:
			self.nparams = 3 + 2 * n_pieces
		else:
			self.nparams = 2 + 2 * n_pieces
		self.eps = 1e-4
		self.gap = 1.0 / npix
		xs = torch.arange(-1+self.gap, 1, 2*self.gap)
		# grid_x, grid_y = torch.meshgrid(xs, xs)
		grid_y, grid_x = torch.meshgrid(-xs, xs)
		self.grid_x = grid_x
		self.grid_y = grid_y
		self.grid_r = torch.sqrt(grid_x**2 + grid_y**2)
		self.grid_theta = torch.atan2(grid_y, grid_x)
		self.spline = PeriodicalCubicSpline(n_pieces=self.n_pieces, delta_h=2*np.pi/self.n_pieces, x0=-np.pi)

	def compute_features(self, params):
		r = self.r_range[0]/(0.5*self.fov) + params[:, 0].unsqueeze(-1).unsqueeze(-1) * (self.r_range[1]-self.r_range[0])/(0.5*self.fov)
		sigma = self.width_range[0]/(0.5*self.fov) + params[:, 1].unsqueeze(-1).unsqueeze(-1) * (self.width_range[1]-self.width_range[0])/(0.5*self.fov)
		yb = 0.9 * (2 * params[:, 2:2+self.n_pieces] -1)
		yw = 0.2 * (2 * params[:, (2+self.n_pieces):(2+2*self.n_pieces)] -1)
		if self.flux_flag:
			flux = self.flux_range[0] + (self.flux_range[1]-self.flux_range[0]) * params[:, 2+2*self.n_pieces].unsqueeze(-1).unsqueeze(-1)
			return r, sigma, yb, yw, flux
		else:
			return r, sigma, yb, yw


	def forward(self, params):
		if self.flux_flag:
			r, sigma, yb, yw, flux = self.compute_features(params)
		else:
			r, sigma, yb, yw = self.compute_features(params)

		width_interpolate = self.spline.forward(yw, self.grid_theta.reshape((-1, )))
		sigma_varying = (1 + width_interpolate.reshape((-1, self.npix, self.npix))) * sigma
		ring = torch.exp(- 0.5 * (self.grid_r - r)**2 / (sigma_varying)**2)
		
		y_interpolate = self.spline.forward(yb, self.grid_theta.reshape((-1, )))
		S = 1.0 + y_interpolate.reshape((-1, self.npix, self.npix))
		crescent = S * ring

		if self.flux_flag:
			crescent = flux * crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
		else:
			crescent = crescent / torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)


		return crescent

	def to(self, device):
		self.grid_x = self.grid_x.to(device)
		self.grid_y = self.grid_y.to(device)
		self.grid_r = self.grid_r.to(device)
		self.grid_theta = self.grid_theta.to(device)
		self.spline = self.spline.to(device)
		return self