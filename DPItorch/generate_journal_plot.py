import numpy as np
import corner
import matplotlib.pyplot as plt
import h5py
# plt.ion()


###################################################
## Fig. 1
###################################################

postsamples = np.load('betapic_postsamples_norej_alpha1.0.npy')
postsamples_rej = np.load('betapic_postsamples_rej_alpha1.0.npy')
postsamples_rej2 = np.load('betapic_postsamples_rej2_alpha1.0.npy')



postsamples = np.load('betapic_postsamples_norej_alpha0.99.npy')
postsamples_rej = np.load('betapic_postsamples_rej_alpha0.99.npy')
postsamples_rej2 = np.load('betapic_postsamples_rej2_alpha0.99.npy')


postsamples = np.load('betapic_postsamples_norej_alpha0.9.npy')
postsamples_rej = np.load('betapic_postsamples_rej_alpha0.9.npy')
postsamples_rej2 = np.load('betapic_postsamples_rej2_alpha0.9.npy')


postsamples = np.load('betapic_postsamples_norej_alpha0.5.npy')
postsamples_rej = np.load('betapic_postsamples_rej_alpha0.5.npy')
postsamples_rej2 = np.load('betapic_postsamples_rej2_alpha0.5.npy')

# corner.corner(postsamples, labels=['sma', 'ecc', 'inc', 'aop', 'pan', 'tau', 'plx', 'mtot'], bins=50, quantiles=[0.16, 0.84])

# labels = ['sma', 'ecc', 'inc', 'aop', 'pan', 'tau', 'plx', 'mtot']
labels = ['$a$', '$e$', '$i$', '$\omega$', '$\Omega$', '$\tau$', '$\pi$', '$M_T$']
ranges = [[8, 40], [0.0, 0.9], [85.5, 93.0], [0, 360], [29.6, 33.0], [0.0, 1.0], [50.8, 52.1], [1.5, 2.0]]
corner.corner(postsamples[0:500000], labels=labels, bins=50, quantiles=[0.16, 0.84],
				range=ranges)
corner.corner(postsamples_rej2[0:200000], labels=labels, bins=50, quantiles=[0.16, 0.84],
				range=ranges)



###################################################
## Fig. 1
###################################################

postsamples = np.load('betapic_postsamples_importance_alpha1.0.npy')



postsamples = np.load('./checkpoint/orbit_beta_pic_b/cartesian/alpha099/betapic_postsamples_importance_alpha0.99.npy')



postsamples = np.load('betapic_postsamples_importance_alpha0.9.npy')



postsamples = np.load('./checkpoint/orbit_beta_pic_b/cartesian/alpha05/betapic_postsamples_importance_alpha0.5.npy')



# corner.corner(postsamples, labels=['sma', 'ecc', 'inc', 'aop', 'pan', 'tau', 'plx', 'mtot'], bins=50, quantiles=[0.16, 0.84])

# labels = ['sma', 'ecc', 'inc', 'aop', 'pan', 'tau', 'plx', 'mtot']
labels = ['$a$', '$e$', '$i$', '$\omega$', '$\Omega$', r'$\tau$', '$\pi$', '$M_T$']
ranges = [[8, 40], [0.0, 0.9], [85.5, 93.0], [0, 360], [29.6, 33.0], [0.0, 1.0], [50.8, 52.1], [1.5, 2.0]]
# corner.corner(postsamples[0:500000, 0:8], labels=labels, bins=50, quantiles=[0.16, 0.84],
# 				range=ranges)


# fig = corner.corner(postsamples[0:500000, 0:8], labels=labels, use_math_text=True, max_n_ticks=4, 
# 				bins=50, quantiles=[0.16, 0.84],range=ranges)
fig = corner.corner(postsamples[0:500000, 0:8], use_math_text=True, max_n_ticks=3, 
				bins=50, quantiles=[0.16, 0.84],range=ranges)

count = 0
for ax in fig.get_axes():
	# ax.tick_params(axis='y', rotation=0)
	# ax.tick_params(axis='x', rotation=90)
	# ax.tick_params(axis='both', labelsize=14)
	# ax.xaxis.get_label().set_fontsize(24)
	# ax.yaxis.get_label().set_fontsize(24)
	if count in [0, 9, 18, 27, 36, 45, 54, 63]:
		ax.annotate(labels[count//9], xy=(0.8, 0.6), size=42, xycoords='axes fraction', ha='center')
	count += 1

	# ax.tick_params(axis='both', labelsize=24)
	ax.tick_params(axis='x', labelsize=24, rotation=45)
	ax.tick_params(axis='y', labelsize=24, rotation=0)
	# ax.xaxis.get_label().set_fontsize(40)
	# ax.yaxis.get_label().set_fontsize(40)

# plt.show()
# plt.savefig('/home/groot/BoumanLab/Lab_meeting/Journal_figures/orbit_new2/orbit0.5_corner.pdf')
plt.savefig('/home/groot/BoumanLab/Lab_meeting/Journal_figures/orbit_new2/orbit0.99_corner.pdf')


# fig = corner.corner(postsamples[0:500000, 0:8], labels=labels, weights=postsamples[0:500000, 8], 
# 				 use_math_text=True, bins=50, quantiles=[0.16, 0.84],range=ranges)
fig = corner.corner(postsamples[0:500000, 0:8] , weights=postsamples[0:500000, 8], use_math_text=True, max_n_ticks=3, 
				bins=50, quantiles=[0.16, 0.84],range=ranges)

# for ax in fig.get_axes():
# 	# ax.tick_params(axis='y', rotation=0)
# 	# ax.tick_params(axis='x', rotation=90)
# 	ax.tick_params(axis='both', labelsize=14)
# 	ax.xaxis.get_label().set_fontsize(24)
# 	ax.yaxis.get_label().set_fontsize(24)

count = 0
for ax in fig.get_axes():
	# ax.tick_params(axis='y', rotation=0)
	# ax.tick_params(axis='x', rotation=90)
	# ax.tick_params(axis='both', labelsize=14)
	# ax.xaxis.get_label().set_fontsize(24)
	# ax.yaxis.get_label().set_fontsize(24)
	if count in [0, 9, 18, 27, 36, 45, 54, 63]:
		ax.annotate(labels[count//9], xy=(0.8, 0.6), size=42, xycoords='axes fraction', ha='center')
	count += 1

	# ax.tick_params(axis='both', labelsize=24)
	ax.tick_params(axis='x', labelsize=24, rotation=45)
	ax.tick_params(axis='y', labelsize=24, rotation=0)


# plt.show()
plt.savefig('/home/groot/BoumanLab/Lab_meeting/Journal_figures/orbit_new2/orbit0.5_corner_IS.pdf')


resamp_ind = np.random.choice(np.arange(postsamples.shape[0]), size=500000, replace=True, p=postsamples[:, 8]/np.sum(postsamples[:, 8]))

# fig = corner.corner(postsamples[resamp_ind, 0:8], labels=labels, 
# 			 use_math_text=True, bins=50, quantiles=[0.16, 0.84], range=ranges)

fig = corner.corner(postsamples[resamp_ind, 0:8], use_math_text=True, max_n_ticks=3, 
				bins=50, quantiles=[0.16, 0.84],range=ranges)

# for ax in fig.get_axes():
# 	# ax.tick_params(axis='y', rotation=0)
# 	# ax.tick_params(axis='x', rotation=90)
# 	# ax.tick_params(axis='both', labelsize=14)
# 	# ax.xaxis.get_label().set_fontsize(24)
# 	# ax.yaxis.get_label().set_fontsize(24)

# 	ax.tick_params(axis='both', labelsize=24)
# 	ax.xaxis.get_label().set_fontsize(40)
# 	ax.yaxis.get_label().set_fontsize(40)

count = 0
for ax in fig.get_axes():
	# ax.tick_params(axis='y', rotation=0)
	# ax.tick_params(axis='x', rotation=90)
	# ax.tick_params(axis='both', labelsize=14)
	# ax.xaxis.get_label().set_fontsize(24)
	# ax.yaxis.get_label().set_fontsize(24)
	if count in [0, 9, 18, 27, 36, 45, 54, 63]:
		ax.annotate(labels[count//9], xy=(0.8, 0.6), size=42, xycoords='axes fraction', ha='center')
	count += 1

	# ax.tick_params(axis='both', labelsize=24)
	ax.tick_params(axis='x', labelsize=24, rotation=45)
	ax.tick_params(axis='y', labelsize=24, rotation=0)

# plt.show()

plt.savefig('/home/groot/BoumanLab/Lab_meeting/Journal_figures/orbit_new2/orbit0.5_corner_SIR.pdf')




postsamples = h5py.File('./checkpoint/orbit_beta_pic_b/cartesian/hetest_radec_chains.hdf5', 'r')
postsamples = postsamples.get('post').value
postsamples[:, 2:5] = 180/np.pi * postsamples[:, 2:5]


# fig = corner.corner(postsamples[:, 0:8], labels=labels, 
# 			 use_math_text=True, bins=50, quantiles=[0.16, 0.84], range=ranges)

fig = corner.corner(postsamples[:, 0:8], max_n_ticks=3, 
			 use_math_text=True, bins=50, quantiles=[0.16, 0.84], range=ranges)

# for ax in fig.get_axes():
# 	# ax.tick_params(axis='y', rotation=0)
# 	# ax.tick_params(axis='x', rotation=90)
# 	ax.tick_params(axis='both', labelsize=14)
# 	ax.xaxis.get_label().set_fontsize(24)
# 	ax.yaxis.get_label().set_fontsize(24)

count = 0
for ax in fig.get_axes():
	# ax.tick_params(axis='y', rotation=0)
	# ax.tick_params(axis='x', rotation=90)
	# ax.tick_params(axis='both', labelsize=14)
	# ax.xaxis.get_label().set_fontsize(24)
	# ax.yaxis.get_label().set_fontsize(24)
	if count in [0, 9, 18, 27, 36, 45, 54, 63]:
		ax.annotate(labels[count//9], xy=(0.8, 0.6), size=42, xycoords='axes fraction', ha='center')
	count += 1

	# ax.tick_params(axis='both', labelsize=24)
	ax.tick_params(axis='x', labelsize=24, rotation=45)
	ax.tick_params(axis='y', labelsize=24, rotation=0)

# plt.show()

plt.savefig('/home/groot/BoumanLab/Lab_meeting/Journal_figures/orbit_new2/orbit_mcmc_corner.pdf')




corner.corner(postsamples[0:8192, 0:8], labels=labels, bins=50, quantiles=[0.16, 0.84],
				range=ranges)
corner.corner(postsamples[0:8192, 0:8], labels=labels, weights=postsamples[0:8192, 8], bins=50, quantiles=[0.16, 0.84],
				range=ranges)



###################################################
## Fig. 1
###################################################
###
#lo
###
# postsamples = np.load('postsamples-1_norej_loop9.npy')
# postsamples_rej = np.load('postsamples-1_rej_loop9.npy')
# postsamples_rej2 = np.load('postsamples-1_rej2_loop9.npy')
# postsamples_rej3 = np.load('postsamples-1_rej3.npy')


postsamples = np.load('postsamples-1_importance.npy')
postsamples[:, 0] = 0.5 * postsamples[:, 0] + np.sqrt(0.25 * postsamples[:, 0]**2 + 0.25 * postsamples[:, 1]**2)
postsamples[:, 1] = postsamples[:, 1] * np.sqrt(2 * np.log(2))
postsamples[:, 3] = (postsamples[:, 3] - 90)%360



labels = ['$d$', '$w$', '$a$', r'$\theta_c$', '$V_c$', '$V_F$', 
		'$V_{g, 1}$', '$\delta x_{1}$', '$\delta y_{1}$', '$w_{x, 1}$', '$w_{y, 1}$', r'$\theta_{g, 1}$',
		'$V_{g, 2}$', '$\delta x_{2}$', '$\delta y_{2}$', '$w_{x, 2}$', '$w_{y, 2}$', r'$\theta_{g, 2}$']
# ranges = [[38, 44], [6, 14], [0.7, 0.9], [-120, -110], [0.25, 2], [0, 0.05], 
# 			[0, 2], [-20, 35], [-20, 12], [0, 100], [10, 50], [30, 60], 
# 			[0, 2], [-20, 35], [-20, 12], [0, 100], [10, 50], [30, 60]]


# ranges = [[38, 44], [6, 14], [0.7, 0.9], [-120, -110], [0.25, 2], [0, 0.06], 
# 			[0, 2], [-30, 30], [-25, 10], [0, 100], [10, 50], [30, 60], 
# 			[0, 2], [-30, 30], [-25, 10], [0, 100], [10, 50], [30, 60]]

ranges = [[39, 44.5], [6, 16], [0.68, 0.92], [152, 160], [-0.1, 2.15], [-0.0001, 0.07], 
			[-0.2, 2.1], [-35, 33], [-35, 15], [-6, 103], [10, 63], [29, 61], 
			[-0.2, 2.1], [-35, 33], [-35, 15], [-6, 103], [10, 63], [29, 61]]


fig = corner.corner(postsamples[0:819200, 0:18], bins=60, quantiles=[0.16, 0.84], 
			labels=labels, range=ranges, max_n_ticks=1)

for ax in fig.get_axes():
	ax.tick_params(axis='y', rotation=0)
	ax.tick_params(axis='x', rotation=90)
	ax.tick_params(axis='both', labelsize=24)
	ax.xaxis.get_label().set_fontsize(24)
	ax.yaxis.get_label().set_fontsize(24)

# count = 0
# for ax in fig.get_axes(): 
# 	if count in [0, 7, 14, 21, 28, 35]:
# 		ax.annotate(labels[count//7], xy=(0.2, 0.7), size=48, xycoords='axes fraction', ha='center')
# 	count += 1
# 	# ax.tick_params(axis='x', labelsize=28, rotation=30)
# 	# ax.tick_params(axis='y', labelsize=28, rotation=60)
# 	ax.tick_params(axis='x', labelsize=28)
# 	ax.tick_params(axis='y', labelsize=28)

# fig.set_size_inches(16, 16)
plt.savefig('/home/groot/BoumanLab/Lab_meeting/Journal_figures/m87_new/m87_corner.pdf')




fig = corner.corner(postsamples[:, 0:18], bins=60, quantiles=[0.16, 0.84], 
			weights=postsamples[:, 18], labels=labels, range=ranges, max_n_ticks=1)

for ax in fig.get_axes():
	ax.tick_params(axis='y', rotation=0)
	ax.tick_params(axis='x', rotation=90)
	ax.tick_params(axis='both', labelsize=24)
	ax.xaxis.get_label().set_fontsize(24)
	ax.yaxis.get_label().set_fontsize(24)

# fig.set_size_inches(16, 16)
plt.savefig('/home/groot/BoumanLab/Lab_meeting/Journal_figures/m87_new/m87_corner_IS.pdf')


resamp_ind = np.random.choice(np.arange(postsamples.shape[0]), size=819200, replace=True, p=postsamples[:, 18]/np.sum(postsamples[:, 18]))
fig = corner.corner(postsamples[resamp_ind, 0:18], bins=60, quantiles=[0.16, 0.84], 
			labels=labels, range=ranges, max_n_ticks=1)

for ax in fig.get_axes():
	ax.tick_params(axis='y', rotation=0)
	ax.tick_params(axis='x', rotation=90)
	ax.tick_params(axis='both', labelsize=20)
	ax.xaxis.get_label().set_fontsize(24)
	ax.yaxis.get_label().set_fontsize(24)

# fig.set_size_inches(16, 16)
plt.savefig('/home/groot/BoumanLab/Lab_meeting/Journal_figures/m87_new/m87_corner_SIR2.pdf')



import pandas as pd
chains = pd.read_csv('chain_equal.csv')
postsamples = chains[['diam', 'fwhm', 'ma1', 'mp1', 'f', 'floor', 
					 'fg1', 'xg1', 'yg1', 'σxg1', 'σyg1', 'ξg1',
					'fg2', 'xg2', 'yg2', 'σxg2', 'σyg2', 'ξg2']]

postsamples = np.array(postsamples)

postsamples[:, 2] = 2*postsamples[:, 2]
postsamples[:, 3] = 360-(180/np.pi * postsamples[:, 3])%360

postsamples[:, 11] = 180/np.pi * postsamples[:, 11]

postsamples[:, 17] = 180/np.pi * postsamples[:, 17]


fig = corner.corner(postsamples[:, 0:18], bins=60, quantiles=[0.16, 0.84], 
			labels=labels, max_n_ticks=1)

# fig = corner.corner(postsamples[:, 0:18], bins=100, quantiles=[0.16, 0.84], 
# 			labels=labels, range=ranges, max_n_ticks=1)
# # fig.set_size_inches(18.5, 10.5)

for ax in fig.get_axes():
	ax.tick_params(axis='y', rotation=0)
	ax.tick_params(axis='x', rotation=90)
	ax.tick_params(axis='both', labelsize=20)
	ax.xaxis.get_label().set_fontsize(24)
	ax.yaxis.get_label().set_fontsize(24)

plt.savefig('/home/groot/BoumanLab/Lab_meeting/Journal_figures/m87_new/m87_corner_mcmc.pdf')




ranges = [[39, 44.5], [0, 16], [0.50, 0.92], [142, 160], [-0.1, 2.15], [-0.0001, 0.07]]
resamp_ind = np.random.choice(np.arange(postsamples.shape[0]), size=12274, replace=True, p=postsamples[:, 18]/np.sum(postsamples[:, 18]))


fig = corner.corner(postsamples[resamp_ind, 0:6], bins=60, quantiles=[0.16, 0.84], 
			labels=labels, max_n_ticks=1, ranges=ranges)

corner.corner(postsamples_rose[:, 0:6], bins=60, quantiles=[0.16, 0.84], 
			labels=labels, max_n_ticks=1, color='c', ranges=ranges, fig=fig)


for ax in fig.get_axes():
	ax.tick_params(axis='y', rotation=0)
	ax.tick_params(axis='x', rotation=90)
	ax.tick_params(axis='both', labelsize=24)
	ax.xaxis.get_label().set_fontsize(24)
	ax.yaxis.get_label().set_fontsize(24)

plt.savefig('/home/groot/BoumanLab/Lab_meeting/Journal_figures/m87_new/m87_corner_comp.pdf')



fig = corner.corner(postsamples_rej3, bins=100, quantiles=[0.16, 0.84], 
			labels=labels, max_n_ticks=2)
# corner.corner(postsamples, bins=50, quantiles=[0.16, 0.84], labels=labels)

postsamples = np.load('postsamples-1_norej_loop9.npy')
postsamples_rej = np.load('postsamples-1_rej_loop9.npy')
postsamples_rej2 = np.load('postsamples-1_rej2_loop9.npy')

labels = ['$d$', '$w$', '$a$', '$\theta_c$', '$V_c$', '$V_F$', 
		'$V_{g, 1}$', '$\delta x_{1}$', '$\delta y_{1}$', '$w_{x, 1}$', '$w_{y, 1}$', '$\theta_{g, 1}$']
ranges = [[41, 44], [2, 12], [0.6, 0.9], [-126, -110], [0.25, 1.25], [0, 0.05], 
			[0, 2], [-20, 35], [-20, 12], [0, 100], [10, 50], [30, 60]]

# corner.corner(postsamples, bins=50, quantiles=[0.16, 0.84], labels=labels, range=ranges)
corner.corner(postsamples, bins=50, quantiles=[0.16, 0.84], labels=labels)


postsamples = np.load('postsamples-1_norej_loop9.npy')
postsamples_rej = np.load('postsamples-1_rej_loop9.npy')
postsamples_rej2 = np.load('postsamples-1_rej2_loop9.npy')

labels = ['$d$', '$w$', '$a$', '$\theta_c$', '$V_c$', '$V_F$', 
		'$V_{g, 1}$', '$\delta x_{1}$', '$\delta y_{1}$', '$w_{x, 1}$', '$w_{y, 1}$', '$\theta_{g, 1}$',
		'$V_{g, 2}$', '$\delta x_{2}$', '$\delta y_{2}$', '$w_{x, 2}$', '$w_{y, 2}$', '$\theta_{g, 2}$',
		'$V_{g, 3}$', '$\delta x_{3}$', '$\delta y_{3}$', '$w_{x, 3}$', '$w_{y, 3}$', '$\theta_{g, 3}$']
ranges = [[41, 44], [2, 12], [0.6, 0.9], [-126, -110], [0.25, 1.25], [0, 0.05], 
			[0, 2], [-20, 35], [-20, 12], [0, 100], [10, 50], [30, 60]]

# corner.corner(postsamples, bins=50, quantiles=[0.16, 0.84], labels=labels, range=ranges)
corner.corner(postsamples, bins=50, quantiles=[0.16, 0.84], labels=labels)


###
#hi
###
postsamples = np.load('postsamples-1_norej_loop9.npy')
postsamples_rej = np.load('postsamples-1_rej_loop9.npy')
postsamples_rej2 = np.load('postsamples-1_rej2_loop9.npy')

postsamples[:, -1] = postsamples[:, -1]%90

labels = ['$d$', '$w$', '$a$', '$\theta_c$', '$V_c$', '$V_F$', 
		'$V_{g, 1}$', '$\delta x_{1}$', '$\delta y_{1}$', '$w_{x, 1}$', '$w_{y, 1}$', '$\theta_{g, 1}$',
		'$V_{g, 2}$', '$\delta x_{2}$', '$\delta y_{2}$', '$w_{x, 2}$', '$w_{y, 2}$', '$\theta_{g, 2}$']
ranges = [[41, 46], [8, 16], [0.5, 0.8], [-130, -115], [0.25, 2.0], [0, 0.05], 
			[0, 2], [-100, 100], [-20, 20], [0, 50], [50, 100], [0, 90], 
			[0, 2], [-100, 100], [-20, 20], [0, 50], [50, 100], [0, 90]]

corner.corner(postsamples, bins=50, quantiles=[0.16, 0.84], labels=labels, range=ranges)






#######################################################################################
###################################################
## Fig. 1
###################################################
###
#lo
###
# postsamples = np.load('postsamples-1_importance_loop9.npy')

postsamples = np.load('postsamples_importance_loop9.npy')

labels = ['$d$', '$w$', '$a$', '$\theta_c$', '$V_c$', '$V_F$', 
		'$V_{g, 1}$', '$\delta x_{1}$', '$\delta y_{1}$', '$w_{x, 1}$', '$w_{y, 1}$', '$\theta_{g, 1}$',
		'$V_{g, 2}$', '$\delta x_{2}$', '$\delta y_{2}$', '$w_{x, 2}$', '$w_{y, 2}$', '$\theta_{g, 2}$']
ranges = [[41, 44], [2, 12], [0.6, 0.9], [-126, -110], [0.25, 1.25], [0, 0.05], 
			[0, 2], [-20, 35], [-20, 12], [0, 100], [10, 50], [30, 60], 
			[0, 2], [-20, 35], [-20, 12], [0, 100], [10, 50], [30, 60]]

corner.corner(postsamples[:, 0:18], bins=50, quantiles=[0.16, 0.84], labels=labels, range=ranges)
corner.corner(postsamples[:, 0:18], bins=50, quantiles=[0.16, 0.84], weights=postsamples[:, 18], labels=labels, range=ranges)

# corner.corner(postsamples, bins=50, quantiles=[0.16, 0.84], labels=labels)

postsamples = np.load('postsamples-1_norej_loop9.npy')
postsamples_rej = np.load('postsamples-1_rej_loop9.npy')
postsamples_rej2 = np.load('postsamples-1_rej2_loop9.npy')

labels = ['$d$', '$w$', '$a$', '$\theta_c$', '$V_c$', '$V_F$', 
		'$V_{g, 1}$', '$\delta x_{1}$', '$\delta y_{1}$', '$w_{x, 1}$', '$w_{y, 1}$', '$\theta_{g, 1}$']
ranges = [[41, 44], [2, 12], [0.6, 0.9], [-126, -110], [0.25, 1.25], [0, 0.05], 
			[0, 2], [-20, 35], [-20, 12], [0, 100], [10, 50], [30, 60]]

# corner.corner(postsamples, bins=50, quantiles=[0.16, 0.84], labels=labels, range=ranges)
corner.corner(postsamples, bins=50, quantiles=[0.16, 0.84], labels=labels)


postsamples = np.load('postsamples-1_norej_loop9.npy')
postsamples_rej = np.load('postsamples-1_rej_loop9.npy')
postsamples_rej2 = np.load('postsamples-1_rej2_loop9.npy')

labels = ['$d$', '$w$', '$a$', '$\theta_c$', '$V_c$', '$V_F$', 
		'$V_{g, 1}$', '$\delta x_{1}$', '$\delta y_{1}$', '$w_{x, 1}$', '$w_{y, 1}$', '$\theta_{g, 1}$',
		'$V_{g, 2}$', '$\delta x_{2}$', '$\delta y_{2}$', '$w_{x, 2}$', '$w_{y, 2}$', '$\theta_{g, 2}$',
		'$V_{g, 3}$', '$\delta x_{3}$', '$\delta y_{3}$', '$w_{x, 3}$', '$w_{y, 3}$', '$\theta_{g, 3}$']
ranges = [[41, 44], [2, 12], [0.6, 0.9], [-126, -110], [0.25, 1.25], [0, 0.05], 
			[0, 2], [-20, 35], [-20, 12], [0, 100], [10, 50], [30, 60]]

# corner.corner(postsamples, bins=50, quantiles=[0.16, 0.84], labels=labels, range=ranges)
corner.corner(postsamples, bins=50, quantiles=[0.16, 0.84], labels=labels)


###
#hi
###
postsamples = np.load('postsamples-1_norej_loop9.npy')
postsamples_rej = np.load('postsamples-1_rej_loop9.npy')
postsamples_rej2 = np.load('postsamples-1_rej2_loop9.npy')

postsamples[:, -1] = postsamples[:, -1]%90

labels = ['$d$', '$w$', '$a$', '$\theta_c$', '$V_c$', '$V_F$', 
		'$V_{g, 1}$', '$\delta x_{1}$', '$\delta y_{1}$', '$w_{x, 1}$', '$w_{y, 1}$', '$\theta_{g, 1}$',
		'$V_{g, 2}$', '$\delta x_{2}$', '$\delta y_{2}$', '$w_{x, 2}$', '$w_{y, 2}$', '$\theta_{g, 2}$']
ranges = [[41, 46], [8, 16], [0.5, 0.8], [-130, -115], [0.25, 2.0], [0, 0.05], 
			[0, 2], [-100, 100], [-20, 20], [0, 50], [50, 100], [0, 90], 
			[0, 2], [-100, 100], [-20, 20], [0, 50], [50, 100], [0, 90]]

corner.corner(postsamples, bins=50, quantiles=[0.16, 0.84], labels=labels, range=ranges)




corner.corner(postsamples, bins=50, quantiles=[0.16, 0.84], labels=labels)




###################################################
## Fig. 1 - synthetic
###################################################
###
#lo
###
# postsamples = np.load('postsamples-1_norej_loop9.npy')
# postsamples_rej = np.load('postsamples-1_rej_loop9.npy')
# postsamples_rej2 = np.load('postsamples-1_rej2_loop9.npy')
# postsamples_rej3 = np.load('postsamples-1_rej3.npy')


postsamples = np.load('postsamples-1_importance.npy')
# postsamples[:, 0] = 0.5 * postsamples[:, 0] + np.sqrt(0.25 * postsamples[:, 0]**2 + 0.25 * postsamples[:, 1]**2)
# postsamples[:, 1] = postsamples[:, 1] * np.sqrt(2 * np.log(2))
# postsamples[:, 3] = (postsamples[:, 3] - 90)%360

labels = ['$d$', '$w$', '$a$', r'$\theta_c$', '$V_F$', 
		'$V_{g, 1}$', '$\delta x_{1}$', '$\delta y_{1}$', '$w_{x, 1}$', '$w_{y, 1}$', r'$\theta_{g, 1}$',
		'$V_{g, 2}$', '$\delta x_{2}$', '$\delta y_{2}$', '$w_{x, 2}$', '$w_{y, 2}$', r'$\theta_{g, 2}$']


truth = model_params.squeeze()[np.concatenate([np.arange(4), np.arange(5, 18)])]

truth[4] /= model_params.squeeze()[4]
truth[5] /= model_params.squeeze()[4]
truth[11] /= model_params.squeeze()[4]


resamp_ind = np.random.choice(np.arange(postsamples.shape[0]), size=819200, replace=True, p=postsamples[:, 17]/np.sum(postsamples[:, 17]))
# fig = corner.corner(postsamples[resamp_ind, 0:18], bins=60, quantiles=[0.16, 0.84], 
# 			labels=labels, range=ranges, max_n_ticks=1)
fig = corner.corner(postsamples[resamp_ind, 0:17], bins=60, quantiles=[0.16, 0.84], 
			labels=labels, max_n_ticks=1, truths=truth.squeeze())

for ax in fig.get_axes():
	ax.tick_params(axis='y', rotation=0)
	ax.tick_params(axis='x', rotation=90)
	ax.tick_params(axis='both', labelsize=20)
	ax.xaxis.get_label().set_fontsize(24)
	ax.yaxis.get_label().set_fontsize(24)

# fig.set_size_inches(16, 16)
plt.savefig('/home/groot/BoumanLab/Lab_meeting/Journal_figures/m87_new/m87_corner_SIR2.pdf')



labels = ['$d$', '$w$', '$a$', '$\theta_c$', '$V_c$', '$V_F$', 
		'$V_{g, 1}$', '$\delta x_{1}$', '$\delta y_{1}$', '$w_{x, 1}$', '$w_{y, 1}$', '$\theta_{g, 1}$',
		'$V_{g, 2}$', '$\delta x_{2}$', '$\delta y_{2}$', '$w_{x, 2}$', '$w_{y, 2}$', '$\theta_{g, 2}$',
		'$V_{g, 3}$', '$\delta x_{3}$', '$\delta y_{3}$', '$w_{x, 3}$', '$w_{y, 3}$', '$\theta_{g, 3}$']


resamp_ind = np.random.choice(np.arange(postsamples.shape[0]), size=819200, replace=True, p=postsamples[:, 18]/np.sum(postsamples[:, 18]))
# fig = corner.corner(postsamples[resamp_ind, 0:18], bins=60, quantiles=[0.16, 0.84], 
# 			labels=labels, range=ranges, max_n_ticks=1)
fig = corner.corner(postsamples[resamp_ind, 0:18], bins=60, quantiles=[0.16, 0.84], 
			labels=labels, max_n_ticks=1, truths=truth.squeeze())


resamp_ind = np.random.choice(np.arange(postsamples.shape[0]), size=819200, replace=True, p=postsamples[:, 18]/np.sum(postsamples[:, 18]))
# fig = corner.corner(postsamples[resamp_ind, 0:18], bins=60, quantiles=[0.16, 0.84], 
# 			labels=labels, range=ranges, max_n_ticks=1)
fig = corner.corner(postsamples[resamp_ind, 0:18], bins=60, quantiles=[0.16, 0.84], 
			labels=labels, max_n_ticks=1, truths=truth.squeeze())



postsamples = np.load('postsamples-1_norej_loop5.0.npy')
# postsamples = np.load('postsamples-1_importance_loop2.0.npy')

labels = ['$d$', '$w$', '$a$', r'$\theta_c$', '$V_F$', 
		'$V_{g, 1}$', '$\delta x_{1}$', '$\delta y_{1}$', '$w_{x, 1}$', '$w_{y, 1}$', r'$\theta_{g, 1}$',
		'$V_{g, 2}$', '$\delta x_{2}$', '$\delta y_{2}$', '$w_{x, 2}$', '$w_{y, 2}$', r'$\theta_{g, 2}$']

fig = corner.corner(postsamples[:, 0:17], bins=60, quantiles=[0.16, 0.84], 
		labels=labels, max_n_ticks=1, truths=truth.squeeze())



postsamples = np.load('postsamples-1_importance_loop8.0.npy')


labels = ['$d$', '$w$', '$a$', r'$\theta_c$', '$V_F$', 
		'$V_{g, 1}$', '$\delta x_{1}$', '$\delta y_{1}$', '$w_{x, 1}$', '$w_{y, 1}$', r'$\theta_{g, 1}$',
		'$V_{g, 2}$', '$\delta x_{2}$', '$\delta y_{2}$', '$w_{x, 2}$', '$w_{y, 2}$', r'$\theta_{g, 2}$']

resamp_ind = np.random.choice(np.arange(postsamples.shape[0]), size=204800, replace=True, p=postsamples[:, 17]/np.sum(postsamples[:, 17]))

fig = corner.corner(postsamples[resamp_ind, 0:17], bins=60, quantiles=[0.16, 0.84], 
			labels=labels, max_n_ticks=1, truths=truth.squeeze())




model_params = np.load('/home/groot/BoumanLab/DPI/dataset/interferometry_m87/synthetic3/model_params.npy')
truth = model_params.squeeze()[np.concatenate([np.arange(4), np.arange(5, 18)])]
truth[0] = 0.5*model_params.squeeze()[0] + 0.5*np.sqrt(model_params.squeeze()[0]**2+model_params.squeeze()[1]**2)
truth[1] = model_params.squeeze()[1] * np.sqrt(2*np.log(2))
truth[3] = (model_params.squeeze()[3] - 90)%360
truth[5] /= model_params.squeeze()[4]
truth[11] /= model_params.squeeze()[4]

# truth[8] = truth[9]
# truth[9] = truth[10]

# truth[14] = truth[15]
# truth[15] = truth[16]

truth = np.concatenate([truth[0:5], truth[11:17], truth[5:11]])


import pandas as pd
chains = pd.read_csv('chain_equal.csv')
postsamples = chains[['diam', 'fwhm', 'ma1', 'mp1', 'f', 'floor', 
					 'fg1', 'xg1', 'yg1', 'σxg1', 'σyg1', 'ξg1',
					'fg2', 'xg2', 'yg2', 'σxg2', 'σyg2', 'ξg2']]

postsamples = np.array(postsamples)

postsamples[:, 2] = 2*postsamples[:, 2]
postsamples[:, 3] = 360-(180/np.pi * postsamples[:, 3])%360


postsamples[:, 11] = 180/np.pi * postsamples[:, 11]


postsamples[:, 17] = 180/np.pi * postsamples[:, 17]

postsamples[:, 6] = postsamples[:, 6] / postsamples[:, 4]
postsamples[:, 12] = postsamples[:, 12] / postsamples[:, 4]


postsamples = postsamples[:, np.concatenate([np.arange(4), np.arange(5, 18)])]



# labels = ['$d$', '$w$', '$a$', '$\theta_c$', '$V_c$', '$V_F$', 
# 		'$V_{g, 1}$', '$\delta x_{1}$', '$\delta y_{1}$', '$w_{x, 1}$', '$w_{y, 1}$', '$\theta_{g, 1}$',
# 		'$V_{g, 2}$', '$\delta x_{2}$', '$\delta y_{2}$', '$w_{x, 2}$', '$w_{y, 2}$', '$\theta_{g, 2}$']

# fig = corner.corner(postsamples[:, 0:18], bins=60, quantiles=[0.16, 0.84], 
# 			labels=labels, max_n_ticks=1)


labels = ['$d$', '$w$', '$a$', '$\theta_c$', '$V_F$', 
		'$V_{g, 1}$', '$\delta x_{1}$', '$\delta y_{1}$', '$w_{x, 1}$', '$w_{y, 1}$', '$\theta_{g, 1}$',
		'$V_{g, 2}$', '$\delta x_{2}$', '$\delta y_{2}$', '$w_{x, 2}$', '$w_{y, 2}$', '$\theta_{g, 2}$']

fig = corner.corner(postsamples, bins=60, quantiles=[0.16, 0.84], 
			labels=labels, max_n_ticks=1, truths=truth.squeeze())


im = eh.image.load_fits('/home/groot/BoumanLab/DPI/dataset/interferometry_m87/synthetic3/groundtruth.fits')



postsamples = np.load('postsamples-1_importance_loop4.0.npy')
postsamples[:, 3] = (postsamples[:, 3] - 90)%360
postsamples[:, 0] = 0.5*postsamples[:, 0] + 0.5*np.sqrt(postsamples[:, 0]**2+postsamples[:, 1]**2)
postsamples[:, 1] = postsamples[:, 1] * np.sqrt(2*np.log(2))

# labels = ['$d$', '$w$', '$a$', r'$\theta_c$', '$V_F$', 
# 		'$V_{g, 1}$', '$\delta x_{1}$', '$\delta y_{1}$', '$w_{x, 1}$', '$w_{y, 1}$', r'$\theta_{g, 1}$',
# 		'$V_{g, 2}$', '$\delta x_{2}$', '$\delta y_{2}$', '$w_{x, 2}$', '$w_{y, 2}$', r'$\theta_{g, 2}$']

# resamp_ind = np.random.choice(np.arange(postsamples.shape[0]), size=20480, replace=True, p=postsamples[:, 17]/np.sum(postsamples[:, 17]))

# fig = corner.corner(postsamples[resamp_ind, 0:17], bins=60, quantiles=[0.16, 0.84], 
# 			labels=labels, max_n_ticks=1, truths=truth.squeeze())

# resamp_ind = np.random.choice(np.arange(postsamples.shape[0]), size=20480, replace=True, p=postsamples[:, 17]/np.sum(postsamples[:, 17]))

fig = corner.corner(postsamples[0:20480, 0:17], bins=60, quantiles=[0.16, 0.84], 
			labels=labels, max_n_ticks=1, truths=truth.squeeze())



postsamples = np.load('postsamples-1_importance_loop4.0.npy')

postsamples[:, 3] = (postsamples[:, 3] - 90)%360
postsamples[:, 0] = 0.5*postsamples[:, 0] + 0.5*np.sqrt(postsamples[:, 0]**2+postsamples[:, 1]**2)
postsamples[:, 1] = postsamples[:, 1] * np.sqrt(2*np.log(2))

postsamples[:, 6] = postsamples[:, 6] / postsamples[:, 4]
postsamples[:, 12] = postsamples[:, 12] / postsamples[:, 4]

postsamples = postsamples[:, np.concatenate([np.arange(4), np.arange(5, 19)])]

postsamples = postsamples[:, np.concatenate([np.arange(5), np.arange(11, 17), np.arange(5, 11), np.arange(17, 18)])]


# labels = ['$d$', '$w$', '$a$', r'$\theta_c$', '$V_F$', 
# 		'$V_{g, 1}$', '$\delta x_{1}$', '$\delta y_{1}$', '$w_{x, 1}$', '$w_{y, 1}$', r'$\theta_{g, 1}$',
# 		'$V_{g, 2}$', '$\delta x_{2}$', '$\delta y_{2}$', '$w_{x, 2}$', '$w_{y, 2}$', r'$\theta_{g, 2}$']

# resamp_ind = np.random.choice(np.arange(postsamples.shape[0]), size=20480, replace=True, p=postsamples[:, 17]/np.sum(postsamples[:, 17]))

# fig = corner.corner(postsamples[resamp_ind, 0:17], bins=60, quantiles=[0.16, 0.84], 
# 			labels=labels, max_n_ticks=1, truths=truth.squeeze())

# resamp_ind = np.random.choice(np.arange(postsamples.shape[0]), size=20480, replace=True, p=postsamples[:, 17]/np.sum(postsamples[:, 17]))

fig = corner.corner(postsamples[0:20480, 0:17], bins=60, quantiles=[0.16, 0.84], 
			labels=labels, max_n_ticks=1, truths=truth.squeeze())


###################################################
## Fig. 4 - synthetic
###################################################

sample_path = '/home/groot/BoumanLab/DPI/DPItorch/checkpoint/interferometry_m87_mcfe/lo/3598_processed_sc_rough/cresentfloorgaussian2/alpha095cphaselogcamp3/'
postsamples = np.load(sample_path+'postsamples-1_rej_alpha0.95.npy')
importance_weights = softmax(postsamples[:, -1].squeeze()-np.max(postsamples[:, -1]))
sample_indices = np.random.choice(np.arange(postsamples.shape[0]), size=postsamples.shape[0], replace=True, p=importance_weights)

postsamples[sample_indices, 0] = 0.5 * postsamples[sample_indices, 0] + 0.5 * np.sqrt(postsamples[sample_indices, 0]**2 + postsamples[sample_indices, 1]**2)
postsamples[sample_indices, 3] = (postsamples[sample_indices, 3]-90)%360

# labels = ['$d$', '$w$', '$a$', r'$\theta_c$', '$V_c$', '$V_F$', 
# 		'$V_{g, 1}$', '$\delta x_{1}$', '$\delta y_{1}$', '$w_{x, 1}$', '$w_{y, 1}$', r'$\theta_{g, 1}$',
# 		'$V_{g, 2}$', '$\delta x_{2}$', '$\delta y_{2}$', '$w_{x, 2}$', '$w_{y, 2}$', r'$\theta_{g, 2}$']

labels = ['$d$', '$w$', '$a$', r'$\theta_c$', '$V_c$', '$V_d$', 
		'$V_{g, 1}$', '$\delta x_{1}$', '$\delta y_{1}$', '$w_{x, 1}$', '$w_{y, 1}$', r'$\theta_{g, 1}$',
		'$V_{g, 2}$', '$\delta x_{2}$', '$\delta y_{2}$', '$w_{x, 2}$', '$w_{y, 2}$', r'$\theta_{g, 2}$']



# save_path = '/home/groot/BoumanLab/Lab_meeting/Journal_figures/m87_new/'
save_path = '/home/groot/BoumanLab/Lab_meeting/Journal_figures/m87_new/corner/'

# fig = corner.corner(postsamples[sample_indices, 0:6], bins=60, quantiles=[0.16, 0.84], 
# 			labels=labels[0:6], label_kwargs={"fontsize": 40}, max_n_ticks=2,
# 			range=[[42, 44], [2, 8], [0.6, 0.9], [145, 160], [0, 2], [0, 0.1]])
fig = corner.corner(postsamples[sample_indices, 0:6], bins=60, quantiles=[0.16, 0.84], max_n_ticks=3,
			range=[[42, 44.5], [2, 8.5], [0.6, 0.95], [145, 162], [0, 2], [0, 0.1]])
count = 0
for ax in fig.get_axes(): 
	if count in [0, 7, 14, 21, 28, 35]:
		ax.annotate(labels[count//7], xy=(0.8, 0.7), size=42, xycoords='axes fraction', ha='center')
	count += 1
	# ax.tick_params(axis='x', labelsize=28, rotation=30)
	# ax.tick_params(axis='y', labelsize=28, rotation=60)
	ax.tick_params(axis='x', labelsize=28, rotation=45)
	ax.tick_params(axis='y', labelsize=28, rotation=0)

fig.savefig(save_path+'crescent.pdf',bbox_inches='tight')

fig = corner.corner(postsamples[sample_indices, 6:12], bins=60, quantiles=[0.16, 0.84], max_n_ticks=3,
			range=[[0, 2], [-35, 35], [-25, 0], [0, 20], [0, 50], [0, 90]])

count = 0
for ax in fig.get_axes(): 
	# ax.tick_params(axis='both', labelsize=18)
	if count in [0, 7, 14, 21, 28, 35]:
		ax.annotate(labels[6+count//7], xy=(0.7, 0.7), size=42, xycoords='axes fraction', ha='center')
	count += 1
	# ax.tick_params(axis='x', labelsize=28, rotation=30)
	# ax.tick_params(axis='y', labelsize=28, rotation=60)
	ax.tick_params(axis='x', labelsize=28, rotation=45)
	ax.tick_params(axis='y', labelsize=28, rotation=0)


fig.savefig(save_path+'gaussian1.pdf',bbox_inches='tight')

# fig = corner.corner(postsamples[sample_indices, 12:18], bins=60, quantiles=[0.16, 0.84], 
# 			labels=labels[12:18], label_kwargs={"fontsize": 24}, max_n_ticks=1)
# for ax in fig.get_axes(): 
# 	ax.tick_params(axis='both', labelsize=18)

fig = corner.corner(postsamples[sample_indices, 12:18], bins=60, quantiles=[0.16, 0.84], max_n_ticks=3,
			range=[[0, 2], [-35, 35], [-25, 0], [0, 20], [0, 50], [0, 90]])

count = 0
for ax in fig.get_axes(): 
	# ax.tick_params(axis='both', labelsize=18)
	if count in [0, 7, 14, 21, 28, 35]:
		ax.annotate(labels[12+count//7], xy=(0.7, 0.7), size=42, xycoords='axes fraction', ha='center')
	count += 1
	# ax.tick_params(axis='x', labelsize=28, rotation=30)
	# ax.tick_params(axis='y', labelsize=28, rotation=60)
	ax.tick_params(axis='x', labelsize=28, rotation=45)
	ax.tick_params(axis='y', labelsize=28, rotation=0)

fig.savefig(save_path+'gaussian2.pdf',bbox_inches='tight')



sample_path = '/home/groot/BoumanLab/DPI/DPItorch/checkpoint/interferometry_m87_mcfe/synthetic3598/crescentfloorgaussian2new4/cresentfloorgaussian2/alpha095cphaselogcamp3/'
postsamples = np.load(sample_path+'postsamples-1_rej_alpha0.95.npy')
importance_weights = softmax(postsamples[:, -1].squeeze()-np.max(postsamples[:, -1]))
sample_indices = np.random.choice(np.arange(postsamples.shape[0]), size=postsamples.shape[0], replace=True, p=importance_weights)

postsamples[sample_indices, 0] = 0.5 * postsamples[sample_indices, 0] + 0.5 * np.sqrt(postsamples[sample_indices, 0]**2 + postsamples[sample_indices, 1]**2)
postsamples[sample_indices, 3] = (postsamples[sample_indices, 3]-90)%360

# labels = ['$d$', '$w$', '$a$', r'$\theta_c$', '$V_c$', '$V_F$', 
# 		'$V_{g, 1}$', '$\delta x_{1}$', '$\delta y_{1}$', '$w_{x, 1}$', '$w_{y, 1}$', r'$\theta_{g, 1}$',
# 		'$V_{g, 2}$', '$\delta x_{2}$', '$\delta y_{2}$', '$w_{x, 2}$', '$w_{y, 2}$', r'$\theta_{g, 2}$']

obs_path = '/home/groot/BoumanLab/DPI/dataset/interferometry_m87/synthetic_crescentfloorgaussian2new4/'
truth_params = np.load(obs_path+'model_params.npy')
truth_params = truth_params.squeeze()
truth_params[0] = 0.5 * truth_params[0] + 0.5 * np.sqrt(truth_params[0]**2 + truth_params[1]**2)
truth_params[3] = (truth_params[3]-90)%360


save_path = '/home/groot/BoumanLab/Lab_meeting/Journal_figures/m87_new/corner/'
# fig = corner.corner(postsamples[sample_indices, 0:6], bins=60, quantiles=[0.16, 0.84], 
# 			labels=labels[0:6], label_kwargs={"fontsize": 24}, max_n_ticks=2, truths=truth_params[0:6])
# for ax in fig.get_axes(): 
# 	ax.tick_params(axis='both', labelsize=18)

fig = corner.corner(postsamples[sample_indices, 0:6], bins=60, quantiles=[0.16, 0.84], max_n_ticks=3,
			range=[[43, 49], [3, 14], [0.4, 0.8], [172, 188], [0, 2], [0, 0.32]], truths=truth_params[0:6])
count = 0
for ax in fig.get_axes(): 
	if count in [0, 7, 14, 21, 28, 35]:
		ax.annotate(labels[count//7], xy=(0.8, 0.7), size=42, xycoords='axes fraction', ha='center')
	count += 1
	# ax.tick_params(axis='x', labelsize=28, rotation=30)
	# ax.tick_params(axis='y', labelsize=28, rotation=60)
	ax.tick_params(axis='x', labelsize=28, rotation=45)
	ax.tick_params(axis='y', labelsize=28, rotation=0)

fig.savefig(save_path+'syntheticcrescent.pdf',bbox_inches='tight')

# fig = corner.corner(postsamples[sample_indices, 6:12], bins=60, quantiles=[0.16, 0.84], 
# 			labels=labels[6:12], label_kwargs={"fontsize": 24}, max_n_ticks=1, truths=truth_params[6:12])
# for ax in fig.get_axes(): 
# 	ax.tick_params(axis='both', labelsize=18)

fig = corner.corner(postsamples[sample_indices, 6:12], bins=60, quantiles=[0.16, 0.84], max_n_ticks=3,
			range=[[0, 0.55], [-50, 50], [-45, -35], [0, 20], [0, 25], [30, 70]], truths=truth_params[6:12])
count = 0
for ax in fig.get_axes(): 
	# ax.tick_params(axis='both', labelsize=18)
	if count in [0, 7, 14, 21, 28, 35]:
		ax.annotate(labels[6+count//7], xy=(0.7, 0.7), size=42, xycoords='axes fraction', ha='center')
	count += 1
	# ax.tick_params(axis='x', labelsize=28, rotation=30)
	# ax.tick_params(axis='y', labelsize=28, rotation=60)
	ax.tick_params(axis='x', labelsize=28, rotation=45)
	ax.tick_params(axis='y', labelsize=28, rotation=0)

fig.savefig(save_path+'syntheticgaussian1.pdf',bbox_inches='tight')

# fig = corner.corner(postsamples[sample_indices, 12:18], bins=60, quantiles=[0.16, 0.84], 
# 			labels=labels[12:18], label_kwargs={"fontsize": 24}, max_n_ticks=1, truths=truth_params[12:18])
# for ax in fig.get_axes(): 
# 	ax.tick_params(axis='both', labelsize=18)

fig = corner.corner(postsamples[sample_indices, 12:18], bins=60, quantiles=[0.16, 0.84], max_n_ticks=3,
			range=[[0, 0.55], [-50, 50], [-45, -35], [0, 20], [0, 25], [30, 70]], truths=truth_params[12:18])
count = 0
for ax in fig.get_axes(): 
	# ax.tick_params(axis='both', labelsize=18)
	if count in [0, 7, 14, 21, 28, 35]:
		ax.annotate(labels[6+count//7], xy=(0.7, 0.7), size=42, xycoords='axes fraction', ha='center')
	count += 1
	# ax.tick_params(axis='x', labelsize=28, rotation=30)
	# ax.tick_params(axis='y', labelsize=28, rotation=60)
	ax.tick_params(axis='x', labelsize=28, rotation=45)
	ax.tick_params(axis='y', labelsize=28, rotation=0)
fig.savefig(save_path+'syntheticgaussian2.pdf',bbox_inches='tight')





###################################################
## Fig. 5 - synthetic
###################################################

evidence_list = []
evidence_list2 = []
img_list = []
time_list = []

for k in range(5):
	path = '/home/groot/BoumanLab/DPI/DPItorch/checkpoint/interferometry_m87_mcfe/lo/3598_processed_sc_rough'
	# path = '/home/groot/BoumanLab/DPI/DPItorch/checkpoint/interferometry_sgra_validation_new/ehtim_closure'
	img1 = np.load(path+'/cresentfloorgaussian{}/alpha095cphaselogcamp/img_map-1_alpha0.95.npy'.format(k))

	loss1 = np.load(path+'/cresentfloorgaussian{}/alpha095cphaselogcamp/loss-1_simple_crescent_floor_nuisance_res32flow48logdet1.0_cphase_logcamp.npy'.format(k), allow_pickle=True)

	evidence_list.append(140*np.min(loss1.item()['total'][12000::]))
	evidence_list2.append(np.min(loss1.item()['evidence'][12000::]))
	img_list.append(img1.squeeze())
	time_list.append(loss1.item()['time'])

# plt.plot(6 * np.arange(1, 6), evidence_list, 'b-s', markersize=12)
# plt.plot(np.arange(5), evidence_list, 'b-s', markersize=12)
# plt.plot(np.arange(5), evidence_list2[0:5], 'b-s', markersize=12)

plt.plot(np.arange(4), evidence_list2[0:4], 'b-s', markersize=12)



plt.legend(['synthetic', 'M87'], fontsize=20)
plt.savefig(save_path+'KL_divergenceloss2.pdf',bbox_inches='tight')
# plt.savefig(save_path+'alpha_divergenceloss2.pdf',bbox_inches='tight')

# plt.savefig(save_path+'divergenceloss_new.pdf',bbox_inches='tight')





evidence_list = []
evidence_list2 = []
img_list = []
time_list = []




for k in range(5):
	path = '/home/groot/BoumanLab/DPI/DPItorch/checkpoint/interferometry_m87_mcfe/synthetic3598/crescentfloorgaussian2new4'
	# path = '/home/groot/BoumanLab/DPI/DPItorch/checkpoint/interferometry_sgra_validation_new/ehtim_closure'
	img1 = np.load(path+'/cresentfloorgaussian{}/alpha095cphaselogcamp/img_mean-1_alpha0.95.npy'.format(k))

	loss1 = np.load(path+'/cresentfloorgaussian{}/alpha095cphaselogcamp/loss-1_simple_crescent_floor_nuisance_res32flow48logdet1.0_cphase_logcamp.npy'.format(k), allow_pickle=True)

	evidence_list.append(140*np.min(loss1.item()['total']))
	evidence_list2.append(np.min(loss1.item()['evidence']))
	img_list.append(img1.squeeze())
	time_list.append(loss1.item()['time'])


# plt.figure(), plt.plot(6 * np.arange(1, 6), evidence_list, 'r-^', markersize=12)
# plt.figure(), plt.plot(np.arange(5), evidence_list2, 'r--^', markersize=12)
# plt.figure(figsize=(8, 3)), plt.plot(np.arange(5), evidence_list2[0:5], 'r-^', markersize=12)
# plt.figure(figsize=(8, 3)), plt.plot(np.arange(5), evidence_list[0:5], 'r-^', markersize=12)
plt.figure(figsize=(8, 3)), plt.plot(np.arange(4), evidence_list2[0:4], 'r-^', markersize=12)


plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.xlabel('geometric model dimensions', fontsize=16)
# plt.xlabel('number of additional Gaussian blobs', fontsize=16)
# plt.ylabel(r'$\alpha$-divergence loss', fontsize=20))
plt.ylabel(r'KL divergence loss', fontsize=20)

# plt.xlim([5, 31])

# my_xticks = ['zero\nGaussian\nblob','one\nGaussian\nblob','two\nGaussian\nblobs','three\nGaussian\nblobs','four\nGaussian\nblobs']
# plt.xticks(np.arange(4), my_xticks[0:4], rotation=45)
# my_xticks = ['Crescent\n(0 blob)','Crescent\n(1 blob)','Crescent\n(2 blobs)','Crescent\n(3 blobs)','Crescent\n(4 blobs)','32x32\nimage']
# plt.xticks(np.arange(5), my_xticks[0:5], rotation=45, fontsize=20)
my_xticks = ['crescent +\n0 ellipse\n(dim=6)','crescent +\n1 ellipse\n(dim=12)','crescent +\n2 ellipses\n(dim=18)','crescent +\n3 ellipses\n(dim=24)','crescent +\n4 ellipses\n(dim=30)','full image\n(dim=32x32)']
plt.xticks(np.arange(4), my_xticks[0:4], rotation=45, fontsize=20)




for k in range(5):
	simim.imvec = img_list[k].reshape((-1, ))
	# simim2 = simim.regrid_image(160*eh.RADPERUAS, 64)
	simim2 = simim.regrid_image(120*eh.RADPERUAS, 64)
	# simim2 = simim.copy()
	# simim2.display()
	simim2.display(has_title=False, cfun='afmhot_10us', label_type='scale', has_cbar=False, scale_fontsize=30, cbar_unit=('mJy', 'muas2')) 
	# plt.savefig(save_path+'synthetic_mapimg{}.pdf'.format(k),bbox_inches='tight')
	plt.savefig(save_path+'m87_mapimg{}_new.pdf'.format(k),bbox_inches='tight')





time_list = [5602.118191957474,5905.9359250068665, 6405.047237873077, 6920.59951043129, 7371.7205810546875, 14157.768675327301]
time_list_sampling = [4, 975.3, 4136.2, 141689.1]


# plt.figure(), plt.semilogy(6 * np.arange(1, 6), time_list, 'r-^', markersize=12)
# plt.figure(figsize=(8, 3)), plt.semilogy(np.arange(5), time_list[0:5], 'r-^', markersize=12)
plt.figure(figsize=(8, 3)), plt.semilogy(np.arange(4), time_list[0:4], 'r-^', markersize=12)

plt.semilogy(np.arange(4), time_list_sampling, 'b-o', markersize=12)


plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.xlabel('geometric model dimensions', fontsize=16)
plt.ylabel('computational\ntime (s)', fontsize=20)
# plt.xlim([5, 31])
# plt.semilogy(np.arange(4, 6), time_list[4:6], 'r--^', markersize=12)
plt.semilogy(np.arange(3, 5), [time_list[3], time_list[5]], 'r--^', markersize=12)


plt.legend([r'$\alpha$-DPI', 'nested sampling'], fontsize=20)
# my_xticks = ['Crescent\n(0 blob)','Crescent\n(1 blob)','Crescent\n(2 blobs)','Crescent\n(3 blobs)','Crescent\n(4 blobs)','32x32\nimage']
my_xticks = ['crescent +\n0 ellipse\n(dim=6)','crescent +\n1 ellipse\n(dim=12)','crescent +\n2 ellipses\n(dim=18)','crescent +\n3 ellipses\n(dim=24)','crescent +\n4 ellipses\n(dim=30)','full image\n(dim=32x32)']

# plt.xticks(np.arange(6), my_xticks[0:6], rotation=45)
plt.xticks(np.arange(5), np.array(my_xticks[0:4]+my_xticks[5:6]), rotation=45, fontsize=20)

plt.ylim([1, 1e6])

plt.savefig(save_path+'computationaltime_new2.pdf',bbox_inches='tight')


###################################################
## Fig. 5 - synthetic
###################################################
import ehtim as eh
import numpy as np
from scipy.special import softmax
from geometric_model import *
import ehtplot.color
import matplotlib.pyplot as plt
plt.ion()




obs = eh.obsdata.load_uvfits('/home/groot/BoumanLab/DPI/dataset/interferometry_m87/synthetic_crescentfloorgaussian2new4/obs_mring_synthdata_thermal_phase_only_scanavg_sysnoise2.uvfits')
obs.add_cphase(count='max')
obs.add_logcamp(count='max')
im_truth = eh.image.load_fits('/home/groot/BoumanLab/DPI/dataset/interferometry_m87/synthetic_crescentfloorgaussian2new4/ground_truth.fits')


npix = 96#32
fov = 120#160
n_gaussian = 3#2
im_placeholder = eh.image.make_square(obs, npix, fov*eh.RADPERUAS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flux_flag = False

flux_const = np.median(obs.unpack_bl('AP', 'AA', 'amp')['amp'])
flux_range = [0.8*flux_const, 1.2*flux_const]
r_range = [10.0, 50.0]
width_range = [1.0, 40.0]
floor_range = [1e-4, 0.999]#[0.0, 1.0]#
floor_r_range = [10.0, 50.0]
shift_range = [-80.0, 80.0]#[-160.0, 160.0]#[-200.0, 200.0]
sigma_range = [1.0, 50.0]#[1.0, 40.0]#[1.0, 80.0]#[1.0, 100.0]#


img_converter = SimpleCrescentNuisanceFloor_Param2Img(npix, r_range=r_range, fov=fov, n_gaussian=n_gaussian, shift_range=shift_range, width_range=width_range, floor_range=floor_range, sigma_range=sigma_range, flux_flag=flux_flag, flux_range=flux_range).to(device=device)


# sample_path = '/home/groot/BoumanLab/DPI/DPItorch/checkpoint/interferometry_m87_mcfe/synthetic3598/crescentfloorgaussian2new4/cresentfloorgaussian2/alpha095cphaselogcamp3/'
sample_path = '/home/groot/BoumanLab/DPI/DPItorch/checkpoint/interferometry_m87_mcfe/lo/3598_processed_sc_rough/cresentfloorgaussian{}/alpha095cphaselogcamp/'.format(n_gaussian)
# sample_path = '/home/groot/BoumanLab/DPI/DPItorch/checkpoint/interferometry_m87_mcfe/lo/3598_processed_sc_rough/cresentfloorgaussian2/alpha095cphaselogcamp3/'
postsamples = np.load(sample_path+'postsamples-1_rej_alpha0.95.npy')
importance_weights = softmax(postsamples[:, -1].squeeze())
sample_indices = np.random.choice(np.arange(postsamples.shape[0]), size=postsamples.shape[0], replace=True, p=importance_weights)
# sample_indices = np.argsort(importance_weights)[::-1]
postsamples_rs = postsamples[sample_indices[0:1024], 0:24]

prob_rs = importance_weights[sample_indices[0:1024]]

# ind_sort = np.argsort(prob_rs)[::-1]
# postsamples = postsamples[ind_sort]
# prob_rs = prob_rs[ind_sort]

# prob_rs = np.log(prob_rs)


r = torch.tensor(postsamples_rs[:, 0] / (2 * 0.5 * fov)).to(device).unsqueeze(-1).unsqueeze(-1)
sigma = torch.tensor(postsamples_rs[:, 1]  / (2 * 0.5 * fov)).to(device).unsqueeze(-1).unsqueeze(-1)
s = torch.tensor(postsamples_rs[:, 2]).to(device).unsqueeze(-1).unsqueeze(-1)
eta = torch.tensor(postsamples_rs[:, 3] * np.pi / 180).to(device).unsqueeze(-1).unsqueeze(-1)
crescent_flux = torch.tensor(postsamples_rs[:, 4]).to(device).unsqueeze(-1).unsqueeze(-1)
floor = torch.tensor(postsamples_rs[:, 5]).to(device).unsqueeze(-1).unsqueeze(-1)
if n_gaussian > 0:
	nuisance_scale1 = torch.tensor(postsamples_rs[:, 6]).to(device).unsqueeze(-1).unsqueeze(-1)
	nuisance_x1 = torch.tensor(postsamples_rs[:, 7] / (0.5 * fov)).to(device).unsqueeze(-1).unsqueeze(-1)
	nuisance_y1 = torch.tensor(postsamples_rs[:, 8] / (0.5 * fov)).to(device).unsqueeze(-1).unsqueeze(-1)
	sigma_x1 = torch.tensor(postsamples_rs[:, 9] / (0.5 * fov)).to(device).unsqueeze(-1).unsqueeze(-1)
	sigma_y1 = torch.tensor(postsamples_rs[:, 10] / (0.5 * fov)).to(device).unsqueeze(-1).unsqueeze(-1)
	rho1 = torch.tensor(postsamples_rs[:, 11] * np.pi / 180).to(device).unsqueeze(-1).unsqueeze(-1)

	if n_gaussian > 1:
		nuisance_scale2 = torch.tensor(postsamples_rs[:, 12]).to(device).unsqueeze(-1).unsqueeze(-1)
		nuisance_x2 = torch.tensor(postsamples_rs[:, 13] / (0.5 * fov)).to(device).unsqueeze(-1).unsqueeze(-1)
		nuisance_y2 = torch.tensor(postsamples_rs[:, 14] / (0.5 * fov)).to(device).unsqueeze(-1).unsqueeze(-1)
		sigma_x2 = torch.tensor(postsamples_rs[:, 15] / (0.5 * fov)).to(device).unsqueeze(-1).unsqueeze(-1)
		sigma_y2 = torch.tensor(postsamples_rs[:, 16] / (0.5 * fov)).to(device).unsqueeze(-1).unsqueeze(-1)
		rho2 = torch.tensor(postsamples_rs[:, 17] * np.pi / 180).to(device).unsqueeze(-1).unsqueeze(-1)

	if n_gaussian > 2:
		nuisance_scale3 = torch.tensor(postsamples_rs[:, 18]).to(device).unsqueeze(-1).unsqueeze(-1)
		nuisance_x3 = torch.tensor(postsamples_rs[:, 19] / (0.5 * fov)).to(device).unsqueeze(-1).unsqueeze(-1)
		nuisance_y3 = torch.tensor(postsamples_rs[:, 20] / (0.5 * fov)).to(device).unsqueeze(-1).unsqueeze(-1)
		sigma_x3 = torch.tensor(postsamples_rs[:, 21] / (0.5 * fov)).to(device).unsqueeze(-1).unsqueeze(-1)
		sigma_y3 = torch.tensor(postsamples_rs[:, 22] / (0.5 * fov)).to(device).unsqueeze(-1).unsqueeze(-1)
		rho3 = torch.tensor(postsamples_rs[:, 23] * np.pi / 180).to(device).unsqueeze(-1).unsqueeze(-1)

	if n_gaussian == 1:
		nuisance_scale = [nuisance_scale1]
		nuisance_x = [nuisance_x1]
		nuisance_y = [nuisance_y1]
		sigma_x_list = [sigma_x1]
		sigma_y_list = [sigma_y1]
		theta_list = [rho1]
	if n_gaussian == 2:
		nuisance_scale = [nuisance_scale1, nuisance_scale2]
		nuisance_x = [nuisance_x1, nuisance_x2]
		nuisance_y = [nuisance_y1, nuisance_y2]
		sigma_x_list = [sigma_x1, sigma_x2]
		sigma_y_list = [sigma_y1, sigma_y2]
		theta_list = [rho1, rho2]
	if n_gaussian == 3:
		nuisance_scale = [nuisance_scale1, nuisance_scale2, nuisance_scale3]
		nuisance_x = [nuisance_x1, nuisance_x2, nuisance_x3]
		nuisance_y = [nuisance_y1, nuisance_y2, nuisance_y3]
		sigma_x_list = [sigma_x1, sigma_x2, sigma_x3]
		sigma_y_list = [sigma_y1, sigma_y2, sigma_y3]
		theta_list = [rho1, rho2, rho3]



ring = torch.exp(- 0.5 * (img_converter.grid_r - r)**2 / (sigma)**2)
S = 1 + s * torch.cos(img_converter.grid_theta - eta)
crescent = S * ring
disk = 0.5 * (1 + torch.erf((r - img_converter.grid_r)/(np.sqrt(2)*sigma)))

crescent = crescent / (torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)+img_converter.eps)	
disk = disk / (torch.sum(disk, (-1, -2)).unsqueeze(-1).unsqueeze(-1)+img_converter.eps)
crescent = crescent_flux * ((1-floor) * crescent + floor * disk)
# crescent = crescent_flux * (crescent + floor * disk)

# crescent = crescent + disk * floor * (1-s)
# crescent = crescent_flux * crescent / (torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)+self.eps)


for k in range(img_converter.n_gaussian):
	x_c = img_converter.grid_x - nuisance_x[k]
	y_c = img_converter.grid_y - nuisance_y[k]
	x_rot = x_c * torch.cos(theta_list[k]) + y_c * torch.sin(theta_list[k])
	y_rot = -x_c * torch.sin(theta_list[k]) + y_c * torch.cos(theta_list[k])
	delta = 0.5 * (x_rot**2 / sigma_x_list[k]**2 + y_rot**2 / sigma_y_list[k]**2)
	# nuisance_now = nn.Softmax(dim=1)(-delta.reshape((-1, self.npix**2))).reshape((-1, self.npix, self.npix))#torch.exp(-delta)
	nuisance_now = 1 / (2 * np.pi * sigma_x_list[k] * sigma_y_list[k]) * torch.exp(-delta)
	nuisance_now = nuisance_now / (torch.sum(nuisance_now, (-1, -2)).unsqueeze(-1).unsqueeze(-1)+img_converter.eps)
	# nuisance_now = (2*self.gap)**2 * nuisance_now
	nuisance_now = nuisance_scale[k] * nuisance_now# / torch.sum(nuisance_now, (-1, -2)).unsqueeze(-1).unsqueeze(-1)
	crescent += nuisance_now


crescent = crescent / (torch.sum(crescent, (-1, -2)).unsqueeze(-1).unsqueeze(-1)+img_converter.eps)

im_recon = crescent.cpu().numpy()

im_placeholder.imvec = np.mean(im_recon, 0).reshape((-1, ))
im_placeholder.display(has_title=False, cfun='afmhot_10us', label_type='scale', has_cbar=False, scale_fontsize=30, cbar_unit=('mJy', 'muas2')) 
# plt.savefig(save_path+'synthetic_mapimg{}.pdf'.format(k),bbox_inches='tight')
save_path = '/home/groot/BoumanLab/Lab_meeting/Journal_figures/m87_new/'
plt.savefig(save_path+'m87_mapimg{}_highres.pdf'.format(im_placeholder),bbox_inches='tight')

plt.figure(5), plt.savefig(save_path+'m87_mapimg_highres.pdf',bbox_inches='tight')

syn = {}

n_samples = 16#32
for k in range(n_samples):
	im_placeholder.imvec = im_recon[k].reshape((-1, ))
	obs_recon = im_placeholder.observe_same_nonoise(obs)
	obs_recon.add_cphase(count='max')
	obs_recon.add_logcamp(count='max')
	obs_recon.data['sigma'] *= 0.
	syn[k] = obs_recon
# cpcolor = 0.3 + 0.4 * (prob_rs[0:n_samples] - np.min(prob_rs[0:n_samples])) / (np.max(prob_rs[0:n_samples]) - np.min(prob_rs[0:n_samples]))
cpcolor = (prob_rs[0:n_samples] - np.min(prob_rs[0:n_samples])) / (np.max(prob_rs[0:n_samples]) - np.min(prob_rs[0:n_samples]))

tris = {'azlmaa': ['AZ', 'LM', 'AA'], 'aalmsm': ['AA', 'LM', 'SM'], 'aaazjc': ['AA', 'AZ', 'JC']}
tris_names = {'azlmaa': ['SMT', 'LMT', 'ALMA'], 'aalmsm': ['ALMA', 'LMT', 'SMA'], 'aaazjc': ['ALMA', 'SMT', 'JCMT']}

cpobs  = {}
cptime = {}
cpsyn = {}
for k in range(n_samples):
	cpsyn[k] = {}
for tri in tris:
    cpobs.update({tri: {'cphas': obs.cphase_tri(tris[tri][0], tris[tri][1], tris[tri][2],
                                                              timetype='GMST')['cphase'],
                               'sigma': obs.cphase_tri(tris[tri][0], tris[tri][1], tris[tri][2],
                                                              timetype='GMST')['sigmacp']}})

    for x in cpsyn:
        cpsyn[x].update({tri: syn[x].cphase_tri(tris[tri][0], tris[tri][1], tris[tri][2],
                                                              timetype='GMST')['cphase']})

    cptime.update({tri: obs.cphase_tri(tris[tri][0], tris[tri][1], tris[tri][2], timetype='GMST')['time']})


save_path = '/home/groot/BoumanLab/Lab_meeting/Journal_figures/m87_new/'
from scipy.interpolate import make_interp_spline
for tri in tris:
	plt.figure()

	# plt.errorbar(cptime[tri], cpobs[tri]['cphas'], cpobs[tri]['sigma'],
	#                 fmt='o', mfc='k', mec='k', ecolor='k', ms=10, alpha=0.5, label='Observation')

	cphase0 = cpobs[tri]['cphas']
	cphase0[cphase0<-50] = cphase0[cphase0<-50] + 360
	plt.errorbar(cptime[tri], cphase0, 2.0 * cpobs[tri]['sigma'],
	                fmt='o', mfc='k', mec='k', ecolor='k', ms=10, zorder=101, alpha=0.5, label='Observation')
	for x in cpsyn:
		# plt.plot(cptime[tri], cpsyn[x][tri], '.', mfc=str(cpcolor[x]), mec=str(cpcolor[x]), ms=5, zorder=10)
		# fig.plot(cptime[tri], cpsyn[x][tri], 'o', mfc=cpcolor[x], mec=cpcolor[x],
		#         ms=7, zorder=10, label=label)

		cphase = cpsyn[x][tri]
		cphase[cphase<-50] = cphase[cphase<-50] + 360
		# plt.plot(cptime[tri], cphase, '-', mfc=str(cpcolor[x]), mec=str(cpcolor[x]), ms=5, zorder=10)
		
		# plt.plot(cptime[tri], cphase, '-', color=[0.5, 0.5-0.5*cpcolor[x], 0.5+0.5*cpcolor[x]], ms=5, zorder=10)

		xnew = np.linspace(np.min(cptime[tri]), np.max(cptime[tri]), 300)

		spl = make_interp_spline(cptime[tri], cphase, k=3)  # type: BSpline
		cphase_inter = spl(xnew)
		# cphase_inter = spline(cptime[tri], cphase, xnew)
		plt.plot(xnew, cphase_inter, '-', color=[0.5, 0.5-0.5*cpcolor[x], 0.5+0.5*cpcolor[x]], ms=5, zorder=10, linewidth=3)

		# plt.plot(cptime[tri], cphase, 'o', color=[0.5, 0.5-0.5*cpcolor[x], 0.5+0.5*cpcolor[x]], ms=4, zorder=10)
	plt.locator_params(axis='x', nbins=5)
	plt.locator_params(axis='y', nbins=5)
	plt.xticks(fontsize=25, rotation=45)
	plt.yticks(fontsize=25, rotation=0)
	plt.xlabel('Time (GMST)', fontsize=25)
	plt.ylabel('Closure phase (deg)', fontsize=25)

	title = tris_names[tri][0] + '+' + tris_names[tri][1] + '+' + tris_names[tri][2]
	plt.annotate(f'{title}', xy=(0.5, 0.88), size=25, xycoords='axes fraction', ha='center')
	plt.savefig(save_path+'/cphase_'+title+'.pdf', dpi=300, bbox_inches="tight")



quads = {'azlmaajc': ['AZ', 'LM', 'AA', 'JC'], 'azlmaaap': ['AZ', 'LM', 'AA', 'AP'], 'azlmsmjc': ['AZ', 'LM', 'SM', 'JC']}
quads_names = {'azlmaajc': ['SMT', 'LMT', 'ALMA', 'JCMT'], 'azlmaaap': ['SMT', 'LMT', 'ALMA', 'APEX'], 'azlmsmjc': ['SMT', 'LMT', 'SMA', 'JCMT']}

campobs  = {}
camptime = {}
campsyn = {}
for k in range(n_samples):
	campsyn[k] = {}
for tri in quads:
    campobs.update({tri: {'cphas': obs.camp_quad(quads[tri][0], quads[tri][1], quads[tri][2], quads[tri][3],
                                                              timetype='GMST')['camp'],
                               'sigma': obs.camp_quad(quads[tri][0], quads[tri][1], quads[tri][2], quads[tri][3],
                                                              timetype='GMST')['sigmaca']}})

    for x in cpsyn:
        campsyn[x].update({tri: syn[x].camp_quad(quads[tri][0], quads[tri][1], quads[tri][2], quads[tri][3],
                                                              timetype='GMST')['camp']})

    camptime.update({tri: obs.camp_quad(quads[tri][0], quads[tri][1], quads[tri][2], quads[tri][3], timetype='GMST')['time']})

from scipy.interpolate import make_interp_spline
for tri in quads:
	plt.figure()

	# plt.errorbar(cptime[tri], cpobs[tri]['cphas'], cpobs[tri]['sigma'],
	#                 fmt='o', mfc='k', mec='k', ecolor='k', ms=10, alpha=0.5, label='Observation')

	cphase0 = campobs[tri]['cphas']
	# cphase0[cphase0<-50] = cphase0[cphase0<-50] + 360
	plt.errorbar(camptime[tri], np.log(cphase0), 2.0 * campobs[tri]['sigma']/cphase0,
	                fmt='o', mfc='k', mec='k', ecolor='k', ms=10, zorder=101, alpha=0.5, label='Observation')
	for x in cpsyn:
		# plt.plot(cptime[tri], cpsyn[x][tri], '.', mfc=str(cpcolor[x]), mec=str(cpcolor[x]), ms=5, zorder=10)
		# fig.plot(cptime[tri], cpsyn[x][tri], 'o', mfc=cpcolor[x], mec=cpcolor[x],
		#         ms=7, zorder=10, label=label)

		cphase = campsyn[x][tri]
		# cphase[cphase<-50] = cphase[cphase<-50] + 360
		# plt.plot(cptime[tri], cphase, '-', mfc=str(cpcolor[x]), mec=str(cpcolor[x]), ms=5, zorder=10)
		
		# plt.plot(cptime[tri], cphase, '-', color=[0.5, 0.5-0.5*cpcolor[x], 0.5+0.5*cpcolor[x]], ms=5, zorder=10)


		xnew = np.linspace(np.min(camptime[tri]), np.max(camptime[tri]), 300)

		spl = make_interp_spline(camptime[tri], np.log(cphase), k=3)  # type: BSpline
		cphase_inter = spl(xnew)
		# cphase_inter = spline(cptime[tri], cphase, xnew)
		plt.plot(xnew, cphase_inter, '-', color=[0.5, 0.5-0.5*cpcolor[x], 0.5+0.5*cpcolor[x]], ms=5, zorder=10, linewidth=3)

		# plt.plot(cptime[tri], cphase, 'o', color=[0.5, 0.5-0.5*cpcolor[x], 0.5+0.5*cpcolor[x]], ms=4, zorder=10)
	plt.locator_params(axis='x', nbins=5)
	plt.locator_params(axis='y', nbins=5)
	plt.xticks(fontsize=25, rotation=45)
	plt.yticks(fontsize=25, rotation=0)
	plt.xlabel('Time (GMST)', fontsize=25)
	plt.ylabel('Log closure amplitude', fontsize=25)

	title = quads_names[tri][0] + '+' + quads_names[tri][1] + '+' + quads_names[tri][2] + '+' + quads_names[tri][3]
	plt.annotate(f'{title}', xy=(0.5, 0.88), size=25, xycoords='axes fraction', ha='center')
	plt.savefig(save_path+'/camp_'+title+'.pdf', dpi=300, bbox_inches="tight")


for k in range(n_samples):
	simim.imvec = im_recon[k].reshape((-1, ))
	simim2 = simim.regrid_image(simim.fovx(), 100)
	simim2.display(has_title=False, cfun='afmhot', label_type='scale', has_cbar=False,  cbar_lims=[0, 0.6], scale_fontsize=30, cbar_unit=('mJy', 'muas2')) 
	# simim.display(has_title=False, cfun='afmhot_10us', label_type='none', has_cbar=False,  cbar_lims=[0, 1.], scale_fontsize=30, cbar_unit=('mJy', 'muas2')) 
	plt.savefig(save_path+'/synthetic/recon{}.pdf'.format(k), dpi=300, bbox_inches="tight")


# obs = ../dataset/interferometry_m87_processed/obs_data_m87_day096_sc_0.02_rough.uvfits


img_converter = SimpleCrescentNuisanceFloor_Param2Img(128, r_range=r_range, floor_range=floor_range, fov=fov, n_gaussian=n_gaussian, flux_flag=flux_flag, flux_range=flux_range).to(device=device)
# params = [0.4, 0.20, 0.3, 0.9, 0.6, 0.4, 0.12, 0.20, 0.10, 0.5, 0.4, 0.4, 0.12, 0.10, 0.20, 0.5, 0.2, 0.4]
params = [0.4, 0.20, 0.3, 0.9, 0.6, 0.4, 0.12, 0.20, 0.10, 0.5, 0.4, 0.4, 0.0, 0.10, 0.20, 0.5, 0.0, 0.0]

# img_converter.flux_flag = True
img = img_converter.forward(torch.tensor(np.array(params).reshape((1, -1)),dtype=torch.float32).to(device))
# plt.figure(), plt.imshow(img.cpu().numpy().squeeze(), cmap='gray_r', vmax=3.5e-4)
plt.figure(), plt.imshow(2.1e-4*img.cpu().numpy().squeeze()/np.max(img.cpu().numpy().squeeze()), cmap='gray_r', vmax=3.5e-4)


plt.savefig(save_path+'/synthetic/geometric_model2.pdf', dpi=300, bbox_inches="tight")


# section = img.cpu().numpy().squeeze()[64, :]
# plt.figure(), plt.plot(section)

theta = np.arange(0, 361)
sinusoidal = 1 + 0.3 * np.cos(theta*np.pi/180-45*np.pi/180)

# plt.figure(), plt.plot(theta, sinusoidal, color='orange')
plt.figure(figsize=(8, 3)), plt.plot(theta, sinusoidal, color='orange')
plt.plot(theta, 1.0*np.ones(theta.shape), '--', color='orange')
plt.plot(np.arange(0, 46), 1.3*np.ones(46), '--', color='orange')
plt.plot(np.arange(0, 226), 0.7*np.ones(226), '--', color='orange')
plt.plot(45*np.ones(14), np.arange(0, 1.4, 0.1), '--', color='black')

plt.locator_params(axis='x', nbins=6)
plt.locator_params(axis='y', nbins=5)
degree_list = [r'$0^{\circ}$', r'$60^{\circ}$', r'$120^{\circ}$', r'$180^{\circ}$', r'$240^{\circ}$', r'$300^{\circ}$', r'$360^{\circ}$']

plt.xticks(np.arange(0, 361, 60), degree_list, fontsize=20, rotation=0)

# plt.yticks(fontsize=25, rotation=0)
plt.tick_params(axis='y', left=False, right=False, labelleft=False)
# plt.xlabel(r'$\theta$ ($360^{\circ}$)', fontsize=25)
# plt.ylabel('Intensity', fontsize=25)
plt.ylim([0.5, 1.5])
plt.xlim([0, 360])

title = 'Azimuthal Intensity'
plt.annotate(f'{title}', xy=(0.5, 0.85), size=25, xycoords='axes fraction', ha='center')
plt.savefig(save_path+'/synthetic/azmulthal.pdf', dpi=300, bbox_inches="tight")


save_path = '/home/groot/BoumanLab/Lab_meeting/Journal_figures/m87_new/'
im_truth = eh.image.load_fits('/home/groot/BoumanLab/DPI/dataset/interferometry_m87/synthetic_crescentfloorgaussian2new4/ground_truth.fits')
im_truth.display(has_title=False, cfun='afmhot', label_type='scale', has_cbar=False, scale_fontsize=30, cbar_unit=('mJy', 'muas2')) 
plt.savefig(save_path+'/synthetic/truth{}.pdf'.format(k), dpi=300, bbox_inches="tight")

