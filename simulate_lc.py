import pandas as pd
import numpy as np
from numpy.random import choice
import scipy
from string import atof
import os
from javelin.zylc import get_data, LightCurve
from javelin.lcmodel import Cont_Model, Rmap_Model, Pmap_Model
from javelin.predict import PredictSpear
from scipy import stats
from astropy.io import fits
import scipy.io as sio
import time

epo_up_lim = 100000

class lc_simulation(object):
	def __init__(self, store_path=None, lc_path=None, drw_path=None):
		np.random.seed(int(time.time()))
		self.drw_path = drw_path
		self.drw_band = input("Please input the band of DRW parameters")
		self.lc_path = lc_path
		self.read_drw_params()
		if store_path is not None:
			
			self.store_path = store_path
			self.read_drw_params()
			self.read_lc()
		else:

			self.lc_path = lc_path
			

			if lc_path is None:
				"No input of light curve nor previous results, empty light curve set!"
				pass
			else:
				self.store_path = '/Users/zhanghaowen/Desktop/AGN/high-z_QSO_variability/'
				self.read_lc()
				self.read_drw_params()
				self.store(self.store_path)

		
		
	def restore(self, store_path):
		file_list = os.listdir(store_path)
		prob_files = []
		for file in file_list:
			if '.mat' in file:
				prob_files.append(file)
		if len(prob_files) > 1:
			print "All the .mat files in the directory are the following, please choose the one from which you want to restore."
			print file_list
			target_file = input()
			if target_file == 'default':
				target_file = os.path.join(store_path, 'lc_simulate_params.mat')
			else:
				target_file = os.path.join(store_path, target_file)
		else: 
			target_file = os.path.join(store_path, prob_files[0])
		load_dict = sio.loadmat(target_file)
		self.start_MJD_set = load_dict['start_MJD_set'][0]
		self.mean_mag_set = load_dict['mean_mag_set'][0]
		self.frac_err_set = load_dict['frac_err_set'][0]
		self.mean_frac_err_set = load_dict['mean_frac_err_set'][0]
		self.intv_set = load_dict['intv_set'][0]
		self.baseline_set = load_dict['baseline_set'][0]
		self.gap_pos_set = load_dict['gap_pos_set'][0]
		self.gap_num_set = load_dict['gap_num_set'][0]
		self.gap_sep_set = load_dict['gap_sep_set'][0]
		self.gap_dur_set = load_dict['gap_dur_set'][0]
		self.epoch_set = load_dict['epoch_set'][0]
		self.sigma_set = load_dict['sigma_set'][0]
		self.tau_set = load_dict['tau_set'][0]

	def store(self, store_path):
		if store_path is None:
			print "no store path specified, pass saving."
		else:
			save_dict = {}
			save_dict['start_MJD_set'] = self.start_MJD_set
			save_dict['mean_mag_set'] = self.mean_mag_set
			save_dict['frac_err_set'] = self.frac_err_set
			save_dict['mean_frac_err_set'] = self.mean_frac_err_set
			save_dict['intv_set'] = self.intv_set
			save_dict['baseline_set'] = self.baseline_set
			save_dict['gap_pos_set'] = self.gap_pos_set
			save_dict['gap_num_set'] = self.gap_num_set
			save_dict['gap_sep_set'] = self.gap_sep_set
			save_dict['gap_dur_set'] = self.gap_dur_set
			save_dict['epoch_set'] = self.epoch_set
			save_dict['sigma_set'] = self.sigma_set
			save_dict['tau_set'] = self.tau_set
			filename = 'lc_simulate_params.mat'
			# print save_dict
			# print store_path + filename
			sio.savemat(store_path+filename, save_dict)

		
	def read_lc(self, bad_value=None):
		
		lc_path = self.lc_path
		print self.lc_path

		lc_list = os.listdir(lc_path)

		start_MJD_set = np.array([])
		mean_mag_set =  np.array([])
		frac_err_set = np.array([])
		mean_frac_err_set = np.array([])
		baseline_set = np.array([])
		intv_set = np.array([])
		epoch_set = np.array([])
		gap_num_set  = np.array([])
		gap_pos_set  = np.array([])
		gap_sep_set = np.array([])
		gap_dur_set = np.array([])

		if bad_value == None:
			bad_value = [-50000, -99.99, 100.0]

		for ind, lc in enumerate(lc_list):
			if ind % 500 == 0:
				print "Finished loading of %d lightcurves." %ind
			if lc == '.DS_Store':
				continue
			lc_file = open(os.path.join(lc_path, lc), 'r')
			lc_data = pd.read_csv(lc_file, header=None, names=['MJD', 'mag', 'err'], usecols=[0, 1, 2], sep=' ')

			#remove the bad values in the data
			
			lc_data['MJD'] = lc_data['MJD'].replace(to_replace=bad_value[0], value=np.nan)
			lc_data['MJD'] = lc_data['MJD'].replace(to_replace=np.nan, value=np.nanmean(lc_data['MJD']))

			lc_data['mag'] = lc_data['mag'].replace(to_replace=bad_value[1], value=np.nan)
			lc_data['mag'] = lc_data['mag'].replace(to_replace=np.nan, value=np.nanmean(lc_data['mag']))

			lc_data['err'] = lc_data['err'].replace(to_replace=bad_value[2], value=np.nan)
			lc_data['err'] = lc_data['err'].replace(to_replace=np.nan, value=np.nanmean(lc_data['err']))



			intv = np.array(lc_data['MJD'][1:-1]) - np.array(lc_data['MJD'][0:-2])

			start_MJD_set = np.append(start_MJD_set, lc_data['MJD'][0])

			mean_mag_set = np.append(mean_mag_set, np.mean(np.array(lc_data['mag'])))

			frac_err = np.abs(np.array(lc_data['err']) / np.array(lc_data['mag']))
			bad_mask = frac_err < 0.001
			frac_err = np.ma.array(frac_err, mask=bad_mask).compressed()
			mean_frac_err = np.mean(frac_err)
			frac_err_set = np.append(frac_err_set, frac_err)
			mean_frac_err_set = np.append(mean_frac_err_set, mean_frac_err)

			baseline = np.array(lc_data['MJD'])[-1] - np.array(lc_data['MJD'])[0] 
			baseline_set = np.append(baseline_set, baseline)

			intv_set = np.append(intv_set, intv)
			intv_std = np.std(intv)

			gap_pos = np.array(lc_data['MJD'])[np.where(intv > 3 * intv_std)]
			# print gap_pos
			gap_pos_set = np.append(gap_pos_set, gap_pos)

			gap_num = len(np.where(intv > 3 * intv_std)[0])
			gap_num_set = np.append(gap_num_set, gap_num)

			gap_sep = gap_pos[1:] - gap_pos[0:-1]
			# print gap_sep
			gap_sep_set = np.append(gap_sep_set, gap_sep)

			gap_dur = intv[np.where(intv > 3 * intv_std)]
			gap_dur_set = np.append(gap_dur_set, gap_dur)

			epoch_set = np.append(epoch_set, len(lc_data['MJD']))

			lc_file.close()

		self.start_MJD_set = start_MJD_set
		self.mean_mag_set = mean_mag_set
		self.frac_err_set = frac_err_set
		self.mean_frac_err_set = mean_frac_err_set
		self.intv_set = intv_set
		self.baseline_set = baseline_set
		self.gap_pos_set = gap_pos_set

		self.gap_num_set = gap_num_set
		self.gap_sep_set = gap_sep_set
		# print "gap_sep_set:", self.gap_sep_set
		self.gap_dur_set = gap_dur_set
		self.epoch_set = epoch_set

		# print mean_mag_set.shape

		self.covariance = np.cov(m=[self.mean_mag_set, self.mean_frac_err_set, self.gap_num_set, self.epoch_set])
		
		print self.gap_num_set
		print self.gap_pos_set
		print self.mean_mag_set





		# print self.covariance


	def read_drw_params(self):
		if os.path.exists(os.path.join(self.drw_path, ('s82drw_%s.dat' %self.drw_band))):
			drw_file = os.path.join(self.drw_path, ('s82drw_%s.dat' %self.drw_band))
		else:
			print "There are no DRW parameters of the band you are going to use, use g band instead."
			drw_file = os.path.join(self.drw_path, 's82drw_g.dat')
		drw = pd.read_csv(drw_file, skiprows=3, header=None, \
						  names=['log10tau', 'log10sigma_KBS09', 'edge', 'Plike', 'Pnoise', 'Pinf', 'npts'],\
						  usecols=[7, 8, 13, 14, 15, 16, 18], delim_whitespace=True)
		self.tau_set = 10**np.array(drw['log10tau'])
		self.sigma_set = (self.tau_set / (2 * 365.25))**0.5 * 10**(np.array(drw['log10sigma_KBS09']))
		mask1 = np.array(drw['log10sigma_KBS09']) <= -10
		mask2 = np.array(drw['npts']) < 10
		mask3 = np.array(drw['Plike']) - np.array(drw['Pnoise']) <= 2
		mask4 = np.array(drw['Plike']) - np.array(drw['Pinf']) <= 0.05
		mask5 = np.array(drw['edge']) != 0
		mask = [mask1[i] or mask2[i] or mask3[i] or mask4[i] or mask5[i] for i in range(len(self.tau_set))]

		self.sigma_set = np.ma.array(self.sigma_set, mask=mask).compressed()
		self.tau_set = np.ma.array(self.tau_set, mask=mask).compressed()
		mask = self.sigma_set > 5
		self.sigma_set = np.ma.array(self.sigma_set, mask=mask).compressed()
		self.tau_set = np.ma.array(self.tau_set, mask=mask).compressed()

	def simulate(self, output_path, cat_path=None, lc_number=None, model="DRW", dataset='SDSS_S82', prior='epoch', timescales=None, max_timescales=None):
		
		# function to generate simulate observation time series according to statistics of a 
		# certain dataset, and in the meantime stat the typical variability amplitude on a certain set of
		# timescales.
		# parameters:
		# 
		# cat_path: string, the path of the catalog to simulate. the code needs the redshift and the mean 
		# magnitude information from this path.
		# 
		# output_path: string, the path to store the output light curve files (in .txt format)
		# 
		# model: string, specify the stat model used to describe the variability of AGNs, default: 'DRW'
		# 
		# dataset: string, specify the dataset to use as template to mock light curves, default: 'SDSS_S82'
		# 
		# prior: string, specify the scheme of generating the light curves. Default: 'epoch'
		# 		 choice: 'baseline': extend the simulated baseline until it exceeds the length sampled from 
		# 				the dataset baseline set, regardless of how many epochs it has
		# 				'epoch': extend the simulated light curve until it has the number of epochs sampled
		# 						from the dataset epoch set, regardless of how long the simulated baseline is.
		# 						
		# timescales: ndarray, used to specify the typical timescales on which the maximum magnitude difference
		# 			  should be calculated.
		# 	
		# max_timescales: ndarray, used to specify the typical timescales within which the maximum magnitude difference
		# 				  should be calculated.
		self.output_path = output_path
		self.cat_path = cat_path
		self.lc_number = lc_number
		self.model = model
		self.prior = prior
		self.dataset = dataset
		

		del_mag_dict = {}


		if timescales is None:
			timescales = np.array([10.0, 30.0, 90.0, 180.0, 360.0, 720.0, 1080.0, 2500.0])

		if max_timescales is None:
			max_timescales = np.array([3.0, 6.0, 10.0]) * 365.25

		for ts in timescales:
			del_mag_dict['max_del_mag_on_' + str(ts)] = []
		for mts in max_timescales:
			del_mag_dict['max_del_mag_within_' + str(mts)] = []


		del_mag_dict['number'] = []
		del_mag_dict['redshift'] = []

		if cat_path is None:
			print "No catalog simulation, the mean magnitude will be chosen from the real light curve set."

		else:

			file_list = os.listdir(cat_path)
			prob_files = []
			for file in file_list:
				if '.fits' in file:
					prob_files.append(file)

			print "The usable catalogs are printed below, please indicate the one to be used, 1 as the first and so on."
			print prob_files
			indicator = input("Please input the number.\n")

			hdu = fits.open(os.path.join(cat_path, prob_files[indicator - 1]))

			selected_band = input("Please input the band to simulate.\n")
			self.selected_band = selected_band

			self.mean_mag_set = hdu[1].data[selected_band]
			# create the directory to contain the light curves
			self.output_path = self.output_path + '_' + selected_band + '_band'
			if os.path.exists(self.output_path) == False:
				os.makedirs(self.output_path)

			self.redshift_set = hdu[1].data['redshift']
			if lc_number is None:
				lc_number = len(self.redshift_set)



		i = 0
		err_log = open(self.output_path + '/disgard_record', 'w')
		err_log.write('num' + ' ' + 'mean_mag' + ' ' + 'redshift' + '\n')
		

		for i in range(lc_number):
			try:
				for trial_num in range(10):


					try:
						self.single_simulate(i, prior=self.prior, model=self.model, timescales=timescales, max_timescales=max_timescales, del_mag_dict=del_mag_dict)
						del_mag_dict['number'].append(i)
						del_mag_dict['redshift'].append(self.redshift_set[i])
						break
					except Exception as e:
						if e is not KeyboardInterrupt:
							pass
						elif e is ValueError:
							break
							# print "error when generating light curve."

					
					# self.single_simulate(i, prior=self.prior, model=self.model, timescales=timescales, max_timescales=max_timescales, del_mag_dict=del_mag_dict)
					# del_mag_dict['number'].append(i)
					# del_mag_dict['redshift'].append(self.redshift_set[i])

					
					# print "failed one simulation."
					# yyy = raw_input('just for pause.')
				# print "finished %d simulation." %i
			except KeyboardInterrupt:
				sio.savemat((self.output_path + '/del_mag_stat_%s_band.mat' % selected_band), del_mag_dict)
				print "delta magnitude statistics saved."
				err_log.close()
				exit()


		
		sio.savemat((self.output_path + '/del_mag_stat_%s_band.mat' % selected_band), del_mag_dict)
		print "delta magnitude statistics saved."
		err_log.close()


	def single_simulate(self, i, model, prior, timescales, max_timescales, del_mag_dict):
		# draw a set of descriptive parameters from the ensemble
		
		if i % 1000 == 0:
			print "finished %d simulations." %i 
		
		# print "0"

		gap_num = choice(self.gap_num_set, size=1)
		gap_pos = np.sort(choice(self.gap_pos_set, size=int(gap_num)))

		# print "1"

		if self.cat_path is None:	
			zp1 = 1.0
		else:
			# ind = choice(len(self.mean_mag_set))
			# ind = i is for the mode in which we have to simulate on each object in the catalog.
			
			zp1 = self.redshift_set[i] + 1.0
			mean_mag = self.mean_mag_set[i]
			# print "mean_mag: ", mean_mag
		start_MJD = choice(self.start_MJD_set, size=1)
		frac_err = choice(self.frac_err_set, size=1)

		if not np.isfinite(mean_mag):
			raise ValueError

		# print "2"
		# print self.sigma_set
		# print self.tau_set


		#This configuration is only for the test.
		
		

		# parameters needed by JAVELIN to produce the line+continuum light curve,
		# it's useless in simulating the pure continuum band but we have to pass 
		# it into the function.
		llags = [5, 0]
		lwids = [2.0, 0]
		lscales = [0.05, 1]

		


		# print "4"



		# print "3"



		# print MJD, input_mag, input_err
		


		while True:
			MJD = np.array([])
			flux = np.array([])
			err = np.array([])

			if model == "DRW":
				while True:
					ind = choice(len(self.sigma_set))
					sigma = self.sigma_set[ind]
					# The observed tau would be dilated by a factor of (1+z)
					# print zp1
					tau = self.tau_set[ind] * zp1
					# print "sigma and tau: ", sigma, tau
					# tau = self.tau_set[ind]
					if tau / zp1 <= 4000.0:
						break

			elif model == "pow-law":
				sigma = 0.2
				tau = 0.45


			if prior == 'baseline':
				# baseline = choice(self.baseline_set, size=1)
				baseline = 4000.0
				# for k in range(0, epo_up_lim):
				# 	if k == 0:
				# 		MJD = np.append(MJD, start_MJD)
				# 		frac_err = choice(self.frac_err_set, size=1)
						
				# 	# draw a interval value from the ensemble
				# 	intv = choice(self.intv_set, size=1)

				# 	# mock the gaps
				# 	# for j in range(gap_num):
				# 	# 	if k == gap_pos[j] + 1:
				# 	# 		intv = choice(self.gap_dur_set, size=1)

				# 	# use the interval generated to yield the next observation time point
				# 	MJD = np.append(MJD, MJD[-1] + intv)
					

					
				# 	if MJD[-1] - MJD[0] > baseline:
				# 		break
				# MJD = np.array(MJD)
				# good_all_ts = 1
				# for ts in timescales:
				# 	good_this_ts = np.intersect1d(np.where(MJD[1:] - MJD[0:-1] > 0.9 * ts), np.where(MJD[1:] - MJD[0:-1] < 1.1 * ts))
				# 	if len(good_this_ts):
				# 		pass
				# 	else:
				# 		good_all_ts = 0
				# 	print ts, good_this_ts, good_all_ts
				# 	yyy = raw_input('pause')
				# if not good_all_ts:
				# 	continue
				MJD = np.linspace(0, baseline, baseline / 10.0 + 1)

			elif prior == 'epoch':
				epoch = 4000
				# epoch = choice(self.epoch_set, size=1)
				for k in range(epoch):
					if k == 0:
						MJD = np.append(MJD, start_MJD)
						frac_err = choice(self.frac_err_set, size=1)

					# draw a interval value from the ensemble
					intv = choice(self.intv_set, size=1)

					# mock the gaps
					for j in range(gap_num):
						if k == gap_pos[j] + 1:
							intv = choice(self.gap_dur_set, size=1)

					# use the interval generated to yield the next observation time point
					MJD = np.append(MJD, MJD[-1] + intv)			


			# input_mag = np.ones(len(MJD)) * mean_mag
			# input_err = abs(np.random.normal(0, 1, len(MJD))) * frac_err * mean_mag

			# MJD_benchmark = choice(self.baseline_set, size=1)

			# ps = PredictSpear(sigma, tau, llags, lwids, lscales, spearmode="Pmap", GPmodel=model)
			# lcdat = [[MJD, input_mag, input_err], [MJD, input_mag, input_err]]
			# # print (np.array(lcdat) <= 0.0).any()
			# lcnew = ps.generate(lcdat, set_threading=False)[0]

			lc_mag = [mean_mag + np.random.normal(0, frac_err * mean_mag)]
			for i in range(1, len(MJD)):
				lc_mag.append(np.exp(-(MJD[i] - MJD[i-1]) / tau) * lc_mag[i - 1] +\
							  mean_mag * (1 - np.exp(-(MJD[i] - MJD[i-1]) / tau)))



			# print '5'

			# lc_mag = np.array(lcnew[1])
			# lc_err = np.array(lcnew[2])
			lc_mag = np.array(lc_mag)
			MJD = np.array(MJD)

			# print "mag: ", lc_mag
			# print "max_del_mag: ", np.max(lc_mag) - np.min(lc_mag)

			if np.max(lc_mag) - np.min(lc_mag) > 5:
				continue
			
			

			# print '6'
			# print timescales, max_timescales


			max_mag_timescales = []
			max_mag_max_timescales = []

			for ts in timescales:
				mag_this_ts = []
				for k in range(len(MJD)):
					in_range_pt = np.where(abs(MJD - MJD[k] - ts) <= 0.1 * ts)
					# print lc_mag, len(in_range_pt)
					if in_range_pt[0].size == 0:
						continue
					max_del_mag = np.max(abs(lc_mag[k] - lc_mag[in_range_pt]))
					mag_this_ts.append(max_del_mag)


				if len(mag_this_ts):
					del_mag_dict['max_del_mag_on_' + str(ts)].append(np.max(mag_this_ts))
					max_mag_timescales.append(np.max(mag_this_ts))
				else:
					del_mag_dict['max_del_mag_on_' + str(ts)].append(-1.0)

			# print '7'

			for mts in max_timescales:
				mag_this_mts = []
				for k in range(len(MJD)):
					in_range_pt = np.where(abs(MJD - MJD[k]) <= mts)
					if in_range_pt[0].size == 0:
						continue
					max_del_mag = np.max(abs(lc_mag[k] - lc_mag[in_range_pt]))
					mag_this_mts.append(max_del_mag)

				if len(mag_this_mts):
					del_mag_dict['max_del_mag_within_' + str(mts)].append(np.max(mag_this_mts))
					max_mag_max_timescales.append(np.max(mag_this_mts))
					
				else:
					del_mag_dict['max_del_mag_within_' + str(mts)].append(-1.0)

			# print '8'
			
			break

				# print "oops!"
		# lc_dataframe = pd.DataFrame(data=np.transpose(np.array([MJD, lc_mag, lc_err])), columns=['MJD', 'mag', 'err'])
		# output_name = '%s/z=%f_%s_%d.txt' %(self.output_path, self.redshift_set[i], self.dataset, i)
		# lc_dataframe.to_csv(output_name, sep=' ', columns=['MJD', 'mag', 'err'],\
		# 					header=['MJD', 'mag', 'err'], index=False)

		# print '9'
		

	def simulate_check(self):
		# This is the function to check the statistical properties of the simulated
		# time series. 
		sim_lc_list = os.listdir(self.output_path)

		sim_start_MJD_set = np.array([])
		sim_mean_mag_set =  np.array([])
		sim_frac_err_set = np.array([])
		sim_baseline_set = np.array([])
		sim_intv_set = np.array([])
		sim_epoch_set = np.array([])
		sim_gap_num_set  = np.array([])
		sim_gap_pos_set  = np.array([])
		sim_gap_sep_set = np.array([])
		sim_gap_dur_set = np.array([])

		for ind, sim_lc in enumerate(sim_lc_list):
			sim_lc_file = os.path.join(self.output_path, sim_lc)
			sim_lc_data = pd.read_csv(sim_lc_file, header=0, names=['MJD', 'mag', 'err'], usecols=[0, 1, 2], sep=' ')

			intv = np.array(sim_lc_data['MJD'][1:-1]) - np.array(sim_lc_data['MJD'][0:-2])

			sim_start_MJD_set = np.append(sim_start_MJD_set, sim_lc_data['MJD'][0])

			sim_mean_mag_set = np.append(sim_mean_mag_set, np.mean(np.array(sim_lc_data['mag'])))

			sim_frac_err = np.abs(np.array(sim_lc_data['err']) / np.array(sim_lc_data['mag']))
			bad_mask = sim_frac_err < 0.001
			sim_frac_err = np.ma.array(sim_frac_err, mask=bad_mask)
			sim_frac_err_set = np.append(sim_frac_err_set, sim_frac_err)

			sim_baseline = np.array(sim_lc_data['MJD'])[-1] - np.array(sim_lc_data['MJD'])[0] 
			sim_baseline_set = np.append(sim_baseline_set, sim_baseline)

			sim_intv_set = np.append(sim_intv_set, intv)
			sim_intv_std = np.std(intv)

			sim_gap_pos = intv[np.where(intv > 5 * sim_intv_std)]
			sim_gap_pos_set = np.append(sim_gap_pos_set, sim_gap_pos)

			sim_gap_num = len(np.where(intv > 5 * sim_intv_std)[0])
			sim_gap_num_set = np.append(sim_gap_num_set, sim_gap_num)

			sim_gap_sep = sim_gap_pos[1:-1] - sim_gap_pos[0:-2]
			sim_gap_sep_set = np.append(sim_gap_sep_set, sim_gap_sep)

			sim_gap_dur = intv[np.where(intv > 5 * sim_intv_std)]
			sim_gap_dur_set = np.append(sim_gap_dur_set, sim_gap_dur)

			sim_epoch_set = np.append(sim_epoch_set, len(sim_lc_data['MJD']))


		self.sim_start_MJD_set = sim_start_MJD_set
		self.sim_mean_mag_set = sim_mean_mag_set
		self.sim_frac_err_set = sim_frac_err_set
		self.sim_intv_set = sim_intv_set
		self.sim_baseline_set = sim_baseline_set
		self.sim_gap_pos_set = sim_gap_pos_set
		self.sim_gap_num_set = sim_gap_num_set
		self.sim_gap_sep_set = sim_gap_sep_set
		self.sim_gap_dur_set = sim_gap_dur_set
		self.sim_epoch_set = sim_epoch_set

		self.baseline_ks = stats.ks_2samp(self.baseline_set, self.sim_baseline_set)
		self.epoch_ks = stats.ks_2samp(self.epoch_set, self.sim_epoch_set)

		print "the k-s test between the real and simulated baseline set is: ", self.baseline_ks
		print "the k-s test between the real and simulated epoch set is: ", self.epoch_ks







