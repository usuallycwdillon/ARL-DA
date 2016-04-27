#!/usr/bin/python

import numpy as np 
from numpy.random import random_sample
import pandas as pd 
import scipy.stats as st 
import scipy as sc
import matplotlib.pyplot as plt
import csv
import json


"""Class for representing army_personnel (randomly sim.) data.
	   n represents the number of observations and
	   cov represents the number of covariates. 
	   With this information, we can construct an n*cov dataframe/matrix """

class ArmyDataSim(object):
	""" Class constructor takes in number of observations and initializes data frame with 0s """
	def __init__(self, n):
		super(ArmyDataSim, self).__init__()

		# Number of observations
		self.n = n

		# Create empty data frame with these column names and with n rows arranged by idx (index)
		self.columns = ['age', 'gender', 'education_level', 'years_active_service', 'years_reserv_service',
					'job_specialty', 'job_years', 'rank', 'location', 'is_deployed', 'months_deployed',
					'anxiety', 'self_efficacy', 'motivation', 'handgun_prof', 'last_fired_wp', 'shrange_hours',
					'is_markstrained', 'last_training', 'qualification_performance', 'is_gamer', 'hours_fps_week',
					'email_id']

		idx = xrange(n)
		self.df = pd.DataFrame(0, index = idx, columns = self.columns)

		#initialize primary vars (other covariates are dependent)
		ed_outcomes = np.array([1,2,3,4,5,6,7])
		ed_probs = np.array([0.6, 0.3, 0.05, 0.02, 0.015, 0.01 , 0.005])
		self.df['education_level'] = self.weighted_probs(ed_outcomes, ed_probs, self.n)
		self.df['gender'] = np.random.binomial(n = 1, p = 0.70, size = self.n)
		self.df['is_gamer'] = np.random.binomial(n = 1, p = 0.50, size = self.n)
		self.df['is_deployed'] = np.random.binomial(n = 1, p = 0.50, size = self.n)


	""" Fill in data w randomly generated values """

	def init(self):
		self.df['rank'] = ["Enlisted" if i==1 else "Officer" for i in self.df['education_level']]
		self.df['years_active_service'] = self.rand_range(0,20, self.n)
		self.df['years_reserv_service'] = self.rand_range(0,8, self.n)
		self.df['job_specialty'] = ""
		self.df['job_years'] = self.rand_range(0,20, self.n)
		self.df['location'] = ""
		self.df['anxiety'] = self.rand_range(1,100, self.n)
		self.df['self_efficacy'] = self.rand_range(1,100, self.n)
		self.df['motivation'] = self.rand_range(1,100, self.n)
		self.df['handgun_prof'] = self.rand_range(0,20, self.n)
		self.df['last_fired_wp'] = self.rand_range(0,6, self.n)
		self.df['shrange_hours'] = self.rand_range(0,1080, self.n)
		self.df['is_markstrained'] = np.random.binomial(n = 1, p = 0.50, size = self.n)
		self.df['last_training'] = self.rand_range(0,6, self.n)
		self.df['qualification_performance'] = ["Marksman" if i==1 else "Novice" for i in self.df['is_markstrained']]
		self.df.loc[((self.df['qualification_performance'] == "Novice") &
					 (self.df['handgun_prof'] < 6)), 'qualification_performance'] = "Unexperienced"
		self.df.loc[((self.df['qualification_performance'] == "Marksman") & (self.df['shrange_hours'] < 300) &
					 (np.random.rand() > .50)),'qualification_performance'] = "Sharpshooter"
		self.df.loc[((self.df['qualification_performance']=="Marksman") & (self.df['shrange_hours'] > 300) &
					 (np.random.rand() < .50)), 'qualification_performance'] = "Sharpshooter"
		self.df.loc[((self.df['qualification_performance']=="Marksman") & (self.df['shrange_hours'] > 600) &
					 (np.random.rand() > .50)), 'qualification_performance'] = "Expert"
		self.df['hours_fps_week'] = self.rand_range(0,120, self.n)
		self.df['email_id'] = ["learner" + str(i) + "@example.com" for i in self.df.index]

	""" Recode factors to categorial variables """
	def recode(self):
		self.df['gender'] = ["Male" if i==1 else "Female" for i in self.df['gender']]
		self.df['age'] = [18 if i==1 else 19 if i==2 else 20 if i==3 else 21 if i==4 else 22 if i==5 else 23 if i==6 else 24 for i in self.df['education_level']]
		self.df['education_level'] = ["High School" if i==1 else "GED" if i==2 else "Some college" if i==3 else "Associate's Degree" if i==4 else "Bachelor's Degree" if i==5 else "Master's Degree" if i==6 else "PHD" for i in self.df['education_level']]
		self.df['handgun_prof_recoded'] = [0 if i <= 6 else 1 if i > 6 and i <= 15 else 2 for i in self.df['handgun_prof']]
		self.df['months_deployed'] = self.rand_range(0,36, self.n)


	# Helper function to create weighted rvs
	def weighted_probs(self, outcomes, probabilities, size):
		temp = np.add.accumulate(probabilities)
		return outcomes[np.digitize(random_sample(size), temp)]


	""" methods for byte seq randomness from np. Useful for creating/filling in dataframe"""

	#Return probabilities between 0 and 1
	def rand_prob(self, n):
		return np.random.random((n)).tolist()

	#Return random values between low and high
	def rand_range(self, lo, high, n):
		return np.random.randint(lo,high,n).tolist()

	""" Probability Density Function Estimation and Visualization"""
	# Alternatives (for future): Kolmogorov-smirnov, KDE, Mean integrated squared error, SDE
	

	### TODO: generalize for all columns in data frame
	def calc_mles(self, col):
		mles = []
		#scipy's internal distributions (continuous and discrete)
		cont_distributions = [st.beta, st.bradford, st.cauchy, st.chi, st.chi2, st.expon, st.f, st.genlogistic,
					st.genpareto, st.genexpon, st.gamma, st.gengamma, st.laplace, st.norm, st.pareto, st.rayleigh,
					st.t, st.wald]
		disc_distributions = [st.bernoulli, st.binom, st.nbinom, st.randint]
		# for now, it's only the discrete distributions that are being compared to but it's easy to change to cont
		for dist in disc_distributions:
			pars = dist.fit(col)
			mle = dist.nnlf(pars, col)
			mles.append(mle)

		# Sorting results of MLEs and showing the best fit
		results = [(distribution.name, mle) for distribution, mle in zip(distributions, mles)]
		best_fit = sorted(zip(distributions, mles), key=lambda d: d[1])[0]
		print 'Best fit reached using {}, MLE value: {}'.format(best_fit[0].name, best_fit[1])

	# Visualize the various distributions and compare them to data

	def visualize_fits(self, col):
		#scipy's internal distributions (continuous and discrete)
		cont_distributions = [st.beta, st.bradford, st.cauchy, st.chi, st.chi2, st.expon, st.f, st.genlogistic,
					st.genpareto, st.genexpon, st.gamma, st.gengamma, st.laplace, st.norm, st.pareto, st.rayleigh,
					st.t, st.wald]
		disc_distributions = [st.bernoulli, st.binom, st.nbinom, st.randint]
		
		hist = plt.hist(col, bins=range(100), color='w')
		for dist in disc_distributions:
			d = getattr(scipy.stats, dist)
			param = d.fit(col)
			fitted_pdf = d.pdf(len(col), *param[:-2], loc = param[-2])
			plt.plot(fitted_pdf, label=dist)
			plt.xlim(xrange(100))
		plt.legend(loc='upper right')
		plt.show()



	""" Output Data to json or csv"""
	def class_to_json(self, name):
		self.df.to_json("../data/class_data/" + str(name) + ".json", orient='index')

	def class_to_csv(self, name):
		self.df.to_csv("../data/class_data/" + str(name) + ".csv", orient='index')

	def persons_to_csv(self):
		for index,row in self.df.iterrows():
			output_doc = '..data/person_data/learner' + str(index) + '.csv'
			with open(output_doc, 'wb') as file:
				csv_file = csv.writer(file)
				csv_file.writerow(self.columns)
				csv_file.writerow(row)

	def persons_to_json(self):
		for i in self.df.index:
			self.df.loc[i].to_json("..data/person_data/learner{}.json".format(i))
	


test_data = ArmyDataSim(10000)
test_data.init()
test_data.recode()
test_data.class_to_json("class")
test_data.class_to_csv("class")





		

