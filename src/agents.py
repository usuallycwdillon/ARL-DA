## This is not a valid starting point. 
#  This class file defines agents that activate in the learner_xSim.py file.

import numpy as np 
from numpy.random import random_sample
import pandas as pd 
import scipy.stats as st 
import scipy as sc
import matplotlib.pyplot as plt
import csv
import json

class learner():
   '''
   
   '''
   def __init__(self, id):
      
      ed_outcomes = np.array([1,2,3,4,5,6,7])
 		    ed_probs = np.array([0.6, 0.3, 0.05, 0.02, 0.015, 0.01 , 0.005])
		
   
      self._student_id = 'learner' + id + '@example.com'
      self._age = 
      self._gender = np.random.binomial(n = 1, p = 0.70, size = 1)
      self._education_level = weighted_probs()
      self._years_active_service
      self._years_reserve_service
      self._job_speciality
      self._years_job
      self._rank_class
      self._location
      self._is_deployed
      self._months_deployed
      self._handgun_prof
      self._rifle_prof
      self._last_wp_fire
      self._range_hours
      self._is_markstrained
      self._last_markstraining
      self._is_gamer
      self._weekly_fps_hours
      self._general_anxiety
      self._general_self_efficacy
      self._general_motivation
      
      #initialize primary vars (other covariates are dependent)
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
		self.df.loc[((self.df['qualification_performance'] == "Marksman") & (self.df['shrange_hours'] > 300) &
					 (np.random.rand() > .40)),'qualification_performance'] = "Sharpshooter"
		self.df.loc[((self.df['qualification_performance']=="Sharpshooter") & (self.df['shrange_hours'] > 600) &
					 (np.random.rand() > .60)), 'qualification_performance'] = "Expert"
		self.df['hours_fps_week'] = self.rand_range(0,120, self.n)
		self.df['email_id'] = ["learner" + str(i) + "@example.com" for i in self.df.index]

	""" Recode factors to categorial variables """
	def recode(self):
		self.df['gender'] = ["Male" if i==1 else "Female" for i in self.df['gender']]
		self.df['age'] = [18 if i==1 else 19 if i==2 else 20 if i==3 else 21 if i==4 else 22 if i==5 else 23 if i==6 else 24 for i in self.df['education_level']]
		self.df['education_level'] = ["High School" if i==1 else "GED" if i==2 else "Some college" if i==3 else "Associate's Degree" if i==4 else "Bachelor's Degree" if i==5 else "Master's Degree" if i==6 else "PHD" for i in self.df['education_level']]
		self.df['handgun_prof_recoded'] = [0 if i <= 6 else 1 if i > 6 and i <= 15 else 2 for i in self.df['handgun_prof']]
		self.df['months_deployed'] = self.rand_range(0,36, self.n)

      
      
   @property
   def age(self):
      return self._age
      
   @property
   def gender(self):
      return self._gender
      
   @property
   def education_level(self):
      return self._education_level
      
   @property
   def years_active_service(self):
      return self._years_active_service
      
   @property
   def years_reserve_servce(self):
      return self._years_reserve_service
      
   @property
   def job_specialty(self):
      return self._job_specialty
      
   @property
   def years_job(self):
      return self._years_job
      
   @property
   def rank_class(self):
      return self._rank_class
      
   @property
   def location(self):
      return self._location
      
   @property
   def is_deployed(self):
      return self._is_deployed
      
   @property
   def months_deployed(self):
      return self._months_deployed
      
   @property
   def handgun_prof(self):
      return self._handgun_prof
      
   @property
   def rifle_prof(self):
      return self._rifle_prof
      
   @property
   def last_wp_fire(self):
      return self._last_wp_fire
      
   @property
   def range_hours(self):
      return self._range_hours
      
   @property
   def is_markstrained(self):
      return self._is_markstrained
      
   @property
   def last_markstraining(self):
      return self._last_markstraining   
      
   @property
   def is_gamer(self):
      return self._is_gamer
   
   @property
   def weekly_fps_hours(self):
      return self._weekly_fps_hours
   
   @property
   def general_anxiety(self):
      return self._general_anxiety
   
   @property
   def general_self_efficacy(self):
      return self._general_self_efficacy
      
   @property
   def general_motivation(self):
      return self._general_self_efficacy
      
                   
   #  Helper function to create weighted rvs
   def weighted_probs(outcomes, probabilities, size):
   	   temp = np.add.accumulate(probabilities)
      return outcomes[np.digitize(random_sample(size), temp)]

   #  methods for byte seq randomness from np. Useful for creating/filling in dataframe
   #  Return probabilities between 0 and 1
   def rand_prob(self, n):
      	return np.random.random((n)).tolist()

   #Return random values between low and high
   def rand_range(self, lo, high, n):
      return np.random.randint(lo,high,n).tolist()

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
		self.df.loc[((self.df['qualification_performance'] == "Marksman") & (self.df['shrange_hours'] > 300) &
					 (np.random.rand() > .40)),'qualification_performance'] = "Sharpshooter"
		self.df.loc[((self.df['qualification_performance']=="Sharpshooter") & (self.df['shrange_hours'] > 600) &
					 (np.random.rand() > .60)), 'qualification_performance'] = "Expert"
		self.df['hours_fps_week'] = self.rand_range(0,120, self.n)
		self.df['email_id'] = ["learner" + str(i) + "@example.com" for i in self.df.index]

	""" Recode factors to categorial variables """
	def recode(self):
		self.df['gender'] = ["Male" if i==1 else "Female" for i in self.df['gender']]
		self.df['age'] = [18 if i==1 else 19 if i==2 else 20 if i==3 else 21 if i==4 else 22 if i==5 else 23 if i==6 else 24 for i in self.df['education_level']]
		self.df['education_level'] = ["High School" if i==1 else "GED" if i==2 else "Some college" if i==3 else "Associate's Degree" if i==4 else "Bachelor's Degree" if i==5 else "Master's Degree" if i==6 else "PHD" for i in self.df['education_level']]
		self.df['handgun_prof_recoded'] = [0 if i <= 6 else 1 if i > 6 and i <= 15 else 2 for i in self.df['handgun_prof']]
		self.df['months_deployed'] = self.rand_range(0,36, self.n)

		print self.df['qualification_performance'].value_counts()

   
   
   