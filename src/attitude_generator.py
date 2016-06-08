import numpy as np 
import pandas as pd 
import scipy.stats

class AttitudeData(object):
	"""Construct the motivation, self-efficacy, and anxiety scores 
	   for the survey questions based on education"""
	def __init__(self, n, csv_file):
		super(AttitudeData, self).__init__()

		# First read in class profile data to get education
		temp_df = pd.read_csv(csv_file)
		temp_df = temp_df.drop(temp_df.columns[[0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]], axis=1)

		# N observations 15 questions (5 for motivation, 5 self-efficacy, and 5 anxiety)
		self.n = n
		cols = ["q" + str(i) for i in range(1,16)]
		cols.insert(0, "id")
		cols.insert(1, "attitude")

		survey_matrix = {"GED": [4.47, 4.5, 4.7, 4.2], 
				  "High School": [4.40, 4.4, 4.6, 4.2],
				  "Some College": [3.7, 3.8, 4, 3.3],
				  "Associates": [3.8, 3.8, 3.6, 4],
				  "Bachelors": [3.4, 3.4, 3.1, 3.7],
				  "Masters": [2.6, 1, 3.4, 3.4],
				  "PhD": [2.43, 1, 3.2, 3.1]}

		# Create data frame
		self.df = pd.DataFrame(0, index = xrange(n), columns = cols)
		self.df['id'] = ["learner" + str(i) + "@example.com" for i in xrange(n)]
		self.df = pd.merge(self.df, temp_df)
		

		def init():
			for index,row in self.df.iterrows():
				if row['education_level'] == "GED":
					mot = scipy.stats.norm(survey_matrix["GED"][1], 0.12)
					eff = scipy.stats.norm(survey_matrix["GED"][2], 0.16)
					anx = scipy.stats.norm(survey_matrix["GED"][3], 0.3)
					att = scipy.stats.norm(survey_matrix["GED"][0], 0.4)
					self.df.ix[index, 'q1'],self.df.ix[index, 'q2'],self.df.ix[index, 'q3'],self.df.ix[index, 'q4'],self.df.ix[index, 'q5'] = mot.rvs(size=5)[0:5]
					self.df.ix[index, 'q6'],self.df.ix[index, 'q7'],self.df.ix[index, 'q8'],self.df.ix[index, 'q9'],self.df.ix[index, 'q10'] = eff.rvs(size=5)[0:5]
					self.df.ix[index, 'q11'],self.df.ix[index, 'q12'],self.df.ix[index, 'q13'],self.df.ix[index, 'q14'],self.df.ix[index, 'q15'] = anx.rvs(size=5)[0:5]
					self.df.ix[index, 'attitude'] = att.rvs(1)

				elif row['education_level'] == "High School":
					mot = scipy.stats.norm(survey_matrix["High School"][1], 0.28)
					eff = scipy.stats.norm(survey_matrix["High School"][2], 0.22)
					anx = scipy.stats.norm(survey_matrix["High School"][3], 0.3)
					att = scipy.stats.norm(survey_matrix["High School"][0], 0.36)
					self.df.ix[index, 'q1'],self.df.ix[index, 'q2'],self.df.ix[index, 'q3'],self.df.ix[index, 'q4'],self.df.ix[index, 'q5'] = mot.rvs(size=5)[0:5]
					self.df.ix[index, 'q6'],self.df.ix[index, 'q7'],self.df.ix[index, 'q8'],self.df.ix[index, 'q9'],self.df.ix[index, 'q10'] = eff.rvs(size=5)[0:5]
					self.df.ix[index, 'q11'],self.df.ix[index, 'q12'],self.df.ix[index, 'q13'],self.df.ix[index, 'q14'],self.df.ix[index, 'q15'] = anx.rvs(size=5)[0:5]
					self.df.ix[index, 'attitude'] = att.rvs(1)

				elif row['education_level'] == "Some College":
					mot = scipy.stats.norm(survey_matrix["Some College"][1], 0.31)
					eff = scipy.stats.norm(survey_matrix["Some College"][2], 0.28)
					anx = scipy.stats.norm(survey_matrix["Some College"][3], 0.3)
					att = scipy.stats.norm(survey_matrix["Some College"][0], 0.32)
					self.df.ix[index, 'q1'],self.df.ix[index, 'q2'],self.df.ix[index, 'q3'],self.df.ix[index, 'q4'],self.df.ix[index, 'q5'] = mot.rvs(size=5)[0:5]
					self.df.ix[index, 'q6'],self.df.ix[index, 'q7'],self.df.ix[index, 'q8'],self.df.ix[index, 'q9'],self.df.ix[index, 'q10'] = eff.rvs(size=5)[0:5]
					self.df.ix[index, 'q11'],self.df.ix[index, 'q12'],self.df.ix[index, 'q13'],self.df.ix[index, 'q14'],self.df.ix[index, 'q15'] = anx.rvs(size=5)[0:5]
					self.df.ix[index, 'attitude'] = att.rvs(1)

				elif row['education_level'] == "Associates":
					mot = scipy.stats.norm(survey_matrix["Associates"][1], 0.3)
					eff = scipy.stats.norm(survey_matrix["Associates"][2], 0.26)
					anx = scipy.stats.norm(survey_matrix["Associates"][3], 0.2)
					att = scipy.stats.norm(survey_matrix["Associates"][0], 0.27)
					self.df.ix[index, 'q1'],self.df.ix[index, 'q2'],self.df.ix[index, 'q3'],self.df.ix[index, 'q4'],self.df.ix[index, 'q5'] = mot.rvs(size=5)[0:5]
					self.df.ix[index, 'q6'],self.df.ix[index, 'q7'],self.df.ix[index, 'q8'],self.df.ix[index, 'q9'],self.df.ix[index, 'q10'] = eff.rvs(size=5)[0:5]
					self.df.ix[index, 'q11'],self.df.ix[index, 'q12'],self.df.ix[index, 'q13'],self.df.ix[index, 'q14'],self.df.ix[index, 'q15'] = anx.rvs(size=5)[0:5]
					self.df.ix[index, 'attitude'] = att.rvs(1)


				elif row['education_level'] == "Bachelors":
					mot = scipy.stats.norm(survey_matrix["Associates"][1], 0.26)
					eff = scipy.stats.norm(survey_matrix["Associates"][2], 0.16)
					anx = scipy.stats.norm(survey_matrix["Associates"][3], 0.3)
					att = scipy.stats.norm(survey_matrix["Associates"][0], 0.4)
					self.df.ix[index, 'q1'],self.df.ix[index, 'q2'],self.df.ix[index, 'q3'],self.df.ix[index, 'q4'],self.df.ix[index, 'q5'] = mot.rvs(size=5)[0:5]
					self.df.ix[index, 'q6'],self.df.ix[index, 'q7'],self.df.ix[index, 'q8'],self.df.ix[index, 'q9'],self.df.ix[index, 'q10'] = eff.rvs(size=5)[0:5]
					self.df.ix[index, 'q11'],self.df.ix[index, 'q12'],self.df.ix[index, 'q13'],self.df.ix[index, 'q14'],self.df.ix[index, 'q15'] = anx.rvs(size=5)[0:5]
					self.df.ix[index, 'attitude'] = att.rvs(1)

				elif row['education_level'] == "Masters":
					mot = scipy.stats.norm(survey_matrix["Associates"][1], 0.05)
					eff = scipy.stats.norm(survey_matrix["Associates"][2], 0.12)
					anx = scipy.stats.norm(survey_matrix["Associates"][3], 0.04)
					att = scipy.stats.norm(survey_matrix["Associates"][0], 0.1)
					self.df.ix[index, 'q1'],self.df.ix[index, 'q2'],self.df.ix[index, 'q3'],self.df.ix[index, 'q4'],self.df.ix[index, 'q5'] = mot.rvs(size=5)[0:5]
					self.df.ix[index, 'q6'],self.df.ix[index, 'q7'],self.df.ix[index, 'q8'],self.df.ix[index, 'q9'],self.df.ix[index, 'q10'] = eff.rvs(size=5)[0:5]
					self.df.ix[index, 'q11'],self.df.ix[index, 'q12'],self.df.ix[index, 'q13'],self.df.ix[index, 'q14'],self.df.ix[index, 'q15'] = anx.rvs(size=5)[0:5]
					self.df.ix[index, 'attitude'] = att.rvs(1)

				else:
					mot = scipy.stats.norm(survey_matrix["PhD"][1], 0.05)
					eff = scipy.stats.norm(survey_matrix["PhD"][2], 0.16)
					anx = scipy.stats.norm(survey_matrix["PhD"][3], 0.2)
					att = scipy.stats.norm(survey_matrix["PhD"][0], 0.05)
					self.df.ix[index, 'q1'],self.df.ix[index, 'q2'],self.df.ix[index, 'q3'],self.df.ix[index, 'q4'],self.df.ix[index, 'q5'] = mot.rvs(size=5)[0:5]
					self.df.ix[index, 'q6'],self.df.ix[index, 'q7'],self.df.ix[index, 'q8'],self.df.ix[index, 'q9'],self.df.ix[index, 'q10'] = eff.rvs(size=5)[0:5]
					self.df.ix[index, 'q11'],self.df.ix[index, 'q12'],self.df.ix[index, 'q13'],self.df.ix[index, 'q14'],self.df.ix[index, 'q15'] = anx.rvs(size=5)[0:5]
					self.df.ix[index, 'attitude'] = att.rvs(1)
		
		

		init()
		
	def to_csv(self,name):
			self.df.to_csv(name)

	def printDF(self):
		print self.df


data = AttitudeData(10000, "test.csv")
data.printDF()
data.to_csv("attitude_data")


