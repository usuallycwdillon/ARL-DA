import pandas as pd 
import numpy as np 
from numpy.random import random_sample
import scipy.stats as st 
import matplotlib.pyplot as plt

class ReaperSimulation(object):
	"""Construct a data frame of students + marksman scores based on prior probabililties"""
	def __init__(self, n, csv_file):
		super(ReaperSimulation, self).__init__()
		

		# First read in the data from class profiles

		temp_df = pd.read_csv(csv_file)
		print temp_df.columns
		# temp_df = temp_df.drop(temp_df.columns[[0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]], axis=1)




		# Number of observations
		self.n = n

		# Create empty data frame with these column names + with n rows
		self.columns = ['date','id', 'fire_pos', 'round', 'range', 'time', 'isHit', 'shot_x', 'shot_y']
		self.df = pd.DataFrame(0,index = xrange(n*40), columns = self.columns)

		# Create student_ids, accounting for the 40 rounds
		ids = ["learner"+str(i)+"@example.com" for i in xrange(n)]
		repl_ids = np.repeat(ids, 40)
		self.df['student_id'] = repl_ids

		# Create date and time
		self.df['date'] = "15-JUN-16"
		self.df['time'] = self.df['time'].apply(lambda v: np.random.randint(3,12))

		# Create FirePosition
		def initFirePos(n):
			# handle positions
			fst_pos = np.repeat(1,20).tolist()
			snd_pos = np.repeat(2,10).tolist()
			trd_pos = np.repeat(3,10).tolist()

			temp = fst_pos + snd_pos + trd_pos
			return temp*n

		# Create different rounds for each fired position as well as the range
		def initRoundRange(n):
			fst_pos = [i for i in range(1,21)]
			snd_pos = [i for i in range(1,11)]
			trd_pos = [i for i in range(1,11)]

			range_50 = np.repeat(50,6).tolist()
			range_100 = np.repeat(100,8).tolist()
			range_150 = np.repeat(150,11).tolist()
			range_200 = np.repeat(200,8).tolist()
			range_250 = np.repeat(250,5).tolist()
			range_300 = np.repeat(300,2).tolist()

			shot_range = range_50 + range_100 + range_150 + range_200 + range_250 + range_300
			temp = fst_pos + snd_pos + trd_pos

			return temp*n, shot_range*n

		def initAbility(n):
			# Calculate the probability of being expert, marksman, etc and update ability column
			levels = np.array([1,2,3,4])
			probs = np.array([0.15, 0.35, 0.45, 0.05])
			temp = weighted_probs(levels, probs, n).tolist()
			return_value = np.repeat(temp, 40)
			ability = ["Expert" if i==1 else "Sharpshooter" if i==2 else "Marksman" if i==3 else "Unqualified" for i in return_value]
			
			# Calcualate isHit
			hit = [calc_outcomes(i)[0] for i in ability]
			shot_x = [calc_hit(-2.5,3.5) if i == "Yes" else calc_miss_x(np.random.randint(0,1)) if i == "No" else "NA" for i in hit]
			shot_y = [calc_hit(-3.5,3.5) if i == "Yes" else calc_miss_y(np.random.randint(0,1)) if i == "No" else "NA" for i in hit]
			return ability, hit, shot_x, shot_y

		def calc_outcomes(level):
			expert = np.array([0.86, 0.13, 0.01])
			sharp_s = np.array([0.76, 0.22, 0.02])
			marksman = np.array([0.55, 0.43, 0.02])
			unqual = np.array([0.33, 0.65, 0.02])

			outcomes = np.array(["Yes","No","No Fire"])
			
			if level == "Expert":
				return weighted_probs(outcomes, expert, 40)
			elif level == "Sharpshooter":
				return weighted_probs(outcomes, sharp_s, 40)
			elif level == "Marksman":
				return weighted_probs(outcomes, marksman, 40)
			else:
				return weighted_probs(outcomes, unqual, 40)


		def weighted_probs(outcomes, probabilities, size):
			temp = np.add.accumulate(probabilities)
			return outcomes[np.digitize(random_sample(size), temp)]

		def calc_hit(lo,hi):
			t = np.random.uniform(lo,hi)
			return float("{0:.2f}".format(t))

		def calc_miss_x(op):
			if op == 0:
				t = np.random.uniform(-21, -3.6) 
				return float("{0:.2f}".format(t))
			else:
				t = np.random.uniform(3.6, 21) 
				return float("{0:.2f}".format(t))

		def calc_miss_y(op):
			if op == 0:
				t = np.random.uniform(-8, -2.6) 
				return float("{0:.2f}".format(t))
			if op == 1:
				t = np.random.uniform(2.6, 8) 
				return float("{0:.2f}".format(t))

		def createUnqualified(n):
			# select random number less than 10000 - 500
			lo = np.random.randint(self.n-(0.05*self.n))
			hi = lo + 0.05*self.n
			# create new dataframe that will be merged with main df
			df = pd.DataFrame(0,index = xrange(n*40), columns = self.columns)
			df['date'] = "16-May-16"
			df['time'] = df['time'].apply(lambda v: np.random.randint(3,12))
			df['fire_pos'] = initFirePos(n)
			df['round'], df['range'] = initRoundRange(n)
			df['ability'], df['isHit'], df['shot_x'], df['shot_y'] = initAbility(n)
			ids = ["learner"+str(i)+"@example.com" for i in xrange(int(lo),int(hi))]
			repl_ids = np.repeat(ids, 40)
			df['student_id'] = repl_ids
			return df


		self.df['fire_pos'] = initFirePos(n)
		self.df['round'], self.df['range'] = initRoundRange(n)
		self.df['ability'], self.df['isHit'], self.df['shot_x'], self.df['shot_y'] = initAbility(n)
		temp_df = createUnqualified(int(0.05*n))
		dfs = [self.df, temp_df]
		self.df = pd.concat(dfs)

	def printData(self):
		print self.df

	def writeCSV(self,name):
		self.df.to_csv(str(name) + ".csv", orient = 'index')

	def constructPlot(self):
		x = self.df['shot_x']
		y = self.df['shot_y']
		try:
			plt.scatter(x,y)
			plt.show()
		except:
			pass
		
		

simulated_data = ReaperSimulation(10000, "test.csv")
# simulated_data.printData()
# simulated_data.writeCSV("reaper_data")
