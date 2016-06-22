import pandas as pd 
import numpy as np 
from random import random
from numpy.random import random_sample
from copy import deepcopy

class ReaperSimulation(object):
    """Construct a data frame of students + marksman scores based on prior probabililties"""
    def __init__(self, n):
        super(ReaperSimulation, self).__init__()
		
		# Number of observations
        self.n = n

		# Create empty data frame with these column names + with n rows
        self.columns = ['date_time','student_id', 'fire_pos', 'round', 'range', 'isHit', 'shot_x', 'shot_y', 'ability']
        self.df = pd.DataFrame(0,index = xrange(n*40), columns = self.columns)

		# Create student_ids, accounting for the 40 rounds
        ids = ["learner" + str(i) + "@example.com" for i in xrange(n)]
        repl_ids = np.repeat(ids, 40)
        self.df['student_id'] = repl_ids

		# Create date and time
        self.df['date'] = ["2-JUN-2016" for i in xrange(len(self.df))]
        self.df['time'] = self.df['time'].apply(lambda v: np.random.randint(3,12))

        # Create FirePosition
        def initFirePos(n):
            fst_pos = np.repeat('Prone_Supported',20).tolist()
            snd_pos = np.repeat('Prone_Unsupported',10).tolist()
            trd_pos = np.repeat('Kneeling_Unsupported',10).tolist()

            temp = fst_pos + snd_pos + trd_pos
            self.df['fire_pos'] = temp * n


        # Create different rounds for each fired position as well as the range
        def initRoundRange(n):
            fst_pos = [i for i in range(1,21)]
            snd_pos = [i for i in range(1,11)]
            trd_pos = [i for i in range(1,11)]

            range_50 = np.repeat('50m',6).tolist()
            range_100 = np.repeat('100m',8).tolist()
            range_150 = np.repeat('150m',11).tolist()
            range_200 = np.repeat('200m',8).tolist()
            range_250 = np.repeat('250m',5).tolist()
            range_300 = np.repeat('300m',2).tolist()

            shot_range = range_50 + range_100 + range_150 + range_200 + range_250 + range_300
            temp = fst_pos + snd_pos + trd_pos

            self.df['round'] = temp*n
            self.df['range'] = shot_range*n


        def initAbility(n):
            # Calculate the probability of being expert, marksman, etc and update ability column
            levels = np.array([1,2,3,4])
            probs = np.array([0.15, 0.35, 0.45, 0.05])
            temp = weighted_probs(levels, probs, n).tolist()
            return_value = np.repeat(temp, 40)
            self.df['ability'] = ["Expert" if i==1 else "Sharpshooter" if i==2 else "Marksman" if i==3 else "Unqualified" for i in return_value]

            # Calcualate isHit
            self.df['isHit'] = [calc_outcomes(i)[0] for i in self.df['ability']]
            self.df['shot_x'] = [calc_hit(-2.5,3.5) if i == 'Hit' else np.nan if i == 'No-Fire' else calc_miss_x(
                np.random.randint(0,1)) for i in self.df['isHit']]
            self.df['shot_y'] = [calc_hit(-3.5,3.5) if i == 'Hit' else np.nan if i == 'No-Fire' else calc_miss_y(
                np.random.randint(0,1)) for i in self.df['isHit']]

        def calc_outcomes(level):
            expert = np.array([0.86, 0.13, 0.01])
            sharp_s = np.array([0.76, 0.22, 0.02])
            marksman = np.array([0.55, 0.41, 0.04])
            unqual = np.array([0.33, 0.59, 0.08])

            outcomes = np.array(['Hit','Miss', 'No-Fire'])

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
            else:
                t = np.random.uniform(2.6, 8)
                return float("{0:.2f}".format(t))

        def initTryAgain():
            # make a dataframe of unqulified students
            offset = len(self.df)
            df2 = deepcopy(self.df[self.df.ability == 'Unqualified'])
            #df2.reset_index(inplace=True)
            nn = len(df2)
            # Add the new date and time of the exercise
            dates = ["9-JUN-2016" for i in xrange(nn)]
            df2['date'] = dates
            times = df2['time'].apply(lambda v: np.random.randint(3,12))
            df2['time'] = times
            # Recalculate ability
            # Calculate the probability of being expert, marksman, etc and update ability column
            levels = np.array([1, 2, 3, 4])
            probs = np.array([0.10, 0.30, 0.55, 0.05])
            temp = weighted_probs(levels, probs, (nn / 40)).tolist()
            return_value = np.repeat(temp, 40)
            df2['ability'] = [
                "Expert" if i == 1 else "Sharpshooter" if i == 2 else "Marksman" if i == 3 else "Unqualified" for i in
                return_value]

            # Calcualate isHit
            is_hit = [calc_outcomes(i)[0] for i in df2.ability]
            df2['isHit'] = is_hit
            shot_x = [calc_hit(-2.5, 3.5) if i == 'Hit' else np.nan if i == 'No-Fire' else calc_miss_x(
                np.random.randint(0, 1)) for i in df2['isHit']]
            shot_y = [calc_hit(-3.5, 3.5) if i == 'Hit' else np.nan if i == 'No-Fire' else calc_miss_y(
                np.random.randint(0, 1)) for i in df2['isHit']]
            print len(shot_x)
            df2['shot_x'] = shot_x
            df2['shot_y'] = shot_y

            self.df = pd.concat([self.df, df2])
            self.df.reset_index(inplace=True)

        initFirePos(n)
        initRoundRange(n)
        initAbility(n)
        initTryAgain()

    def printData(self):
		print self.df

    def writeCSV(self, name):
        self.df.to_csv("../data/reaper/" + str(name) + ".csv", orient = 'index')

    def writeJSON(self, name):
        self.df.to_json("../data/reaper/" + str(name) + ".json", orient='records')


simulated_data = ReaperSimulation(100)
# simulated_data.printData()
simulated_data.writeCSV("record-fire-vCWD")
simulated_data.writeJSON("record-fire-vCWD")
