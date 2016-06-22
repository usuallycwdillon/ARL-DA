## This is not a valid starting point. 
#  This class file defines agents that activate in the learner_xSim.py file.

import random
import numpy as np 
from scipy import stats as sps
import csv
from copy import deepcopy
import data_io
import settings
import utility

class Learner():
    '''
    '''
    def __init__(self, id):
        '''
        Given an integer as an id, generate a learner-agent with attributes described by empirical data we have on
        soldiers taking the basic rifle marksmanship training course.
        :param id:
        '''
        ed_outcomes = np.array(["High School", "GED", "Some College", "Associates Degree", "Bachelors Degree", "Masters Degree", "Doctoral Degree"])
        ed_probs = np.array([0.6, 0.3, 0.05, 0.02, 0.015, 0.01 , 0.005])
        pistol_ex = np.array(["None", "Novice", "Experienced"])
        pistol_probs = np.array([0.5, 0.3, 0.2])
        rifle_ex = np.array(["None", "Novice", "Marksman", "Sharpshooter", "Expert"])
        rifle_probs = np.array([0.15, 0.30, 0.25, 0.20, 0.10])
        sex = np.array(["Male", "Female"])
        sex_prob = np.array([0.70, 0.30])
        yes_no = np.array(["Yes", "No"])

        self._learner_id = 'learner' + str(id) + '@example.com'
        self._age = int(random.triangular(18, 45, 19))
        self._sex = str(np.random.choice(sex, 1, p=sex_prob)[0])
        self._education_level = str(np.random.choice(ed_outcomes, 1, p = ed_probs)[0])
        self._years_active_service = int(random.triangular(0, 20, 2))
        self._years_reserve_service = int(random.triangular(0, 8, 0))
        self._years_job = max((self._years_active_service - 1) + (self._years_reserve_service - 1), 0)

        if self._age > 22 and random.random() < .30:
            self._rank_class = "Officer"
        else:
            self._rank_class = "Enlisted"

        self._location = random.choice(["A", "B", "C", "D"])
        self._is_deployed = str(np.random.choice(yes_no, 1, p=[0.1, 0.9])[0])

        if self.is_deployed == "No":
            self._months_deployed = 0
        else:
            self._months_deployed = int(np.random.randint(0, 23, size=1)[0])

        self._handgun_prof = str(np.random.choice(pistol_ex, 1, p=pistol_probs)[0])
        self._rifle_prof = str(np.random.choice(rifle_ex, 1, p=rifle_probs)[0])

        if self.rifle_prof != "None" and self.rifle_prof != "Novice":
            self._last_markstraining = int(np.random.randint(1, 12, size=1)[0])
        else:
            self._last_markstraining = 0

        if self.rifle_prof != "None" and self.rifle_prof != "Novice":
            self._is_markstrained = "No"
        else:
            self._is_markstraining = "Yes"

        if self.rifle_prof != "None" and self.rifle_prof != "Novice":
            self._last_wp_fire = int(np.random.randint(1, 6, size=1)[0])
        else:
            self._last_wp_fire = 0

        if self.rifle_prof == "None" or self.rifle_prof == "Novice":
            self._range_hours = 0
        elif self.rifle_prof == "Marksman":
            self._range_hours = int(random.triangular(39, 312, mode=78))
        elif self.rifle_prof == "Sharpshooter":
            self._range_hours = int(random.triangular(39, 312, mode=117))
        else:
            self._range_hours = int(random.triangular(39, 312, mode=234))

        self._is_gamer = random.choice(['Yes', 'No'])

        if self._is_gamer =='Yes':
            self._weekly_fps_hours = int(random.triangular(2, 40, mode=12))
        else:
            self._weekly_fps_hours = 0

        self._general_anxiety = random.triangular(0, 1, mode=0.15)
        self._general_self_efficacy = random.triangular(0, 1, mode=0.70)
        self._general_motivation = random.triangular(0, 1, mode=0.85)

        self._pre_lesson_test = {}
        self._post_lesson_test = {}
        self._attitude_survey = {}
        self._record_fire = []
        self._reaction_survey = {}
        self._satisfaction_survey = {}


    @property
    def learner_id(self):
        return self._learner_id

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

    @property
    def attitude_survey(self):
        return self._attitude_survey

    @attitude_survey.setter
    def attitude_survey(self, record):
        self._attitude_survey = record

    @property
    def pre_lesson_test(self):
       return self._pre_lesson_test

    @pre_lesson_test.setter
    def pre_lesson_test(self, record):
       self._pre_lesson_test = {}

    @property
    def post_lesson_test(self):
       return self._post_lesson_test

    @post_lesson_test.setter
    def post_lesson_test(self, record):
       self._post_lesson_test = {}

    @property
    def record_fire(self):
       return self._record_fire

    @record_fire.setter
    def record_fire(self, record):
        self._record_fire.append = record

    @property
    def satisfaction_survey(self):
        return self._satisfaction_survey

    @satisfaction_survey.setter
    def satisfaction_survey(self, record):
        self._satisfaction_survey = record

    @property
    def reaction_survey(self):
        return self._reaction_survey

    @reaction_survey.setter
    def reaction_survey(self, record):
        self._reaction_survey = record


    def takeAttitudeSurvey(self):
        # No magic here. We must already know that the 0th index of survey_models is the attitude survey
        model = deepcopy(settings.survey_models[0]) # each learner needs their very own deep copy of the model
        mea = [self._general_motivation,  self.general_self_efficacy, self.general_anxiety]
        response = utility.surveyResponse(self.learner_id, self.education_level, mea, model, "attitudes/")
        self.attitude_survey = response


    def takePreLessonSurvey(self):
        model = deepcopy(settings.survey_models[1])
        # The final parameter below indicates whether this is the pre-lesson (0) or post-lesson (1)
        response = utility.constructResponse(self.learner_id, self.rifle_prof, model, "pre_test/", 0)
        self.pre_lesson_test = response


    def takePostLessonSurvey(self):
        model = deepcopy(settings.survey_models[2])
        # The final parameter below indicates whether this is the pre-lesson (0) or post-lesson (1)
        response = utility.constructResponse(self.learner_id, self.rifle_prof, model, "post_test/", 1)
        self.post_lesson_test = response


    def takeReactionSurvey(self):
        model = deepcopy(settings.survey_models[4])
        mea = [self._general_motivation,  self.general_self_efficacy, self.general_anxiety]
        response = utility.surveyResponse(self.learner_id, self.education_level, mea, model, "reaction/")
        self.reaction_survey = response


    def takeSatisfactionSurvey(self):
        model = deepcopy(settings.survey_models[5])
        mea = [self._general_motivation,  self.general_self_efficacy, self.general_anxiety]
        response = utility.surveyResponse(self.learner_id, self.education_level, mea, model, "satisfaction/")
        self.satisfaction_survey = response


    def doRecordFireExercise(self):
        pass


class Course_Offering():
    '''
    A course offering scoops up a bunch of learner agents and puts them through the paces of surveys and record fire
    testing, etc.
    '''
    def __init__(self, section):
        self._section = section
        self._location = random.choice(["E", "F", "G", "H"])
        self._instructor = random.choice(["X", "Y"])
        self._enrollment = []

    @property
    def location(self):
        return self._location

    @property
    def section(self):
        return self._section

    @property
    def instructor (self):
        return self._instructor

    @property
    def enrollment(self):
        return self._enrollment

    @enrollment.setter
    def enrollment(self, value):
        self._enrollment.append(value)

    def append(self, value):
        self._enrollment = self._enrollment + [value]


class ReaperSimulation(object):
    """Construct a data frame of students + marksman scores based on prior probabilities"""
    def __init__(self, learner):
        super(ReaperSimulation, self).__init__()

        # Number of observations
        self.n = n

        # Create empty data frame with these column names + with n rows
        self.columns = ['date_time', 'student_id', 'fire_pos', 'round', 'range', 'isHit', 'shot_x', 'shot_y', 'ability']
        self.df = pd.DataFrame(0, index=xrange(n * 40), columns=self.columns)

        # Create student_ids, accounting for the 40 rounds
        ids = ["learner" + str(i) + "@example.com" for i in xrange(n)]
        repl_ids = np.repeat(ids, 40)
        self.df['student_id'] = repl_ids

        # Create date and time
        self.df['date'] = ["2-JUN-2016" for i in xrange(len(self.df))]
        self.df['time'] = self.df['time'].apply(lambda v: np.random.randint(3, 12))

        # Create FirePosition
        def initFirePos(n):
            fst_pos = np.repeat('Prone_Supported', 20).tolist()
            snd_pos = np.repeat('Prone_Unsupported', 10).tolist()
            trd_pos = np.repeat('Kneeling_Unsupported', 10).tolist()

            temp = fst_pos + snd_pos + trd_pos
            self.df['fire_pos'] = temp * n

        # Create different rounds for each fired position as well as the range
        def initRoundRange(n):
            fst_pos = [i for i in range(1, 21)]
            snd_pos = [i for i in range(1, 11)]
            trd_pos = [i for i in range(1, 11)]

            range_50 = np.repeat('50m', 6).tolist()
            range_100 = np.repeat('100m', 8).tolist()
            range_150 = np.repeat('150m', 11).tolist()
            range_200 = np.repeat('200m', 8).tolist()
            range_250 = np.repeat('250m', 5).tolist()
            range_300 = np.repeat('300m', 2).tolist()

            shot_range = range_50 + range_100 + range_150 + range_200 + range_250 + range_300
            temp = fst_pos + snd_pos + trd_pos

            self.df['round'] = temp * n
            self.df['range'] = shot_range * n

        def initAbility(n):
            # Calculate the probability of being expert, marksman, etc and update ability column
            levels = np.array([1, 2, 3, 4])
            probs = np.array([0.15, 0.35, 0.45, 0.05])
            temp = weighted_probs(levels, probs, n).tolist()
            return_value = np.repeat(temp, 40)
            self.df['ability'] = [
                "Expert" if i == 1 else "Sharpshooter" if i == 2 else "Marksman" if i == 3 else "Unqualified" for i in
                return_value]

            # Calcualate isHit
            self.df['isHit'] = [calc_outcomes(i)[0] for i in self.df['ability']]
            self.df['shot_x'] = [calc_hit(-2.5, 3.5) if i == 'Hit' else np.nan if i == 'No-Fire' else calc_miss_x(
                np.random.randint(0, 1)) for i in self.df['isHit']]
            self.df['shot_y'] = [calc_hit(-3.5, 3.5) if i == 'Hit' else np.nan if i == 'No-Fire' else calc_miss_y(
                np.random.randint(0, 1)) for i in self.df['isHit']]

        def calc_outcomes(level):
            expert = np.array([0.86, 0.13, 0.01])
            sharp_s = np.array([0.76, 0.22, 0.02])
            marksman = np.array([0.55, 0.41, 0.04])
            unqual = np.array([0.33, 0.59, 0.08])

            outcomes = np.array(['Hit', 'Miss', 'No-Fire'])

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

        def calc_hit(lo, hi):
            t = np.random.uniform(lo, hi)
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
            # df2.reset_index(inplace=True)
            nn = len(df2)
            # Add the new date and time of the exercise
            dates = ["9-JUN-2016" for i in xrange(nn)]
            df2['date'] = dates
            times = df2['time'].apply(lambda v: np.random.randint(3, 12))
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
        self.df.to_csv("../data/reaper/" + str(name) + ".csv", orient='index')

    def writeJSON(self, name):
        self.df.to_json("../data/reaper/" + str(name) + ".json", orient='records')

