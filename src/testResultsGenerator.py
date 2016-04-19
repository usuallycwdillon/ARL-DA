#!/usr/bin/python

import numpy as np
from numpy.random import random_sample
import pandas as pd
import scipy.stats as st
import scipy as sc
import json
from itertools import chain



class surveyDataGenerator(object):
    """
    The surveyDataGenerator Class takes parameters for GIFT course survey JSON file and generates data to simulate
    students taking the survey. Though, in our test case survey, the pre- and post-course surveys are only written
    out once, so we have to fake having the whole survey.
    """
    def __init__(self, learners, survey):
        """
        survey is a GIFT survey, usually named 'something.course.survey.export'. The survey is nested json file that
        includes lists, so there's a lot of digging to get the useful stuff out of it. The goal is to automate the
        the process of identifying the structure and scoring of a survey.
        learners is a list of learners so that the survey is keyed to learners
        """
        super(surveyDataGenerator, self).__init__()

        self.learners = learners

        # Extract the surveys and identify them
        self.survey = open(survey).read()
        survey_json = json.loads(self.survey)
        context = survey_json['Survey_Context_Name']
        all_surveys = survey_json['Survey_Context_Surveys'][0]['Survey_Context_Survey_Survey']['Survey_Pages']
        num_surveys = len(all_surveys)
        all_survey_names = [s['Survey_Page_Name'] for s in all_surveys]
        all_survey_pages = [p['Elements'] for p in all_surveys]
        # Now, all the survey names are in the 'all_survey_names' list and corresponding pages in 'all_survey_pages' list

        # survey pages can have 1 or more 'Elements'





        ## This section

        # These are for setting up random attitude survey responses
        op_choices = ['Strongly Agree', 'Agree', 'Somewhat Agree', 'Neutral', 'Somewhat Disagree', 'Disagree',
                      'Strongly Disagree']
        op_weights = [.05, .23, .38, .18, .10, .05, .01] # there is no math here; just an intuitive pattern


        # Motivation
        question_list = survey_pages




        # Self-efficacy

        # Anxiety



    # Methods to generate the dataframe
    def init(self):



    ## Raghav's stat functions from the data_simulation script to shape the simulated data distributions

    # Helper function to create weighted rvs
        def weighted_probs(self, outcomes, probabilities, size):
        temp = np.add.accumulate(probabilities)
        return outcomes[np.digitize(random_sample(size), temp)]

    # Return probabilities between 0 and 1
    def rand_prob(self, n):
        return np.random.random((n)).tolist()

    # Return random values between low and high
    def rand_range(self, lo, high, n):
        return np.random.randint(lo, high, n).tolist()


    ## methods to extract questions structure from surveys

    #  Extract questions from a list
    def get_question_list(self, survey):
        q_list = survey_pages[]


survey = "../data/Domain/Marksmanship Course.course.surveys.export"
learners = "../data/person_data/tbd"
#