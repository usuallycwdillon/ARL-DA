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

    This faking part is a problem, because the survey is divided into 'pages' and there is no way to programmatically
    know which 'page' belongs to the testing sections. E.g., this survey file includes one attitude survey (with three
    pages--motivation, self-efficacy, anxiety), one lesson survey (with four pages--no reason, just an artifact of
    writing the survey.) and one reaction survey (with two pages (process and overall).  The attitude survey and the
    lesson survey should each appear twice (pre-lesson and post-lesson) and there should be a fourth survey on the
    lesson context (current location, course instructor name, etc.).

    Additionally, in order to show some significance of our simulated course, the simulated responses to the pre-lesson
    surveys and post-lesson surveys must show that some students benefit from taking the course.
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


        ## This section generates data for pre-lesson attitude survey responses

        # These are for setting up random attitude survey responses
        op_choices = ['Strongly Agree', 'Agree', 'Somewhat Agree', 'Neutral', 'Somewhat Disagree', 'Disagree',
                      'Strongly Disagree']
        pre_op_weights = [.05, .23, .38, .18, .10, .05, .01] # there is no math here; just an intuitive pattern


        # Motivation - prelesson
        question_list = survey_pages



        # Self-efficacy - pre-lesson


        # Anxiety - pre-lesson


        ## This section generates data for post-lesson attitude survey responses
        post_op_weights = [.39, .29, .14, .09, .05, .03, .01]  # there is no math here; just an intuitive pattern

        #  Motivation - post-lesson


        #  Self-efficacy - post-lesson


        #  Anxiety - post-lesson


        ## This section generates data for pre-lesson knowledge and skills

        #  Section 1

        #  Section 2

        #  Section 3

        #  Section 4



        ## This section generates data for post-lesson knowledge and skills


        #  Section 1

        #  Section 2

        #  Section 3

        #  Section 4


        ## This section generates data for post-lesson Reactions


    # Methods to generate the dataframe
    def init(self):


    """
    In the wild, we would expect each of the above surveys (not pages) to be stored into the LRS as the learner completes
    them. For example, upon completing the pre-lesson attitude survey, GIFT should write an xAPI statement that
    {Agent: learner x, Verb: completed, Object: pre-lesson attidude survey for course abc with {Extension: responses to
    questions}, Context: JSON of survey questions and scoring rubric}

    For purposes of generating data in this case, we should just generate a single JSON object with questions and answers
    for each learner. Save the file with the: course name, testing date, learner ID. For example:
    'marksmanship_data_simulation-123405T_21April2016-learnerID.domain.json'
    """


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