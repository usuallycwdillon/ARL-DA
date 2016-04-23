#!/usr/bin/python

import numpy as np
from numpy.random import random_sample
import pandas as pd
import scipy.stats as st
import scipy as sc
import json
from itertools import chain
from time import gmtime, strftime


class surveyDataGenerator(object):
    """
    The surveyDataGenerator Class takes parameters for GIFT course survey JSON file and generates data to simulate
    students taking the survey. Though, in our test case survey, the pre- and post-course surveys are only written
    out once, so we have to fake having the whole survey. See comments in each section for instructions specific to that
    section of the survey.
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

        # Now, all the survey names are in the 'all_survey_names' list and corresponding pages in 'all_survey_pages' list

        # Now we want to generate answers for all questions in each survey in turn

        survey_responses = []

        def getSurveyData(pages):
            page_elements = pages['Elements']
            data_objects = getElementData(page_elements)
            survey_response = {"SurveyName":pages['Survey_Page_Name'],
                               "LearnerID":"",
                               "Time":strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()),
                               "SurveyObjects":data_objects
                               }
            return survey_response

        def getElementData(page_elements):
            survey_objects = []
            for each_element in page_elements:
                question_text = each_element['Question']['Question_Text']
                question_type = each_element['Question']['Type']
                scoring = each_element['Properties']['Answer_Weights']['Property_Value']['StringPayload']
                scoring = scoring.split(',')
                scoring = [int(s) for s in scoring]
                survey_object = {"Question":question_text,
                                 "Type":question_type,
                                 "Answer":[0,0,0,0],
                                 "Scoring":scoring,
                                 "Points":0
                                 }
                survey_objects.append(survey_object)
            return survey_objects

        for pages in all_surveys[1:3]:
            data = getSurveyData(pages)
            survey_responses.append(data)


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


survey = "../data/Domain/Marksmanship Course/ModifiedMarksmanshipCourse.course.surveys.export"
learners = "../data/person_data/tbd"
#