#!/usr/bin/python

import numpy as np
from numpy.random import random_sample
import random
import copy
import pandas as pd
import json
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

        #Read all of the learner profiles from the class
        def readClassProfiles(json_file):
            data = pd.read_json(json_file, orient= 'records')
            return data

        # Read in individual data from individual learner profile
        def readIndividualLearner(json_file):
            data = pd.read_json(json_file, typ='series', orient = 'records')
            return data

        self.learners = readClassProfiles(learners)

        # Extract the surveys and identify them
        self.survey = open(survey).read()
        survey_json = json.loads(self.survey)
        context = survey_json['Survey_Context_Name']
        all_surveys = survey_json['Survey_Context_Surveys'][0]['Survey_Context_Survey_Survey']['Survey_Pages']

        # Now we want to generate a list of dicts that will hold the learner's responses to survey questions
        # whether for pre-lesson test or post-lesson test. That list is the survey_responses.
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

        # Generate a survey response object
        for pages in all_surveys[1:3]:
            data = getSurveyData(pages)
            survey_responses.append(data)

        #print survey_responses[0]['SurveyName'], " is at index 0"


        # print survey_responses
        """
        The final value of 'survey_response' is a list containing two survey data dicts: one for the pre-lesson survey
        and one for the post-lesson survey. Construction of the responses should copy the list and iterate over the
        two survey responses and adding simulated responses.
        """



    # Raghav's stat functions from the data_simulation script to shape the simulated data distributions
        def weighted_probs(outcomes, probabilities, size):
            temp = np.add.accumulate(probabilities)
            return outcomes[np.digitize(random_sample(size), temp)]

    # Return probabilities between 0 and 1
        def rand_prob(n):
            return np.random.random((n)).tolist()


        def calculateProbabilities(rating, post):
            rankings = {"Unexperienced":1, "Novice":2, "Marksman":3, "Sharpshooter":4, "Expert":5}
            if rating == "Unexperienced":
                temp = rand_prob(1)[0]
                if post !=1:
                    if temp < 0.75:
                        return 0
                    else:
                        return 1
                else:
                    if temp < 0.375:
                        return 0
                    else:
                        return 1
            elif rating == "Novice":
                if post != 1:
                    outcomes = [0, 1]
                    probs = [0.45, 0.55]
                    return weighted_probs(outcomes, probs, 1)
                else:
                    outcomes = [0, 1]
                    probs = [0.4, 0.6]
                    return weighted_probs(outcomes, probs, 1)
            elif rating == "Marksman":
                if post != 1:
                    outcomes = [0, 1]
                    probs = [0.35, 0.65]
                    return weighted_probs(outcomes, probs, 1)
                else:
                    outcomes = [0, 1]
                    probs = [0.3, 0.7]
                    return weighted_probs(outcomes, probs, 1)
            elif rating == "Sharpshooter":
                if post != 1:
                    outcomes = [0, 1]
                    probs = [0.25, 0.75]
                    return weighted_probs(outcomes, probs, 1)
                else:
                    outcomes = [0, 1]
                    probs = [0.2, 0.8]
                    return weighted_probs(outcomes, probs, 1)
            else:
                if post != 1:
                    outcomes = [0, 1]
                    probs = [0.15, 0.85]
                    return weighted_probs(outcomes, probs, 1)
                else:
                    outcomes = [0, 1]
                    probs = [0.1, 0.9]
                    return weighted_probs(outcomes, probs, 1)

        def fillInAnswer(answer_matrix, actual_matrix, marksman_background, survey_index):
            prob_correctness = calculateProbabilities(marksman_background, survey_index)
            #print prob_correctness
            if prob_correctness == 0:
                # randomly select the index where it is not the right answer
                ans_pos = [i for i,x in enumerate(actual_matrix) if x != 0]
                # return temp[0]
                if ans_pos == 3:
                    random_choice = random.choice([0,1,2])
                    l = [0,0,0,0]
                    l[random_choice] = 1
                    return l
                elif ans_pos == 2:
                    random_choice = random.choice([0,1,3])
                    l = [0,0,0,0]
                    l[random_choice] = 1
                    return l
                elif ans_pos == 1:
                    random_choice = random.choice([0,2,3])
                    l = [0,0,0,0]
                    l[random_choice] = 1
                    return l
                else:
                    random_choice = random.choice([1,2,3])
                    l = [0,0,0,0]
                    l[random_choice] = 1
                    return l
            else:
                return actual_matrix

        def calculateScore(answer_matrix, actual_matrix):
            if answer_matrix == actual_matrix:
                return max(actual_matrix)
            else:
                return 0


        def constructResponses():
            learner_responses = []

            # iterate over each learner
            for i in range(len(self.learners.columns)):
                this_response = copy.deepcopy(survey_responses)
                for si, s in enumerate(this_response):
                    s['LearnerID'] = self.learners[i]['email_id']
                    overall_points = []
                    for so in s['SurveyObjects']:
                        so['Answer'] = fillInAnswer(so['Answer'], so['Scoring'], self.learners[i]['qualification_performance'], si)
                        so['Points'] = calculateScore(so['Answer'], so['Scoring'])
                        overall_points.append(so['Points'])
                    s['OverallScore'] = sum(overall_points)

                learner_responses.append(this_response)
            print "the learner responses list is ", len(learner_responses)

            for lr in learner_responses:
                with open("../data/pre_test_responses/" + lr[0]["LearnerID"] + ".json", "w") as data_out:
                    json.dump(lr[0], data_out)

                with open("../data/post_test_responses/" + lr[1]["LearnerID"] + ".json", "w") as data_out:
                    json.dump(lr[1], data_out)

        constructResponses()

    """
    In the wild, we would expect each of the above surveys (not pages) to be stored into the LRS as the learner completes
    them. For example, upon completing the pre-lesson attitude survey, GIFT should write an xAPI statement that
    {Agent: learner x, Verb: completed, Object: pre-lesson attitude survey for course abc with {Extension: responses to
    questions}, Context: JSON of survey questions and scoring rubric}

    For purposes of generating data in this case, we should just generate a single JSON object for each learner with
    questions and answers. Save the file with the: course name, testing date, learner ID. For example:
    'marksmanship_data_simulation-123405T_21April2016-learnerID.domain.json'
    """

survey = "../data/Domain/Marksmanship Course/ModifiedMarksmanshipCourse.course.surveys.export"
learners = "../data/class_data/class.json"

data = surveyDataGenerator(learners, survey)