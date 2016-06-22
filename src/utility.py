# This is not a valid entry point.

import random
from numpy.random import random_sample, choice
import scipy.stats as sps
import numpy as np
import json
from math import factorial

import settings
import data_io

# Now we want to generate a list of dicts that will hold the learner's responses to survey questions
# whether for pre-lesson test or post-lesson test. That list is the survey_responses.
survey_responses = []

#Return random values between low and high
def rand_range(lo, high, n):
    return np.random.randint(lo,high,n).tolist()


def weighted_probs(outcomes, probabilities, size):
    temp = np.add.accumulate(probabilities)
    return outcomes[np.digitize(random_sample(size), temp)]


# Return probabilities between 0 and 1
def rand_prob(n):
    return np.random.random(n).tolist()


def calculateProbabilities(rating, post):
    if rating == "Unexperienced":
        temp = rand_prob(1)[0]
        if post != 1:
            if temp < 0.5:
                return 0
            else:
                return 1
        else:
            if temp < 0.3:
                return 0
            else:
                return 1
    elif rating == "Novice":
        if post != 1:
            outcomes = [0, 1]
            probs = [0.4, 0.6]
            return weighted_probs(outcomes, probs, 1)
        else:
            outcomes = [0, 1]
            probs = [0.25, 0.75]
            return weighted_probs(outcomes, probs, 1)
    elif rating == "Marksman":
        if post != 1:
            outcomes = [0, 1]
            probs = [0.2, 0.8]
            return weighted_probs(outcomes, probs, 1)
        else:
            outcomes = [0, 1]
            probs = [0.1, 0.9]
            return weighted_probs(outcomes, probs, 1)
    elif rating == "Sharpshooter":
        if post != 1:
            outcomes = [0, 1]
            probs = [0.15, 0.85]
            return weighted_probs(outcomes, probs, 1)
        else:
            outcomes = [0, 1]
            probs = [0.08, 0.92]
            return weighted_probs(outcomes, probs, 1)
    else:
        if post != 1:
            outcomes = [0, 1]
            probs = [0.05, 0.95]
            return weighted_probs(outcomes, probs, 1)
        else:
            outcomes = [0, 1]
            probs = [0.04, 0.96]
            return weighted_probs(outcomes, probs, 1)


def fillInAnswer(actual_matrix, rifle_exp, survey_index):
    prob_correctness = calculateProbabilities(rifle_exp, survey_index)
    # print prob_correctness
    l = [0, 0, 0, 0]
    ans_pos = [i for i, x in enumerate(actual_matrix) if x != 0]
    if prob_correctness == 0:
        # randomly select the index where it is not the right answer
        if ans_pos == 3:
            random_choice = random.choice([0, 1, 2])
            l[random_choice] = 1
            return l
        elif ans_pos == 2:
            random_choice = random.choice([0, 1, 3])
            l[random_choice] = 1
            return l
        elif ans_pos == 1:
            random_choice = random.choice([0, 2, 3])
            l[random_choice] = 1
            return l
        else:
            random_choice = random.choice([1, 2, 3])
            l[random_choice] = 1
            return l
    else:
        pos = ans_pos[0]
        l[pos] = 1
        return l


def calculateScore(answer_matrix, actual_matrix):
    score_matrix = [0, 0, 0, 0]
    for i in range(len(answer_matrix)):
        score_matrix[i] = answer_matrix[i] * actual_matrix[i]
    return max(score_matrix)


def constructResponse(learner_id, rifle_prof, blank_response, dir, post):
    '''
    In the wild, we would expect each of the above surveys (not pages) to be stored into the LRS as the learner completes
    them. For example, upon completing the pre-lesson attitude survey, GIFT should write an xAPI statement that
    {Agent: learner x, Verb: completed, Object: pre-lesson attitude survey for course abc with {Extension: responses to
    questions}, Context: JSON of survey questions and scoring rubric}

    For purposes of generating data in this case, we should just generate a single JSON object for each learner with
    questions and answers. Save the file with the: course name, testing date, learner ID. For example:
    'marksmanship_data_simulation-123405T_21April2016-learnerID.domain.json' or something like that, if we even need
    the date.
    '''
    r = blank_response
    r['learner_id'] = learner_id
    r['date_time'] = data_io.getTimeStamp('long')
    overall_points = []
    for so in r['Survey_Objects']:
        so['Answer'] = fillInAnswer(so['Scoring'], rifle_prof, post)
        so['Points'] = calculateScore(so['Answer'], so['Scoring'])
        overall_points.append(so['Points'])
    r['Overall_Score'] = sum(overall_points)

    with open(settings.DATA_DIR + dir + r['learner_id'] + ".json", "w") as data_out:
        json.dump(r, data_out)

    return r


def agreement(mu):
    k = [0, 1, 2, 3, 4, 5, 6]  # index values, not answer weights
    p = sps.binom.pmf(range(7), 5, mu)
    val = choice(k, 1, p=p)[0]
    return val

def fillBubbles(survey_object, ed, mea):
    o = survey_object
    answer_list = []
    survey_matrix = {"GED": [0.11, 0.10, 0.11, 0],
                     "High School": [0.13, 0.12, 0.11, 0],
                     "Some College": [0.26, 0.24, 0.20, 0],
                     "Associates Degree": [0.24, 0.24, 0.28, 0],
                     "Bachelors Degree": [0.32, 0.28, 0.38, 0],
                     "Masters Degree": [0.52, 0.80, 0.32, 0],
                     "Doctoral Degree": [0.51, 0.38, 0.36, 0]
                     }
    i = 0
    a = 0
    if o['Category'] == 'Motivation':
        a = 1 - mea[0]
        i = 0
    elif o['Category'] == 'Self-efficacy':
        a = 1 - mea[1]
        i = 1
    elif o['Category'] == 'Anxiety':
        a = mea[2]
        i = 2
    elif o['Category'] == 'Not Categorized':
        a = 1 - sum(mea)/3
        i = 3
    par = survey_matrix[ed][i]
    lo = min(a, par)
    hi = max(a, par)
    mu = random.triangular(lo, hi)
    for a in o['Answers']:
        iv = agreement(mu)
        ans = o['Weights'][iv]
        answer_list.append(ans)
    return answer_list


def surveyResponse(learner_id, ed, mea, blank_response, dir):
    r = blank_response
    r['learner_id'] = learner_id
    r['date_time'] = data_io.getTimeStamp('long')
    for o in r['Survey_Objects']:
        if o['Type'] == 'MatrixOfChoices':
            o['Answers'] = fillBubbles(o, ed, mea)
            o['Points'] = sum(o['Answers'])
            o['Min_Points'] = len(o['Answers'])
            o['Max_Points'] = len(o['Answers']) * 7
    with open(settings.DATA_DIR + dir + r['learner_id'] + ".json", "w") as data_out:
        json.dump(r, data_out)
    return r



