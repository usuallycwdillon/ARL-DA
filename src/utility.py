# This is not a valid entry point.

import random
from numpy.random import random_sample, choice
import scipy.stats as sps
import numpy as np
import json
import pandas as pd
from math import factorial

import settings
import data_io

global learners

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
    if settings.verbosity == True:
        with open(settings.DATA_DIR + dir + r['learner_id'] + ".json", "w") as data_out:
            json.dump(r, data_out)
    return r


def agreement(mu):
    k = [0, 1, 2, 3, 4, 5, 6]  # index values, not answer weights
    p = sps.binom.pmf(range(7), 3, mu, loc=1)
    if sum(p) != 1.00:
        d = 1.00 - sum(p)
        if d > 0:
            p[6] = d
    val = choice(k, 1, p=p)[0]
    return val


def fillBubbles(survey_object, ed, mea):
    o = survey_object
    answer_list = []
    survey_matrix = {"GED": [0.21, 0.20, 0.21, 0],
                     "High School": [0.23, 0.22, 0.21, 0],
                     "Some College": [0.36, 0.34, 0.30, 0],
                     "Associates Degree": [0.34, 0.34, 0.38, 0],
                     "Bachelors Degree": [0.42, 0.38, 0.48, 0],
                     "Masters Degree": [0.62, 0.70, 0.42, 0],
                     "Doctoral Degree": [0.61, 0.48, 0.46, 0]
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
    mu = random.uniform(lo, hi)
    for a in o['Answers']:
        iv = agreement(mu)
        ans = o['Weights'][iv]
        answer_list.append(ans)
    return answer_list


def surveyResponse(learner_id, ed, mea, blank_response, dir):
    r = blank_response
    r['learner_id'] = learner_id
    r['date_time'] = data_io.getTimeStamp('long')
    r['total_score'] = 0
    r['min_score'] = 0
    for o in r['Survey_Objects']:
        if o['Type'] == 'MatrixOfChoices':
            o['Answers'] = fillBubbles(o, ed, mea)
            o['Points'] = sum(o['Answers'])
            r['total_score'] += o['Points']
            o['Min_Points'] = len(o['Answers'])
            r['min_score'] += o['Min_Points']
            o['Max_Points'] = len(o['Answers']) * 7

    if settings.verbosity == True:
        with open(settings.DATA_DIR + dir + r['learner_id'] + ".json", "w") as data_out:
            json.dump(r, data_out)
    return r


def getAttrs(learner):
    learner_id = learner.learner_id
    ed_level = learner.education_level
    min_attitude = learner.attitude_survey['min_score']
    tot_attitude = learner.attitude_survey['total_score']
    min_reaction = learner.reaction_survey['min_score']
    tot_reaction = learner.reaction_survey['total_score']
    min_satisfaction = learner.satisfaction_survey['min_score']
    tot_satisfaction = learner.satisfaction_survey['total_score']
    the_dict = {'learner_id': learner_id,
                'education_level': ed_level,
                'attitude_min': min_attitude,
                'attitude_total': tot_attitude,
                'reaction_min': min_reaction,
                'reaction_total': tot_reaction,
                'satisfaction_min': min_satisfaction,
                'satisfaction_total': tot_satisfaction
                }
    return the_dict

def getChis(crosstab, variable):
    chi2, p, dof, ex = sps.chi2_contingency(crosstab)
    print ex
    crit = sps.chi2.ppf(q=0.95, df=dof)
    if (crit < chi2):
        evaluation = True
    else:
        evaluation = False

    obs = crosstab.as_matrix()
    obs_list = obs.tolist()
    ex_list = ex.tolist()
    z_scores = sps.zmap(obs_list, ex_list)
    z_list = z_scores.tolist()
    z_indicators = []
    for z in z_list:
        z_sig = ["+" if i > 1.96 else "-" if i < -1.96 else " " for i in z]
        z_indicators.append(z_sig)

    results = {'chi-sq': chi2,
               'p-val': p,
               'eval': evaluation,
               'dof': dof,
               'explanans': variable,
               'expected': ex_list,
               'observed': obs_list,
               'z_scores': z_indicators,
               'row_lab': crosstab.index.tolist(),
               'col_lab': crosstab.columns.tolist()
               }
    return results


def binning(col, col_min, labels=None):
    lowest = col_min
    highest = col_min * 7
    half = (highest - lowest) / 2
    break_points = [lowest, (lowest + half), highest]
    if not labels:
        labels = range(len(break_points) - 1)
    colBin = pd.cut(col, bins=break_points, labels=labels, include_lowest=True)
    return colBin


def getRelFreqPlus(a_df, a_var):
    div = len(a_df)
    vc = a_df.groupby([a_var, 'education_level']).count()
    a_dict = vc.to_dict()
    w_dict = a_dict['learner_id']
    the_dict = {}
    for k, v in w_dict.iteritems():
        the_dict[k] = float(v) / div
    return the_dict


def repackage(the_expls, the_stats):
    the_dict = {}
    the_dict['explanandum'] = the_expls
    the_dict['explanans'] = the_stats['explanans']
    the_dict['active'] = the_stats['eval']

    the_data = {}
    the_data['rows'] = the_stats['row_lab']
    the_data['cols'] = the_stats['col_lab']
    the_data['obs_frequencies'] = the_stats['observed']
    the_data['exp_frequencies'] = the_stats['expected']
    the_data['significant'] = the_stats['z_scores']

    the_dict['data'] = the_data
    return the_dict