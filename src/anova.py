#!/usr/bin/python

import numpy as np
import os
import pandas as pd
import scipy.stats as st
import scipy as sc
import json
from statsmodels.stats.multicomp import pairwise_tukeyhsd

class VarianceAnalyzer(object):
    """
    This script reads in a list of learner profiles and a list of their scores (each is a json object) then performs
    an analysis of variance (ANOVA) to suggest which groups of learners are improving their marksmanship knowledge by
    completing the computer based training.
    """
    def __init__(self, learners, pre_scores, post_scores):
        """
        from learners we need their email identifier and their marksmanship rating
        from the two score lists we need the learner's total pre-lesson score and post-lesson score
        """
        super(VarianceAnalyzer, self).__init__()

        pre_json = []
        file_list = os.listdir(pre_scores)
        for f in file_list:
            of = open(os.path.join(pre_scores, f))
            data = json.load(of)
            pre_json.append(data)

        post_json = []
        file_list = os.listdir(post_scores)
        for f in file_list:
            of = open(os.path.join(post_scores, f))
            data = json.load(of)
            post_json.append(data)

        # Read all of the learner profiles from the class
        def readClassProfiles(json_file):
            data = pd.read_json(json_file, orient='records')
            return data

        # Read in individual data from individual learner profile
        def readIndividualLearner(dict_list):
            data = pd.DataFrame.from_dict(dict_list[0], orient='index')
            for each_dict in dict_list[1:]:
                this_data = pd.DataFrame.from_dict(each_dict, orient='index')
                data.append(this_data)
            return data

        def getDiff(row):
            return row['OverallScore_post'] - row['OverallScore_pre']

        self.profiles = readClassProfiles(learners)
        self.profiles = pd.DataFrame.transpose(self.profiles)
        self.pre_data = pd.DataFrame(pre_json)
        self.post_data = pd.DataFrame(post_json)
        self.scores = pd.merge(self.pre_data, self.post_data, on='LearnerID', suffixes=['_pre', '_post'])

        self.df = pd.merge(self.scores, self.profiles, left_on='LearnerID', right_on='email_id', how='outer')
        self.df['diffs'] = self.df.apply(lambda row: getDiff(row), axis=1)

        anova = pairwise_tukeyhsd(endog=self.df['diffs'], groups=self.df['qualification_performance'], alpha=0.05)

        print anova.summary()

pre_scores  = "../data/pre_test_responses"
post_scores = "../data/post_test_responses"
learners    = "../data/class_data/class.json"

analysis = VarianceAnalyzer(learners, pre_scores, post_scores)
