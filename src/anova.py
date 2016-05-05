#!/usr/bin/python

import numpy as np
import os
import pandas as pd
import scipy.stats as st
import scipy as sc
import json
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols

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
        self.df['pos_diffs'] = self.df.apply(lambda row: max(getDiff(row), 0), axis=1)
        self.df['diffs'] = self.df.apply(lambda row: getDiff(row), axis=1)
        self.df['adj_diffs'] = self.df.apply(lambda row: row['diffs'] + 100, axis=1)

        fps = []
        def gamer(s):
            if s == 0:
                fps.append('no')
            else:
                fps.append('yes')

        self.df.apply(lambda row: gamer(row['is_gamer']), axis=1)
        self.df['fps'] = fps

        def accumulate(row):
            return "_".join([row['qualification_performance'], row['education_level'], row['fps']])

        self.df['all'] = self.df.apply(lambda row: accumulate(row), axis=1)

        ## Huom! It may or may not be necessary to elementally convert the numpy.ndarrays into plain python lists and
        #  elementally convert numpy complex data types into plain python data types, e.g., numpy.int64 into int.
        #
        #  Definitely! It is necessary to sample the population data to get samples where the size of each sub-category
        #  is the same length. For example, if there are 5 marksmanship cateogories then we need 30n x 5categories = 150
        #  elements in the pyvttbl dataframe. If we want to use that same dataframe for analysis of education level,
        #  then the sample needs to be 30n x 7levels = 210, but if we want to avoid indirect correlation then the sample
        #  size needs to be 30n x 5cats x 7levels = 1050. ...and 2100 if we want to also look at first person shooter
        #  experience. While it may be ~theoretically~ possible to attempt max(60, 150, 210), since we don't know that
        #  the 210n will also be evenly divided 105 wFPS/105 without, for example.

        categories = set()
        for i in self.df['all']:
            if i not in categories:
                categories.add(i)

        qpj = df.qualification_performance.describe().to_json()
        edj = df.education_level.describe().to_json()
        ooj = df.describe().to_json()

        qgb = df.groupby('qualification_performance')
        fgb = df.groupby('fps')
        egb = df.groupby('education_level')

        qpg = qgb.describe().to_json()
        fpg = fgb.describe().to_json()
        edg = egb.describe().to_json()
        oog = "Not Applicable"

        qpd = Summary("Marksmanship", qpj, qpg)
        fpd = Summary("FPS Experience", fpj, fpg)
        edd = Summary("Education", edj, edg)
        ood = Summary("Overall", ooj, oog)

        for d in descriptions:
            with open("data/results/" + d[0] + ".json", "w") as data_out:
                json.dump(d, data_out)




        # Pivot Table methods for mixed 2-way ANOVA
        from pyvttbl import DataFrame
        pyv_df = DataFrame()
        pyv_df['scores'] = self.df['pos_diffs']
        pyv_df['qp'] = self.df['qualification_performance']
        pyv_df['edu'] = self.df['education_level']
        pyv_df['fps'] = self.df['fps']
        pyv_df['all'] = self.df['all']

        # Pairwise ANOVA of score difference from pre-lesson to post-lesson knowledge surveys
        anova = pairwise_tukeyhsd(endog=self.df['diffs'], groups=self.df['qualification_performance'], alpha=0.01)
        print anova.summary()

        # Summary descriptive statistics of pre- post- and differences of the whole group and by subgroups
        df_summary = self.df.describe()
        qp_groups = self.df.groupby(['qualification_performance'])
        qp_summary = qp_groups.describe()

        formula = 'self.df["diffs"] ~ self.df["qualification_performance"] + self.df["education_level"] + self.df["is_gamer"]'
        ols_lm = ols(formula, self.df).fit()

        f = open("../data/anovaResults.txt", "w")
        f.write(anova.__str__())


pre_scores  = "../data/pre_test_responses"
post_scores = "../data/post_test_responses"
learners    = "../data/class_data/class.json"

VarianceAnalyzer(learners, pre_scores, post_scores)
