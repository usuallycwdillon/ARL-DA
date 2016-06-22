#!/usr/bin/python


import pandas as pd
import json
from pyvttbl import DataFrame
from pyvttbl.plotting import *
from collections import namedtuple


class VarianceAnalyzer(object):
    """
    This script reads in a list of learner profiles and a list of their scores (each is a json object) then performs
    an analysis of variance (ANOVA) to suggest which groups of learners are improving their marksmanship knowledge by
    completing the computer based training.
    """
    def __init__(self, learners, pre_scores, post_scores, data_format):
        """
        from learners we need their email identifier and their marksmanship rating
        from the two score lists we need the learner's total pre-lesson score and post-lesson score
        """
        super(VarianceAnalyzer, self).__init__()

        if data_format == 'csv':
            pre_csv = os.path.join(pre_scores, "learner_responses_pre-test.csv")
            post_csv = os.path.join(post_scores, "learner_responses_post-test.csv")
            self.pre_data = pd.read_csv(pre_csv)
            self.post_data = pd.read_csv(post_csv)
        elif data_format == 'json':
            # Read in the pre-lesson test results
            pre_json = []
            file_list = os.listdir(pre_scores)
            for f in file_list:
                if f.endswith(".json"):
                    of = open(os.path.join(pre_scores, f))
                    data = json.load(of)
                    pre_json.append(data)
            # Load the results into a dataframe
            self.pre_data = pd.DataFrame(pre_json)
            # Read in the post-less on test results
            post_json = []
            file_list = os.listdir(post_scores)
            for f in file_list:
                if f.endswith(".json"):
                    of = open(os.path.join(post_scores, f))
                    data = json.load(of)
                    post_json.append(data)
            # Load the results into a dataframe
            self.post_data = pd.DataFrame(post_json)

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

        self.scores = pd.merge(self.pre_data, self.post_data, on='LearnerID', suffixes=['_pre', '_post'])

        self.df = pd.merge(self.scores, self.profiles, left_on='LearnerID', right_on='email_id', how='outer')
        self.df['pos_diffs'] = self.df.apply(lambda row: max(getDiff(row), 0), axis=1)
        self.df['diffs'] = self.df.apply(lambda row: getDiff(row), axis=1)
        self.df['adj_diffs'] = self.df.apply(lambda row: row['diffs'] + 100, axis=1)

        fps = []
        def gamer(s):
            if s == 0: fps.append('no')
            else: fps.append('yes')

        self.df.apply(lambda row: gamer(row['is_gamer']), axis=1)
        self.df['fps'] = fps

        ## Save off the combined dataframe as a csv
        combined_data = "../data/class_data/learners_wTests.csv"
        self.df.to_csv(combined_data, orient="records")

        ## This side is all data munging to build the database
        ###############################################################################
        ## This side is all about adjusting the lengths of columns

        def accumulate(row):
            return "_".join([row['qualification_performance'], row['education_level'], row['fps']])

        self.df['subs'] = self.df.apply(lambda row: accumulate(row), axis=1)

        combo_sizes = self.df['subs'].value_counts()
        combo_sample = min(combo_sizes)
        print(combo_sample)

        qp_sizes = self.df['qualification_performance'].value_counts()
        qp_sample = min(qp_sizes)
        print(qp_sample)

        edu_sizes = self.df['education_level'].value_counts()
        edu_sample = min(edu_sizes)
        print(edu_sample)

        fps_sizes = self.df['fps'].value_counts()
        fps_sample = min(fps_sizes)
        print(fps_sample)

        short_qp_dfs = []
        for level in list(set(self.df['qualification_performance'])):
            simple = self.df[self.df['qualification_performance'] == level]
            short = simple.sample(qp_sample)
            short_qp_dfs.append(short)
        sample_qps = pd.concat(short_qp_dfs)

        short_ed_dfs = []
        for level in list(set(self.df['education_level'])):
            simple = self.df[self.df['education_level'] == level]
            short = simple.sample(edu_sample)
            short_ed_dfs.append(short)
        sample_eds = pd.concat(short_ed_dfs)

        short_fp_dfs = []
        for level in list(set(self.df['fps'])):
            simple = self.df[self.df['fps'] == level]
            short = simple.sample(fps_sample)
            short_fp_dfs.append(short)
        sample_fps = pd.concat(short_fp_dfs)


        # Pivot Table methods for ANOVA
        def str_list(some_array):
            the_list = some_array.tolist()
            the_elements = [str(x) for x in the_list]
            return the_elements

        def int_list(some_array):
            the_list = some_array.tolist()
            the_elements = [int(x) for x in the_list]
            return the_elements

        def get_w(anova):
            top = anova['ssbn'] - anova['dfbn'] * anova['mswn']
            bottom = anova['ssbn'] + anova['sswn'] + anova['mswn']
            return top / bottom

        def do_anovas(some_df, variable):
            '''
            This method takes a dataframe,  returns the pyvttbl
            anova object for that element
            '''
            pyv_df = DataFrame()

            if variable == 'qp':
                pyv_df['qual'] = str_list(some_df['qualification_performance'])
                pyv_df['vals'] = int_list(some_df['adj_diffs'])

            elif variable == 'ed':
                pyv_df['qual'] = str_list(some_df['education_level'])
                pyv_df['vals'] = int_list(some_df['adj_diffs'])

            elif variable == 'fp':
                pyv_df['qual'] = str_list(some_df['fps'])
                pyv_df['vals'] = int_list(some_df['adj_diffs'])

            else:
                return None

            anova = pyv_df.anova1way('vals', 'qual')
            anova['omega-sq'] = get_w(anova)

            return anova

        qpw_anova = do_anovas(sample_qps, 'qp')
        edw_anova = do_anovas(sample_eds, 'ed')
        fpw_anova = do_anovas(sample_fps, 'fp')

        # Rank order variables
        Rank = namedtuple('Rank', 'var omega')

        qpt = Rank(var='Marksmanship', omega=qpw_anova['omega-sq'])
        fpt = Rank(var='FPS_experience', omega=fpw_anova['omega-sq'])
        edt = Rank(var='Education', omega=edw_anova['omega-sq'])



        ranks = [qpt, fpt, edt]
        ranks = sorted(ranks, key=lambda x: x.omega)


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

        def tuple_handler(data_in):
            data_out = {}
            for k, v in data_in.iteritems():
                inner = {}
                for kk, vv in v.iteritems():
                    nk = "_".join(kk)
                    inner[nk] = vv
                data_out[k] = inner
            return data_out

        def tuple_handler2(data_in):
            data_out = {}
            for k, v in data_in.iteritems():
                nk = "_".join(k)
                data_out[nk] = v
            return data_out


        qpj = self.df.qualification_performance.describe().to_dict()
        fpj = self.df.fps.describe().to_dict()
        edj = self.df.education_level.describe().to_dict()
        ooj = self.df.describe().to_dict()

        qgb = (self.df.groupby('qualification_performance')).describe()
        fgb = (self.df.groupby('fps')).describe()
        egb = (self.df.groupby('education_level')).describe()
        oob = "Not Applicable"

        qpg = tuple_handler(qgb)
        fpg = tuple_handler(fgb)
        edg = tuple_handler(egb)

        qptt = tuple_handler2(qpw_anova.multtest)
        fptt = tuple_handler2(fpw_anova.multtest)
        edtt = tuple_handler2(edw_anova.multtest)


        qpd = {"Category": "Marksmanship", "Summary": qpj, "Data": qpg, "ANOVA": qpw_anova, "Ttest": qptt}
        fpd = {"Category": "FPS Experience", "Summary": fpj, "Data": fpg, "ANOVA": fpw_anova, "Ttest": fptt}
        edd = {"Category": "Education", "Summary": edj, "Data": edg, "ANOVA": edw_anova, "Ttest": edtt}
        ood = {"Category": "Overall", "Summary": ooj, "Data": oob, "ANOVA": 'Not Applicable', "Ttest":'Not Applicable'}

        descriptions = [qpd, fpd, edd, ood]

        ## Compact Data
        def compactor(big_dict):
            cat = big_dict['Category']
            datas = []

            for k, v in big_dict['Data']['adj_diffs'].iteritems():
                means = {}
                if k.endswith('_mean'):
                    means['attr'] = k
                    means['value'] = v
                    datas.append(means)
                    #means[k]=v
            f = big_dict['ANOVA']['f']
            p = big_dict['ANOVA']['p']
            w = big_dict['ANOVA']['omega-sq']
            t = {}
            for k, v in big_dict['Ttest'].iteritems():
                nv = {}
                nv['q'] = v['q']
                nv['sig'] = v['sig']
                t[k] = nv
            return [{'KEY': cat, 'Data': datas, 'ANOVA F-stat': f, 'ANOVA p-value': p, 'ANOVA w2': w, 'Ttest': t}]
            intro = {'KEY': cat, 'Data':datas}
            return [intro]


        for d in descriptions[0:-1]:
            cd = compactor(d)
            with open("../data/results/" + d['Category'] + "_compact-v2.json", "w") as data_out:
                json.dump(cd, data_out)
            with open("../data/results/" + d['Category'] + ".json", "w") as data_out:
                json.dump(d, data_out)

        f = open("../data/results/overall.json", "w")
        json.dump(descriptions[0], f)
        f.close()

        f = open('../data/var-rankings.json', "w")
        json.dump(ranks, f, sort_keys=True)
        f.close()

pre_scores  = "../data/pre_test_responses"
post_scores = "../data/post_test_responses"
learners    = "../data/class_data/class.json"

the_format = 'csv'
# the_format = 'json'

anova = VarianceAnalyzer(learners, pre_scores, post_scores, the_format)
