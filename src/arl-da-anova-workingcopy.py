
        anova = pairwise_tukeyhsd(endog=df['diffs'], groups=df['qualification_performance'], alpha=0.05)

        print anova.summary()



        f = open("../data/anovaResults.txt", "w")
        f.write(anova.__str__())

################################################################################

pre_scores  = "data/pre_test_responses"
post_scores = "data/post_test_responses"
learners    = "data/class_data/class.json"

import numpy as np
import os
import pandas as pd
import scipy.stats as st
import scipy as sc
import json
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from copy import deepcopy

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


profiles = readClassProfiles(learners)
profiles = pd.DataFrame.transpose(profiles)
pre_data = pd.DataFrame(pre_json)
post_data = pd.DataFrame(post_json)
scores = pd.merge(pre_data, post_data, on='LearnerID', suffixes=['_pre', '_post'])

df = pd.merge(scores, profiles, left_on='LearnerID', right_on='email_id', how='outer')
df['pos_diffs'] = df.apply(lambda row: max(getDiff(row), 0), axis=1)
df['diffs'] = df.apply(lambda row: getDiff(row), axis=1)
df['adj_diffs'] = df.apply(lambda row: row['diffs'] + 100, axis=1)

fps = []
def gamer(s):
    if s == 0:
        fps.append('no')
    else:
        fps.append('yes')

df.apply(lambda row: gamer(row['is_gamer']), axis=1)
df['fps'] = fps

def logitify(s):
    v = list(set(s))
    for c in v:
        if s == v:
            new_col.append()



## This side is all data munging to build the database
###############################################################################
##                                                                           ##
##  #######################################################################  ##
##
###############################################################################
## This side is all about adjusting the lengths of columns

def accumulate(row):
    return "_".join([row['qualification_performance'], row['education_level'], row['fps']])

df['subs'] = df.apply(lambda row: accumulate(row), axis=1)

combo_sizes = df['subs'].value_counts()
combo_sample = min(combo_sizes)
print(combo_sample)

qp_sizes = df['qualification_performance'].value_counts()
qp_sample = min(qp_sizes)
print(qp_sample)


edu_sizes = df['education_level'].value_counts()
edu_sample = min(edu_sizes)
print(edu_sample)

fps_sizes = df['fps'].value_counts()
fps_sample = min(fps_sizes)
print(fps_sample)

max_sample_sizes = df['subs'].value_counts()
categories = set()
for i in df['subs']:
    if i not in categories:
        categories.add(i)

short_qp_dfs = []
for level in list(set(df['qualification_performance'])):
    simple = df[df['qualification_performance'] == level]
    short = simple.sample(qp_sample)
    short_qp_dfs.append(short)
sample_qps = pd.concat(short_qp_dfs)    

short_ed_dfs = []
for level in list(set(df['education_level'])):
    simple = df[df['education_level'] == level]
    short = simple.sample(edu_sample)
    short_ed_dfs.append(short)
sample_eds = pd.concat(short_ed_dfs)    

short_fp_dfs = []
for level in list(set(df['fps'])):
    simple = df[df['fps'] == level]
    short = simple.sample(8)
    short_fp_dfs.append(short)
sample_fps = pd.concat(short_fp_dfs)


def str_list(some_array):
    the_list = some_array.tolist()
    the_elements = [str(x) for x in the_list]
    return the_elements

def int_list(some_array):
    the_list = some_array.tolist()
    the_elements = [int(x) for x in the_list]
    return the_elements

# Pivot Table methods for ANOVA
# from pyvttbl import DataFrame
def get_o(anova):
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
    anova['omega-sq'] = get_o(anova)

    label = variable

    return label, anova

#qp_anovas = [do_anovas(d, "qp") for d in short_qp_dfs]
#ed_anovas = [do_anovas(d, "ed") for d in short_ed_dfs]
#fp_anovas = [do_anovas(d, "fp") for d in short_fp_dfs]


qp_df['qual'] = str_list(sample_qps['qualification_performance'])
qp_df['vals'] = int_list(sample_qps['adj_diffs'])

ed_df['qual'] = str_list(sample_eds['education_level'])
ed_df['vals'] = int_list(sample_eds['adj_diffs'])

fp_df['qual'] = str_list(sample_fps['fps'])
fp_df['vals'] = int_list(sample_fps['adj_diffs'])

qpw_anova = qp_df.anova1way('vals', 'qual')
edw_anova = ed_df.anova1way('vals', 'qual')
fpw_anova = fp_df.anova1way('vals', 'qual')


qpw_anova['omega-sq'] = get_o(qpw_anova)
edw_anova['omega-sq'] = get_o(edw_anova)
fpw_anova['omega-sq'] = get_o(fpw_anova)



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

qpj = df.qualification_performance.describe().to_dict()
fpj = df.fps.describe().to_dict()
edj = df.education_level.describe().to_dict()
ooj = df.describe().to_dict()

qgb = (df.groupby('qualification_performance')).describe()
fgb = (df.groupby('fps')).describe()
egb = (df.groupby('education_level')).describe()
oob = "Not Applicable"

qpg = tuple_handler(qgb)
fpg = tuple_handler(fgb)
edg = tuple_handler(egb)



qpd = {"Category":"Marksmanship", "Summary":qpj, "Data":qpg, "ANOVA":qpw_anova}
fpd = {"Category":"FPS Experience", "Summary":fpj, "Data":fpg, "ANOVA":fpw_anova}
edd = {"Category":"Education", "Summary":edj, "Data":edg, "ANOVA":edw_anova}
ood = {"Category":"Overall", "Summary":ooj, "Data":oob}

descriptions = [qpd, fpd, edd, ood]


#from collections import namedtuple
#Summary = namedtuple("Summary", ["var", "sum", "data"])
#qpd = Summary("Marksmanship", qpj, qpg)
#fpd = Summary("FPS Experience", fpj, fpg)
#edd = Summary("Education", edj, edg)
#ood = Summary("Overall", ooj, oog)
#descriptions = [qpd, fpd, edd, ood]




qpcs = {"Category":"Marksmanship", "Summary":qpj, "Data":qpg}
fpcs = {"Category":"FPS Experience", "Summary":fpj, "Data":fpg}
edcs = {"Category":"Education", "Summary":edj, "Data":edg}
oocs = {"Category":"Marksmanship", "Summary":ooj, "Data":oog}

descriptions = [qpcs, fpcs, edcs, oocs]

for d in descriptions:
    with open("data/results/" + d['Category'] + ".json", "w") as data_out:
        json.dump(d, data_out)



Here's *some data* but it 's not everything for the task. This is really just enough to plot average scores. I'm working on getting the ANOVA data out now. I FINALLY came up with a way to re-assemble the pandas dataframe into something that could be converted to JSON.

There's a lot more data in here than you need(or want?) but it's got reasonable labels and hopefully we can get away from simple bar charts to show averages and graduate to box plots as we get further along. All the data necessary for that is in here.

Here's what I'm giving you:
4 files: 
    1 for the whole group, 
    1 subitized for education categories, 
    1 subitized for FPS experience, 
    1 subitized for marksmanship experience.
Each file has a 'summary' for the whole subset and a 'data' object for the data and a label (to keep track of it and for labeling the plot).
The data includes a complete description of the shape of the data. For now, we just need the "_mean" for each category.
​
I haven't done much to validate these, so please let me know if there's something really odd.
