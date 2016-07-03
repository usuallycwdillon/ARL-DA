#!/usr/bin/env python


import pandas as pd
import json
from pyvttbl import DataFrame
from pyvttbl.plotting import *
from collections import namedtuple

import anova
import chi_sq
import settings
import data_io



# pre_scores  = settings.DATA_DIR + "pre_test_responses"
# post_scores = settings.DATA_DIR + "post_test_responses"
# learners    = settings.DATA_DIR + "learners/class.json"
#
# the_format = 'csv'
# # the_format = 'json'
#
# # anova = anova.VarianceAnalyzer(learners, pre_scores, post_scores, the_format)
#
#
# l = len(learners)
# idx = xrange(l)
# cols = ['learner_id', 'education_level', 'attitude_min', 'attitude_total', 'reaction_min', 'reaction_total',
#         'satisfaction_min', 'satisfaction_total']
# df = pd.DataFrame(0, index=idx, columns=cols)
# survey_results = [utility.getAttrs(l) for l in learners]
# df = pd.DataFrame(survey_results)
#
#
# labels = ["low", "moderate", "high"]
# df['attitude_bin'] = utility.binning(df['attitude_total'], df['attitude_min'][0], labels=labels)
# df['reaction_bin'] = utility.binning(df['reaction_total'], df['reaction_min'][0], labels=labels)
# df['satisfaction_bin'] = utility.binning(df['satisfaction_total'], df['satisfaction_min'][0], labels=labels)
#
#
# ed_ct_at = pd.crosstab(df.education_level, df.attitude_bin).apply(lambda row: row / row.sum(), axis=1)
# ed_ct_re = pd.crosstab(df.education_level, df.reaction_bin).apply(lambda row: row / row.sum(), axis=1)
# ed_ct_sa = pd.crosstab(df.education_level, df.satisfaction_bin).apply(lambda row: row / row.sum(), axis=1)
#
#
# at_chi = utility.getChis(ed_ct_at, 'education_level')
# re_chi = utility.getChis(ed_ct_re, 'education_level')
# sa_chi = utility.getChis(ed_ct_sa, 'education_level')
# chis = [at_chi, re_chi, sa_chi]
#
# import pprint
# for ds in chis:
#     pp = pprint.PrettyPrinter(indent=4)
#     pp.pprint(ds)
#
#
# explanans = ['Task Attitude', 'Course Reaction', 'Course Satisfaction']
#
#
# plottables = []
# for i in range(0,3):
#     po = utility.repackage(explanans[i], chis[i])
#     plottables.append(po)
#
#
# data_io.saveJSON(plottables, "results")
# data_io.saveDF(df, "learner_attitude-reaction-satisfaction_data")


#
# class VarianceAnalyzer(object):
#     """
#     This script reads in a list of learner profiles and a list of their scores (each is a json object) then performs
#     an analysis of variance (ANOVA) to suggest which groups of learners are improving their marksmanship knowledge by
#     completing the computer based training.
#     """
#     def __init__(self, learners, pre_scores, post_scores, data_format):
#         """
#         from learners we need their email identifier and their marksmanship rating
#         from the two score lists we need the learner's total pre-lesson score and post-lesson score
#         """
#         super(VarianceAnalyzer, self).__init__()
#
#         if data_format == 'csv':
#             pre_csv = os.path.join(pre_scores, "learner_responses_pre-test.csv")
#             post_csv = os.path.join(post_scores, "learner_responses_post-test.csv")
#             self.pre_data = pd.read_csv(pre_csv)
#             self.post_data = pd.read_csv(post_csv)
#         elif data_format == 'json':
#             # Read in the pre-lesson test results
#             pre_json = []
#             file_list = os.listdir(pre_scores)
#             for f in file_list:
#                 if f.endswith(".json"):
#                     of = open(os.path.join(pre_scores, f))
#                     data = json.load(of)
#                     pre_json.append(data)
#             # Load the results into a dataframe
#             self.pre_data = pd.DataFrame(pre_json)
#             # Read in the post-less on test results
#             post_json = []
#             file_list = os.listdir(post_scores)
#             for f in file_list:
#                 if f.endswith(".json"):
#                     of = open(os.path.join(post_scores, f))
#                     data = json.load(of)
#                     post_json.append(data)
#             # Load the results into a dataframe
#             self.post_data = pd.DataFrame(post_json)
#
#         # Read all of the learner profiles from the class
#         def readClassProfiles(json_file):
#             data = pd.read_json(json_file, orient='records')
#             return data
#
#         # Read in individual data from individual learner profile
#         def readIndividualLearner(dict_list):
#             data = pd.DataFrame.from_dict(dict_list[0], orient='index')
#             for each_dict in dict_list[1:]:
#                 this_data = pd.DataFrame.from_dict(each_dict, orient='index')
#                 data.append(this_data)
#             return data
#
#         def getDiff(row):
#             return row['OverallScore_post'] - row['OverallScore_pre']
#
#         self.profiles = readClassProfiles(learners)
#         self.profiles = pd.DataFrame.transpose(self.profiles)
#
#         self.scores = pd.merge(self.pre_data, self.post_data, on='LearnerID', suffixes=['_pre', '_post'])
#
#         self.df = pd.merge(self.scores, self.profiles, left_on='LearnerID', right_on='email_id', how='outer')
#         self.df['pos_diffs'] = self.df.apply(lambda row: max(getDiff(row), 0), axis=1)
#         self.df['diffs'] = self.df.apply(lambda row: getDiff(row), axis=1)
#         self.df['adj_diffs'] = self.df.apply(lambda row: row['diffs'] + 100, axis=1)
#
#         fps = []
#         def gamer(s):
#             if s == 0: fps.append('no')
#             else: fps.append('yes')
#
#         self.df.apply(lambda row: gamer(row['is_gamer']), axis=1)
#         self.df['fps'] = fps
#
#         ## Save off the combined dataframe as a csv
#         combined_data = "../data/class_data/learners_wTests.csv"
#         self.df.to_csv(combined_data, orient="records")
#
#         ## This side is all data munging to build the database
#         ###############################################################################
#         ## This side is all about adjusting the lengths of columns
#
#         def accumulate(row):
#             return "_".join([row['qualification_performance'], row['education_level'], row['fps']])
#
#         self.df['subs'] = self.df.apply(lambda row: accumulate(row), axis=1)
#
#         combo_sizes = self.df['subs'].value_counts()
#         combo_sample = min(combo_sizes)
#         print(combo_sample)
#
#         qp_sizes = self.df['qualification_performance'].value_counts()
#         qp_sample = min(qp_sizes)
#         print(qp_sample)
#
#         edu_sizes = self.df['education_level'].value_counts()
#         edu_sample = min(edu_sizes)
#         print(edu_sample)
#
#         fps_sizes = self.df['fps'].value_counts()
#         fps_sample = min(fps_sizes)
#         print(fps_sample)
#
#         short_qp_dfs = []
#         for level in list(set(self.df['qualification_performance'])):
#             simple = self.df[self.df['qualification_performance'] == level]
#             short = simple.sample(qp_sample)
#             short_qp_dfs.append(short)
#         sample_qps = pd.concat(short_qp_dfs)
#
#         short_ed_dfs = []
#         for level in list(set(self.df['education_level'])):
#             simple = self.df[self.df['education_level'] == level]
#             short = simple.sample(edu_sample)
#             short_ed_dfs.append(short)
#         sample_eds = pd.concat(short_ed_dfs)
#
#         short_fp_dfs = []
#         for level in list(set(self.df['fps'])):
#             simple = self.df[self.df['fps'] == level]
#             short = simple.sample(fps_sample)
#             short_fp_dfs.append(short)
#         sample_fps = pd.concat(short_fp_dfs)
#
#
#         # Pivot Table methods for ANOVA
#         def str_list(some_array):
#             the_list = some_array.tolist()
#             the_elements = [str(x) for x in the_list]
#             return the_elements
#
#         def int_list(some_array):
#             the_list = some_array.tolist()
#             the_elements = [int(x) for x in the_list]
#             return the_elements
#
#         def get_w(anova):
#             top = anova['ssbn'] - anova['dfbn'] * anova['mswn']
#             bottom = anova['ssbn'] + anova['sswn'] + anova['mswn']
#             return top / bottom
#
#         def do_anovas(some_df, variable):
#             '''
#             This method takes a dataframe,  returns the pyvttbl
#             anova object for that element
#             '''
#             pyv_df = DataFrame()
#
#             if variable == 'qp':
#                 pyv_df['qual'] = str_list(some_df['qualification_performance'])
#                 pyv_df['vals'] = int_list(some_df['adj_diffs'])
#
#             elif variable == 'ed':
#                 pyv_df['qual'] = str_list(some_df['education_level'])
#                 pyv_df['vals'] = int_list(some_df['adj_diffs'])
#
#             elif variable == 'fp':
#                 pyv_df['qual'] = str_list(some_df['fps'])
#                 pyv_df['vals'] = int_list(some_df['adj_diffs'])
#
#             else:
#                 return None
#
#             anova = pyv_df.anova1way('vals', 'qual')
#             anova['omega-sq'] = get_w(anova)
#
#             return anova
#
#         qpw_anova = do_anovas(sample_qps, 'qp')
#         edw_anova = do_anovas(sample_eds, 'ed')
#         fpw_anova = do_anovas(sample_fps, 'fp')
#
#         # Rank order variables
#         Rank = namedtuple('Rank', 'var omega')
#
#         qpt = Rank(var='Marksmanship', omega=qpw_anova['omega-sq'])
#         fpt = Rank(var='FPS_experience', omega=fpw_anova['omega-sq'])
#         edt = Rank(var='Education', omega=edw_anova['omega-sq'])
#
#         ranks = [qpt, fpt, edt]
#         ranks = sorted(ranks, key=lambda x: x.omega)
#
#
#         ## Huom! It may or may not be necessary to elementally convert the numpy.ndarrays into plain python lists and
#         #  elementally convert numpy complex data types into plain python data types, e.g., numpy.int64 into int.
#         #
#         #  Definitely! It is necessary to sample the population data to get samples where the size of each sub-category
#         #  is the same length. For example, if there are 5 marksmanship cateogories then we need 30n x 5categories = 150
#         #  elements in the pyvttbl dataframe. If we want to use that same dataframe for analysis of education level,
#         #  then the sample needs to be 30n x 7levels = 210, but if we want to avoid indirect correlation then the sample
#         #  size needs to be 30n x 5cats x 7levels = 1050. ...and 2100 if we want to also look at first person shooter
#         #  experience. While it may be ~theoretically~ possible to attempt max(60, 150, 210), since we don't know that
#         #  the 210n will also be evenly divided 105 wFPS/105 without, for example.
#
#         def tuple_handler(data_in):
#             data_out = {}
#             for k, v in data_in.iteritems():
#                 inner = {}
#                 for kk, vv in v.iteritems():
#                     nk = "_".join(kk)
#                     inner[nk] = vv
#                 data_out[k] = inner
#             return data_out
#
#         def tuple_handler2(data_in):
#             data_out = {}
#             for k, v in data_in.iteritems():
#                 nk = "_".join(k)
#                 data_out[nk] = v
#             return data_out
#
#
#         qpj = self.df.qualification_performance.describe().to_dict()
#         fpj = self.df.fps.describe().to_dict()
#         edj = self.df.education_level.describe().to_dict()
#         ooj = self.df.describe().to_dict()
#
#         qgb = (self.df.groupby('qualification_performance')).describe()
#         fgb = (self.df.groupby('fps')).describe()
#         egb = (self.df.groupby('education_level')).describe()
#         oob = "Not Applicable"
#
#         qpg = tuple_handler(qgb)
#         fpg = tuple_handler(fgb)
#         edg = tuple_handler(egb)
#
#         qptt = tuple_handler2(qpw_anova.multtest)
#         fptt = tuple_handler2(fpw_anova.multtest)
#         edtt = tuple_handler2(edw_anova.multtest)
#
#
#         qpd = {"Category": "Marksmanship", "Summary": qpj, "Data": qpg, "ANOVA": qpw_anova, "Ttest": qptt}
#         fpd = {"Category": "FPS Experience", "Summary": fpj, "Data": fpg, "ANOVA": fpw_anova, "Ttest": fptt}
#         edd = {"Category": "Education", "Summary": edj, "Data": edg, "ANOVA": edw_anova, "Ttest": edtt}
#         ood = {"Category": "Overall", "Summary": ooj, "Data": oob, "ANOVA": 'Not Applicable', "Ttest":'Not Applicable'}
#
#         descriptions = [qpd, fpd, edd, ood]
#
#         ## Compact Data
#         def compactor(big_dict):
#             cat = big_dict['Category']
#             datas = []
#
#             for k, v in big_dict['Data']['adj_diffs'].iteritems():
#                 means = {}
#                 if k.endswith('_mean'):
#                     means['attr'] = k
#                     means['value'] = v
#                     datas.append(means)
#                     #means[k]=v
#             f = big_dict['ANOVA']['f']
#             p = big_dict['ANOVA']['p']
#             w = big_dict['ANOVA']['omega-sq']
#             t = {}
#             for k, v in big_dict['Ttest'].iteritems():
#                 nv = {}
#                 nv['q'] = v['q']
#                 nv['sig'] = v['sig']
#                 t[k] = nv
#             return [{'KEY': cat, 'Data': datas, 'ANOVA F-stat': f, 'ANOVA p-value': p, 'ANOVA w2': w, 'Ttest': t}]
#             intro = {'KEY': cat, 'Data':datas}
#             return [intro]
#
#
#         for d in descriptions[0:-1]:
#             cd = compactor(d)
#             with open(DATA_DIR + "results/" + d['Category'] + "_compact-v2.json", "w") as data_out:
#                 json.dump(cd, data_out)
#             with open(DATA_DIR + "results/" + d['Category'] + ".json", "w") as data_out:
#                 json.dump(d, data_out)
#
#         f = open(DATA_DIR + "results/" + "overall.json", "w")
#         json.dump(descriptions[0], f)
#         f.close()
#
#         f = open(DATA_DIR + "results/" + "var-rankings.json", "w")
#         json.dump(ranks, f, sort_keys=True)
#         f.close()
#
#
# #!/usr/bin/env python
#
# import pandas as pd
# import scipy.stats as sps
# from scipy.stats import zmap
# import json
#
#
# class chiAnalysis(object):
#     '''
#     There is nothing in this block, yet.
#     '''
#
#     def __init__(self, learners, rec_fire):
#
#         '''
#         :param learners is a json file containing all learner profiles used in this analysis
#         :param rec_fire is a json file containing the reaper/live-fire record of marksmanship qualification trial
#         '''
#
#         self.learner_data = learners
#         self.rec_fire_data = rec_fire
#
#         # Read all of the learner profiles from the class
#         def readClassProfiles(json_file):
#             data = pd.read_json(json_file, orient='index')
#             return data
#
#         # Read in the Record Fire data
#         def readRecordFire(json_file):
#             data = pd.read_json(json_file, orient='records')
#             return data
#
#         self.learners = readClassProfiles(self.learner_data)
#         if the_format == 'json':
#             self.records = readRecordFire(self.rec_fire_data)
#         elif the_format == 'csv':
#             self.records = pd.read_csv(rec_fire)
#
#
#         # Get frequencies, for the whole df or a filtered subset
#         def getRelFreq(a_df, a_var):
#             a_div = len(a_df)
#             var_count = a_df.groupby(a_var).count()
#             a_dict = var_count.to_dict()
#             whole_dict = a_dict['student_id']
#             the_dict = {}
#             for k, v in whole_dict.iteritems():
#                 the_dict[k] = float(v) / a_div
#             return the_dict
#
#         def getRelFreqPlus(a_df, a_var):
#             div = len(a_df)
#             vc = a_df.groupby([a_var, 'isHit']).count()
#             a_dict = vc.to_dict()
#             w_dict = a_dict['student_id']
#             the_dict = {}
#             for k, v in w_dict.iteritems():
#                 the_dict[k] = float(v) / div
#             return the_dict
#
#
#         def posConverter(slice):
#             if slice['fire_pos'] == 1:
#                 return 'Prone_Supported'
#             elif slice['fire_pos'] == 2:
#                 return 'Prone_Unsupported'
#             elif slice['fire_pos'] == 3:
#                 return 'Kneeling_Unsupported'
#             else:
#                 return 'huh?'
#
#
#         def oneValuePlotter(a_df, title):
#             the_df = {'KEY': title}
#             data = []
#             for k, v in a_df.iteritems():
#                 element = {}
#                 element['attr'] = k
#                 element['value'] = v
#                 data.append(element)
#             the_df['Data'] = data
#             return the_df
#
#         def qualifiedHitsPlotter(a_df, title):
#             the_df = {'KEY': title}
#             data = []
#             for k, v in a_df.iteritems():
#                 element = {}
#                 element['attr'] = k
#                 for vk, vv in v.iteritems():
#                     element[vk] = vv
#                 data.append(element)
#             the_df['Data'] = data
#             return the_df
#
#         def recordfireHitsPlotter(a_df, title):
#             the_df = {'KEY': title}
#             data = []
#             for k, v in a_df.iteritems():
#                 element = {}
#                 element['attr'] = k[0]
#                 if k[1] == 0:
#                     element['Miss'] = v
#                 elif k[1] == 1:
#                     element['Hit'] = v
#                 elif k[1] == -1:
#                     element['No Fire'] = v
#                 else:
#                     element[k[1]] = v
#                 data.append(element)
#             the_df['Data'] = data
#             return the_df
#
#         def recordfireRangePlotter(a_df, title):
#             the_df = {'KEY': title}
#             data = []
#             for k, v in a_df.iteritems():
#                 element = {}
#                 seq = (k[0], 'm')
#                 element['attr'] = "_".join((str(k[0]), 'm'))
#                 if k[1] == 0:
#                     element['Miss'] = v
#                 elif k[1] == 1:
#                     element['Hit'] = v
#                 elif k[1] == -1:
#                     element['No Fire'] = v
#                 else:
#                     element[k[1]] = v
#                 data.append(element)
#             the_df['Data'] = data
#             return the_df
#
#         positions = []
#         def position(n):
#             if float(n) == 1.0:
#                 positions.append('Prone_Supported')
#             elif float(n) == 2.0:
#                 positions.append('Prone_Unsupported')
#             elif float(n) == 3.0:
#                 positions.append('Kneeling_Unsupported')
#             else:
#                 positions.append('Dunno')
#
#
#         self.df = pd.merge(self.records, self.learners, left_on='student_id', right_on='email_id', how='outer')
#
#
#
#         # Task 1. Relative frequency of rating outcomes (unqualified, marksman, sharpshooter, expert)
#         oqrf = getRelFreq(self.records, 'ability')
#         overall_qr_rf = oneValuePlotter(oqrf, 'Qualification Ratings Relative Frequency')
#
#         results = {}
#         results['ovarall_qr_rf'] = overall_qr_rf
#
#         # Task 2. Relative frequency of attempts to become qualified
#         # Only do task 2 if time permits
#         # TODO: Task 2. Relative frequency of attempts to become qualified
#         # Cannot do; no data yet
#
#
#         # Task 3. Relative frequency by each variable: hits, misses, no-fires for qualified and unqualified
#
#         ql = self.records[(self.records.ability != 'Unqualified')]
#         ul = self.records[(self.records.ability == 'Unqualified')]
#
#         qrf = recordfireHitsPlotter(ql, 'isHit')
#         urf = recordfireHitsPlotter(ul, 'isHit')
#
#         # This is a much better/easier way.
#         # a_h_ct = pd.crosstab(df.ability, df.isHit).apply(lambda row: row / row.sum(), axis=1)
#
#         rf_hits ={'Qualified': qrf, 'Unqualified': urf}
#         rf_pos = getRelFreqPlus(self.records, 'fire_pos')
#
#         rf_qr = getRelFreqPlus(self.records, 'ability')
#         rf_r = getRelFreqPlus(self.records, 'range')
#
#         qual_rec_proportion = qualifiedHitsPlotter(rf_hits, 'Record Fire Proportion by Qualified')
#         results['qual_rec_proportion'] = qual_rec_proportion
#
#
#         # Task 4. Relative frequency of task 3 subset by qualification rating categories
#         ability_rec_proportion = recordfireHitsPlotter(rf_qr, 'Record Fire Proportion by Qualification Rating')
#         results['ability_rec_proportion'] = ability_rec_proportion
#
#         # Task 5. Relative frequency by firing position
#         pos_rec_proportion = recordfireHitsPlotter(rf_pos, 'Record Fire Proportion by Firing Position')
#         results['pos_rec_proportion'] = pos_rec_proportion
#
#         # Task 6. Relative frequency by range
#         range_rec_proportion = recordfireRangePlotter(rf_r, 'Record Fire Proportion by Range')
#         results['range_rec_proportion'] = range_rec_proportion
#
#         # Task 7. Scatterplot of shots on target (this is just the json file)
#         ot = self.records.ix[:, ['ability', 'student_id', 'shot_x', 'shot_y'] ]
#
#         shot_locations = {'KEY':'Shot Location'}
#         data = []
#         for i in range(len(ot)):
#             d = {}
#             d['attr'] = ot.index[i]
#             d['ability'] = ot.ability[i]
#             d['x'] = ot.shot_x[i]
#             d['y'] = ot.shot_y[i]
#             d['student_id'] = "learner" + str(ot.student_id[i]) + "@example.com"
#             data.append(d)
#         shot_locations['Data'] = data
#
#         results['shots_location'] = shot_locations
#
#         # Task 8. Chi-sq testing on each dem variable (associated = True if Chi-sq stat >= 1.96 (two sigmas/95%)
#         # Task 9. For each "associated = True" variable: Frequencies of students in each category of that variable; indicate
#         #         which categories have more students than expected
#         # I'm doing these together. For Task 8, there is a list of dicts. The task 9 data is included if the relationship
#         # is significant.
#
#
#         ## re-index for range and position crosstabs
#
#         def getChis(crosstab):
#             chi2, p, dof, ex = sps.chi2_contingency(crosstab[0])
#             crit = sps.chi2.ppf(q=0.95, df=dof)
#             if (crit < chi2):
#                 evaluation = True
#             else:
#                 evaluation = False
#
#             obs = crosstab[0].as_matrix()
#             obs_list = obs.tolist()
#             ex_list = ex.tolist()
#             z_scores = zmap(obs_list, ex_list)
#             z_list = z_scores.tolist()
#             z_indicators = []
#             for z in z_list:
#                 z_sig = ["+" if i > 1.96 else "-" if i < -1.96 else " " for i in z]
#                 z_indicators.append(z_sig)
#
#             results = {'chi-sq': chi2,
#                        'p-val': p,
#                        'eval': evaluation,
#                        'dof': dof,
#                        'explanandum':crosstab[1],
#                        'expected': ex_list,
#                        'observed': obs_list,
#                        'z_scores': z_indicators,
#                        'row_lab': crosstab[0].index.tolist(),
#                        'col_lab': crosstab[0].columns.tolist()
#                        }
#             return results
#
#         # def getChis2(crosstab):
#         #     chi2, p, dof, ex = sps.chi2_contingency(crosstab[0])
#         #     crit = sps.chi2.ppf(q=0.95, df=dof)
#         #     if (crit < chi2):
#         #         evaluation = True
#         #     else:
#         #         evaluation = False
#         #     results = {'chi-sq': chi2,
#         #                'crit': sps.chi,
#         #                'p-val': p,
#         #                'eval': evaluation,
#         #                'dof': dof,
#         #                'explanandum': crosstab[1],
#         #                'expected': ex.tolist(),
#         #                'observed': crosstab[0].to_list(),
#         #                'col_lab': crosstab[0].columns}
#         #     for x in range(len(crosstab[0].index)):
#         #         the_index = '_'.join(crosstab[0].index[x])
#         #         results['row_lab'] = the_index
#         #     return results
#
#         #TODO: Figure out why labels/variables cannot be passed to the crosstab function. Doing this manually until then.
#         a_df_ed = pd.crosstab(self.df.ability, self.df.education_level)
#         a_df_hp = pd.crosstab(self.df.ability, self.df.handgun_prof)
#         a_df_ig = pd.crosstab(self.df.ability, self.df.is_gamer)
#         a_df_qp = pd.crosstab(self.df.ability, self.df.qualification_performance)
#         a_df_r = pd.crosstab(self.df.ability, self.df['rank'])
#         a_dfs = [a_df_ed, a_df_hp, a_df_ig, a_df_qp, a_df_r]
#
#         dems = ['Education Level', 'Handgun Profiency', 'Computer Gamer', 'Rifle Experience',
#                 'Officer or Enlisted']
#
#         facs = ['Qualification Rating', 'Fire Position', 'Range']
#
#
#         a_chi =  [getChis(i) for i in zip(a_dfs, dems)]
#
#         # Final outputs:
#         # ...for Record Fire (generally) in 6th slide of req't presentation
#         soft_surveys = []
#
#         qual_rating_srv = {'explanans': facs[0]}
#
#         def repackage(the_dict, the_list):
#             var_list = []
#             for d in the_list:
#                 model = {}
#                 model['explanandum'] = d['explanandum']
#                 model['active'] = d['eval']
#                 if (d['eval'] == True):
#                     the_data = {}
#                     the_data['rows'] = d['row_lab']
#                     the_data['cols'] = d['col_lab']
#                     the_data['obs_frequencies'] = d['observed']
#                     the_data['exp_frequencies'] = d['expected']
#                     the_data['significant'] = d['z_scores']
#                     model['data'] = the_data
#                 var_list.append(model)
#             the_dict['Data'] = var_list
#             return the_dict
#
#         stat_rel_variables.append(repackage(qual_rating_srv, a_chi))
#
#         print stat_rel_variables[0]
#
#         results['significant_relationships'] = stat_rel_variables
#
#         for k, v in results.iteritems():
#             with open("../data/results/" + k + ".json", "w") as data_out:
#                 json.dump(v, data_out)
#
# ## identify data files
# learners = "../data/class_data/class.json"
# rec_fire = "../data/reaper/record-fire-vCWD.csv"
#
# the_format = 'csv'
# # the_format = 'json'
#
# ## create the Chi-sq analysis object
# results = chiAnalysis(learners, rec_fire)