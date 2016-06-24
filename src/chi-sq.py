#!/usr/bin/python

import pandas as pd
import scipy.stats as sps
from scipy.stats import zmap
import json


class chiAnalysis(object):
    '''
    There is nothing in this block, yet.
    '''

    def __init__(self, learners, rec_fire):

        '''
        :param learners is a json file containing all learner profiles used in this analysis
        :param rec_fire is a json file containing the reaper/live-fire record of marksmanship qualification trial
        '''

        self.learner_data = learners
        self.rec_fire_data = rec_fire

        # Read all of the learner profiles from the class
        def readClassProfiles(json_file):
            data = pd.read_json(json_file, orient='index')
            return data

        # Read in the Record Fire data
        def readRecordFire(json_file):
            data = pd.read_json(json_file, orient='records')
            return data

        self.learners = readClassProfiles(self.learner_data)
        if the_format == 'json':
            self.records = readRecordFire(self.rec_fire_data)
        elif the_format == 'csv':
            self.records = pd.read_csv(rec_fire)


        # Get frequencies, for the whole df or a filtered subset
        def getRelFreq(a_df, a_var):
            a_div = len(a_df)
            var_count = a_df.groupby(a_var).count()
            a_dict = var_count.to_dict()
            whole_dict = a_dict['student_id']
            the_dict = {}
            for k, v in whole_dict.iteritems():
                the_dict[k] = float(v) / a_div
            return the_dict

        def getRelFreqPlus(a_df, a_var):
            div = len(a_df)
            vc = a_df.groupby([a_var, 'isHit']).count()
            a_dict = vc.to_dict()
            w_dict = a_dict['student_id']
            the_dict = {}
            for k, v in w_dict.iteritems():
                the_dict[k] = float(v) / div
            return the_dict


        def posConverter(slice):
            if slice['fire_pos'] == 1:
                return 'Prone_Supported'
            elif slice['fire_pos'] == 2:
                return 'Prone_Unsupported'
            elif slice['fire_pos'] == 3:
                return 'Kneeling_Unsupported'
            else:
                return 'huh?'


        def oneValuePlotter(a_df, title):
            the_df = {'KEY': title}
            data = []
            for k, v in a_df.iteritems():
                element = {}
                element['attr'] = k
                element['value'] = v
                data.append(element)
            the_df['Data'] = data
            return the_df

        def qualifiedHitsPlotter(a_df, title):
            the_df = {'KEY': title}
            data = []
            for k, v in a_df.iteritems():
                element = {}
                element['attr'] = k
                for vk, vv in v.iteritems():
                    element[vk] = vv
                data.append(element)
            the_df['Data'] = data
            return the_df

        def recordfireHitsPlotter(a_df, title):
            the_df = {'KEY': title}
            data = []
            for k, v in a_df.iteritems():
                element = {}
                element['attr'] = k[0]
                if k[1] == 0:
                    element['Miss'] = v
                elif k[1] == 1:
                    element['Hit'] = v
                elif k[1] == -1:
                    element['No Fire'] = v
                else:
                    element[k[1]] = v
                data.append(element)
            the_df['Data'] = data
            return the_df

        def recordfireRangePlotter(a_df, title):
            the_df = {'KEY': title}
            data = []
            for k, v in a_df.iteritems():
                element = {}
                seq = (k[0], 'm')
                element['attr'] = "_".join((str(k[0]), 'm'))
                if k[1] == 0:
                    element['Miss'] = v
                elif k[1] == 1:
                    element['Hit'] = v
                elif k[1] == -1:
                    element['No Fire'] = v
                else:
                    element[k[1]] = v
                data.append(element)
            the_df['Data'] = data
            return the_df

        positions = []
        def position(n):
            if float(n) == 1.0:
                positions.append('Prone_Supported')
            elif float(n) == 2.0:
                positions.append('Prone_Unsupported')
            elif float(n) == 3.0:
                positions.append('Kneeling_Unsupported')
            else:
                positions.append('Dunno')


        self.df = pd.merge(self.records, self.learners, left_on='student_id', right_on='email_id', how='outer')



        # Task 1. Relative frequency of rating outcomes (unqualified, marksman, sharpshooter, expert)
        oqrf = getRelFreq(self.records, 'ability')
        overall_qr_rf = oneValuePlotter(oqrf, 'Qualification Ratings Relative Frequency')

        results = {}
        results['ovarall_qr_rf'] = overall_qr_rf

        # Task 2. Relative frequency of attempts to become qualified
        # Only do task 2 if time permits
        # TODO: Task 2. Relative frequency of attempts to become qualified
        # Cannot do; no data yet


        # Task 3. Relative frequency by each variable: hits, misses, no-fires for qualified and unqualified

        ql = self.records[(self.records.ability != 'Unqualified')]
        ul = self.records[(self.records.ability == 'Unqualified')]

        qrf = recordfireHitsPlotter(ql, 'isHit')
        urf = recordfireHitsPlotter(ul, 'isHit')

        # This is a much better/easier way.
        # a_h_ct = pd.crosstab(df.ability, df.isHit).apply(lambda row: row / row.sum(), axis=1)

        rf_hits ={'Qualified': qrf, 'Unqualified': urf}
        rf_pos = getRelFreqPlus(self.records, 'fire_pos')

        rf_qr = getRelFreqPlus(self.records, 'ability')
        rf_r = getRelFreqPlus(self.records, 'range')

        qual_rec_proportion = qualifiedHitsPlotter(rf_hits, 'Record Fire Proportion by Qualified')
        results['qual_rec_proportion'] = qual_rec_proportion


        # Task 4. Relative frequency of task 3 subset by qualification rating categories
        ability_rec_proportion = recordfireHitsPlotter(rf_qr, 'Record Fire Proportion by Qualification Rating')
        results['ability_rec_proportion'] = ability_rec_proportion

        # Task 5. Relative frequency by firing position
        pos_rec_proportion = recordfireHitsPlotter(rf_pos, 'Record Fire Proportion by Firing Position')
        results['pos_rec_proportion'] = pos_rec_proportion

        # Task 6. Relative frequency by range
        range_rec_proportion = recordfireRangePlotter(rf_r, 'Record Fire Proportion by Range')
        results['range_rec_proportion'] = range_rec_proportion

        # Task 7. Scatterplot of shots on target (this is just the json file)
        ot = self.records.ix[:, ['ability', 'student_id', 'shot_x', 'shot_y'] ]

        shot_locations = {'KEY':'Shot Location'}
        data = []
        for i in range(len(ot)):
            d = {}
            d['attr'] = ot.index[i]
            d['ability'] = ot.ability[i]
            d['x'] = ot.shot_x[i]
            d['y'] = ot.shot_y[i]
            d['student_id'] = "learner" + str(ot.student_id[i]) + "@example.com"
            data.append(d)
        shot_locations['Data'] = data

        results['shots_location'] = shot_locations

        # Task 8. Chi-sq testing on each dem variable (associated = True if Chi-sq stat >= 1.96 (two sigmas/95%)
        # Task 9. For each "associated = True" variable: Frequencies of students in each category of that variable; indicate
        #         which categories have more students than expected
        # I'm doing these together. For Task 8, there is a list of dicts. The task 9 data is included if the relationship
        # is significant.


        ## re-index for range and position crosstabs

        def getChis(crosstab):
            chi2, p, dof, ex = sps.chi2_contingency(crosstab[0])
            crit = sps.chi2.ppf(q=0.95, df=dof)
            if (crit < chi2):
                evaluation = True
            else:
                evaluation = False

            obs = crosstab[0].as_matrix()
            obs_list = obs.tolist()
            ex_list = ex.tolist()
            z_scores = zmap(obs_list, ex_list)
            z_list = z_scores.tolist()
            z_indicators = []
            for z in z_list:
                z_sig = ["+" if i > 1.96 else "-" if i < -1.96 else " " for i in z]
                z_indicators.append(z_sig)

            results = {'chi-sq': chi2,
                       'p-val': p,
                       'eval': evaluation,
                       'dof': dof,
                       'explanandum':crosstab[1],
                       'expected': ex_list,
                       'observed': obs_list,
                       'z_scores': z_indicators,
                       'row_lab': crosstab[0].index.tolist(),
                       'col_lab': crosstab[0].columns.tolist()
                       }
            return results

        # def getChis2(crosstab):
        #     chi2, p, dof, ex = sps.chi2_contingency(crosstab[0])
        #     crit = sps.chi2.ppf(q=0.95, df=dof)
        #     if (crit < chi2):
        #         evaluation = True
        #     else:
        #         evaluation = False
        #     results = {'chi-sq': chi2,
        #                'crit': sps.chi,
        #                'p-val': p,
        #                'eval': evaluation,
        #                'dof': dof,
        #                'explanandum': crosstab[1],
        #                'expected': ex.tolist(),
        #                'observed': crosstab[0].to_list(),
        #                'col_lab': crosstab[0].columns}
        #     for x in range(len(crosstab[0].index)):
        #         the_index = '_'.join(crosstab[0].index[x])
        #         results['row_lab'] = the_index
        #     return results

        #TODO: Figure out why labels/variables cannot be passed to the crosstab function. Doing this manually until then.
        a_df_ed = pd.crosstab(self.df.ability, self.df.education_level)
        a_df_hp = pd.crosstab(self.df.ability, self.df.handgun_prof)
        a_df_ig = pd.crosstab(self.df.ability, self.df.is_gamer)
        a_df_qp = pd.crosstab(self.df.ability, self.df.qualification_performance)
        a_df_r = pd.crosstab(self.df.ability, self.df['rank'])
        a_dfs = [a_df_ed, a_df_hp, a_df_ig, a_df_qp, a_df_r]

        dems = ['Education Level', 'Handgun Profiency', 'Computer Gamer', 'Rifle Experience',
                'Officer or Enlisted']

        facs = ['Qualification Rating', 'Fire Position', 'Range']


        a_chi =  [getChis(i) for i in zip(a_dfs, dems)]

        # Final outputs:
        # ...for Record Fire (generally) in 6th slide of req't presentation
        soft_surveys = []

        qual_rating_srv = {'explanans': facs[0]}

        def repackage(the_dict, the_list):
            var_list = []
            for d in the_list:
                model = {}
                model['explanandum'] = d['explanandum']
                model['active'] = d['eval']
                if (d['eval'] == True):
                    the_data = {}
                    the_data['rows'] = d['row_lab']
                    the_data['cols'] = d['col_lab']
                    the_data['obs_frequencies'] = d['observed']
                    the_data['exp_frequencies'] = d['expected']
                    the_data['significant'] = d['z_scores']
                    model['data'] = the_data
                var_list.append(model)
            the_dict['Data'] = var_list
            return the_dict

        stat_rel_variables.append(repackage(qual_rating_srv, a_chi))

        print stat_rel_variables[0]

        results['significant_relationships'] = stat_rel_variables

        for k, v in results.iteritems():
            with open("../data/results/" + k + ".json", "w") as data_out:
                json.dump(v, data_out)

## identify data files
learners = "../data/class_data/class.json"
rec_fire = "../data/reaper/record-fire-vCWD.csv"

the_format = 'csv'
# the_format = 'json'

## create the Chi-sq analysis object
results = chiAnalysis(learners, rec_fire)