import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from scenarios_and_approaches import *
import pathlib
data_path = str(pathlib.Path(__file__).parent.parent.resolve()) + '\Storing_Database.db'
store_path = str(pathlib.Path(__file__).parent.parent.resolve()) +'\obj2_plots'



conn = sqlite3.connect(data_path) 
c = conn.cursor()

c.execute("SELECT rowid,* FROM experiments")
experiments1 = c.fetchall()
experiments1 = np.asarray(experiments1)
conn.commit()
conn.close() 


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def set_box_multicolor(bp, color_vec):
    for index, color in enumerate(color_vec):
        plt.setp(bp['boxes'][index], color=color)
        plt.setp(bp['whiskers'][2*index+1], color=color)
        plt.setp(bp['caps'][2*index+1], color=color)
        plt.setp(bp['whiskers'][2 * index], color=color)
        plt.setp(bp['caps'][2 * index ], color=color)
        plt.setp(bp['medians'][index], color=color)


def add_scatter(point_data, color_vector, offset, alpha = 0.4):
    vals, color, xs = [], [],  []
    for i, col in enumerate(point_data):
        for d in col:
            vals.append(d)
            xs.append(np.random.normal(i, 0.02)*2 +offset)
            color.append(color_vector[i])
    plt.scatter(xs, vals, color=color, alpha=alpha, s=2)

def _percentage_greater_than(utility, utility_array):
    n = np.size(utility_array)
    score = 0
    for u in utility_array:
        if utility >= u:
            score += 1
    score = 100*score/n
    return score

def percentage_greater_than(utility, utility_array):
    return_list = []
    for u in utility:
        p = _percentage_greater_than(u, utility_array)
        return_list.append(p)
    return return_list

saturating_approaches = ['Saturating_Uniform_30','Saturating_3_Stage_SoftMax','SaturatingCRMExploit','SaturatingCRMBalanced_2']
peak_approaches = ['Peaking_Uniform_30','Peak_3_Stage_SoftMax','PeakCRMExploit','PeakCRMBalanced_2']
weight_approaches = ['Weight_Uniform_30','Weight_3_Stage_SoftMax','WeightCRMExploit','WeightCRMBalanced_2']
approaches = [saturating_approaches, peak_approaches, weight_approaches]
scenarios = [scenarioS1, scenarioS2, scenarioS3, scenarioS4, scenarioS5,
             scenarioP1, scenarioP2, scenarioP3, scenarioP4, scenarioP5,
             scenarioX1, scenarioX2, scenarioX3, scenarioX4]

ticks = [
'$Full Uniform Exploration$\n $Objective 1$', '$3Stage_{Softmax}$',  '$FullyContinual_{Standard}$', '$FullyContinual_{Balanced}$',
'$Full Uniform Exploration$\n $Objective 1$',  '$3Stage_{Softmax}$',  '$FullyContinual_{Standard}$', '$FullyContinual_{Balanced}$',
'$Full Uniform Exploration$\n $Objective 1$', '$3Stage_{Softmax}$',  '$FullyContinual_{Standard}$', '$FullyContinual_{Balanced}$',
]



data_uniform_all = np.empty((3,0))
data_Softmax_all = np.empty((3,0))
data_CRMExploit_all = np.empty((3,0))
data_CRMBalanced_all = np.empty((3,0))
data_all = [data_uniform_all,
            data_Softmax_all,
            data_CRMExploit_all, data_CRMBalanced_all]


data_uniform_all_p = np.empty((3,0))
data_Softmax_all_p = np.empty((3,0))
data_CRMExploit_all_p = np.empty((3,0))
data_CRMBalanced_all_p = np.empty((3,0))
data_all_p = [data_uniform_all_p,
              data_Softmax_all_p,
              data_CRMExploit_all_p, data_CRMBalanced_all_p]


data_uniform_all_i = np.empty((3,0))
data_Softmax_all_i = np.empty((3,0))
data_CRMExploit_all_i = np.empty((3,0))
data_CRMBalanced_all_i = np.empty((3,0))
data_all_i = [data_uniform_all_i,
              data_Softmax_all_i,
              data_CRMExploit_all_i, data_CRMBalanced_all_i]

data_uniform_all_ai = np.empty((3,0))

data_Softmax_all_ai = np.empty((3,0))
data_CRMExploit_all_ai = np.empty((3,0))
data_CRMBalanced_all_ai = np.empty((3,0))
data_all_ai = [data_uniform_all_ai,
              data_Softmax_all_ai,
              data_CRMExploit_all_ai, data_CRMBalanced_all_ai]

data_uniform_all_cr = np.empty((3,0))
data_Softmax_all_cr = np.empty((3,0))
data_CRMExploit_all_cr = np.empty((3,0))
data_CRMBalanced_all_cr = np.empty((3,0))
data_all_cr = [data_uniform_all_cr,
              data_Softmax_all_cr,
              data_CRMExploit_all_cr, data_CRMBalanced_all_cr]

data_uniform_all_cpr = np.empty((3,0))
data_Softmax_all_cpr = np.empty((3,0))
data_CRMExploit_all_cpr = np.empty((3,0))
data_CRMBalanced_all_cpr = np.empty((3,0))
data_all_cpr = [data_uniform_all_cpr,
              data_Softmax_all_cpr,
              data_CRMExploit_all_cpr, data_CRMBalanced_all_cpr]


data_uniform_all_cent = np.empty((3,0))
data_Softmax_all_cent = np.empty((3,0))
data_CRMExploit_all_cent = np.empty((3,0))
data_CRMBalanced_all_cent = np.empty((3,0))
data_all_cent = [data_uniform_all_cent,
              data_Softmax_all_cent,
              data_CRMExploit_all_cent, data_CRMBalanced_all_cent]


data_uniform_all_uti = np.empty((3,0))
data_Softmax_all_uti = np.empty((3,0))
data_CRMExploit_all_uti = np.empty((3,0))
data_CRMBalanced_all_uti = np.empty((3,0))
data_all_uti = [data_uniform_all_uti,
              data_Softmax_all_uti,
              data_CRMExploit_all_uti, data_CRMBalanced_all_uti]

mapping_names = {'S1':'Saturating 1',
                 'S2':'Saturating 2',
                 'S3':'Saturating 3',
                 'S4':'Saturating 4',
                 'S5':'Saturating 5',
                 'P1':'Peaking 1',
                 'P2':'Peaking 2',
                 'P3':'Peaking 3',
                 'P4':'Peaking 4',
                 'P5':'Peaking 5',
                 'X1':'Other 1',
                 'X2':'Other 2',
                 'X3':'Other 3',
                 'X4':'Other 4'}


query_doses = np.linspace(0,10,201)
for scenario in scenarios:
    scenario_label = scenario.scenario_ID
    utilities = scenario.dose_utility(query_doses)
    _, maximum_utility = scenario.calculate_utility_max()
    _, minimum_utility = scenario.calculate_utility_min()
    rows = np.where(experiments1[:, 1] == scenario_label)
    experiments2 = experiments1[rows]

    data_u = []
    data_sm = []
    data_crme = []
    data_crmb = []
    data = [data_u,  data_sm, data_crme, data_crmb]

    data_u_p = []
    data_sm_p = []
    data_crme_p = []
    data_crmb_p = []
    data_p = [data_u_p, data_sm_p, data_crme_p, data_crmb_p]

    data_u_i = []
    data_sm_i = []
    data_crme_i = []
    data_crmb_i = []
    data_i = [data_u_i, data_sm_i,  data_crme_i, data_crmb_i]

    data_u_ai = []
    data_sm_ai = []
    data_crme_ai = []
    data_crmb_ai = []
    data_ai = [data_u_ai,  data_sm_ai, data_crme_ai, data_crmb_ai]

    data_u_cr = []
    data_sm_cr = []
    data_crme_cr = []
    data_crmb_cr = []
    data_cr = [data_u_cr,  data_sm_cr, data_crme_cr, data_crmb_cr]

    data_u_cpr = []
    data_sm_cpr = []
    data_crme_cpr = []
    data_crmb_cpr = []
    data_cpr = [data_u_cpr, data_sm_cpr, data_crme_cpr, data_crmb_cpr]

    data_u_cent = []
    data_ul_cent = []
    data_sm_cent = []
    data_crme_cent = []
    data_crmb_cent = []
    data_cent = [data_u_cent, data_ul_cent, data_sm_cent,  data_crme_cent, data_crmb_cent]

    data_u_uti = []
    data_sm_uti = []
    data_crme_uti = []
    data_crmb_uti = []
    data_uti = [data_u_uti, data_sm_uti, data_crme_uti, data_crmb_uti]

    _data_u_all = []
    _data_sm_all = []
    _data_crme_all = []
    _data_crmb_all = []
    _data_all = [_data_u_all,_data_sm_all, _data_crme_all, _data_crmb_all]

    _data_u_all_p = []
    _data_sm_all_p = []
    _data_crme_all_p = []
    _data_crmb_all_p = []
    _data_all_p = [_data_u_all_p, _data_sm_all_p, _data_crme_all_p, _data_crmb_all_p]

    _data_u_all_i = []
    _data_sm_all_i = []
    _data_crme_all_i = []
    _data_crmb_all_i = []
    _data_all_i = [_data_u_all_i, _data_sm_all_i, _data_crme_all_i, _data_crmb_all_i]

    _data_u_all_ai = []
    _data_sm_all_ai = []
    _data_crme_all_ai = []
    _data_crmb_all_ai = []
    _data_all_ai = [_data_u_all_ai, _data_sm_all_ai, _data_crme_all_ai, _data_crmb_all_ai]

    _data_u_all_cr = []
    _data_sm_all_cr = []
    _data_crme_all_cr = []
    _data_crmb_all_cr = []
    _data_all_cr = [_data_u_all_cr,
                _data_sm_all_cr,
                _data_crme_all_cr, _data_crmb_all_cr]

    _data_u_all_cpr = []
    _data_sm_all_cpr = []
    _data_crme_all_cpr = []
    _data_crmb_all_cpr = []
    _data_all_cpr = [_data_u_all_cpr,
                    _data_sm_all_cpr,
                    _data_crme_all_cpr, _data_crmb_all_cpr]

    _data_u_all_cent = []
    _data_sm_all_cent = []
    _data_crme_all_cent = []
    _data_crmb_all_cent = []
    _data_all_cent = [_data_u_all_cent,
                    _data_sm_all_cent,
                    _data_crme_all_cent, _data_crmb_all_cent]

    _data_u_all_uti = []
    _data_sm_all_uti = []
    _data_crme_all_uti = []
    _data_crmb_all_uti = []
    _data_all_uti = [_data_u_all_uti,
                      _data_sm_all_uti,
                      _data_crme_all_uti, _data_crmb_all_uti]

    for index_shape, approach_shape in enumerate(approaches):
        print(approach_shape)

        for index_number, approach in enumerate(approach_shape):
            print(approach)
            rows = np.where(experiments2[:, 2] == approach)
            experiments = experiments2[rows]
            predicted_optimal_dose = experiments[:, 3].astype(np.float)
            predicted_optimal_response = experiments[:, 4].astype(np.float)
            actual_utility_at_predicted_optimal = scenario.dose_utility(predicted_optimal_dose)
            experimental_utility = experiments[:, 7].astype(np.float)

            # Wasted-utility
            wasted_utility = maximum_utility - actual_utility_at_predicted_optimal
            wasted_utility = wasted_utility.tolist()
            # percent_utility
            percent_utility = 100 * (1 - scenario.calculate_utility_normalised_score(actual_utility_at_predicted_optimal))
            percent_utility = percent_utility.tolist()
            # Inaccuracy
            inaccuracy = predicted_optimal_response - actual_utility_at_predicted_optimal
            absolute_inaccuracy = np.abs(inaccuracy)
            inaccuracy = inaccuracy.tolist()
            absolute_inaccuracy = absolute_inaccuracy.tolist()
            # Cum. Regret
            per_person_regret = maximum_utility - experimental_utility
            per_person_regret = per_person_regret.tolist()
            # Cum. Percentage Regret
            per_person_percent_regret = 100 * (1 - scenario.calculate_utility_normalised_score(experimental_utility))
            per_person_percent_regret = per_person_percent_regret.tolist()
            # Percentile
            percentile = percentage_greater_than(actual_utility_at_predicted_optimal, utilities)
            # Utility
            utility = actual_utility_at_predicted_optimal
            utility = utility.tolist()


            data[index_number].append(wasted_utility)
            data_p[index_number].append(percent_utility)
            data_i[index_number].append(inaccuracy)
            data_ai[index_number].append(absolute_inaccuracy)
            data_cr[index_number].append(per_person_regret)
            data_cpr[index_number].append(per_person_percent_regret)
            data_cent[index_number].append(percentile)
            data_uti[index_number].append(utility)
            _data_all[index_number].append(wasted_utility)
            _data_all_p[index_number].append(percent_utility)
            _data_all_i[index_number].append(inaccuracy)
            _data_all_ai[index_number].append(absolute_inaccuracy)
            _data_all_cr[index_number].append(per_person_regret)
            _data_all_cpr[index_number].append(per_person_percent_regret)
            _data_all_cent[index_number].append(percentile)
            _data_all_uti[index_number].append(utility)

    data_all = np.concatenate((data_all, _data_all), axis=2)
    data_all_p = np.concatenate((data_all_p, _data_all_p), axis=2)
    data_all_ai = np.concatenate((data_all_ai, _data_all_ai), axis=2)
    data_all_i = np.concatenate((data_all_i, _data_all_i), axis=2)
    data_all_cr = np.concatenate((data_all_cr, _data_all_cr), axis=2)
    data_all_cpr = np.concatenate((data_all_cpr, _data_all_cpr), axis=2)
    data_all_cent = np.concatenate((data_all_cent, _data_all_cent), axis=2)
    data_all_uti = np.concatenate((data_all_uti, _data_all_uti), axis=2)

    plt.figure(figsize=(10, 8))
    bpu = plt.boxplot(data[0], positions=np.array(range(3)) * 2.0 - 0.75, sym='', widths=0.2, whis=[5, 95])
    bpsm = plt.boxplot(data[1], positions=np.array(range(3)) * 2.0 - 0.25 , sym='', widths=0.2, whis=[5, 95])
    bpcrme = plt.boxplot(data[2], positions=np.array(range(3)) * 2.0 + 0.25, sym='', widths=0.2, whis=[5, 95])
    bpcrmb = plt.boxplot(data[3], positions=np.array(range(3)) * 2.0 + 0.75, sym='', widths=0.2, whis=[5, 95])

    add_scatter(data[0], ['red', 'green', 'blue'], -.75, alpha=0.3)
    add_scatter(data[1], ['red', 'green', 'blue'], -0.25, alpha=0.3)
    add_scatter(data[2], ['red', 'green', 'blue'], 0.25, alpha=0.3)
    add_scatter(data[3], ['red', 'green', 'blue'], .75, alpha=0.3)

    set_box_multicolor(bpu, ['red', 'green', 'blue'])
    set_box_multicolor(bpsm, ['red', 'green', 'blue'])
    set_box_multicolor(bpcrme, ['red', 'green', 'blue'])
    set_box_multicolor(bpcrmb, ['red', 'green', 'blue'])

    plt.xticks([-.75, -.25,  .25, .75,
                2 - .75, 2 - .25, 2 + 0.25, 2 + 0.75,
                4 - .75, 4 - .25, 4 + 0.25, 4 + 0.75], ticks)
    plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
    plt.ylim(0, 0.15)
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    plt.xlabel('Approach')
    plt.ylabel('Simple Regret, \n' +  mapping_names[scenario.scenario_ID])
    plt.title('Simple Regret for' + mapping_names[scenario.scenario_ID])
    plt.tight_layout()
    plt.plot([], c='red', label='Saturating')
    plt.plot([], c='green', label='Peaking')
    plt.plot([], c='blue', label='Weighted')
    plt.legend()

    name = store_path + '\_SR' + scenario.scenario_ID + '.png'

    plt.savefig(name)
    plt.close()

    plt.figure(figsize=(10, 8))
    bpu = plt.boxplot(data_p[0], positions=np.array(range(3)) * 2.0 - 0.75, sym='', widths=0.2, whis=[5, 95])
    bpsm = plt.boxplot(data_p[1], positions=np.array(range(3)) * 2.0 - .25, sym='', widths=0.2, whis=[5, 95])
    bpcrme = plt.boxplot(data_p[2], positions=np.array(range(3)) * 2.0 + .25, sym='', widths=0.2, whis=[5, 95])
    bpcrmb = plt.boxplot(data_p[3], positions=np.array(range(3)) * 2.0 + 0.75, sym='', widths=0.2, whis=[5, 95])

    add_scatter(data_p[0], ['red', 'green', 'blue'], -.75, alpha=0.3)
    add_scatter(data_p[1], ['red', 'green', 'blue'], -.25, alpha=0.3)
    add_scatter(data_p[2], ['red', 'green', 'blue'], .25, alpha=0.3)
    add_scatter(data_p[3], ['red', 'green', 'blue'], .75, alpha=0.3)

    set_box_multicolor(bpu, ['red', 'green', 'blue'])
    set_box_multicolor(bpsm, ['red', 'green', 'blue'])
    set_box_multicolor(bpcrme, ['red', 'green', 'blue'])
    set_box_multicolor(bpcrmb, ['red', 'green', 'blue'])

    plt.xticks([-.75, -.25,  .25, .75,
                2 - .75, 2 - .25, 2 + 0.25, 2 + 0.75,
                4 - .75, 4 - .25, 4 + 0.25, 4 + 0.75], ticks)
    plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
    plt.ylim(0, 100)
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    plt.xlabel('Approach')
    plt.ylabel('Percentage Simple Regret, \n' + mapping_names[scenario.scenario_ID])
    plt.ylabel('Percentage Simple Regret for ' + mapping_names[scenario.scenario_ID])
    plt.tight_layout()
    plt.plot([], c='red', label='Saturating')
    plt.plot([], c='green', label='Peaking')
    plt.plot([], c='blue', label='Weighted')
    plt.legend()

    name = store_path + '\_PSR' + scenario.scenario_ID + '.png'
    plt.savefig(name)
    plt.close()

    plt.figure(figsize=(10, 8))
    bpu = plt.boxplot(data_ai[0], positions=np.array(range(3)) * 2.0 - 0.75, sym='', widths=0.2, whis=[5, 95])
    bpsm = plt.boxplot(data_ai[1], positions=np.array(range(3)) * 2.0 - 0.25, sym='', widths=0.2, whis=[5, 95])
    bpcrme = plt.boxplot(data_ai[2], positions=np.array(range(3)) * 2.0 + 0.25, sym='', widths=0.2, whis=[5, 95])
    bpcrmb = plt.boxplot(data_ai[3], positions=np.array(range(3)) * 2.0 + 0.75, sym='', widths=0.2, whis=[5, 95])

    add_scatter(data_ai[0], ['red', 'green', 'blue'], -.75, alpha=0.3)
    add_scatter(data_ai[1], ['red', 'green', 'blue'], -0.25, alpha=0.3)
    add_scatter(data_ai[2], ['red', 'green', 'blue'], 0.25, alpha=0.3)
    add_scatter(data_ai[3], ['red', 'green', 'blue'], .75, alpha=0.3)

    set_box_multicolor(bpu, ['red', 'green', 'blue'])
    set_box_multicolor(bpsm, ['red', 'green', 'blue'])
    set_box_multicolor(bpcrme, ['red', 'green', 'blue'])
    set_box_multicolor(bpcrmb, ['red', 'green', 'blue'])

    plt.xticks([-.75, -.25,  .25, .75,
                2 - .75, 2 - .25, 2 + 0.25, 2 + 0.75,
                4 - .75, 4 - .25, 4 + 0.25, 4 + 0.75], ticks)
    plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
    plt.ylim(0, 0.15)
    plt.axhline(0, color='black')
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    plt.xlabel('Approach')
    plt.ylabel('Absolute Inaccuracy, \n' + mapping_names[scenario.scenario_ID])
    plt.title('Absolute Inaccuracy for' + mapping_names[scenario.scenario_ID])
    plt.tight_layout()
    plt.plot([], c='red', label='Saturating')
    plt.plot([], c='green', label='Peaking')
    plt.plot([], c='blue', label='Weighted')
    plt.legend()

    name = store_path + '\_AbsIA' + scenario.scenario_ID + '.png'
    plt.savefig(name)
    plt.close()

    plt.figure(figsize=(10, 8))
    bpu = plt.boxplot(data_i[0], positions=np.array(range(3)) * 2.0 - 0.75, sym='', widths=0.2, whis=[5, 95])
    bpsm = plt.boxplot(data_i[1], positions=np.array(range(3)) * 2.0 - 0.25, sym='', widths=0.2, whis=[5, 95])
    bpcrme = plt.boxplot(data_i[2], positions=np.array(range(3)) * 2.0 + 0.25, sym='', widths=0.2, whis=[5, 95])
    bpcrmb = plt.boxplot(data_i[3], positions=np.array(range(3)) * 2.0 + 0.75, sym='', widths=0.2, whis=[5, 95])

    add_scatter(data_i[0], ['red', 'green', 'blue'], -.75, alpha=0.3)
    add_scatter(data_i[1], ['red', 'green', 'blue'], -0.25, alpha=0.3)
    add_scatter(data_i[2], ['red', 'green', 'blue'], 0.25, alpha=0.3)
    add_scatter(data_i[3], ['red', 'green', 'blue'], .75, alpha=0.3)

    set_box_multicolor(bpu, ['red', 'green', 'blue'])
    set_box_multicolor(bpsm, ['red', 'green', 'blue'])
    set_box_multicolor(bpcrme, ['red', 'green', 'blue'])
    set_box_multicolor(bpcrmb, ['red', 'green', 'blue'])

    plt.xticks([-.75, -.25,  .25, .75,
                2 - .75, 2 - .25, 2 + 0.25, 2 + 0.75,
                4 - .75, 4 - .25, 4 + 0.25, 4 + 0.75], ticks)
    plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
    plt.ylim(-0.15, 0.15)
    plt.axhline(0, color='black')
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    plt.xlabel('Approach')
    plt.ylabel('Inaccuracy, \n' + mapping_names[scenario.scenario_ID])
    plt.title('Inaccuracy for ' + mapping_names[scenario.scenario_ID])
    plt.tight_layout()
    plt.plot([], c='red', label='Saturating')
    plt.plot([], c='green', label='Peaking')
    plt.plot([], c='blue', label='Weighted')
    plt.legend()

    name = store_path + '\_IA' + scenario.scenario_ID + '.png'
    plt.savefig(name)
    plt.close()

    plt.figure(figsize=(10, 8))
    bpu = plt.boxplot(data_cent[0], positions=np.array(range(3)) * 2.0 - 0.75, sym='', widths=0.2, whis=[5, 95])
    bpsm = plt.boxplot(data_cent[1], positions=np.array(range(3)) * 2.0 - 0.25, sym='', widths=0.2, whis=[5, 95])
    bpcrme = plt.boxplot(data_cent[2], positions=np.array(range(3)) * 2.0 + 0.25, sym='', widths=0.2, whis=[5, 95])
    bpcrmb = plt.boxplot(data_cent[3], positions=np.array(range(3)) * 2.0 + 0.75, sym='', widths=0.2, whis=[5, 95])

    add_scatter(data_cent[0], ['red', 'green', 'blue'], -.75, alpha=0.3)
    add_scatter(data_cent[1], ['red', 'green', 'blue'], -0.25, alpha=0.3)
    add_scatter(data_cent[2], ['red', 'green', 'blue'], 0.25, alpha=0.3)
    add_scatter(data_cent[3], ['red', 'green', 'blue'], .75, alpha=0.3)

    set_box_multicolor(bpu, ['red', 'green', 'blue'])
    set_box_multicolor(bpsm, ['red', 'green', 'blue'])
    set_box_multicolor(bpcrme, ['red', 'green', 'blue'])
    set_box_multicolor(bpcrmb, ['red', 'green', 'blue'])

    plt.xticks([-.75, -.25,  .25, .75,
                2 - .75, 2 - .25, 2 + 0.25, 2 + 0.75,
                4 - .75, 4 - .25, 4 + 0.25, 4 + 0.75], ticks)
    plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
    plt.ylim(0, 100)
    plt.axhline(0, color='black')
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    plt.xlabel('Approach')
    plt.ylabel('Percentile Utility, \n' + mapping_names[scenario.scenario_ID])
    plt.tight_layout()
    plt.plot([], c='red', label='Saturating')
    plt.plot([], c='green', label='Peaking')
    plt.plot([], c='blue', label='Weighted')
    plt.legend()

    name = store_path + '\_CENT' + scenario.scenario_ID + '.png'
    plt.savefig(name)
    plt.close()

    plt.figure(figsize=(10, 8))
    bpu = plt.boxplot(data_cr[0], positions=np.array(range(3)) * 2.0 - 0.75, sym='', widths=0.2, whis=[5, 95])
    bpsm = plt.boxplot(data_cr[1], positions=np.array(range(3)) * 2.0 -0.25, sym='', widths=0.2, whis=[5, 95])
    bpcrme = plt.boxplot(data_cr[2], positions=np.array(range(3)) * 2.0 +0.25, sym='', widths=0.2, whis=[5, 95])
    bpcrmb = plt.boxplot(data_cr[3], positions=np.array(range(3)) * 2.0 + 0.75, sym='', widths=0.2, whis=[5, 95])

    add_scatter(data_cr[0], ['red', 'green', 'blue'], -.75, alpha=0.3)
    add_scatter(data_cr[1], ['red', 'green', 'blue'], -0.25, alpha=0.3)
    add_scatter(data_cr[2], ['red', 'green', 'blue'], 0.25, alpha=0.3)
    add_scatter(data_cr[3], ['red', 'green', 'blue'], .75, alpha=0.3)

    set_box_multicolor(bpu, ['red', 'green', 'blue'])
    set_box_multicolor(bpsm, ['red', 'green', 'blue'])
    set_box_multicolor(bpcrme, ['red', 'green', 'blue'])
    set_box_multicolor(bpcrmb, ['red', 'green', 'blue'])

    plt.xticks([-.75, -.25,  .25, .75,
                2 - .75, 2 - .25, 2 + 0.25, 2 + 0.75,
                4 - .75, 4 - .25, 4 + 0.25, 4 + 0.75], ticks)
    plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
    plt.ylim(0, 0.15)
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    plt.xlabel('Approach')
    plt.ylabel('Average Regret, \n' + mapping_names[scenario.scenario_ID])
    plt.title('Average Regret for ' + mapping_names[scenario.scenario_ID])
    plt.tight_layout()
    plt.plot([], c='red', label='Saturating')
    plt.plot([], c='green', label='Peaking')
    plt.plot([], c='blue', label='Weighted')
    plt.legend()

    name = store_path + '\_AR' + scenario.scenario_ID + '.png'
    plt.savefig(name)
    plt.close()

    plt.figure(figsize=(10, 8))
    bpu = plt.boxplot(data_cpr[0], positions=np.array(range(3)) * 2.0 - 0.75, sym='', widths=0.2, whis=[5, 95])
    bpsm = plt.boxplot(data_cpr[1], positions=np.array(range(3)) * 2.0 - 0.25, sym='', widths=0.2, whis=[5, 95])
    bpcrme = plt.boxplot(data_cpr[2], positions=np.array(range(3)) * 2.0 + 0.25, sym='', widths=0.2, whis=[5, 95])
    bpcrmb = plt.boxplot(data_cpr[3], positions=np.array(range(3)) * 2.0 + 0.75, sym='', widths=0.2, whis=[5, 95])

    add_scatter(data_cpr[0], ['red', 'green', 'blue'], -.75, alpha=0.3)
    add_scatter(data_cpr[1], ['red', 'green', 'blue'], -0.25, alpha=0.3)
    add_scatter(data_cpr[2], ['red', 'green', 'blue'], 0.25, alpha=0.3)
    add_scatter(data_cpr[3], ['red', 'green', 'blue'], .75, alpha=0.3)

    set_box_multicolor(bpu, ['red', 'green', 'blue'])
    set_box_multicolor(bpsm, ['red', 'green', 'blue'])
    set_box_multicolor(bpcrme, ['red', 'green', 'blue'])
    set_box_multicolor(bpcrmb, ['red', 'green', 'blue'])

    plt.xticks([-.75, -.25, .25, .75,
                2 - .75, 2 - .25, 2 + 0.25, 2 + 0.75,
                4 - .75, 4 - .25, 4 + 0.25, 4 + 0.75], ticks)
    plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
    plt.ylim(-10,110)
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    plt.xlabel('Approach')
    plt.ylabel('Percentage Average Regret, \n' + mapping_names[scenario.scenario_ID])
    plt.title('Percentage Average Regret for ' + mapping_names[scenario.scenario_ID])
    plt.tight_layout()
    plt.plot([], c='red', label='Saturating')
    plt.plot([], c='green', label='Peaking')
    plt.plot([], c='blue', label='Weighted')
    plt.legend()

    name = store_path + '\_PAR' + scenario.scenario_ID + '.png'
    plt.savefig(name)
    plt.close()

    plt.figure(figsize=(10, 8))
    bpu = plt.boxplot(data_uti[0], positions=np.array(range(3)) * 2.0 - 0.75, sym='', widths=0.2, whis=[5, 95])
    bpsm = plt.boxplot(data_uti[1], positions=np.array(range(3)) * 2.0 - 0.25, sym='', widths=0.2, whis=[5, 95])
    bpcrme = plt.boxplot(data_uti[2], positions=np.array(range(3)) * 2.0 + 0.25, sym='', widths=0.2, whis=[5, 95])
    bpcrmb = plt.boxplot(data_uti[3], positions=np.array(range(3)) * 2.0 + 0.75, sym='', widths=0.2, whis=[5, 95])

    add_scatter(data_uti[0], ['red', 'green', 'blue'], -.75, alpha=0.3)
    add_scatter(data_uti[1], ['red', 'green', 'blue'], -0.25, alpha=0.3)
    add_scatter(data_uti[2], ['red', 'green', 'blue'], 0.25, alpha=0.3)
    add_scatter(data_uti[3], ['red', 'green', 'blue'], .75, alpha=0.3)

    set_box_multicolor(bpu, ['red', 'green', 'blue'])
    set_box_multicolor(bpsm, ['red', 'green', 'blue'])
    set_box_multicolor(bpcrme, ['red', 'green', 'blue'])
    set_box_multicolor(bpcrmb, ['red', 'green', 'blue'])

    plt.xticks([-.75, -.25,  .25, .75,
                2 - .75, 2 - .25, 2 + 0.25, 2 + 0.75,
                4 - .75, 4 - .25, 4 + 0.25, 4 + 0.75], ticks)
    plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
    plt.ylim(minimum_utility, maximum_utility)
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    plt.xlabel('Approach')
    plt.ylabel('Utility, \n' + mapping_names[scenario.scenario_ID])
    plt.tight_layout()
    plt.plot([], c='red', label='Saturating')
    plt.plot([], c='green', label='Peaking')
    plt.plot([], c='blue', label='Weighted')
    plt.legend()

    name = store_path + '\_U' + scenario.scenario_ID + '.png'
    plt.savefig(name)
    plt.close()


plt.figure(figsize=(10, 8))
bpu = plt.boxplot(data_all[0].T, positions=np.array(range(3)) * 2.0 - 0.75, sym='', widths=0.2, whis=[5, 95])
bpsm = plt.boxplot(data_all[1].T, positions=np.array(range(3)) * 2.0 - 0.25 , sym='', widths=0.2, whis=[5, 95])
bpcrme = plt.boxplot(data_all[2].T, positions=np.array(range(3)) * 2.0 + 0.25, sym='', widths=0.2, whis=[5, 95])
bpcrmb = plt.boxplot(data_all[3].T, positions=np.array(range(3)) * 2.0 + 0.75, sym='', widths=0.2, whis=[5, 95])



add_scatter(data_all[0],['red', 'green', 'blue'], -.75,alpha=0.1)
add_scatter(data_all[1], ['red', 'green', 'blue'], -0.25,alpha=0.1)
add_scatter(data_all[2], ['red', 'green', 'blue'], 0.25,alpha=0.1)
add_scatter(data_all[3], ['red', 'green', 'blue'], .75,alpha=0.1)

set_box_multicolor(bpu, ['red', 'green', 'blue'])
set_box_multicolor(bpsm, ['red', 'green', 'blue'])
set_box_multicolor(bpcrme, ['red', 'green', 'blue'])
set_box_multicolor(bpcrmb, ['red', 'green', 'blue'])



plt.xticks([-.75, -.25,  .25, .75,
                2 - .75, 2 - .25, 2 + 0.25, 2 + 0.75,
                4 - .75, 4 - .25, 4 + 0.25, 4 + 0.75], ticks)
plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
plt.ylim(0, 0.15)
plt.xticks(rotation=90)
plt.yticks(rotation=90)
plt.ylabel('Simple Regret')
plt.title('Simple Regret for all environments')
plt.xlabel('Approach')
plt.tight_layout()
plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.legend()

name = store_path + '\_SRAll.png'
plt.savefig(name)
plt.close()




plt.figure(figsize=(10, 8))
bpu = plt.boxplot(data_all_p[0].T, positions=np.array(range(3)) * 2.0 - 0.75, sym='', widths=0.2, whis=[5, 95])
bpsm = plt.boxplot(data_all_p[1].T, positions=np.array(range(3)) * 2.0 -0.25 , sym='', widths=0.2, whis=[5, 95])
bpcrme = plt.boxplot(data_all_p[2].T, positions=np.array(range(3)) * 2.0 + 0.25, sym='', widths=0.2, whis=[5, 95])
bpcrmb = plt.boxplot(data_all_p[3].T, positions=np.array(range(3)) * 2.0 + 0.75, sym='', widths=0.2, whis=[5, 95])



add_scatter(data_all_p[0],['red', 'green', 'blue'], -.75,alpha=0.1)
add_scatter(data_all_p[1], ['red', 'green', 'blue'], -0.25,alpha=0.1)
add_scatter(data_all_p[2], ['red', 'green', 'blue'], 0.25,alpha=0.1)
add_scatter(data_all_p[3], ['red', 'green', 'blue'], .75,alpha=0.1)

set_box_multicolor(bpu, ['red', 'green', 'blue'])
set_box_multicolor(bpsm, ['red', 'green', 'blue'])
set_box_multicolor(bpcrme, ['red', 'green', 'blue'])
set_box_multicolor(bpcrmb, ['red', 'green', 'blue'])



plt.xticks([-.75, -.25,  .25, .75,
                2 - .75, 2 - .25, 2 + 0.25, 2 + 0.75,
                4 - .75, 4 - .25, 4 + 0.25, 4 + 0.75], ticks)
plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
plt.ylim(0, 100)
plt.xticks(rotation=90)
plt.yticks(rotation=90)
plt.ylabel('Percentage Simple Regret')
plt.title('Percentage Simple Regret for all environments')
plt.xlabel('Approach')
plt.tight_layout()
plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.legend()

name = store_path + '\_PSRAll.png'
plt.savefig(name)
plt.close()
#
#
#
#
plt.figure(figsize=(10, 8))
bpu = plt.boxplot(data_all_ai[0].T, positions=np.array(range(3)) * 2.0 - 0.75, sym='', widths=0.2, whis=[5, 95])
bpsm = plt.boxplot(data_all_ai[1].T, positions=np.array(range(3)) * 2.0 - 0.25 , sym='', widths=0.2, whis=[5, 95])
bpcrme = plt.boxplot(data_all_ai[2].T, positions=np.array(range(3)) * 2.0 + 0.25, sym='', widths=0.2, whis=[5, 95])
bpcrmb = plt.boxplot(data_all_ai[3].T, positions=np.array(range(3)) * 2.0 + 0.75, sym='', widths=0.2, whis=[5, 95])



add_scatter(data_all_ai[0],['red', 'green', 'blue'], -.75,alpha=0.1)
add_scatter(data_all_ai[1], ['red', 'green', 'blue'], -0.25,alpha=0.1)
add_scatter(data_all_ai[2], ['red', 'green', 'blue'], 0.25,alpha=0.1)
add_scatter(data_all_ai[3], ['red', 'green', 'blue'], .75,alpha=0.1)

set_box_multicolor(bpu, ['red', 'green', 'blue'])
set_box_multicolor(bpsm, ['red', 'green', 'blue'])
set_box_multicolor(bpcrme, ['red', 'green', 'blue'])
set_box_multicolor(bpcrmb, ['red', 'green', 'blue'])



plt.xticks([-.75, -.25,  .25, .75,
                2 - .75, 2 - .25, 2 + 0.25, 2 + 0.75,
                4 - .75, 4 - .25, 4 + 0.25, 4 + 0.75], ticks)
plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
plt.ylim(0, 0.15)
plt.axhline(0, color='black')
plt.xticks(rotation=90)
plt.yticks(rotation=90)
plt.ylabel('Absolute Inaccuracy')
plt.title('Absolute Inaccuracy for all environments')
plt.xlabel('Approach')
plt.tight_layout()
plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.legend()

name = store_path + '\_AbsIAAll.png'
plt.savefig(name)
plt.close()
#
#
#
plt.figure(figsize=(10, 8))
bpu = plt.boxplot(data_all_i[0].T, positions=np.array(range(3)) * 2.0 - 0.75, sym='', widths=0.2, whis=[5, 95])
bpsm = plt.boxplot(data_all_i[1].T, positions=np.array(range(3)) * 2.0 - 0.25 , sym='', widths=0.2, whis=[5, 95])
bpcrme = plt.boxplot(data_all_i[2].T, positions=np.array(range(3)) * 2.0 + 0.25, sym='', widths=0.2, whis=[5, 95])
bpcrmb = plt.boxplot(data_all_i[3].T, positions=np.array(range(3)) * 2.0 + 0.75, sym='', widths=0.2, whis=[5, 95])



add_scatter(data_all_i[0],['red', 'green', 'blue'], -.75,alpha=0.1)
add_scatter(data_all_i[1], ['red', 'green', 'blue'], -0.25,alpha=0.1)
add_scatter(data_all_i[2], ['red', 'green', 'blue'], 0.25,alpha=0.1)
add_scatter(data_all_i[3], ['red', 'green', 'blue'], .75,alpha=0.1)

set_box_multicolor(bpu, ['red', 'green', 'blue'])
set_box_multicolor(bpsm, ['red', 'green', 'blue'])
set_box_multicolor(bpcrme, ['red', 'green', 'blue'])
set_box_multicolor(bpcrmb, ['red', 'green', 'blue'])



plt.xticks([-.75, -.25,  .25, .75,
                2 - .75, 2 - .25, 2 + 0.25, 2 + .75,
                4 - .75, 4 - .25, 4 + 0.25, 4 + .75], ticks)
plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
plt.ylim(-0.15, 0.15)
plt.axhline(0, color='black')
plt.xticks(rotation=90)
plt.yticks(rotation=90)
plt.ylabel('Inaccuracy')
plt.title('Inaccuracy for all environments')
plt.xlabel('Approach')
plt.tight_layout()
plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.legend()

name = store_path + '\_IAAll.png'
plt.savefig(name)
plt.close()




plt.figure(figsize=(10, 8))
bpu = plt.boxplot(data_all_cr[0].T, positions=np.array(range(3)) * 2.0 - .75, sym='', widths=0.2, whis=[5, 95])
bpsm = plt.boxplot(data_all_cr[1].T, positions=np.array(range(3)) * 2.0 -0.25 , sym='', widths=0.2, whis=[5, 95])
bpcrme = plt.boxplot(data_all_cr[2].T, positions=np.array(range(3)) * 2.0 + 0.25, sym='', widths=0.2, whis=[5, 95])
bpcrmb = plt.boxplot(data_all_cr[3].T, positions=np.array(range(3)) * 2.0 + .75, sym='', widths=0.2, whis=[5, 95])



add_scatter(data_all_cr[0],['red', 'green', 'blue'], -.75,alpha=0.1)
add_scatter(data_all_cr[1], ['red', 'green', 'blue'], -0.25,alpha=0.1)
add_scatter(data_all_cr[2], ['red', 'green', 'blue'], 0.25,alpha=0.1)
add_scatter(data_all_cr[3], ['red', 'green', 'blue'], .75,alpha=0.1)

set_box_multicolor(bpu, ['red', 'green', 'blue'])
set_box_multicolor(bpsm, ['red', 'green', 'blue'])
set_box_multicolor(bpcrme, ['red', 'green', 'blue'])
set_box_multicolor(bpcrmb, ['red', 'green', 'blue'])



plt.xticks([-.75, -.25,  .25, .75,
                2 - .75, 2 - .25, 2 + 0.25, 2 + .75,
                4 - .75, 4 - .25, 4 + 0.25, 4 + .75], ticks)
plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
plt.ylim(0, 0.15)
plt.xticks(rotation=90)
plt.yticks(rotation=90)
plt.ylabel('Average Regret')
plt.title('Average Regret for all environments')
plt.xlabel('Approach')
plt.tight_layout()
plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.legend()

name = store_path + '\_ARAll.png'
plt.savefig(name)
plt.close()



plt.figure(figsize=(10, 8))
bpu = plt.boxplot(data_all_cpr[0].T, positions=np.array(range(3)) * 2.0 - .75, sym='', widths=0.2, whis=[5, 95])
bpsm = plt.boxplot(data_all_cpr[1].T, positions=np.array(range(3)) * 2.0 -0.25 , sym='', widths=0.2, whis=[5, 95])
bpcrme = plt.boxplot(data_all_cpr[2].T, positions=np.array(range(3)) * 2.0 + 0.25, sym='', widths=0.2, whis=[5, 95])
bpcrmb = plt.boxplot(data_all_cpr[3].T, positions=np.array(range(3)) * 2.0 + .75, sym='', widths=0.2, whis=[5, 95])



add_scatter(data_all_cpr[0],['red', 'green', 'blue'], -.75,alpha=0.1)
add_scatter(data_all_cpr[1], ['red', 'green', 'blue'], -0.25,alpha=0.1)
add_scatter(data_all_cpr[2], ['red', 'green', 'blue'], 0.25,alpha=0.1)
add_scatter(data_all_cpr[3], ['red', 'green', 'blue'], .75,alpha=0.1)

set_box_multicolor(bpu, ['red', 'green', 'blue'])
set_box_multicolor(bpsm, ['red', 'green', 'blue'])
set_box_multicolor(bpcrme, ['red', 'green', 'blue'])
set_box_multicolor(bpcrmb, ['red', 'green', 'blue'])



plt.xticks([-.75, -.25,  .25, .75,
                2 - .75, 2 - .25, 2 + 0.25, 2 + .75,
                4 - .75, 4 - .25, 4 + 0.25, 4 + .75], ticks)
plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
plt.ylim(-10, 110)
plt.xticks(rotation=90)
plt.yticks(rotation=90)
plt.ylabel('Percentage Average Regret')
plt.title('Percentage Average Regret for all environments')
plt.xlabel('Approach')
plt.tight_layout()
plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.legend()

name = store_path + '\_PARAll.png'
plt.savefig(name)
plt.close()




plt.figure(figsize=(10, 8))
bpu = plt.boxplot(data_all_cent[0].T, positions=np.array(range(3)) * 2.0 - .75, sym='', widths=0.2, whis=[5, 95])
bpsm = plt.boxplot(data_all_cent[1].T, positions=np.array(range(3)) * 2.0 - 0.25  , sym='', widths=0.2, whis=[5, 95])
bpcrme = plt.boxplot(data_all_cent[2].T, positions=np.array(range(3)) * 2.0 + 0.25, sym='', widths=0.2, whis=[5, 95])
bpcrmb = plt.boxplot(data_all_cent[3].T, positions=np.array(range(3)) * 2.0 + .75, sym='', widths=0.2, whis=[5, 95])



add_scatter(data_all_cent[0],['red', 'green', 'blue'], -.75,alpha=0.1)
add_scatter(data_all_cent[1], ['red', 'green', 'blue'], -0.25,alpha=0.1)
add_scatter(data_all_cent[2], ['red', 'green', 'blue'], 0.25,alpha=0.1)
add_scatter(data_all_cent[3], ['red', 'green', 'blue'], .75,alpha=0.1)

set_box_multicolor(bpu, ['red', 'green', 'blue'])
set_box_multicolor(bpsm, ['red', 'green', 'blue'])
set_box_multicolor(bpcrme, ['red', 'green', 'blue'])
set_box_multicolor(bpcrmb, ['red', 'green', 'blue'])



plt.xticks([-.75, -.25,  .25, .75,
                2 - .75, 2 - .25, 2 + 0.25, 2 + .75,
                4 - .75, 4 - .25, 4 + 0.25, 4 + .75], ticks)
plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
plt.ylim(0, 100)
plt.axhline(0, color='black')
plt.xticks(rotation=90)
plt.yticks(rotation=90)
plt.title('Percentile Utility for all environments')
plt.xlabel('Approach')
plt.ylabel('Percentile Utility')
plt.tight_layout()
plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.legend()

name = store_path + '\_CENTAll.png'
plt.savefig(name)
plt.close()
#
print(data_all_cent)
print(np.shape(data_all_cent))
print(data_all_uti)
print(np.shape(data_all_uti))
plt.figure(figsize=(10, 8))
bpu = plt.boxplot(data_all_uti[0].T, positions=np.array(range(3)) * 2.0 - 0.75, sym='', widths=0.2, whis=[5, 95])
bpsm = plt.boxplot(data_all_uti[1].T, positions=np.array(range(3)) * 2.0 - 0.25 , sym='', widths=0.2, whis=[5, 95])
bpcrme = plt.boxplot(data_all_uti[2].T, positions=np.array(range(3)) * 2.0 + 0.25, sym='', widths=0.2, whis=[5, 95])
bpcrmb = plt.boxplot(data_all_uti[3].T, positions=np.array(range(3)) * 2.0 + 0.75, sym='', widths=0.2, whis=[5, 95])



add_scatter(data_all_uti[0],['red', 'green', 'blue'], -.75,alpha=0.1)
add_scatter(data_all_uti[1], ['red', 'green', 'blue'], -0.25,alpha=0.1)
add_scatter(data_all_uti[2], ['red', 'green', 'blue'], 0.25,alpha=0.1)
add_scatter(data_all_uti[3], ['red', 'green', 'blue'], .75,alpha=0.1)

set_box_multicolor(bpu, ['red', 'green', 'blue'])
set_box_multicolor(bpsm, ['red', 'green', 'blue'])
set_box_multicolor(bpcrme, ['red', 'green', 'blue'])
set_box_multicolor(bpcrmb, ['red', 'green', 'blue'])



plt.xticks([-.75, -.25,  .25, .75,
                2 - .75, 2 - .25, 2 + 0.25, 2 + 0.75,
                4 - .75, 4 - .25, 4 + 0.25, 4 + 0.75], ticks)
plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
# plt.ylim(0, 0.15)
plt.xticks(rotation=90)
plt.yticks(rotation=90)
plt.ylabel('Utility')
plt.title('Utility for all environments')
plt.xlabel('Approach')
plt.tight_layout()
plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.legend()

name = store_path + '\_UAll.png'
plt.savefig(name)
plt.close()
