'''
Plots results from objective 1
'''

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from scenarios_and_approaches import *

import pathlib
data_path = str(pathlib.Path(__file__).parent.parent.resolve()) + '\Storing_Database.db'
store_path = str(pathlib.Path(__file__).parent.parent.resolve()) +'\obj1_plots'


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

saturating_approaches = ['Saturating_Uniform_10','Saturating_Uniform_30','Saturating_Uniform_60','Saturating_Uniform_100']
peak_approaches = ['Peaking_Uniform_10','Peaking_Uniform_30','Peaking_Uniform_60','Peaking_Uniform_100']
weight_approaches = ['Weight_Uniform_10','Weight_Uniform_30','Weight_Uniform_60','Weight_Uniform_100']
approaches = [saturating_approaches, peak_approaches, weight_approaches]
scenarios = [scenarioS1, scenarioS2, scenarioS3, scenarioS4, scenarioS5,
             scenarioP1, scenarioP2, scenarioP3, scenarioP4, scenarioP5,
             scenarioX1, scenarioX2, scenarioX3, scenarioX4]

ticks = ['10','30','60','100',
         '10','30','60','100',
         '10','30','60','100']

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



data_10_all = np.empty((3,0))
data_30_all = np.empty((3,0))
data_60_all = np.empty((3,0))
data_100_all = np.empty((3,0))
data_all = [data_10_all, data_30_all, data_60_all, data_100_all]

data_10_all_p = np.empty((3,0))
data_30_all_p = np.empty((3,0))
data_60_all_p = np.empty((3,0))
data_100_all_p = np.empty((3,0))
data_all_p = [data_10_all_p, data_30_all_p, data_60_all_p, data_100_all_p]

data_10_all_i = np.empty((3,0))
data_30_all_i = np.empty((3,0))
data_60_all_i = np.empty((3,0))
data_100_all_i = np.empty((3,0))
data_all_i = [data_10_all_i, data_30_all_i, data_60_all_i, data_100_all_i]

data_10_all_ai = np.empty((3,0))
data_30_all_ai = np.empty((3,0))
data_60_all_ai = np.empty((3,0))
data_100_all_ai = np.empty((3,0))
data_all_ai = [data_10_all_ai, data_30_all_ai, data_60_all_ai, data_100_all_ai]

data_10_all_cr = np.empty((3,0))
data_30_all_cr = np.empty((3,0))
data_60_all_cr = np.empty((3,0))
data_100_all_cr = np.empty((3,0))
data_all_cr = [data_10_all_cr, data_30_all_cr, data_60_all_cr, data_100_all_cr]

data_10_all_cpr = np.empty((3,0))
data_30_all_cpr = np.empty((3,0))
data_60_all_cpr = np.empty((3,0))
data_100_all_cpr = np.empty((3,0))
data_all_cpr = [data_10_all_cpr, data_30_all_cpr, data_60_all_cpr, data_100_all_cpr]

data_10_all_cent = np.empty((3,0))
data_30_all_cent = np.empty((3,0))
data_60_all_cent = np.empty((3,0))
data_100_all_cent = np.empty((3,0))
data_all_cent = [data_10_all_cent, data_30_all_cent, data_60_all_cent, data_100_all_cent]

data_10_all_u = np.empty((3,0))
data_30_all_u = np.empty((3,0))
data_60_all_u = np.empty((3,0))
data_100_all_u = np.empty((3,0))
data_all_u = [data_10_all_u, data_30_all_u, data_60_all_u, data_100_all_u]


query_doses = np.linspace(0,10,201)
for scenario in scenarios:
    print(mapping_names[scenario.scenario_ID])
    scenario_label = scenario.scenario_ID
    _, maximum_utility = scenario.calculate_utility_max()
    utilities = scenario.dose_utility(query_doses)
    rows = np.where(experiments1[:, 1] == scenario_label)
    experiments2 = experiments1[rows]
    data_10 = []
    data_30 = []
    data_60 = []
    data_100 = []
    data = [data_10, data_30, data_60, data_100]

    data_10_p = []
    data_30_p = []
    data_60_p = []
    data_100_p = []
    data_p = [data_10_p, data_30_p, data_60_p, data_100_p]

    data_10_i = []
    data_30_i = []
    data_60_i = []
    data_100_i = []
    data_i = [data_10_i, data_30_i, data_60_i, data_100_i]

    data_10_ai = []
    data_30_ai = []
    data_60_ai = []
    data_100_ai = []
    data_ai = [data_10_ai, data_30_ai, data_60_ai, data_100_ai]

    data_10_cr = []
    data_30_cr = []
    data_60_cr = []
    data_100_cr = []
    data_cr = [data_10_cr, data_30_cr, data_60_cr, data_100_cr]

    data_10_cpr = []
    data_30_cpr = []
    data_60_cpr = []
    data_100_cpr = []
    data_cpr = [data_10_cpr, data_30_cpr, data_60_cpr, data_100_cpr]

    data_10_cent = []
    data_30_cent = []
    data_60_cent = []
    data_100_cent = []
    data_cent = [data_10_cent , data_30_cent, data_60_cent, data_100_cent]

    data_10_u = []
    data_30_u = []
    data_60_u = []
    data_100_u = []
    data_u = [data_10_u, data_30_u, data_60_u, data_100_u]

    _data_10_all = []
    _data_30_all = []
    _data_60_all = []
    _data_100_all = []
    _data_all = [_data_10_all, _data_30_all, _data_60_all, _data_100_all]

    _data_10_all_i = []
    _data_30_all_i = []
    _data_60_all_i = []
    _data_100_all_i = []
    _data_all_i = [_data_10_all_i, _data_30_all_i, _data_60_all_i, _data_100_all_i]

    _data_10_all_ai = []
    _data_30_all_ai = []
    _data_60_all_ai = []
    _data_100_all_ai = []
    _data_all_ai = [_data_10_all_ai, _data_30_all_ai, _data_60_all_ai, _data_100_all_ai]

    _data_10_all_p = []
    _data_30_all_p = []
    _data_60_all_p = []
    _data_100_all_p = []
    _data_all_p = [_data_10_all_p, _data_30_all_p, _data_60_all_p, _data_100_all_p]

    _data_10_all_cr = []
    _data_30_all_cr = []
    _data_60_all_cr = []
    _data_100_all_cr = []
    _data_all_cr = [_data_10_all_cr, _data_30_all_cr, _data_60_all_cr, _data_100_all_cr]

    _data_10_all_cpr = []
    _data_30_all_cpr = []
    _data_60_all_cpr = []
    _data_100_all_cpr = []
    _data_all_cpr = [_data_10_all_cpr, _data_30_all_cpr, _data_60_all_cpr, _data_100_all_cpr]

    _data_10_all_cent = []
    _data_30_all_cent = []
    _data_60_all_cent = []
    _data_100_all_cent = []
    _data_all_cent= [_data_10_all_cent, _data_30_all_cent, _data_60_all_cent, _data_100_all_cent]

    _data_10_all_u = []
    _data_30_all_u = []
    _data_60_all_u = []
    _data_100_all_u = []
    _data_all_u = [_data_10_all_u, _data_30_all_u, _data_60_all_u, _data_100_all_u]


    for index_shape, approach_shape in enumerate(approaches):
        print(index_shape, approach_shape)

        for index_number, approach in enumerate(approach_shape):
            print(index_number, approach)
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
            percent_utility = 100 * (1- scenario.calculate_utility_normalised_score(actual_utility_at_predicted_optimal))
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
            data_u[index_number].append(utility)
            _data_all[index_number].append(wasted_utility)
            _data_all_i[index_number].append(inaccuracy)
            _data_all_ai[index_number].append(absolute_inaccuracy)
            _data_all_cr[index_number].append(per_person_regret)
            _data_all_cpr[index_number].append(per_person_percent_regret)
            _data_all_p[index_number].append(percent_utility)
            _data_all_cent[index_number].append(percentile)
            _data_all_u[index_number].append(utility)

    # print(np.shape(data_all), np.shape(_data_all), np.shape(data_all_p), np.shape(_data_all_p))
    data_all = np.concatenate((data_all, _data_all), axis=2)
    data_all_i = np.concatenate((data_all_i, _data_all_i), axis=2)
    data_all_ai = np.concatenate((data_all_ai, _data_all_ai), axis=2)
    data_all_p = np.concatenate((data_all_p, _data_all_p), axis=2)
    data_all_cr = np.concatenate((data_all_cr, _data_all_cr), axis=2)
    data_all_cpr = np.concatenate((data_all_cpr, _data_all_cpr), axis=2)
    data_all_cent = np.concatenate((data_all_cent, _data_all_cent), axis=2)
    data_all_u = np.concatenate((data_all_u, _data_all_u), axis=2)
    #
    plt.figure(figsize=(10, 8))
    bp1 = plt.boxplot(data[0], positions=np.array(range(len(data[0]))) * 2.0 - 0.6, sym='', widths=0.3, whis=[5, 95])
    bp3 = plt.boxplot(data[1], positions=np.array(range(len(data[1]))) * 2.0 - 0.2, sym='', widths=0.3, whis=[5, 95])
    bp6 = plt.boxplot(data[2], positions=np.array(range(len(data[2]))) * 2.0 + 0.2, sym='', widths=0.3, whis=[5, 95])
    bp10 = plt.boxplot(data[3], positions=np.array(range(len(data[3]))) * 2.0 + 0.6, sym='', widths=0.3, whis=[5, 95])



    add_scatter(data[0],['red', 'green', 'blue'], -.6)
    add_scatter(data[1], ['red', 'green', 'blue'], -.2)
    add_scatter(data[2], ['red', 'green', 'blue'], .2)
    add_scatter(data[3], ['red', 'green', 'blue'], .6)

    set_box_multicolor(bp1, ['red', 'green', 'blue'])
    set_box_multicolor(bp3, ['red', 'green', 'blue'])
    set_box_multicolor(bp6, ['red', 'green', 'blue'])
    set_box_multicolor(bp10, ['red', 'green', 'blue'])

    plt.plot([], c='red', label='Saturating')
    plt.plot([], c='green', label='Peaking')
    plt.plot([], c='blue', label='Weighted')
    plt.legend()



    plt.xticks([-.6, -.2, .2, .6,
                2-.6, 2-.2, 2+.2, 2+.6,
                4-.6, 4-.2, 4+.2, 4+.6], ticks)
    plt.xlim(-1, ((len(ticks) * 2) / 4)-1 )
    plt.ylim(0, 0.15)
    plt.title('Simple Regret for ' + mapping_names[scenario.scenario_ID])
    plt.xlabel('Approach')
    plt.ylabel('Simple Regret')
    plt.plot([], c='red', label='Saturating')
    plt.plot([], c='green', label='Peaking')
    plt.plot([], c='blue', label='Weighted')
    plt.tight_layout()
    name = store_path + '\_SR'+ scenario.scenario_ID +'.png'
    plt.savefig(name)
    plt.close()

    plt.figure(figsize=(10, 8))
    bp1 = plt.boxplot(data_p[0], positions=np.array(range(len(data[0]))) * 2.0 - 0.6, sym='', widths=0.3, whis=[5, 95])
    bp3 = plt.boxplot(data_p[1], positions=np.array(range(len(data[1]))) * 2.0 - 0.2, sym='', widths=0.3, whis=[5, 95])
    bp6 = plt.boxplot(data_p[2], positions=np.array(range(len(data[2]))) * 2.0 + 0.2, sym='', widths=0.3, whis=[5, 95])
    bp10 = plt.boxplot(data_p[3], positions=np.array(range(len(data[3]))) * 2.0 + 0.6, sym='', widths=0.3, whis=[5, 95])

    add_scatter(data_p[0], ['red', 'green', 'blue'], -.6)
    add_scatter(data_p[1], ['red', 'green', 'blue'], -.2)
    add_scatter(data_p[2], ['red', 'green', 'blue'], .2)
    add_scatter(data_p[3], ['red', 'green', 'blue'], .6)

    set_box_multicolor(bp1, ['red', 'green', 'blue'])
    set_box_multicolor(bp3, ['red', 'green', 'blue'])
    set_box_multicolor(bp6, ['red', 'green', 'blue'])
    set_box_multicolor(bp10, ['red', 'green', 'blue'])

    plt.plot([], c='red', label='Saturating')
    plt.plot([], c='green', label='Peaking')
    plt.plot([], c='blue', label='Weighted')
    plt.legend()

    plt.xticks([-.6, -.2, .2, .6,
                2 - .6, 2 - .2, 2 + .2, 2 + .6,
                4 - .6, 4 - .2, 4 + .2, 4 + .6], ticks)
    plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
    plt.ylim(0, 100)
    plt.title('Percentage Simple Regret for ' + mapping_names[scenario.scenario_ID])
    plt.xlabel('Approach')
    plt.ylabel('Percentage Simple Regret')
    plt.plot([], c='red', label='Saturating')
    plt.plot([], c='green', label='Peaking')
    plt.plot([], c='blue', label='Weighted')
    plt.tight_layout()
    name = store_path + '\_PSR' + scenario.scenario_ID + '.png'
    plt.savefig(name)
    plt.close()

    plt.figure(figsize=(10, 8))
    bp1 = plt.boxplot(data_cent[0], positions=np.array(range(len(data[0]))) * 2.0 - 0.6, sym='', widths=0.3, whis=[5, 95])
    bp3 = plt.boxplot(data_cent[1], positions=np.array(range(len(data[1]))) * 2.0 - 0.2, sym='', widths=0.3, whis=[5, 95])
    bp6 = plt.boxplot(data_cent[2], positions=np.array(range(len(data[2]))) * 2.0 + 0.2, sym='', widths=0.3, whis=[5, 95])
    bp10 = plt.boxplot(data_cent[3], positions=np.array(range(len(data[3]))) * 2.0 + 0.6, sym='', widths=0.3, whis=[5, 95])

    add_scatter(data_cent[0], ['red', 'green', 'blue'], -.6)
    add_scatter(data_cent[1], ['red', 'green', 'blue'], -.2)
    add_scatter(data_cent[2], ['red', 'green', 'blue'], .2)
    add_scatter(data_cent[3], ['red', 'green', 'blue'], .6)

    set_box_multicolor(bp1, ['red', 'green', 'blue'])
    set_box_multicolor(bp3, ['red', 'green', 'blue'])
    set_box_multicolor(bp6, ['red', 'green', 'blue'])
    set_box_multicolor(bp10, ['red', 'green', 'blue'])

    plt.plot([], c='red', label='Saturating')
    plt.plot([], c='green', label='Peaking')
    plt.plot([], c='blue', label='Weighted')
    plt.legend()

    plt.xticks([-.6, -.2, .2, .6,
                2 - .6, 2 - .2, 2 + .2, 2 + .6,
                4 - .6, 4 - .2, 4 + .2, 4 + .6], ticks)
    plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
    plt.ylim(0, 100)
    plt.title('Percentile Utility for ' + mapping_names[scenario.scenario_ID])
    plt.xlabel('Approach')
    plt.ylabel('Percentile Utility')
    plt.plot([], c='red', label='Saturating')
    plt.plot([], c='green', label='Peaking')
    plt.plot([], c='blue', label='Weighted')
    plt.tight_layout()
    name = store_path + '\_CENT' + scenario.scenario_ID + '.png'
    plt.savefig(name)
    plt.close()

    plt.figure(figsize=(10, 8))
    bp1 = plt.boxplot(data_u[0], positions=np.array(range(len(data_u[0]))) * 2.0 - 0.6, sym='', widths=0.3, whis=[5, 95])
    bp3 = plt.boxplot(data_u[1], positions=np.array(range(len(data_u[1]))) * 2.0 - 0.2, sym='', widths=0.3, whis=[5, 95])
    bp6 = plt.boxplot(data_u[2], positions=np.array(range(len(data_u[2]))) * 2.0 + 0.2, sym='', widths=0.3, whis=[5, 95])
    bp10 = plt.boxplot(data_u[3], positions=np.array(range(len(data_u[3]))) * 2.0 + 0.6, sym='', widths=0.3, whis=[5, 95])



    add_scatter(data_u[0],['red', 'green', 'blue'], -.6)
    add_scatter(data_u[1], ['red', 'green', 'blue'], -.2)
    add_scatter(data_u[2], ['red', 'green', 'blue'], .2)
    add_scatter(data_u[3], ['red', 'green', 'blue'], .6)

    set_box_multicolor(bp1, ['red', 'green', 'blue'])
    set_box_multicolor(bp3, ['red', 'green', 'blue'])
    set_box_multicolor(bp6, ['red', 'green', 'blue'])
    set_box_multicolor(bp10, ['red', 'green', 'blue'])

    plt.plot([], c='red', label='Saturating')
    plt.plot([], c='green', label='Peaking')
    plt.plot([], c='blue', label='Weighted')
    plt.legend()



    plt.xticks([-.6, -.2, .2, .6,
                2-.6, 2-.2, 2+.2, 2+.6,
                4-.6, 4-.2, 4+.2, 4+.6], ticks)
    plt.xlim(-1, ((len(ticks) * 2) / 4)-1 )
    plt.ylim(-.15, 0.15)
    plt.title('Utility for ' + mapping_names[scenario.scenario_ID])
    plt.xlabel('Approach')
    plt.ylabel('Utility')
    plt.tight_layout()
    name = store_path + '\_U'+ scenario.scenario_ID +'.png'
    plt.savefig(name)
    plt.close()








    plt.figure(figsize=(10, 8))
    bp1 = plt.boxplot(data_i[0], positions=np.array(range(len(data[0]))) * 2.0 - 0.6, sym='', widths=0.3, whis=[5, 95])
    bp3 = plt.boxplot(data_i[1], positions=np.array(range(len(data[1]))) * 2.0 - 0.2, sym='', widths=0.3, whis=[5, 95])
    bp6 = plt.boxplot(data_i[2], positions=np.array(range(len(data[2]))) * 2.0 + 0.2, sym='', widths=0.3, whis=[5, 95])
    bp10 = plt.boxplot(data_i[3], positions=np.array(range(len(data[3]))) * 2.0 + 0.6, sym='', widths=0.3, whis=[5, 95])

    set_box_multicolor(bp1, ['red', 'green', 'blue'])
    set_box_multicolor(bp3, ['red', 'green', 'blue'])
    set_box_multicolor(bp6, ['red', 'green', 'blue'])
    set_box_multicolor(bp10, ['red', 'green', 'blue'])

    add_scatter(data_i[0], ['red', 'green', 'blue'], -.6)
    add_scatter(data_i[1], ['red', 'green', 'blue'], -.2)
    add_scatter(data_i[2], ['red', 'green', 'blue'], .2)
    add_scatter(data_i[3], ['red', 'green', 'blue'], .6)

    plt.plot([], c='red', label='Saturating')
    plt.plot([], c='green', label='Peaking')
    plt.plot([], c='blue', label='Weighted')
    plt.legend()

    plt.xticks([-.6, -.2, .2, .6,
                2 - .6, 2 - .2, 2 + .2, 2 + .6,
                4 - .6, 4 - .2, 4 + .2, 4 + .6], ticks)
    plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
    plt.ylim(-.15, 0.15)
    plt.axhline(0, color='black')
    plt.title('Inaccuracy for ' + mapping_names[scenario.scenario_ID])
    plt.xlabel('Approach')
    plt.ylabel('Inaccuracy')
    plt.plot([], c='red', label='Saturating')
    plt.plot([], c='green', label='Peaking')
    plt.plot([], c='blue', label='Weighted')
    plt.tight_layout()
    name = store_path + '\_IA' + scenario.scenario_ID + '.png'
    plt.savefig(name)
    plt.close()

    plt.figure(figsize=(10, 8))
    bp1 = plt.boxplot(data_ai[0], positions=np.array(range(len(data[0]))) * 2.0 - 0.6, sym='', widths=0.3, whis=[5, 95])
    bp3 = plt.boxplot(data_ai[1], positions=np.array(range(len(data[1]))) * 2.0 - 0.2, sym='', widths=0.3, whis=[5, 95])
    bp6 = plt.boxplot(data_ai[2], positions=np.array(range(len(data[2]))) * 2.0 + 0.2, sym='', widths=0.3, whis=[5, 95])
    bp10 = plt.boxplot(data_ai[3], positions=np.array(range(len(data[3]))) * 2.0 + 0.6, sym='', widths=0.3, whis=[5, 95])

    set_box_multicolor(bp1, ['red', 'green', 'blue'])
    set_box_multicolor(bp3, ['red', 'green', 'blue'])
    set_box_multicolor(bp6, ['red', 'green', 'blue'])
    set_box_multicolor(bp10, ['red', 'green', 'blue'])

    add_scatter(data_ai[0], ['red', 'green', 'blue'], -.6)
    add_scatter(data_ai[1], ['red', 'green', 'blue'], -.2)
    add_scatter(data_ai[2], ['red', 'green', 'blue'], .2)
    add_scatter(data_ai[3], ['red', 'green', 'blue'], .6)

    plt.plot([], c='red', label='Saturating')
    plt.plot([], c='green', label='Peaking')
    plt.plot([], c='blue', label='Weighted')
    plt.legend()

    plt.xticks([-.6, -.2, .2, .6,
                2 - .6, 2 - .2, 2 + .2, 2 + .6,
                4 - .6, 4 - .2, 4 + .2, 4 + .6], ticks)
    plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
    plt.ylim(0, 0.15)
    plt.title('Absolute Inaccuracy for ' + mapping_names[scenario.scenario_ID])
    plt.xlabel('Approach')
    plt.ylabel('Absolute Inaccuracy')
    plt.plot([], c='red', label='Saturating')
    plt.plot([], c='green', label='Peaking')
    plt.plot([], c='blue', label='Weighted')
    plt.tight_layout()
    name = store_path + '\_AI' + scenario.scenario_ID + '.png'
    plt.savefig(name)
    plt.close()





    plt.figure(figsize=(10, 8))
    bp1 = plt.boxplot(data_cr[0], positions=np.array(range(len(data[0]))) * 2.0 - 0.6, sym='', widths=0.3, whis=[5, 95])
    bp3 = plt.boxplot(data_cr[1], positions=np.array(range(len(data[1]))) * 2.0 - 0.2, sym='', widths=0.3, whis=[5, 95])
    bp6 = plt.boxplot(data_cr[2], positions=np.array(range(len(data[2]))) * 2.0 + 0.2, sym='', widths=0.3, whis=[5, 95])
    bp10 = plt.boxplot(data_cr[3], positions=np.array(range(len(data[3]))) * 2.0 + 0.6, sym='', widths=0.3, whis=[5, 95])

    set_box_multicolor(bp1, ['red', 'green', 'blue'])
    set_box_multicolor(bp3, ['red', 'green', 'blue'])
    set_box_multicolor(bp6, ['red', 'green', 'blue'])
    set_box_multicolor(bp10, ['red', 'green', 'blue'])

    add_scatter(data_cr[0], ['red', 'green', 'blue'], -.6)
    add_scatter(data_cr[1], ['red', 'green', 'blue'], -.2)
    add_scatter(data_cr[2], ['red', 'green', 'blue'], .2)
    add_scatter(data_cr[3], ['red', 'green', 'blue'], .6)

    plt.plot([], c='red', label='Saturating')
    plt.plot([], c='green', label='Peaking')
    plt.plot([], c='blue', label='Weighted')
    plt.legend()

    plt.xticks([-.6, -.2, .2, .6,
                2 - .6, 2 - .2, 2 + .2, 2 + .6,
                4 - .6, 4 - .2, 4 + .2, 4 + .6], ticks)
    plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
    plt.ylim(-0.02, 0.15)
    plt.title('Average Regret for ' + mapping_names[scenario.scenario_ID])
    plt.xlabel('Approach')
    plt.ylabel('Average Regret')
    plt.plot([], c='red', label='Saturating')
    plt.plot([], c='green', label='Peaking')
    plt.plot([], c='blue', label='Weighted')
    plt.tight_layout()
    name = store_path + '\_AR' + scenario.scenario_ID + '.png'
    plt.savefig(name)
    plt.close()

    plt.figure(figsize=(10, 8))
    bp1 = plt.boxplot(data_cpr[0], positions=np.array(range(len(data[0]))) * 2.0 - 0.6, sym='', widths=0.3, whis=[5, 95])
    bp3 = plt.boxplot(data_cpr[1], positions=np.array(range(len(data[1]))) * 2.0 - 0.2, sym='', widths=0.3, whis=[5, 95])
    bp6 = plt.boxplot(data_cpr[2], positions=np.array(range(len(data[2]))) * 2.0 + 0.2, sym='', widths=0.3, whis=[5, 95])
    bp10 = plt.boxplot(data_cpr[3], positions=np.array(range(len(data[3]))) * 2.0 + 0.6, sym='', widths=0.3, whis=[5, 95])

    set_box_multicolor(bp1, ['red', 'green', 'blue'])
    set_box_multicolor(bp3, ['red', 'green', 'blue'])
    set_box_multicolor(bp6, ['red', 'green', 'blue'])
    set_box_multicolor(bp10, ['red', 'green', 'blue'])

    add_scatter(data_cpr[0], ['red', 'green', 'blue'], -.6)
    add_scatter(data_cpr[1], ['red', 'green', 'blue'], -.2)
    add_scatter(data_cpr[2], ['red', 'green', 'blue'], .2)
    add_scatter(data_cpr[3], ['red', 'green', 'blue'], .6)

    plt.xticks([-.6, -.2, .2, .6,
                    2 - .6, 2 - .2, 2 + .2, 2 + .6,
                    4 - .6, 4 - .2, 4 + .2, 4 + .6], ticks)
    plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
    plt.ylim(-10, 100)
    plt.title('Percentage Average Regret for ' + scenario.scenario_ID)
    plt.xlabel('Approach')
    plt.ylabel('Percentage Average Regret, \n' + scenario.scenario_ID)
    plt.tight_layout()
    plt.plot([], c='red', label='Saturating')
    plt.plot([], c='green', label='Peaking')
    plt.plot([], c='blue', label='Weighted')
    plt.legend()
    plt.tight_layout()
    name = store_path + '\_PAR' + scenario.scenario_ID + '.png'
    plt.savefig(name)
    plt.close()




plt.figure(figsize=(10, 8))
bp1 = plt.boxplot(data_all[0].tolist(), positions=np.array(range(3)) * 2.0 - 0.6, sym='', widths=0.3, whis=[5, 95])
bp3 = plt.boxplot(data_all[1].tolist(), positions=np.array(range(3)) * 2.0 - 0.2, sym='', widths=0.3, whis=[5, 95])
bp6 = plt.boxplot(data_all[2].tolist(), positions=np.array(range(3)) * 2.0 + 0.2, sym='', widths=0.3, whis=[5, 95])
bp10 = plt.boxplot(data_all[3].tolist(), positions=np.array(range(3)) * 2.0 + 0.6, sym='', widths=0.3, whis=[5, 95])

set_box_multicolor(bp1, ['red', 'green', 'blue'])
set_box_multicolor(bp3, ['red', 'green', 'blue'])
set_box_multicolor(bp6, ['red', 'green', 'blue'])
set_box_multicolor(bp10, ['red', 'green', 'blue'])

add_scatter(data_all[0].tolist(),['red', 'green', 'blue'], -.6, alpha = 0.2)
add_scatter(data_all[1].tolist(), ['red', 'green', 'blue'], -.2, alpha = 0.2)
add_scatter(data_all[2].tolist(), ['red', 'green', 'blue'], .2, alpha = 0.2)
add_scatter(data_all[3].tolist(), ['red', 'green', 'blue'], .6, alpha = 0.2)

plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.legend()
plt.xticks([-.6, -.2, .2, .6,
                2 - .6, 2 - .2, 2 + .2, 2 + .6,
                4 - .6, 4 - .2, 4 + .2, 4 + .6], ticks)
plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
plt.ylim(0, 0.15)
plt.title('Simple Regret for all environments')
plt.xlabel('Approach')
plt.ylabel('Simple Regret')
plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.tight_layout()
name = store_path + '\_All_SR.png'
plt.savefig(name)
plt.close()











plt.figure(figsize=(10, 8))
bp1 = plt.boxplot(data_all_i[0].tolist(), positions=np.array(range(3)) * 2.0 - 0.6, sym='', widths=0.3, whis=[5, 95])
bp3 = plt.boxplot(data_all_i[1].tolist(), positions=np.array(range(3)) * 2.0 - 0.2, sym='', widths=0.3, whis=[5, 95])
bp6 = plt.boxplot(data_all_i[2].tolist(), positions=np.array(range(3)) * 2.0 + 0.2, sym='', widths=0.3, whis=[5, 95])
bp10 = plt.boxplot(data_all_i[3].tolist(), positions=np.array(range(3)) * 2.0 + 0.6, sym='', widths=0.3, whis=[5, 95])

set_box_multicolor(bp1, ['red', 'green', 'blue'])
set_box_multicolor(bp3, ['red', 'green', 'blue'])
set_box_multicolor(bp6, ['red', 'green', 'blue'])
set_box_multicolor(bp10, ['red', 'green', 'blue'])

add_scatter(data_all_i[0].tolist(),['red', 'green', 'blue'], -.6, alpha = 0.2)
add_scatter(data_all_i[1].tolist(), ['red', 'green', 'blue'], -.2, alpha = 0.2)
add_scatter(data_all_i[2].tolist(), ['red', 'green', 'blue'], .2, alpha = 0.2)
add_scatter(data_all_i[3].tolist(), ['red', 'green', 'blue'], .6, alpha = 0.2)

plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.legend()
plt.xticks([-.6, -.2, .2, .6,
                2 - .6, 2 - .2, 2 + .2, 2 + .6,
                4 - .6, 4 - .2, 4 + .2, 4 + .6], ticks)
plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
plt.ylim(-0.15, 0.15)
plt.axhline(0, color='black')
plt.title('Inaccuracy for all environments')
plt.xlabel('Approach')
plt.ylabel('Inaccuracy')
plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.tight_layout()
name = store_path + '\_All_IA.png'
plt.savefig(name)
plt.close()





plt.figure(figsize=(10, 8))
bp1 = plt.boxplot(data_all_ai[0].tolist(), positions=np.array(range(3)) * 2.0 - 0.6, sym='', widths=0.3, whis=[5, 95])
bp3 = plt.boxplot(data_all_ai[1].tolist(), positions=np.array(range(3)) * 2.0 - 0.2, sym='', widths=0.3, whis=[5, 95])
bp6 = plt.boxplot(data_all_ai[2].tolist(), positions=np.array(range(3)) * 2.0 + 0.2, sym='', widths=0.3, whis=[5, 95])
bp10 = plt.boxplot(data_all_ai[3].tolist(), positions=np.array(range(3)) * 2.0 + 0.6, sym='', widths=0.3, whis=[5, 95])

set_box_multicolor(bp1, ['red', 'green', 'blue'])
set_box_multicolor(bp3, ['red', 'green', 'blue'])
set_box_multicolor(bp6, ['red', 'green', 'blue'])
set_box_multicolor(bp10, ['red', 'green', 'blue'])

add_scatter(data_all_ai[0].tolist(),['red', 'green', 'blue'], -.6, alpha = 0.2)
add_scatter(data_all_ai[1].tolist(), ['red', 'green', 'blue'], -.2, alpha = 0.2)
add_scatter(data_all_ai[2].tolist(), ['red', 'green', 'blue'], .2, alpha = 0.2)
add_scatter(data_all_ai[3].tolist(), ['red', 'green', 'blue'], .6, alpha = 0.2)

plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.legend()
plt.xticks([-.6, -.2, .2, .6,
                2 - .6, 2 - .2, 2 + .2, 2 + .6,
                4 - .6, 4 - .2, 4 + .2, 4 + .6], ticks)
plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
plt.ylim(0, 0.15)
plt.title('Absolute Inaccuracy for all environments')
plt.xlabel('Approach')
plt.ylabel('Absolute Inaccuracy')
plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.tight_layout()
name = store_path + '\_All_IAAbs.png'
plt.savefig(name)
plt.close()







plt.figure(figsize=(10, 8))
bp1 = plt.boxplot(data_all_cr[0].tolist(), positions=np.array(range(3)) * 2.0 - 0.6, sym='', widths=0.3, whis=[5, 95])
bp3 = plt.boxplot(data_all_cr[1].tolist(), positions=np.array(range(3)) * 2.0 - 0.2, sym='', widths=0.3, whis=[5, 95])
bp6 = plt.boxplot(data_all_cr[2].tolist(), positions=np.array(range(3)) * 2.0 + 0.2, sym='', widths=0.3, whis=[5, 95])
bp10 = plt.boxplot(data_all_cr[3].tolist(), positions=np.array(range(3)) * 2.0 + 0.6, sym='', widths=0.3, whis=[5, 95])

set_box_multicolor(bp1, ['red', 'green', 'blue'])
set_box_multicolor(bp3, ['red', 'green', 'blue'])
set_box_multicolor(bp6, ['red', 'green', 'blue'])
set_box_multicolor(bp10, ['red', 'green', 'blue'])

add_scatter(data_all_cr[0].tolist(),['red', 'green', 'blue'], -.6, alpha = 0.2)
add_scatter(data_all_cr[1].tolist(), ['red', 'green', 'blue'], -.2, alpha = 0.2)
add_scatter(data_all_cr[2].tolist(), ['red', 'green', 'blue'], .2, alpha = 0.2)
add_scatter(data_all_cr[3].tolist(), ['red', 'green', 'blue'], .6, alpha = 0.2)

plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.legend()
plt.xticks([-.6, -.2, .2, .6,
                2 - .6, 2 - .2, 2 + .2, 2 + .6,
                4 - .6, 4 - .2, 4 + .2, 4 + .6], ticks)
plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
plt.ylim(-0.02, 0.15)
plt.title('Average Regret for all environments')
plt.xlabel('Approach')
plt.ylabel('Average Regret')
plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.tight_layout()
name = store_path + '\_All_CR.png'
plt.savefig(name)
plt.close()





plt.figure(figsize=(10, 8))
bp1 = plt.boxplot(data_all_cpr[0].tolist(), positions=np.array(range(3)) * 2.0 - 0.6, sym='', widths=0.3, whis=[5, 95])
bp3 = plt.boxplot(data_all_cpr[1].tolist(), positions=np.array(range(3)) * 2.0 - 0.2, sym='', widths=0.3, whis=[5, 95])
bp6 = plt.boxplot(data_all_cpr[2].tolist(), positions=np.array(range(3)) * 2.0 + 0.2, sym='', widths=0.3, whis=[5, 95])
bp10 = plt.boxplot(data_all_cpr[3].tolist(), positions=np.array(range(3)) * 2.0 + 0.6, sym='', widths=0.3, whis=[5, 95])

set_box_multicolor(bp1, ['red', 'green', 'blue'])
set_box_multicolor(bp3, ['red', 'green', 'blue'])
set_box_multicolor(bp6, ['red', 'green', 'blue'])
set_box_multicolor(bp10, ['red', 'green', 'blue'])

add_scatter(data_all_cpr[0].tolist(),['red', 'green', 'blue'], -.6, alpha = 0.2)
add_scatter(data_all_cpr[1].tolist(), ['red', 'green', 'blue'], -.2, alpha = 0.2)
add_scatter(data_all_cpr[2].tolist(), ['red', 'green', 'blue'], .2, alpha = 0.2)
add_scatter(data_all_cpr[3].tolist(), ['red', 'green', 'blue'], .6, alpha = 0.2)

plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.legend()
plt.xticks([-.6, -.2, .2, .6,
                2 - .6, 2 - .2, 2 + .2, 2 + .6,
                4 - .6, 4 - .2, 4 + .2, 4 + .6], ticks)
plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
plt.ylim(-10, 110)
plt.title('Percentage Average Regret for all environments')
plt.xlabel('Approach')
plt.ylabel('Percentage Average Regret')
plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.tight_layout()
name = store_path + '\_All_PCR.png'
plt.savefig(name)
plt.close()





plt.figure(figsize=(10, 8))
bp1 = plt.boxplot(data_all_cent[0].tolist(), positions=np.array(range(3)) * 2.0 - 0.6, sym='', widths=0.3, whis=[5, 95])
bp3 = plt.boxplot(data_all_cent[1].tolist(), positions=np.array(range(3)) * 2.0 - 0.2, sym='', widths=0.3, whis=[5, 95])
bp6 = plt.boxplot(data_all_cent[2].tolist(), positions=np.array(range(3)) * 2.0 + 0.2, sym='', widths=0.3, whis=[5, 95])
bp10 = plt.boxplot(data_all_cent[3].tolist(), positions=np.array(range(3)) * 2.0 + 0.6, sym='', widths=0.3, whis=[5, 95])

set_box_multicolor(bp1, ['red', 'green', 'blue'])
set_box_multicolor(bp3, ['red', 'green', 'blue'])
set_box_multicolor(bp6, ['red', 'green', 'blue'])
set_box_multicolor(bp10, ['red', 'green', 'blue'])

add_scatter(data_all_cent[0].tolist(),['red', 'green', 'blue'], -.6, alpha = 0.2)
add_scatter(data_all_cent[1].tolist(), ['red', 'green', 'blue'], -.2, alpha = 0.2)
add_scatter(data_all_cent[2].tolist(), ['red', 'green', 'blue'], .2, alpha = 0.2)
add_scatter(data_all_cent[3].tolist(), ['red', 'green', 'blue'], .6, alpha = 0.2)

plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.legend()
plt.xticks([-.6, -.2, .2, .6,
                2 - .6, 2 - .2, 2 + .2, 2 + .6,
                4 - .6, 4 - .2, 4 + .2, 4 + .6], ticks)
plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
plt.ylim(0,100)
plt.title('Percentile Utility for all environments')
plt.xlabel('Approach')
plt.ylabel('Percentile Utility')
plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.tight_layout()
name = store_path + '\_All_CENT.png'
plt.savefig(name)
plt.close()















plt.figure(figsize=(10, 8))
bp1 = plt.boxplot(data_all_p[0].tolist(), positions=np.array(range(3)) * 2.0 - 0.6, sym='', widths=0.3, whis=[5, 95])
bp3 = plt.boxplot(data_all_p[1].tolist(), positions=np.array(range(3)) * 2.0 - 0.2, sym='', widths=0.3, whis=[5, 95])
bp6 = plt.boxplot(data_all_p[2].tolist(), positions=np.array(range(3)) * 2.0 + 0.2, sym='', widths=0.3, whis=[5, 95])
bp10 = plt.boxplot(data_all_p[3].tolist(), positions=np.array(range(3)) * 2.0 + 0.6, sym='', widths=0.3, whis=[5, 95])

set_box_multicolor(bp1, ['red', 'green', 'blue'])
set_box_multicolor(bp3, ['red', 'green', 'blue'])
set_box_multicolor(bp6, ['red', 'green', 'blue'])
set_box_multicolor(bp10, ['red', 'green', 'blue'])

add_scatter(data_all_p[0].tolist(),['red', 'green', 'blue'], -.6, alpha = 0.2)
add_scatter(data_all_p[1].tolist(), ['red', 'green', 'blue'], -.2, alpha = 0.2)
add_scatter(data_all_p[2].tolist(), ['red', 'green', 'blue'], .2, alpha = 0.2)
add_scatter(data_all_p[3].tolist(), ['red', 'green', 'blue'], .6, alpha = 0.2)

plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.legend()
plt.xticks([-.6, -.2, .2, .6,
                2 - .6, 2 - .2, 2 + .2, 2 + .6,
                4 - .6, 4 - .2, 4 + .2, 4 + .6], ticks)
plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
plt.ylim(0, 100)
plt.title('Percentage Simple Regret for all environments')
plt.xlabel('Approach')
plt.ylabel('Percentage Simple Regret')
plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.tight_layout()
name = store_path + '\_All_PSR.png'
plt.savefig(name)
plt.close()






plt.figure(figsize=(10, 8))
bp1 = plt.boxplot(data_all_u[0].tolist(), positions=np.array(range(3)) * 2.0 - 0.6, sym='', widths=0.3, whis=[5, 95])
bp3 = plt.boxplot(data_all_u[1].tolist(), positions=np.array(range(3)) * 2.0 - 0.2, sym='', widths=0.3, whis=[5, 95])
bp6 = plt.boxplot(data_all_u[2].tolist(), positions=np.array(range(3)) * 2.0 + 0.2, sym='', widths=0.3, whis=[5, 95])
bp10 = plt.boxplot(data_all_u[3].tolist(), positions=np.array(range(3)) * 2.0 + 0.6, sym='', widths=0.3, whis=[5, 95])

set_box_multicolor(bp1, ['red', 'green', 'blue'])
set_box_multicolor(bp3, ['red', 'green', 'blue'])
set_box_multicolor(bp6, ['red', 'green', 'blue'])
set_box_multicolor(bp10, ['red', 'green', 'blue'])

add_scatter(data_all_u[0].tolist(),['red', 'green', 'blue'], -.6, alpha = 0.2)
add_scatter(data_all_u[1].tolist(), ['red', 'green', 'blue'], -.2, alpha = 0.2)
add_scatter(data_all_u[2].tolist(), ['red', 'green', 'blue'], .2, alpha = 0.2)
add_scatter(data_all_u[3].tolist(), ['red', 'green', 'blue'], .6, alpha = 0.2)

plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.legend()
plt.xticks([-.6, -.2, .2, .6,
                2 - .6, 2 - .2, 2 + .2, 2 + .6,
                4 - .6, 4 - .2, 4 + .2, 4 + .6], ticks)
plt.xlim(-1, ((len(ticks) * 2) / 4) - 1)
plt.ylim(-.15, .15)
plt.title('Utility for all environments')
plt.xlabel('Approach')
plt.ylabel('Utility')
plt.plot([], c='red', label='Saturating')
plt.plot([], c='green', label='Peaking')
plt.plot([], c='blue', label='Weighted')
plt.tight_layout()
name = store_path + '\_All_U.png'
plt.savefig(name)
plt.close()





