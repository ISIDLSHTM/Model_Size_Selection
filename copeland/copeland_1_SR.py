'''
Runs the Copeland analysis for Simple Regret
'''

import sqlite3
from scenarios_and_approaches import *
import pathlib

data_path = str(pathlib.Path(__file__).parent.parent.resolve()) + '\Storing_Database.db'
store_path = str(pathlib.Path(__file__).parent.parent.resolve()) +'\copeland\copeland_tables'


conn = sqlite3.connect(data_path)
c = conn.cursor()

c.execute("SELECT rowid,* FROM experiments")
experiments1 = c.fetchall()
experiments1 = np.asarray(experiments1)
conn.commit() 
conn.close()

def copeland(c1,c2):
    if c1 > c2:
        return 1 , 0
    elif c1 < c2:
        return 0 , 1
    return 0.5, 0.5


scenarios = [scenarioS1, scenarioS2, scenarioS3, scenarioS4, scenarioS5,
             scenarioP1, scenarioP2, scenarioP3, scenarioP4, scenarioP5,
             scenarioX1, scenarioX2, scenarioX3, scenarioX4]

limit_to_approaches = ['Peaking_Uniform_30',
                       'Peak_3_Stage_SoftMax','PeakCRMExploit','PeakCRMBalanced_2',
                       'Saturating_Uniform_30',
                       'Saturating_3_Stage_SoftMax', 'SaturatingCRMExploit', 'SaturatingCRMBalanced_2',
                       'Weight_Uniform_30',
                       'Weight_3_Stage_SoftMax', 'WeightCRMExploit', 'WeightCRMBalanced_2',
                       ]
experiments1 = experiments1[np.isin(experiments1[:, 2], limit_to_approaches)]
approaches = np.unique(experiments1[:, 2])

copeland_array = []
a = []
sums = []
p = []
for scenario in scenarios:
    copeland_array = []
    print(scenario)
    scenario_label = scenario.scenario_ID
    rows = np.where(experiments1[:, 1] == scenario_label)
    experiments = experiments1[rows]
    experiments = experiments[:, (1,2,3)]
    suggested_doses = experiments[:,2].astype(np.float)
    actual_utilty = scenario.dose_utility(suggested_doses)
    approach_utility = np.column_stack((experiments[:,1],actual_utilty))

    number_of_rows = np.shape(approach_utility)[0]
    for index1, row1 in enumerate(approach_utility):
        approach1, u1 = row1
        comparison_rows = approach_utility[index1+1:, :]
        for index2, row2 in enumerate(comparison_rows):
            approach2, u2 = row2
            if not approach1 == approach2:
                c1, c2 = copeland(u1,u2)
                copeland_array.append([approach1, c1, 1])
                copeland_array.append([approach2, c2, 1])
    copeland_array = np.asarray(copeland_array).reshape((-1, 3))

    _a = []
    _p = []
    _sums = []
    for approach in approaches:
        rows = np.where(copeland_array[:, 0] == approach)
        c_array = copeland_array[rows]
        c_val = np.sum(c_array[:, 1].astype(np.float))
        c_test = np.sum(c_array[:, 2].astype(np.float))
        a.append(approach)
        sums.append(c_val)
        p.append(c_test)
        _a.append(approach)
        _sums.append(c_val)
        _p.append(c_test)
    _a = np.asarray(_a)
    _sums = np.asarray(_sums)
    _p = np.asarray(_p)
    a2 = []
    sums2 = []
    p2 = []
    for approach in approaches:
        _rows = np.where(_a == approach)
        a2.append(_a[_rows][0])
        sums2.append(_sums[_rows][0])
        p2.append(_sums[_rows][0]/_p[_rows][0])
    array = np.column_stack((a2, sums2, p2))

    sorted_array = array[np.argsort(array[:, 1].astype(np.float))]
    sorted_array = np.flipud(sorted_array)
    sorted_array = np.delete(sorted_array, 1, axis=1)

    with open(store_path +"\copeland_by_scenario\Copeland_SR"+ scenario.scenario_ID +".txt", "w") as f:
        f.write("\n".join(" ".join(map(str, x)) for x in sorted_array))

a = np.asarray(a)
sums = np.asarray(sums)
p = np.asarray(p)
a2 = []
sums2 = []
p2 = []
for approach in approaches:
    rows = np.where(a == approach)
    c_array = a[rows]
    c_val = np.sum(sums[rows].astype(np.float))
    c_test = np.sum(p[rows].astype(np.float))
    a2.append(approach)
    sums2.append(c_val)
    p2.append((c_val/c_test))



array = np.column_stack((a2, sums2, p2))

sorted_array = array[np.argsort(array[:, 1].astype(np.float))]


with open(store_path +"\copeland_SR.txt", "w") as f:
    f.write("\n".join(" ".join(map(str, x)) for x in sorted_array))

