"""
Iterates over all pairings of approaches in approach_list and scenario_plots in scenario_list
"""

from important_functions.simulate_trial_functions import *
from scenarios_and_approaches import *
from important_functions.sql_functions import *
import sqlite3
from time import time
from config import Config

config = Config()
ta = time()

approach_list = [
    approachSaturating_Uniform_10,
    approachSaturating_Uniform_30,
    approachSaturating_Uniform_60,
    approachSaturating_Uniform_100,
    approachPeak_Uniform_10,
    approachPeak_Uniform_30,
    approachPeak_Uniform_60,
    approachPeak_Uniform_100,
    approachWeight_Uniform_10,
    approachWeight_Uniform_30,
    approachWeight_Uniform_60,
    approachWeight_Uniform_100,
    approach3_Step_saturating_SoftMax,
    approach3_Step_Peak_SoftMax,
    approach3_Step_Weight_SoftMax,
    approachSaturating_CRM_Exploit,
    approachPeak_CRM_Exploit,
    approachWeight_CRM_Exploit,
    approachSaturating_CRM_Balanced_2,
    approachPeak_CRM_Balanced_2,
    approachWeight_CRM_Balanced_2,

    # The below were not included in the formal analysis of the paper,
    # but can be simulated to show that the benefit of exploration increases for n = 60, 100
    # Run plot_objective3.py
    approach3_Step_Weight_SoftMax_10,
    approach3_Step_Weight_SoftMax_60,
    approach3_Step_Weight_SoftMax_100,
    approachWeight_CRM_Exploit_10,
    approachWeight_CRM_Exploit_60,
    approachWeight_CRM_Exploit_100,
    approachWeight_CRM_Balanced_2_10,
    approachWeight_CRM_Balanced_2_60,
    approachWeight_CRM_Balanced_2_100
                         ]

scenario_list = [
    scenarioS1,
    scenarioS2,
    scenarioS3,
    scenarioS4,
    scenarioS5,
    scenarioP1,
    scenarioP2,
    scenarioP3,
    scenarioP4,
    scenarioP5,
    scenarioX1,
    scenarioX2,
    scenarioX3,
    scenarioX4
                         ]

conn = sqlite3.connect('Storing_Database.db')

experiment_ID = find_most_recent_experiment(conn)
if experiment_ID is None:
    experiment_ID = 1
else:
    experiment_ID = experiment_ID + 1

for index, approach in enumerate(approach_list):
    print(approach.Approach_ID, 'This is approach:', index+1)
    for scenario in scenario_list:
        print(scenario.scenario_ID)
        t0 = time()
        for i in range(config.Iterations_Per_Pair):
            np.random.seed(experiment_ID)  # included to ensure reproducibility
            result = simulatetrial(scenario, approach)

            suggested, response, theoretical, is_explored, actual_data, params, total_utility, average_utility = result

            experiment = Experiment(scenario.scenario_ID, approach.Approach_ID,
                                    suggested, response, is_explored, total_utility, average_utility)

            dose, eff, _, tox = actual_data

            insert_experiment(experiment, conn)

            experiment_ID += 1
        t1 = time()
        print(t0, t1, t1 - t0)


conn.close()


tb = time()
print(ta, tb, ta - tb)
