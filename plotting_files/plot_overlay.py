"""
Plots the predicted optimal dose and response for every scenario and approach pairing
"""

import sqlite3
from scenarios_and_approaches import *
import pathlib
data_path = str(pathlib.Path(__file__).parent.parent.resolve()) + '\Storing_Database.db'
store_path = str(pathlib.Path(__file__).parent.parent.resolve()) +'\overlaid_clinical_trial_plots'


conn = sqlite3.connect(data_path)
c = conn.cursor()
c.execute("SELECT rowid,* FROM experiments")
experiments1 = c.fetchall()
experiments1 = np.asarray(experiments1)
conn.commit() 
conn.close() 

saturating_approaches = ['Saturating_Uniform_10','Saturating_Uniform_30','Saturating_Uniform_60','Saturating_Uniform_100','Saturating_3_Stage_SoftMax','SaturatingCRMExploit','SaturatingCRMBalanced_2'
                  ]
peak_approaches = ['Peaking_Uniform_10','Peaking_Uniform_30','Peaking_Uniform_60','Peaking_Uniform_100','Peak_3_Stage_SoftMax','PeakCRMExploit','PeakCRMBalanced_2']
weight_approaches = ['Weight_Uniform_10','Weight_Uniform_30','Weight_Uniform_60','Weight_Uniform_100',
                     'Weight_3_Stage_SoftMax_10','Weight_3_Stage_SoftMax','Weight_3_Stage_SoftMax_60','Weight_3_Stage_SoftMax_100',
                     'WeightCRMExploit_10','WeightCRMExploit']
approaches = [saturating_approaches, peak_approaches, weight_approaches]
scenarios = [scenarioS1, scenarioS2, scenarioS3, scenarioS4, scenarioS5,
             scenarioP1, scenarioP2, scenarioP3, scenarioP4, scenarioP5,
             scenarioX1, scenarioX2, scenarioX3, scenarioX4]

#
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

a_mapping_names = {'Saturating_Uniform_10':'Saturating Uniform, 10',
                   'Saturating_Uniform_30':'Saturating Uniform, 30',
                   'Saturating_Uniform_60':'Saturating Uniform, 60',
                   'Saturating_Uniform_100':'Saturating Uniform, 100',
                   'Saturating_3_Stage_SoftMax':'Saturating, Softmax 3 Stage',
                   'SaturatingCRMExploit':'Saturating, Fully Continual, Standard',
                   'SaturatingCRMBalanced_2':'Saturating, Fully Continual, Balanced',
                   'Peaking_Uniform_10':'Peaking Uniform, 10',
                   'Peaking_Uniform_30':'Peaking Uniform, 30',
                   'Peaking_Uniform_60':'Peaking Uniform, 60',
                   'Peaking_Uniform_100':'Peaking Uniform, 100',
                   'Peak_3_Stage_SoftMax':'Peaking, Softmax 3 Stage',
                   'PeakCRMExploit':'Peaking, Fully Continual, Standard',
                   'PeakCRMBalanced_2':'Peaking, Fully Continual, Balanced',
                   'Weight_Uniform_10':'Weighted Uniform, 10',
                   'Weight_Uniform_30':'Weighted Uniform, 30',
                   'Weight_Uniform_60':'Weighted Uniform, 60',
                   'Weight_Uniform_100':'Weighted Uniform, 100',
                   'Weight_3_Stage_SoftMax_10':'Weighted, Softmax 3 Stage 10',
                   'Weight_3_Stage_SoftMax':'Weighted, Softmax 3 Stage 30',
                   'Weight_3_Stage_SoftMax_60':'Weighted, Softmax 3 Stage 60',
                   'Weight_3_Stage_SoftMax_100':'Weighted, Softmax 3 Stage 100',
                   'WeightCRMExploit_10':'Weighted, Fully Continual, Standard 10',
                   'WeightCRMExploit':'Weighted, Fully Continual, Standard 30',
                   'WeightCRMExploit_60':'Weighted, Fully Continual, Standard 60',
                   'WeightCRMExploit_100':'Weighted, Fully Continual, Standard 100',
                   'WeightCRMBalanced_2_10':'Weighted, Fully Continual, Balanced 10',
                   'WeightCRMBalanced_2':'Weighted, Fully Continual, Balanced 30',
                   'WeightCRMBalanced_2_60':'Weighted, Fully Continual, Balanced 60',
                   'WeightCRMBalanced_2_100':'Weighted, Fully Continual, Balanced 100',
                   'Saturating_3_Dose_M_10':'Saturating, Medium 3 Dose, 10',
                   'Saturating_3_Dose_M_30':'Saturating, Medium 3 Dose, 30',
                   'Saturating_3_Dose_M_60':'Saturating, Medium 3 Dose, 60',
                  'Saturating_3_Dose_M_100':'Saturating, Medium 3 Dose, 100',
                  'Saturating_3_Dose_H_10':'Saturating, High 3 Dose, 10',
                  'Saturating_3_Dose_H_30':'Saturating, High 3 Dose, 30',
                  'Saturating_3_Dose_H_60':'Saturating, High 3 Dose, 60',
                  'Saturating_3_Dose_H_100':'Saturating, High 3 Dose, 100',
                   'Peak_3_Dose_M_10':'Peaking, Medium 3 Dose, 10',
                   'Peak_3_Dose_M_30':'Peaking, Medium 3 Dose, 30',
                   'Peak_3_Dose_M_60':'Peaking, Medium 3 Dose, 60',
                  'Peak_3_Dose_M_100':'Peaking, Medium 3 Dose, 100',
                  'Peak_3_Dose_H_10':'Peaking, High 3 Dose, 10',
                  'Peak_3_Dose_H_30':'Peaking, High 3 Dose, 30',
                  'Peak_3_Dose_H_60':'Peaking, High 3 Dose, 60',
                  'Peak_3_Dose_H_100':'Peaking, High 3 Dose, 100',
}


query_doses = np.linspace(0,10,101)
for scenario in scenarios:
    print(mapping_names[scenario.scenario_ID])
    scenario_label = scenario.scenario_ID
    _, maximum_utility = scenario.calculate_utility_max()
    utilities = scenario.dose_utility(query_doses)
    rows = np.where(experiments1[:, 1] == scenario_label)
    experiments2 = experiments1[rows]

    for index_shape, approach_shape in enumerate(approaches):
        print(index_shape, approach_shape)

        for index_number, approach in enumerate(approach_shape):
            print(index_number, approach)
            rows = np.where(experiments2[:, 2] == approach)
            experiments = experiments2[rows]
            predicted_optimal_dose = experiments[:, 3].astype(np.float)
            predicted_optimal_response = experiments[:, 4].astype(np.float)
            mean_d, mean_r = np.mean(predicted_optimal_dose), np.mean(predicted_optimal_response)

            scenario.plot_utility()
            plt.scatter(predicted_optimal_dose, predicted_optimal_response, c='Red')
            plt.scatter(mean_d, mean_r, c='orange')
            plt.title('Predicted optimal doses for \n' + mapping_names[scenario.scenario_ID] + ': ' + a_mapping_names[approach])
            plt.xlabel('Dose')
            plt.ylabel('Utility')
            plt.ylim(-0.1,0.2)
            plt.legend(['True Utility', 'Predicted Optimal Doses', 'Mean of Predicted Optimal Doses'])
            name = store_path + '\scen' + scenario.scenario_ID + '_app' + approach + '.png'
            plt.savefig(name)
            plt.close()

