'''
Creates reformatted Copeland Data for publication
'''

import numpy as np
import pandas as pd

with open("copeland_tables/Copeland_SR.txt") as f:
    lines = [line.rstrip().split() for line in f]
frame = np.asarray(lines)


dataset = pd.DataFrame({'Approach': frame[:, 0],
                        'Rank': frame[:, 1],
                        'Score': frame[:, 2].astype(np.float)})

dataset.Score = dataset.Score.round(3)

for index in dataset.index:
    if dataset.loc[index, 'Approach'] == 'Saturating_Uniform_10':
        dataset.loc[index, 'Approach'] = 'Saturating Uniform, 10'
    if dataset.loc[index, 'Approach'] == 'Saturating_Uniform_30':
        dataset.loc[index, 'Approach'] = 'Saturating Uniform, 30'
    if dataset.loc[index, 'Approach'] == 'Saturating_Uniform_60':
        dataset.loc[index, 'Approach'] = 'Saturating Uniform, 60'
    if dataset.loc[index, 'Approach'] == 'Saturating_Uniform_100':
        dataset.loc[index, 'Approach'] = 'Saturating Uniform, 100'
    if dataset.loc[index, 'Approach'] == 'Peaking_Uniform_10':
        dataset.loc[index, 'Approach'] = 'Peaking Uniform, 10'
    if dataset.loc[index, 'Approach'] == 'Peaking_Uniform_30':
        dataset.loc[index, 'Approach'] = 'Peaking Uniform, 30'
    if dataset.loc[index, 'Approach'] == 'Peaking_Uniform_60':
        dataset.loc[index, 'Approach'] = 'Peaking Uniform, 60'
    if dataset.loc[index, 'Approach'] == 'Peaking_Uniform_100':
        dataset.loc[index, 'Approach'] = 'Peaking Uniform, 100'
    if dataset.loc[index, 'Approach'] == 'Weight_Uniform_10':
        dataset.loc[index, 'Approach'] = 'Weighted Uniform, 10'
    if dataset.loc[index, 'Approach'] == 'Weight_Uniform_30':
        dataset.loc[index, 'Approach'] = 'Weighted Uniform, 30'
    if dataset.loc[index, 'Approach'] == 'Weight_Uniform_60':
        dataset.loc[index, 'Approach'] = 'Weighted Uniform, 60'
    if dataset.loc[index, 'Approach'] == 'Weight_Uniform_100':
        dataset.loc[index, 'Approach'] = 'Weighted Uniform, 100'
    if dataset.loc[index, 'Approach'] == 'SaturatingCRMExploit':
        dataset.loc[index, 'Approach'] = 'Saturating CRM, Exploit'
    if dataset.loc[index, 'Approach'] == 'PeakCRMExploit':
        dataset.loc[index, 'Approach'] = 'Peaking CRM, Exploit'
    if dataset.loc[index, 'Approach'] == 'WeightCRMExploit':
        dataset.loc[index, 'Approach'] = 'Weighted CRM, Exploit'
    if dataset.loc[index, 'Approach'] == 'SaturatingCRMBalanced_2':
        dataset.loc[index, 'Approach'] = 'Saturating CRM, Balanced'
    if dataset.loc[index, 'Approach'] == 'PeakCRMBalanced_2':
        dataset.loc[index, 'Approach'] = 'Peaking CRM, Balanced'
    if dataset.loc[index, 'Approach'] == 'WeightCRMBalanced_2':
        dataset.loc[index, 'Approach'] = 'Weighted CRM, Balanced'
    if dataset.loc[index, 'Approach'] == 'Saturating_3_Stage_SoftMax':
        dataset.loc[index, 'Approach'] = 'Saturating, Softmax 3 Step'
    if dataset.loc[index, 'Approach'] == 'Peak_3_Stage_SoftMax':
        dataset.loc[index, 'Approach'] = 'Peaking, Softmax 3 Step'
    if dataset.loc[index, 'Approach'] == 'Weight_3_Stage_SoftMax':
        dataset.loc[index, 'Approach'] = 'Weighted, Softmax 3 Step'
dataset['Rank'] = dataset.index + 1
dataset = dataset.iloc[::-1]
dataset = dataset[['Rank', 'Approach','Score']]
dataset['Rank'] = dataset['Rank'].values[::-1]
dataset.to_csv('copeland_tables/Copeland_Frame_SR.csv', index = False)




with open("copeland_tables/Copeland_IA.txt") as f:
    lines = [line.rstrip().split() for line in f]
frame = np.asarray(lines)


dataset = pd.DataFrame({'Approach': frame[:, 0],
                        'Rank': frame[:, 1],
                        'Score': frame[:, 2].astype(np.float)})

dataset.Score = dataset.Score.round(3)

for index in dataset.index:
    if dataset.loc[index, 'Approach'] == 'Saturating_Uniform_10':
        dataset.loc[index, 'Approach'] = 'Saturating Uniform, 10'
    if dataset.loc[index, 'Approach'] == 'Saturating_Uniform_30':
        dataset.loc[index, 'Approach'] = 'Saturating Uniform, 30'
    if dataset.loc[index, 'Approach'] == 'Saturating_Uniform_60':
        dataset.loc[index, 'Approach'] = 'Saturating Uniform, 60'
    if dataset.loc[index, 'Approach'] == 'Saturating_Uniform_100':
        dataset.loc[index, 'Approach'] = 'Saturating Uniform, 100'
    if dataset.loc[index, 'Approach'] == 'Peaking_Uniform_10':
        dataset.loc[index, 'Approach'] = 'Peaking Uniform, 10'
    if dataset.loc[index, 'Approach'] == 'Peaking_Uniform_30':
        dataset.loc[index, 'Approach'] = 'Peaking Uniform, 30'
    if dataset.loc[index, 'Approach'] == 'Peaking_Uniform_60':
        dataset.loc[index, 'Approach'] = 'Peaking Uniform, 60'
    if dataset.loc[index, 'Approach'] == 'Peaking_Uniform_100':
        dataset.loc[index, 'Approach'] = 'Peaking Uniform, 100'
    if dataset.loc[index, 'Approach'] == 'Weight_Uniform_10':
        dataset.loc[index, 'Approach'] = 'Weighted Uniform, 10'
    if dataset.loc[index, 'Approach'] == 'Weight_Uniform_30':
        dataset.loc[index, 'Approach'] = 'Weighted Uniform, 30'
    if dataset.loc[index, 'Approach'] == 'Weight_Uniform_60':
        dataset.loc[index, 'Approach'] = 'Weighted Uniform, 60'
    if dataset.loc[index, 'Approach'] == 'Weight_Uniform_100':
        dataset.loc[index, 'Approach'] = 'Weighted Uniform, 100'
    if dataset.loc[index, 'Approach'] == 'SaturatingCRMExploit':
        dataset.loc[index, 'Approach'] = 'Saturating CRM, Exploit'
    if dataset.loc[index, 'Approach'] == 'PeakCRMExploit':
        dataset.loc[index, 'Approach'] = 'Peaking CRM, Exploit'
    if dataset.loc[index, 'Approach'] == 'WeightCRMExploit':
        dataset.loc[index, 'Approach'] = 'Weighted CRM, Exploit'
    if dataset.loc[index, 'Approach'] == 'SaturatingCRMBalanced_2':
        dataset.loc[index, 'Approach'] = 'Saturating CRM, Balanced'
    if dataset.loc[index, 'Approach'] == 'PeakCRMBalanced_2':
        dataset.loc[index, 'Approach'] = 'Peaking CRM, Balanced'
    if dataset.loc[index, 'Approach'] == 'WeightCRMBalanced_2':
        dataset.loc[index, 'Approach'] = 'Weighted CRM, Balanced'
    if dataset.loc[index, 'Approach'] == 'Saturating_3_Stage_SoftMax':
        dataset.loc[index, 'Approach'] = 'Saturating, Softmax 3 Step'
    if dataset.loc[index, 'Approach'] == 'Peak_3_Stage_SoftMax':
        dataset.loc[index, 'Approach'] = 'Peaking, Softmax 3 Step'
    if dataset.loc[index, 'Approach'] == 'Weight_3_Stage_SoftMax':
        dataset.loc[index, 'Approach'] = 'Weighted, Softmax 3 Step'
dataset['Rank'] = dataset.index + 1
dataset = dataset.iloc[::-1]
dataset = dataset[['Rank', 'Approach','Score']]
dataset['Rank'] = dataset['Rank'].values[::-1]
dataset.to_csv('copeland_tables/Copeland_Frame_IA.csv', index = False)



with open("copeland_tables/Copeland_R.txt") as f:
    lines = [line.rstrip().split() for line in f]
frame = np.asarray(lines)


dataset = pd.DataFrame({'Approach': frame[:, 0],
                        'Rank': frame[:, 1],
                        'Score': frame[:, 2].astype(np.float)})

dataset.Score = dataset.Score.round(3)

for index in dataset.index:
    if dataset.loc[index, 'Approach'] == 'Saturating_Uniform_10':
        dataset.loc[index, 'Approach'] = 'Saturating Uniform, 10'
    if dataset.loc[index, 'Approach'] == 'Saturating_Uniform_30':
        dataset.loc[index, 'Approach'] = 'Saturating Uniform, 30'
    if dataset.loc[index, 'Approach'] == 'Saturating_Uniform_60':
        dataset.loc[index, 'Approach'] = 'Saturating Uniform, 60'
    if dataset.loc[index, 'Approach'] == 'Saturating_Uniform_100':
        dataset.loc[index, 'Approach'] = 'Saturating Uniform, 100'
    if dataset.loc[index, 'Approach'] == 'Peaking_Uniform_10':
        dataset.loc[index, 'Approach'] = 'Peaking Uniform, 10'
    if dataset.loc[index, 'Approach'] == 'Peaking_Uniform_30':
        dataset.loc[index, 'Approach'] = 'Peaking Uniform, 30'
    if dataset.loc[index, 'Approach'] == 'Peaking_Uniform_60':
        dataset.loc[index, 'Approach'] = 'Peaking Uniform, 60'
    if dataset.loc[index, 'Approach'] == 'Peaking_Uniform_100':
        dataset.loc[index, 'Approach'] = 'Peaking Uniform, 100'
    if dataset.loc[index, 'Approach'] == 'Weight_Uniform_10':
        dataset.loc[index, 'Approach'] = 'Weighted Uniform, 10'
    if dataset.loc[index, 'Approach'] == 'Weight_Uniform_30':
        dataset.loc[index, 'Approach'] = 'Weighted Uniform, 30'
    if dataset.loc[index, 'Approach'] == 'Weight_Uniform_60':
        dataset.loc[index, 'Approach'] = 'Weighted Uniform, 60'
    if dataset.loc[index, 'Approach'] == 'Weight_Uniform_100':
        dataset.loc[index, 'Approach'] = 'Weighted Uniform, 100'
    if dataset.loc[index, 'Approach'] == 'SaturatingCRMExploit':
        dataset.loc[index, 'Approach'] = 'Saturating CRM, Exploit'
    if dataset.loc[index, 'Approach'] == 'PeakCRMExploit':
        dataset.loc[index, 'Approach'] = 'Peaking CRM, Exploit'
    if dataset.loc[index, 'Approach'] == 'WeightCRMExploit':
        dataset.loc[index, 'Approach'] = 'Weighted CRM, Exploit'
    if dataset.loc[index, 'Approach'] == 'SaturatingCRMBalanced_2':
        dataset.loc[index, 'Approach'] = 'Saturating CRM, Balanced'
    if dataset.loc[index, 'Approach'] == 'PeakCRMBalanced_2':
        dataset.loc[index, 'Approach'] = 'Peaking CRM, Balanced'
    if dataset.loc[index, 'Approach'] == 'WeightCRMBalanced_2':
        dataset.loc[index, 'Approach'] = 'Weighted CRM, Balanced'
    if dataset.loc[index, 'Approach'] == 'Saturating_3_Stage_SoftMax':
        dataset.loc[index, 'Approach'] = 'Saturating, Softmax 3 Step'
    if dataset.loc[index, 'Approach'] == 'Peak_3_Stage_SoftMax':
        dataset.loc[index, 'Approach'] = 'Peaking, Softmax 3 Step'
    if dataset.loc[index, 'Approach'] == 'Weight_3_Stage_SoftMax':
        dataset.loc[index, 'Approach'] = 'Weighted, Softmax 3 Step'
dataset['Rank'] = dataset.index + 1
dataset = dataset.iloc[::-1]
dataset = dataset[['Rank', 'Approach','Score']]
dataset['Rank'] = dataset['Rank'].values[::-1]
dataset.to_csv('copeland_tables/Copeland_Frame_R.csv', index = False)

