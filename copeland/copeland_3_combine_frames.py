'''
Combines the Copeland metrics to find the combine rank and scores
'''

import numpy as np
import pandas as pd
import sys
import pathlib
data_path = str(pathlib.Path(__file__).parent.parent.resolve()) + '\copeland\copeland_tables'
store_path = str(pathlib.Path(__file__).parent.parent.resolve()) +'\copeland\copeland_tables'

vals = np.array((range(1, 13)))[::-1]

with open(data_path+ "\Copeland_SR.txt") as f:
    lines = [line.rstrip().split() for line in f]
SR = np.asarray(lines)
SR = np.insert(SR,0, vals, 1)

with open(data_path+ "\Copeland_IA.txt") as f:
    lines = [line.rstrip().split() for line in f]
IA = np.asarray(lines)
IA = np.insert(IA,0, vals, 1)

with open(data_path+ "\Copeland_R.txt") as f:
    lines = [line.rstrip().split() for line in f]
R = np.asarray(lines)
R = np.insert(R,0, vals, 1)


frame = []
approaches = np.unique(R[:,1])
for approach in approaches:
    rank = 0
    score = 0


    row = np.where(SR[:, 1] == approach)
    row = SR[row]
    _rank = row[0][0].astype(np.float)
    _score = row[0][3].astype(np.float)
    rank += _rank
    score += _score / 3
    SR_Rank = _rank
    SR_Score = _score

    row = np.where(R[:, 1] == approach)
    row = R[row]
    _rank = row[0][0].astype(np.float)
    _score = row[0][3].astype(np.float)
    rank += _rank
    score += _score / 3
    R_Rank = _rank
    R_Score = _score

    row = np.where(IA[:, 1] == approach)
    row = IA[row]
    _rank = row[0][0].astype(np.float)
    _score = row[0][3].astype(np.float)
    rank += _rank
    score += _score / 3
    IA_Rank = _rank
    IA_Score = _score

    approach_label = None
    if approach == 'Saturating_Uniform_10':
        approach_label = 'Saturating Uniform, 10'
    if approach == 'Saturating_Uniform_30':
        approach_label = 'Saturating Uniform, 30'
    if approach == 'Saturating_Uniform_60':
        approach_label = 'Saturating Uniform, 60'
    if approach == 'Saturating_Uniform_100':
        approach_label = 'Saturating Uniform, 100'
    if approach == 'Peaking_Uniform_10':
        approach_label = 'Peaking Uniform, 10'
    if approach == 'Peaking_Uniform_30':
        approach_label = 'Peaking Uniform, 30'
    if approach == 'Peaking_Uniform_60':
        approach_label = 'Peaking Uniform, 60'
    if approach == 'Peaking_Uniform_100':
        approach_label = 'Peaking Uniform, 100'
    if approach == 'Weight_Uniform_10':
        approach_label = 'Weighted Uniform, 10'
    if approach == 'Weight_Uniform_30':
        approach_label = 'Weighted Uniform, 30'
    if approach == 'Weight_Uniform_60':
        approach_label = 'Weighted Uniform, 60'
    if approach == 'Weight_Uniform_100':
        approach_label = 'Weighted Uniform, 100'
    if approach == 'SaturatingCRMExploit':
        approach_label = 'Saturating, Fully Continual, Standard'
    if approach == 'PeakCRMExploit':
        approach_label = 'Peaking, Fully Continual, Standard'
    if approach == 'WeightCRMExploit':
        approach_label = 'Weighted, Fully Continual, Standard'
    if approach == 'SaturatingCRMBalanced_2':
        approach_label = 'Saturating, Fully Continual, Balanced'
    if approach == 'PeakCRMBalanced_2':
        approach_label = 'Peaking, Fully Continual, Balanced'
    if approach == 'WeightCRMBalanced_2':
        approach_label = 'Weighted, Fully Continual, Balanced'
    if approach == 'Saturating_3_Stage_SoftMax':
        approach_label = 'Saturating, Softmax 3 Step'
    if approach == 'Peak_3_Stage_SoftMax':
        approach_label = 'Peaking, Softmax 3 Step'
    if approach == 'Weight_3_Stage_SoftMax':
        approach_label = 'Weighted, Softmax 3 Step'
    if approach_label == None:
        sys.exit()


    result = [approach_label, int(rank), score, int(SR_Rank), SR_Score, int(R_Rank), R_Score, int(IA_Rank), IA_Score]
    frame.append(result)

frame = np.asarray(frame)


frame = frame[np.argsort(frame[:, 1].astype(np.float))]


dataset = pd.DataFrame({'Approach': frame[:, 0],
                        'Rank_Sum': frame[:, 1],
                        'Score_Mean': frame[:, 2].astype(np.float),
                        'SR_Rank': frame[:, 3],
                        'SR_Score': frame[:, 4].astype(np.float),
                        'R_Rank': frame[:, 5],
                        'R_Score': frame[:, 6].astype(np.float),
                        'IA_Rank': frame[:, 7],
                        'IA_Score': frame[:, 8].astype(np.float)})


import csv
dataset.to_csv(data_path+ "\Copeland_Frame_Combined.csv", index = False, float_format='%.3f')

