
"""
Contains all scenarios and dose optimisation approaches
"""

from crm_classes import *

## Approaches
# Full Explore
approachSaturating_Uniform_10 = Approach('Saturating_Uniform_10', 'saturating_uniform', Trial_Size=10,
                     Starting_Min=0, Starting_Max=10, Step_Size=1, Softmax=False, pseudo_weight_final = 0)
approachSaturating_Uniform_30 = Approach('Saturating_Uniform_30', 'saturating_uniform', Trial_Size=30,
                     Starting_Min=0, Starting_Max=10, Step_Size=1, Softmax=False, pseudo_weight_final = 0)
approachSaturating_Uniform_60 = Approach('Saturating_Uniform_60', 'saturating_uniform', Trial_Size=60,
                     Starting_Min=0, Starting_Max=10, Step_Size=1, Softmax=False, pseudo_weight_final = 0)
approachSaturating_Uniform_100 = Approach('Saturating_Uniform_100', 'saturating_uniform', Trial_Size=100,
                     Starting_Min=0, Starting_Max=10, Step_Size=1, Softmax=False, pseudo_weight_final = 0)

approachPeak_Uniform_10 = Approach('Peaking_Uniform_10', 'peaking_uniform', Trial_Size=10,
                     Starting_Min=0, Starting_Max=10, Step_Size=1, Softmax=False, pseudo_weight_final = 0)
approachPeak_Uniform_30 = Approach('Peaking_Uniform_30', 'peaking_uniform', Trial_Size=30,
                     Starting_Min=0, Starting_Max=10, Step_Size=1, Softmax=False, pseudo_weight_final = 0)
approachPeak_Uniform_60 = Approach('Peaking_Uniform_60', 'peaking_uniform', Trial_Size=60,
                     Starting_Min=0, Starting_Max=10, Step_Size=1, Softmax=False, pseudo_weight_final = 0)
approachPeak_Uniform_100 = Approach('Peaking_Uniform_100', 'peaking_uniform', Trial_Size=100,
                     Starting_Min=0, Starting_Max=10, Step_Size=1, Softmax=False, pseudo_weight_final = 0)


approachWeight_Uniform_10 = Approach('Weight_Uniform_10', 'weighted_uniform', Trial_Size=10,
                     Starting_Min=0, Starting_Max=10, Step_Size=1, Softmax=False, pseudo_weight_final = 0)
approachWeight_Uniform_30 = Approach('Weight_Uniform_30', 'weighted_uniform', Trial_Size=30,
                     Starting_Min=0, Starting_Max=10, Step_Size=1, Softmax=False, pseudo_weight_final = 0)
approachWeight_Uniform_60 = Approach('Weight_Uniform_60', 'weighted_uniform', Trial_Size=60,
                     Starting_Min=0, Starting_Max=10, Step_Size=1, Softmax=False, pseudo_weight_final = 0)
approachWeight_Uniform_100 = Approach('Weight_Uniform_100', 'weighted_uniform', Trial_Size=100,
                     Starting_Min=0, Starting_Max=10, Step_Size=1, Softmax=False, pseudo_weight_final = 0)


# CRM Exploit
approachSaturating_CRM_Exploit = Approach('SaturatingCRMExploit', 'saturating', Trial_Size= 30,
                                   Starting_Min=5, Starting_Max=5, Step_Size=.5, resolution=101,
                                   Softmax=False, pseudo_weight_run= 0.01, pseudo_weight_final=0)
approachPeak_CRM_Exploit = Approach('PeakCRMExploit', 'peaking', Trial_Size= 30,
                                   Starting_Min=5, Starting_Max=5, Step_Size=.5, resolution=101,
                                   Softmax=False, pseudo_weight_run= 0.01, pseudo_weight_final=0)
approachWeight_CRM_Exploit = Approach('WeightCRMExploit', 'weighted', Trial_Size= 30,
                                   Starting_Min=5, Starting_Max=5, Step_Size=.5, resolution=101,
                                   Softmax=False, pseudo_weight_run=0.01, pseudo_weight_final=0)
# CRM Balanced
approachSaturating_CRM_Balanced = Approach('SaturatingCRMBalanced', 'saturating', Trial_Size= 30,
                                   Starting_Min=5, Starting_Max=5, Step_Size=.5, resolution=101,
                                   Softmax=True, Inv_Temp=220, pseudo_weight_run= 0.01, pseudo_weight_final=0)
approachPeak_CRM_Balanced = Approach('PeakCRMBalanced', 'peaking', Trial_Size= 30,
                                   Starting_Min=5, Starting_Max=5, Step_Size=.5, resolution=101,
                                   Softmax=True, Inv_Temp=220, pseudo_weight_run= 0.01, pseudo_weight_final=0)
approachWeight_CRM_Balanced = Approach('WeightCRMBalanced', 'weighted', Trial_Size= 30,
                                   Starting_Min=5, Starting_Max=5, Step_Size=.5, resolution=101,
                                   Softmax=True, Inv_Temp=220, pseudo_weight_run=0.01, pseudo_weight_final=0)
# 3 Stage Soft Max
approach3_Step_Saturatingmoid_SoftMax = Approach('Saturating_3_Stage_SoftMax', 'step_saturatingmoid', Trial_Size=3,
                     Starting_Min=0, Starting_Max=10, Step_Size=1, Softmax=False, pseudo_weight_final= 0,resolution=101,
                    Step_Populations = [10,10,10], Step_Pseudo_Weights = [0, 0.01, 0.001, 0],
                    Step_Explore_Params = ['U', 58.88, 294], # 95% confidence interval within 0.05, 0.01
                    Threshold = False)
approach3_Step_Peak_SoftMax = Approach('Peak_3_Stage_SoftMax', 'step_peaking', Trial_Size=3,
                     Starting_Min=0, Starting_Max=10, Step_Size=1, Softmax=False, pseudo_weight_final= 0,resolution=101,
                    Step_Populations = [10,10,10], Step_Pseudo_Weights = [0, 0.01, 0.001, 0],
                    Step_Explore_Params = ['U', 58.88, 294], # 95% confidence interval within 0.05, 0.01
                    Threshold = False)
approach3_Step_Weight_SoftMax = Approach('Weight_3_Stage_SoftMax', 'step_weighted', Trial_Size=3,
                     Starting_Min=0, Starting_Max=10, Step_Size=1, Softmax=False, pseudo_weight_final= 0,resolution=101,
                    Step_Populations = [10,10,10], Step_Pseudo_Weights = [0, 0.01, 0.001, 0],
                    Step_Explore_Params = ['U', 58.88, 294], # 95% confidence interval within 0.05, 0.01
                    Threshold = False)

# 3 Stage Pigeon Satisficer
approach3_Step_Saturatingmoid_Threshold = Approach('Saturating_3_Stage_Satisficer', 'step_saturatingmoid', Trial_Size=3,
                     Starting_Min=0, Starting_Max=10, Step_Size=1, Softmax=False, pseudo_weight_final=0,resolution=101,
                    Step_Populations = [10,10,10], Step_Pseudo_Weights = [0, 0.01, 0.001, 0],
                    Step_Explore_Params = ['U',0.05,0.01],
                    Threshold = True, Step_Random = False)
approach3_Step_Peak_Threshold = Approach('Peak_3_Stage_Satisficer', 'step_peaking', Trial_Size=3,
                     Starting_Min=0, Starting_Max=10, Step_Size=1, Softmax=False, pseudo_weight_final=0,resolution=101,
                    Step_Populations = [10,10,10], Step_Pseudo_Weights = [0, 0.01, 0.001, 0],
                    Step_Explore_Params = ['U',0.05,0.01],
                    Threshold = True, Step_Random = False)
approach3_Step_Weight_Threshold = Approach('Weight_3_Stage_Satisficer', 'step_weighted', Trial_Size=3,
                     Starting_Min=0, Starting_Max=10, Step_Size=1, Softmax=False, pseudo_weight_final=0,resolution=101,
                    Step_Populations = [10,10,10], Step_Pseudo_Weights = [0, 0.01, 0.001, 0],
                    Step_Explore_Params = ['U',0.05,0.01],
                    Threshold = True, Step_Random = False)

# CRM Balanced_2
approachSaturating_CRM_Balanced_2 = Approach('SaturatingCRMBalanced_2', 'saturating', Trial_Size= 30,
                                   Starting_Min=5, Starting_Max=5, Step_Size=.5, resolution=101,
                                   Softmax=True, Inv_Temp=69, pseudo_weight_run= 0.01, pseudo_weight_final=0)
approachPeak_CRM_Balanced_2 = Approach('PeakCRMBalanced_2', 'peaking', Trial_Size= 30,
                                   Starting_Min=5, Starting_Max=5, Step_Size=.5, resolution=101,
                                   Softmax=True, Inv_Temp=69, pseudo_weight_run= 0.01, pseudo_weight_final=0)
approachWeight_CRM_Balanced_2 = Approach('WeightCRMBalanced_2', 'weighted', Trial_Size= 30,
                                   Starting_Min=5, Starting_Max=5, Step_Size=.5, resolution=101,
                                   Softmax=True, Inv_Temp=69, pseudo_weight_run=0.01, pseudo_weight_final=0)


# CRM Weighted Balanced 10, 60, 100
approachWeight_CRM_Balanced_2_10 = Approach('WeightCRMBalanced_2_10', 'weighted', Trial_Size= 10,
                                   Starting_Min=5, Starting_Max=5, Step_Size=.5, resolution=101,
                                   Softmax=True, Inv_Temp=69, pseudo_weight_run=0.01, pseudo_weight_final=0)
approachWeight_CRM_Balanced_2_60 = Approach('WeightCRMBalanced_2_60', 'weighted', Trial_Size= 60,
                                   Starting_Min=5, Starting_Max=5, Step_Size=.5, resolution=101,
                                   Softmax=True, Inv_Temp=69, pseudo_weight_run=0.01, pseudo_weight_final=0)
approachWeight_CRM_Balanced_2_100 = Approach('WeightCRMBalanced_2_100', 'weighted', Trial_Size= 100,
                                   Starting_Min=5, Starting_Max=5, Step_Size=.5, resolution=101,
                                   Softmax=True, Inv_Temp=69, pseudo_weight_run=0.01, pseudo_weight_final=0)

# CRM Exploit 10,60,100
approachWeight_CRM_Exploit_10 = Approach('WeightCRMExploit_10', 'weighted', Trial_Size= 10,
                                   Starting_Min=5, Starting_Max=5, Step_Size=.5, resolution=101,
                                   Softmax=False, pseudo_weight_run=0.01, pseudo_weight_final=0)
approachWeight_CRM_Exploit_60 = Approach('WeightCRMExploit_60', 'weighted', Trial_Size= 60,
                                   Starting_Min=5, Starting_Max=5, Step_Size=.5, resolution=101,
                                   Softmax=False, pseudo_weight_run=0.01, pseudo_weight_final=0)
approachWeight_CRM_Exploit_100 = Approach('WeightCRMExploit_100', 'weighted', Trial_Size= 100,
                                   Starting_Min=5, Starting_Max=5, Step_Size=.5, resolution=101,
                                   Softmax=False, pseudo_weight_run=0.01, pseudo_weight_final=0)


# 3 Stage Soft Max, 10, 60, 100
approach3_Step_Weight_SoftMax_10 = Approach('Weight_3_Stage_SoftMax_10', 'step_weighted', Trial_Size=3,
                     Starting_Min=0, Starting_Max=10, Step_Size=1, Softmax=False, pseudo_weight_final= 0,resolution=101,
                    Step_Populations = [4,3,3], Step_Pseudo_Weights = [0, 0.01, 0.001, 0],
                    Step_Explore_Params = ['U', 58.88, 294], # 95% confidence interval within 0.05, 0.01
                    Threshold = False)
approach3_Step_Weight_SoftMax_60 = Approach('Weight_3_Stage_SoftMax_60', 'step_weighted', Trial_Size=3,
                     Starting_Min=0, Starting_Max=10, Step_Size=1, Softmax=False, pseudo_weight_final= 0,resolution=101,
                    Step_Populations = [20,20,20], Step_Pseudo_Weights = [0, 0.01, 0.001, 0],
                    Step_Explore_Params = ['U', 58.88, 294], # 95% confidence interval within 0.05, 0.01
                    Threshold = False)
approach3_Step_Weight_SoftMax_100 = Approach('Weight_3_Stage_SoftMax_100', 'step_weighted', Trial_Size=3,
                     Starting_Min=0, Starting_Max=10, Step_Size=1, Softmax=False, pseudo_weight_final= 0,resolution=101,
                    Step_Populations = [34,33,33], Step_Pseudo_Weights = [0, 0.01, 0.001, 0],
                    Step_Explore_Params = ['U', 58.88, 294], # 95% confidence interval within 0.05, 0.01
                    Threshold = False)



## Scenarios

scenarioS1 = Scenario('S1', # Steep around 7
                     'saturating',
                     np.array([1, 6,.9]),
                     np.array([1,3,9,10.5]),
                     np.array([.133, -0.006, -.051, -0.133]))

scenarioS2 = Scenario('S2', # wide
                     'saturating',
                     np.array([1.8, 2.5,.9]),
                     np.array([.5,1,4,5]),
                     np.array([.133, -0.006, -.051, -0.133]))


scenarioS3 = Scenario('S3', # low dose best
                     'saturating',
                     np.array([2.5, 2,.9]),
                     np.array([.5,.1,2.5,3]),
                     np.array([.133, -0.006, -.051, -0.133]))

scenarioS4 = Scenario('S4', # max best
                     'saturating',
                     np.array([.7, 7.5,.9]),
                     np.array([.5,1,2,5]),
                     np.array([.266, -0.006, -.051, -0.133]))

scenarioS5 = Scenario('S5', # similar efficacy throughout
                     'saturating',
                     np.array([.1, 8,1]),
                     np.array([.5,1,4,5]),
                     np.array([.266, -0.006, -.051, -0.133]))

scenarioP1 = Scenario('P1', # about 7 peaking
                     'peaking',
                     np.array([-9,3,-3/14]),
                     np.array([1,3,9,10.5]),
                     np.array([.133, -0.006, -.051, -0.133]))

scenarioP2 = Scenario('P2', # wide
                     'peaking',
                     np.array([-4,2, -2/12]),
                     np.array([.1,.1,.4,1.5]),
                     np.array([.133, -0.006, -.051, -0.133]))

scenarioP3 = Scenario('P3', # low dose best
                     'peaking',
                     np.array([-6, 5,-6/(2*4)]),
                     np.array([.5,1,3,5]),
                     np.array([.133, -0.006, -.051, -0.133]))

scenarioP4 = Scenario('P4', # max best
                     'peaking',
                     np.array([-12, 2.5,-2.5/(2*11)]),
                     np.array([.3,1,1.5,2]),
                     np.array([.266, -0.006, -.051, -0.133]))


scenarioP5 = Scenario('P5', # similar efficacy throughout
                     'peaking',
                     np.array([0,0.8, -0.8/(2*6)]),
                     np.array([.1,.1,.4,1.5]),
                     np.array([.133, -0.006, -.051, -0.133]))



scenarioX1 = Scenario('X1', # basically no response
                     'flat',
                     np.array([.02]),
                     np.array([.5,0,3,5]),
                     np.array([.133, -0.006, -.051, -0.133]))


scenarioX2 = Scenario('X2', # biphasic increasing
                     'biphasic',
                     np.array([.5,3,4,6,.9,.5]),
                     np.array([.5,1,3,5]),
                     np.array([.133, -0.006, -.051, -0.133]))

scenarioX3 = Scenario('X3', # biphasic peaking
                     'biphasic',
                     np.array([1,2,4,7,.5,2]),
                     np.array([.5,1,3,5]),
                     np.array([.133, -0.006, -.051, -0.133]))

scenarioX4 = Scenario('X4', # hill
                     'hill',
                     np.array([3,1.2]),
                     np.array([.2,1,1.2,2]),
                     np.array([.133, -0.006, -.051, -0.133]))

"""
Template
"""
# scenarioNAME = Scenario('NAME',
#                      'CURVE',
#                      np.array([X,..., X]),
#                      np.array([Y,Y,Y,Y]),
#                      np.array([Z,Z,Z,Z]))

