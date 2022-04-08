"""
Plots scenario_plots
"""

from scenarios_and_approaches import *
from crm_classes import *
import pathlib

store_path = str(pathlib.Path(__file__).parent.parent.resolve()) +'\scenario_plots'


scenarios = [scenarioS1, scenarioS2, scenarioS3, scenarioS4, scenarioS5,
             scenarioP1, scenarioP2, scenarioP3, scenarioP4, scenarioP5,
             scenarioX1, scenarioX2, scenarioX3, scenarioX4]


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


for scenario in scenarios:
    print(scenario.scenario_ID, scenario.calculate_utility_max(resolution=101))
    print(mapping_names[scenario.scenario_ID])
    scenario.plot_toxicity()
    name = store_path+ '\_tox' + scenario.scenario_ID + '.png'
    plt.title('Dose-Toxicity ' + mapping_names[scenario.scenario_ID] )
    plt.savefig(name)
    plt.close()

    scenario.plot_efficacy()
    name = store_path+ '\_eff' + scenario.scenario_ID + '.png'
    plt.title('Dose-Efficacy ' + mapping_names[scenario.scenario_ID])
    plt.savefig(name)
    plt.close()

    scenario.plot_utility()
    plt.title('Dose-Utility ' + mapping_names[scenario.scenario_ID])
    name = store_path + '\_utility' + scenario.scenario_ID + '.png'
    plt.savefig(name)
    plt.close()


