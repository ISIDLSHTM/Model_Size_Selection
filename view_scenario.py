"""
Plots the efficacy, toxicity, and utility curve for a given scenario (specified on line 8)
"""

from scenarios_and_approaches import *
from crm_classes import *

scenario = scenarioS1 # Change this to visualise a scenario


## Run
print(scenario.calculate_utility_max())
scenario.plot_toxicity()
plt.show()
scenario.plot_efficacy()
plt.show()
scenario.plot_utility()
plt.show()
