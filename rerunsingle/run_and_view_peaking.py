'''
Visualises a single clinical trial where the approach assumes peaking efficacy
'''

from important_functions.simulate_trial_functions import *
from scenarios_and_approaches import *

approach = approachPeak_Uniform_100 # Change this
scenario = scenarioP1 # Change this




experiment_ID = np.random.randint(0, 10000) # change to a sppecific seed if desired
np.random.seed(experiment_ID)  # included to ensure reproducibility

result = simulatetrial(scenario, approach)

suggested, response, theoretical, is_explored, actual_data, params, total_utility, average_utility = result
eff_params, tox_params = params

doses, eff_data, _ , tox_data = actual_data


query_doses = np.linspace(0, 10, 101)
pred_efficacy = scaled_peaking(query_doses,eff_params)
true_efficacy = scenario.efficacy_probability(query_doses)
plt.plot(query_doses, pred_efficacy, c= 'orange')
plt.plot(query_doses, true_efficacy, alpha = .5)
plt.scatter(doses, eff_data)
plt.show()
plot_toxicity(query_doses,tox_params)
plot_toxicity(query_doses,scenario.tox_params)
pred_utilities = dose_utility(query_doses, 'peaking', eff_params, tox_params, scenario.outcome_scores)
true_utilities = scenario.dose_utility(query_doses)
plt.plot(query_doses, pred_utilities, c='orange')
plt.plot(query_doses, true_utilities, alpha=.5)
plt.show()



