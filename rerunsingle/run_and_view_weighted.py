'''
Visualises a single clinical trial where the approach assumes weighted efficacy
'''


from important_functions.simulate_trial_functions import *
from scenarios_and_approaches import *

approach = approachWeight_Uniform_100
scenario = scenarioP5



experiment_ID = np.random.randint(0, 10000) # change to a sppecific seed if desired
np.random.seed(experiment_ID)  # included to ensure reproducibility

result = simulatetrial(scenario, approach)

suggested, response, theoretical, is_explored, actual_data, params, total_utility, average_utility = result

eff_params, tox_params, aic_params = params
saturating_params, peaking_params, _  = eff_params
doses, eff_data, _ , tox_data = actual_data


query_doses = np.linspace(0, 10, 101)
pred_efficacy_saturating = scaled_saturating(query_doses,saturating_params)
pred_efficacy_peak = scaled_peaking(query_doses, peaking_params)
pred_efficacy = pred_efficacy_peak* aic_params[1] + pred_efficacy_saturating* aic_params[0]
true_efficacy = scenario.efficacy_probability(query_doses)
plt.plot(query_doses, pred_efficacy, color = 'y')
plt.plot(query_doses, pred_efficacy_saturating, color = 'b')
plt.plot(query_doses, pred_efficacy_peak, color = 'g')
plt.plot(query_doses, true_efficacy, alpha = .5)
plt.scatter(doses, eff_data)
plt.show()


plot_toxicity(query_doses,tox_params)
plot_toxicity(query_doses,scenario.tox_params)
pred_utilities = dose_utility(query_doses, 'weighted', eff_params, tox_params, scenario.outcome_scores)
true_utilities = scenario.dose_utility(query_doses)
plt.plot(query_doses, pred_utilities, c = 'orange', ls = '--')
plt.plot(query_doses, true_utilities, c = 'blue',alpha=.5)
plt.show()


