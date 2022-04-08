"""
Contains the functions used to simulate the clinical trials
"""

from crm_classes import *


def _final_dose_selection(scenario, approach, actual_data):
    outcome_scores = scenario.outcome_scores
    eff_pseudo_x = approach.eff_pseudo_x
    eff_pseudo_y = approach.eff_pseudo_y
    tox_pseudo_x = approach.tox_pseudo_x
    tox_pseudo_y = approach.tox_pseudo_y
    eff_actual_x, eff_actual_y, tox_actual_x, tox_actual_y = actual_data

    starting_min = approach.Starting_Min
    starting_max = approach.Starting_Max

    lowest_dose = approach.Lowest_Dose
    highest_dose = approach.Highest_Dose
    resolution = approach.Resolution
    pseudo_weight = approach.Pseudo_Weight_Final
    if approach.Approach_Type == 'saturating'\
            or approach.Approach_Type == 'saturating_uniform' or approach.Approach_Type == 'saturating_fixed' :
        suggested, response, theoretical, params = next_dose_saturating(eff_pseudo_x, eff_pseudo_y, eff_actual_x, eff_actual_y,
                                  tox_pseudo_x, tox_pseudo_y, tox_actual_x, tox_actual_y,
                                  outcome_scores, final=True, softmax=False,
                                  starting_min=starting_min, starting_max=starting_max,
                                  lowest_dose=lowest_dose, highest_dose=highest_dose, resolution=resolution,
                                  weight_pseudo=pseudo_weight)
        is_explored = (suggested == theoretical)
    elif approach.Approach_Type == 'peaking' \
            or approach.Approach_Type == 'peaking_uniform' or approach.Approach_Type == 'peaking_fixed' :
        suggested, response, theoretical, params = next_dose_peaking(eff_pseudo_x, eff_pseudo_y, eff_actual_x, eff_actual_y,
                                  tox_pseudo_x, tox_pseudo_y, tox_actual_x, tox_actual_y,
                                  outcome_scores, final=True, softmax=False,
                                  starting_min=starting_min, starting_max=starting_max,
                                  lowest_dose=lowest_dose, highest_dose=highest_dose, resolution=resolution,
                                  weight_pseudo=pseudo_weight)
        is_explored = (suggested == theoretical)
    elif approach.Approach_Type == 'weighted' \
            or approach.Approach_Type == 'weighted_uniform' or approach.Approach_Type == 'weighted_fixed':
        suggested, response, theoretical, params = next_dose_weighted(eff_pseudo_x, eff_pseudo_y, eff_actual_x, eff_actual_y,
                                  tox_pseudo_x, tox_pseudo_y, tox_actual_x, tox_actual_y,
                                  outcome_scores, final=True,
                                  softmax=False, starting_min=starting_min, starting_max=starting_max,
                                  lowest_dose=lowest_dose, highest_dose=highest_dose, resolution=resolution,
                                  weight_pseudo=pseudo_weight)
        is_explored = (suggested == theoretical)

    elif approach.Approach_Type == 'step_saturating':
        suggested, response, theoretical, params = next_dose_saturating(eff_pseudo_x, eff_pseudo_y, eff_actual_x, eff_actual_y,
                                  tox_pseudo_x, tox_pseudo_y, tox_actual_x, tox_actual_y,
                                  outcome_scores, final=True, softmax=False,
                                  starting_min=starting_min, starting_max=starting_max,
                                  lowest_dose=lowest_dose, highest_dose=highest_dose, resolution=resolution,
                                  weight_pseudo=approach.Step_Pseudo_Weights[-1])
        is_explored = (suggested == theoretical)

    elif approach.Approach_Type == 'step_peaking':
        suggested, response, theoretical, params = next_dose_peaking(eff_pseudo_x, eff_pseudo_y, eff_actual_x, eff_actual_y,
                                  tox_pseudo_x, tox_pseudo_y, tox_actual_x, tox_actual_y,
                                  outcome_scores, final=True, softmax=False,
                                  starting_min=starting_min, starting_max=starting_max,
                                  lowest_dose=lowest_dose, highest_dose=highest_dose, resolution=resolution,
                                  weight_pseudo=approach.Step_Pseudo_Weights[-1])
        is_explored = (suggested == theoretical)

    elif approach.Approach_Type == 'step_weighted':
        suggested, response, theoretical, params = next_dose_weighted(eff_pseudo_x, eff_pseudo_y, eff_actual_x, eff_actual_y,
                                  tox_pseudo_x, tox_pseudo_y, tox_actual_x, tox_actual_y,
                                  outcome_scores, final=True, softmax=False,
                                  starting_min=starting_min, starting_max=starting_max,
                                  lowest_dose=lowest_dose, highest_dose=highest_dose, resolution=resolution,
                                  weight_pseudo=approach.Step_Pseudo_Weights[-1])
        is_explored = (suggested == theoretical)
    else:
        print('approach unknown')
    return suggested, response, theoretical, is_explored, params

def _simulatetrial_main_loop(scenario, approach, actual_data):
    outcome_scores = scenario.outcome_scores
    eff_pseudo_x = approach.eff_pseudo_x
    eff_pseudo_y = approach.eff_pseudo_y
    tox_pseudo_x = approach.tox_pseudo_x
    tox_pseudo_y = approach.tox_pseudo_y
    eff_actual_x, eff_actual_y, tox_actual_x, tox_actual_y = actual_data

    starting_min = approach.Starting_Min
    starting_max = approach.Starting_Max
    soft_max = approach.Softmax
    inv_temp = approach.Inv_Temp
    step_size = approach.Step_Size
    lowest_dose = approach.Lowest_Dose
    highest_dose = approach.Highest_Dose
    resolution = approach.Resolution
    pseudo_weight = approach.Pseudo_Weight_Run
    step_populations = approach.Step_Populations
    step_psuedo = approach.Step_Pseudo_Weights
    step_explore_params = approach.Step_Explore_Params
    threshold = approach.Threshold
    step_random = approach.Step_Random
    fixed_doses = approach.Fixed_Doses
    if approach.Approach_Type == 'saturating':
        new_x = next_dose_saturating(eff_pseudo_x, eff_pseudo_y, eff_actual_x, eff_actual_y,
                                  tox_pseudo_x, tox_pseudo_y, tox_actual_x, tox_actual_y,
                                  outcome_scores,step_size=step_size, softmax = soft_max, inverse_temperature = inv_temp,
                                  starting_min=starting_min, starting_max=starting_max,
                                  lowest_dose=lowest_dose, highest_dose=highest_dose, resolution=resolution,
                                  weight_pseudo=pseudo_weight)
    elif approach.Approach_Type == 'peaking':
        new_x = next_dose_peaking(eff_pseudo_x, eff_pseudo_y, eff_actual_x, eff_actual_y,
                                  tox_pseudo_x, tox_pseudo_y, tox_actual_x, tox_actual_y,
                                  outcome_scores, step_size=step_size, softmax=soft_max, inverse_temperature=inv_temp,
                                  starting_min=starting_min, starting_max=starting_max,
                                  lowest_dose=lowest_dose, highest_dose=highest_dose, resolution=resolution,
                                  weight_pseudo=pseudo_weight)
    elif approach.Approach_Type == 'weighted':
        new_x = next_dose_weighted(eff_pseudo_x, eff_pseudo_y, eff_actual_x, eff_actual_y,
                                  tox_pseudo_x, tox_pseudo_y, tox_actual_x, tox_actual_y,
                                  outcome_scores, step_size=step_size, softmax=soft_max, inverse_temperature=inv_temp,
                                  starting_min=starting_min, starting_max=starting_max,
                                  lowest_dose=lowest_dose, highest_dose=highest_dose, resolution=resolution,
                                  weight_pseudo=pseudo_weight)


    elif approach.Approach_Type == 'step_saturating':
        new_x = next_dose_step_saturating(eff_pseudo_x, eff_pseudo_y, eff_actual_x, eff_actual_y,
                                   tox_pseudo_x, tox_pseudo_y, tox_actual_x, tox_actual_y,
                                   outcome_scores,
                                    step_populations, step_psuedo, step_explore_params,
                                    threshold = threshold, step_random = step_random,
                                   lowest_dose=lowest_dose, highest_dose=highest_dose, resolution=resolution)


    elif approach.Approach_Type == 'step_peaking':
        new_x = next_dose_step_peaking(eff_pseudo_x, eff_pseudo_y, eff_actual_x, eff_actual_y,
                                   tox_pseudo_x, tox_pseudo_y, tox_actual_x, tox_actual_y,
                                   outcome_scores,
                                    step_populations, step_psuedo, step_explore_params,
                                    threshold = threshold,  step_random = step_random,
                                   lowest_dose=lowest_dose, highest_dose=highest_dose, resolution=resolution)

    elif approach.Approach_Type == 'step_weighted':
        new_x = next_dose_step_weighted(eff_pseudo_x, eff_pseudo_y, eff_actual_x, eff_actual_y,
                                   tox_pseudo_x, tox_pseudo_y, tox_actual_x, tox_actual_y,
                                   outcome_scores,
                                    step_populations, step_psuedo, step_explore_params,
                                    threshold = threshold,  step_random = step_random,
                                   lowest_dose=lowest_dose, highest_dose=highest_dose, resolution=resolution)


    elif approach.Approach_Type == 'saturating_uniform':
        new_x = next_dose_uniform(eff_actual_x, trial_size=approach.Trial_Size,
                                  starting_min=starting_min, starting_max=starting_max)
    elif approach.Approach_Type == 'peaking_uniform':
        new_x = next_dose_uniform(eff_actual_x, trial_size=approach.Trial_Size,
                                  starting_min=starting_min, starting_max=starting_max)
    elif approach.Approach_Type == 'weighted_uniform':
        new_x = next_dose_uniform(eff_actual_x, trial_size=approach.Trial_Size,
                                  starting_min=starting_min, starting_max=starting_max)

    elif approach.Approach_Type == 'saturating_fixed'\
            or approach.Approach_Type == 'peaking_fixed' \
            or approach.Approach_Type == 'weighted_fixed':
        new_x = next_dose_fixed(eff_actual_x, fixed_doses)
    else:
        print('Unknown Approach', approach.Approach_Type)

    new_x = np.array([new_x]).reshape(-1)
    new_eff = scenario.efficacy_sample(new_x)
    new_tox = scenario.toxicity_sample(new_x)

    eff_actual_x = np.concatenate((eff_actual_x, new_x), axis=0)
    eff_actual_y = np.concatenate((eff_actual_y, new_eff), axis=0)
    tox_actual_x = np.concatenate((tox_actual_x, new_x), axis=0)
    tox_actual_y = np.concatenate((tox_actual_y, new_tox), axis=0)
    updated_data = eff_actual_x, eff_actual_y, tox_actual_x, tox_actual_y
    return updated_data


def simulatetrial(scenario, approach):
    # Get parameters from approach
    trial_size = approach.Trial_Size

    # set up storing arrays
    eff_actual_x = np.array([])
    eff_actual_y = np.array([])
    tox_actual_x = np.array([])
    tox_actual_y = np.array([])

    actual_data = eff_actual_x, eff_actual_y, tox_actual_x, tox_actual_y

    # Main loop + concatenate
    for Individual in range(trial_size):
        updated_data = _simulatetrial_main_loop(scenario, approach, actual_data)
        actual_data = updated_data


    # final estimation of optimal dose (within acceptance criteria, report further work needed if not)
    suggested, response, theoretical, is_explored, params = _final_dose_selection(scenario, approach, actual_data)
    # output all results needed for the trial class
    response = np.asscalar(response)

    # calculation of total utility and average utility
    eff_actual_y = actual_data[1]
    tox_actual_y = actual_data[3]


    total_utility, average_utility = calculate_experiment_utility(eff_actual_y, tox_actual_y, scenario.outcome_scores)

    return suggested, response, theoretical, is_explored, actual_data, params, total_utility, average_utility

