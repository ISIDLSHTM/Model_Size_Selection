'''
Every function involved in modelling, callibrations, and prediction of optimal dose for the various curve shapes
'''

import numpy as np
from scipy.optimize import minimize
import scipy.stats as stats
import matplotlib.pyplot as plt


np.seterr(divide='ignore', invalid='ignore') 

def scale_saturating_params(params):
    grad, mid, maximum = params
    grad, mid, maximum = min(6, abs(grad)), abs(mid), abs(maximum) 
    maximum = max(0, maximum)
    maximum = min(1, maximum)
    return_array = np.array([grad, mid, maximum])
    return return_array

def scale_peaking_params(params):
    mu, b1, b2 = params
    mu, b1, b2 = mu, min(6, abs(b1)), -abs(b2) 
    return_array = np.array([mu, b1, b2])
    return return_array

def scale_ord_params(params):
    w, t1, t2, t3 = params
    w = min(abs(w), 6) 
    return_array = np.array([w, t1, t2, t3])
    return return_array

def scaled_saturating(independent, params):
    grad, mid, maximum = scale_saturating_params(params)
    p = np.exp(-grad * (independent - mid))
    p = maximum / (1 + p)
    dependant = p
    return(dependant)

def scaled_peaking(independent, params):
    mu, b1, b2 = scale_peaking_params(params)
    alpha = mu + b1 * independent + b2 * independent ** 2
    p = np.exp(- alpha)
    p = 1 / (1 + p)
    dependant = p
    return dependant


def biphasic(independent, params): 
    grad1, grad2, mid1, mid2, maximum, frac = params
    _section1 = maximum * frac / (1 + np.exp(-grad1 * (independent - mid1)))
    _section2 = maximum * (1-frac) / (1 + np.exp(-grad2 * (independent - mid2)))
    dependant = _section1 + _section2
    if np.any(dependant <= 0) or np.any(dependant >= 1):
        print('Warning: Some probabilities are nonsensical, alter biphasic parameters')
    return dependant

def flat(independent, params): 
    value = params
    length = np.size(independent)
    dependant = np.ones(length) * value
    return dependant

def hill(independent, params): # called 'linear(dose)' in the work
    grad, maximum = params
    dependant = maximum * independent / (grad+independent)
    return dependant

def ordinal(dose, params):
    w, t1, t2, t3 = scale_ord_params(params)
    scale_1 = t1 - w * dose
    scale_2 = t2 - w * dose
    scale_3 = t3 - w * dose
    phi_0 = 0
    phi_1 = stats.norm.cdf(scale_1)
    phi_2 = stats.norm.cdf(scale_2)
    phi_3 = stats.norm.cdf(scale_3)
    phi_4 = 1
    P0 = phi_1 - phi_0
    P1 = phi_2 - phi_1
    P2 = phi_3 - phi_2
    P3 = phi_4 - phi_3
    return(P0,P1,P2,P3)

def sample_from_probability(prob_vec):
    vec_length = len(prob_vec)
    randarray = np.random.rand(vec_length)
    sample = prob_vec > randarray
    sample = sample.astype(int)
    return (sample)

def saturating_sample(dose, params):
    prob_vec = scaled_saturating(dose, params)
    sample = sample_from_probability(prob_vec)
    return(sample)

def peaking_sample(dose, params):
    prob_vec = scaled_peaking(dose, params)
    sample = sample_from_probability(prob_vec)
    return(sample)

def ordinal_sample(dose, params): # takes x and parameters and returns samples
    P0, P1, P2, P3= ordinal(dose, params)
    N = len(P1)
    randarray = np.random.rand(N)
    sample3 = P0 + P1 + P2 < randarray
    sample3 = sample3.astype(int)
    sample2 = P0 + P1 < randarray
    sample2 = sample2.astype(int)
    sample1 = P0 < randarray
    sample1 = sample1.astype(int)
    return_array = sample1+sample2+sample3
    return(return_array)

def likelihood_saturating(params, doses,results, weight_eff):
    probs = scaled_saturating(doses, params)
    yhat = probs 
    likelihood_array = yhat * results + (1 - yhat) * (1 - results)
    likelihood_array = np.log(likelihood_array)
    likelihood_array = weight_eff * likelihood_array
    negLL = -np.sum(likelihood_array)
    return(negLL)

def likelihood_peaking(params, doses,results, weight_eff):
    probs = scaled_peaking(doses, params)
    yhat = probs  
    likelihood_array = yhat * results + (1 - yhat) * (1 - results)
    likelihood_array = weight_eff * np.log(likelihood_array)
    negLL = -np.sum(likelihood_array)
    return (negLL)

def likelihood_ord(params, doses, results, weights_tox):
    p = ordinal(doses, params)
    yhat = p 
    likelihood_array0 = (results == 0).astype(int) * yhat[0]
    likelihood_array1 = (results == 1).astype(int) * yhat[1]
    likelihood_array2 = (results == 2).astype(int) * yhat[2]
    likelihood_array3 = (results == 3).astype(int) * yhat[3]
    likelihood_array = likelihood_array0 + likelihood_array1 + likelihood_array2 + likelihood_array3
    likelihood_array2 = np.log(likelihood_array)
    likelihood_array3 = weights_tox * likelihood_array2
    negLL = -np.sum(likelihood_array3)
    return (negLL)

def saturating_callibrate(eff_x, eff_y, guess = None, weight_eff = None):
    if guess is None:
        guess = np.array([1, 5, .8]) # arbritary guess, but appears reasonable and prevents errors
    if weight_eff is None:
        length = np.size(eff_x)
        weight_eff = np.ones(length)
    results = minimize(likelihood_saturating, guess, args=(eff_x, eff_y, weight_eff), method="nelder-mead")
    return results

def peaking_callibrate(eff_x, eff_y, guess = None, weight_eff = None):
    if guess is None:
        guess = np.array([-1,0.6,-0.05]) # arbritary guess, but appears reasonable and prevents errors
    if weight_eff is None:
        length = np.size(eff_x)
        weight_eff = np.ones(length)
    results = minimize(likelihood_peaking, guess, args=(eff_x, eff_y, weight_eff), method="nelder-mead")
    return results

def ordinal_callibrate(tox_x, tox_y, guess = None, weight_tox = None):
    if guess is None:
        guess = np.array([0.2, 0.2, 1, 2])
    if weight_tox is None:
        length = np.size(tox_x)
        weight_tox = np.ones(length)
    results = minimize(likelihood_ord, guess, args=(tox_x, tox_y, weight_tox), method="nelder-mead", options={'disp': False} )
    return results

def combine_data_and_pseudo(pseudo_x, pseudo_y, actual_x, actual_y, weight_pseudo = 0.01):
    x = np.concatenate((pseudo_x, actual_x), axis=0)
    y = np.concatenate((pseudo_y, actual_y), axis=0).astype(int)
    weight_pseudo = np.full((len(pseudo_x)), weight_pseudo)
    weight_actual = np.full((len(actual_x)), 1)
    weight = np.concatenate((weight_pseudo, weight_actual), axis=0)
    return x, y, weight

def soft_max(utilities, inverse_temperature):
    scaling = inverse_temperature * utilities


    max_scaling = np.amax(scaling) 
    if max_scaling >= 705:
        subtraction = max_scaling - 705
        scaling = scaling - subtraction

    utility_score = np.exp(scaling)
    total_score = np.sum(utility_score)
    np.nan_to_num(total_score, copy=False)
    scaled_utilities = utility_score/total_score
    return scaled_utilities

def soft_max_selector(utilities, inverse_temperature, samples = None):
    if samples is None:
        scaled_utilities = soft_max(utilities,inverse_temperature)
        number_of_doses = np.size(scaled_utilities)
        dose_index = np.arange(number_of_doses)
        chosen_index = np.random.choice(dose_index, p = scaled_utilities)
        return chosen_index
    else:
        chosen_indexes = np.empty(samples)
        for i in range(samples):
            scaled_utilities = soft_max(utilities, inverse_temperature)
            number_of_doses = np.size(scaled_utilities)
            dose_index = np.arange(number_of_doses)
            chosen_indexes[i] = np.random.choice(dose_index, p=scaled_utilities)
        return chosen_indexes



def generate_efficacy_pseudodata(doses, efficacy_at_dose, no_efficacy_at_dose):
    pseudo_x = np.array([])
    pseudo_y = np.array([])
    for index, dose in enumerate(doses):
        total = efficacy_at_dose[index] + no_efficacy_at_dose[index]
        x_for_this_dose = np.full(total, dose)
        eff_for_this_dose = np.full(efficacy_at_dose[index], 1)
        no_eff_for_this_dose = np.full(no_efficacy_at_dose[index], 0)
        pseudo_x = np.concatenate((pseudo_x, x_for_this_dose), axis=0)
        pseudo_y = np.concatenate((pseudo_y, eff_for_this_dose), axis=0)
        pseudo_y = np.concatenate((pseudo_y, no_eff_for_this_dose), axis=0)
    return pseudo_x, pseudo_y

def generate_toxicity_pseudodata(doses, Tox0_at_dose, Tox1_at_dose, Tox2_at_dose, Tox3_at_dose):
    pseudo_x = np.array([])
    pseudo_y = np.array([])
    for index, dose in enumerate(doses):
        total = Tox0_at_dose[index] + Tox1_at_dose[index] \
                + Tox2_at_dose[index] + Tox3_at_dose[index]
        x_for_this_dose = np.full(total, dose)
        Tox0_for_this_dose = np.full(Tox0_at_dose[index], 0)
        Tox1_for_this_dose = np.full(Tox1_at_dose[index], 1)
        Tox2_for_this_dose = np.full(Tox2_at_dose[index], 2)
        Tox3_for_this_dose = np.full(Tox3_at_dose[index], 3)
        pseudo_x = np.concatenate((pseudo_x, x_for_this_dose), axis=0)
        pseudo_y = np.concatenate((pseudo_y, Tox0_for_this_dose), axis=0)
        pseudo_y = np.concatenate((pseudo_y, Tox1_for_this_dose), axis=0)
        pseudo_y = np.concatenate((pseudo_y, Tox2_for_this_dose), axis=0)
        pseudo_y = np.concatenate((pseudo_y, Tox3_for_this_dose), axis=0)
    return pseudo_x, pseudo_y

def min_and_max(previous_doses,starting_min, starting_max, step_size):  
    if previous_doses.size == 0: 
        return starting_min, starting_max
    min_previous = np.min(previous_doses)
    max_previous = np.max(previous_doses)
    min_acceptable = min(min_previous - step_size, starting_min)
    max_acceptable = max(max_previous + step_size, starting_max)
    return min_acceptable, max_acceptable

def acceptable_dose_from_chosen(chosen_dose, query_doses, previous_doses, starting_min, starting_max, step_size=0.5):
    present_min, present_max = min_and_max(previous_doses, starting_min, starting_max, step_size)

    if not(np.isin(present_min, query_doses)): 
        present_min = min(np.extract(query_doses > present_min, query_doses))
    if not(np.isin(present_max, query_doses)): 
        present_max = max(np.extract(query_doses < present_max, query_doses))

    if chosen_dose <= present_min: 
        return present_min
    if chosen_dose >= present_max : 
        return present_max

    return chosen_dose

def plot_toxicity(query_doses, params):
    P0, P1, P2, P3 = ordinal(query_doses, params)
    plt.stackplot(query_doses, P0, P1, P2, P3, labels=('Tox0', 'Tox1', 'Tox2', 'Tox3'))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=4, fancybox=True, shadow=True)
    plt.show()



def calculate_utility_max(eff_function, eff_params, tox_params, outcome_scores,
                          query_doses):
    utility = dose_utility(query_doses, eff_function, eff_params, tox_params, outcome_scores)
    index = np.argmax(utility)
    optimal_dose = query_doses[index]
    optimal_response = utility[index]
    return optimal_dose, optimal_response

def next_dose_uniform(eff_actual_x, starting_min=0, starting_max=10, trial_size = 10):
    uniform_doses = np.linspace(starting_min, starting_max, trial_size)
    if np.size(eff_actual_x) == 0:
        next_dose = np.min(uniform_doses)
        return next_dose


    largest_previously_tested = np.max(eff_actual_x)
    larger_doses = uniform_doses[uniform_doses > largest_previously_tested]
    next_dose = np.min(larger_doses)

    return next_dose

def next_dose_fixed(eff_actual_x, fixed_doses=None):

    size_fixed_doses = len(fixed_doses)
    indi_number = np.size(eff_actual_x)
    next_index = indi_number % size_fixed_doses

    next_dose = fixed_doses[next_index]
    next_dose = np.float64(next_dose)
    return next_dose

def next_dose_saturating(eff_pseudo_x, eff_pseudo_y, eff_actual_x, eff_actual_y,
                   tox_pseudo_x, tox_pseudo_y, tox_actual_x, tox_actual_y,
                    outcome_scores, starting_min=0, starting_max=0, step_size=0.5,
                   weight_pseudo=0.01, softmax=True, inverse_temperature=None,
                      final=False, limit_doses=True,
                      lowest_dose= 0, highest_dose=10, resolution=1001, guess_eff=None, guess_tox=None):
    # Takes the avaliable data and psuedodata and generates next dose

    # CHECK SIZES AND SANITY
    if (softmax == True) and (inverse_temperature == None):
        print('Error: Please specify inverse temperature')
        return
    if len(eff_pseudo_x) != len(eff_pseudo_y):
        print('Failed: Eff Pseudodata of non-matching lengths', len(eff_pseudo_x), len(eff_pseudo_y))
        return()
    if len(eff_actual_x) != len(eff_actual_y):
        print('Failed: Efficacy data of non-matching lengths', len(eff_actual_x), len(eff_actual_y))
        return()
    if len(tox_pseudo_x) != len(tox_pseudo_y):
        print('Failed: Tox Pseudodata of non-matching lengths', len(tox_pseudo_x), len(tox_pseudo_y))
        return()
    if len(tox_actual_x) != len(tox_actual_y):
        print('Failed: Tox data of non-matching lengths',  len(tox_actual_x), len(tox_actual_y))
        return()

    # GENERATE QUERY DOSES
    query_doses = np.linspace(lowest_dose, highest_dose, resolution)

    # COMBINE ACTUAL AND PSEUDO DATA
    eff_x, eff_y, eff_weight = combine_data_and_pseudo(eff_pseudo_x, eff_pseudo_y,
                                                       eff_actual_x, eff_actual_y,
                                                       weight_pseudo= weight_pseudo)
    tox_x, tox_y, tox_weight = combine_data_and_pseudo(tox_pseudo_x, tox_pseudo_y,
                                                       tox_actual_x, tox_actual_y,
                                                       weight_pseudo=weight_pseudo)

    # CALLIBRATE CURVES
    eff_result = saturating_callibrate(eff_x, eff_y, weight_eff=eff_weight)
    eff_params = eff_result.x
    tox_result = ordinal_callibrate(tox_x, tox_y, weight_tox=tox_weight)
    tox_params = tox_result.x

    # CALCULATE DOSE-UTILITY AND PREDICT OPTIMAL OR SOFTMAX OPTIMAL
    if softmax == False: # chooses optimal with no exploration
        predicted_optimal_dose, predicted_optimal_response = calculate_utility_max('saturating', eff_params, tox_params,
                                                                                   outcome_scores, query_doses)
    else:
        utilities = dose_utility(query_doses, 'saturating', eff_params, tox_params, outcome_scores)
        predicted_optimal_dose_index = soft_max_selector(utilities, inverse_temperature=inverse_temperature)
        predicted_optimal_dose = query_doses[predicted_optimal_dose_index]



    # ADJUST OPTIMAL BASED ON ACCEPTANCE RULES
    if final:
        acceptable_dose = acceptable_dose_from_chosen(predicted_optimal_dose, query_doses, eff_actual_x,
                                                starting_min, starting_max, step_size)
        predicted_response = dose_utility(acceptable_dose, 'saturating', eff_params, tox_params, outcome_scores)

        theoretical_optimal = predicted_optimal_dose
        params = eff_params, tox_params
        return acceptable_dose, predicted_response, theoretical_optimal, params
    elif limit_doses:
        next_dose = acceptable_dose_from_chosen(predicted_optimal_dose, query_doses, eff_actual_x,
                                      starting_min, starting_max, step_size)
    else:
        next_dose = predicted_optimal_dose

    predicted_optimal_dose_no_softmax, _ = calculate_utility_max('saturating', eff_params, tox_params,
                                                                               outcome_scores, query_doses)

    return next_dose

def next_dose_peaking(eff_pseudo_x, eff_pseudo_y, eff_actual_x, eff_actual_y,
                   tox_pseudo_x, tox_pseudo_y, tox_actual_x, tox_actual_y,
                    outcome_scores, starting_min=0, starting_max=0, step_size=0.5,
                   weight_pseudo=0.01, softmax=True, inverse_temperature=None,
                      final=False, limit_doses=True,
                      lowest_dose= 0, highest_dose=10, resolution=1001, guess_eff=None, guess_tox=None):
    # Takes the avaliable data and psuedodata and generates next dose

    # CHECK SIZES AND SANITY
    if (softmax == True) and (inverse_temperature == None):
        print('Error: Please specify inverse temperature')
        return
    if len(eff_pseudo_x) != len(eff_pseudo_y):
        print('Failed: Eff Pseudodata of non-matching lengths', len(eff_pseudo_x), len(eff_pseudo_y))
        return()
    if len(eff_actual_x) != len(eff_actual_y):
        print('Failed: Efficacy data of non-matching lengths', len(eff_actual_x), len(eff_actual_y))
        return()
    if len(tox_pseudo_x) != len(tox_pseudo_y):
        print('Failed: Tox Pseudodata of non-matching lengths', len(tox_pseudo_x), len(tox_pseudo_y))
        return()
    if len(tox_actual_x) != len(tox_actual_y):
        print('Failed: Tox data of non-matching lengths',  len(tox_actual_x), len(tox_actual_y))
        return()

    # GENERATE QUERY DOSES
    query_doses = np.linspace(lowest_dose, highest_dose, resolution)

    # COMBINE ACTUAL AND PSEUDO DATA
    eff_x, eff_y, eff_weight = combine_data_and_pseudo(eff_pseudo_x, eff_pseudo_y,
                                                       eff_actual_x, eff_actual_y,
                                                       weight_pseudo= weight_pseudo)
    tox_x, tox_y, tox_weight = combine_data_and_pseudo(tox_pseudo_x, tox_pseudo_y,
                                                       tox_actual_x, tox_actual_y,
                                                       weight_pseudo=weight_pseudo)

    # CALLIBRATE CURVES
    eff_result = peaking_callibrate(eff_x, eff_y, weight_eff=eff_weight)
    eff_params = eff_result.x
    tox_result = ordinal_callibrate(tox_x, tox_y, weight_tox=tox_weight)
    tox_params = tox_result.x

    # CALCULATE DOSE-UTILITY AND PREDICT OPTIMAL OR SOFTMAX OPTIMAL
    if softmax == False: # chooses optimal with no exploration
        predicted_optimal_dose, predicted_optimal_response = calculate_utility_max('peaking', eff_params, tox_params,
                                                                                   outcome_scores, query_doses)
    else:
        utilities = dose_utility(query_doses, 'peaking', eff_params, tox_params, outcome_scores)

        predicted_optimal_dose_index = soft_max_selector(utilities, inverse_temperature=inverse_temperature)
        predicted_optimal_dose = query_doses[predicted_optimal_dose_index]



    # ADJUST OPTIMAL BASED ON ACCEPTANCE RULES
    if final:
        acceptable_dose = acceptable_dose_from_chosen(predicted_optimal_dose, query_doses, eff_actual_x,
                                                starting_min, starting_max, step_size)
        _acceptable_dose = np.array([acceptable_dose]) # just done to ensure that this is of the form of an array
        predicted_response = dose_utility(_acceptable_dose, 'peaking', eff_params, tox_params, outcome_scores)
        theoretical_optimal = predicted_optimal_dose
        params = eff_params, tox_params
        return acceptable_dose, predicted_response, theoretical_optimal, params
    elif limit_doses:
        next_dose = acceptable_dose_from_chosen(predicted_optimal_dose, query_doses, eff_actual_x,
                                      starting_min, starting_max, step_size)
    else:
        next_dose = predicted_optimal_dose

    return next_dose



def next_dose_weighted(eff_pseudo_x, eff_pseudo_y, eff_actual_x, eff_actual_y,
                   tox_pseudo_x, tox_pseudo_y, tox_actual_x, tox_actual_y,
                    outcome_scores, starting_min=0, starting_max=0, step_size=0.5,
                   weight_pseudo=0.01, softmax=True, inverse_temperature=None,
                      final=False, limit_doses=True,
                      lowest_dose= 0, highest_dose=10, resolution=1001, guess_eff=None, guess_tox=None):
    # Takes the avaliable data and psuedodata and generates next dose

    # CHECK SIZES AND SANITY
    if (softmax == True) and (inverse_temperature == None):
        print('Error: Please specify inverse temperature')
        return
    if len(eff_pseudo_x) != len(eff_pseudo_y):
        print('Failed: Eff Pseudodata of non-matching lengths', len(eff_pseudo_x), len(eff_pseudo_y))
        return()
    if len(eff_actual_x) != len(eff_actual_y):
        print('Failed: Efficacy data of non-matching lengths', len(eff_actual_x), len(eff_actual_y))
        return()
    if len(tox_pseudo_x) != len(tox_pseudo_y):
        print('Failed: Tox Pseudodata of non-matching lengths', len(tox_pseudo_x), len(tox_pseudo_y))
        return()
    if len(tox_actual_x) != len(tox_actual_y):
        print('Failed: Tox data of non-matching lengths',  len(tox_actual_x), len(tox_actual_y))
        return()

    # GENERATE QUERY DOSES
    query_doses = np.linspace(lowest_dose, highest_dose, resolution)

    # COMBINE ACTUAL AND PSEUDO DATA
    eff_x, eff_y, eff_weight = combine_data_and_pseudo(eff_pseudo_x, eff_pseudo_y,
                                                       eff_actual_x, eff_actual_y,
                                                       weight_pseudo= weight_pseudo)
    tox_x, tox_y, tox_weight = combine_data_and_pseudo(tox_pseudo_x, tox_pseudo_y,
                                                       tox_actual_x, tox_actual_y,
                                                       weight_pseudo=weight_pseudo)

    # CALLIBRATE CURVES
    eff_result_saturating = saturating_callibrate(eff_x, eff_y, weight_eff=eff_weight)
    eff_params_saturating = eff_result_saturating.x
    eff_result_peaking = peaking_callibrate(eff_x, eff_y, weight_eff=eff_weight)
    eff_params_peaking = eff_result_peaking.x
    tox_result = ordinal_callibrate(tox_x, tox_y, weight_tox=tox_weight)
    tox_params = tox_result.x

    ## Calulate relative likelihoods
    length = np.size(eff_actual_x)
    weight_eff = np.ones(length)

    saturating_negloglik = likelihood_saturating(eff_params_saturating, eff_actual_x, eff_actual_y, weight_eff)
    peaking_negloglik = likelihood_peaking(eff_params_peaking, eff_actual_x, eff_actual_y, weight_eff)
    negative_likelihoods = np.array([saturating_negloglik,peaking_negloglik])

    lik_params = weight_likelihoods(negative_likelihoods)
    eff_params = eff_params_saturating, eff_params_peaking, lik_params


    # CALCULATE DOSE-UTILITY AND PREDICT OPTIMAL OR SOFTMAX OPTIMAL
    if softmax == False: # chooses optimal with no exploration
        predicted_optimal_dose, predicted_optimal_response = calculate_utility_max('weighted', eff_params, tox_params,
                                                                                   outcome_scores, query_doses)

    else:
        utilities = dose_utility(query_doses, 'weighted', eff_params, tox_params, outcome_scores)
        predicted_optimal_dose_index = soft_max_selector(utilities, inverse_temperature=inverse_temperature)
        predicted_optimal_dose = query_doses[predicted_optimal_dose_index]



    # ADJUST OPTIMAL BASED ON ACCEPTANCE RULES
    if final:
        acceptable_dose = acceptable_dose_from_chosen(predicted_optimal_dose, query_doses, eff_actual_x,
                                                starting_min, starting_max, step_size)
        _acceptable_dose = np.array([acceptable_dose]) # just done to ensure that this is of the form of an array
        predicted_response = dose_utility(_acceptable_dose, 'weighted', eff_params, tox_params, outcome_scores)
        theoretical_optimal = predicted_optimal_dose
        params = eff_params, tox_params, lik_params
        return acceptable_dose, predicted_response, theoretical_optimal, params
    elif limit_doses:
        next_dose = acceptable_dose_from_chosen(predicted_optimal_dose, query_doses, eff_actual_x,
                                      starting_min, starting_max, step_size)
    else:
        next_dose = predicted_optimal_dose

    return next_dose



def weight_likelihoods(likelihood_array):
    min_NLL = np.min(likelihood_array)
    _likelihood_array = likelihood_array - min_NLL
    unscaled_weights = np.exp(-(_likelihood_array))
    weights = unscaled_weights / np.sum(unscaled_weights)
    return weights




def dose_utility(dose, eff_function, eff_params, tox_params, outcome_scores):
    eff_score = outcome_scores[0]
    tox1_score = outcome_scores[1]
    tox2_score = outcome_scores[2]
    tox3_score = outcome_scores[3]

    #  Calculate dose-efficacy
    if eff_function == 'saturating':
        efficacy = scaled_saturating(dose, eff_params)
    elif eff_function == 'peaking':
        efficacy = scaled_peaking(dose, eff_params)
    elif eff_function == 'weighted':
        eff_params_saturating, eff_params_peaking, lik_params = eff_params
        # saturating eff
        efficacy_saturating = scaled_saturating(dose, eff_params_saturating)
        # peaking_eff
        efficacy_peaking = scaled_peaking(dose, eff_params_peaking)
        # lik weights
        saturating_lik, peaking_lik = lik_params
        efficacy = saturating_lik * efficacy_saturating + peaking_lik * efficacy_peaking
    else:
        print('Error: Eff function not implemented')
        return None

    #  Calculate dose-toxicity
    P0, P1, P2, P3 = ordinal(dose, tox_params)

    #  Calculate utility
    utility = efficacy * eff_score
    utility += (0 * P0 + tox1_score * P1 + tox2_score * P2 + tox3_score * P3)
    return utility





def calculate_experiment_utility(eff_data, tox_data, outcome_scores):
    total_efficacy = np.count_nonzero(eff_data == 1)
    total_tox1 = np.count_nonzero(tox_data == 1)
    total_tox2 = np.count_nonzero(tox_data == 2)
    total_tox3 = np.count_nonzero(tox_data == 3)
    outcome_table = np.asarray([total_efficacy, total_tox1, total_tox2, total_tox3])
    total_utility = np.sum(outcome_table * outcome_scores)
    average_utility = total_utility/np.size(eff_data)
    return total_utility, average_utility



def next_dose_step_saturating(eff_pseudo_x, eff_pseudo_y, eff_actual_x, eff_actual_y,
                   tox_pseudo_x, tox_pseudo_y, tox_actual_x, tox_actual_y,
                    outcome_scores, step_populations, step_psuedo_weights, step_explore_params,
                    threshold=False, step_random=False,
                      lowest_dose= 0, highest_dose=10, resolution=1001):
    # Takes the avaliable data and psuedodata and generates next dose

    # CHECK SIZES AND SANITY
    if len(eff_pseudo_x) != len(eff_pseudo_y):
        print('Failed: Eff Pseudodata of non-matching lengths', len(eff_pseudo_x), len(eff_pseudo_y))
        return()
    if len(eff_actual_x) != len(eff_actual_y):
        print('Failed: Efficacy data of non-matching lengths', len(eff_actual_x), len(eff_actual_y))
        return()
    if len(tox_pseudo_x) != len(tox_pseudo_y):
        print('Failed: Tox Pseudodata of non-matching lengths', len(tox_pseudo_x), len(tox_pseudo_y))
        return()
    if len(tox_actual_x) != len(tox_actual_y):
        print('Failed: Tox data of non-matching lengths',  len(tox_actual_x), len(tox_actual_y))
        return()

    ### Work out what step we're on
    step_totals = np.cumsum(step_populations)
    individuals_done = len(eff_actual_x)
    which_step = np.argmax(step_totals>individuals_done)
    next_population_size = step_populations[which_step]
    next_psuedo = step_psuedo_weights[which_step]
    next_rule = step_explore_params[which_step]

    if next_rule == 'U':
        next_doses = np.linspace(lowest_dose, highest_dose, next_population_size)
        return next_doses


    # GENERATE QUERY DOSES
    query_doses = np.linspace(lowest_dose, highest_dose, resolution)

    # COMBINE ACTUAL AND PSEUDO DATA
    eff_x, eff_y, eff_weight = combine_data_and_pseudo(eff_pseudo_x, eff_pseudo_y,
                                                       eff_actual_x, eff_actual_y,
                                                       weight_pseudo= next_psuedo)
    tox_x, tox_y, tox_weight = combine_data_and_pseudo(tox_pseudo_x, tox_pseudo_y,
                                                       tox_actual_x, tox_actual_y,
                                                       weight_pseudo=next_psuedo)

    # CALLIBRATE CURVES
    eff_result = saturating_callibrate(eff_x, eff_y, weight_eff=eff_weight)
    eff_params = eff_result.x
    tox_result = ordinal_callibrate(tox_x, tox_y, weight_tox=tox_weight)
    tox_params = tox_result.x

    # CALCULATE DOSE-UTILITY AND PREDICT OPTIMAL OR SOFTMAX OPTIMAL
    utilities = dose_utility(query_doses, 'saturating', eff_params, tox_params, outcome_scores)

    if threshold == True:
        predicted_optimal_dose_indexes = threshold_selection(utilities, next_rule)
        predicted_optimal_dose_indexes = random_satisfice(predicted_optimal_dose_indexes, next_population_size)
        predicted_optimal_doses = [query_doses[i] for i in predicted_optimal_dose_indexes]

    else:
        predicted_optimal_dose_indexes = soft_max_selector(utilities, inverse_temperature=next_rule,
                                                           samples=next_population_size)
        predicted_optimal_dose_indexes = predicted_optimal_dose_indexes.astype('int')
        predicted_optimal_doses = [query_doses[i] for i in predicted_optimal_dose_indexes]

    next_doses = predicted_optimal_doses

    return next_doses


def threshold_selection(utility_vector, tolerance):
    best=np.max(utility_vector)
    subtracted_vector = utility_vector - best
    args = np.argwhere(subtracted_vector >= -tolerance).reshape(-1)
    return args

def random_satisfice(accepted_x, number_of_samples):
    length = np.size(accepted_x)
    chosen_args = np.random.choice(range(length), number_of_samples, replace=True)
    chosen_x = accepted_x[chosen_args]
    return chosen_x






def next_dose_step_peaking(eff_pseudo_x, eff_pseudo_y, eff_actual_x, eff_actual_y,
                   tox_pseudo_x, tox_pseudo_y, tox_actual_x, tox_actual_y,
                    outcome_scores, step_populations, step_psuedo_weights, step_explore_params,
                    threshold=False, step_random=False,
                    lowest_dose= 0, highest_dose=10, resolution=1001):
    # Takes the avaliable data and psuedodata and generates next dose

    # CHECK SIZES AND SANITY
    if len(eff_pseudo_x) != len(eff_pseudo_y):
        print('Failed: Eff Pseudodata of non-matching lengths', len(eff_pseudo_x), len(eff_pseudo_y))
        return()
    if len(eff_actual_x) != len(eff_actual_y):
        print('Failed: Efficacy data of non-matching lengths', len(eff_actual_x), len(eff_actual_y))
        return()
    if len(tox_pseudo_x) != len(tox_pseudo_y):
        print('Failed: Tox Pseudodata of non-matching lengths', len(tox_pseudo_x), len(tox_pseudo_y))
        return()
    if len(tox_actual_x) != len(tox_actual_y):
        print('Failed: Tox data of non-matching lengths',  len(tox_actual_x), len(tox_actual_y))
        return()

    ### Work out what step we're on
    step_totals = np.cumsum(step_populations)
    individuals_done = len(eff_actual_x)
    which_step = np.argmax(step_totals>individuals_done)
    next_population_size = step_populations[which_step]
    next_psuedo = step_psuedo_weights[which_step]
    next_rule = step_explore_params[which_step]

    if next_rule == 'U':
        next_doses = np.linspace(lowest_dose, highest_dose, next_population_size)
        return next_doses


    # GENERATE QUERY DOSES
    query_doses = np.linspace(lowest_dose, highest_dose, resolution)

    # COMBINE ACTUAL AND PSEUDO DATA
    eff_x, eff_y, eff_weight = combine_data_and_pseudo(eff_pseudo_x, eff_pseudo_y,
                                                       eff_actual_x, eff_actual_y,
                                                       weight_pseudo= next_psuedo)
    tox_x, tox_y, tox_weight = combine_data_and_pseudo(tox_pseudo_x, tox_pseudo_y,
                                                       tox_actual_x, tox_actual_y,
                                                       weight_pseudo=next_psuedo)

    # CALLIBRATE CURVES
    eff_result = peaking_callibrate(eff_x, eff_y, weight_eff=eff_weight)
    eff_params = eff_result.x
    tox_result = ordinal_callibrate(tox_x, tox_y, weight_tox=tox_weight)
    tox_params = tox_result.x

    # CALCULATE DOSE-UTILITY AND PREDICT OPTIMAL OR SOFTMAX OPTIMAL
    utilities = dose_utility(query_doses, 'peaking', eff_params, tox_params, outcome_scores)

    if threshold == True:
        predicted_optimal_dose_indexes = threshold_selection(utilities, next_rule)
        predicted_optimal_dose_indexes = random_satisfice(predicted_optimal_dose_indexes, next_population_size)
        predicted_optimal_doses = [query_doses[i] for i in predicted_optimal_dose_indexes]

    else:
        predicted_optimal_dose_indexes = soft_max_selector(utilities, inverse_temperature=next_rule,
                                                           samples=next_population_size)
        predicted_optimal_dose_indexes = predicted_optimal_dose_indexes.astype('int')
        predicted_optimal_doses = [query_doses[i] for i in predicted_optimal_dose_indexes]

    next_doses = predicted_optimal_doses
    return next_doses






def next_dose_step_weighted(eff_pseudo_x, eff_pseudo_y, eff_actual_x, eff_actual_y,
                   tox_pseudo_x, tox_pseudo_y, tox_actual_x, tox_actual_y,
                    outcome_scores, step_populations, step_psuedo_weights, step_explore_params,
                    threshold=False, step_random=False,
                    lowest_dose= 0, highest_dose=10, resolution=1001):
    # Takes the avaliable data and psuedodata and generates next dose

    # CHECK SIZES AND SANITY
    if len(eff_pseudo_x) != len(eff_pseudo_y):
        print('Failed: Eff Pseudodata of non-matching lengths', len(eff_pseudo_x), len(eff_pseudo_y))
        return()
    if len(eff_actual_x) != len(eff_actual_y):
        print('Failed: Efficacy data of non-matching lengths', len(eff_actual_x), len(eff_actual_y))
        return()
    if len(tox_pseudo_x) != len(tox_pseudo_y):
        print('Failed: Tox Pseudodata of non-matching lengths', len(tox_pseudo_x), len(tox_pseudo_y))
        return()
    if len(tox_actual_x) != len(tox_actual_y):
        print('Failed: Tox data of non-matching lengths',  len(tox_actual_x), len(tox_actual_y))
        return()

    ### Work out what step we're on
    step_totals = np.cumsum(step_populations)
    individuals_done = len(eff_actual_x)
    which_step = np.argmax(step_totals>individuals_done)
    next_population_size = step_populations[which_step]
    next_psuedo = step_psuedo_weights[which_step]
    next_rule = step_explore_params[which_step]

    if next_rule == 'U':
        next_doses = np.linspace(lowest_dose, highest_dose, next_population_size)
        return next_doses


    # GENERATE QUERY DOSES
    query_doses = np.linspace(lowest_dose, highest_dose, resolution)

    # COMBINE ACTUAL AND PSEUDO DATA
    eff_x, eff_y, eff_weight = combine_data_and_pseudo(eff_pseudo_x, eff_pseudo_y,
                                                       eff_actual_x, eff_actual_y,
                                                       weight_pseudo= next_psuedo)
    tox_x, tox_y, tox_weight = combine_data_and_pseudo(tox_pseudo_x, tox_pseudo_y,
                                                       tox_actual_x, tox_actual_y,
                                                       weight_pseudo=next_psuedo)

    # CALLIBRATE CURVES
    eff_result_saturating = saturating_callibrate(eff_x, eff_y, weight_eff=eff_weight)
    eff_params_saturating = eff_result_saturating.x
    eff_result_peaking = peaking_callibrate(eff_x, eff_y, weight_eff=eff_weight)
    eff_params_peaking = eff_result_peaking.x
    tox_result = ordinal_callibrate(tox_x, tox_y, weight_tox=tox_weight)
    tox_params = tox_result.x

    ## Calulate relative likelihoods
    length = np.size(eff_actual_x)
    weight_eff = np.ones(length)

    saturating_negloglik = likelihood_saturating(eff_params_saturating, eff_actual_x, eff_actual_y, weight_eff)
    peaking_negloglik = likelihood_peaking(eff_params_peaking, eff_actual_x, eff_actual_y, weight_eff)
    negative_likelihoods = np.array([saturating_negloglik, peaking_negloglik])

    lik_params = weight_likelihoods(negative_likelihoods)
    eff_params = eff_params_saturating, eff_params_peaking, lik_params

    # CALCULATE DOSE-UTILITY AND PREDICT OPTIMAL OR SOFTMAX OPTIMAL
    utilities = dose_utility(query_doses, 'weighted', eff_params, tox_params, outcome_scores)

    if threshold == True:
        predicted_optimal_dose_indexes = threshold_selection(utilities, next_rule)
        predicted_optimal_dose_indexes = random_satisfice(predicted_optimal_dose_indexes, next_population_size)
        predicted_optimal_doses = [query_doses[i] for i in predicted_optimal_dose_indexes]

    else:
        predicted_optimal_dose_indexes = soft_max_selector(utilities, inverse_temperature=next_rule,
                                                           samples=next_population_size)
        predicted_optimal_dose_indexes = predicted_optimal_dose_indexes.astype('int')
        predicted_optimal_doses = [query_doses[i] for i in predicted_optimal_dose_indexes]

    next_doses = predicted_optimal_doses
    # ADJUST OPTIMAL BASED ON ACCEPTANCE RULES
    return next_doses





