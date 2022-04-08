"""
Contains the key classes for scenarios and dose-optimisation approaches
"""

from important_functions.shape_functions import *
import matplotlib.pyplot as plt


class Scenario:
    def __init__(self,scenario_ID, eff_function, eff_params, tox_params, outcome_scores, lowest_dose=0, highest_dose=10, resolution=1001):
        self.scenario_ID = scenario_ID
        self.eff_function = eff_function
        self.eff_params = eff_params
        self.tox_params = tox_params 
        self.outcome_scores = outcome_scores 
        1 / self.check_param_number_and_function() 
        self.optimal_dose, self.optimal_response = self.calculate_utility_max(lowest_dose, highest_dose, resolution)


    def check_param_number_and_function(self):
        if np.size(self.outcome_scores) == 4:
            if np.size(self.tox_params) == 4:
                if self.eff_function == 'saturating' and np.size(self.eff_params) == 3:
                    return 1
                if self.eff_function == 'peaking' and np.size(self.eff_params) == 3:
                    return 1
                if self.eff_function == 'biphasic' and np.size(self.eff_params) == 6:
                    return 1
                if self.eff_function == 'flat' and np.size(self.eff_params) == 1:
                    return 1
                if self.eff_function == 'hill' and np.size(self.eff_params) == 2:
                    return 1
        print('Error: Incorrect number of parameters or non-implemented function')

    def efficacy_probability(self, dose):
        if self.eff_function == 'saturating':
            efficacy = scaled_saturating(dose,self.eff_params)
            return efficacy
        elif self.eff_function == 'peaking':
            efficacy = scaled_peaking(dose,self.eff_params)
            return efficacy
        elif self.eff_function == 'biphasic':
            efficacy = biphasic(dose,self.eff_params)
            return efficacy
        elif self.eff_function == 'flat':
            efficacy = flat(dose,self.eff_params)
            return efficacy
        elif self.eff_function == 'hill':
            efficacy = hill(dose,self.eff_params)
            return efficacy
        else:
            print('Error: Efficacy Function not implemented')
            return None

    def efficacy_sample(self, dose):
        efficacy_probability =  self.efficacy_probability(dose)
        efficacy = sample_from_probability(efficacy_probability)
        return efficacy

    def toxicity_probability(self,dose):
        P0, P1, P2, P3 = ordinal(dose, self.tox_params)
        return P0, P1, P2, P3

    def toxicity_sample(self,dose):
        toxicity = ordinal_sample(dose, self.tox_params)
        return toxicity

    def dose_utility(self, dose):
        eff_score = self.outcome_scores[0]
        tox1_score = self.outcome_scores[1]
        tox2_score = self.outcome_scores[2]
        tox3_score = self.outcome_scores[3]
        efficacy = self.efficacy_probability(dose)
        P0, P1, P2, P3 = self.toxicity_probability(dose)
        utility = efficacy * eff_score
        utility += (0 * P0 + tox1_score * P1 + tox2_score * P2 + tox3_score * P3)
        return utility

    def calculate_utility_max(self, lowest_dose=0, highest_dose=10, resolution=1001):
        query_doses = np.linspace(lowest_dose, highest_dose, resolution)
        utility = self.dose_utility(query_doses)
        index = np.argmax(utility)
        optimal_dose = query_doses[index]
        optimal_response = utility[index]
        return optimal_dose, optimal_response

    def calculate_utility_min(self, lowest_dose=0, highest_dose=10, resolution=1001):
        query_doses = np.linspace(lowest_dose, highest_dose, resolution)
        utility = self.dose_utility(query_doses)
        index = np.argmin(utility)
        optimal_dose = query_doses[index]
        optimal_response = utility[index]
        return optimal_dose, optimal_response

    def calculate_utility_normalised_score(self, dose_utility):
        _, max = self.calculate_utility_max()
        _, min = self.calculate_utility_min()
        if max > 0:
            delta = max - min
            better_than_worst = dose_utility - min
            score = better_than_worst / delta
            return score
        delta = max - min
        better_than_worst = dose_utility - min
        score = better_than_worst / delta
        return score

    def plot_efficacy(self, lowest_dose=0, highest_dose=10, resolution=1001):
        query_doses = np.linspace(lowest_dose, highest_dose, resolution)
        efficacy = self.efficacy_probability(query_doses)
        plt.plot(query_doses, efficacy)
        plt.ylim((0,1))
        plt.xlabel('loP10 Dose')
        plt.ylabel('Efficacy Probability')
        plt.title("Dose-Efficacy "+ self.scenario_ID)

    def plot_toxicity(self, lowest_dose=0, highest_dose=10, resolution=1001):
        query_doses = np.linspace(lowest_dose, highest_dose, resolution)
        P_T0, P_T1, P_T2, P_T3 = self.toxicity_probability(query_doses)
        plt.margins(x=0, y = 0)
        plt.plot(query_doses, P_T0, color = 'black')
        plt.plot(query_doses, P_T1+P_T0, color='black')
        plt.plot(query_doses, P_T2+P_T1+P_T0, color='black')
        plt.plot(query_doses, P_T3+P_T2+P_T1+P_T0, color='black')
        plt.stackplot(query_doses, P_T0, P_T1, P_T2, P_T3, labels=('Grade 0', 'Grade 1', 'Grade 2', 'Grade 3'))
        plt.title("Dose-Toxicity " + self.scenario_ID)
        plt.xlabel('loP10 Dose')
        plt.ylabel('Graded Toxicity Probability')

    def plot_toxicity_unstacked(self, lowest_dose=0, highest_dose=10, resolution=1001):
        query_doses = np.linspace(lowest_dose, highest_dose, resolution)
        P_T0, P_T1, P_T2, P_T3 = self.toxicity_probability(query_doses)
        plt.figure(figsize=(14, 8))
        plt.margins(x=0, y=0)
        plt.plot(query_doses, P_T0, color='C0', label = 'Grade 0')
        plt.plot(query_doses, P_T1, color='C1', label='Grade 1')
        plt.plot(query_doses, P_T2, color='C2', label='Grade 2')
        plt.plot(query_doses, P_T3, color='C3', label='Grade 3')

        plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
                   ncol=1, fancybox=True, shadow=True)
        plt.title("Dose-Toxicity " + self.scenario_ID)
        plt.xlabel('loP10 Dose')
        plt.ylabel('Graded Toxicity Probability')
        plt.show()

    def plot_toxicity_score(self, lowest_dose=0, highest_dose=10, resolution=1001):
        query_doses = np.linspace(lowest_dose, highest_dose, resolution)
        P_T0, P_T1, P_T2, P_T3 = self.toxicity_probability(query_doses)
        total_tox = self.outcome_scores[1] * P_T1 + self.outcome_scores[2] * P_T2 + self.outcome_scores[3] * P_T3
        plt.plot(query_doses, total_tox)
        plt.title("Dose-Toxicity " + self.scenario_ID)
        plt.xlabel('loP10 Dose')
        plt.ylabel('Toxicity Score')
        plt.show()

    def plot_utility(self, lowest_dose=0, highest_dose=10, resolution=1001):
        query_doses = np.linspace(lowest_dose, highest_dose, resolution)
        utility = self.dose_utility(query_doses)
        plt.plot(query_doses, utility)
        plt.title("Dose-Utility " + self.scenario_ID)
        plt.xlabel('loP10 Dose')

    def plot_overlaid_utility(self, predicted_doses, predicted_responses, same_scales=False, lowest_dose=0, highest_dose=10, resolution=1001):
        query_doses = np.linspace(lowest_dose, highest_dose, resolution)
        utility = self.dose_utility(query_doses)
        plt.plot(query_doses, utility)
        plt.scatter(predicted_doses, predicted_responses, c='red')
        plt.title("Scenario S4, Saturating with Optimised Exploration")
        plt.xlabel('loP10 Dose')
        if same_scales == True:
            plt.xlim([0, 10])
            plt.ylim([-.1, .2])
            plt.yticks(np.arange(-.1, .2, 0.05))
            plt.xticks(np.arange(0, 11, 1))
        plt.show()

    def plot_shaded_utility(self,lowest_dose=0, highest_dose=10, resolution=1001):
        query_doses = np.linspace(lowest_dose, highest_dose, resolution)
        utility = self.dose_utility(query_doses)
        plt.plot(query_doses, utility)
        plt.fill_between(query_doses, 0, utility, where = (utility>-1), alpha = 0.3, interpolate=True, color='red')

        plt.title("95% confidence interval for dose selection, Pure Exploration")
        plt.xlim([0, 10])
        plt.ylim([-.1, .2])
        plt.yticks(np.arange(-.1, .2, 0.05))
        plt.xticks(np.arange(0, 11, 1))
        plt.xlabel('loP10 Dose')
        plt.show()

    def plot_with_random_points(self,lowest_dose=0, highest_dose=10, resolution=1001):
        query_doses = np.linspace(lowest_dose, highest_dose, resolution)
        utility = self.dose_utility(query_doses)
        plt.plot(query_doses, utility)

        doses = np.random.normal(0,0.04,50) + self.optimal_dose
        prediction = self.dose_utility(doses) + np.random.normal(0,0.005,50)

        plt.scatter(doses,prediction, color='red')

        plt.title("High Inaccuracy, High Utility Wastage")
        plt.xlim([0, 10])
        plt.ylim([-.1, .2])
        plt.xlabel('loP10 Dose')
        plt.yticks(np.arange(-.1, .2, 0.05))
        plt.xticks(np.arange(0, 11, 1))
        plt.show()





class Experiment:
    def __init__(self, Scenario_ID, Approach_ID, Predicted_Optimal, Predicted_Optimal_Response, Reached, Total_Utility, Average_Utility):
        self.Scenario_ID = Scenario_ID
        self.Approach_ID = Approach_ID
        self.Predicted_Optimal = Predicted_Optimal
        self.Predicted_Optimal_Response = Predicted_Optimal_Response
        self.Total_Utility = Total_Utility
        self.Average_Utility = Average_Utility
        if Reached == True:
            self.Reached = 1
        elif Reached == False:
            self.Reached = 0
        else:
            self.Reached = Reached

class Approach:
    def __init__(self, Approach_ID, Approach_Type, Trial_Size, Step_Size,
                 Starting_Min = 0, Starting_Max = 0, Softmax = True, Inv_Temp = None,
                 lowest_dose=0, highest_dose=10, resolution=1001,
                 pseudo_weight_run=0.01,  pseudo_weight_final=0.01,
                 Step_Populations = None, Step_Pseudo_Weights = None, Step_Explore_Params = None,
                 Threshold=False, Step_Random = None, Fixed_Doses=None):
        if (Softmax == True) and (Inv_Temp== None):
            print('Error: Please specify inverse temperature or disable softmax')
            return
        self.Approach_ID = Approach_ID
        self.Approach_Type = Approach_Type
        self.Trial_Size = Trial_Size
        self.Inv_Temp = Inv_Temp
        self.Softmax = Softmax
        self.Step_Size = Step_Size
        self.Starting_Min = Starting_Min
        self.Starting_Max =Starting_Max
        self.Lowest_Dose = lowest_dose
        self.Highest_Dose = highest_dose
        self.Resolution = resolution
        self.Pseudo_Weight_Run = pseudo_weight_run
        self.Pseudo_Weight_Final = pseudo_weight_final
        self.Step_Populations = Step_Populations
        self.Step_Pseudo_Weights = Step_Pseudo_Weights
        self.Step_Explore_Params = Step_Explore_Params
        self.Threshold = Threshold
        self.Step_Random = Step_Random
        self.Fixed_Doses = Fixed_Doses
        self.generate_pseudo()
        return

    def generate_pseudo(self):
        doses = np.array([1, 5, 9])
        eff = np.array([10, 50, 90])
        no_eff = np.array([90, 50, 10])
        self.eff_pseudo_x, self.eff_pseudo_y = generate_efficacy_pseudodata(doses, eff, no_eff)

        Doses = np.array([1, 9])
        T0 = np.array([45, 2])
        T1 = np.array([35, 3])
        T2 = np.array([10, 5])
        T3 = np.array([10, 90])

        self.tox_pseudo_x, self.tox_pseudo_y = generate_toxicity_pseudodata(Doses, T0, T1, T2, T3)

