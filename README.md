# Model_Size_Selection

# How to use
This repository does not include the databases that result from simulation, as they are too large. Instead, below are instructions on how to recreate the data and results.
Minor differences may result due to the random elements of the simulation. A seed is used to ensure reproduction is possible, but please ensure that you are using the same version of python and the packages (requires.txt).

Please ensure that you have at least 2GB of storage available if Iterations_Per_Pair = 100

1. Ensure file structure in python directory is the same as in the repository.
2. Run making_database.py to initialise the database.
3. Update config Iterations_Per_Pair to however many repetitions are desired (default 100)
4. Run Simulate_Trials.py. This will take several days if Iterations_Per_Pair = 100, time requirement is O(n).
5. Run plotting_files/plot_objective1.py
6. Run plotting_files/plot_objective2.py
7. Run plotting_files/plot_objective3.py if desired
8. Run plotting_files/plot_overlay.py
9. Run plotting_files/scenarios_plotting.py
10. Run all .py files in copeland 



If desired, new scenarios can be created in scenarios_and_approaches.py. This is done by inserting the following into that document:

scenarioNAME = Scenario('NAME', 
                     'CURVE',
                     np.array([X,..., X]),
                     np.array([Y,Y,Y,Y]),
                     np.array([Z,Z,Z,Z]))
                     
Replace NAME with the desired name.

Replace 'CURVE' with any of 'peaking', 'saturating', 'hill', 'flat', 'biphasic', representing the true efficacy curve shape for the scenario.

Replace [X,..., X] with the parameters for the efficacy curve. These have respectively 3, 3, 2, 1, and 6 parameters.

Replace [Y,Y,Y,Y] with the toxicity model parameters.

Replace [Z,Z,Z,Z] with the utility weights for efficacy, grade 1 AE, grade 2 AE, and grade 3 AE respectively.

Replace 'scenario = scenarioS1' with 'scenario = scenarioNAME' in view_scenario.py to view the scenario.

Append scenarioNAME to scenario_list in 'Simulate_Trials.py' to include the new scenario in the simulation.

Append scenarioNAME to scenarios in the plotting_files .py files to include the new scenario in the visualisation.

Also append 'NAME':'Desired Longform Name' to mapping_names in plot_overlay.py.

Append scenarioNAME to scenarios in the copeland/copeland_1_XX.py files to include the new scenario in the copeland analysis.

Scenarios can also be deleted from the analysis by reversing the above steps. 

A single simulation of a clinical trial can be run and visualised using the relevant files in rerun_single.
The plots shown are
1. Predicted Efficacy (orange) vs True Efficacy (blue) 
2. Predicted Toxicity
3. True Toxicity
4. Predicted Utility (orange) vs True Utility (blue) 
