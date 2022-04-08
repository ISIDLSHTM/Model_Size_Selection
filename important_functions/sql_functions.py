"""
Functions used for storing the data from experiments into the Database
"""

def insert_experiment(exp, connection):
    with connection:
        c = connection.cursor()
        c.execute("""INSERT INTO experiments VALUES(
        :scenario_id,
        :approach_id,
        :pred_optimal,
        :pred_optimal_response,
        :reached,
        :total_utility,
        :average_utility)""", {'scenario_id': exp.Scenario_ID,
                               'approach_id': exp.Approach_ID,
                               'pred_optimal': exp.Predicted_Optimal,
                               'pred_optimal_response': exp.Predicted_Optimal_Response,
                               'reached': exp.Reached,
                               'total_utility': exp.Total_Utility,
                               'average_utility': exp.Average_Utility})


def find_most_recent_experiment(connection):
    with connection:
        c = connection.cursor()
        c.execute('SELECT max(rowid) FROM experiments')
        most_recent_id = c.fetchone()[0]
    return most_recent_id

def delete_experiment_by_id(id, connection):
    with connection:
        c = connection.cursor()
        c.execute("""DELETE FROM experiments WHERE rowid=:id""", {'id': id})

def delete_experiment_by_approach(approach_id, connection):
    with connection:
        c = connection.cursor()
        c.execute("""DELETE FROM experiments WHERE approach_id=:approach_id""", {'approach_id': approach_id})


