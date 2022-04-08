"""
Creates the experiments table that is used to store simulation results
"""

import sqlite3

conn = sqlite3.connect('Storing_Database.db') # Creates the file if it doesn't exist
c = conn.cursor() 

c.execute("""CREATE TABLE experiments (
            scenario_id str,
            approach_id str,
            pred_optimal real,
            pred_optimal_response real,
            reached int1,
            total_utility real,
            average_utility real
            )""")
conn.commit() 

conn.close()
