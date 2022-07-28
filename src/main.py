import sys
from datetime import datetime
"""

print("Available simulations are:")
print(" - Simulation 1: Neural networks failure to approximate discontinuous value functions.")
print(" - Simulation 2: Low-level policy pre-training convergence demonstration.")
print(" - Simulation 3: Full algorithm simulation in complex environment.")
print(" - Simulation 4: Full algorithm simulation in complex environment and comparison with others approaches.")

nb_trials = 4
simulation_id_higher_bound = 5

for x in range(nb_trials):
    simulation_id = input("Chose the simulation to run by selecting a value from 1 to 4:")

    # Verify type
    try:
        simulation_id = int(simulation_id)
    except:
        print("The selected value should be an integer. Try again.")
        continue

    # Verify value
    try:
        assert 0 < simulation_id < simulation_id_higher_bound
    except:
        print("The selected value should be greater than 0 and lower than " + str(simulation_id_higher_bound)
              + ". Try again.")
        continue
    break

else:
    print("You tried too many times. We failed launching a simulation")
    sys.exit()

print(" You selected simulation ", simulation_id, ".")
"""
simulation_id = 4

simulation_start_time = datetime.now()

if simulation_id == 1:
    from src.simulations import pre_training_convergence
elif simulation_id == 2:
    from src.simulations import value_function_over_wall
elif simulation_id == 3:
    from src.simulations import model_free_topology_learning
elif simulation_id == 4:
    from src.simulations import model_free_topology_learning_point_maze
elif simulation_id == 5:
    from src.simulations import her_ant_maze
    # from src.simulations import model_free_topology_learning_ant_maze

simulation_end_time = datetime.now()
duration = simulation_end_time - simulation_start_time
print("simulation duration: ", duration.seconds, "seconds.")
