from statistics import mean

import numpy as np

from settings import settings


class DataHolder:
    def __init__(self):
        """
        This class helps to simplify the code inside the test function inside the main script.
        To help you understand the vocabulary used:
         - Test: Episode during which a tested agent will show off its skills to reach a given goal.
         - Evaluation: Series of multiples tests on a goal reaching task. We do one evaluation every n episodes.

        We consider that this data holder is used for many seeds. Once a seed is done, we store its results inside a
        memory.
        """
        self.current_seed = 0

        # Goal reaching accuracies
        self.old_seeds_accuracies = None
        #  '-> It will be a numpy array, and we can't guess how many evaluations will be done in a seed. So it will be
        #  initialised at the end of the first seed
        self.tests_agent_distances = []
        self.evaluations_average_agent_distances = []  # Each element is an average over every tests inside the evaluation.

        # Goal closest node distance
        self.old_seeds_node_distances = None
        self.tests_node_distances = []
        self.evaluations_average_node_distances = []

    def on_evaluation_start(self):
        self.tests_agent_distances.append([])
        self.tests_node_distances.append([])

    def on_test(self, params):
        """
        Called once a test is done
        """
        goal_closest_node_distance, goal_reaching_success = params

        self.tests_agent_distances[-1].append(goal_reaching_success)
        self.tests_node_distances[-1].append(goal_closest_node_distance)

    def on_evaluation_end(self):
        self.evaluations_average_agent_distances.append(mean(self.tests_agent_distances[-1]))
        self.evaluations_average_node_distances.append(mean(self.tests_node_distances[-1]))

    def on_seed_end(self):
        """
        This function is called once a seed of the simulation is done.
        It stores data from the current seed memories, and store it into old seeds data memories.
        """
        if self.old_seeds_accuracies is None:
            nb_evaluations = len(self.evaluations_average_agent_distances)
            self.old_seeds_accuracies = np.zeros((settings.nb_seeds, nb_evaluations))
            self.old_seeds_node_distances = np.zeros((settings.nb_seeds, nb_evaluations))

        self.old_seeds_accuracies[self.current_seed] = np.array(self.evaluations_average_agent_distances)
        self.old_seeds_node_distances[self.current_seed] = np.array(self.evaluations_average_node_distances)

        self.tests_agent_distances = []
        self.evaluations_average_agent_distances = []
        self.tests_node_distances = []
        self.evaluations_average_node_distances = []

        self.current_seed += 1

    def get_accuracy_evolution(self):
        if self.current_seed == 0:
            return np.array(self.evaluations_average_agent_distances), \
                   np.zeros(len(self.evaluations_average_agent_distances))
        old_seeds_data = self.old_seeds_accuracies[:self.current_seed]
        return np.mean(old_seeds_data, axis=0), np.std(old_seeds_data, axis=0)

    def get_node_distances_evolution(self):
        if self.current_seed == 0:
            return np.array(self.evaluations_average_node_distances), \
                   np.zeros(len(self.evaluations_average_node_distances))
        old_seeds_data = self.old_seeds_node_distances[:self.current_seed]
        return np.mean(old_seeds_data, axis=0), np.std(old_seeds_data, axis=0)
