import numpy as np

from .strategy import Strategy

"""
Tom0 class: the tom0 strategy
"""
class Tom0(Strategy):
    def __init__(self, agent):
        """
        Initializes the ToM0 strategy
        """
        super().__init__("tom0", agent)
        
    def calculate_phi(self, b0, possible_actions, reward_value, costs_vector, expected_amount_catch):
        """
        Calculates the subjective value phi of all possible actions and all their distributions
        :param b0: The zero-order beliefs
        :param possible_actions: The possible actions that the agent can take
        :param reward_value: The reward that the agent receives when finding a packages
        :param costs_vector: The vector with for all containers the costs of using that container
        :param expected_amount_catch: The expected amount of packages catched in one catch
        """
        phi = np.zeros(len(possible_actions))

        for (idx,aa) in enumerate(possible_actions):
            if self.agent == "customs": phi[idx] = reward_value * np.dot(aa, b0) - np.dot(costs_vector,[int(c>0) for c in aa])
            elif self.agent == "smuggler": phi[idx] = reward_value * (max(possible_actions[0]) - np.dot(aa, b0)) - np.dot(costs_vector,[int(c>0) for c in aa])

        return phi
    
    def choose_action_softmax(self, phi, possible_actions):
        """
        Chooses an action to play based on the softmax over the subjective value phi
        :param phi: The phi values
        :param possible_actions: The possible action that the agent can take
        """
        softmax_phi = np.exp(phi) / np.sum(np.exp(phi))
        if self.print: print(f"softmax of phi is : {softmax_phi}")
        chosen_actionindex = np.random.choice([i for i in range(0,len(phi))], 1, p=softmax_phi)[0]
        action = possible_actions[chosen_actionindex]

        return action
    
    def choose_action(self, possible_actions, b0, b1, b2, conf1, conf2, reward_value, costs_vector, expected_amount_catch, simulation_rewardo, simulation_rewarda):
        """
        Chooses an action associated with zero-order theory of mind reasoning
        :param possible_actions: The possible actions that the agent can take
        :param b0: The zero-order beliefs
        :param b1: The first-order beliefs
        :param b2: The second-order beliefs
        :param conf1: The confidence in that first-order theory of mind is accurate
        :param conf2: The confidence in that second-order theory of mind is accurate
        :param reward_value: The reward that the agent receives when finding a packages
        :param costs_vector: The vector with for all containers the costs of using that container
        :param expected_amount_catch: The expected amount of packages catched in one catch
        :param simulation_rewardo: The simulated reward for the opponent agent
        :param simulation_rewarda: The simulated reward for the agent itself
        """
        if self.print: print(f"possible actions are : {possible_actions}")
        phi = self.calculate_phi(b0, possible_actions, reward_value, costs_vector, expected_amount_catch)
        if self.print: print(f"phi is : {phi}")
        action = self.choose_action_softmax(phi, possible_actions)
        return action