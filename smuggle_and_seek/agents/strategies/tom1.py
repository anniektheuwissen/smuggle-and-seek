import numpy as np

from .strategy import Strategy

"""
Tom1 class: the tom1 strategy
"""
class Tom1(Strategy):
    def __init__(self, agent):
        """
        Initializes the ToM1 strategy
        """
        super().__init__("tom1", agent)

        self.prediction_a1 = []
        self.W1 = []
        
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
            if self.agent == "customs": phi[idx] = reward_value * expected_amount_catch * np.dot(aa, b0) - np.dot(costs_vector,[int(c>0) for c in aa])
            elif self.agent == "smuggler": phi[idx] = reward_value * (max(possible_actions[0]) - np.dot(aa, b0)) - np.dot(costs_vector,[int(c>0) for c in aa])

        return phi
    
    def calculate_simulation_phi(self, belief, simulated_reward, sim_agent):
        """
        Calculates the subjective value phi of the opponent by simulating the phi function of the opponent
        :param b1: The first-order beliefs
        :param simulated_reward: The simulated reward
        :param sim_agent: The agent of which the phi values are simulated
        """
        simulation_phi = np.zeros(len(belief))
        for c in range(len(belief)):
            aa = [0]*len(belief); aa[c] = 1
            if (self.agent == "smuggler" and sim_agent == "self") or (self.agent == "customs" and sim_agent == "other"): simulation_phi[c] = (sum(aa) - np.dot(aa, belief)) * simulated_reward[c][0] - np.dot(aa, belief) * simulated_reward[c][0] - simulated_reward[c][1]
            elif (self.agent == "customs" and sim_agent == "self") or (self.agent == "smuggler" and sim_agent == "other"): simulation_phi[c] = np.dot(aa, belief) * simulated_reward[c][0] - (sum(belief) - np.dot(aa, belief)) * simulated_reward[c][0] 
        return simulation_phi

    def merge_prediction(self, prediction, belief, confidence):
        """
        Merges prediction with its belief b0
        :param prediction: The prediction made 
        :param belief: The belief
        :param confidence: The confidence in that the prediction is correct
        """
        W = np.zeros(len(belief))
        for c in range(len(belief)):
            prediction[c] *= np.sum(belief)
            W[c] = confidence * prediction[c] + (1-confidence) * belief[c]
        return W
    
    def choose_action_softmax(self, phi, possible_actions):
        """
        Chooses an action to play based on the softmax over the subjective value phi
        :param phi: The phi values
        :param possible_actions: The possible action that the agent can take
        """
        softmax_phi = self.softmax(phi)
        if self.print: print(f"softmax of phi is : {softmax_phi}")
        chosen_actionindex = np.random.choice([i for i in range(0,len(phi))], 1, p=softmax_phi)[0]
        action = possible_actions[chosen_actionindex]

        return action
    
    def choose_action(self, possible_actions, b0, b1, b2, conf1, conf2, reward_value, costs_vector, expected_amount_catch, simulation_rewardo, simulation_rewarda):
        """
        Chooses an action associated with first-order theory of mind reasoning
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
        # Make prediction about behavior of opponent
        simulation_phi = self.calculate_simulation_phi(b1, simulation_rewardo, "other")
        if self.print: print(f"simulation phi is : {simulation_phi}")   
        self.prediction_a1 = self.softmax(simulation_phi)
        if self.print: print(f"prediction a1 is : {self.prediction_a1}")

        # Merge prediction with zero-order belief
        self.W1 = self.merge_prediction(self.prediction_a1, b0, conf1)
        if self.print: print(f"W is : {self.W1}")

        # Choose action based on W
        phi = self.calculate_phi(self.W1, possible_actions, reward_value, costs_vector, expected_amount_catch)
        if self.print: print(f"phi is : {phi}")
        action = self.choose_action_softmax(phi, possible_actions)
        return action