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
            if len(possible_actions) == (2**len(b0) - 1):
                phi[idx] = reward_value * expected_amount_catch * np.dot(aa, b0) - np.dot(costs_vector,[int(c>0) for c in aa])
            else: phi[idx] = reward_value * (max(possible_actions[0]) - np.dot(aa, b0)) - np.dot(costs_vector,[int(c>0) for c in aa])

        return phi
    
    def calculate_simulation_phi(self, b1, simulated_reward, order):
        """
        Calculates the subjective value phi of the opponent by simulating the phi function of the opponent
        :param b1: The first-order beliefs
        :param simulated_reward: The simulated reward
        :param order: The order of theory of mind at which the agent reasons
        """
        simulation_phi = np.zeros(len(b1))
        for c in range(len(b1)):
            aa = [0]*len(b1); aa[c] = 1
            if (self.agent == "smuggler" and order%2 == 1) or (self.agent == "customs" and order%2 == 0) :sumas = sum(b1)
            else: sumas = sum(aa)
            simulation_phi[c] = np.dot(aa, b1) * simulated_reward[c][1] + (sumas - np.dot(aa, b1)) * simulated_reward[c][0]
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
        softmax_phi = np.exp(phi) / np.sum(np.exp(phi))
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
        simulation_phi = self.calculate_simulation_phi(b1, simulation_rewardo, 1)
        if self.print: print(f"simulation phi is : {simulation_phi}")   
        self.prediction_a1 = np.exp(simulation_phi) / np.sum(np.exp(simulation_phi))
        if self.print: print(f"prediction a1 is : {self.prediction_a1}")

        # Merge prediction with zero-order belief
        W = self.merge_prediction(self.prediction_a1, b0, conf1)
        if self.print: print(f"W is : {W}")

        # Choose action based on W
        phi = self.calculate_phi(W, possible_actions, reward_value, costs_vector, expected_amount_catch)
        if self.print: print(f"phi is : {phi}")
        action = self.choose_action_softmax(phi, possible_actions)
        return action