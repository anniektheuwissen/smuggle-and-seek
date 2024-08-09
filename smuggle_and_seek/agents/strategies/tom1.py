import numpy as np
from more_itertools import powerset

from .strategy import Strategy

"""
Tom1 class: the tom1 strategy
"""
class Tom1(Strategy):
    def __init__(self, agent):
        super().__init__("tom1", agent)

        self.prediction_a1 = []
        
    def calculate_phi(self, b0, possible_actions, reward_value, costs_vector):
        """
        Calculates the subjective value phi of all possible actions and all their distributions
        :param beliefs: The beliefs based on which the phi values have to be calculated
        """
        phi = np.zeros(len(possible_actions))

        for (idx,aa) in enumerate(possible_actions):
            # for c in range(len(b0)):
            #     ao = [0] * len(b0); ao[c] = 1
            #     if len(possible_actions) == (2**len(b0) - 1):
            #         phi[idx] += b0[c] * (reward_value * np.dot(aa, ao) - np.dot(costs_vector,[int(c>0) for c in aa]))
            #     else: phi[idx] += b0[c] * (reward_value * (max(possible_actions[0]) - np.dot(aa, ao)) - np.dot(costs_vector,[int(c>0) for c in aa])) #MAX IS PACKAGES MOET MOOIER DAN DIT
            # ANDERE METHODE (MAKKELIJKER OP TE SCHRIJVEN), OM DIT WERKEND TE KRIJGEN ALLE PREDICTIONS GESCHAALD NAAR INITIAL BELIEFS
            if len(possible_actions) == (2**len(b0) - 1):
                phi[idx] = reward_value * np.dot(aa, b0) - np.dot(costs_vector,[int(c>0) for c in aa])
            else: phi[idx] = reward_value * (max(possible_actions[0]) - np.dot(aa, b0)) - np.dot(costs_vector,[int(c>0) for c in aa])

        return phi
    
    def calculate_simulation_phi(self, b1, simulated_reward, order):
        """
        Calculates the subjective value phi of the opponent by using beliefs b1 and a simulated reward function
        """    
        simulation_phi = np.zeros(len(b1))
        for c in range(len(b1)):
            aa = [0]*len(b1); aa[c] = 1
            if (self.agent == "smuggler" and order%2 == 1) or (self.agent == "police" and order%2 == 0) :sumas = sum(b1)
            else: sumas = sum(aa)
            simulation_phi[c] = np.dot(aa, b1) * simulated_reward[c][1] + (sumas - np.dot(aa, b1)) * simulated_reward[c][0]
        return simulation_phi


    def merge_prediction(self, prediction, belief, confidence):
        """
        Merges prediction with its belief b0
        """
        W = np.zeros(len(belief))
        for c in range(len(belief)):
            W[c] = confidence * sum(belief) * prediction[c] + (1-confidence) * belief[c]
        return W
    
    def choose_action_softmax(self, phi, possible_actions):
        """
        Chooses an action to play based on the softmax over the subjective value phi
        """
        softmax_phi = np.exp(phi) / np.sum(np.exp(phi))
        if self.print: print(f"softmax of phi is : {softmax_phi}")
        chosen_actionindex = np.random.choice([i for i in range(0,len(phi))], 1, p=softmax_phi)[0]
        action = possible_actions[chosen_actionindex]

        return action
    
    def choose_action(self, possible_actions, b0, b1, b2, conf1, conf2, reward_value, costs_vector, simulation_rewardo, simulation_rewarda):
        """
        Chooses an action associated with first-order theory of mind reasoning
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
        phi = self.calculate_phi(W, possible_actions, reward_value, costs_vector)
        if self.print: print(f"phi is : {phi}")
        action = self.choose_action_softmax(phi, possible_actions)
        return action