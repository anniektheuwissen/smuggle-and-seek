import numpy as np
from more_itertools import powerset

from .strategy import Strategy

"""
Tom0 class: the tom0 strategy
"""
class Tom0(Strategy):
    def __init__(self, agent):
        super().__init__("tom0", agent)
        
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
        Chooses an action associated with zero-order theory of mind reasoning
        """
        if self.print: print(f"possible actions are : {possible_actions}")
        phi = self.calculate_phi(b0, possible_actions, reward_value, costs_vector)
        if self.print: print(f"phi is : {phi}")
        action = self.choose_action_softmax(phi, possible_actions)
        return action