#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: BOUKHAIRAT Hind
Inspired by bayesrl
"""

from agent import Agent
import numpy as np
from globalvars import *


class ModelBasedAgent(Agent):
    
    def __init__(self, T, **kwargs):
        super(ModelBasedAgent, self).__init__(**kwargs)
        self.T = T

        self.policy_step = self.T # To keep track of where in T-step policy the agent is in; initialized to recompute policy
        self.transition_observations = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.value_table = np.zeros((self.num_states, self.num_actions))
        

    def reset(self):
        super(ModelBasedAgent, self).reset()
        self.policy_step = self.T # To keep track of where in T-step policy the agent is in; initialized to recompute policy
        self.transition_observations.fill(0)
        self.value_table.fill(0)
        
    @classmethod    
    def _value_iteration(cls, transition_probs):
        
        """
        The value iteration algorithm to find the maximum action value

        Parameters
        ----------
        transition_probs : np.ndarray()

        Raises
        ------
        Exception
            indicate the divergence of the algorithm.

        Returns
        -------
        None.

        """

        value_dim = transition_probs.shape[0]
        value = np.zeros(value_dim)
        k = 0
        while True:
            diff = 0
            for s in range(value_dim):
                old = value[s]
                value[s] = np.max(np.sum(transition_probs[s]*(cls.reward[s] +
                           cls.discount_factor*np.array([value,]*cls.num_actions)),
                           axis=1))
                diff = max(0, abs(old - value[s]))
            k += 1
            if diff < EPSILON:
                break
            if k > DIVERGENCE_PARAM:
                raise Exception("Value iteration not converging.")
        for s in range(value_dim):
            cls.value_table[s] = np.sum(transition_probs[s]*(cls.reward[s] +
                   cls.discount_factor*np.array([value,]*cls.num_actions)),
                   axis=1)
        
    
    