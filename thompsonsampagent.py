#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: BOUKHAIRAT Hind
Inspired by bayesrl
"""

from modelbasedagent import ModelBasedAgent
import numpy as np

class ThompsonSampAgent(ModelBasedAgent):
    """
        the thompson sampling algorithm for the best action selection a,ong the beliefs 

        Parameters
        ----------
        dirichlet_param : float
        reward_param : int
        **kwargs 

        """    
    
    def __init__(self, dirichlet_param, reward_param, **kwargs):

        super(ThompsonSampAgent, self).__init__(**kwargs)
        self.dirichlet_param = dirichlet_param
        self.reward_param = reward_param
        self.reward = np.full((self.num_states, self.num_actions, self.num_states), self.reward_param)

    def reset(self):
        super(ThompsonSampAgent, self).reset()
        self.reward.fill(self.reward_param)
        
    @classmethod    
    def act(cls, reward, next_state, next_state_is_terminal, idx)-> np.ndarray():
        """
        the implimentation of the act function 

        Parameters
        ----------
        reward : int 
            the lenght of the control interval.
        next_state : np.array()
        next_state_is_terminal : boolean
        idx : int
            The episode index.

        Returns
        -------
        np.array()
            the choosen action as the last action.

        """
        # Handle start of episode.
        if reward is None:
            # Return random action since there is no information.
            next_action = np.random.randint(cls.num_actions)
            cls.last_state = next_state
            cls.last_action = next_action
            return cls.last_action

        # Handle completion of episode.
        if next_state_is_terminal:
            # Proceed as normal.
            pass

        # Update the reward associated with (s,a,s') if first time.
        if cls.reward[cls.last_state, cls.last_action, next_state] == cls.reward_param:
            cls.reward[cls.last_state, cls.last_action, next_state] = reward

        # Update set of states reached by playing a.
        cls.transition_observations[cls.last_state, cls.last_action, next_state] += 1

        # Update transition probabilities after every T steps
        if cls.policy_step == cls.T:
            cls.__compute_policy()

        # Choose next action according to policy.
        next_action = cls._argmax_breaking_ties_randomly(cls.value_table[next_state])

        cls.policy_step += 1
        cls.last_state = next_state
        cls.last_action = next_action

        return cls.last_action
    
    def __compute_policy(self):
        """
        Compute an optimal T-step policy for the current state.

        Returns
        -------
        None.

        """
        self.policy_step = 0
        transition_probs = np.zeros((self.num_states, self.num_actions, self.num_states))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                transition_probs[s,a] = np.random.dirichlet(self.transition_observations[s,a] +\
                                                            self.dirichlet_param, size=1)
        self._value_iteration(transition_probs)