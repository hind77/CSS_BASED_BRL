#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: BOUKHAIRAT Hind
Inspired by bayesrl

"""

from thompsonsampagent import ThompsonSampAgent
import numpy as np

class POMDP(ThompsonSampAgent):
    
    
    def __init__(self, observation_model, dirichlet_param, reward_param, **kwargs):
        super(POMDP, self).__init__(dirichlet_param, reward_param, **kwargs)
        self.observation_model = observation_model
        self.reset_belief()
        self.__compute_policy()
        
    def reset_belief(self):
        self.belief = np.array([1./self.num_states for _ in range(self.num_states)])

    def reset(self):
        super(POMDP, self).reset()
        self.reset_belief()
        
    @classmethod    
    def act(cls, reward, observation, next_state_is_terminal, idx):
        """
        

        Parameters
        ----------
        reward : int 
            the lenght of the control interval.
        observation : np.array()
                the occupency states of sensed channels
        next_state : np.array()
        next_state_is_terminal : boolean
        idx : int
            The episode index.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # Handle start of episode.
        if reward is None:
            # Return random action since there is no information.
            next_action = np.random.randint(cls.num_actions)
            cls.last_action = next_action
            cls.__observe(observation)
            return cls.last_action

        # Handle completion of episode.
        if next_state_is_terminal:
            # Proceed as normal.
            pass

        for last_state,next_state in [(s,s_) for s in range(cls.num_states) for s_ in range(cls.num_states)]:
            tp = cls.belief[last_state]*cls.transition_probs[last_state,cls.last_action,next_state]
            # Update the reward associated with (s,a,s') if first time.
            #if self.reward[last_state, self.last_action, next_state] == self.reward_param:
            cls.reward[last_state, cls.last_action, next_state] *= (1-tp)
            cls.reward[last_state, cls.last_action, next_state] += reward*tp

            # Update set of states reached by playing a.
            cls.transition_observations[last_state, cls.last_action, next_state] += tp

        # Update transition probabilities after every T steps
        if cls.policy_step == cls.T:
            cls.__compute_policy()

        cls.__update_belief(cls.last_action,observation)
        # Choose next action according to policy.
        value_table = sum(cls.belief[s]*cls.value_table[s] for s in range(cls.num_states))
        next_action = cls._argmax_breaking_ties_randomly(value_table)

        cls.policy_step += 1
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
        self.transition_probs = np.zeros((self.num_states, self.num_actions, self.num_states))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.transition_probs[s,a] = np.random.dirichlet(self.transition_observations[s,a] +\
                                                            self.dirichlet_param, size=1)
        self._value_iteration(self.transition_probs)