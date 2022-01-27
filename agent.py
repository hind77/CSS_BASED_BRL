#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: BOUKHAIRAT Hind
Inspired by bayesrl
"""

import numpy as np

class Agent(object):
    """
        the base class of the secondary user agent

        Parameters
        ----------
        num_states : int
            the number of possible states 2powerN 
        num_actions : int
            number of possible actions
        discount_factor : float
            the discount factor for the RL
           

     """
    
    def __init__(self,num_states, num_actions, discount_factor):

        
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.num_states = num_states
        
        self.last_state = None
        self.last_action = None 
        
    
    def reset(self):
               
        self.last_state = None
        self.last_action = None
        
    
    def act(self, reward, next_state, next_state_is_terminal):
        """
        

        Parameters
        ----------
        reward : int 
            the lenght of the control interval.
        next_state : np.array()
        next_state_is_terminal : boolean
            
        Raises
        ------
        NameError
            Error of the function implimentation 

        Returns
        -------
        None.

        """
        raise NameError("you didnt impliment this function")
