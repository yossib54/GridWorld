import random
from typing import Any
import numpy as np
from Graphics import *
from Action import Action
from Environement import Environement as Env


class Random_Agent:
    def __init__(self, env) -> None:
        self.env : Env = env
        self.Reward = 0
    
    def get_action(self, state):
        actions = self.env.get_actions(state)
        return random.choice(actions)
            
    def add_reward(self, reward):
        self.Reward += reward

    def __call__(self, state):
        return self.get_action(state)
    
class AI_Agent:
    def __init__(self, env) -> None:
        self.env : Env = env
        self.Reward = 0
        self.Policy = np.full((ROWS, COLS), 3)  # Random policy always down
        self.Value = np.zeros((ROWS, COLS))     # Random Value all zero
        self.gamma = 0.9

    def get_action(self, state):
        return Action(self.Policy[state])
            
    def add_reward(self, reward):
        self.Reward += reward

    def policy_eval(self):
        accuracy = 0.0001

        while True:
            delta = 0
            for row in range(ROWS):
                for col in range(COLS):
                    state = (row, col)
                    if self.env.end_of_game(state):
                        continue
                    v = self.Value[state]
                    action = Action(self.Policy[state])
                    new_state, reward = self.env.move(state, action)
                    self.Value[state] = reward + self.gamma * self.Value[new_state]
                    delta = max(delta, abs(v - self.Value[state]))
            if delta < accuracy:
                break


    def Policy_improv(self):
        stable = True
        for row in range(ROWS):
            for col in range(COLS):
                state = (row, col)
                if self.env.end_of_game(state):
                    continue
                old_action = self.Policy[state]
                best_action = old_action
                best_value = float('-inf')
                for action in Action:
                    new_state, reward = self.env.move(state, action)
                    value = reward + self.gamma * self.Value[new_state]
                    if value > best_value:
                        best_value = value
                        best_action = action.value
                self.Policy[state] = best_action
                if best_action != old_action:
                    stable = False
        return stable

    def Policy_Iteration(self):
        while True:
            self.policy_eval()
            stable = self.Policy_improv()
            if stable:
                break

    
    def Value_Iteration(self):
        pass

    def __call__(self, state):
        return self.get_action(state)
    

    