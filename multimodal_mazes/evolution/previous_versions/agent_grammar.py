import numpy as np
from multimodal_mazes.agents.agent import Agent


class Agent_Grammar(Agent):
    def __init__(self, location, channels, sensor_noise_scale, grammar_tree):
        super().__init__(location, channels)
        self.type = "AgentGrammar"
        self.sensor_noise_scale = sensor_noise_scale
        self.grammar_tree = grammar_tree
        
    def policy(self):
        """Assign a value to each action by performing a forward pass through the grammar tree."""
        self.outputs = self.grammar_tree.forward_pass(self.channel_inputs.reshape(-1))
        self.outputs += np.random.rand(len(self.outputs)) / 1000
