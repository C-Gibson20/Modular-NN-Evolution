import numpy as np
from multimodal_mazes.agents.agent import Agent


class AgentGenome(Agent):
    def __init__(self, location, channels, sensor_noise_scale, genome):
        """
        Initialize the AgentGenome
        Arguments:
            location (array): The location of the agent [r, c].
            channels (list): The list of input channels.
            genome (Genome): The genome representation of the agent.
            sensor_noise_scale (float): The scale of noise to add to sensor inputs.
        Properties:
            type (str): The type of the agent.
            outputs (array): The outputs of the agent policy.
        """
        super().__init__(location, channels)
        self.type = "AgentGenome"
        self.genome = genome
        self.sensor_noise_scale = sensor_noise_scale
        
    def policy(self):
        """
        Assign a value to each action by performing a forward pass through the grammar tree.
        Generates:
            outputs (array): The outputs of the agent policy.
        """
        self.outputs = self.genome.forward_pass(self.channel_inputs.reshape(-1))
        self.outputs += np.random.rand(len(self.outputs)) / 1000
        
