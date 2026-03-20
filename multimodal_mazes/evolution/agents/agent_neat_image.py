# Evolved network agents

import numpy as np
import neat
import copy


class AgentNeatImage():
    def __init__(self, sensor_noise_scale, drop_connect_p, genome, config):
        """
        Creates a NEAT agent. 
        Arguments:
            location (array): initial position [r,c].
            channels (list): list of active (1) and inative (0) channels e.g. [0,1].
            genome (Genome): neat generated genome.
            sensor_noise_scale (float): the scale of the noise applied to every sensor. 
            drop_connect_p (float): the probability of edge drop out, per time step. 
            config (Config): the neat configuration holder.
        Properties:
            type (str): The type of the agent.
            net (RecurrentNetwork): The neural network representation of the agent.
            netmemory (list): The memory of the neural network.
        """
        self.type = "AgentNeatImage"
        self.genome = genome
        self.sensor_noise_scale = sensor_noise_scale
        self.drop_connect_p = drop_connect_p
        self.config = config
        self.net = neat.nn.RecurrentNetwork.create(genome, config)
        self.netmemory = copy.deepcopy(self.net.node_evals)

    def policy(self, img, img_inputs, processing=True):
        """
        AgentNeat policy is a forward pass through an evolved neural network.
        Generates:
            outputs (array): The outputs of the agent policy.
        """

        # Drop connect
        self.net.node_evals = copy.deepcopy(self.netmemory)
        if self.drop_connect_p > 0.0:
            for a, _ in enumerate(self.net.node_evals):
                for b, _ in enumerate(self.net.node_evals[a][-1]):
                    if self.drop_connect_p > np.random.sample():
                        self.net.node_evals[a][-1][b] = (self.net.node_evals[a][-1][b][0], 0.0)

        # Add sensor noise
        img_inputs += np.random.normal(0, self.sensor_noise_scale, img_inputs.shape)

        # NEAT forward pass
        outputs = self.net.activate(list(img_inputs.reshape(-1)))

        # One output case - no processing required
        if not processing:
            self.output = [outputs[0] // img.size, outputs[0] % img.size]
            return

        # Argmax - add small random noise to avoid bias
        outputs += np.random.rand(len(outputs)) / 1000
        i = np.argmax(outputs)
        self.output = [i // img.size, i % img.size]
