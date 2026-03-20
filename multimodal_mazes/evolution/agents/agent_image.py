import numpy as np

class AgentImage():
    def __init__(self, sensor_noise_scale, genome):
        """
        Initialize the AgentImage
        Arguments:
            genome (Genome): The genome representation of the agent.
            sensor_noise_scale (float): The scale of noise to add to sensor inputs.
        Properties:
            type (str): The type of the agent.
            output (array): The output of the agent policy.
        """
        self.type = "AgentImg"
        self.genome = genome
        self.sensor_noise_scale = sensor_noise_scale
        
    def policy(self, img, img_inputs, processing=True):
        """
        Assign a value to each output by performing a forward pass through the genome.
        Generates:
            outputs (array): The outputs of the agent policy.
        """

        # Add sensor noise
        img_inputs += np.random.normal(loc=0.0, scale=self.sensor_noise_scale, size=img_inputs.shape)

        # Genome forward pass
        outputs = self.genome.forward_pass(img_inputs.reshape(-1))
        
        # One output case - no processing required
        if not processing:
            self.output = [outputs[0]//img.size, outputs[0] % img.size]
            return 
        
        # Argmax - add small random noise to avoid bias
        outputs += np.random.rand(len(outputs)) / 1000
        i = np.argmax(outputs)

        # Map output index to 2D coordinates
        self.output = [i//img.size, i % img.size]