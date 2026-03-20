import numpy as np

class MazeModuleBank:
    def __init__(self):
        """
        Initialise the maze module bank.
        Properties:
            bank (dict): A dictionary containing all modules and their distribution.
            scalar_bank (dict): A dictionary containing scalar modules and their distribution.
        """
        self.bank = {"recurrent": [Recurrent, 0.0], "feedforward": [Feedforward, 1.0]}
        self.scalar_bank = {"recurrent": [Recurrent, 0.0], "feedforward": [Feedforward, 1.0]}

class Module:
    def __init__(self, n_inputs=0, n_outputs=0, layer=None):
        """
        Initialise the module.
        Arguments:
            n_inputs (int): The number of input connections.
            n_outputs (int): The number of output connections.
            layer (int): The layer to which the module belongs.
        Properties:
            module_id (int): The unique identifier for the module.
            tag_inputs (bool): Whether the module tags its inputs.
            previous_output (float): The previous output of the module.
            scalar_out (bool): Whether the module produces scalar outputs.
        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.module_id = 0
        self.layer = layer
        self.tag_inputs = False
        self.previous_output = 0.0
        self.scalar_out = True

    def forward_pass(self, input_vec):
        """
        Perform a forward pass through the module - to be implemented by subclasses.
        Arguments:
            input_vec (np.ndarray): The input vector to the module.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def reset(self):
        """
        Reset the module state.
        Resets:
            previous_output (float): The previous output of the module.
        """
        self.previous_output = 0.0


class Feedforward(Module):
    def __init__(self, inputs=2, outputs=1, layer=None):
        """
        Initialise the feedforward module.
        Arguments:
            n_inputs (int): The number of input connections.
            n_outputs (int): The number of output connections.
            layer (int): The layer to which the module belongs.
        Properties:
            type (str): The type of the module.
            module_id (int): The unique identifier for the module.
            tag_inputs (bool): Whether the module tags its inputs.
            scalar_out (bool): Whether the module produces scalar outputs.
        """
        super().__init__(inputs, outputs, layer)
        self.module_id = 0
        self.type = 'feedforward'

    def forward_pass(self, input_vector):
        """
        Forward pass through the feedforward module.
        Arguments:
            input_vector (np.ndarray): The input vector to the module.
        Returns:
            np.ndarray: The output vector from the module.
        """
        # Connection weights
        w_forward = 0.5
        
        # Feedforward - apply weights and activation
        output_vector = np.maximum(0, (input_vector[0] * w_forward + input_vector[1] * w_forward))
        return np.array([output_vector])


class Recurrent(Module):
    def __init__(self, inputs=2, outputs=1, layer=None):
        """
        Initialise the recurrent module.
        Arguments:
            n_inputs (int): The number of input connections.
            n_outputs (int): The number of output connections.
            layer (int): The layer to which the module belongs.
        Properties:
            type (str): The type of the module.
            module_id (int): The unique identifier for the module.
            tag_inputs (bool): Whether the module tags its inputs.
            scalar_out (bool): Whether the module produces scalar outputs.
        """
        super().__init__(inputs, outputs, layer)
        self.module_id = 1
        self.type = 'recurrent'

    def forward_pass(self, input_vector):
        """
        Forward pass through the recurrent module.
        Arguments:
            input_vector (np.ndarray): The input vector to the module.
        Returns:
            np.ndarray: The output vector from the module.
        """
        # Connection weights
        w_forward = 0.5
        w_recurrent = 0.9

        # Recurrent - apply weights and activation and update previous_output
        output_vector = np.maximum(0, (input_vector[0] * w_forward + input_vector[1] * w_forward + self.previous_output * w_recurrent))
        self.previous_output = output_vector
        return np.array([output_vector])