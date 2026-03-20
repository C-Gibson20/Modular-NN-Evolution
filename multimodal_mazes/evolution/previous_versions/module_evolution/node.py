import numpy as np

class Node():
    def __init__(self, activation, layer=-1):
        """
        Initialise the Node with an activation function and layer.
        Arguments:
            activation (str): The activation function to use ('linear' or 'relu').
            layer (int): The layer the node belongs to (-1 for input, 0-n for hidden, -1 for output).
        Properties:
            previous_output (np.ndarray): The output of the node from the previous forward pass.
        """
        self.activation = activation
        self.previous_output = None
        self.layer = layer

    def activation_fn(self, inputs):
        """
        Apply the activation function to the inputs.
        Arguments:
            inputs (np.ndarray): The inputs to the activation function.
        Returns:
            np.ndarray: The output of the activation function.
        """
        match self.activation:
            case 'linear':
                return inputs

            case 'relu':
                return np.maximum(0, inputs)

    def forward_pass(self, inputs):
        """Perform a forward pass through the node.
        Arguments:
            inputs (np.ndarray): Input value to the node.
        Returns: 
            The output of the node after applying the activation function.
        """
        return self.activation_fn(inputs)
    
    def reset(self):
        """Reset the node's state."""
        self.previous_output = None