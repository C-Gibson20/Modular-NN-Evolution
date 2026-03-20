import numpy as np

class Module():
    def __init__(self, inputs=2, outputs=1, layer=None):
        """
        Initialize the module with the given parameters.
        Arguments:
            inputs: The number of input connections to the module.
            outputs: The number of output connections from the module.
            layer: The layer to which the module belongs.
        Properties:
            module_id: The unique identifier for the module.
            previous_output: The output from the previous forward pass.
        """
        self.module_id = 0
        self.n_inputs = inputs
        self.n_outputs = outputs
        self.previous_output = 0.0
        self.layer = layer

    def forward_pass(self, input_vector):
        """Perform a forward pass through the module - to be overwritten"""
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def reset(self):
        """
        Reset the module's state.
        Resets:
            previous_output: The output from the previous forward pass.
        """
        self.previous_output = 0.0
    
class RecurrentModule(Module):
    def __init__(self, inputs=2, outputs=1, layer=None):
        """
        Initialize the recurrent module.
        """
        super().__init__(inputs, outputs, layer)
        self.module_id = 1
        self.type = 'recurrent'

    def forward_pass(self, input_vector):
        """Forward pass through the recurrent module."""
        
        # Weights
        w_forward = 0.5
        w_recurrent = 0.9

        # Output node
        output_vector = np.maximum(0, (input_vector[0] * w_forward + input_vector[1] * w_forward + self.previous_output * w_recurrent))
        self.previous_output = output_vector

        return np.array([output_vector])
    
class FeedforwardModule(Module):
    def __init__(self, inputs=2, outputs=1, layer=None):
        super().__init__(inputs, outputs, layer)
        self.module_id = 0
        self.type = 'feedforward'

    def forward_pass(self, input_vector):
        """Forward pass through the feedforward module."""
        # Weights
        w_forward = 0.5

        # Output node
        output_vector = np.maximum(0, (input_vector[0] * w_forward + input_vector[1] * w_forward))

        return np.array([output_vector])

class SquareClassificationModule(Module):
    def __init__(self, inputs=9, outputs=9, layer=None):
        super().__init__(inputs, outputs, layer)
        self.module_id = 2
        self.type = 'square_classification'

    def forward_pass(self, input_vector):
        """Forward pass through the square classification module."""
        input_vector = np.array(input_vector).reshape((3, 3))
        output_vector = np.zeros((3, 3))
        contrib_map = {-1: 0, 0:-1}

        rows, cols = input_vector.shape
        for i in range(rows):
            for j in range(cols):
                neighbours = [
                    input_vector[i-1][j] if i > 0 else -1,       
                    input_vector[i][j+1] if j < cols - 1 else -1, 
                    input_vector[i+1][j] if i < rows - 1 else -1, 
                    input_vector[i][j-1] if j > 0 else -1        
                ]

                contribution = sum(contrib_map.get(val, val) for val in neighbours)
                output_vector[i][j] = input_vector[i][j] + contribution
        return output_vector.flatten()
    
class CrossClassificationModule(Module):
    def __init__(self, inputs=9, outputs=9, layer=None):
        super().__init__(inputs, outputs, layer)
        self.module_id = 3
        self.type = 'cross_classification'

    def forward_pass(self, input_vector):
        """Forward pass through the cross classification module."""
        input_vector = np.array(input_vector).reshape((3, 3))
        output_vector = np.zeros((3, 3))
        contrib_map = {-1: 0, 0:-1}

        rows, cols = input_vector.shape
        for i in range(rows):
            for j in range(cols):
                neighbours = [
                    input_vector[i-1][j-1] if i > 0 and j > 0 else -1,
                    input_vector[i+1][j+1] if i < rows - 1 and j < cols - 1 else -1,
                    input_vector[i+1][j-1] if i < rows - 1 and j > 0 else -1,
                    input_vector[i-1][j+1] if i > 0 and j < cols - 1 else -1
                ]
                contribution = sum(contrib_map.get(val, val) for val in neighbours)
                output_vector[i][j] = input_vector[i][j] + contribution

                neighbours = [
                    input_vector[i-1][j] if i > 0 else -1,
                    input_vector[i][j+1] if j < cols - 1 else -1,
                    input_vector[i+1][j] if i < rows - 1 else -1,
                    input_vector[i][j-1] if j > 0 else -1
                ]
                contribution = sum(contrib_map.get(val, val) for val in neighbours)
                output_vector[i][j] -= contribution

        return output_vector.flatten()
