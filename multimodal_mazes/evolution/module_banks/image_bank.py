import numpy as np
from scipy.signal import convolve2d, correlate2d

class ImageModuleBank:
    def __init__(self, img_type='square'):
        """
        Initialise the image module bank.
        Arguments:
            img_type: The type of image (e.g., 'square', 'cross').
        Properties:
            scalar_bank: A dictionary containing scalar modules and their distribution.
            bank: A dictionary containing all modules and their distribution.
        """
        self.img_type = img_type
        self.scalar_bank = {"feedforward": [Feedforward, 1.0], "argmax": [Argmax, 0.0]}
        self.bank = {}
        self.initialise_module_bank()

    def initialise_module_bank(self):
        """
        Initialise the module bank based on the image type.
        Updates:
            bank: A dictionary containing all modules and their distribution.
        """
        if self.img_type == 'square':
            self.bank = {"bus": [Bus, 0.0], "convolution": [Convolution, 0.0], "feedforward": [Feedforward, 1.0], "argmax": [Argmax, 0.0]}
        elif self.img_type == 'cross':
            self.bank = {"bus": [Bus, 0.0], "correlation": [Correlation, 0.0], "feedforward": [Feedforward, 1.0], "argmax": [Argmax, 0.0]}


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
            scalar_out (bool): Whether the module produces scalar outputs.
        """
        self.module_id = 0
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.layer = layer
        self.tag_inputs = False
        self.scalar_out = True

    def forward_pass(self, input_vec):
        """Perform a forward pass through the module - to be implemented by subclasses."""
        raise NotImplementedError("This method should be overridden by subclasses")

    def reset(self):
        """Reset the module state - to be implemented by subclasses if needed."""
        return


class Bus(Module):
    def __init__(self, layer=None):
        """
        Initialise the bus module.
        Arguments:
            n_inputs (int): The number of input connections.
            n_outputs (int): The number of output connections.
            layer (int): The layer to which the module belongs.
        Properties:
            type (str): The type of the module.
            module_id (int): The unique identifier for the module.
            tag_inputs (bool): Whether the module tags its inputs.
            input_map (list): The mapping of input connections.
            scalar_out (bool): Whether the module produces scalar outputs.
        """
        super().__init__(n_inputs=81, n_outputs=81, layer=layer)
        self.type = "bus"
        self.tag_inputs = True
        self.input_map = list(np.full(self.n_inputs, -1, dtype=int))
        self.scalar_out = False

    def forward_pass(self, input_vec):
        """
        Perform a forward pass through the bus module.
        Arguments:
            input_vec (np.ndarray): The input vector to the module.
        Returns:
            np.ndarray: The output vector from the module.
        """
        input_vec = input_vec.flatten()

        # Rearrange input vector according to input_map
        output = np.zeros(self.n_outputs, dtype=np.float32)
        for i, idx in enumerate(self.input_map):
            if idx == -1:
                continue
            output[idx] = input_vec[i]
        
        return output.flatten()


class Convolution(Module):
    def __init__(self, layer=None):
        """
        Initialise the convolution module.
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
        super().__init__(n_inputs=81, n_outputs=81, layer=layer)
        self.type = "convolution"
        self.module_id = 1
        self.scalar_out = False

    def forward_pass(self, input):
        """
        Perform a forward pass through the convolution module.
        Arguments:
            input (np.ndarray): The input vector to the module.
        Returns:
            np.ndarray: The output vector from the module.
        """
        # Flatten and pad the input if necessary
        input = input.flatten()
        if input.shape[0] < 81:
            input = np.pad(input, (0, 81 - input.shape[0]), mode="constant")
            
        # Reshape input to 9x9 and apply convolution
        input = input.reshape(9, 9)
        kernel = np.ones((3, 3))
        return convolve2d(input, kernel, mode="same").flatten()
    

class Correlation(Module):
    def __init__(self, layer=None):
        """
        Initialise the correlation module.
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
        super().__init__(n_inputs=81, n_outputs=81, layer=layer)
        self.type = "correlation"
        self.module_id = 1
        self.scalar_out = False

    def forward_pass(self, input):
        """
        Perform a forward pass through the correlation module.
        Arguments:
            input (np.ndarray): The input vector to the module.
        Returns:
            np.ndarray: The output vector from the module.
        """
        # Flatten and pad the input if necessary
        input = input.flatten()
        if input.shape[0] < 81:
            input = np.pad(input, (0, 81 - input.shape[0]), mode="constant")

        # Reshape input to 9x9 and apply correlation
        input = input.reshape(9, 9)
        kernel = np.array([
            [1, -1, 1],
            [-1, 1, -1],
            [1, -1, 1]
        ], dtype=np.float32)
        return correlate2d(input, kernel, mode="same").flatten()


class Argmax(Module):
    def __init__(self, layer=None):
        """
        Initialise the argmax module.
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
        super().__init__(n_inputs=81, n_outputs=1, layer=layer)
        self.type = "argmax"
        self.module_id = 2

    def forward_pass(self, input_vec):
        """
        Perform a forward pass through the argmax module.
        Arguments:
            input (np.ndarray): The input vector to the module.
        Returns:
            np.ndarray: The output vector from the module.
        """
        input_vec = input_vec.flatten()

        # Argmax - add small random noise to avoid bias
        input_vec += np.random.rand(len(input_vec)) / 1000
        return np.array([input_vec.argmax()]).flatten()

# Null module for unbiased initialisation 
class Feedforward(Module):
    def __init__(self, layer=None):
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
        super().__init__(n_inputs=81, n_outputs=1, layer=layer)
        self.type = "feedforward"
        self.module_id = 3

    def forward_pass(self, input_vec):
        """
        Perform a forward pass through the feedforward module.
        Arguments:
            input (np.ndarray): The input vector to the module.
        Returns:
            np.ndarray: The output vector from the module.
        """
        return np.array(input_vec[0]).flatten()
    
# ------------------------------------------------------------- #
# --------------------- Old Image Modules --------------------- #
# ------------------------------------------------------------- #

class SquareClassificationModule(Module):
    def __init__(self, inputs=9, outputs=9, layer=None):
        """
        Initialize the square classification module.
        Arguments:
            inputs (int): The number of input connections.
            outputs (int): The number of output connections.
            layer (int): The layer to which the module belongs.
        Properties:
            module_id (int): The unique identifier for the module.
            type (str): The type of the module.
        """
        super().__init__(inputs, outputs, layer)
        self.module_id = 0
        self.type = 'square_classification'

    def forward_pass(self, input_vector):
        """
        Forward pass through the square classification module.
        Returns:
            output_vector (np.ndarray): The output vector from the module.
        """
        input_vector = np.array(input_vector).reshape((3, 3))
        output_vector = np.zeros((3, 3))

        # Weights for neighbouring cells
        contrib_map = {-1: 0, 0:-1}

        rows, cols = input_vector.shape
        for i in range(rows):
            for j in range(cols):
                # Get the values of the neighbouring cells
                neighbours = [
                    input_vector[i-1][j] if i > 0 else -1,
                    input_vector[i][j+1] if j < cols - 1 else -1,
                    input_vector[i+1][j] if i < rows - 1 else -1,
                    input_vector[i][j-1] if j > 0 else -1
                ]

                # Calculate the contribution from the neighbouring cells
                contribution = sum(contrib_map.get(val, val) for val in neighbours)
                output_vector[i][j] = input_vector[i][j] + contribution
                
        return output_vector.flatten()
    

class CrossClassificationModule(Module):
    def __init__(self, inputs=9, outputs=9, layer=None):
        """
        Initialize the square classification module.
        Arguments:
            inputs (int): The number of input connections.
            outputs (int): The number of output connections.
            layer (int): The layer to which the module belongs.
        Properties:
            module_id (int): The unique identifier for the module.
            type (str): The type of the module.
        """
        super().__init__(inputs, outputs, layer)
        self.module_id = 1
        self.type = 'cross_classification'

    def forward_pass(self, input_vector):
        """
        Forward pass through the cross classification module.
        Returns:
            output_vector (np.ndarray): The output vector from the module.
        """
        input_vector = np.array(input_vector).reshape((3, 3))
        output_vector = np.zeros((3, 3))

        # Weights for neighbouring cells
        contrib_map = {-1: 0, 0:-1}

        rows, cols = input_vector.shape
        for i in range(rows):
            for j in range(cols):
                # Get the values of the diagonally neighbouring cells
                neighbours = [
                    input_vector[i-1][j-1] if i > 0 and j > 0 else -1,
                    input_vector[i+1][j+1] if i < rows - 1 and j < cols - 1 else -1,
                    input_vector[i+1][j-1] if i < rows - 1 and j > 0 else -1,
                    input_vector[i-1][j+1] if i > 0 and j < cols - 1 else -1
                ]

                # Calculate the contribution from the diagonally neighbouring cells
                contribution = sum(contrib_map.get(val, val) for val in neighbours)
                output_vector[i][j] = input_vector[i][j] + contribution

                # Get the values of the neighbouring cells
                neighbours = [
                    input_vector[i-1][j] if i > 0 else -1,
                    input_vector[i][j+1] if j < cols - 1 else -1,
                    input_vector[i+1][j] if i < rows - 1 else -1,
                    input_vector[i][j-1] if j > 0 else -1
                ]

                # Calculate the contribution from the neighbouring cells
                contribution = sum(contrib_map.get(val, val) for val in neighbours)
                output_vector[i][j] -= contribution

        return output_vector.flatten()