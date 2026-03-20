# Square classification
import numpy as np
import matplotlib.pyplot as plt
import cv2

class ImageClassification:
    def __init__(self, size):
        """
        Initialize the image classification model.
        Arguments:
            size (int): size of the input images.
        Properties:
            dmaps (list): list of distance maps for the generated images.
        """
        self.size = size
        self.dmaps = []
        self.images = []
        self.goal_locations = []

    def distance_map(self, img, target):
        """
        Compute the distance map from the target location.
        Arguments:
            img (np.ndarray): input image.
            target (tuple): target location (y, x) in the image.
        Returns:
            np.ndarray: distance map.
        """
        dmap = np.zeros_like(img)
        dmap.reshape((self.size, self.size))

        # Calculate the distance from each pixel to the target
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                dmap[i, j] = np.linalg.norm(np.array([i, j]) - np.array(target))
        return dmap
    
    def plot_image(self, img, guess=None):
        """
        Plot a single image.
        Arguments:
            img (np.ndarray): input image.
            guess (tuple): guessed location (y, x) in the image.
        """
        fig, ax = plt.subplots()
        ax.imshow(img.reshape((self.size, self.size)), cmap='gray_r', origin='lower')
        
        # Plot the guessed location for results analysis
        if guess:
            ax.scatter(guess[1], guess[0], color='red', s=100)

        # Set ticks for the grid lines but hide labels
        ax.set_xticks(np.arange(-0.5, self.size, 1))
        ax.set_yticks(np.arange(-0.5, self.size, 1))
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)  # hide ticks/labels
        ax.grid(True)
        
        # Show axis as a border
        for spine in ax.spines.values():
            spine.set_visible(True)
            
        plt.show()
    

class CrossClassification(ImageClassification):
    def __init__(self, size):
        """
        Initialize the cross classification model.
        Arguments:
            size (int): size of the input images.
        Properties:
            classification_type (str): type of classification (Cross).
        """
        super().__init__(size)
        self.classification_type = "Cross"

    def generate(self, number):
        """
        Generates cross classification images.
        Arguments:
            number (int): number of images to generate.
        Generates:
            images (list): set of generated images.
            goal_locations (list): locations of the center of the last cross for each image.
            dmaps (list): list of distance maps for the generated images.
        """
        img_size = (self.size, self.size)
        
        for _ in range(number):
            img = np.zeros(img_size)
            
            # Generate the X in a random location
            x_loc = np.random.randint(1, img_size[0]-1, size=(2,))
            img[x_loc[0], x_loc[1]] = 1.0
            img[x_loc[0]-1, x_loc[1]-1] = 1.0
            img[x_loc[0]-1, x_loc[1]+1] = 1.0
            img[x_loc[0]+1, x_loc[1]-1] = 1.0
            img[x_loc[0]+1, x_loc[1]+1] = 1.0

            # Define the bounds of the X 
            bound_y = np.arange(x_loc[0]-2, x_loc[0]+3, 1)
            bound_x = np.arange(x_loc[1]-2, x_loc[1]+3, 1)

            # Generate the + in a random location that does not overlap with the X
            while True:
                cross_loc = np.random.randint(1, img_size[0]-1, size=(2,))
                if not (cross_loc[0] in bound_y) and not (cross_loc[1] in bound_x):
                    break

            img[cross_loc[0], cross_loc[1]] = 1.0
            img[cross_loc[0]-1, cross_loc[1]] = 1.0
            img[cross_loc[0]+1, cross_loc[1]] = 1.0
            img[cross_loc[0], cross_loc[1]-1] = 1.0
            img[cross_loc[0], cross_loc[1]+1] = 1.0

            # Update the properties
            self.images.append(img.flatten() / 255)
            self.goal_locations.append(x_loc)
            self.dmaps.append(self.distance_map(img, x_loc))


class SquareClassification(ImageClassification):
    def __init__(self, size):
        """
        Initialize the square classification model.
        Arguments:
            size (int): size of the input images.
        Properties:
            classification_type (str): type of classification (Square).
        """
        super().__init__(size)
        self.classification_type = "Square"

    def generate(self, number):
        """
        Generates square classification images.
        Arguments:
            number (int): number of images to generate.
        Generates:
            images (list): set of generated images.
            goal_locations (list): locations of the center of the last square for each image.
            dmaps (list): list of distance maps for the generated images.
        """

        img_size, lrg_sqr_size = (self.size, self.size), (3, 3)
        lrg_sqr_hwidth = np.floor(lrg_sqr_size[0] / 2).astype(int)

        for _ in range(number):
            img = np.zeros(img_size)

            # Generate the small square in a random location
            sml_sqr_location = np.random.randint(0, img_size[0], size=(2,))
            img[sml_sqr_location[0], sml_sqr_location[1]] = 1.0  

            # Define the bounds of the small square
            bound_y = np.arange(sml_sqr_location[0] - 2, sml_sqr_location[0] + 3)
            bound_x = np.arange(sml_sqr_location[1] - 2, sml_sqr_location[1] + 3)

            # Generate the large square in a random location that does not overlap with the small square
            while True:
                lrg_sqr_location = np.random.randint(lrg_sqr_hwidth, img_size[0] - lrg_sqr_hwidth, size=(2,))
                if not (lrg_sqr_location[0] in bound_y) and not (lrg_sqr_location[1] in bound_x):
                    break

            img[lrg_sqr_location[0] - lrg_sqr_hwidth:lrg_sqr_location[0] + lrg_sqr_hwidth + 1, lrg_sqr_location[1] - lrg_sqr_hwidth:lrg_sqr_location[1] + lrg_sqr_hwidth + 1] = 1.0

            # Update the properties
            self.images.append(img.flatten())
            self.goal_locations.append(lrg_sqr_location)
            self.dmaps.append(self.distance_map(img, lrg_sqr_location))