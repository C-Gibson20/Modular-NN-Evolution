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
        plt.imshow(img.reshape((self.size, self.size)), cmap='gray_r', origin='lower')
        if guess:
            plt.scatter(guess[1], guess[0], color='red', s=100)
        plt.axis('off')
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
        images, goal_locations, dmaps = [], [], []

        for n in range(number):
            img = np.zeros(img_size)
            
            x_loc = np.random.randint(1, img_size[0]-1, size=(2,))
            img[x_loc[0], x_loc[1]] = 1.0
            img[x_loc[0]-1, x_loc[1]-1] = 1.0
            img[x_loc[0]-1, x_loc[1]+1] = 1.0
            img[x_loc[0]+1, x_loc[1]-1] = 1.0
            img[x_loc[0]+1, x_loc[1]+1] = 1.0

            range_0x = np.arange(x_loc[0]-2, x_loc[0]+3, 1)
            range_1x = np.arange(x_loc[1]-2, x_loc[1]+3, 1)

            while True:
                cross_loc = np.random.randint(1, img_size[0]-1, size=(2,))
                if (cross_loc[0] in range_0x) or (cross_loc[1] in range_1x):
                    continue
                else:
                    break

            img[cross_loc[0], cross_loc[1]] = 1.0
            img[cross_loc[0]-1, cross_loc[1]] = 1.0
            img[cross_loc[0]+1, cross_loc[1]] = 1.0
            img[cross_loc[0], cross_loc[1]-1] = 1.0
            img[cross_loc[0], cross_loc[1]+1] = 1.0

            images.append(img.flatten() / 255)

            goal_locations.append(x_loc)
            dmap = self.distance_map(img, x_loc)
            dmaps.append(dmap)
            
        self.images = images
        self.goal_locations = goal_locations
        self.dmaps = dmaps


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
        images, goal_locations, dmaps = [], [], []

        for n in range(number):
            img = np.zeros(img_size)
            
            sml_sqr_location = np.random.randint(0, img_size[0], size=(2,))
            img[sml_sqr_location[0], sml_sqr_location[1]] = 1.0  

            # Prohibited area around small square
            prohib_vert = np.arange(sml_sqr_location[0] - 2, sml_sqr_location[0] + 3)
            prohib_horz = np.arange(sml_sqr_location[1] - 2, sml_sqr_location[1] + 3)

            while True:
                lrg_sqr_location = np.random.randint(lrg_sqr_hwidth, img_size[0] - lrg_sqr_hwidth, size=(2,))
                if (lrg_sqr_location[0] in prohib_vert) or (lrg_sqr_location[1] in prohib_horz):
                    continue
                else:
                    break

            tmp = img.copy()
            tmp[lrg_sqr_location[0] - lrg_sqr_hwidth:lrg_sqr_location[0] + lrg_sqr_hwidth + 1,
                 lrg_sqr_location[1] - lrg_sqr_hwidth:lrg_sqr_location[1] + lrg_sqr_hwidth + 1] = 1.0
            images.append(tmp.flatten())

            goal_locations.append(lrg_sqr_location)
            dmap = self.distance_map(tmp, lrg_sqr_location)
            dmaps.append(dmap)
            
        self.images = images
        self.goal_locations = goal_locations
        self.dmaps = dmaps