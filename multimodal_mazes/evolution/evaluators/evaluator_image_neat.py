# Maze trial
import copy
import numpy as np
from multimodal_mazes.evolution.agents.agent_neat_image import AgentNeatImage as AgentNeat


def image_trial(image, img, genome, sensor_noise_scale, drop_connect_p, config):
    """
    Tests a single agent on a single image.
    Arguments:
        img (np.array): a np array of size (x, x).
        sensor_noise_scale (float): the scale of the noise applied to every sensor.
        drop_connect_p (float): the probability of edge drop out, per time step.
        genome (NEAT Genome): neat generated genome.
        config (Config): the neat configuration holder.
    Returns:
        agent_output: the output of the agent after processing the image.
    """
    # Instantiate agent
    agnt = AgentNeat(sensor_noise_scale=sensor_noise_scale, drop_connect_p=drop_connect_p, genome=genome, config=config)

    # Run agent policy
    agnt.policy(image, img)

    # Validate and return output
    return [min(int(agnt.output[0]), 8), min(int(agnt.output[1]), 8)]


def eval_fitness(genome, config, sensor_noise_scale, drop_connect_p, image):
    """
    Evalutes the fitness of the provided genome across a set of mazes.
    Arguments:
        genome (NEAT Genome): neat generated genome.
        config (Config): the neat configuration holder.
        sensor_noise_scale (float): the scale of the noise applied to every sensor.
        drop_connect_p (float): the probability of edge drop out, per time step.
        image (ImageClassification): a class containing a set of images.
    Returns:
        dict: a dictionary containing the fitness, outputs, and success lists.
    """
    fitness, outputs, success = [], [], []

    for n, img in enumerate(image.images):
        # Run trial
        agent_output = image_trial(image, img, genome, sensor_noise_scale, drop_connect_p, config)

        # Process trial data
        outputs.append(agent_output)
        success.append(np.array_equal(agent_output, img))
        fitness.append((image.dmaps[n].max() - image.dmaps[n][agent_output[0], agent_output[1]]) / image.dmaps[n].max())

    return {'score': np.nanmean(fitness), 'outputs': copy.deepcopy(outputs), 'success': copy.deepcopy(success)}