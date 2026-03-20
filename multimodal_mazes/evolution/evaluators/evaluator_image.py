import numpy as np
import copy
from multimodal_mazes.evolution.agents.agent_image import AgentImage

_global_img, _global_imgs, _global_goals, _global_noise = None, None, None, None

def _init_evaluator(img, sensor_noise_scale):
    """
    Initialize the evaluator with global variables.
    Arguments:
        img (ImageClassification): The image object containing the task information.
        sensor_noise_scale (float): The scale of noise to add to sensor inputs.
    Properties:
        global_img (ImageClassification): The global image object.
        global_imgs (list): The global list of images.
        global_goals (list): The global list of goal locations.
        global_noise (float): The global noise scale.
    """
    global _global_img, _global_imgs, _global_goals, _global_noise
    _global_img = img
    _global_imgs = img.images
    _global_goals = img.goal_locations
    _global_noise = sensor_noise_scale

def genome_image_trial(genome, img, sensor_noise_scale):
    """Run a trial with the given genome with the specified image.
    Args:
        genome (Genome): the genome to evaluate.
        img (np.array): the image to use for the trial.
        sensor_noise_scale (float): scale of noise to apply to the agent's sensors.
    Returns:
        agent_output (array): the output of the agent after processing the image.
    """
    # Instantiate agent and reset modules
    agent = AgentImage(sensor_noise_scale=sensor_noise_scale, genome=genome)
    for module in genome.modules:
        module.reset()

    # Run agent policy
    agent.policy(_global_img, img, processing=False)
    
    # Validate and return output
    return [min(int(agent.output[0]), 8), min(int(agent.output[1]), 8)]

def genome_eval_fitness(genome):
    """Evaluate the fitness of a genome by testing it with an image.
    Args:
        genome (Genome): the genome to evaluate.
    Returns:
        data (dict): a dictionary containing the evaluation results.
    """
    fitness, outputs, success = [], [], []

    for n, img in enumerate(_global_imgs):
        # Run trial
        agent_output = genome_image_trial(genome, img, _global_noise)

        # Process trial results
        outputs.append(agent_output)
        success.append(np.array_equal(agent_output, _global_goals[n]))
        fitness.append((_global_img.dmaps[n].max() - _global_img.dmaps[n][agent_output[0], agent_output[1]]) / _global_img.dmaps[n].max())

    return {'score': np.nanmean(fitness), 'outputs': copy.deepcopy(outputs), 'success': copy.deepcopy(success)}