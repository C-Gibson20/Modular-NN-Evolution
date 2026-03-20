import numpy as np
import copy
from multimodal_mazes.evolution.agents.agent_genome import AgentGenome

_global_maze, _global_mazes, _global_starts, _global_goals, _global_noise, _global_n_steps = None, None, None, None, None, None

def _init_evaluator(maze, sensor_noise_scale, n_steps):
    """
    Initialize the evaluator with global variables.
    Arguments:
        maze (Maze): The maze object containing the task information.
        sensor_noise_scale (float): The scale of noise to add to sensor inputs.
        n_steps (int): The number of steps to run the agent in the maze.
    Properties:
        global_maze (Maze): The global maze object.
        global_mazes (list): The global list of mazes.
        global_starts (list): The global list of start locations.
        global_goals (list): The global list of goal locations.
        global_noise (float): The global noise scale.
        global_n_steps (int): The global number of steps.
    """
    global _global_maze, _global_mazes, _global_starts, _global_goals, _global_noise, _global_n_steps
    _global_maze = maze
    _global_mazes = maze.mazes
    _global_starts = maze.start_locations
    _global_goals = maze.goal_locations
    _global_noise = sensor_noise_scale
    _global_n_steps = n_steps

def genome_maze_trial(genome, mz, maze_start_loc, maze_goal_loc, sensor_noise_scale, n_steps):
    """Run a trial with the given genome in the specified maze.
    Args:
        genome (Genome): the genome to evaluate.
        mz (Maze): the maze environment to run the trial in.
        maze_start_loc (array): starting location in the maze.
        maze_goal_loc (array): goal location in the maze.
        sensor_noise_scale (float): scale of noise to apply to the agent's sensors.
        n_steps (int): number of steps to run the agent in the maze.
    Returns:
        time (int): number of steps taken to reach the goal.
        path (list): list of locations visited by the agent.
    """
    for module in genome.modules:
        module.reset()
    agent = AgentGenome(location=np.copy(maze_start_loc), channels=[1, 1], sensor_noise_scale=sensor_noise_scale, genome=genome)
    path = [list(agent.location)]

    for time in range(n_steps):
        agent.sense(mz)
        agent.policy()
        agent.act(mz)

        path.append(list(agent.location))

        if np.array_equal(agent.location, maze_goal_loc):
            break

    return time, path


def genome_eval_fitness(genome):
    """Evaluate the fitness of a genome by running it in the maze.
    Args:
        genome (Genome): the genome tree to evaluate.
    Returns:
        data (dict): a dictionary containing the evaluation results.
    """
    fitness, norm_times, norm_paths, times, paths, success = [], [], [], [], [], []

    for maze_n, mz in enumerate(_global_mazes):
        trial_results = genome_maze_trial(genome, mz, _global_starts[maze_n], _global_goals[maze_n], _global_noise, _global_n_steps)
        time, path = trial_results

        success.append(time < _global_n_steps - 1)
        times.append(time)
        paths.append(path)
        norm_times.append(1 - ((time - _global_maze.fastest_solutions[maze_n])/ (_global_n_steps - 1 - _global_maze.fastest_solutions[maze_n])))
        norm_paths.append((_global_maze.d_maps[maze_n].max() - _global_maze.d_maps[maze_n][path[-1][0], path[-1][1]])/ _global_maze.d_maps[maze_n].max())

    fitness = (np.array(norm_times) + np.array(norm_paths)) * 0.5

    normalised_data = {'paths': norm_paths, 'times': norm_times, 'fitness': fitness}
    data = {'score': np.nanmean(fitness), 'normalised_data': copy.deepcopy(normalised_data), 'times': copy.deepcopy(times), 'paths': copy.deepcopy(paths), 'success': copy.deepcopy(success)}

    return data


