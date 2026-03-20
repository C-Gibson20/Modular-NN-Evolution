import numpy as np
import copy
from multimodal_mazes.evolution.agents.agent_genome import AgentGenome

_global_maze_set, _global_noise, _global_n_steps = None, None, None

def _init_evaluator(maze_set, sensor_noise_scale, n_steps):
    """
    Initialize the evaluator with global variables.
    Arguments:
        maze_set (list): The set of mazes to evaluate the agent on.
        sensor_noise_scale (float): The scale of noise to add to sensor inputs.
        n_steps (int): The number of steps to run the agent in each maze.
    """
    global _global_maze_set, _global_noise, _global_n_steps
    _global_maze_set = maze_set
    _global_noise = sensor_noise_scale
    _global_n_steps = n_steps

def genome_maze_trial(genome, mz, maze_start_loc, maze_goal_loc, sensor_noise_scale, n_steps):
    """
    Run a trial with the given genome in the specified maze.
    Args:
        genome (Genome): the genome to evaluate.
        maze (Maze): the maze environment to run the trial in.
        maze_start_loc (array): starting location in the maze.
        maze_goal_loc (array): goal location in the maze.
        sensor_noise_scale (float): scale of noise to apply to the agent's sensors.
        n_steps (int): number of steps to run the agent in the maze.
    Returns:
        time (int): number of steps taken to reach the goal.
        path (list): list of locations visited by the agent.
    """
    # Reset genome, instantiate agent and initialise path
    genome.reset()
    agent = AgentGenome(location=np.copy(maze_start_loc), channels=[1, 1], sensor_noise_scale=sensor_noise_scale, genome=genome)
    path = [list(agent.location)]

    for time in range(n_steps):
        # Agent sensation, policy, and action
        agent.sense(mz)
        agent.policy()
        agent.act(mz)

        # Update path
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

    # Randomly select 4 maze sets from all maze objects if there are multiple maze objects
    idxs = [0] if len(_global_maze_set) == 1 else np.random.choice(range(len(_global_maze_set)), 4, replace=False)
    for i in idxs:
        # Select the maze set
        maze = _global_maze_set[i]
        
        for maze_n, mz in enumerate(maze.mazes):
            # Run trial
            time, path = genome_maze_trial(genome, mz, maze.start_locations[maze_n], maze.goal_locations[maze_n], _global_noise, _global_n_steps)
            
            # Process trial results
            success.append(time < _global_n_steps - 1)
            times.append(time)
            paths.append(path)
            norm_times.append(1 - ((time - maze.fastest_solutions[maze_n])/ (_global_n_steps - 1 - maze.fastest_solutions[maze_n])))
            norm_paths.append((maze.d_maps[maze_n].max() - maze.d_maps[maze_n][path[-1][0], path[-1][1]])/ maze.d_maps[maze_n].max())

    fitness = (np.array(norm_times) + np.array(norm_paths)) * 0.5
    normalised_data = {'paths': norm_paths, 'times': norm_times, 'fitness': fitness}
    return {'score': np.nanmean(fitness), 'normalised_data': copy.deepcopy(normalised_data), 'times': copy.deepcopy(times), 'paths': copy.deepcopy(paths), 'success': copy.deepcopy(success)}


