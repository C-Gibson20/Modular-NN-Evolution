import numpy as np
import copy
from multimodal_mazes.evolution.agents.agent_genome import AgentGenome

def genome_maze_trial(genome, mz, maze_start_loc, maze_goal_loc, sensor_noise_scale, n_steps):
    """Run a trial with the given genome in the specified maze.
    Args:
        genome: the genome to evaluate.
        maze: the maze environment to run the trial in.
        maze_start_loc: starting location in the maze.
        maze_goal_loc: goal location in the maze.
        sensor_noise_scale: scale of noise to apply to the agent's sensors.
        n_steps: number of steps to run the agent in the maze.
    Returns:
        time: number of steps taken to reach the goal.
        path: list of locations visited by the agent.
    """
    agent = AgentGenome(location=np.copy(maze_start_loc), channels=[1, 1], sensor_noise_scale=sensor_noise_scale, genome=genome)
    # agent.genome.module.previous_output = [None, None, None, None]  # Reset previous output for each trial
    path = [list(agent.location)]

    for time in range(n_steps):
        prev_loc = np.copy(agent.location)
        
        agent.sense(mz)
        agent.policy()
        agent.act(mz)

        path.append(list(agent.location))

        if np.array_equal(agent.location, maze_goal_loc):
            break

    return time, path


def genome_eval_fitness(genome, maze, sensor_noise_scale, n_steps):
    """Evaluate the fitness of a genome by running it in the maze.
    Args:
        genome: the genome tree to evaluate.
        maze: the maze environment to run the trial in.
        sensor_noise_scale: scale of noise to apply to the agent's sensors.
        n_steps: number of steps to run the agent in the maze.
    Returns:
        fitness: average fitness across all mazes in the environment.
    """
    fitness, norm_times, norm_paths, times, paths, success = [], [], [], [], [], []
    
    for maze_n, mz in enumerate(maze.mazes):
        trial_results = genome_maze_trial(genome, mz, maze.start_locations[maze_n], maze.goal_locations[maze_n], sensor_noise_scale, n_steps)
        time, path = trial_results

        success.append(time < n_steps - 1)
        times.append(time)
        paths.append(path)
        norm_times.append(1 - ((time - maze.fastest_solutions[maze_n])/ (n_steps - 1 - maze.fastest_solutions[maze_n])))
        norm_paths.append((maze.d_maps[maze_n].max() - maze.d_maps[maze_n][path[-1][0], path[-1][1]])/ maze.d_maps[maze_n].max())

    fitness = (np.array(norm_times) + np.array(norm_paths)) * 0.5

    normalised_data = {'paths': norm_paths, 'times': norm_times, 'fitness': fitness}
    data = {'score': np.nanmean(fitness), 'normalised_data': copy.deepcopy(normalised_data), 'times': copy.deepcopy(times), 'paths': copy.deepcopy(paths), 'success': copy.deepcopy(success)}

    return data