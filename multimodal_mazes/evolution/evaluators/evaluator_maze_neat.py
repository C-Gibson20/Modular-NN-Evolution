import numpy as np
import multimodal_mazes

def maze_trial(mz, mz_start_loc, mz_goal_loc, channels, sensor_noise_scale, drop_connect_p, n_steps, agnt=None, genome=None, config=None):
    """
    Tests a single agent on a single maze.
    Arguments:
        mz (np.array): a np array of size x size x channels + 1.
            Where [:,:,-1] stores the maze structure.
        mz_start_loc (array): the location of the start [r,c].
        mz_goal_loc (array): the location of the goal [r,c].
        channels (list): list of active (1) and inative (0) channels e.g. [0,1].
        sensor_noise_scale (float): the scale of the noise applied to every sensor.
        drop_connect_p (float): the probability of edge drop out, per time step.
        n_steps (int): number of simulation steps.
        agnt (Agent): an instance of an agent.
        genome (NEAT Genome): neat generated genome.
        config (Config): the neat configuration holder.
    Returns:
        time (int): the number of steps taken to solve the maze.
            Returns n_steps-1 if the agent fails.
        path (list): a list with the agent's location at each time step [r,c].
        states (list): a list of the agent's states at each time step.
            Each entry is a tuple storing tensors of:
            inputs, prev_inputs, hidden, prev_outputs, outputs.
    """
    # Instantiate agent and initialise path
    agnt = multimodal_mazes.AgentNeat(location=mz_start_loc, channels=channels, sensor_noise_scale=sensor_noise_scale, drop_connect_p=drop_connect_p, genome=genome, config=config)
    path = [list(agnt.location)]

    for time in range(n_steps):
        # Agent sensation, policy, and action
        agnt.sense(mz)
        agnt.policy()
        agnt.act(mz)

        # Update path
        path.append(list(agnt.location))
        if np.array_equal(agnt.location, mz_goal_loc):
            break

    return time, path


def eval_fitness(genome, config, channels, sensor_noise_scale, drop_connect_p, maze, n_steps, agnt=None):
    """
    Evalutes the fitness of the provided genome across a set of mazes.
    Arguments:
        genome (NEAT Genome): neat generated genome.
        config (Config): the neat configuration holder.
        channels (list): list of active (1) and inative (0) channels e.g. [0,1].
        sensor_noise_scale (float): the scale of the noise applied to every sensor.
        drop_connect_p (float): the probability of edge drop out, per time step.
        maze (Maze): a class containing a set of mazes.
        n_steps (int): the max number of simulation steps per maze.
        agnt (Agent): an instance of an agent.
    Returns:
        float: the mean fitness across mazes, between [0,1].
    """
    times, paths = [], []
    
    for mz_n, mz in enumerate(maze.mazes):
        # Run trial
        time, path = maze_trial(mz=mz, mz_start_loc=maze.start_locations[mz_n], mz_goal_loc=maze.goal_locations[mz_n], channels=channels, sensor_noise_scale=sensor_noise_scale, drop_connect_p=drop_connect_p, n_steps=n_steps, agnt=agnt, genome=genome, config=config)

        # Process trial results
        times.append(1 - ((time - maze.fastest_solutions[mz_n]) / (n_steps - 1 - maze.fastest_solutions[mz_n])))
        paths.append((maze.d_maps[mz_n].max() - maze.d_maps[mz_n][path[-1][0], path[-1][1]]) / maze.d_maps[mz_n].max())

    return np.nanmean((np.array(times) + np.array(paths)) * 0.5)