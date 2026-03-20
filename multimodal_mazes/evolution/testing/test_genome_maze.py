import multimodal_mazes
import pickle
import numpy as np
from multimodal_mazes.evolution.evaluators.evaluator_motif import genome_maze_trial

def test_genome(genome, maze, sensor_noise_scale, n_steps):
    """
    Test the genome on the given maze.
    Arguments:
        genome (Genome): The genome to test.
        maze (Maze): The maze to test the genome on.
        sensor_noise_scale (float): The sensor noise scale to use during testing.
        n_steps (int): The number of steps to simulate.
    Returns:
        dict: A dictionary containing the test results.
    """
    mazes = maze.mazes
    starts = maze.start_locations
    goals = maze.goal_locations
    fitness, norm_times, norm_paths, times, paths, success = [], [], [], [], [], []

    for maze_n, mz in enumerate(mazes):
        trial_results = genome_maze_trial(genome, mz, starts[maze_n], goals[maze_n], sensor_noise_scale, n_steps)
        time, path = trial_results

        times.append(time)
        paths.append(path)
        success.append(time < n_steps - 1)
        norm_times.append(1 - ((time - maze.fastest_solutions[maze_n])/ (n_steps - 1 - maze.fastest_solutions[maze_n])))
        norm_paths.append((maze.d_maps[maze_n].max() - maze.d_maps[maze_n][path[-1][0], path[-1][1]])/ maze.d_maps[maze_n].max())

    fitness = (np.array(norm_times) + np.array(norm_paths)) * 0.5
    data = {
        'fitness': fitness,
        'times': times,
        'paths': paths,
        'success': success
    }
    return data

def test_fittest_genome(fittest_genome, task, track, trial_maze = None):
    """
    Test the fittest genome on the given task and track.
    Arguments:
        fittest_genome (Genome): The fittest genome to test.
        task (str): The task to test the genome on.
        track (bool): Whether to use the track maze or not.
        trial_maze (Maze, optional): A specific maze to test the genome on.
    """
    if task == 'maze':
        if track:
            maze = multimodal_mazes.TrackMaze(size=11, n_channels=2, four_dof=True)
            maze.generate(number=200, noise_scale=0.0, gaps=1)
        else:
            maze = multimodal_mazes.GeneralMaze(size=9, n_channels=2)
            maze.generate(number=200, noise_scale=0.0, wall_sparsity=0.2, cue_sparsity=0.2)

    if trial_maze:
        maze = trial_maze

    test_data = test_genome(fittest_genome, maze, sensor_noise_scale=0.0, n_steps=5)
    test_objects = {
        'maze': maze,
        'test_data': test_data
    }

    # Save the results
    with open("genome_motif_test.pkl", "wb") as f:
        pickle.dump(test_objects, f)