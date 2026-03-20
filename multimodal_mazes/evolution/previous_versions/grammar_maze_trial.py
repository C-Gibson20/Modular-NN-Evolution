import multimodal_mazes
import numpy as np
import multiprocessing as mp
import pickle
import copy
from multimodal_mazes.evolution.previous_versions.grammar_EA import Grammar_EA
from multimodal_mazes.evolution.previous_versions.agent_grammar import Agent_Grammar

def grammar_maze_trial(genome, mz, maze_start_loc, maze_goal_loc, sensor_noise_scale, n_steps):
    """Run a trial with the given genome in the specified maze.
    Args:
        genome: the grammar tree to evaluate.
        maze: the maze environment to run the trial in.
        maze_start_loc: starting location in the maze.
        maze_goal_loc: goal location in the maze.
        sensor_noise_scale: scale of noise to apply to the agent's sensors.
        n_steps: number of steps to run the agent in the maze.
    Returns:
        time: number of steps taken to reach the goal.
        path: list of locations visited by the agent.
    """
    agent = Agent_Grammar(location=np.copy(maze_start_loc), channels=[1, 1], sensor_noise_scale=sensor_noise_scale, grammar_tree=genome)
    
    path = [list(agent.location)]

    for time in range(n_steps):
        agent.sense(mz)
        agent.policy()
        agent.act(mz)

        path.append(list(agent.location))

        if np.array_equal(agent.location, maze_goal_loc):
            break

    return time, path


def grammar_eval_fitness(genome, maze, sensor_noise_scale, n_steps):
    """Evaluate the fitness of a genome by running it in the maze.
    Args:
        genome: the grammar tree to evaluate.
        maze: the maze environment to run the trial in.
        sensor_noise_scale: scale of noise to apply to the agent's sensors.
        n_steps: number of steps to run the agent in the maze.
    Returns:
        fitness: average fitness across all mazes in the environment.
    """
    fitness, norm_times, norm_paths, times, paths = [], [], [], [], []

    for maze_n, mz in enumerate(maze.mazes):
        trial_results = grammar_maze_trial(genome, mz, maze.start_locations[maze_n], maze.goal_locations[maze_n], sensor_noise_scale, n_steps)
        time, path = trial_results
        times.append(time)
        paths.append(path)
        norm_times.append(1 - ((time - maze.fastest_solutions[maze_n])/ (n_steps - 1 - maze.fastest_solutions[maze_n])))
        norm_paths.append((maze.d_maps[maze_n].max() - maze.d_maps[maze_n][path[-1][0], path[-1][1]])/ maze.d_maps[maze_n].max())

    fitness = (np.array(norm_times) + np.array(norm_paths)) * 0.5
    
    normalised_data = {'paths': norm_paths, 'times': norm_times, 'fitness': fitness}
    data = {'score': np.nanmean(fitness), 'normalised_data': copy.deepcopy(normalised_data), 'times': copy.deepcopy(times), 'paths': copy.deepcopy(paths)}

    return copy.deepcopy(data) #np.nanmean(fitness) 


def run_grammar_exp(ea, maze, n_generations):
    """Run the grammar evolution experiment.
    Args:
        ea: instance of Grammar_EA to run the evolution.
        maze: the maze environment to evaluate the genomes in.
        n_generations: number of generations to run the evolution.
    """
    for _ in range(n_generations):
        ea.evaluate(grammar_eval_fitness, maze, sensor_noise_scale=0.1, n_steps=15)
        ea.evolve()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  
    
    HYPERPARAMETERS = {
        'module_function': None,
        'num_modules': 4,
        'population_size': 200,
        'top_n': 10,
        'top_genomes': 100, 
        'mutation_rate': 0.8, 
        'crossover_rate': 0.8,
    }

    maze = multimodal_mazes.TrackMaze(size=11, n_channels=2)
    maze.generate(number=20, noise_scale=0.1, gaps=1)

    ea = Grammar_EA(inputs=8, outputs=4, hyperparameters=HYPERPARAMETERS)
    ea.generate()

    run_grammar_exp(ea, maze, 100)

    trial_objects = {'ea': ea, 'maze': maze}

    with open("ea_track_maze.pkl", "wb") as f:
        pickle.dump(trial_objects, f)
