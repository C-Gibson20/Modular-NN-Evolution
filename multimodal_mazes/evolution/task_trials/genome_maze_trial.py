import multimodal_mazes
import pickle
from multimodal_mazes.evolution.algorithms.genome_EA import GenomeEA
from multimodal_mazes.evolution.evaluators.evaluator import genome_eval_fitness


def run_genome_exp(ea, n_generations):
    """Run the genome evolution experiment.
    Args:
        ea: instance of GenomeEA to run the evolution.
        n_generations: number of generations to run the evolution.
    """
    for _ in range(n_generations):
        ea.evaluate()
        ea.evolve()


if __name__ == "__main__":
    HYPERPARAMETERS = {
        'task': 'maze',
        'n_module_types': 2,
        'n_inputs': 8,
        'n_outputs': 4,
        'n_modules': 4,
        'weight_sharing': True,
        'uniform_weights': True,
        'connectivity': 'RANDOM', # Options: 'FULLY CONNECTED', 'UNCONNECTED', 'SPARSE', 'RANDOM', 'IDEAL'
        'connection_density' : {'input_density': 0.125, 'output_density': 0.25}, # For 'RANDOM' connectivity
        'population_size': 40,
        'top_genomes': 8, 
        'mutation_rate': 1.0, 
        'crossover_rate': 0.0,
        'one_to_one': True, # Only compatible with UNCONNECTED, SPARSE, and RANDOM if density < 1.0
        'mutation_rates': [0.5, 0.05, 0.45, 0]
    }

    maze = multimodal_mazes.TrackMaze(size=11, n_channels=2, four_dof=True)
    maze.generate(number=20, noise_scale=0.01, gaps=1)

    ea = GenomeEA(hyperparameters=HYPERPARAMETERS)
    ea.generate(maze, sensor_noise_scale=0.01, n_steps=12)

    run_genome_exp(ea, 20)

    trial_objects = {'ea': ea, 'maze': maze, 'hyperparameters': HYPERPARAMETERS}

    ea._pool.shutdown(wait=False)
    ea._pool = None

    with open("genome_track_maze.pkl", "wb") as f:
        pickle.dump(trial_objects, f)
