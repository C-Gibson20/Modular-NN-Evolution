import multimodal_mazes
import pickle
from multimodal_mazes.evolution.algorithms.genome_EA import GenomeEA
from multimodal_mazes.image_classification.image_classification import SquareClassification, CrossClassification
from multimodal_mazes.evolution.task_trials.neat_maze_trial import run_neat_exp as run_neat_maze_exp
from multimodal_mazes.evolution.task_trials.neat_image_trial import run_neat_exp as run_neat_image_exp



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
        'task': 'image',
        'n_module_types': 4,
        'n_inputs': 81,
        'n_outputs': 81,
        'n_modules': 9,
        'weight_sharing': True,
        'uniform_weights': True,
        'connectivity': 'RANDOM', # Options: 'FULLY CONNECTED', 'UNCONNECTED', 'SPARSE', 'RANDOM', 'IDEAL'
        'connection_density' : {'input_density': 0.25, 'output_density': 0.25}, # For 'RANDOM' connectivity
        'population_size': 40,
        'top_genomes': 8, 
        # 'population_size': 1,
        # 'top_genomes': 1, 
        'mutation_rate': 1.0, 
        'crossover_rate': 0.0,
        'one_to_one': True, # Only compatible with UNCONNECTED, SPARSE, and RANDOM if density < 1.0
        'class_type': 'square',
        'mutation_rates': [0.5, 0.05, 0.45, 0]
    }

    task = HYPERPARAMETERS['task']

    if task == 'maze':
        trial_obj = []
        for _ in range(1):
            # maze = multimodal_mazes.TrackMaze(size=11, n_channels=2, four_dof=True)
            maze = multimodal_mazes.TrackMaze(size=11, n_channels=2, four_dof=False)
            maze.generate(number=20, noise_scale=0.0, gaps=1)
            trial_obj.append(maze)
        sensor_noise_scale = 0.0
        steps = 5
        obj = 'maze'
    elif task == 'image' and HYPERPARAMETERS['class_type'] == 'square':
        trial_obj = SquareClassification(size=9)
        trial_obj.generate(number=50)
        sensor_noise_scale = 0.0
        steps = None
        obj = 'img'
    elif task == 'image' and HYPERPARAMETERS['class_type'] == 'cross':
        trial_obj = CrossClassification(size=9)
        trial_obj.generate(number=50)
        sensor_noise_scale = 0.0
        steps = None
        obj = 'img'

    types_to_compare = {
        'NEAT': ['NEAT'],  # NEAT as a baseline
        # 'EA': ['EA']
    }

    fitness_results = {}

    for type_key, types in types_to_compare.items():
        HYPERPARAMETERS['motif_types'] = types

        if 'H' in type_key:
            HYPERPARAMETERS['n_motifs'] = 8

        fitness_results[type_key] = {}
        
        for i in range(1):
            if type_key == 'NEAT' and task == 'maze':
                generations, fitness = run_neat_maze_exp()
                fitness_results[type_key][i] = (generations, fitness)
            elif type_key == 'NEAT' and task == 'image':
                generations, fitness = run_neat_image_exp(trial_obj=trial_obj)
                fitness_results[type_key][i] = (generations, fitness)
            else:
                ea = GenomeEA(hyperparameters=HYPERPARAMETERS)
                ea.generate(trial_obj, sensor_noise_scale=sensor_noise_scale, n_steps=steps)

                run_genome_exp(ea, 100)
            
                ea._pool.shutdown(wait=False)
                ea._pool = None

                generations, fitness_over_generations = ea.fitness_over_generations()
                fitness_results[type_key][i] = (generations, fitness_over_generations)

    results = {'fitness_results': fitness_results}

    with open("genome_comparison.pkl", "wb") as f:
        pickle.dump(results, f)
