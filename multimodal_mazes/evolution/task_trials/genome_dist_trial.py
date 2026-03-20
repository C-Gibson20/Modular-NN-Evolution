import multimodal_mazes
import pickle
from multimodal_mazes.evolution.algorithms.genome_EA_dist import GenomeEADist as GenomeEA
from multimodal_mazes.image_classification.image_classification import SquareClassification, CrossClassification
from multimodal_mazes.evolution.evaluators.evaluator import genome_eval_fitness as genome_eval_fitness_maze
from multimodal_mazes.evolution.evaluators.evaluator_image import genome_eval_fitness as genome_eval_fitness_square


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
        # 'task': 'maze',
        'n_module_types': 4,
        # 'n_module_types': 2,
        'n_inputs': 81,
        # 'n_inputs': 8,
        'n_outputs': 1,
        # 'n_outputs': 4,
        'n_modules': 3,
        # 'n_modules': 4,
        'weight_sharing': True,
        'uniform_weights': True,
        'connectivity': 'RANDOM', # Options: 'FULLY CONNECTED', 'UNCONNECTED', 'SPARSE', 'RANDOM'
        'connection_density' : {'input_density': 0.2, 'output_density': 1.0}, # For 'RANDOM' connectivity
        'population_size': 40,
        'top_genomes': 8, 
        # 'population_size': 10,
        # 'top_genomes': 2, 
        'mutation_rate': 1.0, 
        'crossover_rate': 0.0,
        'one_to_one': True,
        # 'network_type': 0,
        'network_type': 1,
        'mutation_rates': [0.6, 0.0, 0.0, 0.4],
        'class_type': 'square'
    }

    task = HYPERPARAMETERS['task']

    if task == 'maze':
        trial_obj = multimodal_mazes.TrackMaze(size=11, n_channels=2, four_dof=True)
        trial_obj.generate(number=20, noise_scale=0.0, gaps=1)
        sensor_noise_scale = 0.0
        steps = 5
        obj = 'maze'
        ea = GenomeEA(hyperparameters=HYPERPARAMETERS)
    elif task == 'image' and HYPERPARAMETERS['class_type'] == 'square':
        trial_obj = SquareClassification(size=9)
        trial_obj.generate(number=50)
        sensor_noise_scale = 0.0
        steps = None
        obj = 'img'
        ea = GenomeEA(hyperparameters=HYPERPARAMETERS)
    elif task == 'image' and HYPERPARAMETERS['class_type'] == 'cross':
        trial_obj = CrossClassification(size=9)
        trial_obj.generate(number=50)
        sensor_noise_scale = 0.0
        steps = None
        obj = 'img'
        ea = GenomeEA(hyperparameters=HYPERPARAMETERS)

    ea.generate(trial_obj, sensor_noise_scale=sensor_noise_scale, n_steps=steps)

    # run_genome_exp(ea, 10)
    run_genome_exp(ea, 100)

    trial_objects = {'ea': ea, 'trial_object': trial_obj, 'hyperparameters': HYPERPARAMETERS}

    ea._pool.shutdown(wait=False)
    ea._pool = None

    with open("genome_dist_trial.pkl", "wb") as f:
        pickle.dump(trial_objects, f)
