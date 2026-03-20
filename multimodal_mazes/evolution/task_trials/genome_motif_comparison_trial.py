import multimodal_mazes
import pickle
from multimodal_mazes.evolution.algorithms.genome_EA_motif import GenomeEAMotif as GenomeEA
from multimodal_mazes.image_classification.image_classification import SquareClassification
from multimodal_mazes.evolution.task_trials.neat_maze_trial import run_neat_exp


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
        'n_motif_types': 1,
        'n_inputs': 8,
        'n_outputs': 4,
        'n_motifs': 4,
        'motif_types': [], # List of motif types to use
        'weight_sharing': True,
        'uniform_weights': True, # For weight sharing
        'connectivity': 'RANDOM', # Options: 'FULLY CONNECTED', 'UNCONNECTED', 'SPARSE', 'RANDOM'
        'connection_density' : {'input_density': 0.25, 'output_density': 0.25}, # For 'RANDOM' connectivity
        'population_size': 40,
        'top_genomes': 8, 
        'mutation_rate': 1.0, 
        'crossover_rate': 0.0,
        'one_to_one': True, # Only compatible with UNCONNECTED, SPARSE, and RANDOM if density < 1.0
        'homogeneous': True,
        'class_type': 'square'
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

    # types_to_compare = {
    #     'I2_O1': ['I2_O1', 'I2_O1.R', 'I2_O1.T', 'I2_O1.RT'], 
    #     'I1_H1_O1': ['I1_H1_O1', 'I1_H1_O1.R', 'I1_H1_O1.T', 'I1_H1_O1.RT']
    # }

    types_to_compare = {
        'NEAT': ['NEAT'],  # NEAT as a baseline
        # 'I2_O1': ['I2_O1'],
        # 'I2_O1.R': ['I2_O1.R'],
        # 'I2_O1.T': ['I2_O1.T'],
        # 'I2_O1.RT': ['I2_O1.RT'],
        # 'ALL_I2_O1': ['I2_O1', 'I2_O1.R', 'I2_O1.T', 'I2_O1.RT']
        # 'I1_H1_O1': ['I1_H1_O1'],
        # 'I1_H1_O1.R': ['I1_H1_O1.R'],
        # 'I1_H1_O1.T': ['I1_H1_O1.T'],
        # 'I1_H1_O1.RT': ['I1_H1_O1.RT']
    }

    fitness_results = {}
    motif_distribution_results = {}
    type_distribution_results = {}

    for type_key, types in types_to_compare.items():
        HYPERPARAMETERS['motif_types'] = types

        if 'H' in type_key:
            HYPERPARAMETERS['n_motifs'] = 8

        fitness_results[type_key] = {}
        motif_distribution_results[type_key] = {}
        type_distribution_results[type_key] = {}
        
        for i in range(3):
            if type_key == 'NEAT':
                generations, fitness = run_neat_exp()
                fitness_results[type_key][i] = (generations, fitness)
            else:
                ea = GenomeEA(hyperparameters=HYPERPARAMETERS)
                ea.generate(trial_obj, sensor_noise_scale=sensor_noise_scale, n_steps=steps)

                run_genome_exp(ea, 60)
            
                ea._pool.shutdown(wait=False)
                ea._pool = None

                generations, fitness_over_generations = ea.fitness_over_generations()
                fitness_results[type_key][i] = (generations, fitness_over_generations)

                generations, distributions = ea.motif_distribution_over_generations()
                motif_distribution_results[type_key][i] = (generations, distributions)

                generations, type_distributions = ea.type_distribution_over_generations()
                type_distribution_results[type_key][i] = (generations, type_distributions)

    results = {
        'fitness_results': fitness_results,
        'motif_distribution_results': motif_distribution_results,
        'type_distribution_results': type_distribution_results
    }

    with open("genome_motif_comparison.pkl", "wb") as f:
        pickle.dump(results, f)
