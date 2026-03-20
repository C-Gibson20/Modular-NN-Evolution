import pickle
from multimodal_mazes.image_classification.image_classification import SquareClassification
from multimodal_mazes.evolution.algorithms.genome_EA import GenomeEA
from multimodal_mazes.evolution.evaluators.evaluator_image import genome_eval_fitness


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
        'n_module_types': 2,
        'n_inputs': 81,
        'n_outputs': 81,
        'n_modules': 9,
        'weight_sharing': True,
        'uniform_weights': True,
        'connectivity': 'IDEAL', # Options: 'FULLY CONNECTED', 'UNCONNECTED', 'SPARSE', 'RANDOM', 'IDEAL'
        'connection_density' : {'input_density': 0.25, 'output_density': 0.25}, # For 'RANDOM' connectivity
        # 'population_size': 40,
        # 'top_genomes': 8, 
        'population_size': 1,
        'top_genomes': 1, 
        'mutation_rate': 1.0, 
        'crossover_rate': 0.0,
        'one_to_one': False, # Only compatible with UNCONNECTED, SPARSE, and RANDOM if density < 1.0
        'class_type': 'square',
        'mutation_rates': [0.5, 0.05, 0.45, 0]
    }

    img = SquareClassification(size=9)
    img.generate(number=50)

    ea = GenomeEA(hyperparameters=HYPERPARAMETERS)
    ea.generate(img, sensor_noise_scale=0.0)

    # run_genome_exp(ea, 2)
    run_genome_exp(ea, 50)

    trial_objects = {'ea': ea, 'img': img}

    ea._pool.shutdown(wait=False)
    ea._pool = None

    with open("genome_img_square.pkl", "wb") as f:
        pickle.dump(trial_objects, f)
