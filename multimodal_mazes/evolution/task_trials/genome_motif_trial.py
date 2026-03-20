import multimodal_mazes
import pickle
from multimodal_mazes.evolution.algorithms.genome_EA_motif import GenomeEAMotif as GenomeEA

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
        # 'n_motifs': 8,
        'motif_types': ['I2_O1', 'I2_O1.R', 'I2_O1.T', 'I2_O1.RT'], # List of motif types to use
        # 'motif_types': ['I1_H1_O1'], #'I1_H1_O1.R', 'I1_H1_O1.T', 'I1_H1_O1.RT'], # List of motif types to use
        # 'motif_types': ['I1_O2'], #'I1_O2.R', 'I1_O2.T', 'I1_O2.RT'], # List of motif types to use
        'weight_sharing': True,
        'uniform_weights': True, # For weight sharing
        'connectivity': 'RANDOM', # Options: 'FULLY CONNECTED', 'UNCONNECTED', 'SPARSE', 'RANDOM'
        'connection_density' : {'input_density': 0.125, 'output_density': 0.25}, # For 'RANDOM' connectivity
        'population_size': 40,
        'top_genomes': 8,
        # 'population_size': 1,
        # 'top_genomes': 1,
        'mutation_rate': 1.0,
        'crossover_rate': 0.0,
        'one_to_one': True, # Only compatible with UNCONNECTED, SPARSE, and RANDOM if density < 1.0
        'homogeneous': True,
        'class_type': 'track',
        'mutation_rates': [0.4, 0.05, 0.3, 0.25]
    }

    task = HYPERPARAMETERS['task']

    if task == 'maze':
        trial_obj = []
        for _ in range(1):
            if HYPERPARAMETERS['class_type'] == 'track':
                maze = multimodal_mazes.TrackMaze(size=11, n_channels=2, four_dof=True)
                # maze = multimodal_mazes.TrackMaze(size=11, n_channels=2, four_dof=False)
                maze.generate(number=40, noise_scale=0.0, gaps=1)
                steps = 5
            else:
                maze = multimodal_mazes.GeneralMaze(size=9, n_channels=2)
                maze.generate(number=40, noise_scale=0.0, wall_sparsity=0.25, cue_sparsity=0)
                steps = 25

            trial_obj.append(maze)
        sensor_noise_scale = 0.0
        obj = 'maze'

    ea = GenomeEA(hyperparameters=HYPERPARAMETERS)
    ea.generate(trial_obj, sensor_noise_scale=sensor_noise_scale, n_steps=steps)

    run_genome_exp(ea, 30)
    # run_genome_exp(ea, 2)

    trial_objects = {'ea': ea, obj: trial_obj, 'hyperparameters': HYPERPARAMETERS}

    ea._pool.shutdown(wait=False)
    ea._pool = None

    with open("genome_motif_trial.pkl", "wb") as f:
        pickle.dump(trial_objects, f)
