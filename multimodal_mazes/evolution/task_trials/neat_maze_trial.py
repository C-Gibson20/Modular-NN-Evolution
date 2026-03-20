import numpy as np
import neat
import os
import shutil
import multimodal_mazes
from multimodal_mazes.evolution.evaluators.evaluator_maze_neat import eval_fitness


def eval_genomes(genomes, config):
    """
    Evaluates the fitness of each genome in the population.
    Arguments:
        genomes: the list of genomes in the current population.
        config: the neat configuration holder.
    Updates: 
        fitness: fitness of each genome.
    """
    global fitness
    scores = []
    for genome_id, genome in genomes:
        genome.fitness = eval_fitness(genome=genome, config=config, channels=exp_config["channels"], sensor_noise_scale=exp_config["sensor_noise_scale"], drop_connect_p=exp_config["drop_connect_p"], maze=maze, n_steps=exp_config["n_steps"])
        scores.append(genome.fitness)
    scores = sorted(scores, reverse=True)
    fitness.append(np.mean(scores[:8]))

def run_neat_exp():
    # Config files
    global exp_config, maze, p, fitness
    neat_config_path = "../../neat_config.ini"
    exp_config = multimodal_mazes.load_exp_config("../../exp_config.ini")

    # Track maze
    maze = multimodal_mazes.TrackMaze(size=exp_config["maze_size"], n_channels=len(exp_config["channels"]))
    maze.generate(number=exp_config["n_mazes"], noise_scale=exp_config["maze_noise_scale"], gaps=exp_config["maze_gaps"])
    fitness = []
    generations = [i for i in range(exp_config["n_generations"])]
    
    # Run
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        neat_config_path,
    )

    # Create the population
    p = neat.Population(config)

    # Init with positive weights
    for n in p.population:
        for c in p.population[n].connections:
            p.population[n].connections[c].weight = abs(p.population[n].connections[c].weight)

    # Run
    _ = p.run(eval_genomes, n=exp_config["n_generations"])

    fitness = np.array(fitness).reshape(exp_config["n_generations"])

    return generations, fitness