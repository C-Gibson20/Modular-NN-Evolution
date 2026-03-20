from multimodal_mazes.evolution.previous_versions.genome import Genome
from multimodal_mazes.evolution.previous_versions.evaluator import genome_eval_fitness
import random
import heapq
import os
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import math

_global_maze = None
_global_noise_scale = None
_global_n_steps = None

def _init_worker(maze, sensor_noise_scale, n_steps):
    """Initialize the worker with the maze and parameters."""
    global _global_maze, _global_noise_scale, _global_n_steps
    _global_maze = maze
    _global_noise_scale = sensor_noise_scale
    _global_n_steps = n_steps

def _evaluate_genome(genome):
    """Evaluate a genome in the global maze."""
    return genome_eval_fitness(genome, _global_maze, _global_noise_scale, _global_n_steps)

def _evaluate_genome_chunk(genomes):
    """Evaluate a chunk of genomes in the global maze."""
    return [genome_eval_fitness(g, _global_maze, _global_noise_scale, _global_n_steps) for g in genomes]

class GenomeEA:
    def __init__(self, hyperparameters = None, evaluator=None):
        self.hyperparameters = hyperparameters
        self.evaluator = evaluator
        
        self.population = hyperparameters['population_size']
        self.top_genomes = hyperparameters['top_genomes']
        self.mutation_rate = hyperparameters['mutation_rate']
        self.crossover_rate = hyperparameters['crossover_rate']
        self.generation = 0
        self.last_genome_id = 0

        self.genomes = []
        self.fittest_network = None
        self.fittest_networks = []

        self._n_workers = max(1, os.cpu_count() - 1)
        self._pool = None
        self.chunk_size = math.ceil(self.population / self._n_workers)

        self.trial_data = {}
    
    def generate(self, maze, sensor_noise_scale, n_steps): 
        """Generate the initial population of grammar trees."""
        for _ in range(self.population):
            genome = Genome(self.last_genome_id, self.hyperparameters, False)
            self.genomes.append(genome)
            self.last_genome_id += 1

        self.fittest_network = self.genomes[0]
        self.fittest_networks = self.genomes[:self.top_genomes]

        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers  = self._n_workers,
                initializer  = _init_worker,
                initargs     = (maze, sensor_noise_scale, n_steps)
            )

    def evaluate(self, maze, sensor_noise_scale, n_steps):
        """Evaluate the fitness of each genome in the population.
        Args:
            maze: the maze environment to evaluate the genomes in.
            sensor_noise_scale: scale of noise to apply to the agent's sensors.
            n_steps: number of steps to run the agent in the maze.
        """
        self.trial_data[self.generation] = {}

        batches = [self.genomes[i:i + self.chunk_size] for i in range(0, len(self.genomes), self.chunk_size)]

        results_iter = self._pool.map(_evaluate_genome_chunk, batches)

        for batch, batch_results in zip(batches, results_iter):
            for genome, res in zip(batch, batch_results):
                genome.fitness = res['score']
                self.trial_data[self.generation][genome.genome_id] = res

        self.update_fittest()

    def update_fittest(self):
        """Update the fittest genome based on the current generation."""
        self.fittest_networks = heapq.nlargest(self.top_genomes, self.genomes, key=lambda g: g.fitness)
        self.fittest_network = self.fittest_networks[0]
        self.trial_data[self.generation]['fittest'] = (self.fittest_network.fitness, self.fittest_network)
    
    def evolve(self):
        """Evolve the population of genomes by selecting the fittest individuals and applying crossover and mutation."""
        top_genomes = list(self.fittest_networks)
        n_new_genomes = self.population - self.top_genomes

        total_op = self.crossover_rate + self.mutation_rate
        p_cross = self.crossover_rate / total_op
        n_cross = np.random.binomial(n_new_genomes, p_cross)
        n_mut = n_new_genomes - n_cross

        parents_cross = random.choices(top_genomes, k=2 * n_cross)
        parents_mut = random.choices(top_genomes, k=n_mut)

        new_genomes = []   

        for i in range(0, 2*n_cross, 2):
            p1, p2 = parents_cross[i], parents_cross[i+1]
            child = p1.crossover(self.last_genome_id, p2)
            new_genomes.append(child)
            self.last_genome_id += 1

        for p in parents_mut:
            child = p.clone(self.last_genome_id)
            child.mutate()
            new_genomes.append(child)
            self.last_genome_id += 1

        self.genomes = top_genomes + new_genomes
        self.generation += 1

    def plot_fitness_over_generations(self):
        """Plot the fitness of the fittest genome over generations."""
        generations = list(self.trial_data.keys())
        fitness = [self.trial_data[gen]['fittest'][0] for gen in generations]

        plt.plot(generations, fitness)
        plt.title('Fitness Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.ylim([0, 1])
        plt.show()
