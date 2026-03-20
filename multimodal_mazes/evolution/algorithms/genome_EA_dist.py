from multimodal_mazes.evolution.genomes.genome_dist import GenomeDist as Genome
from multimodal_mazes.evolution.evaluators.evaluator import genome_eval_fitness as genome_eval_fitness_maze
from multimodal_mazes.evolution.evaluators.evaluator import _init_evaluator as _init_evaluator_maze
from multimodal_mazes.evolution.evaluators.evaluator_image import genome_eval_fitness as genome_eval_fitness_square
from multimodal_mazes.evolution.evaluators.evaluator_image import _init_evaluator as _init_evaluator_square
from multimodal_mazes.evolution.module_banks.maze_bank import MazeModuleBank
from multimodal_mazes.evolution.module_banks.image_bank import ImageModuleBank
import random
import copy
import heapq
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import math

def _evaluate_genome_chunk(args):
    """
    Evaluate a chunk of genomes.
    Arguments:
        args (tuple): A tuple containing the genomes to evaluate and the task type.
    Returns:
        (list): A list of evaluation data for the evaluated genomes.
    """
    genomes, task = args
    if task == 'maze':
        return [genome_eval_fitness_maze(g) for g in genomes]
    elif task == 'image':
        return [genome_eval_fitness_square(g) for g in genomes]
    
class GenomeEADist:
    def __init__(self, hyperparameters = None):
        """
        Initialise the evolutionary algorithm.
        Arguments:
            hyperparameters (dict): The hyperparameters for evolution.
        Properties:
            generation (int): The current generation number.
            genomes (list): The current genome population.
            fittest_network (GenomeMotif): The fittest network in the current population.
            fittest_networks (list): The fittest networks in the current population.
            last_genome_id (int): The ID of the last generated genome.
            trial_data (dict): The data collected during the trials.
            population (int): The population size.
            top_genomes (int): The number of elite genomes.
            mutation_rate (float): The rate of mutation.
            crossover_rate (float): The rate of crossover.
            task (str): The task type for the evolutionary algorithm.
            _pool (ProcessPoolExecutor): The process pool for evaluating genomes.
        """
        self.generation = 0
        self.genomes = []
        self.fittest_network = None
        self.fittest_networks = []
        self.last_genome_id = 0
        self.trial_data = {}

        self.hyperparameters = hyperparameters
        self.population = hyperparameters['population_size']
        self.top_genomes = hyperparameters['top_genomes']
        self.mutation_rate = hyperparameters['mutation_rate']
        self.crossover_rate = hyperparameters['crossover_rate']
        self.task = hyperparameters['task']
        
        self.initialise_module_bank()

        self._pool = None

    def initialise_module_bank(self):
        """
        Initialise the module bank and module distribution for the population..
        Generates:
            self.module_bank (ModuleBank): The module bank containing all modules.
            self.mod_distribution (dict): The mapping of modules types to their distributions.
        """
        if self.task == 'maze':
            self.module_bank = MazeModuleBank()
        elif self.task == 'image':
            self.module_bank = ImageModuleBank(self.hyperparameters['class_type'])

        # Initialise module distribution
        # self.mod_distribution = {k: v[1] for k, v in self.module_bank.bank.items()}

    def initialise_process(self, args):
        """
        Initialise the process pool for evaluating genomes.
        Arguments:
            args (tuple): The arguments for the initializer.
        Generates:
            self._n_workers (int): The number of worker processes to use.
            self._chunk_size (int): The size of each chunk of genomes to process.
            self._pool (ProcessPoolExecutor): The process pool for evaluating genomes.
        """
        self._n_workers = max(1, os.cpu_count() - 1)
        self._chunk_size = math.ceil(self.population / self._n_workers)
        
        if self.task == 'maze':
            self._pool = ProcessPoolExecutor(
                max_workers = self._n_workers,
                initializer = _init_evaluator_maze,
                initargs = args
            )
        elif self.task == 'image':
            self._pool = ProcessPoolExecutor(
                max_workers = self._n_workers,
                initializer = _init_evaluator_square,
                initargs = args[:-1]
            )

    def generate(self, task_obj, sensor_noise_scale, n_steps=None): 
        """Generate the initial population of genomes.
        Args:
            task_obj (Maze or ImageClassification): The task object to evaluate the genomes on.
            sensor_noise_scale (float): Scale of noise to apply to the agent's sensors.
            n_steps (int): Number of steps to run the agent in the task.
        Generates:
            self.genomes (list): The current genome population.
            self.fittest_network (Genome): The fittest network in the current generation.
            self.fittest_networks (list): The fittest networks in the current generation.
        """
        id, genomes, hyperparams, mod_bank = self.last_genome_id, self.genomes, self.hyperparameters, self.module_bank
        # Generate initial population
        for _ in range(self.population):
            g = Genome(id, hyperparams, mod_bank)
            genomes.append(g)
            id += 1

        # Update fittest networks
        self.fittest_network = genomes[0]
        self.fittest_networks = genomes[:self.top_genomes]
        self.genomes = genomes

        if self._pool is None:
            self.initialise_process(args=(task_obj, sensor_noise_scale, n_steps))
            
    def evaluate(self):
        """
        Evaluate the fitness of each genome in the population.
        Generates:
            self.trial_data[generation] (dict): The data collected during the trials in the current generation.
        """
        genomes, chunk, task = self.genomes, self._chunk_size, self.task
        # Evaluate each genome in parallel in batches
        batches = [genomes[i:i + chunk] for i in range(0, len(genomes), chunk)]
        results_iter = self._pool.map(_evaluate_genome_chunk, [(batch, task) for batch in batches])

        # Collect results
        data = {}
        for batch, batch_results in zip(batches, results_iter):
            for g, res in zip(batch, batch_results):
                g.fitness = res['score']
                data[g.genome_id] = res
        self.trial_data[self.generation] = data

        self.update_fittest()
        self.update_module_distribution()

    def update_fittest(self):
        """
        Update the fittest genome based on the current generation.
        Updates:
            self.fittest_network (Genome): The fittest genome in the current generation.
            self.fittest_networks (list): The fittest genomes in the current generation.
        Generates:
            self.trial_data[generation]['fittest']: The fitness and genome of the fittest individual in the current generation.
            self.trial_data[generation]['fitness']: The average fitness of the fittest individuals in the current generation.
        """
        top, genomes = self.top_genomes, self.genomes
        # Sort genomes by fitness and update fittest networks
        self.fittest_networks = fittest_n = heapq.nlargest(top, genomes, key=lambda g: g.fitness)
        self.fittest_network = fittest = fittest_n[0]

        # Store fittest genome and average fitness in trial_data
        data = self.trial_data[self.generation]
        data['fittest'] = (fittest.fitness, fittest)
        data['fitness'] = np.mean([g.fitness for g in self.fittest_networks])

    def update_module_distribution(self):
        """
        Update the module distribution based on the current population.
        Updates:
            self.mod_distribution (dict): The mapping of modules to their distributions.
        Generates:
            self.trial_data[generation]['mod_distribution']: The module distribution of the fittest individuals in the current generation.
        """
        # Reset and calculate module distribution
        mot_dist = {m: 0 for m in self.module_bank.bank.keys()}
        for g in self.fittest_networks:
            for m, r in g.mod_dist.items():
                mot_dist[m] += np.array(r)

        # Normalize module distribution
        for _, c in mot_dist.items():
            c /= self.top_genomes

        self.trial_data[self.generation]['mod_distribution'] = copy.deepcopy(mot_dist)

    def evolve(self):
        """
        Evolve the population of genomes by selecting the fittest individuals and applying crossover and mutation.
        Updates:
            self.genomes (list): The current genome population.
            self.generation (int): The current generation number.
        """
        fittest_n, id = self.fittest_networks, self.last_genome_id
        # Calculate the number of new genomes to create via crossover and mutation
        n_new_genomes = self.population - self.top_genomes
        total_op = self.crossover_rate + self.mutation_rate
        p_cross = self.crossover_rate / total_op if total_op > 0 else 0
        n_cross = np.random.binomial(n_new_genomes, p_cross)
        
        # Select crossover and mutation parents
        parents_cross = random.choices(fittest_n, k=2 * n_cross)
        parents_mut = random.choices(fittest_n, k=n_new_genomes - n_cross)
        new_genomes = []

        # Crossover
        for p1, p2 in zip(parents_cross[::2], parents_cross[1::2]):
            child = p1.crossover(id, p2)
            new_genomes.append(child)
            id += 1

        # Mutation
        for p in parents_mut:
            child = p.clone(id)
            child.mutate()
            new_genomes.append(child)
            id += 1

        # Update genomes and generation
        self.last_genome_id = id
        self.genomes = fittest_n + new_genomes
        self.generation += 1

    def fitness_over_generations(self):
        """
        Return the fitness of the fittest genome over generations.
        Returns:
            (tuple): The generations and their corresponding fitness values.
        """
        t_data = self.trial_data
        generations = list(t_data.keys())
        fitness = [t_data[g]['fitness'] for g in generations]
        return generations, fitness

    def plot_fitness_over_generations(self):
        """Plot the fitness of the fittest genome over generations."""
        generations, fitness = self.fitness_over_generations()

        plt.plot(generations, fitness)
        plt.title('Fitness Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.ylim([0, 1])
        plt.show()

    def module_distribution_over_generations(self):
        """
        Return the module distribution over generations.
        Returns:
            (tuple): The generations and their corresponding module distributions.
        """
        t_data, mods = self.trial_data, self.module_bank.bank.keys()
        # Initialise generations and module distribution
        generations = list(t_data.keys())
        mod_dist = {mod: [] for mod in mods}

        # Populate module distribution over generations
        for gen in generations:
            for mod in mods:
                mod_dist[mod].append(t_data[gen]['mod_distribution'][mod])
        return generations, mod_dist

    def plot_module_distribution_over_generations(self):
        """Plot the module distribution over generations."""
        generations, mod_dist = self.module_distribution_over_generations()
        cmap = cm.get_cmap('viridis', len(mod_dist))
        mod_names = list(mod_dist.keys())

        for i, mod in enumerate(mod_names):
            plt.plot(generations, mod_dist[mod], label=mod, color=cmap(i))

        plt.title('Module Distribution Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Module Ratio')
        plt.legend()
        plt.show()
