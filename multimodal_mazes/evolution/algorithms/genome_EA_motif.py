from multimodal_mazes.evolution.genomes.genome_motifs import GenomeMotif as Genome
from multimodal_mazes.evolution.module_banks.motifs import MotifBank
from multimodal_mazes.evolution.evaluators.evaluator_motif import genome_eval_fitness as genome_eval_fitness_maze
from multimodal_mazes.evolution.evaluators.evaluator_motif import _init_evaluator as _init_evaluator_maze
from multimodal_mazes.evolution.previous_versions.evaluator_image_motif import genome_eval_fitness as genome_eval_fitness_square
from multimodal_mazes.evolution.previous_versions.evaluator_image_motif import _init_evaluator as _init_evaluator_square
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
    
class GenomeEAMotif:
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
        self.types = self.hyperparameters['motif_types']
        self.task = hyperparameters['task']
        
        self.motif_bank = MotifBank(self.hyperparameters['motif_types'])

        self._pool = None

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

    def generate(self, task_obj=None, sensor_noise_scale=None, n_steps=None): 
        """Generate the initial population of genomes.
        Args:
            task_obj (Maze): The task object to evaluate the genomes on.
            sensor_noise_scale (float): Scale of noise to apply to the agent's sensors.
            n_steps (int): Number of steps to run the agent in the task.
        Updates:
            self.genomes (list): The current genome population.
            self.fittest_networks (list): The fittest networks in the current generation.
            self.fittest_network (Genome): The fittest network in the current generation.
        """
        genomes, hyperparams, mod_bank = self.genomes, self.hyperparameters, self.motif_bank
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
        # Evaluate genomes in parallel in batches
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
        self.update_motif_distribution()

    def update_fittest(self):
        """
        Update the fittest genome based on the current generation.
        Updates:
            self.fittest_network (Genome): The fittest genome in the current generation.
            self.fittest_networks (list): The fittest genomes in the current generation
        Generates:
            self.trial_data[generation]['fittest']: The fitness and genome of the fittest individual in the current generation.
            self.trial_data[generation]['fitness']: The average fitness of the fittest individuals in the current generation.
        """
        top, genomes = self.top_genomes, self.genomes
        # Sort genomes by fitness and update fittest networks
        # self.fittest_networks = heapq.nlargest(self.top_genomes, self.genomes, key=lambda g: (g.fitness, -g.complexity))
        self.fittest_networks = fittest_n = heapq.nlargest(top, genomes, key=lambda g: g.fitness)
        self.fittest_network = fittest = fittest_n[0]

        # Store fittest genome and average fitness in trial_data
        data = self.trial_data[self.generation]
        data['fittest'] = (fittest.fitness, fittest)
        data['fitness'] = np.mean([g.fitness for g in fittest_n])

    def update_motif_distribution(self):
        """
        Update the motif distribution based on the current population.
        Generates:
            self.trial_data[generation]['mot_distribution']: The motif distribution of the fittest individuals in the current generation.
        """
        types, structs = self.types, self.motif_bank.structures
        # Reset and calculate motif distribution
        mot_dist = {t: np.zeros(len(structs[t])) for t in types}
        for g in self.fittest_networks:
            for t, r in g.mot_dist.items():
                mot_dist[t] += np.array(r)

        # Normalize motif distribution
        for _, c in mot_dist.items():
            c /= self.top_genomes

        self.trial_data[self.generation]['mot_distribution'] = copy.deepcopy(mot_dist)
        self.update_type_distribution()

    def update_type_distribution(self):
        """
        Compute the distribution of motif types in the population.
        Updates:
            self.type_distribution (dict): The mapping of motif types to their distribution.
        Generates:
            self.trial_data[generation]['type_distribution']: The type distribution of the population in the current generation.
        """
        # Reset and update type distribution
        t_dist = {type: 0 for type in self.types}
        for genome in self.genomes:
            for t, c in genome.mot_count.items():
                t_dist[t] += sum(c)

        # Normalize type distribution
        total = sum(t_dist.values())
        if total > 0:
            for t in self.types:
                t_dist[t] /= total

        self.trial_data[self.generation]['type_distribution'] = copy.deepcopy(t_dist)

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

    def motif_distribution_over_generations(self):
        """
        Return the motif distribution over generations.
        Returns:
            (tuple): The generations and their corresponding motif distributions.
        """
        t_data, types = self.trial_data, self.types
        # Initialise generations and motif distribution
        generations = list(t_data.keys())
        structs = self.motif_bank.structures
        mot_ratios = {t: {id: [] for id in range(len(structs[t]))} for t in types}

        # Populate motif distribution over generations
        for g in generations:
            for t in types:
                ratios = t_data[g]['mot_distribution'][t]
                for id, r in enumerate(ratios):
                    mot_ratios[t][id].append(r)

        return generations, mot_ratios

    def plot_motif_distribution_over_generations(self):
        """Plot the motif distribution over generations."""
        types = self.types
        generations, ratios = self.motif_distribution_over_generations()
        cmap = cm.get_cmap('viridis', len(types))
        type_colors = {t: cmap(i) for i, t in enumerate(types)}

        for t in types:
            for id, r in ratios[t].items():
                label = t if id == 0 else None
                plt.plot(generations, r, label=label, color=type_colors[t])

        plt.title('Motif Distribution Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Motif Ratio')
        plt.legend()
        plt.show()

    def type_distribution_over_generations(self):
        """
        Returns the type distribution over generations.
        Returns:
            (tuple): The generations and their corresponding type distributions.
        """
        t_data, types = self.trial_data, self.types
        # Initialise generations and type distribution
        generations = list(t_data.keys())
        type_ratios = {t: [] for t in types}

        # Populate type distribution over generations
        for gen in generations:
            type_dist = t_data[gen]['type_distribution']
            for t in types:
                type_ratios[t].append(type_dist[t])

        return generations, type_ratios
        