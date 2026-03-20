from multimodal_mazes.evolution.previous_versions.genome_motifs import GenomeMotif as Genome
from multimodal_mazes.evolution.module_banks.motifs import MotifBank, Motif
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
        list: A list of evaluation data for the evaluated genomes.
    """
    genomes, task = args
    if task == 'maze':
        return [genome_eval_fitness_maze(g) for g in genomes]
    elif task == 'image':
        return [genome_eval_fitness_square(g) for g in genomes]
    
class GenomeEAMotif:
    def __init__(self, hyperparameters = None):
        """
        Initialise the evolutionary algorithm manager for genome evolution.
        Arguments:
            hyperparameters (dict): A dictionary containing the hyperparameters for the evolution process.
        Properties:
            generation (int): The current generation number.
            genomes (list): The list of genomes currently in the population.
            fittest_network (GenomeMotif): The fittest network in the current population.
            fittest_networks (list): The list of the fittest networks in the current population.
            last_genome_id (int): The ID of the last generated genome.
            trial_data (dict): The data collected during the trials.
            population (int): The size of the population.
            top_genomes (int): The number of top genomes to keep.
            mutation_rate (float): The rate of mutation.
            crossover_rate (float): The rate of crossover.
            task (str): The task type for the evolutionary algorithm.
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
        
        self.initialise_motif_distribution()

        self._pool = None

    def initialise_motif_distribution(self):
        """
        Initialise the motif manager and motif distribution for the population.
        Generates:
            motif_manager (MotifBank): The motif manager for the population.
            types (list): The list of motif types.
            mot_count (dict): The count of each motif type in the population.
            mot_distribution (dict): The distribution of each motif type in the population.
        """
        # Instantiate motif manager
        self.motif_manager = MotifBank(self.hyperparameters['motif_types'])
        
        # Set types and counts
        self.types = self.hyperparameters['motif_types']
        n_types = len(self.hyperparameters['motif_types'])
        n_motifs = self.hyperparameters['n_motifs']

        # Initialize motif counts and distributions
        self.mot_count = {type: np.zeros(len(self.motif_manager.motif_bank[type])) for type in self.types}
        self.mot_distribution = {type: np.zeros(len(self.motif_manager.motif_bank[type])) for type in self.types}
        
        # Initialise all modules as one type for homogenous case
        if self.hyperparameters['homogeneous']:
            self.mot_count[self.types[0]][0] = n_motifs
        
        # Initialise modules as even distribution of simplest motif of each type
        else: 
            count = 0

            # Evenly distribute simplest motifs
            for type, id in self.motif_manager.simplest_motifs.items():
                n_mot = n_motifs // n_types
                self.mot_count[type][id] = n_mot
                count += n_mot

            # If there are remaining motifs, randomly assign from simplest motifs
            if count < n_motifs:
                types = np.random.choice(list(self.types), n_motifs - count, replace=True)
                for type in types:
                    id = self.motif_manager.simplest_motifs[type]
                    self.mot_count[type][id] += 1

            # Normalize distribution
            for type, counts in self.mot_count.items():
                self.mot_distribution[type] = counts / n_motifs

    def initialise_process(self, args):
        """
        Initialise the process pool for evaluating genomes.
        Arguments:
            args (tuple): A tuple containing the arguments for the initializer.
        Generates:
            _n_workers (int): The number of worker processes to use.
            chunk_size (int): The size of each chunk of genomes to process.
            _pool (ProcessPoolExecutor): The process pool for evaluating genomes.
        """
        self._n_workers = max(1, os.cpu_count() - 1)
        self.chunk_size = math.ceil(self.population / self._n_workers)

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

    def generate(self, task_obj=None, sensor_noise_scale=None, n_steps=None): 
        """Generate the initial population of genomes.
        Args:
            task_obj (Maze or ImageClassification): the task environment to evaluate the genomes in.
            sensor_noise_scale (float): scale of noise to apply to the agent's sensors.
            n_steps (int): number of steps to run the agent in the task.
        Updates:
            genomes (list): The list of genomes in the current generation.
            fittest_networks (list): The list of the fittest networks in the current generation.
            fittest_network (Genome): The fittest network in the current generation.
            _pool (ProcessPoolExecutor): The process pool for evaluating genomes.
        """
        motif_objects = self.mot_count, self.mot_distribution

        # Generate genomes
        for _ in range(self.population):
            genome = Genome(self.last_genome_id, self.motif_manager, motif_objects, self.hyperparameters)
            self.genomes.append(genome)
            self.last_genome_id += 1

        # Update fittest networks
        self.fittest_network = self.genomes[0]
        self.fittest_networks = self.genomes[:self.top_genomes]

        if self._pool is None:
            self.initialise_process(args=(task_obj, sensor_noise_scale, n_steps))

    def evaluate(self):
        """
        Evaluate the fitness of each genome in the population.
        Generates:
            trial_data (dict): A dictionary containing the trial data for the current generation.
        """
        self.trial_data[self.generation] = {}

        # Evaluate genomes in parallel in batches
        batches = [self.genomes[i:i + self.chunk_size] for i in range(0, len(self.genomes), self.chunk_size)]
        results_iter = self._pool.map(_evaluate_genome_chunk, [(batch, self.task) for batch in batches])

        # Collect results
        for batch, batch_results in zip(batches, results_iter):
            for genome, res in zip(batch, batch_results):
                genome.fitness = res['score']
                self.trial_data[self.generation][genome.genome_id] = res

        self.update_fittest()
        self.update_motif_distribution()

    def update_fittest(self):
        """
        Update the fittest genome based on the current generation.
        Updates:
            fittest_network (Genome): The fittest genome in the current generation.
            fittest_networks (list): The list of the top fittest genomes in the current generation.
        Generates:
            trial_data[generation]['fittest']: A tuple containing the fitness and genome of the fittest individual.
            trial_data[generation]['fitness']: The average fitness of the fittest individuals.
        """
        # Sort genomes by fitness and update fittest networks
        self.fittest_networks = heapq.nlargest(self.top_genomes, self.genomes, key=lambda g: (g.fitness, -g.complexity))
        self.fittest_network = self.fittest_networks[0]

        # Store fittest genome and average fitness in trial_data
        self.trial_data[self.generation]['fittest'] = (self.fittest_network.fitness, self.fittest_network)
        self.trial_data[self.generation]['fitness'] = np.mean([g.fitness for g in self.fittest_networks])

    def update_motif_distribution(self):
        """
        Update the motif distribution based on the current population.
        Updates:
            mot_distribution (dict): The updated motif distribution.
        Generates:
            trial_data[generation]['mot_distribution']: The motif distribution of the fittest individuals.
        """
        # Reset and calculate motif distribution
        self.mot_distribution = {type: np.zeros(len(self.motif_manager.motif_bank[type])) for type in self.types}
        for genome in self.fittest_networks:
            for type, ratios in genome.mot_dist.items():
                self.mot_distribution[type] += np.array(ratios)

        # Normalize motif distribution
        for type, counts in self.mot_distribution.items():
            counts /= self.top_genomes

        self.trial_data[self.generation]['mot_distribution'] = copy.deepcopy(self.mot_distribution)
        self.update_type_distribution()

    def update_type_distribution(self):
        """
        Compute the distribution of motif types in the population.
        Updates:
            type_distribution (dict): The updated type distribution.
        Generates:
            trial_data[generation]['type_distribution']: The type distribution of the population.
        """
        # Reset and update type distribution
        type_distribution = {type: 0 for type in self.types}
        for genome in self.genomes:
            for type, count in genome.mot_count.items():
                type_distribution[type] += sum(count)

        # Normalize type distribution
        total_count = sum(type_distribution.values())
        if total_count > 0:
            for type in self.types:
                type_distribution[type] /= total_count

        self.trial_data[self.generation]['type_distribution'] = copy.deepcopy(type_distribution)

    def evolve(self):
        """
        Evolve the population of genomes by selecting the fittest individuals and applying crossover and mutation.
        Updates:
            genomes (list): The list of genomes in the current generation.
            generation (int): The current generation number.
        """
        # Calculate the number of new genomes to create via crossover and mutation
        n_new_genomes = self.population - self.top_genomes
        total_op = self.crossover_rate + self.mutation_rate
        p_cross = self.crossover_rate / total_op if total_op > 0 else 0
        n_cross = np.random.binomial(n_new_genomes, p_cross)

        # Select crossover and mutation parents
        parents_cross = random.choices(self.fittest_networks, k=2 * n_cross)
        parents_mut = random.choices(self.fittest_networks, k=n_new_genomes - n_cross)
        new_genomes = []   

        # Crossover
        for p1, p2 in zip(parents_cross[::2], parents_cross[1::2]):
            child = p1.crossover(self.last_genome_id, p2)
            new_genomes.append(child)
            self.last_genome_id += 1

        # Mutation
        for p in parents_mut:
            child = p.clone(self.last_genome_id)
            child.mutate()
            new_genomes.append(child)
            self.last_genome_id += 1

        # Update genomes and generation
        self.genomes = self.fittest_networks + new_genomes
        self.generation += 1

    def fitness_over_generations(self):
        """
        Get the fitness of the fittest genome over generations.
        Returns:
            tuple: A tuple containing the generations and their corresponding fitness values.
        """
        generations = list(self.trial_data.keys())
        fitness = [self.trial_data[gen]['fitness'] for gen in generations]
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
        Get the motif distribution over generations.
        Returns:
            tuple: A tuple containing the generations and their corresponding motif distributions.
        """
        # Initialise generations and motif distribution
        generations = list(self.trial_data.keys())
        mot_ratios = {type: {id: [] for id in range(len(self.motif_manager.motif_bank[type]))} for type in self.types}

        # Populate motif distribution over generations
        for gen in generations:
            for type in self.types:
                ratios = self.trial_data[gen]['mot_distribution'][type]
                for id, ratio in enumerate(ratios):
                    mot_ratios[type][id].append(ratio)

        return generations, mot_ratios

    def plot_motif_distribution_over_generations(self):
        """Plot the motif distribution over generations."""
        generations, mot_ratios = self.motif_distribution_over_generations()
        cmap = cm.get_cmap('viridis', len(self.types))
        type_colors = {type: cmap(i) for i, type in enumerate(self.types)}
        
        for type in self.types:
            for id, ratios in mot_ratios[type].items():
                label = type if id == 0 else None
                plt.plot(generations, ratios, label=label, color=type_colors[type])

        plt.title('Motif Distribution Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Motif Ratio')
        plt.legend()
        plt.show()

    def type_distribution_over_generations(self):
        """Get the type distribution over generations."""
        # Initialise generations and type distribution
        generations = list(self.trial_data.keys())
        type_ratios = {type: [] for type in self.types}

        # Populate type distribution over generations
        for gen in generations:
            type_dist = self.trial_data[gen]['type_distribution']
            for type in self.types:
                type_ratios[type].append(type_dist[type])

        return generations, type_ratios
        