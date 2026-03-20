from multimodal_mazes.evolution.previous_versions.grammar import Grammar_Tree
import random
import multiprocessing as mp
import matplotlib.pyplot as plt

class Grammar_EA:
    def __init__(self, inputs = 8, outputs = 4, hyperparameters = None):
        self.inputs = inputs
        self.outputs = outputs

        self.population = hyperparameters['population_size']

        self.hyperparameters = hyperparameters
        self.generation = 0
        self.last_genome_id = 0
        self.genomes = []

        self.fittest_network = None
        self.fittest_networks = []

        self.trial_data = {}
    
    def generate(self): 
        """Generate the initial population of grammar trees."""
        for _ in range(self.population):
            genome = Grammar_Tree(self.last_genome_id, self.inputs, self.outputs, self.hyperparameters, True)
            self.genomes.append(genome)
            self.last_genome_id += 1

        self.fittest_network = self.genomes[0]
        self.fittest_networks = self.genomes[:self.hyperparameters['top_n']]

    def evaluate(self, evaluator, maze, sensor_noise_scale, n_steps):
        """Evaluate the fitness of each genome in the population.
        Args:
            evaluator: function to evaluate the genome.
            maze: the maze environment to evaluate the genomes in.
            sensor_noise_scale: scale of noise to apply to the agent's sensors.
            n_steps: number of steps to run the agent in the maze.
        """
        self.trial_data[self.generation] = {}
        results = {}
        max_processors = max(mp.cpu_count()-1, 1)
        pool = mp.Pool(processes=max_processors)

        for i, genome in enumerate(self.genomes):
            results[i] = pool.apply_async(evaluator, args=(genome, maze, sensor_noise_scale, n_steps))
            
        for i in range(len(results)):
            result = results[i].get()
            self.genomes[i].fitness = result['score']
            self.trial_data[self.generation][self.genomes[i].grammar_id] = result

        pool.close()
        pool.join()
        self.update_fittest()

    def update_fittest(self):
        """Update the fittest genome based on the current generation."""
        self.genomes.sort(key=lambda x: x.fitness, reverse=True)
        self.fittest_network = self.genomes[0]
        self.fittest_networks = self.genomes[:self.hyperparameters['top_n']]
        self.trial_data[self.generation]['fittest'] = self.fittest_network.fitness
    
    def evolve(self):
        """Evolve the population of grammar trees by selecting the fittest individuals and applying crossover and mutation."""
        self.genomes = self.genomes[:self.hyperparameters['top_genomes']]
        new_genomes = []   
        n_new_genomes = self.population - len(self.genomes)
        
        while len(new_genomes) < n_new_genomes:
            for genome in self.genomes:
                if random.random() < self.hyperparameters['crossover_rate']:
                    parent_idx = random.randint(0, len(self.genomes) - 1)  
                    crossover_offspring = genome.crossover(self.last_genome_id, self.genomes[parent_idx])
                    new_genomes.append(crossover_offspring)
                    self.last_genome_id += 1

                if random.random() < self.hyperparameters['mutation_rate']:
                    new_genome = genome.clone(self.last_genome_id)
                    new_genome.mutate()
                    new_genomes.append(new_genome)
                    self.last_genome_id += 1

        self.genomes.extend(new_genomes)
        self.genomes = self.genomes[:self.population]
        self.generation += 1

    def plot_fitness_over_generations(self):
        """Plot the fitness of the fittest genome over generations."""
        generations = list(self.trial_data.keys())
        fitness = [self.trial_data[gen]['fittest'] for gen in generations]

        plt.plot(generations, fitness)
        plt.title('Fitness Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.ylim([0, 1])
        plt.show()
