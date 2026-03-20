import random
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import bisect
from enum import IntEnum
from collections import defaultdict
from multimodal_mazes.evolution.module_banks.motifs import Motif

class RuleType(IntEnum):
    """Enumeration for different rule types in the genome."""
    MODULE = 0
    CONNECT_IN = 1
    CONNECT_OUT = 2

# Motif tuples
# (
#     ('rule_type', np.int32), # 0: motif, 1: connect_in, 2: connect_out
#     ('motif_id', np.int32),  # motif ID in network
# )

# Input connection tuples
# (
    # ('rule_type', np.int32), # 0: motif, 1: connect_in, 2: connect_out
    # ('src', np.int32),       # source node ID: network input
    # ('dst', np.int32),       # destination node ID: motif_id
    # ('dst_port', np.int32),  # motif input port
    # ('weight', np.float32)   # weight of the connection
# )

# Output connection tuples
# (
#     ('rule_type', np.int32), # 0: motif, 1: connect_in, 2: connect_out
#     ('src', np.int32),       # source node ID: motif_id
#     ('src_port', np.int32),  # motif output port
#     ('dst', np.int32),       # destination node ID: network output
#     ('weight', np.float32)   # weight of the connection
# )

# Complexity definition:
# Feedforward edge adds 1 to complexity
# Recurrent and temporal edges add 2 to complexity

class GenomeMotif():
    def __init__(self, genome_id, motif_manager, motif_objects,  hyperparameters):
        """
        Initialise a genome with motifs and connection rules.
        Arguments:
            genome_id (int): The ID of the genome.
            motif_manager (MotifManager): The manager for handling motifs.
            motif_objects (list): The list of motif objects.
            hyperparameters (dict): The hyperparameters for the genome.
        Properties:
            fitness (float): The fitness of the genome.
            complexity (float): The complexity of the genome.
            offspring_hyperparameters (dict): The hyperparameters for the offspring genomes.
            n_inputs (int): The number of inputs to the genome.
            n_outputs (int): The number of outputs from the genome.
            n_motifs (int): The number of motifs in the genome.
            weight_sharing (bool): Whether to use weight sharing in the genome.
            uniform_weights (bool): Whether to use uniform weights in the genome.
            mot_grouped_in (dict): The input connections grouped by destination motif.
            out_grouped_out (dict): The output connections grouped by destination output.
            task (str): The task to be solved by the genome.
            types (list): The types of motifs in the genome.
            mot_count (dict): The count of each motif type in the genome.
            mot_dist (dict): The distribution of each motif type in the genome.
            homogeneous (bool): Whether the genome is homogeneous.
            motifs (list): The list of motifs in the genome.
            n_motif_types (int): The number of motif types in the genome.
            one_to_one (bool): Whether the genome connections are one-to-one.
            max_conn_in (int): The maximum number of input connections.
            max_conn_out (int): The maximum number of output connections.
            mot_rules (list): The rules for motif connections.
            conn_in_rules (list): The rules for input connections.
            conn_out_rules (list): The rules for output connections.
            compile_flag (list): The flags for compilation.
            connectivity (str): The connectivity pattern of the genome.
            connection_density (float): The density of connections in the genome.
        """
        self.genome_id = genome_id
        self.fitness = 0.0
        self.complexity = 0.0 

        self.motif_manager = motif_manager
        
        self.hyperparameters = hyperparameters
        self.offspring_hyperparameters = dict(hyperparameters)
        self.n_inputs = hyperparameters['n_inputs']
        self.n_outputs = hyperparameters['n_outputs']
        self.n_motifs = hyperparameters['n_motifs']
        self.weight_sharing = hyperparameters['weight_sharing']
        self.uniform_weights = hyperparameters['uniform_weights']

        self.mot_grouped_in = {i: [] for i in range(self.n_motifs)} # Group input connections by destination motif
        self.out_grouped_out = {i: [] for i in range(self.n_outputs)} # Group output connections by destination output

        self.task = hyperparameters['task']

        self.types = hyperparameters['motif_types']
        self.mot_count, self.mot_dist = motif_objects
        self.homogeneous = hyperparameters['homogeneous']
        self.motifs = []
        self.initialise_motifs()
        self.n_motif_types = hyperparameters['n_motif_types']

        if self.weight_sharing:
            self.initialise_weight_sharing()

        self.one_to_one = hyperparameters['one_to_one']
        if self.one_to_one:
            self.initialise_one_to_one()

        self.max_conn_in = self.n_inputs * sum([mot.n_inputs for mot in self.motifs]) if not self.one_to_one else self.n_inputs
        self.max_conn_out = self.n_outputs * sum([mot.n_outputs for mot in self.motifs]) if not self.one_to_one else self.n_outputs

        self.mot_rules = []
        self.conn_in_rules = []
        # self.conn_mot_rules = [] # Connections between motifs 
        self.conn_out_rules = []

        self.compile_flag = [1, 1] # [input, output]

        self.connectivity = hyperparameters['connectivity']
        self.connection_density = hyperparameters['connection_density']
        self.offspring_hyperparameters['connectivity'] = 'UNCONNECTED'
        self.initialise_genome()

    def initialise_motifs(self):
        """Initialise the motifs in the genome based on the motif manager."""
        if self.mot_count:
            for mot_type, counts in self.mot_count.items():
                for id, count in enumerate(counts):
                    while count > 0:
                        n_inputs, n_outputs = self.motif_manager.motifs[mot_type][id].n_inputs, self.motif_manager.motifs[mot_type][id].n_outputs
                        complexity = self.motif_manager.motifs[mot_type][id].complexity
                        self.motifs.append(Motif(mot_type, id, n_inputs, n_outputs, complexity))
                        self.complexity += complexity
                        count -= 1

        self.motifs = np.random.permutation(self.motifs)

    def initialise_weight_sharing(self):
        """Initialise weight sharing for input and output connections."""
        if self.uniform_weights:
            self.shrd_mot_in_wts = {mot_type: [1.0 for _ in range(len(self.motif_manager.motif_bank[mot_type]))] for mot_type in self.types}
            self.shrd_mot_out_wts = {mot_type: [1.0 for _ in range(len(self.motif_manager.motif_bank[mot_type]))] for mot_type in self.types}
        else:
            self.shrd_mot_in_wts = {mot_type: np.random.uniform(0.1, 1.0, len(self.motif_manager.motif_bank[mot_type])) for mot_type in self.types}
            self.shrd_mot_out_wts = {mot_type: np.random.uniform(0.1, 1.0, len(self.motif_manager.motif_bank[mot_type])) for mot_type in self.types}

    def initialise_one_to_one(self):
        """Initialise one-to-one connections for the genome."""
        self.input_set = set(range(self.n_inputs))
        self.used_inputs = set()
        self.output_set = set(range(self.n_outputs))
        self.used_outputs = set()
        self.mot_in_set = set((i, j) for i in range(self.n_motifs) for j in range(self.motifs[i].n_inputs))
        self.used_mot_in = set()
        self.mot_out_set = set((i, j) for i in range(self.n_motifs) for j in range(self.motifs[i].n_outputs))
        self.used_mot_out = set()

    def add_motif_rule(self, mot_rule):
        """Add a motif rule to the genome."""
        self.mot_rules.append(mot_rule)

    def add_motif_rules(self, rules):
        """Add multiple motif rules to the genome."""
        for rule in rules:
            self.add_motif_rule(rule)

    def add_connect_rule(self, conn_rule):
        """Add a connection rule to the genome.
        Arguments:
            conn_rule (tuple): A tuple representing the connection rule.
        """
        match conn_rule[0]:
            case 1:  
                _, src, dst, dst_pt, wt = conn_rule
                if any(t[0] == dst_pt and t[1] == src for t in self.mot_grouped_in[dst]):
                    return
                bisect.insort(self.mot_grouped_in[dst], (dst_pt, src, wt))
                self.conn_in_rules.append(conn_rule)
                self.compile_flag[0] = 1
                self.complexity += 1

                if self.one_to_one:
                    self.used_mot_in.add((dst, dst_pt))
                    self.used_inputs.add(src)
            case 2:  
                _, src, src_pt, dst, wt = conn_rule
                if any(t[0] == src_pt and t[1] == src for t in self.out_grouped_out[dst]):
                    return
                bisect.insort(self.out_grouped_out[dst], (src_pt, src, wt))
                self.conn_out_rules.append(conn_rule)
                self.compile_flag[1] = 1
                self.complexity += 1

                if self.one_to_one:
                    self.used_mot_out.add((src, src_pt))
                    self.used_outputs.add(dst)

    def remove_connect_rule(self, conn_rule):
        """Remove a connection rule from the genome.
        Arguments:
            conn_rule (tuple): A tuple representing the connection rule to remove.
        """
        match conn_rule[0]:
            case 1:
                _, src, dst, dst_pt, wt = conn_rule
                self.mot_grouped_in[dst].remove((dst_pt, src, wt))
                self.compile_flag[0] = 1
                self.complexity -= 1

                if self.one_to_one:
                    self.used_mot_in.remove((dst, dst_pt))
                    self.used_inputs.remove(src)
            case 2:
                _, src, src_pt, dst, wt = conn_rule
                self.out_grouped_out[dst].remove((src_pt, src, wt))
                self.compile_flag[1] = 1
                self.complexity -= 1

                if self.one_to_one:
                    self.used_mot_out.remove((src, src_pt))
                    self.used_outputs.remove(dst)

    def add_connect_in_rules(self, rules, group = 0, size = 0):
        """Add multiple input connection rules to the genome.
        Arguments:
            rules (list): List of connection rules to add.
            group (int): Group type for the rules (0: MODULES, 1: INPUTS).
            size (int): Number of rules to generate if no rules are provided.
        """
        if rules:
            for rule in rules:
                self.add_connect_rule(rule)
        else:
            match group:
                case 0: 
                    motif_ids = list(range(self.n_motifs))
                    pts = [np.random.choice(self.motifs[i].n_inputs) for i in motif_ids]
                case 1: 
                    if self.one_to_one:
                        size = min(size, len(self.mot_in_set), len(self.input_set))
                        mots = random.sample(list(self.mot_in_set), size)
                        motif_ids = [mot[0] for mot in mots]
                        pts = [mot[1] for mot in mots]
                    else:
                        motif_ids = np.random.choice(range(self.n_motifs), size, replace=True)
                        pts = [np.random.choice(self.motifs[i].n_inputs) for i in motif_ids]
            
            srcs = np.random.choice([i for i in range(self.n_inputs) for _ in range(int(np.ceil(size / self.n_inputs)))], size, replace=False)
            wts = [self.shrd_mot_in_wts[self.motifs[i].motif_type][self.motifs[i].id] for i in motif_ids] if self.weight_sharing else np.random.uniform(0.1, 1.0, size=size)

            for src, dst, dst_pt, wt in zip(srcs, motif_ids, pts, wts):
                rule = 1, src, dst, dst_pt, wt
                self.add_connect_rule(rule)
        
    def add_connect_out_rules(self, rules, group = 0, size = 0):
        """Add multiple outputs connection rules to the genome.
        Arguments:
            rules (list): List of connection rules to add.
            group (int): Group type for the rules (0: MODULES, 1: OUTPUTS).
            size (int): Number of rules to generate if no rules are provided.
        """
        if rules:
            for rule in rules:
                self.add_connect_rule(rule)
        else:
            match group:
                case 0:  
                    motif_ids = list(range(self.n_motifs))
                    pts = [np.random.choice(self.motifs[i].n_outputs) for i in motif_ids]
                case 1:  
                    if self.one_to_one:
                        size = min(size, len(self.mot_out_set), len(self.output_set))
                        mots = random.sample(list(self.mot_out_set), size)
                        motif_ids = [mot[0] for mot in mots]
                        pts = [mot[1] for mot in mots]
                    else:
                        motif_ids = np.random.choice(range(self.n_motifs), size, replace=True)
                        pts = [np.random.choice(self.motifs[i].n_outputs) for i in motif_ids]

            dsts = np.random.choice([i for i in range(self.n_outputs) for _ in range(int(np.ceil(size / self.n_outputs)))], size, replace=False)
            wts = [self.shrd_mot_out_wts[self.motifs[i].motif_type][self.motifs[i].id] for i in motif_ids] if self.weight_sharing else np.random.uniform(0.1, 1.0, size=size)

            for src, src_pt, dst, wt in zip(motif_ids, pts, dsts, wts):
                rule = 2, src, src_pt, dst, wt
                self.add_connect_rule(rule)

    def compile_rules(self):
        """Compile network rules into a ragged array for evaluation."""
        if self.compile_flag[0]:
            self.compile_conn_in_rules()
        if self.compile_flag[1]:
            self.compile_conn_out_rules()
        self.compile_flag = [0, 0]  

    def compile_conn_in_rules(self):
        """Compile input connection rules into a ragged array."""
        self.in_srcs, self.in_pts, self.in_wts = np.empty(len(self.conn_in_rules), dtype=np.int32), np.empty(len(self.conn_in_rules), dtype=np.int32), np.empty(len(self.conn_in_rules), dtype=np.float32)
        self.in_indptr = np.zeros(self.n_motifs + 1, dtype=np.int32)
        idx = 0

        for motif_id in range(self.n_motifs):
            for pt, src, wt in self.mot_grouped_in[motif_id]:
                self.in_srcs[idx], self.in_pts[idx], self.in_wts[idx] = src, pt, wt
                idx += 1
            self.in_indptr[motif_id + 1] = idx

    def compile_conn_out_rules(self):
        """Compile output connection rules into a ragged array."""
        self.out_srcs, self.out_pts, self.out_wts = np.empty(len(self.conn_out_rules), dtype=np.int32), np.empty(len(self.conn_out_rules), dtype=np.int32), np.empty(len(self.conn_out_rules), dtype=np.float32)
        self.out_indptr = np.zeros(self.n_outputs + 1, dtype=np.int32)
        idx = 0

        for out_id in range(self.n_outputs):
            for pt, src, wt in self.out_grouped_out[out_id]:
                self.out_srcs[idx], self.out_pts[idx], self.out_wts[idx] = src, pt, wt
                idx += 1
            self.out_indptr[out_id + 1] = idx

    def initialise_genome(self):
        """Initialise the genome."""
        for i, mot in enumerate(self.motifs):
            rule = 0, i
            self.add_motif_rule(rule)

        match self.connectivity:
            case 'UNCONNECTED': return
            case 'SPARSE':
                # Sparse connectivity: Connect each motif with one input and one output
                self.add_connect_in_rules([], 0, self.n_motifs)
                self.add_connect_out_rules([], 0, self.n_motifs)

            case 'RANDOM':
                # Random connectivity: Connect a random subset of inputs and outputs to random motifs
                n_conn_in = int(self.connection_density['input_density'] * self.n_inputs)
                n_conn_out = int(self.connection_density['output_density'] * self.n_outputs)
                self.add_connect_in_rules([], 1, n_conn_in)
                self.add_connect_out_rules([], 1, n_conn_out)

            case 'FULLY CONNECTED':
                # Fully connected: Connect all inputs and outputs to all motifs
                for i, mot in enumerate(self.motifs):
                    in_wt, out_wt = self.shrd_mot_in_wts[mot.motif_type][mot.id], self.shrd_mot_out_wts[mot.motif_type][mot.id] if self.weight_sharing else np.random.uniform(0.1, 1.0), np.random.uniform(0.1, 1.0)
                    for pt in range(mot.n_inputs):
                        for src in range(self.n_inputs):
                            rule = 1, src, i, pt, in_wt
                            self.add_connect_rule(rule)

                    for pt in range(mot.n_outputs):
                        for dst in range(self.n_outputs):
                            rule = 2, i, pt, dst, out_wt
                            self.add_connect_rule(rule)

            case 'IDEAL':
                if self.task == 'maze':
                    for i in range(self.n_inputs):
                        mot = i // 2
                        self.add_connect_rule((1, i, mot, 0, 1.0))
                        self.add_connect_rule((1, i, mot, 1, 1.0))
                    
                    for i in range(self.n_outputs):
                        self.add_connect_rule((2, i, 0, i, 1.0))

                if self.task == 'image':
                    side_len = int(np.sqrt(self.n_inputs))
                    mot_side_len = int(np.sqrt(side_len))
                    for row in range(side_len):
                        for col in range(side_len):
                            in_idx = row * side_len + col
                            mot = (row // mot_side_len) * mot_side_len + (col // mot_side_len)
                            prt = col % mot_side_len + (row % mot_side_len) * mot_side_len
                            self.add_connect_rule((1, in_idx, mot, prt, 1.0))  
                            self.add_connect_rule((2, mot, prt, in_idx, 1.0))  

    def forward_pass(self, input_vector):
        """Forward pass through the genome using execution plan.
        Arguments:
            input_vector (list): Input vector of size equal to the number of inputs.
        Returns:
            outputs (list): Output vector of size equal to the number of outputs.
        """
        if sum(self.compile_flag) > 0:
            self.compile_rules()

        total_nodes = self.n_inputs + self.n_motifs + self.n_outputs
        values = np.zeros(total_nodes, dtype=np.float32)
        values[:self.n_inputs] = input_vector
        
        mot_outputs = self.layer_forward_pass(0, values[:self.n_inputs], self.n_motifs)
        outputs = self.layer_forward_pass(1, mot_outputs, self.n_outputs)
        values[-self.n_outputs:] = outputs

        return outputs
    
    def layer_forward_pass(self, layer, in_vec, size):
        """Forward pass through a specific layer of the genome.
        Arguments:
            layer (int): The layer type (0 for 'MODULE' or 1 for 'OUTPUT').
            in_vec (np.ndarray): The input vector for the layer.
            size (int): The size of the layer.
        Returns:
            np.array: The output vector after processing the layer.
        """
        match layer:
            case 0: 
                inter_vec = [np.zeros(mot.n_inputs, dtype=np.float32) for mot in self.motifs]
                out_vec = [np.zeros(mot.n_outputs, dtype=np.float32) for mot in self.motifs]
                indptr, layer_srcs, layer_pts, layer_wts = self.in_indptr, self.in_srcs, self.in_pts, self.in_wts
                
            case 1: 
                inter_vec = [0.0 for _ in range(size)]
                out_vec = [0.0 for _ in range(size)]
                indptr, layer_srcs, layer_pts, layer_wts = self.out_indptr, self.out_srcs, self.out_pts, self.out_wts
                
        for i in range(size):
            start, end = indptr[i], indptr[i + 1]
            if start == end:
                continue
            srcs, pts, wts = layer_srcs[start:end], layer_pts[start:end], layer_wts[start:end]
            
            if layer == 0:
                for src, pt, wt in zip(srcs, pts, wts):
                    inter_vec[i][pt] += in_vec[src] * wt
                out_vec[i] = self.motif_manager.forward_pass(self.motifs[i], inter_vec[i])
            elif layer == 1:
                for src, pt, wt in zip(srcs, pts, wts):
                    inter_vec[i] += in_vec[src][pt] * wt
                out_vec[i] = inter_vec[i]

        return np.array(out_vec)

    def crossover(self, new_id, parent_2):
        """Perform crossover between two genomes to create a new child.
        Arguments:
            new_id (int): ID for the new child genome.
            parent_2 (Genome): The second parent genome for crossover.
        Returns:
            child (Genome): A new child genome resulting from the crossover.
        """
        cut = np.random.randint(1, min(self.n_motifs, parent_2.n_motifs))
        motifs = copy.deepcopy(self.motifs[:cut]) + copy.deepcopy(parent_2.motifs[cut:])
        motif_objects = (None, None) 
        child = GenomeMotif(new_id, self.motif_manager, motif_objects, self.offspring_hyperparameters)
        child.motifs = motifs
        child.count_motifs()
        child.recompute_motif_distribution()
        child.reset()

        self.cut_rules(child, cut, self.mot_rules, parent_2.mot_rules, 0)
        self.cut_rules(child, cut, self.conn_in_rules, parent_2.conn_in_rules, 1)
        self.cut_rules(child, cut, self.conn_out_rules, parent_2.conn_out_rules, 2)

        return child

    def cut_rules(self, child, cut, rule_list1, rule_list2, rule_type):
        """Cut rules at a specified index.
        Arguments:
            cut (int): The index to cut the rules at.
            rule_list1 (list): The list of rules to cut from parent 1.
            rule_list2 (list): The list of rules to cut from parent 2.
            rule_type (RuleType): The type of rules being cut.
        """
        match rule_type:
            case 0: 
                rules = [rule for rule in rule_list1 if rule[1] < cut] + [rule for rule in rule_list2 if rule[1] > cut]
                child.add_motif_rules(rules)
            case 1: 
                if self.one_to_one:
                    used_in, used_mot_in = set(), set()
                    for rule in rule_list1:
                        if rule[2] < cut and rule[1] not in used_in and (rule[2], rule[3]) not in used_mot_in:
                            used_in.add(rule[1])
                            used_mot_in.add((rule[2], rule[3]))
                            child.add_connect_rule(rule)
                    for rule in rule_list2:
                        if rule[2] > cut and rule[1] not in used_in and (rule[2], rule[3]) not in used_mot_in:
                            used_in.add(rule[1])
                            used_mot_in.add((rule[2], rule[3]))
                            child.add_connect_rule(rule)        
                else:
                    rules = [rule for rule in rule_list1 if rule[2] < cut] + [rule for rule in rule_list2 if rule[2] > cut]
                    child.add_connect_in_rules(rules)
            case 2: 
                used_out, used_mot_out = set(), set()
                if self.one_to_one:
                    for rule in rule_list1:
                        if rule[1] < cut and (rule[1], rule[2]) not in used_mot_out and rule[3] not in used_out:
                            used_mot_out.add((rule[1], rule[2]))
                            used_out.add(rule[3])
                            child.add_connect_rule(rule)
                    for rule in rule_list2:
                        if rule[1] > cut and (rule[1], rule[2]) not in used_mot_out and rule[3] not in used_out:
                            used_mot_out.add((rule[1], rule[2]))
                            used_out.add(rule[3])
                            child.add_connect_rule(rule)
                else:
                    rules = [rule for rule in rule_list1 if rule[1] < cut] + [rule for rule in rule_list2 if rule[1] > cut]
                    child.add_connect_out_rules(rules)

    def mutate(self):
        """Mutate the genome by randomly motifying its structure."""
        if not self.conn_in_rules or not self.conn_out_rules:
            mut = 0
        else:
            mut = np.random.choice([0, 1, 2, 3], p=[0.4, 0.05, 0.3, 0.25])
            # mut = np.random.choice([0, 1, 2, 3], p=[0, 0, 0, 1.0])
            
        match mut:
            case 0: self.add_connection()
            case 1: self.remove_connection()
            case 2: self.swap_connection()
            case 3: self.swap_motif()
            case 4: self.modify_connection_weight()

    def add_connection(self):
        """Add a new connection from a genome input or to a genome output."""
        conn_type = np.random.randint(1, 3)  # 1: CONNECT_IN, 2: CONNECT_OUT
        
        match conn_type:
            case 1:
                if len(self.conn_in_rules) >= self.max_conn_in:
                    self.swap_connection(conn_type=1)
                    return
                if self.one_to_one:
                    motif_id, dst_pt = random.sample(list(self.mot_in_set - self.used_mot_in), 1)[0]
                    mot = self.motifs[motif_id]
                    src = np.random.choice(list(self.input_set - self.used_inputs))
                else:
                    motif_id = np.random.randint(0, self.n_motifs)
                    mot = self.motifs[motif_id]
                    dst_pt = np.random.randint(0, mot.n_inputs)
                    src = np.random.randint(0, self.n_inputs)

                
                wt = self.shrd_mot_in_wts[mot.motif_type][mot.id] if self.weight_sharing else np.random.uniform(0.1, 1.0)
                rule = 1, src, motif_id, dst_pt, wt
            case 2:
                if len(self.conn_out_rules) >= self.max_conn_out:
                    return
                if len(self.conn_out_rules) >= self.max_conn_out:
                    self.swap_connection(conn_type=2)
                    return
                if self.one_to_one:
                    motif_id, src_pt = random.sample(list(self.mot_out_set - self.used_mot_out), 1)[0]
                    mot = self.motifs[motif_id]
                    dst = np.random.choice(list(self.output_set - self.used_outputs))
                else:
                    motif_id = np.random.randint(0, self.n_motifs)
                    mot = self.motifs[motif_id]
                    src_pt = np.random.randint(0, mot.n_outputs)
                    dst = np.random.randint(0, self.n_outputs)
                
                wt = self.shrd_mot_out_wts[mot.motif_type][mot.id] if self.weight_sharing else np.random.uniform(0.1, 1.0)
                rule = 2, motif_id, src_pt, dst, wt
        self.add_connect_rule(rule)

    def remove_connection(self):
        """Remove a connection from a genome input or to a genome output."""
        conn_type = np.random.randint(1, 3) # 1: CONNECT_IN, 2: CONNECT_OUT
        match conn_type:
            case 1:
                if not self.conn_in_rules:
                    return
                idx = random.randrange(len(self.conn_in_rules))
                rule = self.conn_in_rules.pop(idx)
            case 2:
                if not self.conn_out_rules:
                    return
                idx = random.randrange(len(self.conn_out_rules))
                rule = self.conn_out_rules.pop(idx)
        self.remove_connect_rule(rule)
        
    def swap_connection(self, conn_type=None):
        """
        Swap two random input or output connections.
        Arguments:
            conn_type (int): The type of connection to swap (1: CONNECT_IN, 2: CONNECT_OUT). If not provided, a random type will be chosen.
        """
        conn_type = conn_type if conn_type else np.random.randint(1, 3)  # 1: CONNECT_IN, 2: CONNECT_OUT

        match conn_type:
            case 1:
                if len(self.conn_in_rules) < 2:
                    return
                idx1, idx2 = random.sample(range(len(self.conn_in_rules)), 2)
                rule1, rule2 = self.conn_in_rules[idx1], self.conn_in_rules[idx2]
                (_, src1, dst1, pt1, wt1), (_, src2, dst2, pt2, wt2) = rule1, rule2
                self.conn_in_rules[idx1], self.conn_in_rules[idx2] = (1, src2, dst1, pt1, wt1), (1, src1, dst2, pt2, wt2)
                self.mot_grouped_in[dst1].remove((pt1, src1, wt1))
                self.mot_grouped_in[dst2].remove((pt2, src2, wt2))
                self.mot_grouped_in[dst1].append((pt1, src2, wt1))
                self.mot_grouped_in[dst2].append((pt2, src1, wt2))
                self.compile_flag[0] = 1
            case 2:
                if len(self.conn_out_rules) < 2:
                    return
                idx1, idx2 = random.sample(range(len(self.conn_out_rules)), 2)
                rule1, rule2 = self.conn_out_rules[idx1], self.conn_out_rules[idx2]
                (_, src1, pt1, dst1, wt1), (_, src2, pt2, dst2, wt2) = rule1, rule2
                self.conn_out_rules[idx1], self.conn_out_rules[idx2] = (2, src1, pt1, dst2, wt1), (2, src2, pt2, dst1, wt2)

                self.out_grouped_out[dst1].remove((pt1, src1, wt1))
                self.out_grouped_out[dst2].remove((pt2, src2, wt2))
                self.out_grouped_out[dst1].append((pt2, src2, wt2))
                self.out_grouped_out[dst2].append((pt1, src1, wt1))
                self.compile_flag[1] = 1

    def swap_motif(self):
        """
            Swap one motif type for another if not homogenous.
            Otherwise swap all motifs of one type for another.
        """
        if len(self.types) == 1 and len(self.motif_manager.motif_bank[self.types[0]]) == 1:
            return
        match self.homogeneous:
            case True:
                mot_type, old_id = self.motifs[0].motif_type, self.motifs[0].id
                new_type = np.random.choice([t for t in self.types])
                new_id = np.random.choice(list(self.motif_manager.motif_bank[new_type].keys()))
                old_complexity = self.motif_manager.motifs[mot_type][old_id].complexity
                complexity = self.motif_manager.motifs[new_type][new_id].complexity
                self.motifs = [Motif(new_type, new_id, self.motif_manager.motifs[new_type][new_id].n_inputs, self.motif_manager.motifs[new_type][new_id].n_outputs, complexity) for _ in range(self.n_motifs)]
                self.mot_count[mot_type][old_id] -= self.n_motifs
                self.mot_count[new_type][new_id] += self.n_motifs
                self.complexity += (complexity - old_complexity) * self.n_motifs
            case False:
                motif_idx = np.random.randint(0, self.n_motifs)
                mot = self.motifs[motif_idx]
                motif_type, old_id = mot.motif_type, mot.id
                new_type = np.random.choice([t for t in self.types])
                new_id = np.random.choice(list(self.motif_manager.motif_bank[new_type].keys()))
                old_complexity = self.motif_manager.motifs[motif_type][old_id].complexity
                complexity = self.motif_manager.motifs[new_type][new_id].complexity
                self.motifs[motif_idx] = Motif(new_type, new_id, self.motif_manager.motifs[new_type][new_id].n_inputs, self.motif_manager.motifs[new_type][new_id].n_outputs, complexity)
                self.mot_count[motif_type][old_id] -= 1
                self.mot_count[new_type][new_id] += 1
                self.complexity += complexity - old_complexity

        self.recompute_motif_distribution()

    def count_motifs(self):
        """Count the number of motifs in the genome."""
        self.mot_count = {motif_type: np.zeros(len(self.motif_manager.motif_bank[motif_type])) for motif_type in self.types}
        for mot in self.motifs:
            self.mot_count[mot.motif_type][mot.id] += 1

    def recompute_motif_distribution(self):
        """Recompute the motif distribution based on the current motif counts."""
        self.mot_dist = {motif_type: np.zeros(len(self.motif_manager.motif_bank[motif_type])) for motif_type in self.types}
        for motif_type, counts in self.mot_count.items():
            self.mot_dist[motif_type] = counts / self.n_motifs

    def modify_connection_weight(self):
        """Modify the weight of a randomly selected connection."""
        conn_type = np.random.randint(1, 3)  # 1: CONNECT_IN, 2: CONNECT_OUT

        match conn_type:
            case 1: 
                if not self.conn_in_rules:
                    return
                idx = random.randrange(len(self.conn_in_rules))
                tp, src, dst, dst_pt, wt = self.conn_in_rules[idx]
                new_wt = np.clip(wt + np.random.normal(0, 0.1), 0.0, 1.0)
                self.conn_in_rules[idx] = (tp, src, dst, dst_pt, new_wt)
                self.mot_grouped_in[dst].remove((dst_pt, src, wt))
                self.mot_grouped_in[dst].append((dst_pt, src, new_wt))
                # self.compile_flag = 1 if self.compile_flag == 0 else 3
                self.compile_flag[0] = 1

            case 2:
                if not self.conn_out_rules:
                    return
                idx = random.randrange(len(self.conn_out_rules))
                tp, src, src_pt, dst, wt = self.conn_out_rules[idx]
                new_wt = np.clip(wt + np.random.normal(0, 0.1), 0.0, 1.0)
                self.conn_out_rules[idx] = (tp, src, src_pt, dst, new_wt)
                self.out_grouped_out[dst].remove((src_pt, src, wt))
                self.out_grouped_out[dst].append((src_pt, src, new_wt))
                # self.compile_flag = 2 if self.compile_flag == 0 else 3
                self.compile_flag[1] = 1

    def clone(self, new_id):
        """Clone the genome with a new ID.
        Arguments:
            new_id (int): The ID for the cloned genome.
        Returns:
            child (Genome): A new instance of Genome with the same structure but a different ID.
        """ 
        motif_objects = (copy.deepcopy(self.mot_count), copy.deepcopy(self.mot_dist))
        child = GenomeMotif(new_id, self.motif_manager, motif_objects, self.offspring_hyperparameters)
        child.reset()
        child.mot_rules = copy.deepcopy(self.mot_rules)
        child.add_connect_in_rules(self.conn_in_rules)
        child.add_connect_out_rules(self.conn_out_rules)
        return child
    
    def reset(self):
        """Reset the genome to its initial state."""
        for mot in self.motifs:
            mot.reset()

        self.compile_flag = [1, 1]  
    
    def plot_genome(self):
        """Visualize the genome as a directed graph."""
        graph = nx.DiGraph()

        pos = {}
        node_colors = []
        y_gap = 1
        node_sz = 500

        for i in range(self.n_inputs):
            name = f'I_{i}'
            graph.add_node(name, type='input')
            pos[name] = (-2, i * y_gap)
            node_colors.append('lightblue')
        for i in range(self.n_motifs):
            name = f'M_{i}'
            graph.add_node(name, type='motif')
            pos[name] = (0, i * y_gap)
            node_colors.append('lightgreen')
        for i in range(self.n_outputs):
            name = f'O_{i}'
            graph.add_node(name, type='output')
            pos[name] = (2, i * y_gap)
            node_colors.append('lightcoral')

        edge_label_groups = defaultdict(list)
        edge_labels = {}

        for rule in self.conn_in_rules:
            _, src_id, dst_id, dst_pt, wt = rule
            
            src_nm = f"I_{src_id}"
            dst_nm = f"M_{dst_id}"

            graph.add_edge(src_nm, dst_nm, weight=wt)
            edge_label_groups[(src_nm, dst_nm)].append(f"P{dst_pt}: {wt:.2f}")

        for rule in self.conn_out_rules:
            _, src_id, src_pt, dst_id, wt = rule

            src_nm = f"M_{src_id}"
            dst_nm = f"O_{dst_id}"

            graph.add_edge(src_nm, dst_nm, weight=wt)
            edge_label_groups[(src_nm, dst_nm)].append(f"P{src_pt}: {wt:.2f}")

        edge_labels = {k: " | ".join(v) for k, v in edge_label_groups.items()}

        plt.figure(figsize=(6, 8))
        nx.draw(graph, pos, with_labels=True, node_color=node_colors, arrows=True, node_size=node_sz, font_size=10, font_color='black', edge_color='gray')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red', font_size=8, horizontalalignment='left')
        max_y = max(self.n_motifs, self.n_outputs, self.n_inputs) * y_gap
        plt.ylim(-y_gap, max_y + y_gap)
        plt.xlim(-2.5, 2.5)
        plt.axis('off')
        plt.show()
    