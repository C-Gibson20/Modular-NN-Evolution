import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import bisect
from enum import IntEnum
from collections import defaultdict
from multimodal_mazes.evolution.module_banks.motifs import Motif
from dataclasses import dataclass

class RuleType(IntEnum):
    """
    Enumeration for different rule types in the genome.
    Values:
        CONNECT_IN (int): Connections from network inputs to motifs.
        CONNECT_OUT (int): Connections from motifs to network outputs.
    """
    CONNECT_IN = 1
    CONNECT_OUT = 2

@dataclass
class Edges:
    """
    Class representing the edges in the genome.
    Properties:
        indptr (ndarray): The index pointer for the edges.
        srcs (ndarray): The source nodes for the edges.
        src_pts (ndarray): The source ports for the edges.
        dst_pts (ndarray): The destination ports for the edges.
        wts (ndarray): The weights for the edges.
        dst (ndarray): The destination for the edges.
    """
    indptr: np.ndarray
    srcs: np.ndarray
    src_pts: np.ndarray
    dst_pts: np.ndarray
    wts: np.ndarray
    dst: np.ndarray

@dataclass
class Ragged:
    """
    Class representing a ragged array.
    Properties:
        vals (ndarray): The values in the ragged array.
        ptr (ndarray): The pointer array for the ragged array.
    """
    vals: np.ndarray
    ptr: np.ndarray

    def view(self, i):
        """
        View the values for a specific index in the ragged array.
        Arguments:
            i (int): The index to view.
        Returns:
            (ndarray): The values for the specified index.
        """
        start, end = self.ptr[i], self.ptr[i + 1]
        return self.vals[start:end]

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
    def __init__(self, genome_id, hyperparameters, motif_bank):
        """
        Initialise the GenomeMotif object.
        Arguments:
            genome_id (int): The ID of the genome.
            hyperparameters (dict): The hyperparameters for the genome.
            motif_bank (MotifBank): The bank for handling motifs.
        Properties:
            fitness (float): The fitness of the genome.
            complexity (float): The complexity of the genome.
            offspring_hyperparameters (dict): The hyperparameters for the offspring genomes.
            n_inputs (int): The number of inputs to the genome.
            n_outputs (int): The number of outputs from the genome.
            n_motifs (int): The number of motifs in the genome.
            weight_sharing (bool): Whether to use weight sharing in the genome.
            uniform_weights (bool): Whether to use uniform weights in the genome.
            mutation_rates (dict): The mutation rates for the genome.
            rng (np.random.Generator): The random number generator for the genome.
            task (str): The task to be solved by the genome.
            types (list): The possible types of motifs.
            homogeneous (bool): Whether the genome is homogeneous.
            one_to_one (bool): Whether the genome connections are one-to-one.
            conn_in_rules (list): The rules for input connections.
            conn_out_rules (list): The rules for output connections.
            compile_flag (int): The flag indicating whether compilation is required.
            connectivity (str): The initial connectivity of the genome.
            connection_density (float): The initial density of connections in the genome.
        """
        self.genome_id = genome_id
        self.fitness = 0.0
        self.complexity = 0.0 

        self.motif_bank = motif_bank
        
        self.hyperparameters = hyperparameters
        self.task = hyperparameters['task']
        self.offspring_hyperparameters = dict(hyperparameters)
        self.n_inputs = hyperparameters['n_inputs']
        self.n_outputs = hyperparameters['n_outputs']
        self.n_motifs = hyperparameters['n_motifs']
        self.weight_sharing = hyperparameters['weight_sharing']
        self.uniform_weights = hyperparameters['uniform_weights']
        self.mutation_rates = hyperparameters['mutation_rates']

        self.rng = np.random.default_rng()

        self.types = hyperparameters['motif_types']
        self.homogeneous = hyperparameters['homogeneous']
        self.initialise_motifs()
        
        self.initialise_grouped_connections()
        
        if self.weight_sharing:
            self.initialise_weight_sharing()

        self.one_to_one = hyperparameters['one_to_one']
        if self.one_to_one:
            self.initialise_one_to_one()

        self.conn_in_rules = []
        self.conn_out_rules = []

        self.compile_flag = 1

        self.connectivity = hyperparameters['connectivity']
        self.connection_density = hyperparameters['connection_density']
        self.offspring_hyperparameters['connectivity'] = 'UNCONNECTED'
        self.initialise_genome()

    def initialise_motifs(self):
        """
        Initialise the motifs for the genome.
        Generates:
            self.mot_count (dict): The mapping of motif types to their counts.
            self.motifs (list): The motif instances in the genome.
        """
        bank, types, n_mots = self.motif_bank, self.types, self.n_motifs
        structs = bank.structures
        # Initialize motifs and motif count
        motifs = []
        mot_count = {t: [0] * (len(structs[t])) for t in types}

        # Initialise all modules as one type for homogenous case
        if self.homogeneous:
            mot_count[types[0]][0] = n_mots
        # If heterogeneous, distribute motifs according to initial bank
        else:
            # Distribute motifs according to initial bank
            count = 0
            for t, (id, v) in bank.init_bank.items():
                c = int(v * n_mots)
                mot_count[t][id] = c
                count += c

            # Ensure the total number of modules matches the necessary count
            if count < n_mots:
                mot_count[types[0]][0] += n_mots - count

        # Populate module instances
        for t, counts in mot_count.items():
            for id, c in enumerate(counts):
                if c == 0:
                    continue
                for _ in range(c):
                    motifs.append(Motif(structs[t][id]))
        idx = np.arange(len(motifs))
        self.rng.shuffle(idx)
        self.motifs = [motifs[i] for i in idx]
        
        self.mot_count = mot_count
        self.recompute_motif_distribution(False)

    def initialise_grouped_connections(self):
        """
        Initialise grouped connections for the genome.
        Generates:
            self.mot_grouped_in (dict): Input connections grouped by destination motif.
            self.out_grouped_out (dict): Output connections grouped by destination output.
            self._in_keys (list): The keys for each mot_grouped_in entry.
            self._out_keys (list): The keys for each out_grouped_out entry.
        """
        m, o = self.n_motifs, self.n_outputs
        # Grouped dictionaries
        self.mot_grouped_in = {i: [] for i in range(m)} # Group input connections by destination motif
        self.out_grouped_out = {i: [] for i in range(o)} # Group output connections by destination output

        # Key sets
        self._in_keys = [set() for _ in range(m)]
        self._out_keys = [set() for _ in range(o)]

    def initialise_one_to_one(self):
        """
        Initialise one-to-one connections for the genome.
        Generates:
            self.input_set (set): All input indices.
            self.used_inputs (set): All used input indices.
            self.output_set (set): All output indices.
            self.used_outputs (set): All used output indices.
            self.mot_in_set (set): All input indice-port pairs for each motif.
            self.used_mot_in (set): All used input indice-port pairs for each motif.
            self.mot_out_set (set): All output indice-port pairs for each motif.
            self.used_mot_out (set): All used output indice-port pairs for each motif.
        """
        motifs, m = self.motifs, self.n_motifs
        # Input indices
        self.input_set, self.used_inputs = set(range(self.n_inputs)), set()

        # Output indices
        self.output_set, self.used_outputs = set(range(self.n_outputs)), set()

        # Module input port indices
        self.mot_in_set, self.used_mot_in = set((i, j) for i in range(m) for j in range(motifs[i].structure.n_inputs)), set()

        # Module output port indices
        self.mot_out_set, self.used_mot_out = set((i, j) for i in range(m) for j in range(motifs[i].structure.n_outputs)), set()

    def initialise_weight_sharing(self):
        """
        Initialise weight sharing for the genome.
        Generates:
            self.shrd_mot_in_wts (dict): The shared weights for each motif's input connections.
            self.shrd_mot_out_wts (dict): The shared weights for each motif's output connections.
        """
        structs, types, rng = self.motif_bank.structures, self.types, self.rng
        # Uniform weights - all weights equal to 1.0
        if self.uniform_weights:
            self.shrd_mot_in_wts = {t: np.ones(len(structs[t])) for t in types}
            self.shrd_mot_out_wts = {t: np.ones(len(structs[t])) for t in types}
        # Shared random weights - weights per module sampled from a uniform distribution
        else:
            self.shrd_mot_in_wts = {t: rng.uniform(0.1, 1.0, len(structs[t])) for t in types}
            self.shrd_mot_out_wts = {t: rng.uniform(0.1, 1.0, len(structs[t])) for t in types}

    def add_connect_rule(self, conn_rule):
        """Add a connection rule to the genome.
        Arguments:
            conn_rule (tuple): The connection rule.
        Updates:
            self.conn_in_rules (list): The rules for input connections.
            self.conn_out_rules (list): The rules for output connections.
            self.used_inputs (set): All used input connections.
            self.used_outputs (set): All used output connections.
            self.used_mot_in (set): All used input connections for each motif.
            self.used_mot_out (set): All used output connections for each motif.
            self.mot_grouped_in (dict): Input connections grouped by destination motif.
            self.out_grouped_out (dict): Output connections grouped by destination output.
            self._in_keys (list): The keys for each mot_grouped_in entry.
            self._out_keys (list): The keys for each out_grouped_out entry.
            self.complexity (int): The complexity of the genome.
            self.compile_flag (int): The flag indicating whether compilation is required.
        """
        self.compile_flag = 1
                
        match conn_rule[0]:
            case 1:  
                _, src, dst, dst_pt, wt = conn_rule
                key, item = (dst_pt, src), (dst_pt, src, wt)
                bucket, existing_keys = self.mot_grouped_in[dst], self._in_keys[dst]

                # Check connection does not already exist
                if key in existing_keys:
                    return

                # Insert new connection to mod_grouped_in and conn_in_rules
                if not bucket:
                    bucket.append(item)
                else:
                    bisect.insort(bucket, item)
                self.conn_in_rules.append(conn_rule)
                self.complexity += 1

                # Update keys
                existing_keys.add(key)

                # Update used inputs and module inputs
                if self.one_to_one:
                    self.used_mot_in.add((dst, dst_pt))
                    self.used_inputs.add(src)
            case 2:  
                _, src, src_pt, dst, wt = conn_rule
                key, item = (src_pt, src), (src_pt, src, wt)
                bucket, existing_keys = self.out_grouped_out[dst], self._out_keys[dst]

                # Check connection does not already exist
                if key in existing_keys:
                    return

                # Insert new connection to out_grouped_out and conn_out_rules
                if not bucket:
                    bucket.append(item)
                else:
                    bisect.insort(bucket, item)
                self.conn_out_rules.append(conn_rule)
                self.complexity += 1

                # Update keys
                existing_keys.add(key)

                # Update used outputs and module outputs
                if self.one_to_one:
                    self.used_mot_out.add((src, src_pt))
                    self.used_outputs.add(dst)

    def remove_connect_rule(self, conn_rule):
        """Remove a connection rule from the genome.
        Arguments:
            conn_rule (tuple): The connection rule to remove.
        Updates:
            self.conn_in_rules (list): The rules for input connections.
            self.conn_out_rules (list): The rules for output connections.
            self.used_inputs (set): All used input connections.
            self.used_outputs (set): All used output connections.
            self.used_mot_in (set): All used input connections for each motif.
            self.used_mot_out (set): All used output connections for each motif.
            _in_keys (list): The keys for each mot_grouped_in entry.
            _out_keys (list): The keys for each out_grouped_out entry.
            complexity (int): The complexity of the genome.
            compile_flag (int): The flag indicating whether compilation is required.
        """
        self.compile_flag = 1
        self.complexity -= 1
        
        match conn_rule[0]:
            case 1:
                _, src, dst, dst_pt, wt = conn_rule
                # Remove connection from mod_grouped_in
                self.mot_grouped_in[dst].remove((dst_pt, src, wt))
                self._in_keys[dst].remove((dst_pt, src))
                
                # Update used inputs and module inputs
                if self.one_to_one:
                    self.used_mot_in.remove((dst, dst_pt))
                    self.used_inputs.remove(src)
            case 2:
                _, src, src_pt, dst, wt = conn_rule
                # Remove connection from out_grouped_out
                self.out_grouped_out[dst].remove((src_pt, src, wt))
                self._out_keys[dst].remove((src_pt, src))

                # Update used outputs and module outputs
                if self.one_to_one:
                    self.used_mot_out.remove((src, src_pt))
                    self.used_outputs.remove(dst)

    def initialise_connect_in_rules(self, connectivity, n_rules):
        """Add multiple input connection rules to the genome at initialisation.
        Arguments:
            connectivity (int): Connectivity type for the rules (0: SPARSE, 1: RANDOM).
            n_rules (int): Number of rules to generate.
        """
        mots, n_mots, n_in, rng = self.motifs, self.n_motifs, self.n_inputs, self.rng
        match connectivity:
            case 0: 
                mot_ids = list(range(n_mots))

                # Reduce number of motifs if one-to-one mapping and n_motifs > n_inputs
                if self.one_to_one and n_rules > n_in:
                    n_rules = n_in
                    mot_ids = rng.choice(mot_ids, size=n_rules, replace=False).tolist()

                # Randomly sample one motif-input pair for each selected motif
                pts = [rng.integers(mots[i].structure.n_inputs) for i in mot_ids]

            case 1: 
                if self.one_to_one:
                    mot_set = list(self.mot_in_set)
                    # Size is the minimum of the given number of rules, available motif ports, and available inputs
                    n_rules = min(n_rules, len(mot_set), len(self.input_set))

                    # Randomly sample `n_rules` unique motif-input pairs
                    idxs = rng.choice(len(mot_set), size=n_rules, replace=False)
                    mot_ids = [mot_set[i][0] for i in idxs]
                    pts = [mot_set[i][1] for i in idxs]
                else:
                    # Sample motif-port pairs with replacement
                    mot_ids = rng.choice(n_mots, size=n_rules, replace=True).tolist()
                    pts = [rng.integers(mots[i].structure.n_inputs) for i in mot_ids]

        # Randomly sample source inputs
        if self.one_to_one:
            srcs = rng.choice(n_in, size=n_rules, replace=False).tolist()
        else:
            src_pool = np.tile(np.arange(n_in), int(np.ceil(n_rules / n_in)))
            rng.shuffle(src_pool)
            srcs = src_pool[:n_rules]

        # Randomly sample weights from uniform distribution or shared weights
        if self.weight_sharing:
            shrd_wts = self.shrd_mot_in_wts
            mot_type_keys = np.fromiter(([mots[i].structure.type, mots[i].structure.id] for i in mot_ids), dtype=object)
            wts = np.array([shrd_wts[mot_type][mot_id] for mot_type, mot_id in mot_type_keys])
        else:
            wts = rng.uniform(0.1, 1.0, size=n_rules)

        # Create connection rules
        for src, dst, dst_pt, wt in zip(srcs, mot_ids, pts, wts):
            self.add_connect_rule((1, src, dst, dst_pt, wt))

    def initialise_connect_out_rules(self, connectivity, n_rules):
        """Add multiple outputs connection rules to the genome at initialisation.
        Arguments:
            connectivity (int): Connectivity type for the rules (0: SPARSE, 1: RANDOM).
            n_rules (int): Number of rules to generate.
        """
        mots, n_mots, n_out, rng = self.motifs, self.n_motifs, self.n_outputs, self.rng
        match connectivity:
            case 0:  
                mot_ids = list(range(n_mots))

                # Reduce number of motifs if one-to-one mapping and n_motifs > n_outputs
                if self.one_to_one and n_rules > n_out:
                    n_rules = n_out
                    mot_ids = rng.choice(mot_ids, size=n_rules, replace=False).tolist()

                pts = [rng.choice(mots[i].structure.n_outputs) for i in mot_ids]
            case 1:  
                if self.one_to_one:
                    mot_set = list(self.mot_out_set)
                    # Size is the minimum of the given n_rules, available motif ports, and available outputs
                    n_rules = min(n_rules, len(mot_set), len(self.output_set))

                    # Randomly sample `n_rules` unique motif-output pairs
                    idxs = rng.choice(len(mot_set), size=n_rules, replace=False).tolist()
                    mot_ids = [mot_set[i][0] for i in idxs]
                    pts = [mot_set[i][1] for i in idxs]
                else:
                    # Sample motif-output port pairs with replacement
                    mot_ids = rng.choice(range(n_mots), size=n_rules, replace=True)
                    pts = [rng.choice(mots[i].n_outputs) for i in mot_ids]

        # Randomly sample destination outputs
        if self.one_to_one:
            dsts = rng.choice(n_out, size=n_rules, replace=False).tolist()
        else:
            dst_pool = np.tile(np.arange(n_out), int(np.ceil(n_rules / n_out)))
            rng.shuffle(dst_pool)
            dsts = dst_pool[:n_rules]

        # Randomly sample weights from uniform distribution or shared weights
        if self.weight_sharing:
            shrd_wts = self.shrd_mot_out_wts    
            mot_type_keys = np.fromiter([(mots[i].structure.type, mots[i].structure.id) for i in mot_ids], dtype=object)
            wts = np.array([shrd_wts[mot_type][mot_id] for mot_type, mot_id in mot_type_keys])
        else:
            wts = rng.uniform(0.1, 1.0, size=n_rules)

        # Create connection rules
        for src, src_pt, dst, wt in zip(mot_ids, pts, dsts, wts):
            self.add_connect_rule((2, src, src_pt, dst, wt))

    def compile_rules(self):
        """
        Compile connection rules into a ragged arrays.
        Generates:
            self.in_edges (Edges): The edges for input connections.
            self.out_edges (Edges): The edges for output connections.
        """
        
        # ------------------------------------------- #
        # ------------ Input Connections ------------ #
        # ------------------------------------------- #

        # Initialise arrays grouped by motif
        mot_g_in, n_mots = self.mot_grouped_in, self.n_motifs
        in_lens = np.fromiter((len(mot_g_in[i]) for i in range(n_mots)), count=n_mots, dtype=np.int32)
        in_indptr = np.concatenate(([0], np.cumsum(in_lens, dtype=np.int32)))
        n = in_indptr[-1]
        in_srcs, in_src_pts, in_dst_pts, in_wts = np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.float32)

        # Fill arrays
        for m in range(n_mots):
            start, end = in_indptr[m], in_indptr[m + 1]
            if start == end:
                continue
            entries = mot_g_in[m]
            in_srcs[start:end] = [t[1] for t in entries]
            in_dst_pts[start:end] = [t[0] for t in entries]
            in_wts[start:end] = [t[2] for t in entries]

        # Create edges from arrays
        deg = in_indptr[1:] - in_indptr[:-1]
        dst = np.repeat(np.arange(n_mots), deg)
        self.in_edges = Edges(in_indptr, in_srcs, in_src_pts, in_dst_pts, in_wts, dst)

        # -------------------------------------------- #
        # ------------ Output Connections ------------ #
        # -------------------------------------------- #

        # Initialise arrays grouped by motif
        out_g_out, n_out = self.out_grouped_out, self.n_outputs
        out_lens = np.fromiter((len(out_g_out[i]) for i in range(n_out)), count=n_out, dtype=np.int32)
        out_indptr = np.concatenate(([0], np.cumsum(out_lens, dtype=np.int32)))
        n = out_indptr[-1]
        out_srcs, out_src_pts, out_dst_pts, out_wts = np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.float32)

        # Fill arrays
        for m in range(n_out):
            start, end = out_indptr[m], out_indptr[m + 1]
            if start == end:
                continue
            entries = out_g_out[m]
            out_srcs[start:end] = [t[1] for t in entries]
            out_src_pts[start:end] = [t[0] for t in entries]
            out_wts[start:end] = [t[2] for t in entries]

        # Create edges from arrays 
        deg = out_indptr[1:] - out_indptr[:-1]
        dst = np.repeat(np.arange(self.n_outputs), deg)
        self.out_edges = Edges(out_indptr, out_srcs, out_src_pts, out_dst_pts, out_wts, dst)

    def initialise_genome(self):
        """Initialise the genome."""
        match self.connectivity:
            case 'UNCONNECTED': return
            case 'SPARSE':
                n = self.n_motifs
                # Sparse connectivity: Connect each motif with one input and one output
                self.initialise_connect_in_rules(0, n)
                self.initialise_connect_out_rules(0, n)

            case 'RANDOM':
                # Random connectivity: Connect a random subset of inputs and outputs to random motifs
                n_conn_in = int(self.connection_density['input_density'] * self.n_inputs)
                n_conn_out = int(self.connection_density['output_density'] * self.n_outputs)
                if n_conn_in > 0:
                    self.initialise_connect_in_rules(1, n_conn_in)
                if n_conn_out > 0:
                    self.initialise_connect_out_rules(1, n_conn_out)

            case 'FULLY CONNECTED':
                # Fully connected: Connect all inputs and outputs to all motifs
                rng, n_in, n_out = self.rng, self.n_inputs, self.n_outputs
                shr_wts, shrd_in_wts, shrd_out_wts = self.weight_sharing, self.shrd_mot_in_wts, self.shrd_mot_out_wts
                for i, mot in enumerate(self.motifs):
                    mot_s = mot.structure
                    # Sample weights from uniform distribution or shared weights
                    if shr_wts:
                        in_wt, out_wt = shrd_in_wts[mot_s.type][mot_s.id], shrd_out_wts[mot_s.type][mot_s.id]
                    else:
                        in_wt, out_wt = rng.uniform(0.1, 1.0, 2)

                    # Connect all network inputs
                    for pt in range(mot_s.n_inputs):
                        for src in range(n_in):
                            self.add_connect_rule((1, src, i, pt, in_wt))

                    # Connect all network outputs
                    for pt in range(mot_s.n_outputs):
                        for dst in range(n_out):
                            self.add_connect_rule((2, i, pt, dst, out_wt))

            case 'IDEAL':
                # Ideal: Connect each module and node ideally - for maze task prototyping only
                assert self.task == 'maze'
                for i in range(self.n_inputs):
                    mot = i // 2
                    self.add_connect_rule((1, i, mot, 0, 1.0))
                    self.add_connect_rule((1, i, mot, 1, 1.0))
                
                for i in range(self.n_outputs):
                    self.add_connect_rule((2, i, 0, i, 1.0))

    def forward_pass(self, input_vector):
        """Forward pass through the genome using execution plan.
        Arguments:
            input_vector (list): Input vector of size equal to the number of inputs.
        Returns:
            outputs.vals (array): Output array of size equal to the number of outputs.
        """
        # Compile if the edges are outdated
        if self.compile_flag:
            self.compile_rules()
            self.compile_flag = 0

        # Forward pass through the motifs and outputs
        mot_outputs = self.layer_forward_pass(0, input_vector, self.n_motifs)
        outputs = self.layer_forward_pass(1, mot_outputs, self.n_outputs)
        
        return outputs.vals
    
    def layer_forward_pass(self, layer, in_vec, lyr_size):
        """Forward pass through a specific layer of the genome.
        Arguments:
            layer (int): The layer type (0 for 'MODULE' or 1 for 'OUTPUT').
            in_vec (np.ndarray): The input vector for the layer.
            lyr_size (int): The size of the layer.
        Returns:
            (Ragged): The output data structure after processing the layer.
        """
        # Prepare edges and sizes based on layer type
        match layer:
            case 0: 
                mots = self.motifs
                edges = self.in_edges
                inter_sizes = np.fromiter((m.structure.n_inputs for m in mots), dtype=np.int32)
                out_sizes = np.fromiter((m.structure.n_outputs for m in mots), dtype=np.int32)

            case 1: 
                edges = self.out_edges
                inter_sizes = np.ones(lyr_size, dtype=np.int32)

        # Build intermediate pointers and values
        inter_ptr = self.build_ptr_from_size(inter_sizes)
        inter_vals = np.zeros(inter_ptr[-1], dtype=np.float32)

        # Compute contributions from input vector
        if layer == 0:
            input_vec = in_vec[edges.srcs]
        else:
            src_idx = in_vec.ptr[edges.srcs] + edges.src_pts
            input_vec = in_vec.vals[src_idx]
        dst_idx = inter_ptr[edges.dst] + edges.dst_pts
        np.add.at(inter_vals, dst_idx, input_vec * edges.wts)

        # Early return for output layer
        if layer == 1:
            return Ragged(inter_vals, inter_ptr)
        
        # Prepare output pointers and values
        out_ptr = self.build_ptr_from_size(out_sizes)
        out_vals = np.zeros(out_ptr[-1], dtype=np.float32)

        # Compute contributions from motifs
        for i in range(lyr_size):
            # Check the motif has edges
            start, end = inter_ptr[i], inter_ptr[i + 1]
            if start == end:
                continue

            # Fill in motif forward pass outputs
            mot = self.motifs[i]
            out_start, out_end = out_ptr[i], out_ptr[i + 1]
            out_vals[out_start:out_end] = mot.forward_pass(inter_vals[start:end])

        return Ragged(out_vals, out_ptr)

    # Todo
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
        child = GenomeMotif(new_id, self.offspring_hyperparameters, self.motif_bank)
        child.motifs = motifs
        child.count_motifs()
        child.recompute_motif_distribution()
        child.reset()

        self.cut_rules(child, cut, self.conn_in_rules, parent_2.conn_in_rules, 1)
        self.cut_rules(child, cut, self.conn_out_rules, parent_2.conn_out_rules, 2)

        return child

    # Todo
    def cut_rules(self, child, cut, rule_list1, rule_list2, rule_type):
        """Cut rules at a specified index.
        Arguments:
            cut (int): The index to cut the rules at.
            rule_list1 (list): The list of rules to cut from parent 1.
            rule_list2 (list): The list of rules to cut from parent 2.
            rule_type (RuleType): The type of rules being cut.
        """
        match rule_type:
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
                    for r in rules:
                        child.add_connect_rule(r)
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
                    for r in rules:
                        child.add_connect_out_rule(r)

    def mutate(self):
        """Mutate the genome by randomly motifying its structure."""
        # If there is no path through the network then adding connections is the only useful mutation
        if not self.conn_in_rules or not self.conn_out_rules:
            mut = 0
        # Choose mutations with given probability distribution
        else:
            mut = np.random.choice([0, 1, 2, 3], p=self.mutation_rates)
            
        match mut:
            case 0: self.add_connection()
            case 1: self.remove_connection()
            case 2: self.swap_connection()
            case 3: self.swap_motif()
            case 4: self.modify_connection_weight()

    def add_connection(self):
        """Add a new connection from a genome input or to a genome output."""
        rng = self.rng
        conn_type = rng.integers(1, 3)  # 1: CONNECT_IN, 2: CONNECT_OUT

        match conn_type:
            case 1:
                if self.one_to_one:
                    # Get set of available motif-port pairs and sources
                    mot_set = list(self.mot_in_set - self.used_mot_in)
                    src_set = list(self.input_set - self.used_inputs)

                    # If there are no available motif-port pairs or sources, swap connections
                    if not mot_set or not src_set:
                        self.swap_connection(conn_type=1)
                        return
                    
                    # Get a random motif-port pair and input
                    mot_id, dst_pt = mot_set[rng.integers(len(mot_set))]
                    mot = self.motifs[mot_id]
                    src = int(rng.choice(src_set))

                else:
                    # Get the set of available motifs
                    ids = [i for i in range(self.n_motifs)]

                    # If there are no available motifs, swap connections
                    if not ids:
                        self.swap_connection(conn_type=1)
                        return

                    # Get a random motif-port pair and input
                    mot_id = rng.choice(ids)
                    mot = self.motifs[mot_id]
                    dst_pt = rng.integers(mot.structure.n_inputs)
                    src = rng.integers(self.n_inputs)


                wt = self.shrd_mot_in_wts[mot.structure.type][mot.structure.id] if self.weight_sharing else np.random.uniform(0.1, 1.0)
                rule = 1, src, mot_id, dst_pt, wt
            case 2:
                if self.one_to_one:
                    # Get the set of available motif-port pairs and destinations
                    mot_set = list(self.mot_out_set - self.used_mot_out)
                    dst_set = list(self.output_set - self.used_outputs)

                    # If there are no available motif-port pairs or destinations, swap connections
                    if not mot_set or not dst_set:
                        self.swap_connection(conn_type=2)
                        return

                    # Get a random motif-port pair and destination
                    mot_id, src_pt = mot_set[rng.integers(len(mot_set))]
                    mot = self.motifs[mot_id]
                    dst = rng.choice(dst_set)
                else:
                    # Get the set of available motifs
                    ids = [i for i in range(self.n_motifs)]

                    # If there are no available motifs, swap connections
                    if not ids:
                        self.swap_connection(conn_type=2)
                        return
                    
                    # Get a random motif-port pair and output
                    mot_id = rng.choice(ids)
                    mot = self.motifs[mot_id]
                    src_pt = rng.integers(mot.n_outputs)
                    dst = rng.integers(self.n_outputs)

                mot_s = mot.structure
                wt = self.shrd_mot_out_wts[mot_s.type][mot_s.id] if self.weight_sharing else rng.uniform(0.1, 1.0)
                rule = 2, mot_id, src_pt, dst, wt
        self.add_connect_rule(rule)

    def remove_connection(self):
        """Remove a connection from a genome input or to a genome output."""
        rng = self.rng
        conn_type = rng.integers(1, 3)  # 1: CONNECT_IN, 2: CONNECT_OUT
        match conn_type:
            # Input connections
            case 1:
                conn_in = self.conn_in_rules
                if not conn_in:
                    return
                idx = rng.integers(len(conn_in))
                rule = conn_in.pop(idx)
            # Output connections
            case 2:
                conn_out = self.conn_out_rules
                if not conn_out:
                    return
                idx = rng.integers(len(conn_out))
                rule = conn_out.pop(idx)
        self.remove_connect_rule(rule)
        
    def swap_connection(self, conn_type=None):
        """
        Swap two random input or output connections.
        Arguments:
            conn_type (int): The type of connection to swap (1: CONNECT_IN, 2: CONNECT_OUT).
        Updates:
            self.conn_in_rules (list): The rules for input connections.
            self.conn_out_rules (list): The rules for output connections.
            self.mot_grouped_in (dict): Input connections grouped by destination motif.
            self.out_grouped_out (dict): Output connections grouped by their destination output.
            self._in_keys (list): The keys for each mot_grouped_in entry.
            self._out_keys (list): The keys for each out_grouped_out entry.
            self.compile_flag (int): The flag indicating whether compilation is required.
        """
        rng = self.rng
        conn_type = conn_type if conn_type else rng.integers(1, 3)  # 1: CONNECT_IN, 2: CONNECT_OUT
        self.compile_flag = 1

        match conn_type:
            case 1:
                in_rules = self.conn_in_rules
                n = len(in_rules)
                if n < 2:
                    return
                # Sample two random input connection rules
                idx1, idx2 = rng.choice(n, size=2, replace=False).tolist()
                rule1, rule2 = in_rules[idx1], in_rules[idx2]
                (_, src1, dst1, pt1, wt1), (_, src2, dst2, pt2, wt2) = rule1, rule2

                # Swap the connection sources
                in_rules[idx1], in_rules[idx2] = (1, src2, dst1, pt1, wt1), (1, src1, dst2, pt2, wt2)

                # Update grouped dictionaries and key sets
                mot_g_in, in_keys = self.mot_grouped_in, self._in_keys
                mot_g_in[dst1].remove((pt1, src1, wt1))
                in_keys[dst1].discard((pt1, src1))
                mot_g_in[dst2].remove((pt2, src2, wt2))
                in_keys[dst2].discard((pt2, src2))
                mot_g_in[dst1].append((pt1, src2, wt1))
                in_keys[dst1].add((pt1, src2))
                mot_g_in[dst2].append((pt2, src1, wt2))
                in_keys[dst2].add((pt2, src1))
            case 2:
                out_rules = self.conn_out_rules
                n = len(out_rules)
                if n < 2:
                    return
                # Sample two random output connection rules
                idx1, idx2 = rng.choice(n, size=2, replace=False).tolist()
                rule1, rule2 = out_rules[idx1], out_rules[idx2]
                (_, src1, pt1, dst1, wt1), (_, src2, pt2, dst2, wt2) = rule1, rule2
                
                # Swap the connection destinations
                out_rules[idx1], out_rules[idx2] = (2, src1, pt1, dst2, wt1), (2, src2, pt2, dst1, wt2)

                # Update grouped dictionaries and key sets
                mot_g_out, out_keys = self.out_grouped_out, self._out_keys
                mot_g_out[dst1].remove((pt1, src1, wt1))
                out_keys[dst1].discard((pt1, src1))
                mot_g_out[dst2].remove((pt2, src2, wt2))
                out_keys[dst2].discard((pt2, src2))
                mot_g_out[dst1].append((pt2, src2, wt2))
                out_keys[dst1].add((pt2, src2))
                mot_g_out[dst2].append((pt1, src1, wt1))
                out_keys[dst2].add((pt1, src1))

    def swap_motif(self):
        """
        Swap one motif type for another if not homogeneous.
        Otherwise swap all motifs of one type for another.
        Updates:
            self.motifs (list): The motif instances in the genome.
            self.mot_count (dict): The mapping of motif types to their counts.
            self.complexity (int): The complexity of the genome.
            self.compile_flag (int): The flag indicating whether compilation is required.
        """
        structs, types, rng = self.motif_bank.structures, self.types, self.rng
        n_mots = self.n_motifs
        # Insufficient motif types to swap
        if len(types) == 1 and len(structs[types[0]]) == 1:
            return
        
        self.compile_flag = 1
        
        # Swap all motifs of one type for another
        if self.homogeneous:
            # Old motif information
            mot_s = self.motifs[0].structure
            old_t, old_id, old_c = mot_s.type, mot_s.id, mot_s.complexity
            
            # New motif information
            new_t = rng.choice([t for t in types])
            # Ensure new motif is different from old
            if new_t == old_t:
                if len(structs[new_t]) > 1:
                    new_id = rng.choice([i for i in structs[new_t].keys() if i != old_id])
                else:
                    new_t = rng.choice([t for t in types if t != old_t])
                    new_id = rng.choice(list(structs[new_t].keys()))
            else:
                new_id = rng.choice(list(structs[new_t].keys()))

            # Repopulate motif instances
            self.motifs = [Motif(structs[new_t][new_id]) for _ in range(n_mots)]
            
            # Update complexity
            new_c = structs[new_t][new_id].complexity
            self.complexity += (new_c - old_c) * n_mots

            # Update distributions and validate swap
            self.recompute_motif_distribution()
            self.rebuild_motif_sets()
            for i in range(n_mots):
                self.validate_swap(i)
        else:
            # Randomly sample a motif and get its information
            idx = rng.integers(n_mots)
            mot_s = self.motifs[idx].structure
            old_t, old_id, old_c = mot_s.type, mot_s.id, mot_s.complexity

            # Randomly choose new motif
            new_t = rng.choice([t for t in types])
            # Ensure new motif is different from old
            if new_t == old_t:
                if len(structs[new_t]) > 1:
                    new_t = rng.choice([t for t in types if t != old_t])
                    new_id = rng.choice(list(structs[new_t].keys()))
                else:
                    new_id = rng.choice([i for i in structs[new_t].keys() if i != old_id])
            else:
                new_id = rng.choice(list(structs[new_t].keys()))

            # Update motif instance
            self.motifs[idx] = Motif(structs[new_t][new_id])

            # Update complexity
            new_c = structs[new_t][new_id].complexity
            self.complexity += (new_c - old_c)

            # Update distributions and validate swap
            self.recompute_motif_distribution()
            self.rebuild_motif_sets()
            self.validate_swap(idx)

    def validate_swap(self, mod_idx):
        """
        Validate the swap of a motif and update connection rules accordingly.
        Arguments:
            mod_idx (int): The index of the modified motif.
        """
        # New motif properties
        new_mot_s = self.motifs[mod_idx].structure
        new_in, new_out = new_mot_s.n_inputs, new_mot_s.n_outputs

        # Prune out of range input connections
        rules_to_remove = [r for r in self.conn_in_rules if r[2] == mod_idx and r[3] >= new_in]
        for r in rules_to_remove:
            self.remove_connect_rule(r)

        # Prune out of range output connections
        rules_to_remove = [r for r in self.conn_out_rules if r[1] == mod_idx and r[2] >= new_out]
        for r in rules_to_remove:
            self.remove_connect_rule(r)

    def rebuild_motif_sets(self):
        """
        Rebuild the sets of input and output connections for each motif.
        Generates:
            self.mot_in_set (set): All input connections for each motif.
            self.mot_out_set (set): All output connections for each motif.
        """
        n_mots, mots = self.n_motifs, self.motifs
        self.mot_in_set = set((i, j) for i in range(n_mots) for j in range(mots[i].structure.n_inputs))
        self.mot_out_set = set((i, j) for i in range(n_mots) for j in range(mots[i].structure.n_outputs))

    def count_motifs(self):
        """
        Count the number of motifs in the genome.
        Updates:
            self.mot_count (dict): The mapping of motif types to their counts.
        """
        structs, types = self.motif_bank.structures, self.types
        # Precompute length of each type
        lengths = {t: len(structs[t]) for t in types}

        # Gather motif ids per type
        buckets = defaultdict(list)
        for mot in self.motifs:
            s = mot.structure
            buckets[s.type].append(s.id)

        # Count occurrences per type
        mot_count = {}
        for t in types:
            mot_count[t] = np.bincount(buckets[t], minlength=lengths[t]).astype(np.int32)
        self.mot_count = mot_count

    def recompute_motif_distribution(self, count = True):
        """
        Recompute the motif distribution based on the current motif counts.
        Arguments:
            count (bool): Whether to count motifs before recomputing the distribution. 
        Updates:
            self.mot_dist (dict): The mapping of motif types to their distributions.
        """
        if count:
            self.count_motifs()
        # Compute ratio from count
        mot_count, n_mots, types = self.mot_count, self.n_motifs, self.types
        self.mot_dist = {t: np.asarray(mot_count[t], dtype=np.int32) / n_mots for t in types}

    def modify_connection_weight(self):
        """
        Modify the weight of a randomly selected connection.
        Updates:
            self.conn_in_rules (list): The rules for input connections.
            self.conn_out_rules (list): The rules for output connections.
            self.mot_grouped_in (dict): Input connections grouped by destination motif.
            self.out_grouped_out (dict): Output connections grouped by destination output.
            self.compile_flag (int): The flag indicating whether compilation is required.
        """
        if self.weight_sharing:
            return
        rng = self.rng
        conn_type = rng.integers(1, 3)  # 1: CONNECT_IN, 2: CONNECT_OUT
        self.compile_flag = 1

        match conn_type:
            case 1: 
                conn_in = self.conn_in_rules
                if not conn_in:
                    return
                # Randomly select a connection rule
                idx = rng.integers(len(conn_in))
                tp, src, dst, dst_pt, wt = conn_in[idx]

                # Randomly generate new weight
                new_wt = np.clip(wt + rng.normal(0, 0.1), 0.0, 1.0)

                # Update connection rule with the new weight
                self.conn_in_rules[idx] = (tp, src, dst, dst_pt, new_wt)
                self.mot_grouped_in[dst].remove((dst_pt, src, wt))
                self.mot_grouped_in[dst].append((dst_pt, src, new_wt))
                
            case 2:
                conn_out = self.conn_out_rules
                if not conn_out:
                    return
                # Randomly select a connection rule
                idx = rng.integers(len(conn_out))
                tp, src, src_pt, dst, wt = conn_out[idx]

                # Randomly generate new weight
                new_wt = np.clip(wt + rng.normal(0, 0.1), 0.0, 1.0)

                # Update connection rule with the new weight
                self.conn_out_rules[idx] = (tp, src, src_pt, dst, new_wt)
                self.out_grouped_out[dst].remove((src_pt, src, wt))
                self.out_grouped_out[dst].append((src_pt, src, new_wt))

    def clone(self, new_id):
        """Clone the genome with a new ID.
        Arguments:
            new_id (int): The ID for the cloned genome.
        Returns:
            child (Genome): A clone of the genome with the new ID.
        """ 
        # New genome instance
        child = GenomeMotif(new_id, self.offspring_hyperparameters, self.motif_bank)
        
        # Clone motif information
        child.motifs = copy.deepcopy(self.motifs)
        child.mot_count = copy.deepcopy(self.mot_count)
        child.mot_dist = copy.deepcopy(self.mot_dist)

        # Clone module and connection rules
        child.conn_in_rules = copy.deepcopy(self.conn_in_rules)
        child.conn_out_rules = copy.deepcopy(self.conn_out_rules)
        child.mot_grouped_in = copy.deepcopy(self.mot_grouped_in)
        child.out_grouped_out = copy.deepcopy(self.out_grouped_out)
        child._in_keys = copy.deepcopy(self._in_keys)
        child._out_keys = copy.deepcopy(self._out_keys)
        child.used_inputs = copy.deepcopy(self.used_inputs)
        child.used_outputs = copy.deepcopy(self.used_outputs)
        child.used_mot_in = copy.deepcopy(self.used_mot_in)
        child.used_mot_out = copy.deepcopy(self.used_mot_out)

        # Reset and return new genome
        child.reset()
        return child
    
    def reset(self):
        """
        Reset the genome to its initial state.
        Resets:
            (Motif): Each motif instance in self.motifs
            self.compile_flag (int): The flag indicating whether compilation is required.
        """
        # Reset motifs
        for mot in self.motifs:
            mot.reset()

        # Reset compile flag
        self.compile_flag = 1

    def build_ptr_from_size(self, arr_sizes):
        """
        Build a pointer array from the given sizes.
        Arguments:
            arr_sizes (np.ndarray): An array of sizes.
        Returns:
            ptr (np.ndarray): A pointer array indicating the start of each segment.
        """
        # Initialize pointer array
        ptr = np.zeros(arr_sizes.size + 1, dtype=np.int32)

        # Build and return pointer array
        np.cumsum(arr_sizes, out=ptr[1:])
        return ptr
    
    def plot_genome(self):
        """Visualize the genome as a directed graph."""
        graph = nx.DiGraph()

        pos, node_colors = {}, []

        # Input nodes
        for i in range(self.n_inputs):
            name = f'I_{i}'
            graph.add_node(name, type='input')
            pos[name] = (-2, i)
            node_colors.append('lightblue')

        # Motif nodes
        for i in range(self.n_motifs):
            name = f'M_{i}'
            graph.add_node(name, type='motif')
            pos[name] = (0, i)
            node_colors.append('lightgreen')

        # Output nodes
        for i in range(self.n_outputs):
            name = f'O_{i}'
            graph.add_node(name, type='output')
            pos[name] = (2, i)
            node_colors.append('lightcoral')

        edge_label_groups = defaultdict(list)

        # Input connections
        for rule in self.conn_in_rules:
            _, src_id, dst_id, dst_pt, wt = rule
            src_nm = f"I_{src_id}"
            dst_nm = f"M_{dst_id}"
            graph.add_edge(src_nm, dst_nm, weight=wt)
            edge_label_groups[(src_nm, dst_nm)].append(f"P{dst_pt}: {wt:.2f}")

        # Output connections
        for rule in self.conn_out_rules:
            _, src_id, src_pt, dst_id, wt = rule
            src_nm = f"M_{src_id}"
            dst_nm = f"O_{dst_id}"
            graph.add_edge(src_nm, dst_nm, weight=wt)
            edge_label_groups[(src_nm, dst_nm)].append(f"P{src_pt}: {wt:.2f}")

        # Combine edge labels for multiple connections
        edge_labels = {k: " | ".join(v) for k, v in edge_label_groups.items()}

        # Plot the graph
        plt.figure(figsize=(6, 8))
        nx.draw(graph, pos, with_labels=True, node_color=node_colors, arrows=True, node_size=2000, font_size=10, font_color='black', edge_color='gray')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red', font_size=8, horizontalalignment='left')
        max_y = max(self.n_motifs, self.n_outputs, self.n_inputs)
        plt.ylim(-1, max_y + 1)
        plt.xlim(-2.5, 2.5)
        plt.axis('off')
        plt.show()
    