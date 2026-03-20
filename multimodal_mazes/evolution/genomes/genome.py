import random
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import bisect
from enum import IntEnum
from multimodal_mazes.evolution.module_banks.maze_bank import Recurrent, Feedforward
from multimodal_mazes.evolution.module_banks.image_bank import SquareClassificationModule, CrossClassificationModule 
from dataclasses import dataclass
from collections import defaultdict

class RuleType(IntEnum):
    """
    Enumeration for different rule types in the genome.
    Values:
        CONNECT_IN: Represents an input connection rule.
        CONNECT_OUT: Represents an output connection rule.
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
    # ('rule_type', np.int32), # 0: module, 1: connect_in, 2: connect_out
    # ('src', np.int32),       # source node ID: network input
    # ('dst', np.int32),       # destination node ID: module_id
    # ('dst_port', np.int32),  # module input port
    # ('weight', np.float32)   # weight of the connection
# )

# Output connection tuples
# (
#     ('rule_type', np.int32), # 0: module, 1: connect_in, 2: connect_out
#     ('src', np.int32),       # source node ID: module_id
#     ('src_port', np.int32),  # module output port
#     ('dst', np.int32),       # destination node ID: network output
#     ('weight', np.float32)   # weight of the connection
# )

class Genome():
    def __init__(self, genome_id, hyperparameters):
        """
        Initialise the Genome object.
        Arguments:
            genome_id (int): The ID of the genome.
            hyperparameters (dict): The hyperparameters for the genome.
        Properties:
            fitness (float): The fitness of the genome.
            offspring_hyperparameters (dict): The hyperparameters for the offspring genomes.
            n_inputs (int): The number of inputs to the genome.
            n_outputs (int): The number of outputs from the genome.
            n_modules (int): The number of modules in the genome.
            weight_sharing (bool): Whether to use weight sharing in the genome.
            uniform_weights (bool): Whether to use uniform weights in the genome.
            mutation_rates (dict): The mutation rates for the genome.
            rng (np.random.Generator): The random number generator for the genome.
            task (str): The task to be solved by the genome.
            one_to_one (bool): Whether the genome connections are one-to-one.
            conn_in_rules (list): The rules for input connections.
            conn_out_rules (list): The rules for output connections.
            compile_flag (int): The flag indicating whether compilation is required.
            connectivity (str): The initial connectivity of the genome.
            connection_density (float): The initial density of connections in the genome.
            n_modules (int): The number of modules in the genome.
            weight_sharing (bool): Whether to use weight sharing in the genome.
        """
        self.genome_id = genome_id
        self.fitness = 0.0
        
        self.hyperparameters = hyperparameters
        self.task = hyperparameters['task']
        self.offspring_hyperparameters = dict(hyperparameters)
        self.n_inputs = hyperparameters['n_inputs']
        self.n_outputs = hyperparameters['n_outputs']
        self.n_modules = hyperparameters['n_modules']
        self.weight_sharing = hyperparameters['weight_sharing']
        self.uniform_weights = hyperparameters['uniform_weights']
        self.mutation_rates = hyperparameters['mutation_rates']

        self.rng = np.random.default_rng()

        self.initialise_modules()

        self.initialise_grouped_connections()

        self.n_module_types = hyperparameters['n_module_types']
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

    def initialise_modules(self):
        """
        Initialise the modules for the genome.
        Generates:
            self.modules (list): The list of modules for the genome.
        """
        task = self.task
        if task == 'maze':
            self.modules = [Recurrent() for _ in range(self.n_modules)]
        elif task == 'image':
            c_type = self.hyperparameters['class_type']
            if c_type == 'square':
                self.modules = [SquareClassificationModule() for _ in range(self.n_modules)]
            elif c_type == 'cross':
                self.modules = [CrossClassificationModule() for _ in range(self.n_modules)]

    def initialise_grouped_connections(self):
        """
        Initialise grouped connections for the genome.
        Generates:
            self.mod_grouped_in (dict): Input connections grouped by destination module.
            self.out_grouped_out (dict): Output connections grouped by destination output.
            self._in_keys (list): The keys for each mod_grouped_in entry.
            self._out_keys (list): The keys for each out_grouped_out entry.
        """
        m, o = self.n_modules, self.n_outputs
        # Grouped dictionaries
        self.mod_grouped_in = {i: [] for i in range(m)} # Group input connections by destination module
        self.out_grouped_out = {i: [] for i in range(o)} # Group output connections by destination output

        # Key sets
        self._in_keys = [set() for _ in range(m)]
        self._out_keys = [set() for _ in range(o)] 

    def initialise_weight_sharing(self):
        """"
        Initialise the weight sharing for the genome.
        Generates:
            self.shrd_mod_in_wts (list): The shared weights for each modules's input connections.
            self.shrd_mod_out_wts (list): The shared weights for each modules's output connections.
        """
        n_mod_types = self.n_module_types
        # Uniform weights - all weights equal to 1.0
        # Shared random weights - weights per module sampled from a uniform distribution
        self.shrd_mod_in_wts, self.shrd_mod_out_wts = np.ones((2, n_mod_types), dtype=np.float64) if self.uniform_weights else self.rng.uniform(0.1, 1.0, (2, n_mod_types))

    def initialise_one_to_one(self):
        """
        Initialise the one-to-one connections for the genome.
        Generates:
            self.input_set (set): All input indices.
            self.used_inputs (set): All used input indices.
            self.output_set (set): All output indices.
            self.used_outputs (set): All used output indices.
            self.mod_in_set (set): All input indice-port pairs.
            self.used_mod_in (set): All used input indice-port pairs for each module.
            self.mod_out_set (set): All output indice-port pairs for each module.
            self.used_mod_out (set): All used output indice-port pairs for each module.
        """
        n_mods, mods = self.n_modules, self.modules
        # Input indices
        self.input_set, self.used_inputs = set(range(self.n_inputs)), set()

        # Output indices
        self.output_set, self.used_outputs = set(range(self.n_outputs)), set()

        # Module input port indices
        self.mod_in_set, self.used_mod_in = set((i, j) for i in range(n_mods) for j in range(mods[i].n_inputs)), set()

        # Module output port indices
        self.mod_out_set, self.used_mod_out = set((i, j) for i in range(n_mods) for j in range(mods[i].n_outputs)), set()

    def add_connect_rule(self, conn_rule):
        """Add a connection rule to the genome.
        Arguments:
            conn_rule (tuple): The connection rule.
        Updates:
            self.conn_in_rules (list): The rules for input connections.
            self.conn_out_rules (list): The rules for output connections.
            self.mod_grouped_in (dict): Input connections grouped by destination module.
            self.out_grouped_out (dict): Output connections grouped by destination output.
            self._in_keys (list): The keys for each mod_grouped_in entry.
            self._out_keys (list): The keys for each out_grouped_out entry.
            used_inputs (set): All used input indices.
            used_outputs (set): All used output indices.
            used_mod_in (set): All used input indice-port pairs for each module.
            used_mod_out (set): All used output indice-port pairs for each module.
            self.compile_flag (int): The flag indicating whether compilation is required.
        """
        match conn_rule[0]:
            case 1:  
                _, src, dst, dst_pt, wt = conn_rule
                key, item = (dst_pt, src), (dst_pt, src, wt)
                bucket, existing_keys = self.mod_grouped_in[dst], self._in_keys[dst]

                # Check connection does not already exist
                if key in existing_keys:
                    return
                
                # Insert new connection to mod_grouped_in and conn_in_rules
                if not bucket:
                    bucket.append(item)
                else:
                    bisect.insort(bucket, item)
                self.conn_in_rules.append(conn_rule)
                
                # Update keys
                existing_keys.add(key)

                # Update used inputs and module inputs
                if self.one_to_one:
                    self.used_mod_in.add((dst, dst_pt))
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

                # Update keys
                existing_keys.add(key)

                # Update used outputs and module outputs
                if self.one_to_one:
                    self.used_mod_out.add((src, src_pt))
                    self.used_outputs.add(dst)

    def remove_connect_rule(self, conn_rule):
        """Remove a connection rule from the genome.
        Arguments:
            conn_rule (tuple): The connection rule.
        Updates:
            self.conn_in_rules (list): The rules for input connections.
            self.conn_out_rules (list): The rules for output connections.
            self.mod_grouped_in (dict): Input connections grouped by destination module.
            self.out_grouped_out (dict): Output connections grouped by destination output.
            self._in_keys (list): The keys for each mod_grouped_in entry.
            self._out_keys (list): The keys for each out_grouped_out entry.
            self.used_inputs (set): All used input indices.
            self.used_outputs (set): All used output indices.
            self.used_mod_in (set): All used input indice-port pairs for each module.
            self.used_mod_out (set): All used output indice-port pairs for each module.
            self.compile_flag (int): The flag indicating whether compilation is required.
        """
        match conn_rule[0]:
            case 1:
                _, src, dst, dst_pt, wt = conn_rule
                # Remove connection from mod_grouped_in
                self.mod_grouped_in[dst].remove((dst_pt, src, wt))
                self._in_keys[dst].remove((dst_pt, src))

                # Update used inputs and module inputs
                if self.one_to_one:
                    self.used_mod_in.remove((dst, dst_pt))
                    self.used_inputs.remove(src)

            case 2:
                _, src, src_pt, dst, wt = conn_rule
                # Remove connection from out_grouped_out
                self.out_grouped_out[dst].remove((src_pt, src, wt))
                self._out_keys[dst].remove((src_pt, src))

                # Update used inputs and module inputs
                if self.one_to_one:
                    self.used_mod_out.remove((src, src_pt))
                    self.used_outputs.remove(dst)

    def initialise_connect_in_rules(self, connectivity, n_rules):
        """Add multiple input connection rules to the genome at initialisation.
        Arguments:
            connectivity (int): Connectivity type for the rules (0: SPARSE, 1: RANDOM).
            n_rules (int): Number of rules to generate if no rules are provided.
        """
        mods, rng, n_in, n_mods = self.modules, self.rng, self.n_inputs, self.n_modules
        match connectivity:
            case 0: 
                mod_ids = list(range(n_mods))

                # Reduce number of modules if one-to-one mapping and n_modules > n_inputs
                if self.one_to_one and n_rules > n_in:
                    n_rules = n_in
                    mod_ids = rng.choice(mod_ids, size=n_rules, replace=False).tolist()

                # Randomly sample one module-input port pair for each selected module
                pts = [rng.integers(mods[i].n_inputs) for i in mod_ids]

            case 1: 
                if self.one_to_one:
                    mod_set = list(self.mod_in_set)
                    # Size is the minimum of the given n_rules, available module ports, and available inputs
                    n_rules = min(n_rules, len(mod_set), len(self.input_set))

                    # Randomly sample `n_rules` unique module-input port pairs
                    idxs = rng.choice(len(mod_set), size=n_rules, replace=False).tolist()
                    mod_ids = [mod_set[i][0] for i in idxs]
                    pts = [mod_set[i][1] for i in idxs]
                else:
                    # Sample module-port pairs with replacement
                    mod_ids = rng.choice(n_mods, size=n_rules, replace=True).tolist()
                    pts = [rng.integers(mods[i].n_inputs) for i in mod_ids]

        # Randomly sample source inputs
        if self.one_to_one:
            srcs = rng.choice(n_in, size=n_rules, replace=False).tolist()
        else:
            src_pool = np.tile(np.arange(n_in), int(np.ceil(n_rules / n_in)))
            rng.shuffle(src_pool)
            srcs = src_pool[:n_rules]

        # Randomly sample weights from uniform distribution or use module shared weights
        if self.weight_sharing:
            mod_type_ids = np.fromiter((mods[i].module_id for i in mod_ids), count=n_rules, dtype=np.int32)
            wts = np.asarray(self.shrd_mod_in_wts)[mod_type_ids]
        else:
            wts = rng.uniform(0.1, 1.0, size=n_rules)

        # Create connection rules
        for src, dst, dst_pt, wt in zip(srcs, mod_ids, pts, wts):
            self.add_connect_rule((1, src, dst, dst_pt, wt))
        
    def initialise_connect_out_rules(self, connectivity, n_rules):
        """Add multiple outputs connection rules to the genome at initialisation.
        Arguments:
            connectivity (int): Connectivity type for the rules (0: SPARSE, 1: RANDOM).
            n_rules (int): Number of rules to generate if no rules are provided.
        """
        mods, n_mods, n_out, rng = self.modules, self.n_modules, self.n_outputs, self.rng
        match connectivity:
            case 0:  
                mod_ids = list(range(n_mods))

                # Reduce number of modules if one-to-one mapping and n_modules > n_outputs
                if self.one_to_one and n_rules > n_out:
                    n_rules = n_out
                    mod_ids = rng.choice(mod_ids, size=n_rules, replace=True).tolist()

                pts = [rng.integers(mods[i].n_outputs) for i in mod_ids]
            case 1:  
                if self.one_to_one:
                    mod_set = list(self.mod_out_set)
                    # Size is the minimum of the given n_rules, available modules ports, and available outputs
                    n_rules = min(n_rules, len(mod_set), len(self.output_set))

                    # Randomly sample `n_rules` unique module-output port pairs
                    idxs = rng.choice(len(mod_set), size=n_rules, replace=False).tolist()
                    mod_ids = [mod_set[i][0] for i in idxs]
                    pts = [mod_set[i][1] for i in idxs]
                else:
                    # Sample module-output port pairs with replacement
                    mod_ids = rng.choice(self.n_modules, size=n_rules, replace=True).tolist()
                    pts = [rng.integers(mods[i].n_outputs) for i in mod_ids]

        # Randomly sample output ports  
        if self.one_to_one:
            dsts = rng.choice(n_out, size=n_rules, replace=False).tolist()
        else:
            dst_pool = np.tile(np.arange(n_out), int(np.ceil(n_rules / n_out)))
            rng.shuffle(dst_pool)
            dsts = dst_pool[:n_rules]

        # Randomly sample weights from uniform distribution or use module shared weights
        if self.weight_sharing:
            mod_type_ids = np.fromiter((mods[i].module_id for i in mod_ids), count=n_rules, dtype=np.int32)
            wts = np.asarray(self.shrd_mod_in_wts)[mod_type_ids]
        else:
            wts = rng.uniform(0.1, 1.0, size=n_rules)

        # Create connection rules
        for src, src_pt, dst, wt in zip(mod_ids, pts, dsts, wts):
            self.add_connect_rule((2, src, src_pt, dst, wt))

    def compile_rules(self):
        """
        Compile connection rules into ragged arrays.
        Generates:
            in_edges (Edges): The edges for input connections.
            out_edges (Edges): The edges for output connections.
            mod_edges (list[Edges]): The edges for module connections.
            mod_vec_edges (list[Edges]): The edges for module vector connections.
        """
        # ------------------------------------------- #
        # ------------ Input Connections ------------ #
        # ------------------------------------------- #

        # Initialise arrays grouped by module
        mod_g_in, n_mods = self.mod_grouped_in, self.n_modules
        in_lens = np.fromiter((len(mod_g_in[m]) for m in range(n_mods)), count=n_mods, dtype=np.int32)
        in_indptr = np.concatenate(([0], np.cumsum(in_lens, dtype=np.int32)))
        n = in_indptr[-1]
        in_srcs, in_src_pts, in_dst_pts, in_wts = np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.float32)

        # Fill arrays
        for m in range(n_mods):
            start, end = in_indptr[m], in_indptr[m + 1]
            if start == end:
                continue
            entries = mod_g_in[m]
            in_srcs[start:end] = [t[1] for t in entries]
            in_dst_pts[start:end] = [t[0] for t in entries]
            in_wts[start:end] = [t[2] for t in entries]

        # Create edges from arrays
        deg = in_indptr[1:] - in_indptr[:-1]
        dst = np.repeat(np.arange(n_mods, dtype=np.int32), deg)
        self.in_edges = Edges(in_indptr, in_srcs, in_src_pts, in_dst_pts, in_wts, dst)

        # -------------------------------------------- #
        # ------------ Output Connections ------------ #
        # -------------------------------------------- #

        # Initialise arrays grouped by output
        out_g_out, n_out = self.out_grouped_out, self.n_outputs
        out_lens = np.fromiter((len(out_g_out[o]) for o in range(n_out)), count=n_out, dtype=np.int32)
        out_indptr = np.concatenate(([0], np.cumsum(out_lens, dtype=np.int32)))
        n = out_indptr[-1]
        out_srcs, out_src_pts, out_dst_pts, out_wts = np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.float32)

        # Fill arrays
        for o in range(n_out):
            start, end = out_indptr[o], out_indptr[o + 1]
            if start == end:
                continue
            entries = out_g_out[o]
            out_srcs[start:end] = [t[1] for t in entries]
            out_src_pts[start:end] = [t[0] for t in entries]
            out_wts[start:end] = [t[2] for t in entries]

        # Create edges from arrays
        deg = out_indptr[1:] - out_indptr[:-1]
        dst = np.repeat(np.arange(n_out, dtype=np.int32), deg)
        self.out_edges = Edges(out_indptr, out_srcs, out_src_pts, out_dst_pts, out_wts, dst) 

    def initialise_genome(self):
        """Initialise the genome."""
        match self.connectivity:
            case 'UNCONNECTED': return
            case 'SPARSE':
                n = self.n_modules
                # Sparse connectivity: Connect each module with one input and one output
                self.initialise_connect_in_rules(0, n)
                self.initialise_connect_out_rules(0, n)

            case 'RANDOM':
                conn_d = self.connection_density
                # Random connectivity: Connect a random subset of inputs and outputs to random modules
                n_in, n_out = int(conn_d['input_density'] * self.n_inputs), int(conn_d['output_density'] * self.n_outputs)
                if n_in > 0:
                    self.initialise_connect_in_rules(1, n_in)
                if n_out > 0:
                    self.initialise_connect_out_rules(1, n_out)

            case 'FULLY CONNECTED':
                # Fully connected: Connect all inputs and outputs to all motifs
                rng, n_in, n_out = self.rng, self.n_inputs, self.n_outputs
                shr_wts, shrd_in_wts, shrd_out_wts = self.weight_sharing, self.shrd_mod_in_wts, self.shrd_mod_out_wts

                for i, m in enumerate(self.modules):
                    # Sample weights from uniform distribution or shared weights
                    if shr_wts:
                        in_wt, out_wt = shrd_in_wts[m.module_id], shrd_out_wts[m.module_id]
                    else:
                        in_wt, out_wt = rng.uniform(0.1, 1.0, 2)

                    # Connect all network inputs
                    for pt in range(m.n_inputs):
                        for src in range(n_in):
                            self.add_connect_rule((1, src, i, pt, in_wt))

                    # Connect all network outputs
                    for pt in range(m.n_outputs):
                        for dst in range(n_out):
                            self.add_connect_rule((2, i, pt, dst, out_wt))

            case 'IDEAL':
                # Ideal: Connect each module and node ideally - for prototyping only
                if self.task == 'maze':
                    for i in range(self.n_inputs):
                        mod = i // 2
                        self.add_connect_rule((1, i, mod, 0, 1.0))
                        self.add_connect_rule((1, i, mod, 1, 1.0))
                    
                    for i in range(self.n_outputs):
                        self.add_connect_rule((2, i, 0, i, 1.0))

                if self.task == 'image':
                    side_len = int(np.sqrt(self.n_inputs))
                    mod_side_len = int(np.sqrt(side_len))
                    for row in range(side_len):
                        for col in range(side_len):
                            in_idx = row * side_len + col
                            mod = (row // mod_side_len) * mod_side_len + (col // mod_side_len)
                            prt = col % mod_side_len + (row % mod_side_len) * mod_side_len
                            self.add_connect_rule((1, in_idx, mod, prt, 1.0))  
                            self.add_connect_rule((2, mod, prt, in_idx, 1.0))  
    
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
        mod_outputs = self.layer_forward_pass(0, input_vector, self.n_modules)
        outputs = self.layer_forward_pass(1, mod_outputs, self.n_outputs)
        
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
                mods = self.modules
                edges = self.in_edges
                inter_sizes = np.fromiter((m.n_inputs for m in mods), dtype=np.int32)
                out_sizes = np.fromiter((m.n_outputs for m in mods), dtype=np.int32)

            case 1: 
                edges = self.out_edges
                inter_sizes = np.ones(lyr_size, dtype=np.float32)

        # Build intermediate pointers and values
        inter_ptr = self.build_ptr_from_sizes(inter_sizes)
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
        out_ptr = self.build_ptr_from_sizes(out_sizes)
        out_vals = np.zeros(out_ptr[-1], dtype=np.float32)

        # Compute contributions from modules
        for i in range(lyr_size):
            # Check the module has edges
            start, end = inter_ptr[i], inter_ptr[i + 1]
            if start == end:
                continue

            # Fill in the module forward pass outputs
            mod = self.modules[i]
            out_start, out_end = out_ptr[i], out_ptr[i + 1]
            out_vals[out_start:out_end] = mod.forward_pass(inter_vals[start:end])

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
        child = Genome(new_id, self.offspring_hyperparameters)
        cut = np.random.randint(1, min(self.n_modules, parent_2.n_modules))
        child.modules = copy.deepcopy(self.modules[:cut]) + copy.deepcopy(parent_2.modules[cut:])
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
                    used_in, used_mod_in = set(), set()
                    for rule in rule_list1:
                        if rule[2] < cut and rule[1] not in used_in and (rule[2], rule[3]) not in used_mod_in:
                            used_in.add(rule[1])
                            used_mod_in.add((rule[2], rule[3]))
                            child.add_connect_rule(rule)
                    for rule in rule_list2:
                        if rule[2] > cut and rule[1] not in used_in and (rule[2], rule[3]) not in used_mod_in:
                            used_in.add(rule[1])
                            used_mod_in.add((rule[2], rule[3]))
                            child.add_connect_rule(rule)        
                else:
                    rules = [rule for rule in rule_list1 if rule[2] < cut] + [rule for rule in rule_list2 if rule[2] > cut]
                    for r in rules:
                        child.add_connect_rules(r)
            case 2: 
                used_out, used_mod_out = set(), set()
                if self.one_to_one:
                    for rule in rule_list1:
                        if rule[1] < cut and (rule[1], rule[2]) not in used_mod_out and rule[3] not in used_out:
                            used_mod_out.add((rule[1], rule[2]))
                            used_out.add(rule[3])
                            child.add_connect_rule(rule)
                    for rule in rule_list2:
                        if rule[1] > cut and (rule[1], rule[2]) not in used_mod_out and rule[3] not in used_out:
                            used_mod_out.add((rule[1], rule[2]))
                            used_out.add(rule[3])
                            child.add_connect_rule(rule)
                else:
                    rules = [rule for rule in rule_list1 if rule[1] < cut] + [rule for rule in rule_list2 if rule[1] > cut]
                    for r in rules:
                        child.add_connect_rule(r)

    def mutate(self):
        """Mutate the genome by randomly modifying its structure."""
        if not self.conn_in_rules or not self.conn_out_rules:
            mut = 0
        else:
            mut = np.random.choice([0, 1, 2, 3], p=self.mutation_rates)

        match mut:
            case 0: self.add_connection()
            case 1: self.remove_connection()
            case 2: self.swap_connection()
            case 3: self.modify_connection_weight()

    def add_connection(self):
        """Add a new connection from a genome input or to a genome output."""
        rng, modules = self.rng, self.modules
        conn_type = rng.integers(1, 3)  # 1: CONNECT_IN, 2: CONNECT_OUT
        
        match conn_type:
            case 1:
                if self.one_to_one:
                    # Get the set of available module-port pairs
                    mod_set = list(self.mod_in_set - self.used_mod_in)
                    # Get the set of available inputs
                    src_set = list(self.input_set - self.used_inputs)

                    # If there are no available module-port pairs or inputs, swap connections
                    if not mod_set or not src_set:
                        self.swap_connection(conn_type=1)
                        return

                    # Get a random module-port pair and input
                    mod_id, dst_pt = mod_set[rng.integers(len(mod_set))]
                    mod = modules[mod_id]
                    src = rng.choice(src_set)
                else:
                    # Get the set of available modules
                    ids = [i for i in range(self.n_modules)]
                    
                    # If there are no available modules, swap connections
                    if not ids:
                        self.swap_connection(conn_type=1)
                        return

                    # Get a random module-port pair and input
                    mod_id = rng.choice(ids)
                    mod = modules[mod_id]
                    dst_pt = rng.integers(0, mod.n_inputs)
                    src = rng.integers(0, self.n_inputs)

                wt = self.shrd_mod_in_wts[mod.module_id] if self.weight_sharing else rng.uniform(0.1, 1.0)
                rule = 1, src, mod_id, dst_pt, wt
            case 2:
                if self.one_to_one:
                    # Get the set of available module-port pairs
                    mod_set = list(self.mod_out_set - self.used_mod_out)
                    # Get the set of available outputs
                    dst_set = list(self.output_set - self.used_outputs)

                    # If there are no available module-port pairs or outputs, swap connections
                    if not mod_set or not dst_set:
                        self.swap_connection(conn_type=2)
                        return

                    # Get a random module-port pair and output
                    mod_id, src_pt = mod_set[rng.integers(len(mod_set))]
                    # Vector connections cannot be swapped
                    mod = modules[mod_id]
                    if not mod.scalar_out:
                        return
                    dst = rng.choice(dst_set)
                else:
                    # Get the set of available modules
                    ids = [i for i in range(self.n_modules)]
                    
                    # If there are no available modules, swap connections
                    if not ids:
                        self.swap_connection(conn_type=2)
                        return

                    # Get a random module-port pair and output
                    mod_id = rng.choice(ids)
                    mod = modules[mod_id]
                    # Vector connections cannot be swapped
                    if not mod.scalar_out:
                        return
                    src_pt = rng.integers(0, mod.n_outputs)
                    dst = rng.integers(0, self.n_outputs)

                wt = self.shrd_mod_out_wts[mod.module_id] if self.weight_sharing else rng.uniform(0.1, 1.0)
                rule = 2, mod_id, src_pt, dst, wt
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
            conn_type (int): The type of connection to swap (1: CONNECT_IN, 2: CONNECT_OUT). If not provided, a random type will be chosen.
        Updates:
            self.conn_in_rules (list): The rules for input connections.
            self.conn_out_rules (list): The rules for output connections.
            self.mod_grouped_in (dict): Input connections grouped by destination module.
            self.out_grouped_out (dict): Output connections grouped by destination output.
            self._in_keys (list): The keys for each mod_grouped_in entry.
            self._out_keys (list): The keys for each out_grouped_out entry.
            self.used_inputs (set): All used input indices.
            self.used_outputs (set): All used output indices.
            self.used_mod_in (set): All used input indice-port pairs for each module.
            self.used_mod_out (set): All used output indice-port pairs for each module.
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

                # Swap the connection srcs
                in_rules[idx1], in_rules[idx2] = (1, src2, dst1, pt1, wt1), (1, src1, dst2, pt2, wt2)

                # Update grouped dictionaries and key sets
                mod_g_in, in_keys = self.mod_grouped_in, self._in_keys
                mod_g_in[dst1].remove((pt1, src1, wt1))
                in_keys[dst1].discard((pt1, src1))
                mod_g_in[dst2].remove((pt2, src2, wt2))
                in_keys[dst2].discard((pt2, src2))
                mod_g_in[dst1].append((pt1, src2, wt1))
                in_keys[dst1].add((pt1, src2))
                mod_g_in[dst2].append((pt2, src1, wt2))
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

                # Swap the connection dsts
                out_rules[idx1], out_rules[idx2] = (2, src1, pt1, dst2, wt1), (2, src2, pt2, dst1, wt2)

                # Update grouped dictionaries and key sets
                out_g_out, out_keys = self.out_grouped_out, self._out_keys
                out_g_out[dst1].remove((pt1, src1, wt1))
                out_keys[dst1].discard((pt1, src1))
                out_g_out[dst2].remove((pt2, src2, wt2))
                out_keys[dst2].discard((pt2, src2))
                out_g_out[dst1].append((pt2, src2, wt2))
                out_keys[dst1].add((pt2, src2))
                out_g_out[dst2].append((pt1, src1, wt1))
                out_keys[dst2].add((pt1, src1))

    def modify_connection_weight(self):
        """
        Modify the weight of a randomly selected connection.
        Updates:
            self.conn_in_rules (list): The rules for input connections.
            self.conn_out_rules (list): The rules for output connections.
            self.mod_grouped_in (dict): Input connections grouped by destination module.
            self.out_grouped_out (dict): Output connections grouped by destination output.
            self.compile_flag (int): The flag indicating whether compilation is required.
        """
        rng = self.rng
        if self.weight_sharing:
            return

        conn_type = rng.integers(1, 3)  # 1: CONNECT_IN, 2: CONNECT_OUT
        self.compile_flag = 1

        match conn_type:
            case 1: 
                in_rules = self.conn_in_rules
                if not in_rules:
                    return
                # Randomly select connection rule
                idx = rng.integers(len(in_rules))
                tp, src, dst, dst_pt, wt = in_rules[idx]

                # Randomly generate new weight
                new_wt = np.clip(wt + rng.normal(0, 0.1), 0.0, 1.0)

                # Update the connection rule with the new weight
                in_rules[idx] = (tp, src, dst, dst_pt, new_wt)
                self.mod_grouped_in[dst].remove((dst_pt, src, wt))
                self.mod_grouped_in[dst].append((dst_pt, src, new_wt))

            case 2:
                out_rules = self.conn_out_rules
                if not out_rules:
                    return
                # Randomly select connection rule
                idx = rng.integers(len(out_rules))
                tp, src, src_pt, dst, wt = out_rules[idx]

                # Randomly generate new weight
                new_wt = np.clip(wt + rng.normal(0, 0.1), 0.0, 1.0)

                # Update the connection rule with the new weight
                out_rules[idx] = (tp, src, src_pt, dst, new_wt)
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
        child = Genome(new_id, self.offspring_hyperparameters)

        # Clone module information
        child.modules = copy.deepcopy(self.modules)

        # Clone module and connection rules
        child.conn_in_rules = copy.deepcopy(self.conn_in_rules)
        child.conn_out_rules = copy.deepcopy(self.conn_out_rules)
        child.mod_grouped_in = copy.deepcopy(self.mod_grouped_in)
        child.out_grouped_out = copy.deepcopy(self.out_grouped_out)
        child._in_keys = copy.deepcopy(self._in_keys)
        child._out_keys = copy.deepcopy(self._out_keys)
        child.used_inputs = copy.deepcopy(self.used_inputs)
        child.used_outputs = copy.deepcopy(self.used_outputs)
        child.used_mod_in = copy.deepcopy(self.used_mod_in)
        child.used_mod_out = copy.deepcopy(self.used_mod_out)

        # Reset and return new genome
        child.reset()
        return child
    
    def reset(self):
        """
        Reset the genome to its initial state.
        Resets:
            (Module): Each module instance in self.modules.
            self.compile_flag (int): The flag indicating whether compilation is required.
        """
        # Reset modules
        for mod in self.modules:
            mod.reset()

        # Reset compile flag
        self.compile_flag = 1 
    
    def build_ptr_from_sizes(self, sizes):
        """
        Build a pointer array from the given sizes.
        Arguments:
            sizes (np.ndarray): An array of sizes.
        Returns:
            ptr (np.ndarray): A pointer array indicating the start of each segment.
        """
        # Initialize pointer array
        ptr = np.zeros(sizes.size + 1, dtype=np.int32)

        # Build and return pointer array
        np.cumsum(sizes, out=ptr[1:])
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

        # Module nodes
        for i in range(self.n_modules):
            name = f'M_{i}'
            graph.add_node(name, type='module')
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
        plt_size = (6, 8) if self.task == 'maze' else (6, 30)
        plt.figure(figsize=plt_size)
        nx.draw(graph, pos, with_labels=True, node_color=node_colors, arrows=True, node_size=2000, font_size=10, font_color='black', edge_color='gray')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red', font_size=8, horizontalalignment='left')
        plt.title(f"Genome {self.genome_id}")
        max_y = max(self.n_modules, self.n_outputs, self.n_inputs)
        plt.ylim(-1, max_y + 1)
        plt.xlim(-2.5, 2.5)
        plt.axis('off')
        plt.show()
    