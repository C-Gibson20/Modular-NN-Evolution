import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import bisect
from enum import IntEnum
from collections import defaultdict
from dataclasses import dataclass
from collections import Counter

class RuleType(IntEnum):
    """
    Enumeration for different rule types in the genome.
    Values:
        CONNECT_IN: Connections from network inputs to modules
        CONNECT_OUT: Connections from modules to network outputs
        CONNECT_MOD: Connections between modules
    """
    CONNECT_IN = 1
    CONNECT_OUT = 2
    CONNECT_MOD = 3

class NetworkType(IntEnum):
    """
    Enumeration for different network types.
    Values:
        WIDE: Wide network - width = n_modules, depth = 1
        DEEP: Deep network - width = 1, depth = n_modules
        WIDE_AND_DEEP: Wide and deep network - width = n, depth = m where n + m = n_modules
    """
    WIDE = 0
    DEEP = 1
    WIDE_AND_DEEP = 2

@dataclass
class Edges:
    """
    Class representing the edges in the genome.
    Properties:
        indptr (np.ndarray): The index pointer for the edges.
        srcs (np.ndarray): The source nodes for the edges.
        src_pts (np.ndarray): The source points for the edges.
        dst_pts (np.ndarray): The destination points for the edges.
        wts (np.ndarray): The weights for the edges.
        dst (np.ndarray): The destination nodes for the edges.
    """
    indptr: np.ndarray
    srcs: np.ndarray
    src_pts: np.ndarray
    dst_pts: np.ndarray
    wts: np.ndarray
    dst: np.ndarray

@dataclass
class VectorEdges:
    """
    Class representing the vectorized edges in the genome.
    Properties:
        indptr (np.ndarray): The index pointer for the edges.
        src_vecs (np.ndarray): The source vectors for the edges.
        wts (np.ndarray): The weights for the edges.
        dst (np.ndarray): The destination nodes for the edges.
    """
    indptr: np.ndarray
    src_vecs: np.ndarray
    wts: np.ndarray
    dst: np.ndarray

@dataclass
class Ragged:
    """
    Class representing a ragged array.
    Properties:
        vals (np.ndarray): The values in the ragged array.
        ptr (np.ndarray): The pointer array for the ragged array.
    """
    vals: np.ndarray
    ptr: np.ndarray

    def view(self, i):
        """
        View the values for a specific index in the ragged array.
        Arguments:
            i (int): The index to view.
        Returns:
            (np.ndarray): The values for the specified index.
        """
        start, end = self.ptr[i], self.ptr[i + 1]
        return self.vals[start:end]

# Input connection tuples
# (
    # ('rule_type', np.int32), # 1: connect_in
    # ('src', np.int32),       # source node ID: network input
    # ('dst', np.int32),       # destination node ID: module_id
    # ('dst_port', np.int32),  # module input port
    # ('weight', np.float32)   # weight of the connection
# )

# Output connection tuples
# (
#     ('rule_type', np.int32), # 2: connect_out
#     ('src', np.int32),       # source node ID: module_id
#     ('src_port', np.int32),  # module output port
#     ('dst', np.int32),       # destination node ID: network output
#     ('weight', np.float32)   # weight of the connection
# )

# Module connection tuples
# (
#     ('rule_type', np.int32), # 3: connect_mod
#     ('src_lyr', np.int32),   # source layer ID
#     ('src', np.int32),       # source node ID: module_id
#     ('src_port', np.int32),  # module output port
#     ('dst_lyr', np.int32),   # destination layer ID
#     ('dst', np.int32),       # destination node ID: module_id
#     ('dst_port', np.int32),  # module input port
#     ('weight', np.float32)   # weight of the connection
# )
# Module layer will also be a property of the module class

class GenomeDist():
    def __init__(self, genome_id, hyperparameters, module_bank):
        """
        Initialize the GenomeDist object.
        Arguments:
            genome_id (int): The ID of the genome.
            hyperparameters (dict): The hyperparameters for the genome.
            module_bank (ModuleBank): The module bank containing the available modules.
        Properties:
            task (str): The task to be solved by the genome.
            offspring_hyperparameters (dict): The hyperparameters for the offspring genomes.
            n_inputs (int): The number of inputs to the genome.
            n_outputs (int): The number of outputs from the genome.
            n_modules (int): The number of modules in the genome.
            weight_sharing (bool): Whether to use weight sharing in the genome.
            uniform_weights (bool): Whether to use uniform weights in the genome.
            mutation_rates (dict): The mutation rates for the genome.
            rng (np.random.Generator): The random number generator for the genome.
            network_type (int): The type of network (0: wide, 1: deep).
            n_hid_layers (int): The number of hidden layers in the genome.
            conn_in_rules (list): The rules for input connections.
            conn_mod_rules (list): The rules for module connections.
            conn_out_rules (list): The rules for output connections.
            compile_flag (int): The flag indicating whether compilation is required.
            connectivity (float): The initial connectivity of the genome.
            connection_density (float): The initial density of connections in the genome.
        """
        self.genome_id = genome_id
        self.fitness = 0.0

        self.module_bank = module_bank
        
        self.hyperparameters = hyperparameters
        self.task = hyperparameters['task']
        self.offspring_hyperparameters = dict(hyperparameters)
        self.n_inputs = hyperparameters['n_inputs']
        self.n_outputs = hyperparameters['n_outputs']
        self.n_modules = hyperparameters['n_modules']
        self.weight_sharing = hyperparameters['weight_sharing']
        self.uniform_weights = hyperparameters['uniform_weights']
        self.mutation_rates = hyperparameters['mutation_rates']

        self.network_type = hyperparameters['network_type']
        self.n_hid_layers = 0

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
        self.conn_mod_rules = []
        self.conn_out_rules = []

        self.compile_flag = 1

        self.connectivity = hyperparameters['connectivity']
        self.connection_density = hyperparameters['connection_density']
        self.offspring_hyperparameters['connectivity'] = 'UNCONNECTED'
        self.initialise_genome()

    # Todo: wide and deep case
    def initialise_modules(self):    
        """
        Initialise the modules for the genome.
        Generates:
            self.mod_classes (dict): The mapping of module names to their classes.
            self.scalar_mod_classes (dict): The mapping of scalar module names to their classes.
            self.mod_count (dict): The mapping of modules to their counts.
            self.modules (list): A module instances in the genome.
            self.max_hid_layers (int): The maximum number of hidden layers in the genome.
            self.n_hid_layers (int): The number of hidden layers in the genome.
        """
        mod_bank, bank, rng = self.module_bank, self.module_bank.bank, self.rng
        # Initialise module attributes
        mod_classes = {nm: cls for nm, (cls, _) in bank.items()}
        self.scalar_mod_classes = {nm: cls for nm, (cls, _) in mod_bank.scalar_bank.items()}
        mod_count = {nm: 0 for nm in bank}

        # Distribute modules according to initial bank
        n_mods = 0
        n = self.n_modules
        for nm, (_, v) in bank.items():
            if v > 0:
                count = int(v * n)
                mod_count[nm] = count
                n_mods += count

        # Ensure the total number of modules matches the necessary count
        if n_mods < n:
            mod_count['feedforward'] += n - n_mods

        # Populate module instances
        modules = []
        for nm, c in mod_count.items():
            if c == 0:
                continue
            for _ in range(c):
                modules.append(mod_classes[nm]())
        idx = np.arange(len(modules))
        rng.shuffle(idx)
        self.modules = [modules[i] for i in idx.tolist()]
        
        self.mod_classes = mod_classes
        self.mod_count = mod_count
        self.recompute_module_distribution()

        # Update layer properties of the network
        if self.network_type == 0:
            self.max_hid_layers = 1
            self.n_hid_layers = 1
            for m in self.modules:
                m.layer = 1
        elif self.network_type == 1:
            self.max_hid_layers = n
            self.n_hid_layers = n
            for i, m in enumerate(self.modules):
                m.layer = i + 1
        else:
            # Initialise wide and deep networks with one layer
            pass

    def initialise_grouped_connections(self):
        """
        Initialise the grouped connections for the genome.
        Generates:
            self.mod_grouped_in (dict): Input connections grouped by destination module.
            self.out_grouped_out (dict): Output connections grouped by destination output.
            self.mod_grouped_mod (dict): Module connections grouped by destination layer and module.
            self.mod_grouped_mod_vec (dict): Vector connections grouped by destination layer and module.
            self._in_keys (list): The keys for each mod_grouped_in entry.
            self._out_keys (list): The keys for each out_grouped_out entry.
            self._mod_keys (list): The keys for each mod_grouped_mod entry.
            self._mod_vec_keys (list): The keys for each mod_grouped_mod_vec entry.
        """
        n, m = self.n_modules, self.max_hid_layers + 2
        
        # Grouped dictionaries
        self.mod_grouped_in = {i: [] for i in range(n)} # Group input connections by destination module
        self.out_grouped_out = {i: [] for i in range(self.n_outputs)} # Group output connections by destination output
        self.mod_grouped_mod = {i : {j: [] for j in range(n)} for i in range(1, m)} # Group connections between modules by destination layer, then module
        self.mod_grouped_mod_vec = {i : {j: [] for j in range(n)} for i in range(1, m)} # Group vector connections between modules by destination layer, then module

        # Key sets
        self._in_keys, self._out_keys = [set() for _ in range(n)], [set() for _ in range(n)]
        self._mod_keys, self._mod_vec_keys = [[set() for _ in range(n)] for _ in range(m)], [[set() for _ in range(n)] for _ in range(m)]

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

    def initialise_weight_sharing(self):
        """
        Initialise the weight sharing for the genome.
        Generates:
            self.shrd_mod_in_wts (list): The shared weights for each modules's input connections.
            self.shrd_mod_out_wts (list): The shared weights for each modules's output connections.
            self.shrd_mod_wts (list): The shared weights for each modules's module connections.
        """
        n_mod_types = self.n_module_types
        # Uniform weights - all weights equal to 1.0
        # Shared random weights - weights per module sampled from a uniform distribution
        self.shrd_mod_in_wts, self.shrd_mod_out_wts, self.shrd_mod_wts = np.ones((3, n_mod_types), dtype=np.float64) if self.uniform_weights else self.rng.uniform(0.1, 1.0, (3, n_mod_types))

    def add_connect_rule(self, conn_rule):
        """Add a connection rule to the genome.
        Arguments:
            conn_rule (tuple): The connection rule.
        Updates:
            self.conn_in_rules (list): The rules for input connections.
            self.conn_out_rules (list): The rules for output connections.
            self.conn_mod_rules (list): The rules for module connections.
            self.mod_grouped_in (dict): Input connections grouped by destination module.
            self.out_grouped_out (dict): Output connections grouped by destination output.
            self.mod_grouped_mod (dict): Module connections grouped by destination layer and module.
            self.mod_grouped_mod_vec (dict): Vector connections grouped by destination layer and module.
            self._in_keys (list): The keys for each mod_grouped_in entry.
            self._out_keys (list): The keys for each out_grouped_out entry.
            self._mod_keys (list): The keys for each mod_grouped_mod entry.
            self._mod_vec_keys (list): The keys for each mod_grouped_mod_vec entry.
            used_inputs (set): All used input indices.
            used_outputs (set): All used output indices.
            used_mod_in (set): All used input indice-port pairs for each module.
            used_mod_out (set): All used output indice-port pairs for each module.
            self.compile_flag (int): The flag indicating whether compilation is required.
        """
        self.compile_flag = 1
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
            case 3:
                tp, src_lyr, src, src_pt, dst_lyr, dst, dst_pt, wt = conn_rule
                
                # Module with scalar output connections
                if self.modules[src].scalar_out:
                    key, item = (dst_pt, src_lyr, src_pt, src), (dst_pt, src_lyr, src_pt, src, wt)
                    bucket, existing_keys = self.mod_grouped_mod[dst_lyr][dst], self._mod_keys[dst_lyr][dst]

                    # Check connection does not already exist
                    if key in existing_keys:
                        return
                    
                    # Insert new connection to conn_mod_rules
                    self.conn_mod_rules.append(conn_rule)
                else:
                    key, item = (src_lyr, src), (src_lyr, src, wt)
                    bucket, existing_keys = self.mod_grouped_mod_vec[dst_lyr][dst], self._mod_keys[dst_lyr][dst]

                    # Check connection does not already exist
                    if key in existing_keys:
                        return

                    # Insert new connection to conn_mod_rules
                    self.conn_mod_rules.append((tp, src_lyr, src, 0, dst_lyr, dst, 0, wt))

                    # Update ports
                    src_pt, dst_pt = 0, 0
                    

                if not bucket:
                    bucket.append(item)
                else:
                    bisect.insort(bucket, item)

                # Update keys
                existing_keys.add(key)

                # Update used inputs and module inputs
                if self.one_to_one:
                    self.used_mod_out.add((src, src_pt))
                    self.used_mod_in.add((dst, dst_pt))

    def remove_connect_rule(self, conn_rule):
        """Remove a connection rule from the genome.
        Arguments:
            conn_rule (tuple): The connection rule.
        Updates:
            self.conn_in_rules (list): The rules for input connections.
            self.conn_out_rules (list): The rules for output connections.
            self.conn_mod_rules (list): The rules for module connections.
            self.mod_grouped_in (dict): Input connections grouped by destination module.
            self.out_grouped_out (dict): Output connections grouped by destination output.
            self.mod_grouped_mod (dict): Module connections grouped by destination layer and module.
            self.mod_grouped_mod_vec (dict): Vector connections grouped by destination layer and module.
            self._in_keys (list): The keys for each mod_grouped_in entry.
            self._out_keys (list): The keys for each out_grouped_out entry.
            self._mod_keys (list): The keys for each mod_grouped_mod entry.
            self._mod_vec_keys (list): The keys for each mod_grouped_mod_vec entry.
            self.used_inputs (set): All used input indices.
            self.used_outputs (set): All used output indices.
            self.used_mod_in (set): All used input indice-port pairs for each module.
            self.used_mod_out (set): All used output indice-port pairs for each module.
            self.compile_flag (int): The flag indicating whether compilation is required.
        """
        self.compile_flag = 1

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
            case 3:
                _, src_lyr, src, src_pt, dst_lyr, dst, dst_pt, wt = conn_rule
                # Remove connection from mod_grouped_mod or mod_grouped_mod_vec
                if self.modules[src].scalar_out:
                    self.mod_grouped_mod[dst_lyr][dst].remove((dst_pt, src_lyr, src_pt, src, wt))
                    self._mod_keys[dst_lyr][dst].remove((dst_pt, src_lyr, src_pt, src))
                else:
                    self.mod_grouped_mod_vec[dst_lyr][dst].remove((src_lyr, src, wt))
                    self._mod_vec_keys[dst_lyr][dst].remove((src_lyr, src))

                # Update used inputs and module inputs
                if self.one_to_one:
                    self.used_mod_out.remove((src, src_pt))
                    self.used_mod_in.remove((dst, dst_pt))

    def initialise_connect_in_rules(self, connectivity, n_rules):
        """Add multiple input connection rules to the genome at initialisation.
        Arguments:
            connectivity (int): Connectivity type for the rules (0: SPARSE, 1: RANDOM).
            n_rules (int): Number of rules to generate if no rules are provided.
        """
        mods, rng, n_in, n_mods = self.modules, self.rng, self.n_inputs, self.n_modules
        match connectivity:
            case 0: 
                mod_ids = [i for i in range(n_mods) if mods[i].layer == 1]

                # Reduce number of modules if one-to-one mapping and n_modules > n_inputs
                if self.one_to_one and n_rules > n_in:
                    n_rules = n_in
                    mod_ids = rng.choice(mod_ids, size=n_rules, replace=False).tolist()

                # Randomly sample one module-input port pair for each selected module
                pts = [rng.integers(mods[i].n_inputs) for i in mod_ids]

            case 1: 
                if self.one_to_one:
                    # Only consider modules in the first layer
                    mod_set = [(mod, pt) for mod, pt in self.mod_in_set if mods[mod].layer == 1]
                    # Size is the minimum of the given n_rules, available module ports, and available inputs
                    n_rules = min(n_rules, len(mod_set), len(self.input_set))

                    # Randomly sample `n_rules` unique module-input port pairs
                    idxs = rng.choice(len(mod_set), size=n_rules, replace=False).tolist()
                    mod_ids = [mod_set[i][0] for i in idxs]
                    pts = [mod_set[i][1] for i in idxs]
                else:
                    # Sample module-port pairs with replacement
                    mod_ids = rng.choice([i for i, mod in enumerate(mods) if mod.layer == 1], size=n_rules, replace=True).tolist()
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
        mods, n_mods, n_out, rng, n_hid = self.modules, self.n_modules, self.n_outputs, self.rng, self.n_hid_layers
        match connectivity:
            case 0:  
                # Randomly sample 1 module-output port pair per module that connects to the output layer
                mod_ids = [i for i in range(n_mods) if mods[i].layer == n_hid]

                # Reduce number of modules if one-to-one mapping and n_modules > n_outputs
                if self.one_to_one and n_rules > n_out:
                    n_rules = n_out
                    mod_ids = rng.choice(mod_ids, size=n_rules, replace=True).tolist()

                pts = [rng.integers(mods[i].n_outputs) for i in mod_ids]
            case 1:  
                if self.one_to_one:
                    # Only consider modules in the last layer
                    mod_set = [(mod, pt) for mod, pt in self.mod_out_set if mods[mod].layer == n_hid]
                    # Size is the minimum of the given n_rules, available modules ports, and available outputs
                    n_rules = min(n_rules, len(mod_set), len(self.output_set))

                    # Randomly sample `n_rules` unique module-output port pairs
                    idxs = rng.choice(len(mod_set), size=n_rules, replace=False).tolist()
                    mod_ids = [mod_set[i][0] for i in idxs]
                    pts = [mod_set[i][1] for i in idxs]
                else:
                    # Sample module-output port pairs with replacement
                    mod_ids = rng.choice([i for i, mod in enumerate(mods) if mod.layer == n_hid], size=n_rules, replace=True).tolist()
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

    def initialise_connect_mod_rules(self):
        """Add multiple module connection rules to the genome."""
        mods, n, rng = self.modules, self.n_modules, self.rng
        # Randomly sample output ports for all modules except the last
        src_pts = [rng.integers(mods[i].n_outputs) for i in range(n - 1)]
        # Randomly sample input ports for all modules except the first
        dst_pts = [rng.integers(mods[i].n_inputs) for i in range(1, n)]

        # Randomly sample weights from uniform distribution or use module shared weights
        if self.weight_sharing:
            src_type_ids = [mods[i].module_id for i in range(n - 1)]
            wts = np.asarray(self.shrd_mod_wts)[src_type_ids]
        else:
            wts = rng.uniform(0.1, 1.0, size=n - 1)

        # Add connections from each module to the next
        for i in range(n - 1):
            self.add_connect_rule((3, i + 1, i, src_pts[i], i + 2, i + 1, dst_pts[i], wts[i]))

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

        # --------------------------------------------- #
        # ------------ Module Connections ------------- #
        # --------------------------------------------- #

        mod_edges = []
        mod_g_mod, n_hid = self.mod_grouped_mod, self.n_hid_layers

        # Create edges for each layer
        for lyr in range(1, n_hid + 1):
            # Initialise arrays grouped by modules in the next layer
            lens = np.fromiter((len(mod_g_mod[lyr][dst]) for dst in range(n_mods)), count=n_mods, dtype=np.int32)
            indptr = np.concatenate(([0], np.cumsum(lens, dtype=np.int32)))
            n = indptr[-1]
            srcs, src_pts, dst_pts, wts = np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.float32)

            # Fill arrays
            for dst in range(n_mods):
                start, end = indptr[dst], indptr[dst + 1]
                if start == end:
                    continue
                entries = mod_g_mod[lyr][dst]
                srcs[start:end] = [t[3] for t in entries]
                src_pts[start:end] = [t[2] for t in entries]
                dst_pts[start:end] = [t[0] for t in entries]
                wts[start:end] = [t[4] for t in entries]

            # Create edges from arrays 
            deg = indptr[1:] - indptr[:-1]
            dst = np.repeat(np.arange(n_mods, dtype=np.int32), deg)
            mod_edges.append(Edges(indptr, srcs, src_pts, dst_pts, wts, dst))

        self.mod_edges = mod_edges

        # ---------------------------------------------------- #
        # ------------ Module Vector Connections ------------- #
        # ---------------------------------------------------- #

        mod_vec_edges = []
        mod_g_mod_v = self.mod_grouped_mod_vec

        # Create edges for each layer
        for lyr in range(1, n_hid + 1):
            # Initialise arrays grouped by modules in the next layer
            lens = np.fromiter((len(mod_g_mod_v[lyr][dst]) for dst in range(n_mods)), count=n_mods, dtype=np.int32)
            indptr = np.concatenate(([0], np.cumsum(lens, dtype=np.int32)))
            n = indptr[-1]
            src_vecs, wts = np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.float32)

            # Fill arrays
            for dst in range(n_mods):
                start, end = indptr[dst], indptr[dst + 1]
                if start == end:
                    continue
                entries = mod_g_mod_v[lyr][dst]
                src_vecs[start:end] = [t[1] for t in entries]
                wts[start:end] = [t[2] for t in entries]

            # Create edges from arrays
            deg = indptr[1:] - indptr[:-1]
            dst = np.repeat(np.arange(n_mods, dtype=np.int32), deg)
            mod_vec_edges.append(VectorEdges(indptr, src_vecs, wts, dst))

        self.mod_vec_edges = mod_vec_edges

    def initialise_genome(self):
        """Initialise the genome."""
        match self.connectivity:
            case 'UNCONNECTED': return
            case 'SPARSE':
                n = self.n_modules
                # Sparse connectivity: Connect each module with one input and one output
                self.initialise_connect_in_rules(0, n)
                self.initialise_connect_out_rules(0, n)

                if self.network_type == 1:
                    self.initialise_connect_mod_rules()

            case 'RANDOM':
                conn_d = self.connection_density
                # Random connectivity: Connect a random subset of inputs and outputs to random modules
                n_in, n_out = int(conn_d['input_density'] * self.n_inputs), int(conn_d['output_density'] * self.n_outputs)
                if n_in > 0:
                    self.initialise_connect_in_rules(1, n_in)
                if n_out > 0:
                    self.initialise_connect_out_rules(1, n_out)
                if self.network_type == 1:
                    self.initialise_connect_mod_rules()

            case 'FULLY CONNECTED':
                # Fully connected: Connect all inputs and outputs to all modules
                n_mods, rng, n_in, n_out = self.n_modules, self.rng, self.n_inputs, self.n_outputs
                shr_wts, shrd_mod_in, shrd_mod_out = self.weight_sharing, self.shrd_mod_in_wts, self.shrd_mod_out_wts
                # Module port and layer information
                layers_to_mods = {}
                in_counts = [m.n_inputs for m in n_mods]
                out_counts = [m.n_outputs for m in n_mods]
                layers = [m.layer for m in n_mods]
                for i, lyr in enumerate(layers):
                    layers_to_mods.setdefault(lyr, []).append(i)

                for i, m in enumerate(self.modules):
                    lyr = layers[i]

                    # Sample weights from uniform distribution or shared weights
                    if shr_wts:
                        in_wt = shrd_mod_in[m.module_id]
                        out_wt = shrd_mod_out[m.module_id]
                    else:
                        wts = rng.uniform(0.1, 1.0, 2)
                        in_wt, out_wt = wts[0], wts[1]

                    # Connect all network inputs
                    if lyr == 1:
                        for pt in range(in_counts[i]):
                            for src in range(n_in):
                                self.add_connect_rule((1, src, i, pt, in_wt))

                    # Connect all modules
                    else:
                        prev_ids = layers_to_mods[lyr - 1]
                        for pt in range(in_counts[i]):
                            for src in prev_ids:
                                for src_pt in range(out_counts[src]):
                                    self.add_connect_rule((3, lyr - 1, src, src_pt, lyr, i, pt, in_wt))

                    # Connect all network outputs
                    if lyr == self.n_hid_layers:
                        for pt in range(out_counts[i]):
                            for dst in range(n_out):
                                self.add_connect_rule((2, i, pt, dst, out_wt))
                    # Module connections to next module layer already handled by input logic

            case 'IDEAL':
                # Ideal: Connect each module and node ideally - for maze task prototyping only
                assert self.task == 'maze'
                for i in range(self.n_inputs):
                    mod = i // 2
                    self.add_connect_rule((1, i, mod, 0, 1.0))
                    self.add_connect_rule((1, i, mod, 1, 1.0))
                for i in range(self.n_outputs):
                    self.add_connect_rule((2, i, 0, i, 1.0))

    def forward_pass(self, input_vector):
        """Forward pass through the genome using execution plan.
        Arguments:
            input_vector (list): Input vector of size equal to the number of inputs.
        Returns:
            outputs.vals (array): Output array of size equal to the number of outputs.
        """
        n_mods = self.n_modules
        # Compile if the edges are outdated
        if self.compile_flag:
            self.compile_rules()
            self.compile_flag = 0

        # Forward pass through first module layer from inputs
        mod_outputs = self.layer_forward_pass(0, input_vector, n_mods)

        # Forward pass through hidden layers from previous hidden layers
        for i in range(2, self.n_hid_layers + 1):
            mod_outputs = self.layer_forward_pass(i, mod_outputs, n_mods)

        # Forward pass through output layer from last module layer
        outputs = self.layer_forward_pass(1, mod_outputs, self.n_outputs)
        return outputs.vals
    
    def layer_forward_pass(self, layer, in_vec, lyr_size):
        """Forward pass through a specific layer of the genome.
        Arguments:
            layer (int): The layer type (0 for 'MODULE LYR 1', 1 for 'OUTPUT', 2...N for remaining module layers).
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
                vec_edges, v_indptr = None, None
                inter_sizes = np.fromiter((m.n_inputs for m in mods), dtype=np.int32)
                out_sizes = np.fromiter((m.n_outputs for m in mods), dtype=np.int32)

            case 1: 
                edges = self.out_edges
                vec_edges, v_indptr = None, None
                inter_sizes = np.ones(lyr_size, dtype=np.int32)

            case _:
                mods = self.modules
                edges = self.mod_edges[layer-1]
                vec_edges, v_indptr = self.mod_vec_edges[layer-1], None
                inter_sizes = np.fromiter((m.n_inputs for m in mods), dtype=np.int32)
                out_sizes = np.fromiter((m.n_outputs for m in mods), dtype=np.int32)

        # Build intermediate pointers and values
        inter_ptr = self.build_ptr_from_sizes(inter_sizes)
        inter_vals = np.zeros(inter_ptr[-1], dtype=np.float32)

        # Compute contributions from input vector
        if layer == 0:
            input_vec = in_vec[edges.srcs]
        else:
            src_idx = in_vec.ptr[edges.srcs] + edges.src_pts
            input_vec = in_vec.vals[src_idx]

        # Review
        dst_idx = inter_ptr[edges.dst] + edges.dst_pts
        np.add.at(inter_vals, dst_idx, input_vec * edges.wts)

        # Early return for output layer
        if layer == 1:
            return Ragged(inter_vals, inter_ptr)

        # Prepare output pointers and values
        out_ptr = self.build_ptr_from_sizes(out_sizes)
        out_vals = np.zeros(out_ptr[-1], dtype=np.float32)

        # Compute contributions from vector edges
        if layer > 1 and vec_edges is not None and vec_edges.dst.size:
            v_indptr, v_srcs, v_wts = vec_edges.indptr, vec_edges.src_vecs, vec_edges.wts

            for i in np.nonzero(v_indptr[1:] - v_indptr[:-1])[0]:
                # Check for non-empty vector edges
                v_start, v_end = v_indptr[i], v_indptr[i + 1]
                if v_start == v_end:
                    continue

                # Compute destination indices
                v_dst_start, v_dst_end = inter_ptr[i], inter_ptr[i + 1]
                dst_len = v_dst_end - v_dst_start

                for j in range(v_start, v_end):
                    # Compute source indices
                    src = v_srcs[j]
                    src_start, src_end = in_vec.ptr[src], in_vec.ptr[src + 1]

                    # Compute contribution for the valid length of the edge
                    l = min(src_end - src_start, dst_len)
                    if l > 0:
                        inter_vals[v_dst_start:v_dst_start + l] += in_vec.vals[src_start:src_start + l] * v_wts[j]

        # Convert layer to the actual network module layer
        mod_layer = 1 if layer == 0 else layer
        modules = self.modules

        # Compute contributions from module layers
        for i in range(lyr_size):
            # Check the module belongs to this layer
            if mod_layer != modules[i].layer:
                continue

            # Check the module has scalar or vector edges
            start, end = inter_ptr[i], inter_ptr[i + 1]
            has_scalar = edges.indptr[i + 1] > edges.indptr[i]
            has_vector = (layer > 1 and vec_edges is not None and vec_edges.indptr[i + 1] > vec_edges.indptr[i])
            if not (has_scalar or has_vector):
                continue

            mod = modules[i]

            # Reset the input map
            if mod.tag_inputs and layer == 0:
                mod.input_map[:] = [-1] * len(mod.input_map)

            # Fill in the input map for scalar inputs
            # Only works under 1-to-1 assumption
            if mod.tag_inputs and has_scalar:
                t_start, t_end = edges.indptr[i], edges.indptr[i + 1]
                # Layer 0 - connections from inputs
                if layer == 0:
                    for j in range(t_start, t_end):
                        mod.input_map[edges.dst_pts[j]] = edges.srcs[j]
                # Layer 1 - connections from ports 
                # Currently this assumes a deep network where ports are associated with the same module
                elif layer > 1:
                    for j in range(t_start, t_end):
                        mod.input_map[edges.dst_pts[j]] = edges.src_pts[j] 

            # Fill in the input map for vector inputs
            # Only works under 1-to-1 assumption
            if mod.tag_inputs and has_vector and layer > 1:
                dst_len = inter_ptr[i + 1] - inter_ptr[i]
                for k in range(min(dst_len, mod.n_outputs)):
                    mod.input_map[k] = k

            # Fill in module forward pass outputs
            out_start, out_end = out_ptr[i], out_ptr[i + 1]
            out_vals[out_start:out_end] = mod.forward_pass(inter_vals[start:end])

        return Ragged(out_vals, out_ptr)

    # review
    # how to cut deep network
    def crossover(self, new_id, parent_2):
        """Perform crossover between two genomes to create a new child.
        Arguments:
            new_id (int): ID for the new child genome.
            parent_2 (Genome): The second parent genome for crossover.
        Returns:
            child (Genome): A new child genome resulting from the crossover.
        """
        child = GenomeDist(new_id, self.offspring_hyperparameters, self.module_bank)
        cut = int(self.rng.integers(1, min(self.n_modules, parent_2.n_modules)))
        child.modules = copy.deepcopy(self.modules[:cut]) + copy.deepcopy(parent_2.modules[cut:])
        child.reset()

        self.cut_rules(child, cut, self.conn_in_rules, parent_2.conn_in_rules, 1)
        self.cut_rules(child, cut, self.conn_out_rules, parent_2.conn_out_rules, 2)

        child.count_modules()
        child.recompute_module_distribution()

        return child

    # review
    # how to cut deep network
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
                    for rule in rules:
                        child.add_connect_rule(rule)
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
                    for rule in rules:
                        child.add_connect_rule(rule)

    def mutate(self):
        """Mutate the genome by randomly modifying its structure."""
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
            case 3: self.swap_module()
            case 4: self.modify_connection_weight()

    def add_connection(self):
        """Add a new connection from a genome input or to a genome output, or between modules."""
        rng, modules = self.rng, self.modules
        conn_type = rng.integers(1, 4)  # 1: CONNECT_IN, 2: CONNECT_OUT, 3: CONNECT_MOD

        match conn_type:
            case 1:
                if self.one_to_one:
                    # Get the set of available module-port pairs
                    if self.network_type == 0:
                        mod_set = list(self.mod_in_set - self.used_mod_in)
                    else:
                        mod_set = [(m, pt) for m, pt in list(self.mod_in_set - self.used_mod_in) if modules[m].layer == 1]

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
                    ids = [i for i in range(self.n_modules) if modules[i].layer == 1]
                    
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
                    if self.network_type == 0:
                        mod_set = list(self.mod_out_set - self.used_mod_out)
                    else:
                        mod_set = [(m, pt) for m, pt in list(self.mod_out_set - self.used_mod_out) if m.layer == self.n_hid_layers]

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
                    if self.network_type == 0:
                        ids = [i for i in range(self.n_modules)]
                    else:
                        ids = [i for i in range(self.n_modules) if modules[i].layer == self.n_hid_layers]

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
            case 3:
                # Randomly select a source and destination layer
                src_lyr = int(self.rng.integers(0, self.n_hid_layers))
                dst_lyr = src_lyr + 1

                if self.one_to_one:
                    # Get the set of available module-port pairs for source and destination module
                    src_mod_set = [(m, pt) for m, pt in list(self.mod_out_set-self.used_mod_out) if modules[m].layer == src_lyr]
                    dst_mod_set = [(m, pt) for m, pt in list(self.mod_in_set-self.used_mod_in) if modules[m].layer == dst_lyr]

                    # If there are no available module-port pairs, swap connections  
                    if not src_mod_set or not dst_mod_set:
                        self.swap_connection(conn_type=3, layer=src_lyr)
                        return

                    # Get a random module-port pair for source and destination
                    src_mod, src_pt = src_mod_set[rng.integers(len(src_mod_set))]
                    dst_mod, dst_pt = dst_mod_set[rng.integers(len(dst_mod_set))]
                else:
                    # Get the set of available modules for source and destination
                    src_mod_set = [m for m in self.modules if m.layer == src_lyr]
                    dst_mod_set = [m for m in self.modules if m.layer == dst_lyr]

                    # If there are no available modules, swap connections
                    if not src_mod_set or not dst_mod_set:
                        self.swap_connection(conn_type=3, layer=src_lyr)
                        return

                    # Get a random module-port pair for source and destination
                    src_mod = int(rng.choice(src_mod_set))
                    src_pt = int(rng.integers(0, src_mod.n_outputs))
                    dst_mod = int(rng.choice(dst_mod_set))
                    dst_pt = int(rng.integers(0, dst_mod.n_inputs))

                wt = self.shrd_mod_out_wts[self.modules[dst_mod].module_id] if self.weight_sharing else np.random.uniform(0.1, 1.0)
                rule = 3, src_lyr, src_mod, src_pt, dst_lyr, dst_mod, dst_pt, wt

        self.add_connect_rule(rule)

    def remove_connection(self):
        """Remove a connection from a genome input, to a genome output, or between modules."""
        rng = self.rng
        conn_type = rng.integers(1, 4)  # 1: CONNECT_IN, 2: CONNECT_OUT, 3: CONNECT_MOD

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
            # Module connections
            case 3:
                conn_mod = self.conn_mod_rules
                if not conn_mod:
                    return
                idx = rng.integers(len(conn_mod))
                rule = conn_mod.pop(idx)
        self.remove_connect_rule(rule)
        
    def swap_connection(self, conn_type=None, layer=None):
        """
        Swap two random input, module, or output connections.
        Arguments:
            conn_type: The type of connection to swap (1: CONNECT_IN, 2: CONNECT_OUT, 3: CONNECT_MOD).
            layer: The layer to swap connections in (for module connections).
        Updates:
            self.conn_in_rules (list): The rules for input connections.
            self.conn_out_rules (list): The rules for output connections.
            self.conn_mod_rules (list): The rules for module connections.
            self.mod_grouped_in (dict): Input connections grouped by destination module.
            self.out_grouped_out (dict): Output connections grouped by destination output.
            self.mod_grouped_mod (dict): Module connections grouped by destination layer and module.
            self._in_keys (list): The keys for each mod_grouped_in entry.
            self._out_keys (list): The keys for each out_grouped_out entry.
            self._mod_keys (list): The keys for each mod_grouped_mod entry.
            self.used_inputs (set): All used input indices.
            self.used_outputs (set): All used output indices.
            self.used_mod_in (set): All used input indice-port pairs for each module.
            self.used_mod_out (set): All used output indice-port pairs for each module.
            self.compile_flag (int): The flag indicating whether compilation is required.
        """
        rng = self.rng
        conn_type = conn_type if conn_type else rng.integers(1, 4)  # 1: CONNECT_IN, 2: CONNECT_OUT, 3: CONNECT_MOD
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
            case 3:
                # Obtain possible connections for a randomly selected layer
                mod_rules, modules = self.conn_mod_rules, self.modules
                src_lyr = layer if layer else int(rng.integers(0, self.n_hid_layers))
                conn_rules = [(idx, rule) for idx, rule in enumerate(mod_rules) if rule[1] == src_lyr]
                n = len(conn_rules)
                if n < 2:
                    return
                
                # Sample two random module connection rules
                (idx1, rule1), (idx2, rule2) = rng.choice(n, size=2, replace=False).tolist()
                (_, _, src_mod1, src_pt1, dst_lyr1, dst_mod1, dst_pt1, wt1), (_, _, src_mod2, src_pt2, dst_lyr2, dst_mod2, dst_pt2, wt2) = rule1, rule2

                # Only swap scalar connections
                if not modules[src_mod1].scalar_out or not modules[src_mod2].scalar_out:
                    return

                # Swap the connection srcs
                mod_rules[idx1], mod_rules[idx2] = (3, src_lyr, src_mod2, src_pt2, dst_lyr1, dst_mod1, dst_pt1, wt1), (3, src_lyr, src_mod1, src_pt1, dst_lyr2, dst_mod2, dst_pt2, wt2)

                # Update grouped dictionaries
                mod_g_mod, mod_keys = self.mod_grouped_mod, self._mod_keys
                mod_g_mod[src_lyr][dst_mod1].remove((dst_pt1, src_lyr, src_pt1, src_mod1, wt1))
                mod_keys[src_lyr][dst_mod1].discard((dst_pt1, src_lyr, src_pt1, src_mod1))
                mod_g_mod[src_lyr][dst_mod2].append((dst_pt2, src_lyr, src_pt1, src_mod1, wt1))
                mod_keys[src_lyr][dst_mod2].add((dst_pt2, src_lyr, src_pt1, src_mod1))
                mod_g_mod[src_lyr][dst_mod2].remove((dst_pt2, src_lyr, src_pt2, src_mod2, wt2))
                mod_keys[src_lyr][dst_mod2].discard((dst_pt2, src_lyr, src_pt2, src_mod2))
                mod_g_mod[src_lyr][dst_mod1].append((dst_pt1, src_lyr, src_pt2, src_mod2, wt2))
                mod_keys[src_lyr][dst_mod1].add((dst_pt1, src_lyr, src_pt2, src_mod2))

    def swap_module(self):
        """
        Swap one module type for another.
        Updates:
            self.modules (list): The module instances in the genome.
            self.mod_count (dict): The mapping of module types to their counts.
            self.compile_flag (int): The flag indicating whether compilation is required.
        """
        rng, mod_classes, mods = self.rng, self.mod_classes, self.modules
        # Select a random module to swap
        mod_idx = rng.integers(self.n_modules)
        old_mod = mods[mod_idx]
        old_mod_type, old_scalar, old_layer = old_mod.type, old_mod.scalar_out, old_mod.layer

        # Randomly select a new module type
        if old_layer == self.n_hid_layers:
            # Modules connecting to the output layer must have scalar outputs
            scalar_mods = self.scalar_mod_classes
            if len(scalar_mods) < 2:
                return
            new_mod_type = rng.choice([m for m in scalar_mods.keys() if m != old_mod_type])
        else:
            if len(mod_classes) < 2:
                return
            new_mod_type = rng.choice([m for m in mod_classes.keys() if m != old_mod_type])

        # Instantiate new module
        mods[mod_idx] = mod_classes[new_mod_type](layer=old_mod.layer)

        # Update module counts, distribution and module sets
        self.recompute_module_distribution()
        self.rebuild_module_sets()
        self.compile_flag = 1

        # Validate swap
        self.validate_swap(mod_idx, old_scalar)

    def validate_swap(self, mod_idx, old_scalar):
        """
        Validate the swap of a module and update connection rules accordingly.
        Updates:
            self.mod_grouped_mod (dict): Module connections grouped by destination layer and module.
            self.mod_grouped_mod_vec (dict): Vector module connections grouped by destination layer and module.
            self.used_inputs (set): All used input indices.
            self.used_outputs (set): All used output indices.
            self.used_mod_in (set): All used input indice-port pairs for each module.
            self.used_mod_out (set): All used output indice-port pairs for each module.
            self._mod_keys (list): The keys for each mod_grouped_mod entry.
            self._mod_vec_keys (list): The keys for each mod_grouped_mod_vec entry.
        """
        # Get the new module and its properties
        new_mod = self.modules[mod_idx]
        new_in, new_out, new_scalar = new_mod.n_inputs, new_mod.n_outputs, new_mod.scalar_out
        mod_g_mod, mod_g_mod_vec, mod_keys, mod_vec_keys = self.mod_grouped_mod, self.mod_grouped_mod_vec, self._mod_keys, self._mod_vec_keys
        used_mod_in, used_mod_out = self.used_mod_in, self.used_mod_out
        mod_rules = self.conn_mod_rules
        one_to_one = self.one_to_one

        match new_mod.layer:
            case 1:
                # Prune out of range connections
                rules_to_remove = [r for r in self.conn_in_rules if r[2] == mod_idx and r[3] >= new_in]
                for r in rules_to_remove:
                    self.remove_connect_rule(r)

                if not old_scalar or not new_scalar:
                    # No adjustments to be made if both modules have vector outputs
                    if not old_scalar and not new_scalar:
                        return
                    
                    rules_to_rewire = [r for r in mod_rules if r[2] == mod_idx]

                    # Remove rules to be rewired
                    for r in rules_to_rewire:
                        _, src_lyr, src, src_pt, dst_lyr, dst, dst_pt, wt = r
                        mod_rules.remove(r)

                        # Remove from grouped connections and key sets
                        if old_scalar:
                            mod_g_mod[dst_lyr][dst].remove((dst_pt, src_lyr, src_pt, src, wt))
                            mod_keys[dst_lyr][dst].discard((dst_pt, src_lyr, src_pt, src))
                        else:
                            mod_g_mod_vec[dst_lyr][dst].remove((src_lyr, src, wt))
                            mod_vec_keys[dst_lyr][dst].discard((src_lyr, src))

                        # Remove the rule from the used sets if one-to-one
                        if one_to_one:
                            used_mod_out.discard((src, src_pt))
                            used_mod_in.discard((dst, dst_pt))

                    # Re-add the rules with updated properties
                    for r in rules_to_rewire:
                        self.add_connect_rule(r)

                else:
                    # Prune out of range connections
                    rules_to_remove = [r for r in mod_rules if (r[2] == mod_idx and r[3] >= new_out)]
                    for r in rules_to_remove:
                        self.remove_connect_rule(r)

            case self.n_hid_layers:
                # Prune out of range dst connections
                rules_to_remove = [r for r in mod_rules if (r[5] == mod_idx and r[6] >= new_in)]
                for r in rules_to_remove:
                    self.remove_connect_rule(r)

                # Prune out of range src connections
                rules_to_remove = [r for r in self.conn_out_rules if r[1] == mod_idx and r[2] >= new_out]
                for r in rules_to_remove:
                    self.remove_connect_rule(r)
            
            case _:
                if not old_scalar or not new_scalar:
                    # No adjustments to be made if both modules have vector outputs
                    if not old_scalar and not new_scalar:
                        return
                    
                    rules_to_rewire = [r for r in mod_rules if r[2] == mod_idx]

                    # Remove rules to be rewired
                    for r in rules_to_rewire:
                        _, src_lyr, src, src_pt, dst_lyr, dst, dst_pt, wt = r
                        mod_rules.remove(r)

                        # Update grouped connections and key sets
                        if old_scalar:
                            mod_g_mod[dst_lyr][dst].remove((dst_pt, src_lyr, src_pt, src, wt))
                            mod_keys[dst_lyr][dst].discard((dst_pt, src_lyr, src_pt, src))
                        else:
                            mod_g_mod_vec[dst_lyr][dst].remove((src_lyr, src, wt))
                            mod_vec_keys[dst_lyr][dst].discard((src_lyr, src))

                        # Remove the rule from the used sets if one-to-one
                        if one_to_one:
                            used_mod_out.discard((src, src_pt))
                            used_mod_in.discard((dst, dst_pt))

                    # Re-add the rules with updated properties
                    for r in rules_to_rewire:
                        self.add_connect_rule(r)

                else:
                    # Prune out of range connections
                    rules_to_remove = [r for r in mod_rules if (r[2] == mod_idx and r[3] >= new_out) or (r[5] == mod_idx and r[6] >= new_in)]
                    for r in rules_to_remove:
                        self.remove_connect_rule(r)
                        
    def rebuild_module_sets(self):
        """
        Rebuild the sets of input and output connections for each module.
        Generates:
            self.mod_in_set (set): All input indice-port pairs.
            self.mod_out_set (set): All output indice-port pairs for each module.
        """
        n_mods, mods = self.n_modules, self.modules
        self.mod_in_set = set((i, j) for i in range(n_mods) for j in range(mods[i].n_inputs))
        self.mod_out_set = set((i, j) for i in range(n_mods) for j in range(mods[i].n_outputs))

    def count_modules(self):
        """
        Count the number of modules of each type.
        Updates:
            self.mod_count (dict): The mapping of module types to their counts.
        """
        counts = Counter(m.type for m in self.modules)
        self.mod_count = {nm: counts.get(nm, 0) for nm in self.mod_classes.keys()}

    def recompute_module_distribution(self, count=True):
        """
        Recompute the module distribution based on the current module counts.
        Arguments:
            count (bool): Whether to count modules before recomputing distribution.
        Updates:
            self.mod_dist (dict): The mapping of module types to their distribution.
        """
        if count:
            self.count_modules()
        # Compute ratio from count
        n_mods = self.n_modules
        self.mod_dist = {nm: c / n_mods for nm, c in self.mod_count.items()}

    def modify_connection_weight(self):
        """
        Modify the weight of a randomly selected connection.
        Updates:
            self.conn_in_rules (list): The rules for input connections.
            self.conn_out_rules (list): The rules for output connections.
            self.conn_mod_rules (list): The rules for module connections.
            self.mod_grouped_in (dict): Input connections grouped by destination module.
            self.out_grouped_out (dict): Output connections grouped by destination output.
            self.mod_grouped_mod (dict): Module connections grouped by destination layer and module.
            self.compile_flag (int): The flag indicating whether compilation is required.
        """
        rng = self.rng
        if self.weight_sharing:
            return

        conn_type = rng.integers(1, 4)  # 1: CONNECT_IN, 2: CONNECT_OUT, 3: CONNECT_MOD
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

            case 3:
                mod_rules = self.conn_mod_rules
                if not mod_rules:
                    return
                # Randomly select connection rule
                idx = rng.integers(len(mod_rules))
                tp, src_lyr, src_mod, src_pt, dst_lyr, dst_mod, dst_pt, wt = mod_rules[idx]

                # Only modify weights for scalar connections
                if not self.modules[src_mod].scalar_out:
                    return
                
                # Update the connection rule with the new weight
                mod_rules[idx] = (tp, src_lyr, src_mod, src_pt, dst_lyr, dst_mod, dst_pt, new_wt)
                self.mod_grouped_mod[dst_lyr][dst_mod].remove((dst_pt, src_lyr, src_pt, src_mod, wt))
                self.mod_grouped_mod[dst_lyr][dst_mod].append((dst_pt, src_lyr, src_pt, src_mod, new_wt))

    def clone(self, new_id):
        """Clone the genome with a new ID.
        Arguments:
            new_id (int): The ID for the cloned genome.
        Returns:
            child (Genome): A clone of the genome with the new ID.
        """
        # New genome instance 
        child = GenomeDist(new_id, self.offspring_hyperparameters, self.module_bank)

        # Clone module information
        child.mod_count = copy.deepcopy(self.mod_count)
        child.mod_dist = copy.deepcopy(self.mod_dist)
        child.modules = copy.deepcopy(self.modules)

        # Clone layer information
        child.n_hid_layers = self.n_hid_layers
        child.max_hid_layers = self.max_hid_layers

        # Clone module and connection rules
        child.conn_in_rules = copy.deepcopy(self.conn_in_rules)
        child.conn_out_rules = copy.deepcopy(self.conn_out_rules)
        child.conn_mod_rules = copy.deepcopy(self.conn_mod_rules)
        child.mod_grouped_in = copy.deepcopy(self.mod_grouped_in)
        child.out_grouped_out = copy.deepcopy(self.out_grouped_out)
        child.mod_grouped_mod = copy.deepcopy(self.mod_grouped_mod)
        child.mod_grouped_mod_vec = copy.deepcopy(self.mod_grouped_mod_vec)
        child._in_keys = copy.deepcopy(self._in_keys)
        child._out_keys = copy.deepcopy(self._out_keys)
        child._mod_keys = copy.deepcopy(self._mod_keys)
        child._mod_vec_keys = copy.deepcopy(self._mod_vec_keys)
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
        modules, n_hid_lyrs = self.modules, self.n_hid_layers
        lyr_count = [0 for _ in range(n_hid_lyrs)]
        for i in range(self.n_modules):
            name = f'M_{i}'
            graph.add_node(name, type='module')
            pos[name] = (-2 + 0.75 * modules[i].layer, lyr_count[modules[i].layer - 1])
            node_colors.append('lightgreen')
            # Update layer count for positioning
            lyr_count[modules[i].layer - 1] += 1

        # Output nodes
        for i in range(self.n_outputs):
            name = f'O_{i}'
            graph.add_node(name, type='output')
            pos[name] = (-1.25 + 0.75 * n_hid_lyrs, i)
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

        # Module connections
        for rule in self.conn_mod_rules:
            _, _, src_mod, src_pt, _, dst_mod, dst_pt, wt = rule
            src_nm = f"M_{src_mod}"
            dst_nm = f"M_{dst_mod}"
            graph.add_edge(src_nm, dst_nm, weight=wt)
            edge_label_groups[(src_nm, dst_nm)].append(f"P{src_pt} -> P{dst_pt}: {wt:.2f}")

        # Combine edge labels for multiple connections
        edge_labels = {k: " | ".join(v) for k, v in edge_label_groups.items()}

        # Plot the graph
        plt.figure(figsize=(16, 8))
        nx.draw(graph, pos, with_labels=True, node_color=node_colors, arrows=True, node_size=500, font_size=10, font_color='black', edge_color='gray')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red', font_size=8, horizontalalignment='center')
        max_y = max(self.n_modules, self.n_outputs, self.n_inputs)
        plt.ylim(-1, max_y + 1)
        plt.xlim(-2.5, 2.5)
        plt.axis('off')
        plt.show()
    