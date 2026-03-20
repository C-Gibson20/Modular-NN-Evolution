import random
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from enum import IntEnum
from multimodal_mazes.evolution.module_banks.test_modules import RecurrentModule

class RuleType(IntEnum):
    MODULE = 0
    CONNECT = 1

module_rule_dtype = np.dtype([
    ('type', np.int8),
    ('id', np.int32)
])

connect_rule_dtype = np.dtype([
    ('type', np.int8),
    ('src', np.int32),
    ('dst', np.int32),
    ('weight', np.float32)
])

class Genome():
    def __init__(self, genome_id, hyperparameters=None, weight_sharing=False):
        self.genome_id = genome_id
        self.fitness = 0.0
        
        self.hyperparameters = hyperparameters
        self.n_inputs = hyperparameters['n_inputs']
        self.n_outputs = hyperparameters['n_outputs']
        self.n_modules = hyperparameters['n_modules']
        self.weight_sharing = weight_sharing
        
        self.module = RecurrentModule()

        self.modules = [RecurrentModule() for _ in range(self.n_modules)]
        
        self.build_integer_id_blocks()

        self.build_module_maps()

        self.in_port_to_idx   = { pid:i for i,pid in enumerate(self.in_port_ids) }
        self.out_port_to_idx  = { pid:i for i,pid in enumerate(self.out_port_ids) }
        self.in_node_to_idx   = { nid:i for i,nid in enumerate(self.in_node_ids) }
        self.out_node_to_idx  = { nid:i for i,nid in enumerate(self.out_node_ids) }

        self.module_rule_list = []
        self.connect_rule_list = []
        self.module_rules = np.zeros(0, dtype=module_rule_dtype)
        self.connect_rules = np.zeros(0, dtype=connect_rule_dtype)
        self.rules_dirty = True
        
        self.plan_in_idx = np.zeros((self.n_modules, self.module.n_inputs), dtype=np.int32)
        self.plan_in_wt = np.zeros((self.n_modules, self.module.n_inputs), dtype=np.float32)
        self.plan_out_idx = np.zeros((self.n_modules, self.module.n_outputs), dtype=np.int32)
        self.plan_out_wt = np.zeros((self.n_modules, self.module.n_outputs), dtype=np.float32)
        self.execution_ready = False

        if self.weight_sharing:
            self.in_port_weights  = np.random.uniform(0.0, 1.0, size=self.module.n_inputs)
            self.out_port_weights = np.random.uniform(0.0, 1.0, size=self.module.n_outputs)
        
        self.initialise_genome()    

    def build_integer_id_blocks(self):
        """Build integer ID blocks for the genome."""
        idx = 0
        self.in_node_ids = np.arange(idx, idx + self.n_inputs, dtype=np.int32)
        idx += self.n_inputs

        self.module_ids = np.arange(idx, idx + self.n_modules, dtype=np.int32)
        idx += self.n_modules

        self.in_port_ids = np.arange(idx, idx + self.module.n_inputs * self.n_modules, dtype=np.int32)
        idx += self.module.n_inputs * self.n_modules

        self.out_port_ids = np.arange(idx, idx + self.module.n_outputs * self.n_modules, dtype=np.int32)
        idx += self.module.n_outputs * self.n_modules

        self.out_node_ids = np.arange(idx, idx + self.n_outputs, dtype=np.int32)
        idx += self.n_outputs

        self.total_nodes = idx

    def build_module_maps(self):
        """
        Build maps for module nodes and ports.
        port -> (module, port_index)
        ports are ordered by module-major then port-major orders
        """
        self.port_to_module_in = np.repeat(np.arange(self.n_modules), self.module.n_inputs)
        self.port_to_i_in = np.tile(np.arange(self.module.n_inputs), self.n_modules)

        self.port_to_module_out = np.repeat(np.arange(self.n_modules), self.module.n_outputs)
        self.port_to_i_out = np.tile(np.arange(self.module.n_outputs), self.n_modules)

    def add_module_rule(self, mod_id):
        """Add a module rule to the genome."""
        self.module_rule_list.append((RuleType.MODULE, mod_id))
        self.rules_dirty = True
    
    def add_connect_rule(self, src_id, dst_id, weight):
        """Add a connection rule to the genome."""
        self.connect_rule_list.append((RuleType.CONNECT, src_id, dst_id, weight))
        self.rules_dirty = True

    def compile_rules(self):
        """Compile rules into NumPy structured array."""
        if not self.rules_dirty:
            return 
        
        self.module_rules = np.array(self.module_rule_list, dtype=module_rule_dtype)
        self.connect_rules= np.array(self.connect_rule_list, dtype=connect_rule_dtype)
        self.rules_dirty = False

    def parse_rules(self):
        self.compile_rules()

        self.plan_in_idx.fill(0)
        self.plan_in_wt.fill(0.0)
        self.plan_out_idx.fill(0)
        self.plan_out_wt.fill(0.0)

        connect_rules = self.connect_rules
        srcs, dsts, weights = connect_rules['src'], connect_rules['dst'], connect_rules['weight']

        input_mask = np.isin(dsts, self.in_port_ids)
        output_mask = ~input_mask

        in_srcs, in_dsts, in_weights = srcs[input_mask], dsts[input_mask], weights[input_mask]
        in_port_idxs = np.searchsorted(self.in_port_ids, in_dsts)
        in_mod_idxs = self.port_to_module_in[in_port_idxs]
        port_in_idxs = self.port_to_i_in[in_port_idxs]

        self.plan_in_idx[in_mod_idxs, port_in_idxs] = in_srcs
        self.plan_in_wt[in_mod_idxs, port_in_idxs] = in_weights

        out_srcs, out_dsts, out_weights = srcs[output_mask], dsts[output_mask], weights[output_mask]
        out_port_idxs = np.searchsorted(self.out_port_ids, out_srcs)
        out_mod_idxs = self.port_to_module_out[out_port_idxs]
        port_out_idxs = self.port_to_i_out[out_port_idxs]

        self.plan_out_idx[out_mod_idxs, port_out_idxs] = out_dsts
        self.plan_out_wt[out_mod_idxs, port_out_idxs] = out_weights

        self.execution_ready = True

    def initialise_genome(self):
        """Initialise the genome with a set of rules."""
        for mod_id in self.module_ids:
            self.add_module_rule(mod_id)

        used_in = set()
        for idx, dst in enumerate(self.in_port_ids):
            src = np.random.choice([i for i in self.in_node_ids if i not in used_in])
            used_in.add(src)
            port_idx = idx % self.module.n_inputs
            weight = self.in_port_weights[port_idx] if self.weight_sharing else np.random.uniform(0.0, 1.0)
            self.add_connect_rule(src, dst, weight)

        used_out = set()
        for idx, src in enumerate(self.out_port_ids):
            dst = np.random.choice([i for i in self.out_node_ids if i not in used_out])
            used_out.add(dst)
            port_idx = idx % self.module.n_outputs
            weight = self.out_port_weights[port_idx] if self.weight_sharing else np.random.uniform(0.0, 1.0)
            self.add_connect_rule(src, dst, weight)

        # self.add_connect_rule(self.in_node_ids[0], self.in_port_ids[0], 1.0)
        # self.add_connect_rule(self.in_node_ids[1], self.in_port_ids[1], 1.0)
        # self.add_connect_rule(self.in_node_ids[2], self.in_port_ids[2], 1.0)
        # self.add_connect_rule(self.in_node_ids[3], self.in_port_ids[3], 1.0)
        # self.add_connect_rule(self.in_node_ids[4], self.in_port_ids[4], 1.0)
        # self.add_connect_rule(self.in_node_ids[5], self.in_port_ids[5], 1.0)
        # self.add_connect_rule(self.in_node_ids[6], self.in_port_ids[6], 1.0)
        # self.add_connect_rule(self.in_node_ids[7], self.in_port_ids[7], 1.0)
        # self.add_connect_rule(self.out_port_ids[0], self.out_node_ids[0], 1.0)
        # self.add_connect_rule(self.out_port_ids[1], self.out_node_ids[1], 1.0)
        # self.add_connect_rule(self.out_port_ids[2], self.out_node_ids[2], 1.0)
        # self.add_connect_rule(self.out_port_ids[3], self.out_node_ids[3], 1.0)


    def forward_pass(self, input_vector):
        """Forward pass through the genome using execution plan.
        Args:
            input_vector (list): Input vector of size equal to the number of inputs.
        Returns:
            list: Output vector of size equal to the number of outputs.
        """
        if self.rules_dirty or not self.execution_ready:
            self.parse_rules()

        values = np.zeros(self.total_nodes, dtype=np.float32)
        values[self.in_node_ids] = input_vector

        in_vals = values[self.plan_in_idx] * self.plan_in_wt
        mod_outputs = np.stack([self.module.forward_pass(row, i) for i, row in enumerate(in_vals)], axis=0)
        # mod_outputs = np.stack([self.modules[i].forward_pass(in_val) for i, in_val in enumerate(in_vals)], axis=0)

        # print(f"Module outputs: {mod_outputs}")
        # print(f"Module outputs test: {mod_outputs_test}")
        # print(f"in vals: {in_vals}")
        # print("")

        w_outputs = self.plan_out_wt * mod_outputs
        values[self.plan_out_idx] = w_outputs

        return values[self.out_node_ids]
    
    def crossover(self, new_id, parent_2):
        """Perform crossover between two grammar trees to create a new child.
        Args:
            new_id (int): ID for the new child grammar tree.
            parent_2 (Grammar_Tree): The second parent grammar tree for crossover.
        Returns:
            Grammar_Tree: A new child grammar tree resulting from the crossover.
        """
        child = Genome(new_id, self.hyperparameters, self.weight_sharing)
        child.module_rule_list = copy.deepcopy(self.module_rule_list)
        child.connect_rule_list = []
        
        used_in, used_out, used_in_port, used_out_port = set(), set(), set(), set()

        candidate_rules = self.connect_rule_list + parent_2.connect_rule_list
        np.random.shuffle(candidate_rules)

        input_rules = [rule for rule in candidate_rules if rule[1] in self.in_node_ids]
        for rule in input_rules:
            _, src, dst, weight = rule
            if src in used_in or dst in used_in_port:
                continue

            if child.weight_sharing:
                idx = child.in_port_to_idx[dst] % child.module.n_inputs
                weight = child.in_port_weights[idx]
            
            child.add_connect_rule(src, dst, weight)
            used_in.add(src)
            used_in_port.add(dst)

        output_rules = [rule for rule in candidate_rules if rule[2] in self.out_node_ids]
        for rule in output_rules:
            _, src, dst, weight = rule
            if dst in used_out or src in used_out_port:
                continue

            if child.weight_sharing:
                idx = child.out_port_to_idx[src] % child.module.n_outputs
                weight = child.out_port_weights[idx]
            
            child.add_connect_rule(src, dst, weight)
            used_out.add(dst)
            used_out_port.add(src)

        available_inputs = list(set(self.in_node_ids) - used_in)
        available_in_ports = list(set(self.in_port_ids) - used_in_port)
        
        while available_inputs and available_in_ports:
            src = available_inputs.pop()
            dst = available_in_ports.pop()

            if child.weight_sharing:
                idx = child.in_port_to_idx[dst] % child.module.n_inputs
                weight = child.in_port_weights[idx]
            else:
                weight = np.random.uniform(0.0, 1.0)

            child.add_connect_rule(src, dst, weight)

        available_outputs = list(set(self.out_node_ids) - used_out)
        available_out_ports = list(set(self.out_port_ids) - used_out_port)

        while available_outputs and available_out_ports:
            src = available_out_ports.pop()
            dst = available_outputs.pop()

            if child.weight_sharing:
                idx = child.out_port_to_idx[src] % child.module.n_outputs
                weight = child.out_port_weights[idx]
            else:
                weight = np.random.uniform(0.0, 1.0)

            child.add_connect_rule(src, dst, weight)

        child.rules_dirty = True
        return child
    
    def mutate(self):
        """Mutate the grammar tree by randomly modifying its structure."""
        mut = np.random.randint(0, 3)
        
        match mut:
            case 0: self.swap_input()
            case 1: self.swap_output()
            case 2: self.modify_weight()
    
    def swap_input(self):
        """Swap the input connections of two randomly selected modules."""
        rule1, rule2 = random.sample([rule for rule in self.connect_rule_list if (rule[1] in self.in_node_ids and rule[2] in self.in_port_ids)], 2)
        
        i1, i2 = self.connect_rule_list.index(rule1), self.connect_rule_list.index(rule2)
        src1, dst1, w1 = rule1[1:]
        src2, dst2, w2 = rule2[1:]

        self.connect_rule_list[i1] = (RuleType.CONNECT, src2, dst1, w1)
        self.connect_rule_list[i2] = (RuleType.CONNECT, src1, dst2, w2)
        
        self.rules_dirty = True
    
    def swap_output(self):
        """Swap the output connections of two randomly selected modules."""
        rule1, rule2 = random.sample([rule for rule in self.connect_rule_list if (rule[1] in self.out_port_ids and rule[2] in self.out_node_ids)], 2)

        i1, i2 = self.connect_rule_list.index(rule1), self.connect_rule_list.index(rule2)
        src1, dst1, w1 = rule1[1:]
        src2, dst2, w2 = rule2[1:]

        self.connect_rule_list[i1] = (RuleType.CONNECT, src1, dst2, w1)
        self.connect_rule_list[i2] = (RuleType.CONNECT, src2, dst1, w2)
        
        self.rules_dirty = True
        
    def modify_weight(self):
        """Modify the weight of a randomly selected connection."""
        port = int(np.random.choice(np.concatenate([self.in_port_ids, self.out_port_ids])))
        rule = random.choice([rule for rule in self.connect_rule_list if rule[1] == port or rule[2] == port])
        new_weight = np.clip(rule[3] + np.random.normal(0, 0.1), 0.0, 1.0)

        if not self.weight_sharing:
            idx = self.connect_rule_list.index(rule)
            self.connect_rule_list[idx] = (rule[0], rule[1], rule[2], new_weight)
        else:
            if port in self.in_port_ids:
                self.in_port_weights[self.in_port_to_idx[port]] = new_weight
            elif port in self.out_port_ids:
                self.out_port_weights[self.out_port_to_idx[port]] = new_weight
            
            for rule in self.connect_rule_list: 
                if rule[1] == port or rule[2] == port:
                    idx = self.connect_rule_list.index(rule)
                    self.connect_rule_list[idx] = (rule[0], rule[1], rule[2], new_weight)

        self.rules_dirty = True
        
    def clone(self, new_id):
        """Clone the grammar tree with a new ID.
        Args:
            new_id (int): The ID for the cloned grammar tree.
        Returns:
            Grammar_Tree: A new instance of Grammar_Tree with the same structure but a different ID.
        """ 
        child = Genome(new_id, self.hyperparameters, self.weight_sharing)
        
        if self.weight_sharing:
            child.in_port_weights = copy.deepcopy(self.in_port_weights)
            child.out_port_weights = copy.deepcopy(self.out_port_weights)
        
        child.module_rule_list = copy.deepcopy(self.module_rule_list)
        child.connect_rule_list = copy.deepcopy(self.connect_rule_list)
        child.rules_dirty = True
        
        return child
    
    def plot_genome(self):
        """Visualize the grammar tree as a directed graph."""
        if self.rules_dirty or not self.execution_ready:
            self.parse_rules()

        graph = nx.DiGraph()

        pos = {}
        node_colors = []
        y_gap = 1000
        node_size = 2000
        
        input_height = self.n_inputs * (y_gap + node_size)
        output_height = self.n_outputs * (y_gap + node_size)
        module_height = self.n_modules * (y_gap + node_size)
        module_offset = (input_height - module_height) / 6
        output_offset = (input_height - output_height) / 6

        for i in range(self.n_inputs):
            name = f'INPUT_{i}'
            graph.add_node(name, type='input')
            pos[name] = (-2, i * y_gap)
            node_colors.append('lightblue')
        for i in range(self.n_modules):
            name = f'MODULE_{i}'
            graph.add_node(name, type='module')
            pos[name] = (0, i * y_gap + module_offset)
            node_colors.append('lightgreen')
        for i in range(self.n_outputs):
            name = f'OUTPUT_{i}'
            graph.add_node(name, type='output')
            pos[name] = (2, i * y_gap + output_offset)
            node_colors.append('lightcoral')

        edge_labels = {}
            
        for rule in self.connect_rules:
            _, src_id, dst_id, weight = rule
            
            if src_id in self.in_node_to_idx:
                src_name = f"INPUT_{self.in_node_to_idx[src_id]}"
            else:
                idx = self.out_port_to_idx[src_id]
                mod_i = self.port_to_module_out[idx]
                src_name = f"MODULE_{mod_i}"

            if dst_id in self.out_node_to_idx:
                dst_name = f"OUTPUT_{self.out_node_to_idx[dst_id]}"
            else:
                idx = self.in_port_to_idx[dst_id]
                mod_i = self.port_to_module_in[idx]
                dst_name = f"MODULE_{mod_i}"
            
            graph.add_edge(src_name, dst_name, weight=weight)
            edge_labels[(src_name, dst_name)] = f"{weight:.2f}"
    
        plt.figure(figsize=(6, 8))
        nx.draw(graph, pos, with_labels=True, node_color=node_colors, arrows=True, node_size=node_size, font_size=10, font_color='black', edge_color='gray')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red', font_size=8)

        plt.title(f"Genome {self.genome_id}")
        plt.axis('off')
        plt.show()
    
    