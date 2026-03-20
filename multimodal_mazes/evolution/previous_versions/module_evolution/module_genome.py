import random
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import bisect
from enum import IntEnum
from multimodal_mazes.evolution.module_evolution.node import Node

class RuleType(IntEnum):
    NODE_IN = 0
    NODE_HIDDEN = 1
    NODE_OUT = 2
    CONNECT_IN = 3
    CONNECT_HIDDEN = 4
    CONNECT_OUT = 5

# Node tuples
# (
#     ('rule_type', np.int32), # 0: in_node, 1: hidden_node, 2: out_node
#     ('node_id', np.int32),   # node ID
#     ('hidden_layer', np.int32), # layer ID -1: input, 0-n: hidden, -1: output
#     ('activation', np.str_),   # activation function: 'linear', 'relu'
# )

# Input connection tuples
# (
    # ('rule_type', np.int32), # 3: connect_in
    # ('src', np.int32),       # source node ID
    # ('src_layer', np.int32), # source node layer: input, hidden, output
    # ('dst', np.int32),       # destination node ID: input node ID
    # ('weight', np.float32)   # weight of the connection
# )

# Hidden node connection tuples
# (
    # ('rule_type', np.int32),    # 4: connect_hidden
    # ('src', np.int32),          # source node ID
    # ('src_layer', np.int32),    # source node layer: input, hidden, output
    # ('dst', np.int32),          # destination node ID: hidden node ID
    # ('hidden_layer', np.int32), # hidden layer ID
    # ('weight', np.float32)      # weight of the connection
# )

# Output connection tuples
# (
    # ('rule_type', np.int32), # 5: connect_out
    # ('src', np.int32),       # source node ID
    # ('src_layer', np.int32), # source node layer: input, hidden, output
    # ('dst', np.int32),       # destination node ID: output node ID
    # ('weight', np.float32)   # weight of the connection
# )

############################################################################################################
########## IMPORTANT: REVIEW HIDDEN LAYER FORWARD PASS AND COMPILATION - ModuleGenome IS UNTESTED ##########
############################################################################################################

class ModuleGenome():
    def __init__(self, module_id, hyperparameters, dimensions=(2, 1)):
        """
        Initialise the ModuleGenome with a unique ID, hyperparameters, and dimensions.
        Arguments:
            module_id (int): The unique identifier for the module.
            hyperparameters (dict): The hyperparameters for the module.
            dimensions (tuple): The input and output dimensions of the module.
        Properties:
            fitness (float): The fitness score of the module.
            n_layers (int): The number of layers in the module (input, hidden, output).
            n_inputs (int): The number of input nodes in the module.
            n_outputs (int): The number of output nodes in the module.
            n_hidden_curr (int): The number of hidden nodes in the module.
            n_hidden_total (int): The total number of hidden nodes in the module.
            hidden_ids (list): The IDs of the hidden nodes in the module.
            hidden_id_indptr_map (dict): A mapping from hidden node IDs to their index in the hidden layer.
            n_hidden_layers (int): The number of hidden layers in the module.
            node_rules (list): The rules governing the nodes in the module.
            nodes_in (list): The input nodes in the module.
            nodes_hid (dict): The hidden nodes in the module.
            nodes_out (list): The output nodes in the module.
            conn_in_rules (list): The input connections in the module.
            conn_hid_rules (list): The hidden connections in the module.
            conn_out_rules (list): The output connections in the module.
            grouped_in (dict): The input connections grouped by input node.
            grouped_hid (dict): The hidden connections grouped by hidden layer and sorted by hidden node.
            grouped_out (dict): The output connections grouped by output node.
            compile_flag (list): Flags indicating whether each layer is compiled.
        """
        self.module_id = module_id
        self.fitness = 0.0
        
        self.hyperparameters = hyperparameters
        self.crossover_hyperparameters = dict(hyperparameters)
        self.n_layers = 3  # INPUT, HIDDEN, OUTPUT
        self.n_inputs = dimensions[0]
        self.n_outputs = dimensions[1]
        self.n_hidden_curr = 0
        self.n_hidden_total = 0
        self.hidden_ids = []
        self.hidden_id_indptr_map = {}
        self.n_hidden_layers = 1
        
        self.node_rules = []
        self.nodes_in = [Node('linear') for _ in range(self.n_inputs)]
        self.nodes_hid = {}
        self.nodes_out = [Node('linear') for _ in range(self.n_outputs)]
        self.conn_in_rules = []
        self.conn_hid_rules = []
        self.conn_out_rules = []
        
        self.grouped_in = {i: [] for i in range(self.n_inputs)} # connections to input nodes grouped by input node
        self.grouped_hid = {} # connections to hidden nodes grouped by hidden_layer and sorted by hidden node
        self.grouped_out = {i: [] for i in range(self.n_outputs)} # connections to output nodes grouped by output node

        self.compile_flag = [1, 1, 1]  # [node, input, output]

        self.initialise_genome()

    def add_node_rule(self, node_rule):
        """
        Add a node rule to the genome.
        Arguments:
            node_rule (tuple): A tuple representing the node rule.
        """
        self.node_rules.append(node_rule)
        if node_rule[0] != 1:
            return
        self.nodes_hid[node_rule[1]] = Node(node_rule[3], node_rule[2])
        self.grouped_hid[node_rule[2]] = []
        self.n_hidden_curr += 1
        self.n_hidden_total += 1
        self.hidden_ids.append(node_rule[1])
        
    def remove_node_rule(self, node_rule):
        """
        Remove a node rule from the genome.
        Arguments:
            node_rule (tuple): A tuple representing the node rule.
        """
        if node_rule[0] != 1:
            return
        self.node_rules.remove(node_rule)
        self.n_hidden_curr -= 1
        idx = self.hidden_ids.index(node_rule[1])
        self.hidden_ids.pop(idx)
        del self.nodes_hid[node_rule[1], node_rule[2]]

    def add_node_rules(self, node_rules):
        """Add multiple node rules to the genome.
        Arguments:
            node_rules (list): List of node rules to add.
        """
        for rule in node_rules:
            self.add_node_rule(rule)

    def add_connect_rule(self, conn_rule):
        """Add a connection rule to the genome.
        Arguments:
            conn_rule (tuple): A tuple representing the connection rule.
        """
        match conn_rule[0]:
            case 3:
                _, src, src_lyr, dst, wt = conn_rule
                if any(t[0] == src_lyr and t[1] == src for t in self.grouped_in[dst]):
                    return
                self.grouped_in[dst].append((src_lyr, src, wt))
                self.conn_in_rules.append(conn_rule)
                self.compile_flag[0] = 1
            case 4:
                _, src, src_lyr, dst, hid_lyr, wt = conn_rule
                if any(t[0] == dst and t[1] == src_lyr and t[2] == src for t in self.grouped_hid[hid_lyr]):
                    return
                bisect.insort(self.grouped_hid[hid_lyr], (dst, src_lyr, src, wt))
                self.conn_hid_rules.append(conn_rule)
                self.compile_flag[1] = 1
            case 5:
                _, src, src_lyr, dst, wt = conn_rule
                if any(t[0] == src_lyr and t[1] == src for t in self.grouped_out[dst]):
                    return
                self.grouped_out[dst].append((src_lyr, src, wt))
                self.conn_out_rules.append(conn_rule)
                self.compile_flag[2] = 1

    def remove_connect_rule(self, conn_rule):
        """
        Remove a connection rule from the genome.
        Arguments:
            conn_rule (tuple): A tuple representing the connection rule.
        """
        match conn_rule[0]:
            case 3:  
                _, src, src_lyr, dst, wt = conn_rule
                self.grouped_in[dst].remove((src_lyr, src, wt))
                self.compile_flag[0] = 1
            case 4:
                _, src, src_lyr, dst, hid_lyr, wt = conn_rule  
                self.grouped_hid[hid_lyr].remove((dst, src_lyr, src, wt))
                self.compile_flag[1] = 1
            case 5:  
                _, src, src_lyr, dst, wt = conn_rule
                self.grouped_out[dst].remove((src_lyr, src, wt))
                self.compile_flag[2] = 1

    def add_connect_rules(self, rules):
        """Add multiple connection rules to the genome.
        Arguments:
            rules (list): List of connection rules to add.
        """
        for rule in rules:
            self.add_connect_rule(rule)

    def compile_rules(self):
        """Compile network rules into a ragged array for evaluation."""
        if self.compile_flag[0] == 1:
            self.compile_conn_in_rules()
        if self.compile_flag[1] == 1:
            self.compile_conn_hidden_rules()
        if self.compile_flag[2] == 1:
            self.compile_conn_out_rules()
        self.compile_flag = [0, 0, 0]

    def compile_conn_in_rules(self): 
        """Compile connection rules with destination at input layer into a ragged array."""
        self.in_srcs, self.in_lyrs, self.in_wts = np.empty(len(self.conn_in_rules), dtype=np.int32), np.empty(len(self.conn_in_rules), dtype=np.int32), np.empty(len(self.conn_in_rules), dtype=np.float32)
        self.in_indptr = np.zeros(self.n_inputs + 1, dtype=np.int32)
        idx = 0

        for i in range(self.n_inputs):
            for src_lyr, src, wt in self.grouped_in[i]:
                self.in_srcs[idx], self.in_lyrs[idx], self.in_wts[idx] = src, src_lyr, wt
                idx += 1
            self.in_indptr[i + 1] = idx

    # Review
    def compile_conn_hidden_rules(self):
        """Compile connection rules with destination at hidden layer into a ragged array."""
        self.hidden_id_indptr_map = {}
        self.hid_srcs, self.hid_lyrs, self.hid_wts = np.empty(len(self.conn_hid_rules), dtype=np.int32), np.empty(len(self.conn_hid_rules), dtype=np.int32), np.empty(len(self.conn_hid_rules), dtype=np.float32)
        self.hid_indptr = np.zeros(self.n_hidden_layers, dtype=np.int32)
        idx = 0

        for i in range(self.n_hidden_layers):
            if i not in self.grouped_hid:
                continue
            for _, src_lyr, src, wt in self.grouped_hid[i]:
                self.hid_srcs[idx], self.hid_lyrs[idx], self.hid_wts[idx] = src, src_lyr, wt
                idx += 1
            self.hid_indptr[i + 1] = idx

    def compile_conn_out_rules(self):
        """Compile connection rules with destination at output layer into a ragged array."""
        self.out_srcs, self.out_lyrs, self.out_wts = np.empty(len(self.conn_out_rules), dtype=np.int32), np.empty(len(self.conn_out_rules), dtype=np.int32), np.empty(len(self.conn_out_rules), dtype=np.float32)
        self.out_indptr = np.zeros(self.n_outputs + 1, dtype=np.int32)
        idx = 0

        for out_id in range(self.n_outputs):
            for src_lyr, src, wt in self.grouped_out[out_id]:
                self.out_srcs[idx], self.out_lyrs[idx], self.out_wts[idx] = src, src_lyr, wt
                idx += 1
            self.out_indptr[out_id + 1] = idx

    def initialise_genome(self):
        """Initialise the genome."""
        rules = [(0, i, -1, 'linear') for i in range(self.n_inputs)]
        self.add_node_rules(rules)
        rules = [(2, i, -1, 'linear') for i in range(self.n_outputs)]
        self.add_node_rules(rules)

    # Review
    # Currently this does not use the compiled rules
    # Todo: revise to use compiled rules
    def forward_pass(self, in_vec):
        """Forward pass through the genome using execution plan.
        Args:
            input_vector (list): Input vector of size equal to the number of inputs.
        Returns:
            outputs (list): Output vector of size equal to the number of outputs.
        """
        if sum(self.compile_flag) > 0:
            self.compile_rules()

        prev_in = np.array([node.previous_output for node in self.nodes_in])
        prev_hid = np.array([self.nodes_hid[i].previous_output for i in self.hidden_ids])
        prev_out = np.array([node.previous_output for node in self.nodes_out])

        # Input layer
        # Connections from input_vector
        # Connections from input, hidden, and output layer from previous time step
        # rules[1] = src_id, # rules[2] = src_layer, 0: input, 1: hidden, 2: output, # rules[3] = dst_id
        in_rec_rules = [rule for rule in self.conn_in_rules if rule[2] == 0]
        for rule in in_rec_rules:
            in_vec[rule[3]] += prev_in[rule[1]] * rule[4]

        hid_rec_rules = [rule for rule in self.conn_in_rules if rule[2] == 1]
        for rule in hid_rec_rules:
            in_vec[rule[3]] += prev_hid[self.hidden_id_indptr_map[rule[1]]] * rule[4]

        out_rec_rules = [rule for rule in self.conn_in_rules if rule[2] == 2]
        for rule in out_rec_rules:
            in_vec[rule[3]] += prev_out[rule[1]] * rule[4]

        in_outputs = np.zeros(self.n_inputs, dtype=np.float32)
        for i, node in enumerate(self.nodes_in):
            in_outputs[i] = node.forward_pass(in_vec[i])

        # Hidden layer
        # Connections from input layer and shallower deeper hidden layers
        # Connections from deeper hidden layers and output layer from previous time step
        
        hid_in_vec = np.zeros(self.n_hidden_curr, dtype=np.float32)

        in_rules = [rule for rule in self.conn_hid_rules if rule[2] == 0]
        for rule in in_rules:
            hid_in_vec[self.hidden_id_indptr_map[rule[3]]] += in_outputs[rule[1]] * rule[4]

        out_rec_rules = [rule for rule in self.conn_hid_rules if rule[2] == 2]
        for rule in out_rec_rules:
            hid_in_vec[self.hidden_id_indptr_map[rule[3]]] += prev_out[self.hidden_id_indptr_map[rule[1]]] * rule[4]

        lyr_outputs = np.zeros((self.n_hidden_layers, self.n_hidden_curr), dtype=np.float32)
        for lyr in range(self.n_hidden_layers):
            lyr_in_vec = np.zeros(self.n_hidden_curr, dtype=np.float32)

            # Hidden layers deeper or as deep as current hidden layer
            hid_rec_rules = [rule for rule in self.conn_hid_rules if rule[2] == 1 and rule[4] >= lyr]
            for rule in hid_rec_rules:
                lyr_in_vec[self.hidden_id_indptr_map[rule[3]]] += prev_hid[self.hidden_id_indptr_map[rule[1]]] * rule[4]

            if lyr == 0:
                continue

            hid_fwd_rules = [rule for rule in self.conn_hid_rules if rule[2] == 1 and rule[4] < lyr]
            for rule in hid_fwd_rules:
                lyr_in_vec[self.hidden_id_indptr_map[rule[3]]] += lyr_outputs[lyr - 1][self.hidden_id_indptr_map[rule[1]]] * rule[4]

            lyr_in_vec += hid_in_vec

            for i, node in enumerate(self.nodes_hid.values()):
                if node.layer != lyr:
                    continue
                lyr_outputs[lyr][i] = node.forward_pass(lyr_in_vec[i])

        hid_outputs = np.zeros(self.n_hidden_curr, dtype=np.float32)
        for lyr in lyr_outputs:
            hid_outputs += lyr

        # Output layer
        # Connections from input and hidden layer
        # Connections from output layer from previous time step
        out_in_vec = np.zeros(self.n_outputs, dtype=np.float32)

        in_rules = [rule for rule in self.conn_out_rules if rule[2] == 0]
        for rule in in_rules:
            out_in_vec[rule[3]] += in_outputs[rule[1]] * rule[4]
        
        hid_rules = [rule for rule in self.conn_out_rules if rule[2] == 1]
        for rule in hid_rules:
            out_in_vec[rule[3]] += hid_outputs[self.hidden_id_indptr_map[rule[1]]] * rule[4]

        out_rec_rules = [rule for rule in self.conn_out_rules if rule[2] == 2]
        for rule in out_rec_rules:
            out_in_vec[rule[3]] += prev_out[rule[1]] * rule[4]

        outputs = np.zeros(self.n_outputs, dtype=np.float32)
        for i, node in enumerate(self.nodes_out):
            outputs[i] = node.forward_pass(out_in_vec[i])

        return outputs

    def crossover(self, new_id, parent_2):
        """Perform crossover between two genomes to create a new child.
        Args:
            new_id (int): ID for the new child genome.
            parent_2 (Genome): The second parent genome for crossover.
        Returns:
            child (Genome): A new child genome resulting from the crossover.
        """
        dimensions = (max(self.n_inputs, parent_2.n_inputs), max(self.n_outputs, parent_2.n_outputs))
        child = ModuleGenome(new_id, self.crossover_hyperparameters , dimensions)
        child.n_hidden_curr = 0
        child.n_hidden_total = 0
        child.hidden_ids = []

        cut_in = 1 if self.n_inputs == 1 and parent_2.n_inputs == 1 else np.random.randint(1, min(self.n_inputs, parent_2.n_inputs))
        cut_out = 1 if self.n_outputs == 1 and parent_2.n_outputs == 1 else np.random.randint(1, min(self.n_outputs, parent_2.n_outputs))

        self.cut_rules(child, cut_in, self.node_rules, parent_2.node_rules, self.conn_in_rules, parent_2.conn_in_rules, 3)
        self.cut_rules(child, cut_out, self.node_rules, parent_2.node_rules, self.conn_out_rules, parent_2.conn_out_rules, 5)
        if self.n_hidden_curr > 1 and parent_2.n_hidden_curr > 1:
            cut_hid = np.random.randint(1, min(self.n_hidden_curr, parent_2.n_hidden_curr))
            self.cut_rules(child, cut_hid, self.node_rules, parent_2.node_rules, self.conn_hid_rules, parent_2.conn_hid_rules, 4)
        child.reset()

        return child

    def cut_rules(self, child, cut, node_rule_list1, node_rule_list2, conn_rule_list1, conn_rule_list2, rule_type):
        """Cut rules at a specified index.
        Args:
            cut (int): The index to cut the rules at.
            node_rule_list1 (list): The list of node rules to cut from parent 1.
            node_rule_list2 (list): The list of node rules to cut from parent 2.
            conn_rule_list1 (list): The list of connection rules to cut from parent 1.
            conn_rule_list2 (list): The list of connection rules to cut from parent 2.
            rule_type (RuleType): The type of rules being cut.
            
        """
        match rule_type:
            case 3: 
                conn_rules = [rule for rule in conn_rule_list1 if rule[3] < cut] + [rule for rule in conn_rule_list2 if rule[3] > cut]
                child.add_connect_rules(conn_rules)
            case 4: 
                node_rules = [rule for rule in node_rule_list1 if rule[0] == 1 and rule[1] < cut] + [rule for rule in node_rule_list2 if rule[0] == 1 and rule[1] > cut]
                nodes = [rule[1] for rule in node_rules]
                child.nodes_hid = {node_id: Node('relu') for node_id in nodes}
                child.add_node_rules(node_rules)
                conn_rules = [rule for rule in conn_rule_list1 if rule[3] < cut] + [rule for rule in conn_rule_list2 if rule[3] > cut]
                child.add_connect_rules(conn_rules)
            case 5: 
                conn_rules = [rule for rule in conn_rule_list1 if rule[3] < cut] + [rule for rule in conn_rule_list2 if rule[3] > cut]
                child.add_connect_rules(conn_rules)

    def mutate(self):
        """Mutate the genome by randomly modifying its structure."""
        if not self.conn_in_rules or not self.conn_out_rules:
            mut = 0
        else:
            mut = np.random.randint(0, 3)

        match mut:
            case 0: self.add_connection()
            case 1: self.remove_connection()
            case 2: self.modify_connection_weight()
            case 3: self.add_hidden_node_wide()
            case 4: self.add_hidden_node_deep()

    def add_connection(self):
        """Add a new connection from a genome input or to a genome output."""
        conn_type = np.random.choice([3, 4, 5]) if self.hidden_ids else np.random.choice([3, 5])
        lyr = np.random.choice([0, 1, 2]) if self.hidden_ids else np.random.choice([0, 2]) # src_lyr
        wt = np.random.uniform(0.1, 1.0)

        match conn_type:
            # Destination 
            case 3: dst = np.random.randint(0, self.n_inputs)
            case 4: dst = np.random.choice(self.hidden_ids) 
            case 5: dst = np.random.randint(0, self.n_outputs)

        match lyr:
            # Construct new rule tuple
            case 0: rule = (conn_type, np.random.randint(0, self.n_inputs), lyr, dst, wt) 
            case 1: rule = (conn_type, np.random.choice(self.hidden_ids), lyr, dst, self.nodes_hid[dst].layer, wt)
            case 2: rule = (conn_type, np.random.randint(0, self.n_outputs), lyr, dst, wt)
        
        self.add_connect_rule(rule)

    def remove_connection(self):
        """
        Remove a connection from the genome.
        Updates:
            conn_in_rules (list): The input connection rules
            conn_hid_rules (list): The hidden connection rules
            conn_out_rules (list): The output connection rules
            compile_flag (list): The compilation flags for each connection type
        """
        # Select connection type
        conn_type = np.random.randint(3, 6) # 3: CONNECT_IN, 4: CONNECT_HIDDEN, 5: CONNECT_OUT
        self.compile_flag[conn_type - 3] = 1

        match conn_type:
            case 3:
                if not self.conn_in_rules:
                    return
                # Remove input connection
                rule = self.conn_in_rules.pop(random.randrange(len(self.conn_in_rules)))
            case 4:
                if not self.conn_hid_rules:
                    return
                # Remove hidden connection
                rule = self.conn_hid_rules.pop(random.randrange(len(self.conn_hid_rules)))
            case 5:
                if not self.conn_out_rules:
                    return
                # Remove output connection
                rule = self.conn_out_rules.pop(random.randrange(len(self.conn_out_rules)))
                
        self.remove_connect_rule(rule)

    def modify_connection_weight(self):
        """
        Modify the weight of a randomly selected connection.
        Updates:
            conn_in_rules (list): The input connection rules
            conn_hid_rules (list): The hidden connection rules
            conn_out_rules (list): The output connection rules
            grouped_in (dict): The grouped input connections
            grouped_hid (dict): The grouped hidden connections
            grouped_out (dict): The grouped output connections
        """
        # Select connection type and new weight
        conn_type = np.random.randint(3, 6)  # 3: CONNECT_IN, 4: CONNECT_HIDDEN, 5: CONNECT_OUT
        new_wt = np.clip(wt + np.random.normal(0, 0.1), 0.0, 1.0)
        self.compile_flag[conn_type - 3] = 1

        match conn_type:
            case 3: 
                if not self.conn_in_rules:
                    return
                # Connection properties
                idx = random.randrange(len(self.conn_in_rules))
                tp, src, src_lyr, dst, wt = self.conn_in_rules[idx]
                
                # Update connections
                self.conn_in_rules[idx] = (tp, src, src_lyr, dst, new_wt)
                self.grouped_in[dst].remove((src_lyr, src, wt))
                self.grouped_in[dst].append((src_lyr, src, new_wt))

            case 4:
                if not self.conn_hid_rules:
                    return
                # Connection properties
                idx = random.randrange(len(self.conn_hid_rules))
                tp, src, src_lyr, dst, hid_lyr, wt = self.conn_hid_rules[idx]
                
                # Update connections
                self.conn_hid_rules[idx] = (tp, src, src_lyr, dst, hid_lyr, new_wt)
                self.grouped_hid[hid_lyr].remove((dst, src_lyr, src, wt))
                bisect.insort(self.grouped_hid[hid_lyr], (dst, src_lyr, src, new_wt))

            case 5:
                if not self.conn_out_rules:
                    return
                # Connection properties
                idx = random.randrange(len(self.conn_out_rules))
                tp, src, src_lyr, dst, wt = self.conn_out_rules[idx]
                
                # Update connections
                self.conn_out_rules[idx] = (tp, src, src_lyr, dst, new_wt)
                self.grouped_out[dst].remove((src_lyr, src, wt))
                self.grouped_out[dst].append((src_lyr, src, new_wt))

    def add_hidden_node_wide(self):
        """
        Add a wide hidden node to the genome.
        Generates:
            node (tuple): The new hidden node
            conn_rule_in (tuple): The connection rule for the new node input
            conn_rule_out (tuple): The connection rule for the new node output
        """
        # Create new node
        hid_id = self.n_hidden_total
        hid_lyr = np.random.randint(0, self.n_hidden_layers)
        self.add_node_rule((1, hid_id, hid_lyr, 'relu'))

        # Add connection rule from the previous layer to the node
        if hid_lyr == 0:
            # Input layer to hidden layer
            self.add_connect_rule((3, np.random.randint(0, self.n_inputs), hid_id, hid_lyr, np.random.uniform(0.1, 1.0)))
        else:
            # Hidden layer to hidden layer
            src = np.random.choice([rule[3] for rule in self.conn_hid_rules if rule[4] == hid_lyr-1])
            self.add_connect_rule((4, src, hid_lyr-1, hid_id, hid_lyr, np.random.uniform(0.1, 1.0)))

        # Add connection rule from the node to the next layer
        if hid_lyr == self.n_hidden_layers - 1:
            # Hidden layer to output layer
            self.add_connect_rule((5, hid_id, hid_lyr, np.random.randint(0, self.n_outputs), np.random.uniform(0.1, 1.0)))
        else:
            # Hidden layer to hidden layer
            dst = np.random.choice([rule[3] for rule in self.conn_hid_rules if rule[4] == hid_lyr+1])
            self.add_connect_rule((4, hid_id, hid_lyr, dst, hid_lyr + 1, np.random.uniform(0.1, 1.0)))

    def add_hidden_node_deep(self):
        """
        Add a deep hidden node to the genome.
        Updates:
            n_hidden_layers (int): The number of hidden layers
            conn_hid_rules (list): The hidden connection rules
            conn_out_rules (list): The output connection rules
        """
        # Requires a connection from hidden to output to split
        if not self.conn_hid_rules or not self.conn_out_rules:
            return

        # Choose a connection to split
        hid_rule_ids = [rule[3] for rule in self.conn_hid_rules]
        out_rule_ids = [rule[1] for rule in self.conn_out_rules]
        ids = [id for id in self.hidden_ids if id in hid_rule_ids and id in out_rule_ids]
        if not ids:
            return
        split = np.random.choice(ids)

        # Create new node         
        hid_id = self.n_hidden_total
        self.n_hidden_layers += 1
        self.add_node_rule((1, hid_id, self.n_hidden_layers - 1, 'relu'))

        # Split connection from previous hidden layer to new hidden node
        rule = np.random.choice([rule for rule in self.conn_hid_rules if rule[3] == split])
        _, src, src_lyr, _, wt = rule
        self.conn_hid_rules.pop(self.conn_hid_rules.index(rule))
        self.remove_connect_rule(rule)
        self.add_connect_rule((4, src, src_lyr, hid_id, wt))

        # Split connection from new hidden node to output
        rule = np.random.choice([rule for rule in self.conn_out_rules if rule[1] == split])
        _, _, src_lyr, dst, wt = rule
        self.conn_out_rules.pop(self.conn_out_rules.index(rule))
        self.remove_connect_rule(rule)
        self.add_connect_rule((5, hid_id, src_lyr, dst, wt))

    def clone(self, new_id):
        """Clone the genome with a new ID.
        Args:
            new_id (int): The ID for the cloned genome.
        Returns:
            child (Genome): A new instance of Genome with the same structure but a different ID.
        """ 
        child = ModuleGenome(new_id, self.hyperparameters)
        child.n_hidden_curr = self.n_hidden_curr
        child.n_hidden_total = self.n_hidden_total
        child.hidden_ids = copy.deepcopy(self.hidden_ids)
        child.hidden_id_indptr_map = copy.deepcopy(self.hidden_id_indptr_map)
        child.n_hidden_layers = self.n_hidden_layers
        child.node_rules = copy.deepcopy(self.node_rules)
        child.nodes_in = copy.deepcopy(self.nodes_in)
        child.nodes_hid = copy.deepcopy(self.nodes_hid)
        child.nodes_out = copy.deepcopy(self.nodes_out)
        child.add_connect_rules(self.conn_in_rules)
        child.add_connect_rules(self.conn_hid_rules)
        child.add_connect_rules(self.conn_out_rules)
        child.reset()
        return child
    
    def reset(self):
        """
        Reset the genome's state.
        Resets:
            Node states - input, hidden, output
            Compile flag - Flags indicating whether each layer is compiled
        """
        for node in self.nodes_in:
            node.reset()
        for node in self.nodes_hid.values():
            node.reset()
        for node in self.nodes_out:
            node.reset()
        
        self.compile_flag = [1, 1, 1]

    def plot_genome(self):
        """Visualize the genome as a directed graph."""
        graph = nx.DiGraph()

        pos, node_colors = {}, []
        
        # Add input nodes
        for i in range(self.n_inputs):
            name = f'INPUT_{i}'
            graph.add_node(name, type='input')
            pos[name] = (-2, i)
            node_colors.append('lightblue')

        # Add hidden nodes
        for i, id in enumerate(self.hidden_ids):
            name = f'HIDDEN_{id}'
            graph.add_node(name, type='hidden')
            pos[name] = (self.nodes_hid[id].layer * 2, i)
            node_colors.append('lightgreen')

        # Add output nodes
        for i in range(self.n_outputs):
            name = f'OUTPUT_{i}'
            graph.add_node(name, type='output')
            pos[name] = (self.n_hidden_layers * 2, i)
            node_colors.append('lightcoral')

        edge_labels, layer_names = {}, {0: 'INPUT', 1: 'HIDDEN', 2: 'OUTPUT'}

        # Add input connections
        for rule in self.conn_in_rules:
            _, src_id, src_lyr, dst_id, wt = rule
            src_nm = f"{layer_names[src_lyr]}_{src_id}"
            dst_nm = f"INPUT_{dst_id}"
            graph.add_edge(src_nm, dst_nm, weight=wt)
            edge_labels[(src_nm, dst_nm)] = f"{wt:.2f}"

        # Add hidden connections
        for rule in self.conn_hid_rules:
            _, src_id, src_lyr, dst_id, wt = rule
            src_nm = f"{layer_names[src_lyr]}_{src_id}"
            dst_nm = f"HIDDEN_{dst_id}"
            graph.add_edge(src_nm, dst_nm, weight=wt)
            edge_labels[(src_nm, dst_nm)] = f"{wt:.2f}"

        # Add output connections
        for rule in self.conn_out_rules:
            _, src_id, src_lyr, dst_id, wt = rule
            src_nm = f"{layer_names[src_lyr]}_{src_id}"
            dst_nm = f"OUTPUT_{dst_id}"
            graph.add_edge(src_nm, dst_nm, weight=wt)
            edge_labels[(src_nm, dst_nm)] = f"{wt:.2f}"

        plt.figure(figsize=(4, 4))
        nx.draw(graph, pos, with_labels=True, node_color=node_colors, arrows=True, node_size=2000, font_size=10, font_color='black', edge_color='gray')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red', font_size=8, horizontalalignment='left')
        plt.title(f"Module {self.module_id}")
        plt.ylim(-1, max(len(self.hidden_ids), self.n_outputs, self.n_inputs) + 1)
        plt.xlim(-2.5, self.n_hidden_layers * 2 + 0.5)
        plt.axis('off')
        plt.show()
    