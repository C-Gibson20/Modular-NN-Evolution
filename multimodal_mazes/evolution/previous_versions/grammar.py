import random
import numpy as np
import copy
import networkx as nx
import matplotlib.pyplot as plt
from multimodal_mazes.evolution.module_banks.test_modules import RecurrentModule

class Grammar_Tree():
    def __init__(self, grammar_id, inputs=8, outputs=4, hyperparameters=None, weight_sharing=False):
        self.grammar_id = grammar_id
        self.inputs = inputs
        self.outputs = outputs
        self.hyperparameters = hyperparameters

        self.symbolic_rules = []

        self.fitness = 0.0

        self.weight_sharing = weight_sharing
        
        self.module_function = hyperparameters['module_function']
        self.module = RecurrentModule()
        self.initialise_genome()    
    
    def initialise_genome(self):
        """Initialise the genome with a set of symbolic rules."""
        used_inputs = set()
        used_outputs = set()

        if self.weight_sharing:
            # self.input_1_W = random.uniform(-1.0, 1.0)
            # self.input_2_W = random.uniform(-1.0, 1.0)
            # self.output_W = random.uniform(-1.0, 1.0)
            self.input_1_W = random.uniform(0.0, 1.0)
            self.input_2_W = random.uniform(0.0, 1.0)
            self.output_W = random.uniform(0.0, 1.0)

        for mod_id in range(self.hyperparameters['num_modules']):
            self.symbolic_rules.append({'type': 'MODULE', 'name': f"MODULE_{mod_id}"})

            for mod_input in range(2):
                input_node = self.get_unused_node(used_inputs, self.inputs)
                if self.weight_sharing:
                    self.symbolic_rules.append({'type': 'CONNECT', 'src': f"INPUT_{input_node}", 'dst': f"MODULE_{mod_id}.in{mod_input}", 'weight': self.input_1_W if mod_input == 0 else self.input_2_W})
                else:
                    self.symbolic_rules.append({'type': 'CONNECT', 'src': f"INPUT_{input_node}", 'dst': f"MODULE_{mod_id}.in{mod_input}", 'weight': random.uniform(-1.0, 1.0)})
                
            output_node = self.get_unused_node(used_outputs, self.outputs)
            if self.weight_sharing:
                self.symbolic_rules.append({'type': 'CONNECT', 'src': f"MODULE_{mod_id}", 'dst': f"OUTPUT_{output_node}", 'weight': self.output_W})
            else:
                self.symbolic_rules.append({'type': 'CONNECT', 'src': f"MODULE_{mod_id}", 'dst': f"OUTPUT_{output_node}", 'weight': random.uniform(-1.0, 1.0)})
    
    def parse_rules(self):
        """Parse the symbolic rules to create connections and modules."""
        self.connections = {}
        self.modules = {}

        for rule in self.symbolic_rules:
            if rule['type'] == 'MODULE':
                self.modules[rule['name']] = {'inputs': [], 'outputs': []}
            
            elif rule['type'] == 'CONNECT':
                src_node = rule['src']
                dst_node = rule['dst']
                weight = rule['weight']

                self.connections[(src_node, dst_node)] = {'src': src_node, 'dst': dst_node, 'weight': weight}
                
                if 'MODULE' in dst_node:
                    mod_name, port = dst_node.split('.')
                    self.modules[mod_name]['inputs'].append((src_node, port))
                elif 'MODULE' in src_node:
                    mod_name = src_node
                    self.modules[mod_name]['outputs'].append((dst_node, 'out')) 

    def generate_nn(self):
        """Generate the neural network structure from the symbolic rules."""
        self.parse_rules()
        self.execution_order = []
        self.node_values = {}

        for i in range(self.inputs):
            self.node_values[f"INPUT_{i}"] = None

        for mod_key, mod in self.modules.items():
            in_nodes = [inp[0] for inp in mod['inputs']]
            out_nodes = [out[0] for out in mod['outputs']]
            self.execution_order.append((mod_key, in_nodes, out_nodes))
    
    def forward_pass(self, input_vector):
        """Forward pass through the grammar tree.
        Args:
            input_vector (list): Input vector of size equal to the number of inputs.
        Returns:
            list: Output vector of size equal to the number of outputs.
        """
        self.generate_nn()
        
        for i in range(self.inputs):
            self.node_values[f"INPUT_{i}"] = input_vector[i]

        for mod_key, in_nodes, out_nodes in self.execution_order:
            in1 = self.node_values[in_nodes[0]]
            in2 = self.node_values[in_nodes[1]] 

            input_ports = [port for (_, port) in self.modules[mod_key]['inputs']]
            w1 = self.connections[(in_nodes[0], f"{mod_key}.{input_ports[0]}")]['weight']
            w2 = self.connections[(in_nodes[1], f"{mod_key}.{input_ports[1]}")]['weight']
            
            mod_input1 = in1 * w1
            mod_input2 = in2 * w2
            
            out_val = self.module.forward_pass(np.array([mod_input1, mod_input2]))
            
            self.node_values[mod_key] = out_val

            out_node = out_nodes[0]    
            w3 = self.connections[(mod_key, out_node)]['weight']
            self.node_values[out_node] = out_val * w3

        output_vector = np.array([self.node_values[f"OUTPUT_{i}"] for i in range(self.outputs)])
        return output_vector
    
    def crossover(self, new_id, second_parent):
        """Perform crossover between two grammar trees to create a new child.
        Args:
            new_id (int): ID for the new child grammar tree.
            second_parent (Grammar_Tree): The second parent grammar tree for crossover.
        Returns:
            Grammar_Tree: A new child grammar tree resulting from the crossover.
        """
        child = self.clone(new_id)
        child.symbolic_rules = [rule for rule in self.symbolic_rules if rule['type'] == 'MODULE']

        used_inputs = set()
        used_outputs = set()
        used_module_input_ports = set()
        used_module_output_ports = set()

        parent_1_rules = [rule for rule in self.symbolic_rules if rule['type'] == 'CONNECT']
        parent_2_rules = [rule for rule in second_parent.symbolic_rules if rule['type'] == 'CONNECT']

        candidates = parent_1_rules + parent_2_rules
        random.shuffle(candidates)

        for rule in candidates:
            if 'INPUT' in rule['src']:
                input_node = int(rule['src'].split('_')[1])
                module_port = rule['dst']
                if input_node not in used_inputs and module_port not in used_module_input_ports:
                    child.symbolic_rules.append(rule)
                    if self.weight_sharing:
                        weight = self.input_1_W if 'in0' in module_port else self.input_2_W
                        child.symbolic_rules[-1]['weight'] = weight
                    used_inputs.add(input_node)
                    used_module_input_ports.add(module_port)
            elif 'OUTPUT' in rule['dst']:
                output_node = int(rule['dst'].split('_')[1])
                module_port = rule['src']
                if output_node not in used_outputs and module_port not in used_module_output_ports:
                    child.symbolic_rules.append(rule)
                    if self.weight_sharing:
                        child.symbolic_rules[-1]['weight'] = self.output_W
                    used_outputs.add(output_node)
                    used_module_output_ports.add(module_port)
        
        while len(used_inputs) < self.inputs:
            input_node = self.get_unused_node(used_inputs, self.inputs)
            module_port = self.get_unused_port(used_module_input_ports, 'in')
            if self.weight_sharing:
                weight = self.input_1_W if 'in0' in module_port else self.input_2_W
                child.symbolic_rules.append({'type': 'CONNECT', 'src': f"INPUT_{input_node}", 'dst': module_port, 'weight': weight})
            else:
                child.symbolic_rules.append({'type': 'CONNECT', 'src': f"INPUT_{input_node}", 'dst': module_port, 'weight': random.uniform(-1.0, 1.0)})
            used_inputs.add(input_node)
            used_module_input_ports.add(module_port)
                    
        while len(used_outputs) < self.outputs:
            output_node = self.get_unused_node(used_outputs, self.outputs)
            module_port = self.get_unused_port(used_module_output_ports, 'out')
            if self.weight_sharing:
                child.symbolic_rules.append({'type': 'CONNECT', 'src': module_port, 'dst': f"OUTPUT_{output_node}", 'weight': self.output_W})
            else:
                child.symbolic_rules.append({'type': 'CONNECT', 'src': module_port, 'dst': f"OUTPUT_{output_node}", 'weight': random.uniform(-1.0, 1.0)})
            used_outputs.add(output_node)
            used_module_output_ports.add(module_port)

        child.parse_rules()

        return child
    
    def mutate(self):
        """Mutate the grammar tree by randomly modifying its structure."""
        mutation_type = random.choice(["rewire_input", "rewire_output", "modify_weight"])
        if mutation_type == "rewire_input":
            self.swap_input()
        elif mutation_type == "rewire_output":
            # self.swap_output()
            self.swap_input()
        elif mutation_type == "modify_weight":
            self.modify_weight()
    
    def swap_input(self):
        """Swap the input connections of two randomly selected modules."""
        port = random.choice(['in0', 'in1'])
        rules = np.random.choice([rule for rule in self.symbolic_rules if rule["type"] == "CONNECT" and ("INPUT" in rule["src"] and port in rule["dst"])], 2, replace=False)
        rule1, rule2 = rules[0], rules[1]
        rule1["src"], rule2["src"] = rule2["src"], rule1["src"]
    
    def swap_output(self):
        """Swap the output connections of two randomly selected modules."""
        rules = np.random.choice([rule for rule in self.symbolic_rules if rule["type"] == "CONNECT" and "OUTPUT" in rule["dst"]], 2, replace=False)
        rule1, rule2 = rules[0], rules[1]
        rule1["dst"], rule2["dst"] = rule2["dst"], rule1["dst"]
        
    def modify_weight(self):
        """Modify the weight of a randomly selected connection."""
        port = random.choice(['in0', 'in1', 'OUTPUT'])
        rule = random.choice([rule for rule in self.symbolic_rules if rule['type'] == 'CONNECT' and (port in rule['dst'] or port in rule['src'])])
        # new_weight = np.clip(rule["weight"] + np.random.normal(0, 0.1), -1.0, 1.0)
        new_weight = np.clip(rule["weight"] + np.random.normal(0, 0.1), 0.0, 1.0)
            
        # rule = random.choice([rule for rule in self.symbolic_rules if rule["type"] == "CONNECT"])
        # rule["weight"] += np.random.normal(0, 0.1)
        # rule["weight"] = np.clip(rule["weight"], -1.0, 1.0)

        if self.weight_sharing:
            if port == 'in0':
                self.input_1_W = new_weight
            elif port == 'in1':
                self.input_2_W = new_weight
            elif port == 'OUTPUT':
                return
                self.output_W = new_weight
            for sim_rule in self.symbolic_rules:
                if sim_rule['type'] == 'CONNECT' and (port in sim_rule['src'] or port in sim_rule['dst']):
                    sim_rule['weight'] = new_weight
        else:
            rule["weight"] = new_weight
        
    def clone(self, new_id):
        """Clone the grammar tree with a new ID.
        Args:
            new_id (int): The ID for the cloned grammar tree.
        Returns:
            Grammar_Tree: A new instance of Grammar_Tree with the same structure but a different ID.
        """ 
        clone = copy.deepcopy(self)
        clone.grammar_id = new_id
        return clone
    
    def plot_genome(self):
        """Visualize the grammar tree as a directed graph."""
        self.parse_rules()
        graph = nx.DiGraph()

        input_nodes = [f"INPUT_{i}" for i in range(self.inputs)]
        output_nodes = [f"OUTPUT_{i}" for i in range(self.outputs)]
        module_nodes = list(self.modules.keys())

        for node in input_nodes:
            graph.add_node(node, type='input')
        for node in module_nodes:
            graph.add_node(node, type='module')
        for node in output_nodes:
            graph.add_node(node, type='output')
            
        for (src, dst), conn in self.connections.items():
            src = src.split('.')[0]  
            dst = dst.split('.')[0]
            graph.add_edge(src, dst, weight=conn['weight'])

        pos = {}
        y_gap = 1000
        node_size = 2000
        
        input_height = self.inputs * (y_gap + node_size)
        output_height = self.outputs * (y_gap + node_size)
        module_height = len(self.modules) * (y_gap + node_size)
        module_offset = (input_height - module_height) / 6
        output_offset = (input_height - output_height) / 6

        for i in range(self.inputs):
            pos[f"INPUT_{i}"] = (-2, i * y_gap)
        for i, mod in enumerate(self.modules.keys()):
            pos[mod] = (0, i * y_gap + module_offset)
        for i in range(self.outputs):
            pos[f"OUTPUT_{i}"] = (2, i * y_gap + output_offset)

        node_colors = []
        for node in graph.nodes:
            if 'INPUT' in node:
                node_colors.append('lightblue')
            elif 'OUTPUT' in node:
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightcoral')

        plt.figure(figsize=(6, 8))
        nx.draw(graph, pos, with_labels=True, node_color=node_colors, arrows=True, node_size=node_size, font_size=10, font_color='black', edge_color='gray')

        edge_labels = {(src, dst): f"{data['weight']:.2f}" for src, dst, data in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red', font_size=8)

        plt.title(f"Grammar Tree {self.grammar_id}")
        plt.axis('off')
        plt.show()
    
    def get_unused_node(self, node_set, node_max):
        """Get a random unused node from the set of nodes.
        Args:
            node_set (set): Set of currently used nodes.
            node_max (int): Maximum number of nodes.
        Returns:
            int: A random unused node.
        """
        unused_nodes = [i for i in range(node_max) if i not in node_set]
        node = random.choice(unused_nodes)
        node_set.add(node)
        return node
    
    def get_unused_port(self, port_set, port_type):
        """Get a random unused port from the set of ports.
        Args:
            port_set (set): Set of currently used ports.
            port_type (str): Type of port ('in' or 'out').
        Returns:
            str: A random unused port.
        """
        possible_ports = set()
        for i in range(self.hyperparameters['num_modules']):
            if port_type == 'in':
                possible_ports.add(f"MODULE_{i}.in0")
                possible_ports.add(f"MODULE_{i}.in1")
            else:
                possible_ports.add(f"MODULE_{i}")
        
        unused_ports = possible_ports - port_set
        port = random.choice(list(unused_ports))
        port_set.add(port)
        return port
