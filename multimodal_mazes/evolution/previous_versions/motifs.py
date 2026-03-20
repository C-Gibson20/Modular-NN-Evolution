import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from networkx.algorithms.isomorphism import DiGraphMatcher, categorical_node_match

class Motif:
    def __init__(self, motif_type, id, n_inputs, n_outputs, complexity, recursive=False, temporal=False):
        """
        Initialise the motif.
        Arguments:
            motif_type (str): The type of the motif.
            id (int): The unique identifier for the motif.
            n_inputs (int): The number of input connections.
            n_outputs (int): The number of output connections.
            complexity (int): The complexity of the motif.
            recursive (bool): Whether the motif is recursive.
            temporal (bool): Whether the motif is temporal.
        Properties:
            previous_values (array): Stores the previous output values of the motif for recurrent and temporal connections.
        """
        self.motif_type = motif_type
        self.id = id
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.recursive = recursive
        self.temporal = temporal
        self.previous_values = None  
        self.complexity = complexity 

    def reset(self):
        """
        Reset the motif's state.
        Resets:
            previous_values (array): Stores the previous output values of the motif for recurrent and temporal connections.
        """
        self.previous_values = None

class MotifBank:
    def __init__(self, motif_types):
        """
        Initialise the motif bank.
        Arguments:
            motif_types (list): The types of motifs to include in the bank.
        Properties:
            motif_generator (MotifGenerator): The generator used to create motifs.
            motif_bank (dict): The bank of generated motifs.
            simplest_motifs (list): The simplest motifs in the bank.
            motifs (list): All generated motifs.
        """
        self.motif_types = motif_types
        self.motif_generator = MotifGenerator(motif_types)
        self.motif_bank, self.simplest_motifs, self.motifs = self.motif_generator.generate_motifs()

    def forward_pass(self, motif, inputs):
        """
        Perform a forward pass through the specified motif.
        Arguments:
            motif (Motif): The motif to process.
            inputs (array): The input values for the motif.
        Returns:
            outputs (array): The output values from the motif.
        """

        _, matrix = self.motif_bank[motif.motif_type][motif.id]
        values = [0, 0, 0]
        n_hidden = 3 - motif.n_inputs - motif.n_outputs

        idx = 0

        for i in range(motif.n_inputs):
            values[i] = inputs[i]
            for j in range(0, 3):
                if motif.previous_values:
                    values[i] += motif.previous_values[j] * matrix[j][i]

        # print(values)

        idx += motif.n_inputs

        for i in range(idx, idx + n_hidden):
            for j in range(0, idx):
                values[i] += values[j] * matrix[j][i]
            for j in range(i, 3):
                if motif.previous_values:
                    values[i] += motif.previous_values[j] * matrix[j][i]

        # print(values)

        idx += n_hidden

        for i in range(idx, 3):
            for j in range(0, idx):
                values[i] += values[j] * matrix[j][i]
            for j in range(idx, 3):
                if motif.previous_values:
                    values[i] += motif.previous_values[j] * matrix[j][i] 

        motif.previous_values = values

        # print(values)

        outputs = np.array(values[-motif.n_outputs:])
        
        return outputs
    
    def plot_motif_bank(self, type=None):
        """
        Plot the motifs in the bank.
        Arguments:
            type (str): The type of motifs to plot.
        """
        types = self.motif_bank.keys() if type is None else [type]

        for motif_type in types:
            for motif_id, _ in self.motif_bank[motif_type].items():
                self.plot_motif(motif_type, motif_id)

    def plot_motif(self, motif_type, motif_id):
        """
        Plot a specific motif from the bank.
        Arguments:
            motif_type (str): The type of the motif.
            motif_id (int): The ID of the motif.
        """
        graph = self.motif_bank[motif_type][motif_id][0]
        pos = {0: (-1, 0), 1: (0, 1), 2: (1, 0)}
        colours = {'I': 'lightblue', 'H': 'lightgreen', 'O': 'lightcoral'}
        node_colours = [colours[graph.nodes[i]['label']] for i in graph.nodes]
        plt.figure(figsize=(3, 3))
        nx.draw(graph, pos, with_labels=True, node_color=node_colours, node_size=800, font_size=10, arrowsize=20)
        plt.axis('off')
        plt.show()

class MotifGenerator:
    def __init__(self, types_to_generate):
        """
        Initialise the motif generator.
        Arguments:
            types_to_generate (list): The types of motifs to generate.
        Properties:
            types (list): The types of motifs to generate.
            type_properties (dict): Properties for each motif type.
            all_matrices (ndarray): All possible adjacency matrices for 3 nodes.
        """
        self.types = types_to_generate
        self.type_properties = {
            "I2_O1":   {"labels": ["I", "I", "O"], "recursive": False, "temporal": False},
            "I2_O1.R": {"labels": ["I", "I", "O"], "recursive": True, "temporal": False},
            "I2_O1.T": {"labels": ["I", "I", "O"], "recursive": False, "temporal": True},
            "I2_O1.RT": {"labels": ["I", "I", "O"], "recursive": True, "temporal": True},
            "I1_H1_O1": {"labels": ["I", "H", "O"], "recursive": False, "temporal": False},
            "I1_H1_O1.R": {"labels": ["I", "H", "O"], "recursive": True, "temporal": False},
            "I1_H1_O1.T": {"labels": ["I", "H", "O"], "recursive": False, "temporal": True},
            "I1_H1_O1.RT": {"labels": ["I", "H", "O"], "recursive": True, "temporal": True},
            "I1_O2": {"labels": ["I", "O", "O"], "recursive": False, "temporal": False},
            "I1_O2.R": {"labels": ["I", "O", "O"], "recursive": True, "temporal": False},
            "I1_O2.T": {"labels": ["I", "O", "O"], "recursive": False, "temporal": True},
            "I1_O2.RT": {"labels": ["I", "O", "O"], "recursive": True, "temporal": True}
        }
        self.all_matrices = np.array(list(product([0, 1], repeat=9))).reshape(-1, 3, 3)

    def generate_motifs(self):
        """
        Generate a bank of motifs based on the specified types.
        Returns:
            motif_bank (dict): A dictionary containing the generated motif graphs and matrices per type.
            simplest_motifs (dict): A dictionary containing the simplest motif for each type.
            motifs (dict): A dictionary containing the generated motif objects.
        """
        motif_bank = {}
        motifs = {}

        # Generate motifs for each type
        for type in self.types:
            print(f"Generating motif class for {type}...")
            motif_bank[type], motifs[type] = self.generate_motif_type(type, **self.type_properties[type])
            print(f"Generated {len(motif_bank[type])} motifs for {type}.")

        # Find the simplest motif for each type using complexity
        simplest_motifs = {type: min(bank, key=lambda x: len(bank[x][0].edges)) for type, bank in motif_bank.items()}
        simplest_motifs = {type: motif_id for type, motif_id in simplest_motifs.items()}
        return motif_bank, simplest_motifs, motifs

    def generate_motif_type(self, type, labels, recursive, temporal):
        """
        Generate a specific motif type.
        Arguments:
            type (str): The type of the motif.
            labels (list): The labels for the nodes.
            recursive (bool): Whether the motif is recursive.
            temporal (bool): Whether the motif is temporal.
        Returns:
            bank (dict): A dictionary containing the generated motif graphs and matrices.
            motifs (dict): A dictionary containing the generated motif objects.
        """
        bank = {}
        motifs = {}
        id = 0

        for matrix in self.all_matrices:
            # Create the graph
            graph = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
            nx.set_node_attributes(graph, {i: labels[i] for i in range(len(labels))}, 'label')

            # Check validity
            valid, complexity = self.valid_motif(matrix, graph, labels, recursive, temporal)
            if not valid:
                continue

            placed = False
            # Check if the motif's isomorphism is already in the bank
            for (graph2, _) in bank.values():
                if self.check_isomorphism(graph, graph2):
                    placed = True
                    break

            # Add motif to bank if not placed
            if not placed:
                bank[id] = (graph, matrix)
                motifs[id] = Motif(type, id, labels.count('I'), labels.count('O'), complexity, recursive, temporal)
                id += 1

        return bank, motifs

    def valid_motif(self, matrix, motif, labels, recursive=False, temporal=False):
        """Check if the motif is valid.
        Criteria for validity:
            - If not recursive, self-loops are not allowed.
            - If recursive, there must be at least one self-loop.
            - If not temporal, edges must go from lower to higher layers.
            - If temporal, at least one edge must go from higher to lower layer.
            - Every input node must have a path to every hidden node.
            - Every input node must have a path to every output node.
            - Every hidden node must be reachable from every input node.
        Arguments:
            motif (networkx.DiGraph): The motif graph.
            labels (list): List of node labels.
            recursive (bool): Whether the motif has recursive connections.
            temporal (bool): Whether the motif has temporal connections.
        Returns:
            list: A list containing the validity status and complexity.
        """
        layers = {'I': 0, 'H': 1, 'O': 2}
        complexity = 0
        
        # Check for self loops and validate against properties
        self_loops = [i for i in motif.nodes() if motif.has_edge(i, i)]
        if not recursive and self_loops:
            return [False, complexity]
        if recursive and not self_loops:
            return [False, complexity]


        # Check for temporal edges and validate against properties
        temporal_edges = [e for e in motif.edges() if layers[labels[e[0]]] >= layers[labels[e[1]]] and e[0] != e[1]]
        if not temporal and temporal_edges:
            return [False, complexity]
        if temporal and not temporal_edges:
            return [False, complexity]

        # Compute complexity
        complexity += 2 * (len(self_loops) + len(temporal_edges))
        n_forward_edges = len(motif.edges()) - len(self_loops) - len(temporal_edges)
        complexity += n_forward_edges

        # Gather node types
        inputs = [i for i, r in enumerate(labels) if r == 'I']
        hiddens = [i for i, r in enumerate(labels) if r == 'H']
        outputs = [i for i, r in enumerate(labels) if r == 'O']

        # Compute transitive closure
        trans_closure = nx.transitive_closure(motif, reflexive=False)

        if hiddens:
            # Check path from input to hidden
            if not trans_closure.has_edge(inputs[0], hiddens[0]):
                return [False, complexity]

            # Check path from hidden to output
            if not trans_closure.has_edge(hiddens[0], outputs[0]):
                return [False, complexity]

        # Check path from input to output
        for i in inputs:
            for o in outputs:
                if not trans_closure.has_edge(i, o):
                    return [False, complexity]

        return [True, complexity]

    def check_isomorphism(self, motif1, motif2):
        """
        Check if two motifs are isomorphic.
        Arguments:
            motif1 (networkx.DiGraph): The first motif to compare.
            motif2 (networkx.DiGraph): The second motif to compare.
        Returns:
            bool: True if the motifs are isomorphic, False otherwise.
        """
        matcher = DiGraphMatcher(motif1, motif2, node_match=categorical_node_match('label', None))
        return matcher.is_isomorphic()