import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from dataclasses import dataclass
import itertools

@dataclass(frozen=True, slots=True)
class MotifStructure:
    """
    The structural information of a motif.
    Properties:
        type (str): The type of the motif.
        id (int): The ID of the motif.
        labels (tuple): The labels associated with the motif.
        matrix (np.ndarray): The adjacency matrix of the motif.
        n_inputs (int): The number of input connections to the motif.
        n_outputs (int): The number of output connections from the motif.
        layers (np.ndarray): The layer assignments for each node in the motif.
        I (np.ndarray): The input nodes of the motif.
        H (np.ndarray): The hidden nodes of the motif.
        O (np.ndarray): The output nodes of the motif.
        F (np.ndarray): The feedforward connections of the motif.
        NF (np.ndarray): The recurrent connections of the motif.
        complexity (int): The complexity score of the motif.
    """
    type: str
    id: int
    labels: tuple
    matrix: np.ndarray
    n_inputs: int
    n_outputs:int
    layers: np.ndarray
    I: np.ndarray
    H: np.ndarray
    O: np.ndarray
    F: np.ndarray
    NF: np.ndarray
    complexity: int
    recursive: bool
    temporal: bool

class Motif:
    def __init__(self, structure):
        """
        Initialise the motif.
        Arguments:
            structure (MotifStructure): The structural information of the motif.
        Properties:
            previous_values (array): Previous output values of the motif.
        """
        self.structure = structure
        self.previous_values = None  
        
    def reset(self):
        """
        Reset the motif's state.
        Resets:
            previous_values (array): Previous output values of the motif.
        """
        self.previous_values = None

    def forward_pass(self, inputs):
        """
        Perform a forward pass through the motif.
        Arguments:
            inputs (array): The input values.
        Returns:
            (array): The output values after the forward pass.
        """
        s = self.structure
        values = np.zeros(3, dtype=np.float32)
        prev = self.previous_values if self.previous_values is None else np.asarray(self.previous_values, dtype=np.float32)

        # Input layer processing
        values[s.I] = np.asarray(inputs, dtype=np.float32)
        if prev is not None and s.I.size:
            values[s.I] += prev @ s.NF[:, s.I]

        # Hidden layer processing
        if s.H.size:
            if s.F[:, s.H].any():
                values[s.H] += values @ s.F[:, s.H]
            if prev is not None and s.NF[:, s.H].any():
                values[s.H] += prev @ s.NF[:, s.H]

        # Output layer processing
        if s.F[:, s.O].any():
            values[s.O] += values @ s.F[:, s.O]
        if prev is not None and s.NF[:, s.O].any():
            values[s.O] += prev @ s.NF[:, s.O]

        self.previous_values = values
        return values[s.O]


class MotifBank:
    def __init__(self, motif_types):
        """
        Initialise the motif bank.
        Arguments:
            motif_types (list): The types of motifs to include in the bank.
        Properties:
            motif_generator (MotifGenerator): The generator used to create motifs.
            structures (dict): The bank of motif structures.
            simplest_motifs (list): The simplest motif per type.
            init_bank (dict): The bank of motifs with distributions for initialisation.
        """
        self.motif_types = motif_types
        self.motif_generator = MotifGenerator(motif_types)
        self.structures, self.simplest_motifs = self.motif_generator.generate_structures()

        # Initialise the initial distribution of motifs
        n_types = len(self.motif_types)
        self.init_bank = {t: [id, 1 / n_types] for t, id in self.simplest_motifs.items()}

    def plot_motif_bank(self, type=None):
        """
        Plot the motifs in the bank.
        Arguments:
            type (str): The type of motifs to plot.
        """
        structs = self.structures
        types = structs.keys() if type is None else [type]

        for t in types:
            for motif_id in structs[t].keys():
                self.plot_motif(t, motif_id)

    def plot_motif(self, m_type, m_id):
        """
        Plot a specific motif from the bank.
        Arguments:
            m_type (str): The motif type.
            m_id (int): The motif ID.
        """
        # Load structure
        s = self.structures[m_type][m_id]

        # Convert structure to graph
        G = nx.from_numpy_array(s.matrix, create_using=nx.DiGraph)
        nx.set_node_attributes(G, {i: {'label': s.labels[i]} for i in range(3)})

        # Define node positions and colors
        pos = {0: (-1, 0), 1: (0, 1), 2: (1, 0)}
        colours = {'I': 'lightblue', 'H': 'lightgreen', 'O': 'lightcoral'}
        node_colours = [colours[G.nodes[i]['label']] for i in G.nodes]

        # Plot
        plt.figure(figsize=(3, 3))
        nx.draw(G, pos, with_labels=True, node_color=node_colours, node_size=800, font_size=10, arrowsize=20)
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
        self.permutations = self.generate_permutations()
        self.all_matrices = np.array(list(product([0, 1], repeat=9))).reshape(-1, 3, 3)
    
    def generate_permutations(self):
        """
        Generate all permutations for each motif type.
        Returns:
            perms (dict): The permutations for each motif type.
        """
        t_props, types = self.type_properties, self.types
        perms = {t: [] for t in types}
        for t in types:
            # Get labels for the current motif type
            labels = np.asarray(t_props[t]['labels'])

            for p in itertools.permutations(range(3)):
                p = np.array(p)
                # Check if the permutation is valid
                if np.all(labels[p] == labels):
                    perms[t].append(p)
        return perms

    def generate_structures(self):
        """
        Generate motif structures.
        Returns:
            structures (dict): The generated motif structures.
            simples (dict): The simplest motif per type.
        """
        structures, simples = {}, {}
        t_props, permutations, types = self.type_properties, self.permutations, self.types

        for t in types:
            # Get properties, labels, and permutations for the current motif type
            props, labels, perms = t_props[t], t_props[t]['labels'], permutations[t]

            # Initialize data structures
            seen, store = set(), {}
            best_id, best_edges, next_id = None, 999, 0
            
            # Masks for feedforward and non_feedforward connections
            layers =  np.array([0 if L=='I' else 1 if L=='H' else 2 for L in labels], dtype=np.uint8)
            fmask = (layers[:, None] < layers[None, :])
            nfmask = (layers[:, None] >= layers[None, :])

            for m in self.all_matrices:
                # Check for self-loops and rule out invalid motifs
                self_loops = bool(np.any(np.diag(m)))
                if not props['recursive'] and self_loops:
                    continue
                if props['recursive'] and not self_loops:
                    continue

                # Check for non-feedforward connections and rule out invalid motifs
                non_forward = nfmask & ~np.eye(3, dtype=bool) & (m != 0)
                if not props['temporal'] and non_forward.any():
                        continue
                if props['temporal'] and not non_forward.any():
                    continue

                # Check for valid motifs
                if not self.valid_motif(m, labels, nfmask, props['recursive'], props['temporal']):
                    continue

                # Check for canonical form and uniqueness
                key = self.canonical_key(m, perms)
                if key in seen:
                    continue
                seen.add(key)

                # Store the motif node structure
                I = np.where(layers == 0)[0]
                H = np.where(layers == 1)[0]
                O = np.where(layers == 2)[0]

                # Create copy of the matrix
                M = m.astype(np.float32, copy=True)
                # Create view of feedforward connections
                F = (M * fmask).astype(np.float32, copy=False)
                # Create view of non-feedforward connections
                NF = (M * nfmask).astype(np.float32, copy=False)

                # Calculate complexity, n_inputs and n_outputs
                c = self.calculate_complexity(m, nfmask)
                n_input, n_output = np.sum(layers == 0), np.sum(layers == 2)

                # Store the motif structure
                store[next_id] = MotifStructure(t, next_id, tuple(labels), m, n_input, n_output, layers, I, H, O, F, NF, c, props['recursive'], props['temporal'])

                # Update simplest motif if necessary
                edges = np.sum(m)
                if best_id is None or edges < best_edges:
                    best_id = next_id
                    best_edges = edges

                next_id += 1

            # Store the generated structures and simplest motifs
            structures[t] = store
            simples[t] = best_id

            print(f"Generated {len(store)} unique motifs of type {t}.")

        return structures, simples

    def valid_motif(self, matrix, labels, nfmask, recursive, temporal):
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
            matrix (ndarray): The adjacency matrix of the motif.
            labels (list): List of node labels.
            nfmask (ndarray): The non-feedforward mask.
            recursive (bool): Whether the motif has recursive connections.
            temporal (bool): Whether the motif has temporal connections.
        Returns:
            (bool): Whether the motif is valid.
        """
        # Check for self loops and validate against properties
        self_loops = bool(np.any(np.diag(matrix)))
        if not recursive and self_loops:
            return False
        if recursive and not self_loops:
            return False
        
        # Check for temporal edges and validate against properties
        temporal_edges = bool(np.any((nfmask & ~np.eye(3, dtype=bool)) & (matrix != 0)))
        if not temporal and temporal_edges:
            return False
        if temporal and not temporal_edges:
            return False

        # Check reachability and nodes
        r = self.reachability(matrix)
        idx_i = [i for i, l in enumerate(labels) if l == 'I']
        idx_h = [i for i, l in enumerate(labels) if l == 'H']
        idx_o = [i for i, l in enumerate(labels) if l == 'O']

        # Check input to hidden and hidden to output reachability
        if idx_h:
            if not r[idx_i[0], idx_h[0]]:
                return False
            if not r[idx_h[0], idx_o[0]]:
                return False

        # Check input to output reachability
        for i in idx_i:
            for o in idx_o:
                if not r[i, o]:
                    return False
                
        return True
    
    def reachability(self, matrix):
        """
        Compute the reachability matrix using the Floyd-Warshall algorithm.
        Arguments:
            matrix (ndarray): The adjacency matrix of the motif.
        Returns:
            (ndarray): The reachability matrix.
        """
        # Initialize the reachability matrix
        m = np.copy(matrix.astype(bool))

        # Apply the Floyd-Warshall algorithm
        for i in range(3):
            m = m | (m[:, i:i + 1] & (m[i:i + 1, :]))

        return m

    def canonical_key(self, matrix, permutations):
        """
        Compute a canonical key for the motif.
        Arguments:
            matrix (ndarray): The adjacency matrix of the motif.
            permutations (list): A list of node permutations.
        Returns:
            c_key (tuple): The canonical key.
        """
        c_key = None
        for p in permutations:
            # Apply the permutation to the matrix and compute the flattened key
            mp = matrix[np.ix_(p, p)]
            flat = mp.astype(np.uint8, copy=False).reshape(-1)

            # Compute the key
            k = int(flat.dot(1 << np.arange(9)))

            # Update the canonical key if necessary
            if c_key is None or k < c_key:
                c_key = k
        return c_key

    def calculate_complexity(self, matrix, nfmask):
        """Calculate the complexity of the motif.
        Arguments:
            matrix (ndarray): The adjacency matrix of the motif.
            nfmask (ndarray): The non-feedforward mask.
        Returns:
            float: The complexity of the motif.
        """
        # Self-loops
        n_self = np.sum(bool(np.any(np.diag(matrix))))
        # Temporal connections
        n_temporal = np.sum(nfmask & ~np.eye(3, dtype=bool) & (matrix != 0))
        # Feedforward connections
        n_fwd = np.sum(matrix) - n_self - n_temporal
        return 2 * (n_self + n_temporal) + n_fwd