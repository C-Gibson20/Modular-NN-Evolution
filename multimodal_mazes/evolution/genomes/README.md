# Genomes

This directory contains the genome encodings used in evolutionary experiments.Each genome defines a representation of candidate networks, including their modules/motifs,
connection rules, mutation operators, and compilation into an executable forward pass.

The genomes differ in their expressiveness, constraints, and intended use cases,
ranging from prototyping-oriented baselines to fully flexible distributed architectures.

## Files

* **`genome.py` — Prototyping only**
* **`genome_motif.py` — Motif genome** 
* **`genome_dist.py` — Module genome** 
    * Layered wide/deep wiring
    * Inter-module edges
    * Vector edges


## `genome.py` — Prototyping only

A reduced-functionality genome used for prototyping and validation. It wires one selected module homogeneously to inputs/outputs, ensuring the EA pipeline can be tested end-to-end without the full feature set of the other genomes.

### Scope and Behaviour

* **Two-stage topology** 
    * Inputs → Modules → Outputs.
    * No inter-module edges, layers, or vector edges.

* **Task-fixed modules**
    * `maze`: every module is a simple Recurrent unit.
    * `image`: every module is a 3×3 classifier SquareClassificationModule or CrossClassificationModule.
* **Lightweight mutations** 
    * Add/remove/swap connections
    * Optional per-edge weight mutation.
* **Compiled forward pass** 
    * Over ragged buffers for speed.
    * Simple per-module `forward_pass`.

### Use Case
* Quick end-to-end tests of the EA machinery.
* Deterministic check for module prototyping.

## `genome_dist.py`

A full-featured genome supporting wide, deep, and distributed architechtures. 

### Scope and Behaviour

* **Three-stage topology** 
    * Inputs → Modules → Outputs.
        * Modules layered wide and deep
    * Supports inter-module edges for depth and recurrence
    * Vector edges supported in addition to scalar connections.
        * Vector edges connect all output ports of a module at once.
* **Task-dependent modules**
    * `maze`: modules from `maze_bank.py`.
    * `image`: modules from `image_bank.py`
* **Full mutation set** 
    * Add/remove/swap connections
    * Optional per-edge weight mutation.
    * Swap module from the module bank.
* **Flexible connectivity**
    * Connectivity can grow arbritrarily through mutations into deep or distributed topologies.
* **Compiled forward pass** 
    * Over ragged buffers for speed.
    * Layered per-module `forward_pass`.

### Use Case
* Investigating reusable building blocks in evolved networks
* Exploration of deep, wide, and distributed networks.
* Analysis of strucutre function relationships.

## `genome_motif.py` 

A biologically-plausible genome where the building blocks are three-node motifs.

### Scope and Behaviour

* **Two-stage topology** 
    * Inputs → Modules → Outputs.
    * Supports for inter-module edges in progress.
* **Task-independent modules**
    * Motifs from `motifs.py`
* **Full mutation set** 
    * Add/remove/swap connections
    * Optional per-edge weight mutation.
    * Swap module from the module bank.
* **Compiled forward pass** 
    * Over ragged buffers for speed.
    * Layered per-module `forward_pass`.
* **Motif management**
    * Hyperparameters specify homogeneity.
    * Distribution across types tracked.

### Use Case
* Investigating reusable building blocks in evolved networks.
* Studying homogenous versus heteregenous motif populations.
* Analysis of strucutre function relationships.


## Hyperparameters

Each genome class expects a `hyperparameters` dictionary.
The exact fields differ between `genome.py`, `genome_dist.py`, and `genome_motif.py`.

### Common Fields

| Key | Type | Used by | Description |
| --- | ---- | ------- | ----------- |
| `task`| `str`| all | `"maze"` or `"image"`. Determines which modules are used. | 
| `n_inputs` | `int` | all | Number of network inputs. |
| `n_outputs` | `int` | all | Number of network outputs. |
| `weight_sharing`| `bool`| all | Enable shared weights across modules/motifs. |
| `uniform_weights` | `bool` | all | If true, shared weights are uniform across modules. |
| `one_to_one` | `bool` | all | Enforce each input/output/module port to be used at most once. |
| `connectivity` | `str` | all | One of `UNCONNECTED`, `SPARSE`, `RANDOM`, `FULLY CONNECTED`, or `IDEAL`. |
| `connection_density` | `dict`| all | `{ 'input_density': float, 'output_density': float }`. Used for `RANDOM` connectivity. |
| `mutation_rates` | `list[float]` | all |Probabilities for choosing mutation operators. Must sum to 1. |
| `n_modules`| `int` | `genome.py` and <br> `genom_dist.py` |Number of modules in the genome. |
| `n_module_types` | `int` | `genome.py` and <br> `genom_dist.py` | Size of the module bank (even though only one module is used). |
| `class_type`| `str` | all | Specifies the exact task. |
| `network_type` | `int`  | `genome_dist.py` | 0 for wide, 1 for deep, 2 for wide and deep|
| `n_motifs` | `int` | `genome_motif.py` | Number of motifs in the genome. |
| `motif_types` | `list[str]` | `genome_motif.py` |Types of motifs to include (keys in the motif bank). |
| `homogeneous` | `bool`| `genome_motif.py` |If true, all motifs are of the same type.|

### Connectivity Modes

* **UNCONNECTED**: start with no edges.
* **SPARSE**: one input and one output connection per module.
* **RANDOM**: sample counts from `connection_density` for inputs/outputs.
* **FULLY CONNECTED**: every input connects to every module port; every module port connects to every output.
* **IDEAL**: This wiring mode should only be used for prototyping.

### Weight Sharing

Optional **per-module-type** shared weights:

* Each module has shared weights across its inputs and outputs. 
* If `uniform_weights=True`, shared weights are **1.0**; else sampled once per module type
* When weight sharing is enabled, **per-edge weight mutation is disabled**

### One-to-one Constraint

If enabled:

* Each input can feed **at most one** module port; each module port is **used at most once**; and each output is **driven by at most one** module port.
* Operators respect availability; if no slots are free, they attempt a **swap** instead of a failing add.

## Execution Model

To evaluate quickly, connection rules are compiled into compact arrays:

* `Edges(indptr, srcs, src_pts, dst_pts, wts, dst)` for both **CONNECT\_IN**, **CONNECT\_OUT** and **CONNECT\_MOD**.
* A layer run creates a ragged buffer (`Ragged(vals, ptr)`) for the destination layer
* **Layer 0** (Inputs→Modules): accumulates per-module input ports, then calls each module’s `forward_pass`
* **Layer 1** (Modules→Outputs): weighted sum into outputs
* **Layer > 1** (Modules→Modules): accumulates per-module input ports, then calls each module’s `forward_pass`.

## Genetic operators

* **add\_connection**: add either an input→module, module→module or module→output connection (respecting one-to-one and weight sharing)
* **remove\_connection**: remove a random existing connection
* **swap\_connection**: swap sources (for input edges or module) or destinations (for output edges)
* **modify\_connection\_weight**: small Gaussian tweak to a random edge weight (skipped if `weight_sharing=True`)
* **crossover**: single cut point across the **module list**; rules are cut/merged accordingly and validated against one-to-one

## Visualization

`plot_genome()` draws Inputs — Modules — Outputs with edge labels showing **port** and **weight**. The canvas height adapts to task size.


## Genomes Comparison

| Capability | `genome.py` | `genome_dist.py` | `genome_motif.py` |
| ---------- | ----------- | ---------------- | ----------------- |
| **Inter-module wiring** | — | Fully supported | In progress | —                                                        |
| **Vector-edges** | — | Fully supported | — |
| **Depth** | Wide | Wide / Deep / <br>Wide and Deep <br> in progress  | Wide /<br> Deep in progress / <br> Wide and Deep in progress|
| **Building blocks** | Selected module | Module bank | Motif bank |

