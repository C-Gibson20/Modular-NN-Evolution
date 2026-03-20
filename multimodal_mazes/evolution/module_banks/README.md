# Module Banks

This directory defines **module and motif banks** — reusable building blocks that genomes can draw from when constructing solutions to tasks.
Module banks expose simple forward-pass modules (e.g., convolution, feedforward, recurrent) and higher-level motifs (small recurrent/temporal graph structures).

They are central to the **EA’s compositionality**: genomes don’t learn from scratch, but compose and evolve networks from a structured vocabulary of modules and motifs.

## Contents

* **`image_bank.py`**
  Provides the **ImageModuleBank** and a library of modules for image-based tasks.

  * **Modules included:**

    * `Bus` – routes tagged inputs across positions.
    * `Convolution` – 3×3 convolution with uniform kernel.
    * `Correlation` – 3×3 correlation with alternating kernel.
    * `Argmax` – reduces an input map to the index of the maximal value.
    * `Feedforward` – scalar passthrough, unbiased initialization.
  * **Legacy modules:**

    * `SquareClassificationModule` / `CrossClassificationModule` (early prototypes).

* **`maze_bank.py`**
  Provides the **MazeModuleBank** with modules designed for sequential maze navigation.

  * `Feedforward` – combines two inputs with fixed weights and ReLU activation.
  * `Recurrent` – feedforward computation plus a weighted recurrence term (`previous_output`).
  * Both banks (scalar + general) expose initial distributions for genome initialization.

* **`motifs.py`**
  Provides reusable **three-node graph motifs** that encode structured connectivity patterns.

  * `MotifStructure` dataclass encodes adjacency, labels, and properties.
  * `Motif` executes forward passes given inputs and maintains recurrent state.
  * `MotifBank` manages a set of motifs across types (feedforward, recursive, temporal).
  * `MotifGenerator` enumerates and validates motifs given label templates and constraints.
  * Includes utilities to **plot motifs** using NetworkX/Matplotlib.


## Usage Patterns

### Image module bank

```python
from multimodal_mazes.evolution.module_banks.image_bank import ImageModuleBank

# Initialise bank for square-classification tasks
bank = ImageModuleBank(img_type='square')
print(bank.bank.keys())  # dict of available modules and weights

# Sample and apply a module
conv_cls, _ = bank.bank['convolution']
module = conv_cls(layer=0)
output = module.forward_pass(input_vector)
```

### Maze module bank

```python
from multimodal_mazes.evolution.module_banks.maze_bank import MazeModuleBank

bank = MazeModuleBank()
ff_cls, _ = bank.bank['feedforward']
ff_module = ff_cls()
out = ff_module.forward_pass(np.array([1.0, 0.5]))
```

### Motif bank

```python
from multimodal_mazes.evolution.module_banks.motifs import MotifBank

motif_bank = MotifBank(motif_types=["I1_H1_O1", "I2_O1"])
motif_bank.plot_motif_bank()

# Access specific motif structure
structure = motif_bank.structures["I1_H1_O1"][0]
motif = motif_bank.motif_generator
```

## Design Notes

* **Distributions:** Each bank defines initial module/motif distributions. These are evolved during EA runs.
* **Statefulness:** Some modules (`Recurrent`, `Motif`) maintain internal state between forward passes. Always call `reset()` before reusing in a new trial.
