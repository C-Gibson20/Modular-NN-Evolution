# Agents

Agents wrap **genomes / networks** and expose a `policy(...)` that produces an action or prediction from task inputs.
This directory provides agents for **maze** and **image** tasks, including a NEAT-based variant.

Agents expect a genome/network with a **forward pass** API and add **sensor noise** and **tie-breaking jitter** for argmax decisions.

## Contents

* **`agent_genome.py` – `AgentGenome`**
  Maze/grid agent that uses a custom Genome to map **channel inputs → action scores**.
  Inherits from `multimodal_mazes.agents.agent.Agent` which provides `location`, channels and input handling.

* **`agent_image.py` – `AgentImage`**
  Image agent that uses a custom Genome to map **flattened image inputs → pixel prediction** or raw scalar if `processing=False`.

* **`agent_neat_image.py` – `AgentNeatImage`**
  Image agent backed by a **NEAT recurrent network**. Supports **drop-connect** at inference for stochasticity / robustness.

## Conventions

* **Noise:** Agents add small random noise to inputs (`sensor_noise_scale`) and a tiny jitter (`/1000`) to logits before argmax to avoid bias.
* **Forward pass contract:** The provided `genome` (or NEAT network) must expose a method to compute outputs from a **1D float array**.

  * Custom Genome: `genome.forward_pass(np.ndarray[float]) -> np.ndarray[float]`
  * NEAT: handled via `neat.nn.RecurrentNetwork.activate(list_of_floats)`.
* **Output mapping (image agents):**
  If `processing=True`, the argmax index `i` is mapped to **2D coords** as:

  ```
  row = i // img.size
  col = i %  img.size
  ```

  If `processing=False`, the single scalar output is mapped the same way.


## `AgentGenome` (maze)

**File:** `agent_genome.py`<br>
**Use when:** You have a custom **Genome** and a maze state represented as **channel inputs**.

### Init

```python
from multimodal_mazes.evolution.agents.agent_genome import AgentGenome

agent = AgentGenome(
    location=np.array([r, c]),          # agent’s grid position
    channels=[...],                      # active input channels
    sensor_noise_scale=0.01,             # stddev of Gaussian noise on sensors
    genome=my_genome                     # must implement forward_pass(...)
)
```

### Policy

```python
agent.channel_inputs = ...              # shape: (num_channels, H, W) or compatible
agent.policy()
actions = agent.outputs                 # 1D array of action scores
```

**Notes**

* Extends a base `Agent` for grid tasks; expects `channel_inputs` to be set by the environment. 
* The action is determined by a method defined in the `Agent` class.


## `AgentImage`

**File:** `agent_image.py`<br>
**Use when:** You want to apply a custom **Genome** to image tasks.

### Init

```python
from multimodal_mazes.evolution.agents.agent_image import AgentImage

agent = AgentImage(
    sensor_noise_scale=0.01,
    genome=my_genome   # must implement forward_pass(...)
)
```

### Policy

```python
# img: 2D array-like (H, W)
# img_inputs: flattened or feature vector aligned with genome input size
agent.policy(img=img, img_inputs=img_inputs, processing=True)

prediction = agent.output  # [row, col]
```

**Parameters**

* `processing=True`: uses **argmax** over outputs to select a pixel and maps it to `[row, col]`.
* `processing=False`: treats the **first** output as a raw scalar index and maps it to coords. This assumes argmax is performed within the genome.

---

## `AgentNeatImage`

**File:** `agent_neat_image.py`<br>
**Use when:** You’re evolving **NEAT** networks and want an image agent with optional **drop-connect**.

### Init

```python
import neat
from multimodal_mazes.evolution.agents.agent_neat_image import AgentNeatImage

config = neat.Config(...)        # your NEAT config
genome  = ...                    # NEAT genome from population

agent = AgentNeatImage(
    sensor_noise_scale=0.01,
    drop_connect_p=0.05,         # 0.0 disables drop-connect
    genome=genome,
    config=config
)
```

### Policy

```python
agent.policy(img=img, img_inputs=img_inputs, processing=True)
prediction = agent.output  # [row, col]
```

**Drop-connect**

* Before each forward pass, a copy of `node_evals` is made and individual edges are randomly dropped with prob `drop_connect_p`.
* Use to explore robustness or regularize inference; set to `0.0` for deterministic behavior (aside from jitter and sensor noise).

---

