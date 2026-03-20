# Evaluators

This directory contains task-specific **evaluators** used by the EAs. Each evaluator exposes a small, consistent API so EAs can run genomes (or NEAT genomes) across tasks (maze, image) in parallel worker processes.

All evaluators follow the same pattern:

1. **`_init_evaluator(...)`**
   Called **once per worker** (via `ProcessPoolExecutor(initializer=...)`). It caches task data in module-level globals to avoid heavy IPC.
2. **`*_trial(...)`**
   Runs **one agent** for **one episode** (maze) or **one sample** (image) and returns low-level outcomes (e.g., time-to-goal, path, agent output).
3. **`genome_eval_fitness(genome)`** (or `eval_fitness(...)` in NEAT variants)
   Loops over the dataset/mazes using the globals initialized above, aggregates metrics, and returns a **dict** with a `"score"` field (fitness in \[0,1]) plus useful extras (paths, success flags, per-trial outputs).

NEAT versions differ only in the agent and argument signatures (they pass the NEAT `genome` and `config`), and Motif versions reset motifs before each trial.

## Files

### `evaluator.py` (maze, Genome)

Evaluates **`genomes/genome.py` or `genomes/genome_dist.py`** style networks in maze environments.

* **Initializer**

  * `_init_evaluator(maze, sensor_noise_scale, n_steps)`
  * Caches: mazes, start/goal locations, noise scale, episode length.
* **Trial**

  * `genome_maze_trial(genome, mz, start_loc, goal_loc, sensor_noise_scale, n_steps) -> (time, path)`
  * Creates an `AgentGenome`, steps for up to `n_steps`, records the path and stop time.
* **Fitness**

  * `genome_eval_fitness(genome) -> dict`
  * For each maze:

    * **Time component**: normalized by fastest known solution
      `1 - ((time - fastest) / (n_steps - 1 - fastest))`
    * **Path component**: normalized distance to goal from final position.
  * **Final fitness**: mean of 0.5·(time\_norm + path\_norm) across mazes.
  * **Return payload**:

    ```python
    {
      "score": float,                     # mean fitness in [0,1]
      "normalised_data": {"paths": [...], "times": [...], "fitness": [...]},
      "times": [int, ...],
      "paths": [[(r,c), ...], ...],
      "success": [bool, ...]
    }
    ```

### `evaluator_image.py` (image, Genome)

Evaluates **`genomes/genome.py` or `genomes/genome_dist.py`** style networks on a simple image classification/navigation task.

* **Initializer**

  * `_init_evaluator(img_dataset, sensor_noise_scale)`
  * Caches: images, goal locations, noise scale.
* **Trial**

  * `genome_image_trial(genome, img, sensor_noise_scale) -> agent_output`
  * Builds an `AgentImage`, resets module/motif state where relevant, runs policy once over the image, returns output.
* **Fitness**

  * `genome_eval_fitness(genome) -> dict`
  * For each image:

    * **Success**: exact match to target coordinate.
    * **Distance score**: normalized distance map improvement from max to predicted location.
  * **Final fitness**: mean of distance-based scores (NaN-robust via `nanmean`).
  * **Return payload**:

    ```python
    {
      "score": float,
      "outputs": [agent_output, ...],
      "success": [bool, ...]
    }
    ```

### `evaluator_image_neat.py` (image, NEAT)

Evaluates **NEAT** genomes on image tasks.

* **Core API**

  * Like the maze NEAT evaluator but exchanging maze env for image data.
  * Uses a NEAT-compatible agent (e.g., `AgentNeatImage` if present) that consumes `(genome, config)`.
* **Fitness**

  * Typically mirrors `evaluator_image.py`: success boolean and distance-based normalization, averaged across images.
* **Signature differences**

  * Trial/eval functions accept `genome`, `config`, and task args supplied by the EA.

### `evaluator_maze_neat.py` (maze, NEAT)

Evaluates **NEAT** genomes in maze environments.

* **Trial**

  * `maze_trial(mz, start, goal, channels, sensor_noise_scale, drop_connect_p, n_steps, agnt=None, genome=None, config=None) -> (time, path)`
  * Uses `multimodal_mazes.AgentNeat(...)` to step the policy for up to `n_steps`.
* **Fitness**

  * `eval_fitness(genome, config, channels, sensor_noise_scale, drop_connect_p, maze, n_steps, agnt=None) -> float`
  * For each maze:

    * **Time component**: normalized by fastest known solution.
    * **Path component**: normalized by final distance to goal.
  * **Return**: mean of the average of those two components across mazes (scalar in \[0,1]).

### `evaluator_motif.py` (maze, Motif Genome)

Evaluates **`genomes/genome_motifs.py`** motif-based networks in maze (and image variants exist under `*_image_motif`).

* **Initializer**

  * `_init_evaluator(maze, sensor_noise_scale, n_steps)` (maze)
    or `_init_evaluator(img_dataset, sensor_noise_scale)` (image motif variant).
* **Trial**

  * Resets all motifs (`for motif in genome.motifs: motif.reset()`), then runs the corresponding agent (`AgentGenome` for maze, `AgentImage` for image).
* **Fitness**

  * Maze: identical normalization to `evaluator.py`.
  * Image: identical scoring to `evaluator_image.py`.
  * **Return payload** matches the task’s baseline evaluator shape, with `"score"` and diagnostic arrays.

## Common Design Choices

* **Global state per worker**
  Initializers stash heavy objects (mazes, distance maps, images) in module globals:
  `(_global_mazes, _global_starts, _global_goals, _global_noise, _global_n_steps, ...)`
  This keeps worker evaluation light and avoids serializing large arrays on every call.

* **Agents wrap the policy step**
  Evaluators do not implement control logic; they:

  1. Build an appropriate Agent (`AgentGenome`, `AgentImage`, `AgentNeat`, …),
  2. Call `sense → policy → act` (maze) or `policy` (image),
  3. Gather outcomes.

* **Fitness is task-normalized**
  Scores are designed to lie in **\[0,1]**, combining:

  * **time-to-goal** (relative to shortest known path) and
  * **final distance to goal** (via precomputed distance maps),
    or **distance-map quality** for image outputs.

* **Resets**
  Motif and module evaluators reset each motif/module before every trial to avoid state leakage.

* **Return contract**
  Every `genome_eval_fitness(...)` / `eval_fitness(...)` returns a dict (or scalar for NEAT maze) with at least:

  * `"score"`: float in \[0,1] (fitness),
  * Optional diagnostics (paths, times, success flags, raw outputs).
