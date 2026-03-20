# Testing

This subdirectory contains code for **post-evolution evaluation** of genomes on **unseen task instances**.
Use these scripts to assess **generalization** (e.g., spot overfitting by comparing trial vs test performance).

## Contents

* **`test_genome_maze.py`**
  Utilities to evaluate a **maze genome** on newly generated mazes and persist results to a pickle for later analysis.

  * `test_genome(genome, maze, sensor_noise_scale, n_steps) -> dict`
  * `test_fittest_genome(fittest_genome, task, track, trial_maze=None) -> None` 
.

## `test_genome_maze.py`

For each maze instance, `test_genome_maze.py` runs `genome_maze_trial(...)` and collects:

* **`time`**: steps taken (lower is better).
* **`path`**: list of visited coordinates.
* **`success`**: `time < n_steps - 1`.

It then computes **normalized metrics** per maze:

* **Normalized time**

  $$
  \text{norm\_time} = 1 - \frac{time - \text{fastest\_solution}}{(n\_steps - 1) - \text{fastest\_solution}}
  $$

* **Normalized final distance** (via distance map `d_maps`)

  $$
  \text{norm\_path} = \frac{\max(d\_map) - d\_map[\text{last\_row}, \text{last\_col}]}{\max(d\_map)}
  $$

* **Fitness** (per maze):

  $$
  \text{fitness} = 0.5 \times (\text{norm\_time} + \text{norm\_path})
  $$

The returned/test-pickled structure is:

```python
{
  "maze": <Maze>,                  # the generated test maze object
  "test_data": {
    "fitness": np.ndarray,         # shape: (num_mazes,)
    "times": List[int],
    "paths": List[List[Tuple[int,int]]],
    "success": List[bool]
  }
}
```

## Typical Workflow

1. **Train/Evolve** with a trial (see `scripts/evolution_scripts/`).
2. **Select** a genome to test (often the fittest from the final generation).
3. **Choose a test distribution**:

   * `GeneralMaze(...).generate(...)` for broad layouts.
   * `TrackMaze(...).generate(...)` for track-like layouts.
   * Or pass `trial_maze` to reuse a specific maze set.
4. **Run** `test_fittest_genome(...)` or `test_genome(...)`.
5. **Analyze** the pickled results: mean fitness, success rate, time histograms, path visualizations.
