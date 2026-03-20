# Task Trials

This directory contains **end-to-end experiment drivers** that wire up:

1. A task/dataset (maze or image),
2. An evolutionary algorithm (EA; module-based, motif-based, or the simple prototype), or **NEAT**,
3. An evaluator,
4. The training loop across generations, and
5. Saving results to disk (pickled).

All scripts follow a **similar skeleton**:

1. **Build the task**

   * Maze: `TrackMaze(size, n_channels, ...)` → `.generate(number, noise_scale, gaps)`
   * Image: `SquareClassification(size)` or `CrossClassification(size)` → `.generate(number)`
2. **Select and build the optimizer**

   * EA (module / dist): `GenomeEADist`
   * EA (motif): `GenomeEAMotif`
   * EA (prototype): `GenomeEA`
   * NEAT: uses `neat` library and a NEAT evaluator.
3. **Generate the population**
   `ea.generate(trial_obj, sensor_noise_scale, n_steps=None)`
4. **Training loop**

   ```
   for _ in range(n_generations):
       ea.evaluate()
       ea.evolve()
   ```
5. **Collect & persist results**

   * Either dump the EA state plus task config, or aggregate fitness curves (for comparison runs).
   * Files are saved as `.pkl` next to the script.

> Multiprocessing: the EAs initialize a `ProcessPoolExecutor` for evaluator workers. Scripts **shut it down** before pickling (`ea._pool.shutdown(...); ea._pool = None`) to avoid serialization issues.

## Files

### `genome_maze_trial.py`

**Runs:** EA over the prototype `Genome` on **maze** tasks.<br>
**Builds:** One or more `TrackMaze` instances; sets `sensor_noise_scale` and `n_steps`.<br>
**EA:** `GenomeEA` (prototype genome).<br>
**Save:** `genome_track_maze.pkl` containing:
```python
{
  "ea": <Genome without process pool>,
  "trial_object": TrackMaze,
  "hyperparameters": dict
}
```

### `genome_dist_trial.py`

**Runs:** EA over `GenomeDist` on **maze or image** tasks.<br>
**EA:** `GenomeEADist` (uses `genome_dist` with wide/deep wiring, vector edges).<br>
**Save:** `genome_dist_trial.pkl` containing:

```python
{
  "ea": <GenomeEADist without process pool>,
  "trial_object": <TrackMaze or ImageClassification>,
  "hyperparameters": dict
}
```

### `genome_square_trial.py` / `genome_cross_trial.py`

**Run:** EA on **image** tasks with **square**/**cross** classification datasets.<br>
**EA:** `GenomeEA` (prototype).<br>
**Dataset:** `SquareClassification` or `CrossClassification` with `.generate(number=...)`.<br>
**Save:** a pickle with EA/trial/hyperparameters.

### `genome_motif_trial.py`

**Runs:** EA over the **motif** genome on **maze** tasks.<br>
**EA:** `GenomeEAMotif`.<br>
**Save:** per-trial pickle with EA state.

### `genome_comparison_trial.py`

**Runs:** **Comparison** experiments across algorithm families (e.g., NEAT vs EA).<br>
**Flow:**
* Chooses task and builds the corresponding trial object(s).
* Defines `types_to_compare` (e.g., `{ "NEAT": ["NEAT"] }` or various motif/module presets).
* For NEAT entries: calls the NEAT trial runner.
* For EA entries: constructs an EA with the chosen hyperparameters, runs evolution, and gathers data.

**Save:** `genome_comparison.pkl` containing:

```python
{
  "fitness_results": {
    "<label>": {
      <replicate_index>: (generations, fitness_curve)
    },
    ...
  }
}
```

### `genome_motif_comparison_trial.py`

**Runs:** **Motif-focused** comparisons (optionally including NEAT as baseline).<br>
**Additional outputs:** besides fitness curves, it records motif usage over time:

```python
{
  "fitness_results": {...},
  "motif_distribution_results": {
    "<label>": {
      <replicate_index>: (generations, motif_ratios_by_type_and_id)
    }
  },
  "type_distribution_results": {
    "<label>": {
      <replicate_index>: (generations, type_ratios)
    }
  }
}
```

### `neat_maze_trial.py`

**Runs:** **NEAT** on maze tasks.<br>
**Flow:**

* Loads `exp_config.ini` (maze size, channels, steps, generations, etc.).
* Builds `TrackMaze` dataset.
* Standard NEAT `Config` and `Population`.
* `eval_genomes` uses `evaluator_maze_neat.eval_fitness(...)`.
* After each generation, stores the mean of the top-k fitnesses to track progress.
  **Return (from `run_neat_exp`)**: `(generations, fitness_curve)`.

### `neat_image_trial.py`

**Runs:** **NEAT** on image tasks (square/cross).<br>
**Flow:** mirrors `neat_maze_trial.py` but with image datasets and the corresponding evaluator.<br>
**Return:** `(generations, fitness_curve)`; useful for head-to-head comparisons with EA runs.
