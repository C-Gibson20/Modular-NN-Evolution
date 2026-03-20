# Evolutionary Algorithms

This directory contains the evolutionary algorithms (EAs) used to search over different genome encodings. All three classes implement the same high-level loop—**generate → evaluate (in parallel) → select/elites → crossover/mutate → repeat**—but they differ in the type of genome they optimize and the data they record.

## Files

* **`genome_EA.py` – For prototyping only**

  * Optimizes the simplified `Genome` (from `genomes/genome.py`).
  * No module/motif banks.
* **`genome_EA_dist.py` – EA for distributed module genomes**

  * Optimizes `GenomeDist` (from `genomes/genome_dist.py`).
  * Uses a **ModuleBank** (`MazeModuleBank` or `ImageModuleBank`) and tracks module ratios across elites.
* **`genome_EA_motif.py` – EA for motif genomes**

  * Optimizes `GenomeMotif` (from `genomes/genome_motifs.py`).
  * Uses a **MotifBank** and tracks both **motif instance ratios** and **type ratios** across elites.

## Common workflow

All classes expose the same public surface:

1. **`generate(task_obj, sensor_noise_scale, n_steps=None)`**
   Creates the initial population and sets up the parallel evaluator.

   * `task_obj`: task instance (e.g., Maze env or ImageClassification dataset wrapper).
   * `sensor_noise_scale`: noise scale injected into the evaluator.
   * `n_steps`: optional episode length (maze); not used for image evaluators.

2. **`evaluate()`**

   * Splits the population into chunks and evaluates in parallel using a `ProcessPoolExecutor`.
   * Writes per-genome results to `self.trial_data[self.generation][genome_id]`.
   * Updates elite set (`self.fittest_networks`) and summary fitness.

3. **`evolve()`**

   * Elitism: carries over top `top_genomes`.
   * Fills the rest via **crossover** and **mutation** of elites.
   * Increments `self.generation`.

4. **Plotting**

   * `plot_fitness_over_generations()` in all EAs.
   * Distribution plots in the specialized EAs.

Parallel evaluation is driven by small adapter functions (`_evaluate_genome_chunk`) that dispatch to **task-specific evaluators**:

* Maze: `evaluator._init_evaluator`, `evaluator.genome_eval_fitness`
* Image: `evaluator_image._init_evaluator`, `evaluator_image.genome_eval_fitness`
* Motif variants: corresponding `*_motif` evaluators

## Hyperparameters 

Each EA expects a `hyperparameters` dict with the following keys (the EA forwards genome-specific keys to the genome constructors):

| Key | Type | Used by | Description |
| --- | ---- | ------- | ----------- |
| `task` | `str` | all | `"maze"` or `"image"`; selects evaluator and the module/motif bank. |
| `population_size`| `int` | all | Number of genomes per generation.|
| `top_genomes`    | `int` | all | Number of elites carried over unchanged.|
| `mutation_rate`  | `float| all | Relative rate of mutation.|
| `crossover_rate` | `float| all | Relative rate of crossover. |
| `class_type` | `str` | all | Specifies the exact task.  |
| `motif_types` | `list[str]` | `genome_EA_motif.py` | Motif families to include in the MotifBank. |
| *(plus genome fields)* | — | all | All fields required by the underlying genome |

**Operator mix.** The number of offspring is `population_size - top_genomes`. Offspring are split between crossover and mutation with probability proportional to `crossover_rate` and `mutation_rate` (binomial draw per generation).


## Class-specific notes

### `GenomeEA` 

* **Genome**: `genomes/genome.py` (prototyping, homogeneous modules, no inter-module edges).
* **Evaluators**: maze / image.
* **Parallelism**: process pool; chunk size ≈ `ceil(population_size / (cpu_count-1))`.
* **Diagnostics**:

  * `trial_data[g]['fittest']` → `(fitness, genome)` for generation `g`
  * `trial_data[g]['fitness']` → mean fitness across elites
  * No distribution tracking.

**Typical loop**

```python
ea = GenomeEA(hparams)               # includes genome hyperparameters
ea.generate(task_obj, sensor_noise_scale, n_steps)
for _ in range(n_generations):
    ea.evaluate()
    ea.evolve()
ea.plot_fitness_over_generations()
```


### `GenomeEADist` 

* **Genome**: `genomes/genome_dist.py` (layers, inter-module and vector edges, banks).
* **Module banks**:

  * Maze: `MazeModuleBank()`
  * Image: `ImageModuleBank(class_type)`
* **Evaluators**: maze / image.
* **Extra diagnostics**:

  * **Module distribution** of elites: `trial_data[g]['mod_distribution']`
    Aggregated (then averaged) across the top `top_genomes`.

**Extra helper functions**

* `module_distribution_over_generations()` → `(generations, mod_dist)`
* `plot_module_distribution_over_generations()` → line plot per module type.


### `GenomeEAMotif` 

* **Genome**: `genomes/genome_motifs.py` (higher-level building blocks with internal wiring).
* **Bank**: `MotifBank(motif_types)`.
* **Evaluators**:

  * Uses `evaluator_motif` (maze).
* **Extra diagnostics**:

  * **Motif distribution** (per motif instance id within each type): `trial_data[g]['mot_distribution']`.
  * **Type distribution** (fraction of motifs by type in the population): `trial_data[g]['type_distribution']`.

**Extra helper methods**

* `motif_distribution_over_generations()` and `plot_motif_distribution_over_generations()`
* `type_distribution_over_generations()`


## Parallel evaluation details

* Each EA sets up a `ProcessPoolExecutor` with `max_workers = max(1, os.cpu_count() - 1)`.
* The pool is **initialized once** (on the first `generate` call) using the task-specific `_init_evaluator(...)`.
* Populations are split into contiguous slices of size `ceil(population_size / max_workers)`, evaluated with `_evaluate_genome_chunk`.
* Evaluator results must include at least `res['score']` (fitness). Entire per-trial payload is stored in `trial_data[generation][genome_id]`.
