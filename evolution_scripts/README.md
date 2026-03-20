# Evolution Scripts

This directory contains notebooks and utilities for **running task trials, analyzing results, and validating evolutionary algorithm (EA) implementations**.
It serves as the interface layer for executing experiments and inspecting outcomes.


## Contents

* **Experiment notebooks**

  * Run trial files defined in the `evolution/task_trials` directory.
  * Execute evolutionary runs with configurable hyperparameters.
  * Analyze trial outcomes interactively.
  * Pickle trial objects for reproducibility.

* **`results/` subdirectory**

  * Stores serialized (pickled) trial objects after each experiment.
  * Organized according to algorithm type and task for easy loading into analysis notebooks.

* **`unit_testing/` subdirectory**

  * Contains notebooks for verifying correctness of EA variants and genome implementations.
  * **GTest** is used for **basic validation** (ensuring components run as expected).
  * **Custom testing** provides **in-depth functional verification** under varied conditions:

    * Displays intermediate steps and outputs.
    * Allows interactive modification of inputs.
    * Provides an informative debugging and validation environment.
* **`test_scripts/` subdirectory**
    * Run test files defined in the `evolution/testing` directory.
    * Evaluate evolved genomes with configurable hyperparameters.
    * Analyze test outcomes interactively.
    * Pickle test objects for reproducibility.

* **Plotting helpers**

  * A standalone script of reusable plotting functions.
  * Supports visualization of:

    * Fitness curves over generations.
    * Module distribution curves over generations
    * Motif distribution curves over generations
        * Type distribution
        * Property distribution

## File Index

### Experiment Notebooks
* `genome_motif_trial.ipynb` – Runs a genome motif trial, analyzes fitness curves and module use.
* `genome_motif_comparison_trial.ipynb` – Runs a genome motif comparison trial, compares fitness curves and motif use.
* `genome_dist_trial.ipynb`– Runs a genome distribution trial, analyzes fitness curves and module use.
* `genome_comparison_trial.ipynb`– Runs a genome comparison trial, compares fitness curves and module use.

### Unit Testing
* `genome_dist_testing.ipynb` – Validates genome initialization, forward pass, mutation, crossover, and clone behavior.
* `genome_motif_testing.ipynb` – Validates motif generation, genome initialization, forward pass, mutation, crossover, and clone behavior.

### Test Notebooks
* `test_genome_maze.ipynb` – Tests evolved genome population on new set of maze tasks.


## Experiment Notebooks
All trial notebooks follow the same pattern
1. **Imports**
2. **Run trial**
3. **Load results**
4. **Inspect fittest networks**
5. **Visualize task outputs**
6. **Plot training dynamics**
7. **Store results**

The notebooks can also be used to analyse existing notebook by adjusting the filename at the load results stage.

## Unit Testing
Unit testing notebooks vary by test type but share the same approach.
* **gtest cells**
    * Fast structural and basic correctness checks including IDs, shapes, invaritants, and basic behaviours.
* **Custom test cells**
    * Deeper verification across conditions with visible intermediate states, plot, and modifiable inputs.

## Test Notebooks
All test notebooks follow the same pattern
1. **Imports**
2. **Load evolved population**
3. **Inspect fittest networks**
4. **Run Test**
5. **Load results**
6. **Inspect fittest networks**

These notebooks are primarily used to test for overfitting. By comparing fitness from trial results to test results, the generalisability of the evolved genomes can be determined.

## Note
All notebooks are clearly documented and  designed to run top-to-bottom without modification; adjust only filenames or configuration paths when analyzing different results.