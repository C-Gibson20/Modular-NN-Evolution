# Modular-NN-Evolution
# Overview

This repository provides an **evolutionary algorithm (EA) framework** for exploring solutions to **image classification** and **multimodal maze** tasks.
It includes tools for **task execution, analysis, and visualization**.

The included code reflects my personal contribution to the project, covering my work on agent implementation, experimental trial setups, and the collection of results. The broader multimodal_mazes library, which includes additional environments and scenarios, is not included.

## Directory Structure

```
.
├── multimodal_mazes/
│   ├── agents/
│   ├── evolution/
│   ├── image_classification/
│   ├── mazes/
│   ├── plotting/
│   └── ...
└── evolution_scripts/
        └── ...
```

### `multimodal_mazes/image_classification/`

* Provides a library of **synthetic image classification tasks** that serve as benchmarks for the evolutionary algorithm.

* Implements abstract base class ImageClassification.

* **Includes**
    * Tasks which generate images with specific shapes at random positions.
        * CrossClassification task.
        * SquareClassification task.

* **Produces**
    * Synthetic images.
    * Goal locations (targets to classify/detect).
    * Distance maps (for training and evaluation).


### `multimodal_mazes/evolution/`

Contains the **core EA framework** and its components. This directory defines how candidate solutions (genomes) are represented, evolved, and evaluated within task environments.

* **agents/** 
    * Implements agents that operate within task environments.
* **evaluators/** 
    * Scripts that evaluate genome performance on tasks
    * Run evalutations each generations across the entire EA population.
* **algorithms/**
    * Implementations of the evolutionary algorithms.
    * Coordinate population management, evaluation, selection, mutation, and crossover.
* **module_banks/**
    * Classes for storing reusable module and motif banks specific to each task.
* **task_trials/**
    * Trial setup scripts defining the end-to-end experiment.
    * Specify algorithm instantiaion, hyperparamters, and task configurations.
* **testing/**
    * Scripts for testing evolved genomes on new task objects.
    * Supports evaluation of generalisation beyond training tasks.
* **genomes/**
    * Implementations of genomes used by the EA.
    * Defines how candidate solutions are encoded, mutated, and recombined.


### `evolution_scripts/`

Provides tools and notebooks for running evolutionary trials and analyzing results.

* **Experiment notebooks**
    * Launch task trial files from the `evolution/` directory.
    * Perform results analysis after each run.
    * Pickle trial results for later inspection.

* **`results/` subdirectory**
    * Stores serialized trial objects.
    * Contains experiment outputs.

* **`unit_testing/` subdirectory**
    * Contains notebooks to unit test EA variants and genome implementations.
    * GTest is used for basic validation.
    * Custom testing provides interactive and informative validation environment.

* **Plotting helpers**
    * A standalone script with reusable functions for visualizing results.

