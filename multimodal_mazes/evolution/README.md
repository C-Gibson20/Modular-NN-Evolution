# Evolution

This directory contains the **core evolutionary algorithm (EA) framework**.
It defines how candidate solutions (genomes) are **represented, evolved, evaluated, and tested** within task environments.

The framework is organized into modular components, each in its own subdirectory.
For detailed information and usage examples, see the README inside each subdirectory.


## Directory Structure

```
evolution/
├── agents/
├── evaluators/
├── algorithms/
├── module_banks/
├── task_trials/
├── testing/
└── genomes/
```

### `agents/`

Implements agents that operate in task environments.

* Responsible for executing trials and interacting with tasks.
* [See `agents/README.md`](agents/README.md) for details.

### `evaluators/`

Evaluates genome performance per generation.

* Runs evaluations across all genomes in the population.
* Defines metrics and success criteria.
* [See `evaluators/README.md`](evaluators/README.md).

### `algorithms/`

Implements the evolutionary algorithms.

* Handles population initialization, selection, mutation, and crossover.
* Coordinates evaluation and generational updates.
* [See `evolution_managers/README.md`](evolution_managers/README.md).

### `module_banks/`

Provides reusable module banks for each task.

* Store task-specific building blocks that genomes can compose.
* Enable modularity and motif reuse across tasks.
* [See `module_banks/README.md`](module_banks/README.md).

### `task_trials/`

Trial setup scripts for experiments.

* Define task, algorithm instantiation, and hyperparameters.
* Encapsulate end-to-end configurations for reproducible runs.
* Each task trial has a correspnding script with the same name in the `scripts/evolutionary_scripts` directory.
* [See `task_trials/README.md`](task_trials/README.md).

### `testing/`

Tests evolved genomes on **new task objects**.
* Define task, algorithm instantiation, and hyperparameters.
* Encapsulate end-to-end configurations for reproducible runs.
* Each test trial has a correspnding script with the same name in the `scripts/evolutionary_scripts` directory.
* [See `testing/README.md`](testing/README.md).

### `genomes/`

Implements genomes and network topologies.

* Defines representation of candidate solutions.
* Provides mutation and crossover logic.
* [See `topology_evolution/README.md`](topology_evolution/README.md).

---

## Component Interaction

At a high level, the evolutionary process flows as follows:

1. **Task trials** set up an experiment (task, parameters, algorithm).
2. **Algorithms** run the EA process.
3. **Evaluators** score genomes each generation.
4. **Agents** execute the task using genomes.
5. **Module banks** provide reusable building blocks.
6. **Genomes** defines how networks are structured and mutated.
7. **Testing** evaluates evolved solutions on new/unseen tasks.