import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def compute_type_ratios(results):
    """
    Compute the type ratios from the results.
    Arguments:
        results (dict): the results from the evolution process.
    Returns:
        tuple: generations and type_ratios
    """
    # Initialise generations and type ratios
    types = results[0][1].keys()
    generations = {type: results[0][0] for type in types}
    type_ratios = {type: np.zeros(len(results[0][1][type])) for type in types}

    # Collect averagetype ratios
    for i in results.keys():
        for type in types:
            type_ratios[type] += results[i][1][type]
            type_ratios[type] /= len(results)

    return generations, type_ratios

def plot_fitness_over_generations(results):
    """
    Plot the fitness of the fittest genome over generations.
    Arguments:
        results (dict): the results from the evolution process.
    """
    # Initialise fitness and generations
    types = results.keys()
    average_fitness = {type: np.zeros(len(results[type][0][1])) for type in types} 
    generations = {type: results[type][0][0] for type in types}

    # Collect average fitness
    for type in types:
        for i in results[type].keys():
            fitness = results[type][i][1]
            average_fitness[type] += fitness
        average_fitness[type] /= len(results[type])

    # Plot average fitness
    cmap = cm.get_cmap('viridis', len(types))
    type_colors = {type: cmap(i) for i, type in enumerate(types)}
    for type in types:
        generation, fitness = generations[type], average_fitness[type]
        plt.plot(generation, fitness, label=type, color=type_colors[type])

    plt.title('Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

def plot_motif_distribution_over_generations(results):
    """
    Plot the motif distribution over generations.
    Arguments:
        results (dict): the results from the evolution process.
    """
    # generations, mot_ratios = results
    types = results[0][1].keys()
    generations = {type: results[0][0] for type in types}
    average_mot_ratios = {type: {i: np.zeros(len(results[0][1][type][i])) for i in range(len(results[0][1][type]))} for type in types}

    # Collect average motif ratios
    for i in results.keys():
        for type in types:
            for id in average_mot_ratios[type].keys():
                average_mot_ratios[type][id] += results[i][1][type][id]
                average_mot_ratios[type][id] /= len(results)

    # Plot motif ratios
    cmap = cm.get_cmap('viridis', len(average_mot_ratios))
    type_colors = {type: cmap(i) for i, type in enumerate(types)}
    for type in types:
        for id, ratios in average_mot_ratios[type].items():
            label = type if id == 0 else None
            plt.plot(generations[type], ratios, label=label, color=type_colors[type])

    plt.title('Module Distribution Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Module Ratio')
    plt.legend()
    plt.show()

def plot_type_distribution_over_generations(results):
    """
    Plot the type distribution over generations.
    Arguments:
        results (dict): the results from the evolution process.
    """
    types = results[0][1].keys()
    generations, type_ratios = compute_type_ratios(results)

    cmap = cm.get_cmap('viridis', len(types))
    type_colors = {type: cmap(i) for i, type in enumerate(types)}
    for type in types:
        plt.plot(generations[type], type_ratios[type], label=type, color=type_colors[type])

    plt.title('Type Distribution Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Type Ratio')
    plt.legend()
    plt.show()

def plot_property_distribution_over_generations(results):
    """
    Plot the property distribution over generations.
    Arguments:
        results (dict): the results from the evolution process.
    """
    generations, type_ratios = compute_type_ratios(results)
    n_types = len(type_ratios)
    property_ratios = {' ': [], 'R': [], 'T': [], 'RT': []}

    # Collect ratios for motif properties
    for type in type_ratios:
        prop = type.split('.')[-1] if '.' in type else ' '
        property_ratios[prop] += type_ratios[type]

    # Normalize property ratios
    for prop in property_ratios:
        if property_ratios[prop] == []:
            continue
        property_ratios[prop] /= n_types

    # Plot property ratios
    cmap = cm.get_cmap('viridis', len(property_ratios))
    prop_colors = {prop: cmap(i) for i, prop in enumerate(property_ratios.keys())}
    for prop in property_ratios:
        plt.plot(generations, property_ratios[prop], label=prop, color=prop_colors[prop])

    plt.title('Property Distribution Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Property Ratio')
    plt.legend()
    plt.show()