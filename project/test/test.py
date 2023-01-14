import sys
import os

sys.path.insert(0, os.getcwd())

import src.config as config
import src.utils as utils
import src.agents as agents
import src.synergy as synergy
from src.metaheuristics import\
    ArtificialBeeColony, DifferentialEvolution,\
    ParticleSwarmOptimization, WaterCycleAlgorithm


import numpy as np

############################################


def new_differential_evolution():
    """
    Instantiates a new Differential Evolution algorithm with a random population.
    """

    crossover_rate = np.random.uniform()
    differential_weight = np.random.uniform()

    params = {
        'population_size': config.population_size,
        'diff_weight': differential_weight,
        'crossover_rate': crossover_rate
    }

    de = DifferentialEvolution(
        search=config.search,
        **params
    )

    return de


def new_artificial_bee_colony():
    """
    Instantiates a new Artificial Bee Colony algorithm with a random population.
    """

    params = {
        'population_size': config.population_size,
        'max_trials': config.population_size * config.search.dims,
    }

    abc = ArtificialBeeColony(
        search=config.search,
        **params
    )

    return abc


def new_particle_swarm_optimization():
    """
    Instantiates a new Particle Swarm Optimization algorithm with a random population.
    """

    params = {
        'population_size': config.population_size,
        'inertia': 0.5,
        'cognitive': 0.5,
        'social': 0.5,
    }

    pso = ParticleSwarmOptimization(
        search=config.search,
        **params
    )

    return pso


# ms = [new_differential_evolution() for _ in range(0, 10)]  # 10 metaheuristics to be combined of DE
# ms = [new_differential_evolution(), new_artificial_bee_colony()]  # 2 metaheuristics to be combined of DE and ABC
# ms = [new_artificial_bee_colony() for _ in range(0, 10)]  # 10 metaheuristics to be combined of ABC
ms = [new_particle_swarm_optimization() for _ in range(0, 10)]  # 10 metaheuristics to be combined of PSO

params = {
    'runs': config.runs,
    'iterations': config.iterations,
    'convergence_criteria': config.convergence_criteria,
}

synergy_boost = synergy.SynergyBoost(metaheuristics=ms, search=config.search, **params)
stats = synergy_boost.optimize()
print(f'Best agent: {synergy_boost.best_agent} @ Fitness: {synergy_boost.best_agent.fitness} @ Position: {synergy_boost.best_agent.position}')

print("Stats: ", stats)

# params = {
#     'population_size': 30,  # 20 streams + 9 rivers + 1 sea
#     'n_sr': 10,  # 9 rivers + 1 sea,
#     'd_max': 0.5,  # maximum distance between sea and river before evaporation
# }

# wc = WaterCycleAlgorithm(search=config.search, **params)
