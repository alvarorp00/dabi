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


def new_water_cycle_algorithm():
    """
    Instantiates a new Water Cycle Algorithm.
    """

    params = {
        'population_size': 30,  # 20 streams + 9 rivers + 1 sea
        'n_sr': 5,  # 9 rivers + 1 sea,
        'd_max': 0.5,  # maximum distance between sea and river before evaporation
        'd_max_decay': 1e-3,  # decay rate of d_max  --> d_max_{t+1} = d_max_{t} - d_max_decay
    }

    wca = WaterCycleAlgorithm(search=config.search, **params)

    return wca

############################################
# MAIN                                     #
############################################

ms = [new_differential_evolution(), new_artificial_bee_colony(), new_particle_swarm_optimization(), new_water_cycle_algorithm()]

params = {
    'runs': config.runs,
    'iterations': config.iterations,
    'convergence_criteria': config.convergence_criteria,
}

synergy_boost = synergy.SynergyBoost(metaheuristics=ms, search=config.search, **params)
stats = synergy_boost.optimize()
print(f'Best agent: {synergy_boost.best_agent} @ Fitness: {synergy_boost.best_agent.fitness} @ Position: {synergy_boost.best_agent.position}')

print("#################\nStats\n#################")

print("Runs: ", stats["runs"])
print("Converged: ", stats["converged"])

# print("Trace:")

# for i in range(0, len(stats["trace"])):
#     print(f'{stats["trace"][i]} @ {stats["trace"][i].owner} @ fitness: {stats["trace"][i].fitness} @ position: {stats["trace"][i].position}')

print("#################\nTrace\n#################")

for trace in stats["trace"].trace:
    # print(trace)
    print(f"trace @ {trace['name']}:{trace['owner']} @ fitness: {trace['fitness']} @ run: {trace['run']} @ iteration: {trace['iteration']}")
