import argparse
import logging
import os
import sys

# Custom imports
import src.config as config
import src.metaheuristics as metaheuristics
import src.synergy as synergy
import src.plot as plot

import src.utils as utils

import src.config as config


def run():
    # ms = [m for m in metaheuristics.__dict__.values()
    #         if isinstance(m, type) and issubclass(m, metaheuristics.Metaheuristic) and m is not metaheuristics.Metaheuristic]
    ms = [
        metaheuristics.DifferentialEvolution(search=config.search)
        for _ in range(5)
    ]
    # synergy_boost = synergy.SynergyBoost(ms)
    # synergy_boost.optimize(config.objective_function, config.initial_solution)
    for m in ms:
        print(m)
    pass


def parse():
    """
    Pending to be implemented
    """
    # TODO --> Test before with config and maybe after with argparse
    return None


if __name__ == '__main__':
    # args = parse()
    run()
