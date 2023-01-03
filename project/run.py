import argparse
import logging
import os
import sys

# Custom imports
import src.macros as macros
import src.metaheuristics as metaheuristics
import src.synergy as synergy
import src.plot as plot

def run():
    ms = [m for m in metaheuristics.__dict__.values()
            if isinstance(m, type) and issubclass(m, metaheuristics.Metaheuristic) and m is not metaheuristics.Metaheuristic]
    # synergy_boos = synergy.SynergyBoost()
    print(ms)
    pass

"""
Pending to be implemented
"""
def parse():
    # TODO --> Test before with macros and maybe after with argparse
    return None

if __name__ == '__main__':
    # args = parse()
    run()