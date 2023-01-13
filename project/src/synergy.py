from typing import List
import logging
import numpy as np

from src.metaheuristics import Metaheuristic
import src.utils as utils
import src.agents as agents


class SynergyBoost:
    """
    This class implements the synergy boost metaheuristic.
    It takes a list of metaheuristics (sees them as black boxes) and
    returns the best solution found by any of the metaheuristics.

    The idea is that this class will be used to share knowledge between
    metaheuristics, so that they can learn from each other.

    Metaheuristics are expected to have been initialized previously.

    Params:
        - metaheuristics: List[Metaheuristic], list of metaheuristics
        - search: utils.Search, search class object

    This class assumes that all metaheuristics had initialized their
    agents and population size, and that they already have a
    best agent candidate.
    """
    def __init__(self, metaheuristics, search, *args, **kwargs):
        self._metaheuristics = metaheuristics
        self._search = search

        self._parameters = kwargs.copy()

        if 'iterations' not in self._parameters:
            logging.critical('Iterations not specified')

        # Initialize best agent with the best agent
        # among all metaheuristics
        if search.mode == utils.EvalMode.MINIMIZE:
            __m_idx = np.argmin(
                    [m.best_agent.best_fitness for m in self.metaheuristics]
                )
        else:
            __m_idx = np.argmax(
                    [m.best_agent.best_fitness for m in self.metaheuristics]
                )
        self.best_agent = self.metaheuristics[__m_idx].best_agent

    @property
    def metaheuristics(self):
        return self._metaheuristics

    @metaheuristics.setter
    def metaheuristics(self, metaheuristics: List[Metaheuristic]):
        self._metaheuristics = metaheuristics

    @property
    def search(self):
        return self._search

    @search.setter
    def search(self, search):
        self._search = search

    @property
    def best_agent(self) -> agents.Agent:
        return self._best_agent

    @best_agent.setter
    def best_agent(self, best_agent: agents.Agent):
        self._best_agent = best_agent

    @property
    def parameters(self):
        return self._parameters

    def optimize(self):
        for _ in range(self.parameters.get('iterations')):
            for m in self.metaheuristics:
                if m.optimize():
                    if utils.improves(
                        self.best_agent.fitness,
                        m.best_agent.fitness,
                        self.search.mode
                    ):
                        self.best_agent = m.best_agent
