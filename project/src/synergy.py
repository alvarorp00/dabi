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
        - kwargs: dict, dictionary of parameters
            - runs: int, number of runs to perform,
                    namely the number of times the metaheuristics
                    will be executed
            - iterations: int, number of iterations to perform, namely
                            the number of times each metaheuristic will
                            execute per run
            - convergence_criteria: float, convergence criteria

    This class assumes that all metaheuristics had initialized their
    agents and population size, and that they already have a
    best agent candidate.
    """
    def __init__(self, metaheuristics, search, *args, **kwargs):
        self._metaheuristics = metaheuristics
        self._search = search

        self._parameters = kwargs.copy()

        if 'runs' not in self._parameters:
            logging.critical('Runs not specified')

        if 'iterations' not in self._parameters:
            logging.critical('Iterations not specified')

        if 'convergence_criteria' not in self._parameters:
            logging.info('Convergence criteria not specified, not using it')
            self.convergence_criteria = 0.0

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

    @property
    def runs(self) -> int:
        return self.parameters.get('runs', None)

    @property
    def iterations(self) -> int:
        return self.parameters.get('iterations', None)

    @property
    def convergence_criteria(self) -> float:
        return self.parameters.get('convergence_criteria', None)

    @convergence_criteria.setter
    def convergence_criteria(self, convergence_criteria: float):
        self._parameters['convergence_criteria'] = convergence_criteria

    def optimize(self) -> dict[str, any]:
        """
        This method performs the optimization of the metaheuristics
        and returns the best agent found.

        Returns:
            - stats: dict[str, any], dictionary with the following keys:
                - runs: int, number of runs performed
                - converged: bool, whether the best agent has converged
                - trace: List[agents.Agent], list of best agents found
        """
        stats = {'runs': 0, 'converged': 'False', 'trace': utils.Trace()}
        for run in range(self.runs):
            stats['runs'] += 1
            # Extract the best agent before the optimization
            # and check if it has improved after the optimization
            best_agent_before = self.best_agent
            for m in self.metaheuristics:
                for iteration in range(self.iterations):
                    # Check if the metaheuristic has improved its best agent
                    if m.optimize():
                        if utils.improves(
                            self.best_agent.fitness,
                            m.best_agent.fitness,
                            self.search.mode
                        ):
                            self.best_agent = m.best_agent
                            # Add to trace the best fitness found
                            stats['trace'].add(
                                self.best_agent,
                                iteration,
                                run
                            )
                            # Update the best fitness of all metaheuristics
                            for m in self.metaheuristics:
                                m.update_parameters(best_agent=self.best_agent)
            # Check if the best agent has converged
            if utils.converged(
                self.best_agent.fitness,
                best_agent_before.fitness,
                self.convergence_criteria
            ):
                stats['converged'] = 'True'
                break
        return stats
