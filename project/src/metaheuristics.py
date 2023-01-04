from abc import ABC, abstractmethod
from typing import List
import numpy as np
import random
import logging
import src.utils as utils
import src.macros as macros


class Metaheuristic(ABC):
    def __init__(self, search: utils.Search, *args, **kwargs):
        """
            Params:
                - search: Search type, search object
                - kwargs:
                    - agents: List[utils.Agent], list of agents
                    - population_size: int, size of the population
        """
        self._search = search
        self._parameters = kwargs.copy()

        # Check general parameters
        # It is not this class' responsibility to set up the parameters
        if 'agents' not in self._parameters:
            logging.critical('Agents not specified')
        if 'population' not in self._parameters:
            logging.critical('Population size not specified')

        # Initialize best agent with the first agent
        self.__best_agent = self._parameters['agents'][0]

    @property
    def search(self) -> utils.Search:
        return self._search

    @search.setter
    def search(self, search: utils.Search):
        self._search = search

    @property
    def parameters(self) -> dict:
        return self._parameters

    @property
    def agents(self) -> List[utils.Agent]:
        return self._parameters.get('agents', None)

    @property
    def population_size(self) -> int:
        return self._parameters.get('population_size', None)

    @property
    def best_agent(self) -> utils.Agent:
        return self._parameters.get('best_agent', None)

    @best_agent.setter
    def __best_agent(self, best_agent: utils.Agent):
        """
            Set the best agent found so far.
            Params:
                - best_agent: utils.Agent, best agent found so far
        """
        self._parameters['best_agent'] = best_agent

    @abstractmethod
    def optimize(self):
        # code for optimizing an objective function using the metaheuristic
        # goes here, this is an abstract method
        pass

    @abstractmethod
    def update_parameters(self, **kwargs):
        # Update parameters of the metaheuristic
        # Maybe sharing agents between metaheuristics is a good idea,
        # together with their positions and fitnesses, etc
        # Code goes here, this is an abstract method
        pass

    def __str__(self):
        return self.__class__.__name__


class ArtificialBeeColony(Metaheuristic):
    def __init__(self, search: utils.Search, *args, **kwargs):
        super().__init__(search, args, kwargs)
        # specific initialization code for the Artificial Bee Colony
        # goes here
        pass

    def optimize(self):
        # specific code for optimizing an objective function using
        # the Artificial Bee Colony algorithm goes here
        pass

    def update_parameters(self, **kwargs):
        # Update parameters of the metaheuristic
        # Maybe sharing agents between metaheuristics is a good idea,
        # together with their positions and fitnesses, etc
        # Code goes here, this is an abstract method
        pass


class WaterCycleAlgorithm(Metaheuristic):
    def __init__(self, search: utils.Search, *args, **kwargs):
        super().__init__(search, args, kwargs)
        # specific initialization code for the Water Cycle Algorithm
        # goes here
        pass

    def optimize(self):
        # specific code for optimizing an objective function using
        # the Water Cycle Algorithm goes here
        pass

    def update_parameters(self, **kwargs):
        # Update parameters of the metaheuristic
        # Maybe sharing agents between metaheuristics is a good idea,
        # together with their positions and fitnesses, etc
        # Code goes here, this is an abstract method
        pass


class ParticleSwarmOptimization(Metaheuristic):
    def __init__(self, search: utils.Search, *args, **kwargs):
        super().__init__(search, args, kwargs)
        # specific initialization code for the Particle Swarm Optimization
        # goes here
        pass

    def optimize(self):
        # specific code for optimizing an objective function using
        # the Particle Swarm Optimization algorithm goes here
        pass

    def update_parameters(self, **kwargs):
        # Update parameters of the metaheuristic
        # Maybe sharing agents between metaheuristics is a good idea,
        # together with their positions and fitnesses, etc
        # Code goes here, this is an abstract method
        pass


class DifferentialEvolution(Metaheuristic):
    def __init__(self, search: utils.Search, *args, **kwargs):
        super().__init__(search, args, kwargs)
        # Check specific parameters for Differential Evolution
        if 'crossover_rate' not in self._parameters:
            return ValueError('Crossover rate not specified')

        if 'diff_weight' not in self._parameters:
            return ValueError('Differential weight not specified')

    @property
    def crossover_rate(self) -> float:
        return self._parameters.get('crossover_rate', None)

    @property
    def diff_weight(self) -> float:
        return self._parameters.get('diff_weight', None)

    def optimize(self):
        """
        Performs a single iteration of the Differential Evolution algorithm.
        """
        for agent in self.agents:
            # Select 3 random agents
            random_agents = self.agents.copy()
            random_agents.remove(agent)
            random_agents = random.sample(random_agents, 3)

            if len(random_agents) != 3:
                logging.critical('Not enough agents to perform the\
                                 Differential Evolution algorithm')

            # Copy current agent's position
            new_position = agent.position.copy()
            
            # Select i-th dimension to mutate
            i = random.randint(0, len(new_position) - 1)
            new_position[i] = random_agents[0].position[i] + \
                self.diff_weight *\
                (random_agents[1].position[i] - random_agents[2].position[i])

            # Select each j-th dimension to mutate with probability
            for j in range(len(new_position)):
                if j != i and np.random.uniform(low=0, high=1) <\
                              self.crossover_rate:
                    new_position[j] = random_agents[0].position[j] + \
                        self.diff_weight *\
                        (random_agents[1].position[j] -
                         random_agents[2].position[j])

            # Check if new position is within the search space
            new_position = self.search.space.fix_position(new_position)

            # Evaluate new position
            new_fitness = self.search.objective_function(new_position)

            # Update agent's position and fitness if new fitness is better
            if utils.improves(
                agent.fitness,
                new_fitness,
                self.search.mode
            ):
                agent.position = new_position
                agent.fitness = new_fitness
            
            # Update global best agent if new fitness is better
            if utils.improves(
                self.best_agent.fitness,
                new_fitness,
                self.search.mode
            ):
                self.best_agent = agent

    def update_parameters(self, **kwargs):
        # Update parameters of the metaheuristic
        # Maybe sharing agents between metaheuristics is a good idea,
        # together with their positions and fitnesses, etc
        # Code goes here, this is an abstract method
        pass


class FireflyAlgorithm(Metaheuristic):
    def __init__(self, search: utils.Search, *args, **kwargs):
        super().__init__(search, args, kwargs)
        # specific initialization code for the Firefly Algorithm
        # goes here
        pass

    def optimize(self):
        # specific code for optimizing an objective function using
        # the Firefly Algorithm goes here
        pass

    def update_parameters(self, **kwargs):
        # Update parameters of the metaheuristic
        # Maybe sharing agents between metaheuristics is a good idea,
        # together with their positions and fitnesses, etc
        # Code goes here, this is an abstract method
        pass
