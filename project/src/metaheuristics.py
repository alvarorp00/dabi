from abc import ABC, abstractmethod
from scipy import stats as st
from typing import List
import numpy as np
import random
import logging
import src.agents as agents
import src.utils as utils


class Metaheuristic(ABC):
    def __init__(self, search: utils.Search, *args, **kwargs):
        super().__init__()  # Invoke ABC ctor
        """
            Params:
                - search: Search type, search object
                - kwargs:
                    - population_size: int, size of the population
                    - max_trials: int, maximum number of trials without improvement  # noqa: E501

            This class is an abstract class, it must be inherited
        """
        self._search = search
        self._parameters = kwargs.copy()

        # It is not this class' responsibility to set up the parameters
        if 'population_size' not in self._parameters:
            logging.critical('Population size not specified')

    def _start_best_agent(self):
        """
            Start the best agent candidate among all agents
            in the population.

            This method is called just after the agents have been
            initialized, after running super().__init__() in the
            constructor of the child class, and before the
            optimization process starts (just before leaving the
            __init__() method in the child class).
        """
        # Select best agent among all agents
        if self.search.mode == utils.EvalMode.MINIMIZE:
            __a_idx = np.argmin(
                    [a.best_fitness for a in self.agents]
                )
        else:
            __a_idx = np.argmax(
                    [a.best_fitness for a in self.agents]
                )
        self.best_agent = self.agents[__a_idx]

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
    def agents(self):
        return self._parameters.get('agents', None)

    @agents.setter
    def agents(self, agents):
        self._parameters['agents'] = agents

    @property
    def population_size(self) -> int:
        return self._parameters.get('population_size', None)

    @property
    def max_trials(self) -> int:
        return self._parameters.get('max_trials', None)

    @max_trials.setter
    def max_trials(self, max_trials: int):
        self._parameters['max_trials'] = max_trials

    @property
    def use_max_trials(self) -> bool:
        return self._parameters.get('use_max_trials', False)

    @property
    def best_agent(self):
        return self._parameters.get('best_agent', None)

    @best_agent.setter
    def best_agent(self, best_agent):
        """
            Set the best agent found so far.
            Params:
                - best_agent: agents.Agent, best agent found so far
        """
        self._parameters['best_agent'] = best_agent

    @abstractmethod
    def optimize(self) -> bool:
        """
            Optimize an objective function using the metaheuristic.
            Returns:
                - bool, True if the best agent was updated, False otherwise
        """
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
    """
    Artificial Bee Colony metaheuristic.

    Params:
        - max_trials: int, maximum number of trials without improvement
    """

    def __init__(self, search: utils.Search, *args, **kwargs):
        super().__init__(search, *args, **kwargs)  # Invoke Metaheuristic ctor

        if 'max_trials' not in self._parameters:
            logging.critical('Max trials not specified')
        else:
            if self.max_trials <= 0:
                logging.critical('Max trials must be greater than 0')

        # Initialize employed bees
        self.employed_bees = [
            agents.Bee(
                i,
                search.space.random_bounded(search.dims),
                search
            ) for i in range(self.population_size // 2)
        ]  # Half of the bees are employed bees

        # Initialize onlooker bees with positions of employed bees and weights
        self.onlooker_bees = [
            agents.Bee(
                i,
                np.zeros(search.dims),  # It's updated later in the algorithm
                search
            ) for i in range(self.population_size // 2)
        ]  # Half of the bees are onlooker_bees

        self.scouting_bees = []  # Initially 0

        self.scouting_counter = self.population_size *\
            search.dims  # Number of dimensions

        # Initialize best agent
        self._start_best_agent()

    @property
    def employed_bees(self) -> List[agents.Bee]:
        return self._parameters.get('employed_bees', None)

    @employed_bees.setter
    def employed_bees(self, employed_bees: List[agents.Bee]):
        self._parameters['employed_bees'] = employed_bees

    @property
    def onlooker_bees(self) -> List[agents.Bee]:
        return self._parameters.get('onlooker_bees', None)

    @onlooker_bees.setter
    def onlooker_bees(self, onlooker_bees: List[agents.Bee]):
        self._parameters['onlooker_bees'] = onlooker_bees

    @property
    def scouting_bees(self) -> List[agents.Bee]:
        return self._parameters.get('scouting_bees', None)

    @scouting_bees.setter
    def scouting_bees(self, scouting_bees: List[agents.Bee]):
        self._parameters['scouting_bees'] = scouting_bees

    @property
    def bees(self):
        return self.employed_bees + self.onlooker_bees + self.scouting_bees

    @property
    def max_trials(self) -> int:
        return self._parameters.get('max_trials', None)

    # Override agents property to return the employed bees
    @property
    def agents(self) -> List[agents.Agent]:
        # Only employed bees initially, which is when this method is called
        return self.employed_bees

    def optimize(self) -> bool:
        updated = False
        # Perform a step of the metaheuristic

        # Employed bees
        for bee in self.employed_bees:
            if self.send_employee(bee):  # Send employed bee to search
                # Update best agent
                if utils.improves(self.best_agent.fitness, bee.fitness,
                                  self.search.mode):
                    self.best_agent = bee
                    updated = True
            else:
                if bee.trials >= self.max_trials:
                    # Add to scouts and remove from employed bees
                    self.scouting_bees.append(bee)
                    self.employed_bees.remove(bee)

        # Onlooker bees

        for bee in self.onlooker_bees:
            if self.send_onlooker(bee):  # Send employed bee to search
                # Update best agent
                if utils.improves(self.best_agent.fitness, bee.fitness,
                                  self.search.mode):
                    self.best_agent = bee
                    updated = True

        # Scouting bees
        if len(self.scouting_bees) > 0:  # If there are scouts
            # Select bee with the biggest number of trials
            bee = self.employed_bees[np.argmax([bee.trials for bee in
                                                self.scouting_bees])]
            # if self.send_employee(bee):  # Send employed bee to search
            #     # Update best agent
            #     if utils.improves(self.best_agent.fitness, bee.fitness,
            #                       self.search.mode):
            #         self.best_agent = bee
            #         updated = True

            # New position purely random
            bee.position = self.search.space.random_bounded(self.search.dims)
            bee.fitness = self.search.objective_function(bee.position)

            # Delete previous bee from scouts and add it to employed bees
            bee.trials = 0
            self.employed_bees.append(bee)
            self.scouting_bees.remove(bee)

        return updated

    def send_employee(self, bee) -> bool:
        """
            Send an employed bee to search for a new position.

            Params:
                - bee: agents.EmployedBee, employed bee to send

            Returns:
                - bool, True if the employed bee was updated, False otherwise
        """
        # Select a random dimension
        k_i = random.randint(0, self.search.dims - 1)

        # Select a random employed bee b'
        random_bee = self.bees
        random_bee.remove(bee)
        random_bee = random.sample(random_bee, 1)[0]

        # Generate a new position for b_i in the k_i dimension
        candidate_position = bee.position.copy()
        candidate_position[k_i] = candidate_position[k_i] +\
            random.uniform(-1, 1) * (bee.position[k_i] - random_bee.position[k_i])
        # Fix position, must be inside the search space
        candidate_position =\
            self.search.space.fix_position(candidate_position)

        # Check if the candidate position improves the fitness

        candidate_fitness =\
            self.search.objective_function(candidate_position)

        if utils.improves(
            bee.fitness,
            candidate_fitness,
            self.search.mode
        ):
            # Update b_i position and fitness
            bee.position = candidate_position
            bee.fitness = candidate_fitness

            return True
        return False

    def send_onlooker(self, bee):
        selected_id = self.select_bee()
        bee.position = self.employed_bees[selected_id].position
        bee.fitness = self.search.objective_function(bee.position)

        return self.send_employee(bee)

    def probabilities(self) -> np.ndarray:
        """
            Calculate the probability of selecting an employed bee
            in the onlooker bees phase.

            Returns:
                - np.ndarray, array of probabilities

            Notes:
                - If the goal is to maximize,
                  array is sorted in ascending order.
                - If the goal is to minimize,
                  array is sorted in descending order.
        """
        # Gather random samples from the distribution
        x = st.expon.rvs(size=len(self.employed_bees))

        # Retrieve probs of the sample by
        # calculating the probability of each
        probs = st.expon.pdf(x)

        # Normalize the probs
        probs = np.nan_to_num(probs / np.sum(probs))

        # Sort in ascending order
        probs = np.sort(probs)

        # Now, bigger probs are at the end of the array
        # and smaller probs are at the beginning of the array

        # If goal is to minimize, invert the probs
        if self.search.mode == utils.EvalMode.MINIMIZE:
            probs = np.flip(probs)

        return probs

    def select_bee(self) -> int:
        """
            Select a bee to send to the onlooker bees phase.
            
            Returns:
                - int, id of the selected bee

            Notes:
                - If the goal is to maximize,
                  bee with the biggest fitness is selected.

                - If the goal is to minimize,
                  bee with the smallest fitness is selected.
        """
        # Get weights of employed bees fitnesses
        weights = np.array([bee.fitness for bee in self.employed_bees],
                           dtype=np.float64)
        ids = np.arange(len(self.employed_bees))
        mapping = dict(zip(ids, weights))  # Map ids to fitnesses
        # Sort mapping by fitnesses in ascending order
        mapping = sorted(mapping.items(), key=lambda kv: kv[1])

        probs = self.probabilities()

        # Get a random bee given previous probabilities
        selected_id = np.random.choice(ids, p=probs)

        return selected_id

    def update_parameters(self, **kwargs):
        # Update parameters of the metaheuristic
        # Maybe sharing agents between metaheuristics is a good idea,
        # together with their positions and fitnesses, etc
        # Code goes here, this is an abstract method
        pass


class WaterCycleAlgorithm(Metaheuristic):
    def __init__(self, search: utils.Search, *args, **kwargs):
        super().__init__(search, *args, **kwargs)
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
        super().__init__(search, *args, **kwargs)
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
    """
    Differential Evolution metaheuristic.
    Params:
        - search: Search type, search object
        - kwargs:
            - crossover_rate: float, crossover rate
            - diff_weight: float, differential weight
    """
    def __init__(self, search: utils.Search, *args, **kwargs):
        super().__init__(search, *args, **kwargs)
        # Check specific parameters for Differential Evolution
        if 'crossover_rate' not in self.parameters:
            return ValueError('Crossover rate not specified')

        if 'diff_weight' not in self.parameters:
            return ValueError('Differential weight not specified')

        # Initialize agents
        self.agents = [
            agents.Agent(
                i,
                search.space.random_bounded(search.dims),
                search
            ) for i in range(self.population_size)
        ]

        # Initialize best agent
        self._start_best_agent()

    @property
    def crossover_rate(self) -> float:
        return self.parameters.get('crossover_rate', None)

    @property
    def diff_weight(self) -> float:
        return self.parameters.get('diff_weight', None)

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
                return True  # New global best agent found
        return False

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
