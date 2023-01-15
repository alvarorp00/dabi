from abc import ABC, abstractmethod
import copy
from scipy import stats as st
from typing import List
import numpy as np
import random
import logging
import src.agents as agents_module
import src.utils as utils


class Metaheuristic(ABC):
    def __init__(self, search: utils.Search, *args, **kwargs):
        super().__init__()  # Invoke ABC ctor
        """
            Params:
                - search: Search type, search object
                - kwargs:
                    - population_size: int, size of the population

            This class is an abstract class, it must be inherited
        """
        self._search = search
        self._parameters = kwargs.copy()

        # It is not this class' responsibility to set up the parameters
        if 'population_size' not in self._parameters:
            logging.critical('Population size not specified')

    def _start_best_agent(self, population: List[agents_module.Agent] = None):
        """
            Start the best agent candidate among all agents
            in the population.

            This method must be called during the __init__ method
            of the child class, after the population is created.

            Params:
                - population: list of agents, population of agents;
                                if None, the population is taken from
                                self.agents
        """
        if population is None:
            population = self.agents
        # Select best agent among all agents
        if self.search.mode == utils.EvalMode.MINIMIZE:
            __a_idx = np.argmin(
                    [a.best_fitness for a in population]
                )
        else:
            __a_idx = np.argmax(
                    [a.best_fitness for a in population]
                )
        self.best_agent = copy.deepcopy(population[__a_idx])

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
    def best_agent(self) -> agents_module.Agent:
        return self._parameters.get('best_agent', None)

    @best_agent.setter
    def best_agent(self, best_agent):
        """
            Set the best agent found so far.
            Params:
                - best_agent: agents_module.Agent, best agent found so far

            Don't call this method from an outer instance, as some important
            information could be lost. Instead, call the update_parameters
            instead.
        """
        self._parameters['best_agent'] = best_agent
    @property
    def name(self):
        if 'name' in self._parameters:
            return self._parameters['name']
        else:
            return self.__class__.__name__

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

    def update_parameters(self, **kwargs):
        """
            Update parameters of the metaheuristic.

            Any child class can override this method and do its
            own parameter update. This method will update only the
            common parameters.

            Currently only supports best agent update,
            but it can be extended to support other parameters.

            Params:
                - kwargs: dict, parameters to update
                    - best_agent: agents_module.Agent, best agent found so far
        """

        # Update best agent
        if 'best_agent' in kwargs:
            self.best_agent.position = kwargs['best_agent'].position
            self.best_agent.fitness = kwargs['best_agent'].fitness

    @staticmethod
    def sort_population(population: List[agents_module.Agent],
                        mode: utils.EvalMode):
        """
            Sort population by fitness, in ascending order if
            the search mode is MINIMIZE, in descending order
            if the search mode is MAXIMIZE.

            Params:
                - population: list of agents, population of agents

            Returns:
                - list of agents, sorted population
        """
        # If the goal is to maximize, reverse the mapping
        reverse = mode == utils.EvalMode.MAXIMIZE

        # Sort rs set by fitness in ascending order
        return sorted(
            population, key=lambda river: river.fitness, reverse=reverse
        )

    def __str__(self):
        return self.__class__.__name__


class ArtificialBeeColony(Metaheuristic):
    """
    Artificial Bee Colony metaheuristic.

    Params:
        - max_trials: int, maximum number of trials without improvement

    References:
        - https://www.semanticscholar.org/paper/AN-IDEA-BASED-ON-HONEY-BEE-SWARM-FOR-NUMERICAL-Karaboga/cf20e34a1402a115523910d2a4243929f6704db1  # noqa: E501
    """

    def __init__(self, search: utils.Search, *args, **kwargs):
        super().__init__(search, *args, **kwargs)  # Invoke Metaheuristic ctor

        if 'max_trials' not in self._parameters:
            logging.critical('Max trials not specified')
        else:
            if self.max_trials <= 0:
                logging.critical('Max trials must be greater than 0')

        # Initialize employed bees
        self.employed_bees = set([
                agents_module.Bee(
                    i,
                    search.space.random_bounded(search.dims),
                    search,
                    owner=self.name
                ) for i in range(self.population_size // 2)
            ]  # Half of the bees are employed bees
        )

        # Initialize onlooker bees with positions of employed bees and weights
        self.onlooker_bees = set([
                agents_module.Bee(
                    i + self.population_size // 2,  # do not overlap ids
                    np.zeros(search.dims),  # Updated later in the algorithm
                    search,
                    owner=self.name
                ) for i in range(self.population_size // 2)
            ]  # Half of the bees are onlooker_bees
        )

        self.scouting_bees = set()  # Initially 0

        # Initialize bees dict
        ids = np.arange(self.population_size)
        bees = list(self.employed_bees | self.onlooker_bees)
        self.bees = dict(zip(ids, bees))

        # Initialize best agent
        self._start_best_agent()

    @property
    def employed_bees(self) -> set[agents_module.Bee]:
        return self._parameters.get('employed_bees', None)

    @employed_bees.setter
    def employed_bees(self, employed_bees: set[agents_module.Bee]):
        self._parameters['employed_bees'] = employed_bees

    @property
    def onlooker_bees(self) -> set[agents_module.Bee]:
        return self._parameters.get('onlooker_bees', None)

    @onlooker_bees.setter
    def onlooker_bees(self, onlooker_bees: set[agents_module.Bee]):
        self._parameters['onlooker_bees'] = onlooker_bees

    @property
    def scouting_bees(self) -> set[agents_module.Bee]:
        return self._parameters.get('scouting_bees', None)

    @scouting_bees.setter
    def scouting_bees(self, scouting_bees: set[agents_module.Bee]):
        self._parameters['scouting_bees'] = scouting_bees

    @property
    def bees(self) -> dict[int, agents_module.Bee]:
        return self._parameters.get('bees', None)

    @bees.setter
    def bees(self, bees: dict[int, agents_module.Bee]):
        self._parameters['bees'] = bees

    @property
    def max_trials(self) -> int:
        return self._parameters.get('max_trials', None)

    # Override agents property to return the employed bees
    @property
    def agents(self) -> List[agents_module.Agent]:
        # Only employed bees initially, which is when this method is called
        return list(self.employed_bees)

    def optimize(self) -> bool:
        updated = False
        # Perform a step of the metaheuristic

        # Employed bees
        removals = set()
        for bee in self.employed_bees:
            if self.send_employee(bee):  # Send employed bee to search
                # Update best agent
                if utils.improves(self.best_agent.fitness, bee.fitness,
                                  self.search.mode):
                    self.best_agent = copy.deepcopy(bee)
                    updated = True
            else:
                # Increase trials
                bee.trials += 1
                # Check if it's the only bee left in the employed bees
                if len(self.employed_bees) == 1:
                    # If so, force scout immediately
                    self.send_scout(bee)
                if bee.trials >= self.max_trials:
                    # Add to scouts and remove from employed bees
                    self.scouting_bees.add(bee)
                    removals.add(bee)
        # Remove bees from employed bees
        self.employed_bees -= removals

        # Onlooker bees

        for bee in self.onlooker_bees:
            if self.send_onlooker(bee):  # Send employed bee to search
                # Update best agent
                if utils.improves(self.best_agent.fitness, bee.fitness,
                                  self.search.mode):
                    self.best_agent = copy.deepcopy(bee)
                    updated = True

        # Scouting bees
        if len(self.scouting_bees) > 0:  # If there are scouts
            # Select bee with the biggest number of trials
            bee = self.biggest_trial_bee()

            # New position purely random
            self.send_scout(bee)

            # Delete previous bee from scouts and add it to employed bees
            self.employed_bees.add(bee)
            self.scouting_bees.remove(bee)

        return updated

    def send_employee(self, bee) -> bool:
        """
            Send an employed bee to search for a new position.

            Params:
                - bee: agents_module.Bee, employed bee to send

            Returns:
                - bool, True if the employed bee was updated, False otherwise
        """
        # Select a random dimension
        k_i = random.randint(0, self.search.dims - 1)

        # Select a random employed bee b'
        random_bee = self.employed_bees.copy()
        if bee in random_bee:
            random_bee.remove(bee)
        random_bee = random.sample(random_bee, 1)[0]

        # Generate a new position for b_i in the k_i dimension
        candidate_position = bee.position.copy()
        candidate_position[k_i] = candidate_position[k_i] +\
            random.uniform(-1, 1) *\
            (bee.position[k_i] - random_bee.position[k_i])
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

            # Update best known position of the bee
            bee.best_position = bee.position
            # Update best fitness of the bee
            bee.best_fitness = bee.fitness

            return True
        return False

    def send_onlooker(self, bee):
        selected_id = self.select_bee()
        selected_bee = self.get_bee(selected_id)
        bee.position = selected_bee.position.copy()
        bee.fitness = selected_bee.fitness

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
        # Ids of employed bees are not necessarily ordered
        ids = np.array([bee.id for bee in self.employed_bees])  # Get ids
        mapping = dict(zip(ids, weights))  # Map ids to fitnesses
        # Sort mapping by fitnesses in ascending order
        mapping = sorted(mapping.items(), key=lambda kv: kv[1])

        probs = self.probabilities()

        # Get a random bee given previous probabilities
        selected_id = np.random.choice(ids, p=probs)

        return selected_id

    def biggest_trial_bee(self) -> agents_module.Bee:
        """
            Get the biggest number of trials of the employed bees.

            Returns:
                - bee, bee with the biggest number of trials
        """
        # Get the number of trials of the scouting bees
        trials = np.array([bee.trials for bee in self.scouting_bees])

        # Get the ids of the scouting bees
        ids = np.array([bee.id for bee in self.scouting_bees])

        # Map ids to trials
        mapping = dict(zip(ids, trials))
        # Sort mapping by trials in ascending order
        mapping = sorted(mapping.items(), key=lambda kv: kv[1])

        # Return the bee with the biggest number of trials
        bee = self.get_bee(
            mapping[-1][0]  # [0] because mapping is a list of tuples
        )

        return bee

    def get_bee(self, id: int) -> agents_module.Bee:
        """
            Get an employed bee given its id.

            Params:
                - id: int, id of the bee to get

            Returns:
                - agents_module.Bee, bee with the given id
        """
        return self.bees[id]  # Get bee from the dictionary

    def send_scout(self, bee: agents_module.Bee):
        """
            Force the given bee to scout.

            Use this method when the bee has reached the maximum
            number of trials and there are no more bees in the
            employed bees set (skip steps), or whenever you want
            to force a bee to scout.

            It's important to select a bigger number for the population
            or a bigger number for the maximum number of trials.
        """
        bee.position = self.search.space.random_bounded(dims=self.search.dims)
        bee.fitness = self.search.objective_function(bee.position)
        bee.trials = 0

    def update_parameters(self, **kwargs):
        # Update parameters of the metaheuristic
        # Maybe sharing agents between metaheuristics is a good idea,
        # together with their positions and fitnesses, etc
        # Code goes here, this is an abstract method
        pass


class WaterCycleAlgorithm(Metaheuristic):
    """
        Water Cycle Algorithm

        Params:
            - kwargs: dict, dictionary of parameters
                - n_sr: int, sum of rivers & seas, namely
                        the number of rivers + 1 (sea)
                - d_max: float, maximum distance between sea
                            and rivers before evaporation occurs

        References:
            - https://arxiv.org/pdf/1909.08800.pdf

        Notes:
            - From the total population of size N:
                - 1 sea
                - Nrs - 1 rivers
                - N - Nrs streams
    """
    def __init__(self, search: utils.Search, *args, **kwargs):
        super().__init__(search, *args, **kwargs)

        # Check specific parameters
        if 'n_sr' not in kwargs:
            logging.critical(
                "Missing parameter 'n_sr' (sum of rivers & seas) "
                "for Water Cycle Algorithm"
            )
        elif self.n_sr >= self.population_size:
            logging.critical(
                "The number of rivers & seas must be less than the "
                "population size"
            )

        if 'd_max' not in kwargs:
            logging.critical(
                "Missing parameter 'd_max' (maximum distance between sea "
                "and rivers before evaporation occurs) for Water Cycle "
                "Algorithm"
            )

        if 'd_max_decay' not in kwargs:
            logging.critical("Missing parameter 'd_max_decay' for Water Cycle ")

        if 'c_constant' not in kwargs:
            logging.info("Using random value for 'c_constant' parameter [>1,"
                         "best 1 < c < 2]")
            self._parameters['c_constant'] = 2

        # Create agents, namely rivers, streams and sea
        population = [
            agents_module.River(
                i,
                self.search.space.random_bounded(dims=self.search.dims),
                self.search,
                owner=self.name
            ) for i in range(0, self.population_size)
        ]

        # # This will set the sea automatically
        # self._start_best_agent(population=population)

        # rs = set(population) - set([self.sea])  # Rivers & streams

        population = Metaheuristic.sort_population(
            population=population,
            mode=self.search.mode
        )

        self.sea = population[0]

        # Set rivers
        rivers = population[1:self.n_sr]
        self.rivers = dict(zip(
            [r.id for r in rivers],
            rivers
        ))

        # Set streams
        streams = population[self.n_sr:]
        self.streams = dict(zip(
            [s.id for s in streams],
            streams
        ))

        sea_and_rivers = population[:self.n_sr]
        ns_n = []  # Number of streams for each river (and sea)
        costs_n = np.array([
            r.fitness
            for r in sea_and_rivers
        ])

        ns_n = []
        for i in range(0, self.n_sr):
            ns_i = int(np.round(
                np.abs(costs_n[i] / np.sum(np.abs(costs_n))) * self.n_st
            ))
            ns_n.append(ns_i)

        if np.sum(ns_n) < self.n_st:
            ns_n[np.random.randint(0, len(ns_n))] += self.n_st - np.sum(ns_n)

        pending_of_assignment = np.array(list(self.streams.keys()))

        for i in range(0, self.n_sr - 1):
            selected = np.random.choice(
                pending_of_assignment,
                size=ns_n[i],
                replace=False
            )
            pending_of_assignment = np.setdiff1d(
                pending_of_assignment,
                selected
            )
            for s in selected:
                self.streams[s].river_id = sea_and_rivers[i].id
        # Assign the remaining streams to the last river
        for s in pending_of_assignment:
            self.streams[s].river_id = sea_and_rivers[-1].id
            # sea_and_rivers[-1].add_affluent(self.streams[s])

    @property
    def sea(self) -> agents_module.River:
        return self._sea

    @sea.setter
    def sea(self, sea: agents_module.River):
        self._sea = sea

    @property
    def rivers(self) -> dict[int, agents_module.River]:
        return self._rivers

    @rivers.setter
    def rivers(self, rivers: dict[int, agents_module.River]):
        self._rivers = rivers

    @property
    def streams(self) -> dict[int, agents_module.River]:
        return self._streams

    @streams.setter
    def streams(self, streams: dict[int, agents_module.River]):
        self._streams = streams

    @property
    def n_sr(self) -> int:
        """
            Get the sum of rivers & seas, namely the number of rivers + 1 (sea)
        """
        return self._parameters.get('n_sr', None)

    @property
    def n_s(self) -> int:
        """
            Get the number of seas, namely 1
        """
        return 1

    @property
    def n_r(self) -> int:
        """
            Get the number of rivers, namely n_sr - 1
        """
        return self.n_sr - self.n_s

    @property
    def n_st(self) -> int:
        """
            Get the number of streams, namely population_size - n_sr
        """
        return self.population_size - self.n_sr

    @n_st.setter
    def n_st(self, n_st: int):
        self._parameters['n_st'] = n_st

    @property
    def d_max(self) -> float:
        """
            Get the maximum distance between sea and rivers before evaporation
            occurs
        """
        return self._parameters.get('d_max', None)

    @d_max.setter
    def d_max(self, d_max: float):
        self._parameters['d_max'] = d_max

    @property
    def d_max_decay(self) -> float:
        """
            Get the maximum distance between sea and rivers before evaporation
            occurs
        """
        return self._parameters.get('d_max_decay', None)

    @property
    def c_constant(self) -> float:
        """
            Get the constant c for the distance formula
        """
        return self._parameters.get('c_constant', None)

    @property
    def agents(self) -> list[agents_module.Agent]:
        return [self.sea] + \
            list(self.rivers.values()) +\
            list(self.streams.values())

    @property  # override
    def best_agent(self):
        return self.sea  # The sea is the best agent

    @best_agent.setter  # override
    def best_agent(self, agent: agents_module.River):
        self.sea = agent

    def get_river(self, river_id: int) -> agents_module.River:
        if river_id == self.sea.id:
            return self.sea
        else:
            return self.rivers.get(river_id, None)

    def optimize(self) -> bool:
        for _, stream in self.streams.items():
            river = self.get_river(stream.river_id)
            # Stream flows to its corresponding river and sea (Eqs. 16,17)
            next_position = stream.position + \
                np.random.uniform(0, 1) *\
                self.c_constant *\
                (river.position - stream.position)
            # Assign and fix the next position
            stream.position = self.search.space.fix_position(next_position)
            # Compare the stream with its river (or sea if it is the case)

            if utils.improves(
                river.fitness,
                stream.fitness,
                self.search.mode
            ):
                # Stream flows to its river
                river.position = stream.position
                river.fitness = stream.fitness

                # Compare the river with its sea
                if utils.improves(
                    self.sea.fitness,
                    stream.fitness,
                    self.search.mode
                ):
                    # River flows to the sea
                    self.sea.position = river.position
                    self.sea.fitness = river.fitness

            # River flows to its sea (Eq. 18)
            next_position = river.position + \
                np.random.uniform(0, 1) *\
                self.c_constant *\
                (self.sea.position - river.position)
            # Assign and fix the next position
            river.position = self.search.space.fix_position(next_position)
            # Compare the river with its sea
            if utils.improves(
                self.sea.fitness,
                river.fitness,
                self.search.mode
            ):
                # River flows to the sea
                self.sea.position = river.position
                self.sea.fitness = river.fitness

        # Evaporation  --> raining process
        for _, river in self.rivers.items():
            if np.linalg.norm(river.position - self.sea.position) < self.d_max\
                    or np.random.uniform(0, 1) < 1e-1:
                # Create a new stream
                new_position =\
                    self.search.space.random_bounded(self.search.dims)
                stream = agents_module.River(
                    id=river.id,
                    position=new_position,
                    search=self.search,
                    owner=self.name
                )

                # Add the stream to the streams
                self.streams[stream.id] = stream

                # Get the best stream (including the newly created one)
                best_stream = Metaheuristic.sort_population(
                    list(self.streams.values()),
                    self.search.mode
                )[0]

                # Swap ids of the best stream and the new stream
                best_stream.id, stream.id = stream.id, best_stream.id
                """
                    This is a bit tricky, but it is necessary to swap the ids
                    of the best stream and the new stream, because the best
                    stream is now a river and all the streams that were
                    flowing to it must flow to the new river
                """

                if best_stream.id != stream.id:
                    # new stream flows to best_stream
                    stream.river_id = best_stream.id

                # best_stream flows to the sea
                best_stream.river_id = self.sea.id

                # Remove the best stream from the streams
                self.streams.pop(best_stream.id)

                # The best stream replaces the river
                self.rivers[best_stream.id] = best_stream

                # Compare the river with its sea
                if utils.improves(
                    self.sea.fitness,
                    best_stream.fitness,
                    self.search.mode
                ):
                    # River flows to the sea
                    self.sea.position = best_stream.position
                    self.sea.fitness = best_stream.fitness

        # Reduce the d_max parameter
        self.d_max = self.d_max - self.d_max_decay

    def update_parameters(self, **kwargs):
        # Update parameters of the metaheuristic
        # Maybe sharing agents between metaheuristics is a good idea,
        # together with their positions and fitnesses, etc
        # Code goes here, this is an abstract method
        pass


class ParticleSwarmOptimization(Metaheuristic):
    """
        Particle Swarm Optimization

        Params:
            - kwargs: dict, dictionary of parameters for the metaheuristic
                - inertia: float, inertia of the particles
                - cognitive: float, cognitive parameter of the particles
                - social: float, social parameter of the particles

        References:
            - https://ieeexplore.ieee.org/document/488968
            - https://ieeexplore.ieee.org/document/699146
    """
    def __init__(self, search: utils.Search, *args, **kwargs):
        super().__init__(search, *args, **kwargs)

        # Check specific parameters for the Particle Swarm Optimization
        if 'inertia' not in kwargs:
            logging.critical(
                "Inertia not specified for the Particle Swarm Optimization"
            )
        if 'cognitive' not in kwargs:
            logging.critical(
                "Cognitive not specified for the Particle Swarm Optimization"
            )
        if 'social' not in kwargs:
            logging.critical(
                "Social not specified for the Particle Swarm Optimization"
            )

        # specific initialization code for the Particle Swarm Optimization
        self.particles = [
            agents_module.Particle(
                id=i,
                position=self.search.space.random_bounded(search.dims),
                search=self.search,
                velocity=np.zeros(self.search.dims),
                owner=self.name
            ) for i in range(self.population_size)
        ]

        # Initialize best agent
        self._start_best_agent()

    @property
    def particles(self) -> List[agents_module.Particle]:
        return self._particles

    @particles.setter
    def particles(self, particles: List[agents_module.Particle]):
        self._particles = particles

    @property
    def inertia(self) -> float:
        return self.parameters.get('inertia', None)

    @property
    def cognitive(self) -> float:
        return self.parameters.get('cognitive', None)

    @property
    def social(self) -> float:
        return self.parameters.get('social', None)

    @property
    def agents(self):
        return self.particles  # For compatibility with other metaheuristics

    def optimize(self) -> bool:
        update = False
        for particle in self.particles:
            particle.update_velocity(
                inertia=self.inertia,
                cognitive=self.cognitive,
                social=self.social,
                swarm_best=self.best_agent
            )
            particle.position = particle.position + particle.velocity

            # Fix position if it's out of bounds
            particle.position = self.search.space.fix_position(
                particle.position
            )

            # Update fitness
            particle.fitness = self.search.objective_function(
                particle.position
            )

            # Check if improves particle's best position
            if utils.improves(
                particle.best_fitness,
                particle.fitness,
                self.search.mode
            ):
                particle.best_position = particle.position
                particle.best_fitness = particle.fitness

                # Check if improves global best position
                if utils.improves(
                    self.best_agent.fitness,
                    particle.best_fitness,
                    self.search.mode
                ):
                    self.best_agent = copy.deepcopy(particle)

                update = True
        return update

            

    def update_parameters(self, **kwargs):
        # Update parameters of the metaheuristic
        # Maybe sharing agents between metaheuristics is a good idea,
        # together with their positions and fitnesses, etc
        # Code goes here, this is an abstract method
        # TODO: Implement this method
        pass


class DifferentialEvolution(Metaheuristic):
    """
    Differential Evolution metaheuristic.

    Params:
        - search: Search type, search object
        - kwargs:
            - crossover_rate: float, crossover rate
            - diff_weight: float, differential weight

    References:
        - https://link.springer.com/book/10.1007/3-540-31306-0
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
            agents_module.Agent(
                i,
                search.space.random_bounded(search.dims),
                search,
                owner=self.name
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

    def optimize(self) -> bool:
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
                self.best_agent = copy.deepcopy(agent)
                return True  # New global best agent found
        return False

    def update_parameters(self, **kwargs):
        # Update parameters of the metaheuristic
        # Maybe sharing agents between metaheuristics is a good idea,
        # together with their positions and fitnesses, etc
        # Code goes here, this is an abstract method
        pass

# MODULE ENDS HERE
