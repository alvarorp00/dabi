from src.utils import EvalMode, Search, SearchSpace, improves
import numpy as np

# Agent class. Could act as particle, ant, etc.
# It's a generic class for representing positions in a search space
# and keeping track of its fitness,
# together with it's best position and fitness.


#############################
# AGENT CLASS               #
#############################

class Agent():
    """
    Agent class. Could act as particle, ant, etc.
    It's a generic class for representing positions in a search space.

    Agent's best fitness is the result of applying the objective function
    to the best position, so we store it to avoid unnecessary computations.
    """
    def __init__(self, id: int, position: np.ndarray, search: Search):
        """
        Creates an Agent object. Best position and fitness are initialized to
        the given position and fitness.

        Params:
            - id: integer, agent's id
            - position: np.ndarray type, position of the agent in the search
                        space
            - search: utils.Search, search class object
        """
        self._id = id
        self._position = position
        self._fitness = search.objective_function(position)
        self._best_position = position
        self._best_fitness = self.fitness
        self._search_space = search.space
        self._objective_function = search.objective_function

    @property
    def id(self) -> int:
        return self._id

    @property
    def position(self) -> np.ndarray:
        """
        Agent's position, which is a N-th dimensional array.
        """
        return self._position

    @position.setter
    def position(self, position: np.ndarray):
        self._position = position

    @property
    def fitness(self) -> float:
        """
        Agent's fitness is the result of applying the objective function,
        so we store it to avoid unnecessary computations for future calls.
        """
        return self._fitness

    @fitness.setter
    def fitness(self, fitness: float):
        self._fitness = fitness

    @property
    def best_position(self) -> np.ndarray:
        """
        Agent's best position, which is a N-th dimensional array.
        """
        return self._best_position

    @best_position.setter
    def best_position(self, best_position: np.ndarray):
        self._best_position = best_position

    @property
    def best_fitness(self) -> float:
        """
        Agent's best fitness is the result of applying the objective function,
        so we store it to avoid unnecessary computations for future calls.
        """
        return self._best_fitness

    @best_fitness.setter
    def best_fitness(self, best_fitness: float):
        self._best_fitness = best_fitness

    @property
    def search_space(self) -> SearchSpace:
        return self._search_space

    @DeprecationWarning
    def update_best(self, eval_mode: EvalMode, adaptative: bool = False):
        """
        Updates best position and fitness if current fitness is better than
        the best one.
        """
        if improves(self.fitness, self.best_fitness, eval_mode,
                    adaptative=adaptative):
            self.best_fitness = self.fitness
            self.best_position = self.position

    @DeprecationWarning
    def bounded(self) -> bool:
        """
        Checks if the agent is bounded to the search space.
        """
        return self.search_space.bounded(self.position)

    @DeprecationWarning
    def fix_position(self):
        """
        Fixes the position of the agent if
        it's not bounded to the search space.
        """
        self.position = self.search_space.fix_position(self.position)

    @DeprecationWarning
    def random_bounded(self):
        """
        Generates a random position inside the search space.
        """
        self.position = self.search_space.random_bounded(self.position.shape[0])  # noqa: E501

    def __str__(self):
        return f"Agent {self.id}"

    def __repr__(self) -> str:
        return f"Agent {self.id} at {self.position} with fitness {self.fitness} and best position {self.best_position} with fitness {self.best_fitness}."  # noqa: E501

    def __eq__(self, other) -> bool:
        return self.id == other.id

    def __ne__(self, other) -> bool:
        return self.id != other.id

    def __hash__(self) -> int:
        return hash(self.id)


#############################
# BEES CLASSES              #
#############################


class Bee(Agent):
    def __init__(self, id: int, position: np.ndarray, search: Search):
        super().__init__(id, position, search)
        self._trials = 0

    @property
    def trials(self) -> int:
        return self._trials

    @trials.setter
    def trials(self, trials: int):
        self._trials = trials

    def __str__(self):
        return f"Bee {self.id}"

    def __repr__(self) -> str:
        return f"Bee {self.id} at {self.position} with fitness {self.fitness} and best position {self.best_position} with fitness {self.best_fitness}."

    def __eq__(self, other) -> bool:
        return super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()


#############################
# PARTICLES CLASS           #
#############################


class Particle(Agent):
    def __init__(self,
                 id: int,
                 position:
                 np.ndarray,
                 search: Search,
                 velocity: np.ndarray = None
                 ):
        """
        Creates a Particle object.

        Params:
            - position: np.ndarray type, position of the
                        particle in the search space
            - fitness: float type, fitness of the particle
            - search_space: SearchSpace type,
                            search space where the particle is

        Default velocity is zero.
        """
        super().__init__(id, position, search)
        self._velocity = velocity
        self._best_velocity = velocity

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity

    @velocity.setter
    def velocity(self, velocity: np.ndarray):
        self._velocity = velocity

    @property
    def best_velocity(self) -> np.ndarray:
        return self._best_velocity

    @best_velocity.setter
    def best_velocity(self, best_velocity: np.ndarray):
        self._best_velocity = best_velocity

    def update_velocity(self,
                        inertia: float,
                        cognitive: float,
                        social: float,
                        swarm_best: 'Particle' = None):
        """
        Updates the velocity of the particle.
        """
        self.velocity = inertia * self.velocity + \
            cognitive * np.random.rand() * (self.best_position - self.position) + \
            social * np.random.rand() * (swarm_best.position - self.position)  # noqa: E501

    def __str__(self):
        return f"Particle {self.id}"

    def __repr__(self) -> str:
        return f"Particle {self.id} at {self.position} with fitness {self.fitness} and best position {self.best_position} with fitness {self.best_fitness}."  # noqa: E501

    def __eq__(self, other) -> bool:
        return super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()

####################
