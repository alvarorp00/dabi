from enum import Enum
import numpy as np

####################
# Helper functions #
####################


def around(f, decimals=5):
    """
    Keeps up to n (default=5) decimals in given float type
    """
    return np.around(np.float128(f), decimals=decimals)

####################

####################
# Helper classes   #
####################

# Compare mode


class EvalMode(Enum):
    """Evaluation mode."""
    MINIMIZE = 0
    MAXIMIZE = 1


class CompareResult(Enum):
    """Compare result."""
    LESS = -1
    EQUAL = 0
    GREATER = 1


def compare(a, b, mode):
    """
        Compare two values.
        Tells if b is better than a.
    """
    if mode == EvalMode.MINIMIZE:
        if a < b:
            return CompareResult.LESS
        elif a == b:
            return CompareResult.EQUAL
        else:
            return CompareResult.GREATER
    else:
        if a > b:
            return CompareResult.LESS
        elif a == b:
            return CompareResult.EQUAL
        else:
            return CompareResult.GREATER


def improves(a, b, mode, adaptative=False):
    """
        Check if b improves a.
        Adaptative mode: if a and b are equal, then b improves a.
    """
    __r = compare(a, b, mode) == CompareResult.GREATER
    if adaptative:
        __r = __r or compare(a, b, mode) == CompareResult.EQUAL
    return __r


class Search():
    def __init__(self, space, objective_function, mode):
        """
        Creates a search instance.

        Params:
            - space: SearchSpace type, space where the search is going to be performed
            - objective_function: function type, target function to be optimized
            - mode: EvalMode type, evaluation mode
        """
        self._space = space
        self._objective_function = objective_function
        self._mode = mode

    @property
    def space(self):
        return self._space

    @property
    def objective_function(self):
        return self._objective_function

    @property
    def mode(self):
        return self._mode


class SearchSpace():
    def __init__(self, lower_bound: float, upper_bound: float):
        """
        Creates a bounded space. Dimensions are not specified, it'll allow N dimensions.

        Params:
            - lower_bound: float type, minimum value a coordenate can reach
            - upper_bound: float type, maximum value a coordenate can reach
        """
        self._lower_bound = around(lower_bound)
        self._upper_bound = around(upper_bound)

    @property
    def lower_bound(self) -> np.ndarray:
        return self._lower_bound

    @property
    def upper_bound(self) -> np.ndarray:
        return self._upper_bound

    def bounded(self, pos: np.ndarray) -> bool:
        """
        Checks if a given N-th dimensional array is bounded to the space created

        Params:
            - pos: np.ndarray type, whose coords are going to be checked

        Returns:
            - bounded: bool type, true if all coords are inside bounds, false otherwise
        """
        return np.all(np.array(
            [p >= self.lower_bound and p <= self._upper_bound for p in pos]
        ))

    def fix_position(self, pos: np.ndarray) -> np.ndarray:
        """
        Forces values that exceed bound to be in the border of them

        Params:
            - pos: array type, position to be checked and fixed

        Returns:
            - fixed position: np.ndarray type
        """
        _fxd_pos = []
        for p in pos:
            if p < self.lower_bound:
                _fxd_pos.append(self.lower_bound)
            elif p > self.upper_bound:
                _fxd_pos.append(self.upper_bound)
            else:
                _fxd_pos.append(p)
        return np.array(_fxd_pos)

    def random_bounded(self, dims: int) -> np.ndarray:
        """
        Returns a random position inside the bounds of the space

        Params:
            - dims: integer, number of dimensions desired for the position
        """
        return around(np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=dims))

# Agent class. Could act as particle, ant, etc. It's a generic class for representing positions in a search space
# and keeping track of its fitness (together with it's best position and fitness).


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
            - position: np.ndarray type, position of the agent in the search
                        space
            - fitness: float type, fitness of the agent
            - search_space: SearchSpace type, search space where the agent is
        """
        self._id = id
        self._position = position
        self._fitness = around(search._objective_function(position))
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
        Fixes the position of the agent if it's not bounded to the search space.
        """
        self.position = self.search_space.fix_position(self.position)

    @DeprecationWarning
    def random_bounded(self):
        """
        Generates a random position inside the search space.
        """
        self.position = self.search_space.random_bounded(self.position.shape[0])

    def __str__(self):
        return f"Agent {self.id}"

    def __repr__(self) -> str:
        return f"Agent {self.id} at {self.position} with fitness {self.fitness} and best position {self.best_position} with fitness {self.best_fitness}."


class Particle(Agent):
    def __init__(self, id: int, position: np.ndarray, search: Search):
        """
        Creates a Particle object.

        Params:
            - position: np.ndarray type, position of the particle in the search space
            - fitness: float type, fitness of the particle
            - search_space: SearchSpace type, search space where the particle is
        """
        super().__init__(id, position, search)
        self._velocity = np.zeros_like(self.position)
        self._best_velocity = np.zeros_like(self.position)

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

    def __str__(self):
        return f"Particle {self.id}"

    def __repr__(self) -> str:
        return f"Particle {self.id} at {self.position} with fitness {self.fitness} and best position {self.best_position} with fitness {self.best_fitness}."

####################
