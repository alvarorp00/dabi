from enum import Enum
from typing import List
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


def converged(a, b, criteria):
    """
        Check if a and b are close enough.

        Params:
            - a: float type, first value
            - b: float type, second value
            - criteria: float type, maximum difference between a and b

        Returns:
            - converged: bool type, true if a and b are close enough,
                         false otherwise
    """
    return abs(a - b) <= criteria


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


class Search():
    def __init__(self, space: SearchSpace, objective_function,
                 mode: EvalMode, dims: int):
        """
        Creates a search instance.

        Params:
            - space: SearchSpace type, space where the search is going to be performed
            - objective_function: function type, target function to be optimized
            - mode: EvalMode type, evaluation mode
            - dims: integer, number of dimensions of the search space
        """
        self._space = space
        self._objective_function = objective_function
        self._mode = mode
        self._dims = dims

    @property
    def space(self) -> SearchSpace:
        return self._space

    @property
    def objective_function(self):
        return self._objective_function

    @property
    def mode(self):
        return self._mode

    @property
    def dims(self) -> int:
        return self._dims


# Trace class

class Trace:
    """
        Trace class. Stores the trace of an agent.

        Attributes:
            - trace: list of dicts, each dict contains the following keys:
                - iteration: integer, iteration number
                - run: integer, run number
                - position: np.ndarray type, position of the agent
                - fitness: float type, fitness of the agent
    """
    def __init__(self, *args, **kwargs) -> None:
        self._trace = []

    @property
    def trace(self) -> dict:
        return self._trace

    def add(self, agent, iteration, run):
        """
        Adds a new trace to the trace list.

        Params:
            - agent: Agent type, agent whose trace is going to be added
            - iteration: integer, iteration number
            - run: integer, run number
        """
        self._trace.append({
            'name': str(agent),
            'iteration': iteration,
            'run': run,
            'position': agent.position,
            'fitness': agent.fitness,
            'owner': agent.owner,
        })
