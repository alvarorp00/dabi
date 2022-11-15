
import numpy as np
import scipy.stats as st
import logging
import random
from enum import Enum
from typing import List, Set

# Test functions --> change target_func for using each of them

def sphere(vector):
    return np.sum(np.power(vector, 2))
    
def rastrigin(vector, A=10):
    return A + np.sum(np.power(vector, 2) - A * np.cos(2*np.math.pi*vector))

def rosenbrock(vector, A=100, B=20):
    return np.math.exp(-np.sum(np.array([(A*(vector[i+1]-vector[i]**2)**2 + (1-vector[i])**2)/B for i in range(len(vector) - 1)])))


class EvalMode(Enum):
    MINIMUM = 1
    MAXIMUM = 2

class Compare(Enum):
    IMPROVES = 1
    STAYS = 2


class Utils:
    @staticmethod
    def compare(old, new, mode: EvalMode, target_fn) -> Compare:
        if mode == EvalMode.MAXIMUM:
            if target_fn(new) > target_fn(old):
                return Compare.IMPROVES
            else:
                return Compare.STAYS
        else:
            if target_fn(new) < target_fn(old):
                return Compare.IMPROVES
            else:
                return Compare.STAYS

    @staticmethod
    def around(f):
        return np.around(np.float128(f), 5)  # just save 5 decimals


class SearchSpace():
    def __init__(self, lower_bound: float, upper_bound: float):
        """
        Creates a bounded space. Dimensions are not specified, it'll allow N dimensions.

        Params:
            - lower_bound: float type, minimum value a coordenate can reach
            - upper_bound: float type, maximum value a coordenate can reach
        """
        self._lower_bound = Utils.around(lower_bound)
        self._upper_bound = Utils.around(upper_bound)

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
        return Utils.around(np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=dims))


class Agent():
    def __init__(self, id: int, pos: np.ndarray):
        self._id = id
        self.update_position(pos)
        self._hash = self._id.__hash__()

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        self._hash

    def __str__(self):
        return f'Agent {self.id} @ [{self.pos}]'

    @property
    def pos(self):
        return self._pos

    @property
    def id(self):
        return self._id

    def update_position(self, position: np.ndarray):
        self._pos = np.array(Utils.around(position))


class DifferentialEvolution():
    def __init__(self,\
        space: SearchSpace,
        n_agents: int,
        target_fn,
        dims,
        cr,
        f,
        mode: EvalMode
    ):
        self._space = space
        self._n_agents = n_agents
        self._target_fn = target_fn
        self._agents = dict(
            zip(
                [i for i in range(1, self._n_agents + 1)],
                [Agent(i, space.random_bounded(dims)) for i in range(1, self._n_agents + 1)]
            )
        )
        self._dims = dims
        self._cr = cr
        self._f = f
        self._mode = mode

        self._update_best_pos(list(self.agents.values())[0].pos) # first agents' pos

        for a in list(self.agents.values())[1:]:
            eval_res = Utils.compare(self.best_pos, a.pos, self.mode, self.target_fn)
            if eval_res == Compare.IMPROVES:
                self._update_best_pos(a.pos)

    @property
    def space(self) -> SearchSpace:
        return self._space
    
    @property
    def n_agents(self):
        return self._n_agents

    @property
    def target_fn(self):
        return self._target_fn

    @property
    def agents(self):
        return self._agents

    @property
    def dims(self):
        return self._dims

    @property
    def cr(self):
        return self._cr

    @property
    def f(self):
        return self._f

    @property
    def mode(self):
        return self._mode

    @property
    def best_pos(self):
        return self._best_pos

    @property
    def best(self):
        return self._best

    def _update_best_pos(self, best_pos):
        """
        Updates pos and target_fn(pos) a.k.a. best pos & best
        """
        self._best_pos = np.float128(best_pos)
        self.__update_best(self.target_fn(self.best_pos))

    def __update_best(self, best):
        """
        This is called by _update_best_pos
        """
        self._best = np.float128(best)

    def run(self, iterations: int):
        """
        Performs Differential Evolution 
        """
        for _ in range(iterations):
            for id, agent in self.agents.items():
                __subagents = self.agents.copy()
                __subagents.pop(id)
                
                if len(__subagents) < 3:  # check [a,b,c] can be selected
                    logging.critical(f'Agent {id} has no three mates [a,b,c]. Error.')
                    
                __abc: List[Agent] = []
                for _ in range(0,3):  # select agents [a,b,c]
                    __abc.append(random.choice(list(__subagents.values())))
                    __subagents.pop(__abc[-1].id)

                candidate_pos = {}

                __i=0
                for coord in agent.pos:  # register current positions in dictionary
                   candidate_pos[__i] = coord
                   __i+=1 
                
                dims_list: Set[int] = list(candidate_pos.keys())  # current coordinates indexes [0,1,2...n_dims]
                i_dim = random.choice(list(dims_list))
                candidate_pos[i_dim] = __abc[0].pos[i_dim] + self.f * (__abc[1].pos[i_dim] - __abc[2].pos[i_dim])
                dims_list.remove(i_dim)
                
                for j_dim in dims_list:
                    threshold = np.random.uniform(low=0, high=1)
                    if threshold < self.cr:
                        candidate_pos[j_dim] = __abc[0].pos[j_dim] + self.f * (__abc[1].pos[j_dim] - __abc[2].pos[j_dim])
                
                candidate_refined = self.space.fix_position(np.array(list(candidate_pos.values())))

                if Utils.compare(agent.pos, candidate_refined, self.mode, self.target_fn) == Compare.IMPROVES:
                    agent.update_position(candidate_refined)
                    if Utils.compare(self.best_pos, agent.pos, self.mode, self.target_fn) == Compare.IMPROVES:
                        self._update_best_pos(agent.pos)


# Configuration of the simulation
N_DIMS = 5
N_AGENTS = 10**2
ITERATIONS = 10**4
LOW, HIGH = -100, 100
CR, FF = .65, 1.15
MODE = EvalMode.MINIMUM

if __name__=='__main__':
    target_func = lambda vector: sphere(vector)  # [sphere | rastrigin | rosenbrock]

    ss = SearchSpace(LOW, HIGH)
    de = DifferentialEvolution(
        space=ss,
        n_agents=N_AGENTS,
        dims=N_DIMS,
        target_fn=target_func,
        mode=MODE,
        cr=CR,
        f=FF
    )

    # print('Agents: ')
    # for agent in de.agents.values():
    #     print(f'\t{agent}')
    # print('\n')

    agents_best = np.min(np.array([
        de.target_fn(a.pos) for a in list(de.agents.values())
    ]))

    print(f'Starter best: {de.best_pos} @ eval_fn_result: {de.best}')

    de.run(ITERATIONS)  # RUN Differential Evolution

    print(f'Final best: {de.best_pos} @ eval_fn_result: {de.best}')