
import numpy as np
import scipy.stats as st
import logging
from enum import Enum
from typing import List


ITERATIONS = 100


class Direction(Enum):
        STRAIGHT = 1
        REVERSE = 2

class Graph():
    def __init__(self, start, finish,\
        alpha: np.double = np.float64(1), beta: np.double = np.float64(1),\
            ro: np.double = np.float64(1), name='Graph'):
        """
        Parameters:
            - start: starting node
            - finish: objective node
            - alpha: pheromone influence
            - beta: desirability influence
            - name: name of the graph
        
        Returns:
            - new graph
        """
        self._start = start
        self._finish = finish
        self._parameters = dict({
            'name': name,
            'edges': dict({}),
            'alpha': np.float64(alpha),
            'beta': np.float64(beta),
            'ro': np.float64(ro)
        })

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        _str = self.name
        for v in self.edges:
            _str += "\n\t" + str(v)
        return _str
    
    @property
    def name(self):
        return self._parameters['name']

    @property
    def start(self):
        return self._start

    @property
    def finish(self):
        return self._finish

    @property
    def edges(self):
        return self._parameters['edges']

    @property
    def alpha(self):
        return self._parameters['alpha']

    @property
    def beta(self):
        return self._parameters['beta']

    @property
    def ro(self):
        return self._parameters['ro']

    def _reachable(self, source, target) -> bool:
        """
        Checks if target is reachable from given source

        Params:
            - source: Node, where to start the check
            - target: Node, node to reach from source

        Returns:
            - True if target is reachable from source
        """
        if source == target: return True
        edges = self.get_edges_from_node(source, Direction.STRAIGHT)
        nodes = [edge.node_right for edge in edges]
        for option in nodes:
            if self._reachable(option, target):
                return True
        return False

    def validate_reachability(self):
        """
        Checks if the current graph with the correspondant
        edges can reach the target node from the starting one
        """
        return self._reachable(self.start, self.finish)

    def _step_probability(self, edge):
        """
        Given formula that computes the probability
        of choosing an edge

        Params:
            - edge: Edge, whos probability is being computed
        
        Returns:
            - float, probability of walking through an edge
        """
        return np.float64(
            (
                np.math.pow(edge.pheromone, self.alpha)
            *
                np.math.pow(edge.desirability, self.beta)
            )
            /
            (
                np.math.pow(np.sum(np.array([_edge.pheromone for _edge in self.edges.values()])), self.alpha)
            +
                np.math.pow(np.sum(np.array([_edge.desirability for _edge in self.edges.values()])), self.beta)
            )
        )

    def add_edge(self, edge):
        if edge.name in self.edges:
            logging.warning(
                'Edge {' + edge.name + '} already in Graph { ' + self.name + ' }'
            )
        else:
            self.edges[edge.name] = edge

    def add_edges(self, edges):
        for edge in edges:
            self.add_edge(edge=edge)

    def get_edge_by_name(self, edge_name):
        if self.edges[edge_name] is None:
            logging.critical(
                'Edge {' + edge_name + '} not in Graph { ' + self.name + ' }'
            )
            return None
        else:
            return self.edges[edge_name]

    def get_edges_from_node(self, node, direction: Direction):
        """
        Returns the list of edges than can be followed from a node in
        a given direction

        Params:
            - node: Node type, from when edges will be retrieved
            - direction: Direction type

        Returns:
            - List[Edge]
        """
        if direction == Direction.STRAIGHT:
            _ftr_fn = lambda e: e.node_left == node
        else:
            _ftr_fn = lambda e: e.node_right == node
        return list(filter(_ftr_fn, self.edges.values()))

    def step(self, node, direction, burn_in=False):
        """
        At a given node, computes an step and
        returns the edge chosen to be traversed

        Params: 
            - node: Node type, where the ant is placed
            - direction: Direction type, direction of the ant
            - [burn_in]: Optional, bool. If True, edge probability is discarded
                            and path is chosen randomly. Default false

        Returns:
            - edge: Edge type, edge to be traversed in the step
        """
        valid_edges = self.get_edges_from_node(node, direction)
        if len(valid_edges) == 0:
            logging.critical(
                f'Node -{node}- with no path in -{direction}- direction'
            )
            return None
        probs = np.array([self._step_probability(edge) for edge in valid_edges])
        if np.sum(probs) == 0 or burn_in:  # first round is purely random --> burn_in phase
            choice = np.random.choice(valid_edges, size=1)
        else:
            probs /= np.sum(probs)  # normalize in case they don't sum 1
            choice = np.random.choice(valid_edges, size=1, p=probs)
        return choice[0]


class Node():
    def __init__(self, idx: int, name=None):
        self._name = name if name is not None else\
            'Node [' + str(idx) + ']'
        self._parameters = dict({
            'idx': idx
        })

    def __eq__(self, other):
        return self.index == other.index
    
    def __ne__(self, other):
        return not self.__eq__(other)
        
    def __hash__(self):
        return self.index.__hash__()
        
    def __str__(self):
        return self.name

    @property
    def name(self):
        return self._name
        
    @property
    def index(self):
        return self._parameters['idx']

class Edge():
    def __init__(self, node_left, node_right, cost, name=None):
        self._name = name if name is not None else\
            'Edge [' + str(node_left) + ' <--> ' + str(node_right) +\
                ' @ Cost ' + str(cost) + ']'
        if node_left == node_right:
            logging.critical(
                f'Edge -{self._name}- with node -{str(node_left)}- @ self_loop'
            )
            return None
        self._parameters = dict({
            'left': node_left,
            'right': node_right,
            'cost': cost,
            'pheromone': np.float64(0),
            'desirability': np.float64(1 / cost),
            'straight_visits': np.int32(0), # times edge is walked by ants STRAIGHT
            'reverse_visits': np.int32(0)  # times edge is walked by ants REVERSE
        })

    def __eq__(self, other):
        return (self.node_left == other.node_left and\
                self.node_right == other.node_right) or\
                    (self.node_left == other.node_right and\
                        self.node_right == other.node_left)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return int(f'{self.node_left.__hash__()}'+f'{self.node_right.__hash__()}').__hash__()

    def __str__(self):
        return f'{self.name} [P:{str(self.pheromone)}] [SV:{self.straight_visits}| RV:{self.reverse_visits}]'

    @property
    def name(self):
        return self._name

    @property
    def node_left(self):
        return self._parameters['left']

    @property
    def node_right(self):
        return self._parameters['right']

    @property
    def cost(self):
        return self._parameters['cost']

    @property
    def pheromone(self):
        return self._parameters['pheromone']
    
    @property
    def desirability(self):
        return self._parameters['desirability']
    
    @property
    def straight_visits(self):
        return self._parameters['straight_visits']
    
    @property
    def reverse_visits(self):
        return self._parameters['reverse_visits']

    def walk(self, direction: Direction):
        """
        Call this method when the edge is walked by an ant,
        it updates the counter of how many times it's been visited
        """
        if direction == Direction.STRAIGHT:
            self._parameters['straight_visits'] += 1
        else:
            self._parameters['reverse_visits'] += 1
    
    def update_pheromone(self, ro, delta):
        self._parameters['pheromone'] = max(0, np.float64((1-ro)*self.pheromone + delta))

    def nodes_by_direction(self, direction: Direction):
        """
        Given the direction in which edge is traversed,
        it returns the src_node and dst_node in order

        Parameters:
            - direction: Direction type; in which direction node is traversed
        
        Returns:
            - (src_node, dst_node) tuple
        """
        if direction == Direction.STRAIGHT:
            src_node = self.node_left
            dst_node = self.node_right
        else:
            src_node = self.node_right
            dst_node = self.node_left
        return (src_node, dst_node)



class Colony():
    class Iteration():
        def __init__(self):
            self._distance = np.int32(0)
            self._straight_trace: List[Edge] = []
            self._reverse_trace: List[Edge] = []

        @property
        def distance(self):
            return self._distance

        @property
        def straight_pace(self):
            return self._straight_trace

        @property
        def reverse_pace(self):
            return self._reverse_trace
        
        def add_edge(self, edge: Edge, direction: Direction):
            """
            Adds an edge to the trace depending on wether the direction
            is STRAIGHT or REVERSE and adds the cost to the total distance

            It does not change anything inside edge, no pheromone update nor walk count

            Params:
                - edge: Edge type, edge to add to the trace
                - direction: Direction type, how ant is moving
            """
            if direction == Direction.STRAIGHT:
                self.straight_pace.append(edge)
            else:
                self.reverse_pace.append(edge)
            self._distance += edge.cost

        def finish(self):
            """
            This methods MUST BE CALLED after finishing the path
            """
            self.edges = set(self.straight_pace).union(set(self.reverse_pace))

        def contains_edge(self, edge: Edge):
            return edge in self.edges
    
    class Ant():
        def __init__(self, colony, node: Node, finish: Node,\
             identifier: int=0, direction=Direction.STRAIGHT):
            """
            Instantiates an ant

            Params:
                - colony: Colony type, to which ant belongs
                - position: Node type, where the ant is found when instantiated
                - identifier: for debug purposes so we can recognise the ant later (might be unused)
                - direction: Direction type, either STRAIGHT or REVERSE
            """
            self._parameters = dict({
                'node': node,  # current node
                'source': node,  # origin node
                'finish': finish,  # finisher node
                'direction': direction,
                'colony': colony,
                'identifier': identifier,
                'walked': np.float64(0)
            })

            self._iterations: List[Colony.Iteration] = []

        def __str__(self):
            return f'Ant [{self.identifier}]'

        @property
        def node(self):
            return self._parameters['node']

        @property
        def source(self):
            return self._parameters['source']

        @property
        def finish(self):
            return self._parameters['finish']

        @property
        def target(self):
            if self.direction == Direction.STRAIGHT:
                return self.finish  # straight --> finish is the target
            else:
                return self.source  # backwards --> source is the target

        @property
        def direction(self):
            return self._parameters['direction']

        @property
        def colony(self):
            return self._parameters['colony']

        @property
        def identifier(self):
            return self._parameters['identifier']

        @property
        def walked(self):
            return self._parameters['walked']

        @property
        def iterations(self):
            return self._iterations

        def _change_node(self, node: Node):
            self._parameters['node'] = node

        def _change_direction(self, direction: Direction):
            self._parameters['direction'] = direction

        def _add_iteration(self, iteration):
            self._iterations.append(iteration)

        def _iteration(self, iteration, burn_in):
            edge: Edge = self.colony.graph.step(
                node=self.node, direction=self.direction, burn_in=burn_in
            )
            iteration.add_edge(edge, self.direction)
            _, dst_node = edge.nodes_by_direction(self.direction)
            self._change_node(dst_node)
            edge.walk(self.direction)

        def iteration(self, burn_in=False):
            """
            Performs a complete iteration searching the target node
            and returning to the starting one. If burn_in is set to true,
            edges are chosen randomly without taking into account edge probability.
            Set burn_in to true during the first phase.

            Params:
                - [burn_in]: Optional bool, if True then path is chosen randomly at
                                each node without evaluating pheromones or desirability

            Returns:
                - iteration: Colony.Iteration type with the result of the pathing
            """
            iteration = Colony.Iteration()
            self._change_node(self.source)  # current node --> start
            self._change_direction(Direction.STRAIGHT)

            while self.node != self.finish:  # moving STRAIGHT to the finisher node (->)
                self._iteration(iteration, burn_in)

            self._change_direction(Direction.REVERSE)

            while self.node != self.source: # moving REVERSE to the initial node (<-)
                self._iteration(iteration, burn_in)                
            
            iteration.finish()  # finishes the iteration
            self._add_iteration(iteration)  # save iteration done

            return iteration
    
    def __init__(self, graph: Graph, colony_size: int, n_iterations: int=ITERATIONS):
        """
        Instantiates an ant colony optimization
        """
        self._graph = graph
        self._colony_size = colony_size
        self._ants: List[Colony.Ant] = [
            Colony.Ant(self, graph.start, graph.finish, i, Direction.STRAIGHT) for i in range(0, colony_size)
        ]
        self._n_iterations = n_iterations

    @property
    def graph(self):
        return self._graph

    @property
    def ants(self) -> List[Ant]:
        return self._ants

    @property
    def colony_size(self):
        return self._colony_size

    @property
    def n_iterations(self):
        return self._n_iterations

    def _update_edges(self, iterations):
        for edge in self.graph.edges.values():
            delta = np.float64(0)  # amount of pheromone to add to an edge
            for iteration in iterations:
                if iteration.contains_edge(edge):
                    delta += np.float64((1 / np.float64(iteration.distance)))
            edge.update_pheromone(ro=self.graph.ro, delta=delta)

    def run(self):
        """
        Computes the simulation of the aco metaheuristic
        and stores the results inside the edges of the supplied graph
        """
        # burn_in phase
        iterations: List[Colony.Iteration] = []
        for ant in self.ants:
            iterations.append(ant.iteration(burn_in=True))

        self._update_edges(iterations)

        # True iterations
        for _ in range(self.n_iterations):
            iterations: List[Colony.Iteration] = []  # delete previous iterations
            for ant in self.ants:
                iterations.append(ant.iteration())
            self._update_edges(iterations)

    def solve(self):
        """
        Returns a List[Edge] sequence corresponding to the path followed by the ants
        in both ways, first targeting the final node from the source and then reverse it.

        Assumes that graph.validate_reachability is True hence the problem can be computed and
        solved.

        Run this method after run() so the edges have the neccesary stats (namely the visits)
        required.
        """

        """
        edge: Edge = self.colony.graph.step(
                node=self.node, direction=self.direction, burn_in=burn_in
            )
            iteration.add_edge(edge, self.direction)
            _, dst_node = edge.nodes_by_direction(self.direction)
            self._change_node(dst_node)
            edge.walk(self.direction)
        """

        source = self.graph.start
        target = self.graph.finish

        path: List[Edge] = []
        current = source

        while current != target:  # Running with STRAIGHT direction (->)
            valid_edges: List[Edge] = self.graph.get_edges_from_node(current, Direction.STRAIGHT)
            best_edge: Edge = valid_edges[0]  # start with the first one by default
            if len(valid_edges) > 1:
                for edge in valid_edges[1:]:
                    if edge.straight_visits > best_edge.straight_visits:
                        best_edge = edge
            current = best_edge.node_right
            path.append(best_edge)
        
        while current != source:
            valid_edges: List[Edge] = self.graph.get_edges_from_node(current, Direction.REVERSE)
            best_edge: Edge = valid_edges[0]  # start with the first one by default
            if len(valid_edges) > 1:
                for edge in valid_edges[1:]:
                    if edge.reverse_visits > best_edge.reverse_visits:
                        best_edge = edge
            current = best_edge.node_left
            path.append(best_edge)
        
        return path


if __name__=='__main__':
    nodes = [Node(i+1) for i in range(0,6)]
    edges = [
        Edge(nodes[0], nodes[1], 5),
        Edge(nodes[0], nodes[2], 15),
        Edge(nodes[0], nodes[3], 3),
        Edge(nodes[1], nodes[4], 20),
        Edge(nodes[2], nodes[4], 12),
        Edge(nodes[3], nodes[5], 30),
        Edge(nodes[4], nodes[5], 15)
    ]

    alpha = .95
    beta = 1
    ro = .15

    graph = Graph(
        start=nodes[0],
        finish=nodes[-1],
        alpha=alpha,
        beta=beta,
        ro=ro,
        name='Demo Graph'
    )

    graph.add_edges(edges)

    if graph.validate_reachability() is False:
        logging.critical(
            f'Graph -{str(graph)}- with no connection between {graph.start} and {graph.finish}'
        )

    colony_size = 10**3

    aco = Colony(graph=graph, colony_size=colony_size)


    aco.run()


    print('Edge [source <--> dest @ cost/distance] [P: pheromone_level] [SV:straight_visits(->)] [RV:reverse_visits(<-)]\n')

    for e in aco.graph.edges.values():  # Result of all edges after the simulation
        print(e)
    print('\n')

    print('Solving path from walks data')

    path = aco.solve()

    for edge in path:  # Best path found by the ant colony metaheuristic
        print(edge)




