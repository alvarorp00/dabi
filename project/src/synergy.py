class SynergyBoost:
    def __init__(self, metaheuristics):
        self.metaheuristics = metaheuristics
    
    def optimize(self, objective_function, initial_solution):
        best_solution = initial_solution
        for metaheuristic in self.metaheuristics:
            solution = metaheuristic.optimize(objective_function, initial_solution)
            if objective_function(solution) > objective_function(best_solution):
                best_solution = solution
        return best_solution