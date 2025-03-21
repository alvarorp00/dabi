from src.fitness_functions import sphere, rastrigin, rosenbrock, griewank
from src.utils import EvalMode, Search, SearchSpace

# Population
population_size = 10

# Number of runs
runs = 100

# Number of iterations
iterations = 100  # number of iterations per run

# Convergence criteria
convergence_criteria = 1e-12

# Number of dimensions
dimensions = 10

# Lower bound
lower_bound = -5.12

# Upper bound
upper_bound = +5.12

# Target function
target_function = rastrigin

# Evaluation mode
eval_mode = EvalMode.MINIMIZE

# Search configuration
search: Search = Search(
    space=SearchSpace(
        lower_bound=lower_bound,
        upper_bound=upper_bound
    ),
    objective_function=target_function,
    mode=eval_mode,
    dims=dimensions
)
