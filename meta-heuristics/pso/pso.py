from typing import List
import numpy as np
import scipy.stats as st

# Configuration of the simulation
N_DIMS = 2
N_PARTICLES = 10**3
STEPS = 5 * 10**3
LOW, HIGH = -25, 25
MIN_VEL, MAX_VEL = -5, 5

# Test functions --> change target_func for using each of them

def sphere(vector):
    return np.sum(np.power(vector, 2))
    
def rastrigin(vector, A=10):
    return A + np.sum(np.power(vector, 2) - A * np.cos(2*np.math.pi*vector))

def rosenbrock(vector, A=100, B=20):
    return np.math.exp(-np.sum(np.array([(A*(vector[i+1]-vector[i]**2)**2 + (1-vector[i])**2)/B for i in range(len(vector) - 1)])))

target_func = lambda vector: sphere(vector)  # OBJECTIVE FUNCTION USED BY PARTICLES

"""
Particle class with position track & local best position and value

 - random start between bounded domain [LOW; HIGH]
 - random velocity between velocity limits [MIN_VEL, MAX_VEL]
 - first local best is random start position
"""
class Particle():
    def __init__(self, calc_objective=True):
        self.velocity=np.random.uniform(low=MIN_VEL, high=MAX_VEL, size=N_DIMS)
        self.position=np.random.uniform(low=LOW, high=HIGH, size=N_DIMS)
        self.local_best_position=self.position
        self.local_best_value=objective_function(self.local_best_position) if calc_objective else 0
        self.position_trace=np.empty(shape=(STEPS, N_DIMS))
        self.velocity_trace=np.empty(shape=(STEPS, N_DIMS))

# Instantiate particles
def initialize():
    particles = [Particle() for _ in np.arange(0, N_PARTICLES)]
    return particles

# Check if particle is inside domain
def valid_pos(position):
    for pos in position:
        if pos < LOW or pos > HIGH:
            return False
    return True

# If particle is outside domain, no bounce: attach it to the border in the axis surpassed
def fix_position(position):
    fixed_pos = []
    for pos in position:
        if pos < LOW:
            fixed_pos.append(LOW)
        elif pos > HIGH:
            fixed_pos.append(HIGH)
        else:
            fixed_pos.append(pos)
    return np.float64(np.array(fixed_pos))

# Objetive funcion, uses lambda target_func defined above to perform the calculus
def objective_function(position):
    if valid_pos(position) is False:
        return np.finfo(np.float64).max
    return target_func(vector=position)

"""
Algorithm:
    - Searchs for global minimum of the objective function called by target_func.

    - Select w from Normal(1, 0.5) <- close to 1
    - Select q1, q2 (a.k.a. c1, c2) from Normal(2, 0.5) <- close to 2

    1. Retrieve swarm_best_position firstly from particles starting positions
    2. For each step to do
        1. For each particle
            1. Update velocity
            2. Update position
            3. Check if current position improves local_best
            1. Check if current position improves swarm_best
            4. Add velocity and position to the trace --> Not used here but for debugging is useful
"""
def pso(particles: List[Particle]):
    """
    Returns local_bests & global_best
    (particles, local_bests) -> (local_bests, global_best)
    """
    inertia_weight = st.norm(loc=1, scale=.5).rvs(size=1)
    c1, c2 = st.norm(loc=2, scale=.5).rvs(size=2)

    swarm_best_position = particles[0].local_best_position

    for i in np.arange(1, len(particles)):
        if objective_function(particles[i].local_best_position) < objective_function(swarm_best_position):
            swarm_best_position = particles[i].local_best_position
    
    for i in np.arange(STEPS):
        for j in np.arange(N_PARTICLES):
            particles[j].velocity = inertia_weight * particles[j].velocity +\
                c1*np.random.random()*(particles[j].local_best_position - particles[j].position) +\
                    c2*np.random.random()*(swarm_best_position - particles[j].position)

            particles[j].position += particles[j].velocity
            particles[j].position = np.float64(fix_position(particles[j].position))

            candidate = objective_function(particles[j].position)

            if candidate < objective_function(particles[j].local_best_position):
                particles[j].local_best_position = particles[j].position

            if objective_function(particles[j].local_best_position) < objective_function(swarm_best_position):
                swarm_best_position = particles[j].local_best_position

            particles[j].position_trace[i] = particles[j].position
            particles[j].velocity_trace[i] = particles[j].velocity

    return swarm_best_position, particles

if __name__=='__main__':
    particles = initialize()
    swarm_best_position, particles = pso(particles=particles)
    print(f"Swarm Best: -> {swarm_best_position}")
    best_locals = np.array([particle.local_best_position for particle in particles])
    best_unique_locals = np.unique(best_locals, axis=0)
    print("====================")
    print(f"Best Unique Locals: \n {best_unique_locals}")