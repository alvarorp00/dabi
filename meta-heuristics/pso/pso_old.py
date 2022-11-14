from tokenize import Double
from typing import List
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import scipy.stats as st
import matplotlib.pyplot as plt

N_PARTICLES = 10**4
STEPS = 10**3
LOW, HIGH = -50, 50

class Particle():
    def __init__(self):
        self.velocity=np.random.uniform(low=0, high=10, size=2)
        self.position=np.random.uniform(low=LOW, high=HIGH, size=2)
        self.local_best_position=self.position
        self.local_best_value=objective_function(self.local_best_position)
        self.position_trace=np.empty(shape=(STEPS, 2))
        self.velocity_trace=np.empty(shape=(STEPS, 2))

def initialize():
    particles = [Particle() for _ in np.arange(0, N_PARTICLES)]
    return particles

def valid_pos(position):
    x_pos, y_pos = np.float32(position)
    if x_pos < LOW or x_pos > HIGH:
        return False
    if y_pos < LOW or y_pos > HIGH:
        return False
    return True

def fix_position(position):
    x_pos, y_pos = np.float32(position)
    
    if x_pos < LOW:
        x_pos = LOW
    elif x_pos > HIGH:
        x_pos = HIGH
    
    if y_pos < LOW:
        y_pos = LOW
    elif y_pos > HIGH:
        y_pos = HIGH

    return np.array([x_pos, y_pos])

def objective_function(position):
    # Gathered from nitri.ac.in
    if valid_pos(position) is False:
        return -1
    x_pos, y_pos = np.float32(position)
    a = np.math.pow(x_pos, 2)
    b = np.math.pow(y_pos, 2)
    return a*b

def pso(particles: List[Particle]):
    """
    Returns local_bests & global_best
    (particles, local_bests) -> (local_bests, global_best)
    """
    inertia_weight = st.norm(loc=1, scale=.5).rvs(size=1)
    c1, c2 = st.norm(loc=2, scale=.5).rvs(size=2)

    swarm_best_position = particles[0].local_best_position

    for i in np.arange(0, len(particles)):
        if objective_function(swarm_best_position) < objective_function(particles[i].local_best_position):
            swarm_best_position = particles[i].local_best_position
    
    for i in np.arange(STEPS):
        for j in np.arange(N_PARTICLES):
            particles[j].velocity = inertia_weight * particles[j].velocity +\
                c1*np.random.random()*(particles[j].local_best_position - particles[j].position) +\
                    c2*np.random.random()*(swarm_best_position - particles[j].position)

            _position = particles[j].position
            _velocity = particles[j].velocity

            particles[j].position += particles[j].velocity
            particles[j].position = np.float64(fix_position(particles[j].position))

            candidate = objective_function(particles[j].position)

            if candidate > objective_function(particles[j].local_best_position):
                particles[j].local_best_position = particles[j].position
            
            if objective_function(particles[j].local_best_position) > objective_function(swarm_best_position):
                swarm_best_position = particles[j].local_best_position

            particles[j].position_trace[i] = _position
            particles[j].velocity_trace[i] = _velocity

    return swarm_best_position, particles

def plot_particles(particles: List[Particle]):
    # Plot only first particle
    particle = particles[0]
    plt.plot(particle.position_trace)
    pass

if __name__=="__main__":
    particles = initialize()
    swarm_best_position, particles = pso(particles=particles)
    print(swarm_best_position)
    best_locals = np.array([particle.local_best_position for particle in particles])
    best_unique_locals = np.unique(best_locals, axis=0)
    print("====================")
    print(best_unique_locals)
    # for particle in particles:
    #     print(f"\t -> {particle.local_best_position}")
    plot_particles(particles)
