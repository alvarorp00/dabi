import numpy as np

#####################
# Fitness functions #
#####################


def sphere(vector):  # Sphere target function
    return np.sum(np.power(vector, 2))


# def rastrigin(vector, A=10):  # Rastrigin target function
#     return A + np.sum(np.power(vector, 2) - A * np.cos(2*np.math.pi*vector))

def rastrigin(vector):
    return np.float64(np.sum(np.power(vector, 2) -
                      10 * np.cos(2*np.math.pi*vector) + 10))


def rosenbrock(vector, A=100, B=20):  # Rosenbrock target function
    return np.float64(
        np.math.exp(-np.sum(np.array([(A*(vector[i+1]-vector[i]**2)**2 +
                    (1-vector[i])**2)/B for i in range(len(vector) - 1)])))
    )


def griewank(vector, A=1, B=1):  # Griewank target function
    return np.float64(
        np.math.exp(
                -np.sum(np.array([A*(vector[i]**2)/4000
                        for i in range(len(vector))]))
                - np.math.exp(np.sum(np.array(
                    [np.math.cos(vector[i]/np.math.sqrt(i+1))
                     for i in range(len(vector))]))/B)
        )
    )
