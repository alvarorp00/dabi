# %% [markdown]
# # GridWorld 2:
# 
# *GridWorld* is a world in the form of a board widely used as test environment in Reinforcement Learning. This board has several types of cells: initial, free, obstacles, terminal... and now teleportation points! The agents must go from the initial cell to the terminal one avoiding the obstacles and traveling the minimum distance.

# %% [markdown]
# Packages required for *GridWorld 2*:

# %%
import logging
import random
from typing import List
from scipy import stats as st

import numpy as np
import math

# %% [markdown]
# Functions to display information: 

# %%
def printMap(world):
  # Shows GridWorld map
  m = "["
  for i in range(world.size[0]):
    for j in range(world.size[1]):
      if world.map[(i, j)] == 0: 
        m += " O "
      elif world.map[(i, j)] == -1:
        m += " X " 
      elif world.map[(i, j)] == 1:
        m += " F "
      elif world.map[(i, j)] == 2:
        m += " T "
    if i == world.size[0] - 1:
      m += "]\n"
    else:
      m += "\n"
  print(m)

def printPolicy(world, policy):
  # Shows policy
  p = "["
  for i in range(world.size[0]):
    for j in range(world.size[1]):
      if policy[i][j] == 0:
        p += " ^ "
      elif policy[i][j] == 1:
        p += " V "
      elif policy[i][j] == 2:
        p += " < "
      else:
        p += " > "
    if i == world.size[0] - 1:
      p += "]\n" 
    else:
      p += "\n"
  print(p)

# %% [markdown]
# # *World* class: 
# 
# This class stores the information of the world:
# 
# *   *Map*: Matrix that encodes the world with free cells (0), obstacles (-1) and terminal cells (1)
# *   *Size*: Vector with the size of the world encoding matrix (width, height)
# 
# The following data is required to create a world:
# 
# *   Map size (width, height)
# *   Terminal cell list
# *   Obstacle cell list
# *   Teletransportation
# 
# Notes:
# * When the agent falls in an obstacle, it is trapped there forever.
# * When the agent enters in a teletransportation point, it exits through the other.
# 
# For instance: 
# 
# w = World((10, 10), [(9, 9)], [(2, 4), (4, 2)], [(0, 2), (9, 7)])
# 
# Creates a world with 10 rows and 10 columns with a terminal state (9, 9), two obstacles in (2, 4) and (4, 2), and a teletransportation system between (0, 2) and (9, 7).
# 
# ![map2.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmUAAAJsCAMAAACLXiTdAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAIBUExURQAAAP///////////////////////////////////////////////////////////////////////////////////////////////////////////////////wAAAAUGBwkKDAkMDw4PDw4RFhAIAxIVGBMXHhcdJBsfJBwjLB0eHyERByEoMyUqMSUtOSkyPy00PC03RTAZCjA8SzRAUTY+SDdFVjtJXDw9QD9IVD9OYkIjDkRUakdRX0lLTk9aaVErEVZjc1hbXl5rfWIzFGVzh2x7kHE8F3KDmXN2e3qMo39DGoCDiYSXsIqUoouZrYyQlo1LHZKaqpObqZOkupico5tSIJ6dpqCdpaCls6KqtaOxxKSor6hZI66xurK+zrVgJbWjnbm/xrm/x7u8xLykmsFmKMLL2MXJ0sXK0cbM1MnIzcnS3cqnlM1sKtHY4dHY4tPV29PX3NWxnNbT1dbc5dbd5dnf59t0Ldu+rtzf49zi6eDl6+Hl7OLFsuLn7ePe3uXn6uauiebWzuewiujr8Ojs8OmxjOuzjevu8u19Me2vhu7w9PDy9fGzifG4kPKwhPK0ifOxhPOyhvP19/Sxg/SyhfW7k/X2+PX3+fa+mPf5+vjOsfjOsvj5+vn6+/rey/r7/PvfzPv8/Pzn2Pzt4v3v5f3z6/307f717////4pxjDoAAAAedFJOUwAfICYuMTxASlhecICIkJifoKeorrC2uL3AxczT29CMfPIAAAAJcEhZcwAAFxEAABcRAcom8z8AAEdASURBVHhe7Z37n2THedYHkASSCCEJCVGCxuMgO0qMlXXsjbOWh/Wy1njcZsjduZMbicBsInHJrkBZlkt2IckuCbCDIRiJSwL4/JW8dc7b51R1v0/VU++ke7WZ9/uDdHo/83ZXPf2tOpc+3XUQBEEQBAHDn3nppb/o5i+99Fd1y8E3vvQtuuXgW176Jt1y8K0vfaNuOXjppW/QrX6+4WJhf6tuOfimJxj2cwcHzw1BsFNePDh4dhjecvPu8L5uObg3/PHX3PzxcF+fxsHj4Z5uORiG27rVz+2Lhf1YtxzcHx7qloOHFwv7hYODPzsMZ25+dnikWw5+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)

# %%
class World:

  def __init__(self, size, terminal, obstacle, hole):
    # Creates the world
    self.size = size
    self.map = {}
    for i in range(size[0]):
      for j in range(size[1]):
        # Free states
        self.map[(i, j)] = 0
        # Terminal states
        for t in terminal:
          if i==t[0] and j==t[1]:
            self.map[(i, j)] = 1
        # Obstacles
        for o in obstacle:
          if i==o[0] and j==o[1]:
            self.map[(i, j)] = -1
        # Teletransportation
        for h in hole:
          if i==h[0] and j==h[1]:
            self.map[(i, j)] = 2

# %% [markdown]
# Test for the *World* class:

# %%
if __name__ == "__main__":
  w = World((10, 10), [(9, 9)], [(2, 4), (4, 2)], [(0, 2), (9, 7)])
  printMap(w)

# %% [markdown]
# # *Agent* class:
# 
# This class controls the agent that learns by Reinforce Learning in *GridWorld*. 
# 
# The following data is required to create an agent:
# 
# *   *World*: World of the agent.
# *   *Initial State*: Initial state of the agent.
# 
# The following methods are used to control the agent:
# 
# *   *nextState = move(state, action)*: Moves the agent from *state* to *nextState* applying *action*.
# *   *reward = reward(nextState)*: Returns the *reward* received by the agent when going to *nextState*.
# *   *nextState, reward = checkAction(state, action)*: Checks the *nextState* and *reward* when the agent takes the *action* in the *state*. This method do not change the internal state of the agent, so it can be used to sweep the state space.
# *   *nextState, reward = executeAction(action)*: Executes the *action* in the current state and returns the *nextState* and *reward*. This method changes the internal state of the agent, so it should only be used when the agent travels along the world.
# 
# Note: You can change some properties of the agent (reward distribution, behavior with obstacles...) to improve the performance of the algorithms. 

# %%
class Agent:

  def __init__(self, world, initialState):
    # Create an agent
    self.world = world
    self.state = np.array(initialState)

  def move(self, state, action):
    # Manage state transitions
    nextState = state + np.array(action)
    if nextState[0] < 0:
      nextState[0] = 0
    elif nextState[0] >= self.world.size[0]:
      nextState[0] = self.world.size[0] - 1
    if nextState[1] < 0:
      nextState[1] = 0
    elif nextState[1] >= self.world.size[1]:
      nextState[1] = self.world.size[1] - 1
    if self.world.map[(nextState[0], nextState[1])] == 2:
      aux = nextState
      for i in range(self.world.size[0]):
        for j in range(self.world.size[1]):
          if self.world.map[(i, j)] == 2 and (nextState[0] != i and nextState[1] != j):
            aux = np.array([i, j])
            nextState = aux
    return nextState

  def reward(self, nextState):
    # Manage rewards
    if self.world.map[(nextState[0], nextState[1])] == -1:
      # Reward when the agent moves to an obstacle
      reward = -1 # ** Try different values **
    elif self.world.map[(nextState[0], nextState[1])] == 1:
      # Reward when the agent moves to a terminal cell
      reward = 1 # ** Try different values **
    else:
      # Reward when the agent moves to a free cell
      reward = 0 # ** Try different values ** 
    return reward

  def checkAction(self, state, action):
    # Plan the action
    nextState = self.move(state, action)
    if self.world.map[(state[0], state[1])] == -1: 
      nextState = state                            
    reward = self.reward(nextState)
    return nextState, reward

  def executeAction(self, action):
    # Plan and execute the action
    nextState = self.move(self.state, action)
    if self.world.map[(self.state[0], self.state[1])] == -1: 
      nextState = self.state     
    else: 
      self.state = nextState                                 
    reward = self.reward(nextState)
    return self.state, reward  

# %% [markdown]
# Test for the *Agent* class:

# %%
if __name__ == "__main__":
  # Create the world
  w = World((10, 10), [(9, 9)], [(2, 4), (4, 2)], [(0, 2), (9, 7)])
  printMap(w)
  # Create the agent
  a = Agent(w, (0, 0))
  # Move the agent through the main diagonal
  for i in range(1, 5):
    # Show the sates and rewards
    print(a.executeAction((0, 1)))

# %% [markdown]
# # Class work:
# 
# In this work you are going to implement the two most common value-based methods in reinforcement learning: SARSA and QLearning. In addition, you are going to test both algorithms in a set of scenarios to check if they work and compare their performance.
# 
# ## Worlds: 
# 
# The following worlds are provided in multiple sizes to test the algorithms: 
# 
# *   World 1: Easy maze that can be solved in zigzag.
# *   World 2: Random obstacles and useful teletransportation.
# *   World 3: Random obstacles and bad teletransportation.
# *   World 4: Hard maze with right and wrong ways.
# 
# Note: Feel free to use some of these scenarios or create your own scenarios. 

# %%
if __name__ == "__main__":
  
  # Word 1 small
  obstacles = []
  for j in range(0, 4):
    obstacles.append((j, 1))
  for j in range(1, 5):
    obstacles.append((j, 3))
  w1p = World((5, 5), [(4, 4)], obstacles, [])
  print("World 1: ")
  printMap(w1p)

  # World 1 medium
  obstacles = []
  for i in [1, 5]:
    for j in range(0, 8):
      obstacles.append((j, i))
  for i in [3, 7]:
    for j in range(1, 9):
      obstacles.append((j, i))
  w1m = World((9, 9), [(8, 8)], obstacles, [])
  print("World 1: ")
  printMap(w1m)

  # World 1 big
  obstacles = []
  for i in [1, 5, 9, 13, 17]:
    for j in range(0, 20):
      obstacles.append((j, i))
  for i in [3, 7, 11, 15, 19]:
    for j in range(1, 21):
      obstacles.append((j, i))
  w1g = World((21, 21), [(20, 20)], obstacles, [])
  print("World 1: ")
  printMap(w1g)

  # World 2 small
  obstacles = []
  for i in range(3):
    obstacles.append((np.random.randint(1, 4), np.random.randint(1, 4)))  
  w2p = World((5, 5), [(4, 4)], obstacles, [(2, 0), (4, 2)])
  print("World 2: ")
  printMap(w2p)

  # World 2 medium
  obstacles = []
  for i in range(10):
    obstacles.append((np.random.randint(1, 9), np.random.randint(1, 9)))  
  w2m = World((10, 10), [(9, 9)], obstacles, [(3, 1), (8, 6)])
  print("World 2: ")
  printMap(w2m)

  # World 2 big
  obstacles = []
  for i in range(50):
    obstacles.append((np.random.randint(1, 19), np.random.randint(1, 19)))  
  w2g = World((21, 21), [(20, 20)], obstacles, [(6, 2), (18, 14)])
  print("World 2: ")
  printMap(w2g)

  # World 3 small
  obstacles = []
  for i in range(3):
    obstacles.append((np.random.randint(1, 4), np.random.randint(1, 4)))  
  w3p = World((5, 5), [(4, 4)], obstacles, [(4, 0), (0, 4)])
  print("World 3: ")
  printMap(w3p)

  # World 3 medium
  obstacles = []
  for i in range(10):
    obstacles.append((np.random.randint(1, 9), np.random.randint(1, 9)))  
  w3m = World((10, 10), [(9, 9)], obstacles, [(8, 1), (1, 8)])
  print("World 3: ")
  printMap(w3m)

  # World 3 big
  obstacles = []
  for i in range(50):
    obstacles.append((np.random.randint(1, 19), np.random.randint(1, 19)))  
  w3g = World((21, 21), [(20, 20)], obstacles, [(18, 2), (2, 18)])
  print("World 3: ")
  printMap(w3g)

  # World 4
  obstacles = [(0,1),(0,3),(0,9),(0,15),(0,16),(0,17),(0,19),
               (1,1),(1,3),(1,4),(1,5),(1,6),(1,7),(1,9),(1,10),(1,11),(1,12),(1,13),(1,17),(1,19),
               (2,1),(2,9),(2,13),(2,15),(2,16),(2,17),(2,19),
               (3,1),(3,3),(3,5),(3,7),(3,9),(3,11),(3,16),(3,19),
               (4,3),(4,5),(4,7),(4,8),(4,9),(4,10),(4,11),(4,12),(4,13),(4,14),(4,16),(4,18),(4,19),
               (5,0),(5,1),(5,2),(5,3),(5,5),(5,9),(5,16),
               (6,5),(6,6),(6,7),(6,9),(6,10),(6,11),(6,12),(6,13),(6,14),(6,16),(6,17),(6,19),
               (7,0),(7,1),(7,2),(7,3),(7,5),(7,7),(7,9),(7,19),
               (8,3),(8,7),(8,8),(8,9),(8,12),(8,13),(8,14),(8,15),(8,16),(8,17),(8,18),(8,19),
               (9,1),(9,3),(9,5),(9,7),(9,11),(9,12),(9,19),(9,20),
               (10,1),(10,3),(10,5),(10,6),(10,7),(10,9),(10,11),(10,14),(10,15),(10,16),(10,17),
               (11,1),(11,3),(11,5),(11,9),(11,11),(11,13),(11,14),(11,17),(11,18),(11,19),
               (12,1),(12,5),(12,6),(12,8),(12,9),(12,11),(12,13),(12,19),
               (13,1),(13,2),(13,3),(13,4),(13,5),(13,8),(13,15),(13,16),(13,17),(13,19),
               (14,4),(14,7),(14,8),(14,10),(14,12),(14,13),(14,15),(14,19),
               (15,0),(15,1),(15,2),(15,6),(15,7),(15,10),(15,13),(15,14),(15,15),(15,17),(15,18),(15,19),(15,20),
               (16,2),(16,3),(16,5),(16,6),(16,7),(16,8),(16,9),(16,10),(16,11),(16,15),(16,17),
               (17,0),(17,3),(17,5),(17,9),(17,13),(17,14),(17,15),(17,17),(17,19),
               (18,0),(18,1),(18,5),(18,6),(18,7),(18,9),(18,10),(18,11),(18,15),(18,19),
               (19,1),(19,2),(19,4),(19,5),(19,11),(19,13),(19,14),(19,15),(19,16),(19,17),(19,18),(19,19),
               (20,7),(20,8),(20,9),(20,11),(20,19)]          
  print("World 4: ")
  w4 = World((21, 21), [(20, 20)], obstacles, [])
  printMap(w4)

# %% [markdown]
# ## SARSA:
# 
# *SARSA* (State-Action-Reward-State-Action) is a value-based method that solves reinforcement learning problems. *SARSA* computes iteratively the value function $Q(S,A)$ and then determines the optimal policy $\pi$.
# 
# *SARSA* received this name because of the five variables involved in its update function: current state ($S_t$), current action ($A_t$), current reward ($R_t$), next state ($S_{t+1}$) and next action ($A_{t+1}$). This equation is shown below:
# 
# \begin{equation}
# Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha [R_t + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t,A_t)]
# \end{equation}
# 
# Note: $\alpha$ is the episode length and $\gamma$ the discount factor.
# 
# The *SARSA* algorithm follows this scheme: 
# 
# 1.   Initialize $Q(S,A)$ for every state and action 
# 2.   **Loop** (repeat $3-9$ until convergence):
# 3.   Initialize $S_t$
# 4.   Choose $A_t$ for $S_t$ following the policy $Q(S,A)$
# 5.   **Loop** (repeat $6-9$ until $S_t$ is terminal):
# 6.   Take $A_t$ in $S_t$ and observe $R_t$ and $S_{t+1}$
# 7.   Choose $A_{t+1}$ for $S_{t+1}$ following the policy $Q(S,A)$
# 8.   Update the value $Q(S_t, A_t)$ with the update equation
# 9.   Take $S_{t+1}$ and $A_{t+1}$ as the new $S_t$ and $A_t$
# 
# The *SARSA* algorithm uses a parameter $\epsilon \in (0, 1)$ to search a balance between exploration and exploitation. When it chooses $A_t$ for $S_t$, if a random number is less than $\epsilon$, it will take a random action; whereas if that number is more than $\epsilon$, it will take the best action.
# 
# ## Exercise 1:
# 
# Implement the SARSA algorithm for the previously defined agent and world.

# %% [markdown]
# Algorithm is defined below, check the class _ReinforcedLearning_ with the common attributes and functions of SARSA and QLearning and the inner implementations of both algorithms, both defined as _Sarsa_ and _QLearning_ classes that extends the _ReinforcedLearning_ class.

# %% [markdown]
# Below the definition of both algorithms some tests can be found

# %% [markdown]
# ## Q-Learning:
# 
# *Q-Learning* is the most common value-based method to solve reinforcement learning problems. This algorithm receives this name from $Q(S,A)$, the value function that updates during its execution. *Q-Learning* is very similar to *SARSA*, but it uses a different update equation:
# 
# \begin{equation}
# Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha [R_t + \gamma max_a{Q(S_{t+1}}, a) - Q(S_t,A_t)]
# \end{equation}
# 
# In this case, the action $A_{t+1}$ in $S_{t+1}$ is taken to exploit the maximum value.
# 
# The *Q-Learning* algorithm follows this scheme: 
# 
# 1.   Initialize $Q(S,A)$ for every state and action 
# 2.   **Loop** (repeat $3-8$ until convergence):
# 3.   Initialize $S_t$
# 4.   **Loop** (repeat $6-8$ until $S_t$ is terminal):
# 5.   Choose $A_t$ for $S_t$ following the policy $Q(S,A)$
# 6.   Take $A_t$ in $S_t$ and observe $R_t$ and $S_{t+1}$
# 7.   Update the value $Q(S_t, A_t)$ with the update equation
# 8.   Take $S_{t+1}$ as the new $S_t$
# 
# ## Exercise 2:
# Implement the Q-Learning algorithm for the previously defined agent and world.

# %% [markdown]
# Algorithm is defined below, check the class _ReinforcedLearning_ with the common attributes and functions of SARSA and QLearning and the inner implementations of both algorithms, both defined as _Sarsa_ and _QLearning_ classes that extends the _ReinforcedLearning_ class.

# %% [markdown]
# Below the definition of both algorithms some tests can be found

# %% [markdown]
# ## Analysis:
# 
# *SARSA* and *Q-Learning* are very similar, they can be applied to the same problems and usually obtain the same solutions. However, the results of both algorithms can be different in certains problems: e.g., in the Cliffworld, SARSA performs safer movements but obtains less value than Q-Learning ([interesting article](https://medium.com/gradientcrescent/fundamentals-of-reinforcement-learning-navigating-cliffworld-with-sarsa-and-q-learning-cc3c36eb5830)).
# 
# ## Exercise 3:
# 
# Analyze the results of both algorithms:
# 
# 1.   Performance of SARSA and Q-Learning: Which problems do they solve? When do they find the optimal solution? What are the causes of these results?
# 
# 2.   Differences between SARSA and Q-Learning: Which algorithm solves more scenarios? Which one converges faster? Which one obtains more value?
# 
# Note: The following variables can be interesting to analyze the results: Diference between resultant and optimal policies, number of iterations required to converge, total return of the problem, and return per episode.
# 
# 3.   Differences with more exploration (higher $\epsilon$) and more exploitation (lower $\epsilon$). Which converges faster? Which maximizes return?
# 
# 4.   Differences with other parameters: number of episodes, learning rate ($\alpha$), and discount factor ($\gamma$). Which combination gives the best results?

# %%
# Solution: Code required to generate results

# %% [markdown]
# **Solution**: Comments on the results.

# %% [markdown]
# Note: Feel free to add all the text and code blocks that you need to answer the questions.

# %% [markdown]
# ## Classes

# %%
# Solution COMMON class

class ReinforcedLearning():
  def __init__(self, *args, **kwargs):
    self._parameters = kwargs.copy()
    self.__config = dict({
      'ready': False
    })

  @property
  def world(self) -> World:
    return self._parameters['world']

  @property
  def agent(self) -> Agent:
    return self._parameters['agent']

  @property
  def actions(self) -> List:
    return self._parameters['actions']

  @property
  def alpha(self) -> float:
    return self._parameters['alpha']

  @property
  def gamma(self) -> float:
    return self._parameters['gamma']

  @property
  def epsilon(self):
    return self._parameters['epsilon']

  @property
  def q(self) -> np.ndarray:
    return self._parameters['q']

  @property
  def config(self):
    return self.__config.copy()

  def set_alpha(self, alpha):
    self._parameters['alpha'] = alpha

  def set_gamma(self, gamma):
    self._parameters['gamma'] = gamma

  def set_epsilon(self, epsilon):
    self._parameters['epsilon'] = epsilon

  def configure(self, **kwargs):
    self.__config = kwargs.copy()
    if 'max_iterations' not in self.__config:
      self.__config['max_iterations'] = np.int32(10**3)  # Run for at most this time
    if 'max_steps' not in self.__config:
      self.__config['max_steps'] = np.int32(10**2)  # Each iteration will run for this steps
    if 'theta' not in self.__config:
      self.__config['theta'] = np.float32(0.01)  # Difference threshold
    if 'init_value' not in self.__config:
      self.__config['init_value'] = 0
    if 'q' not in self.__config:
      self._parameters['q'] = np.ones(
        (self.world.size[0], self.world.size[1], len(self.actions))
      ) * (self.__config['init_value']) # Q(S,A) = init_value (0, -1, 1, etc)
    if 'qs' not in self.__config:
      self._parameters['qs'] = [self.q.copy()]  # Q matrix history
    self.__config['ready'] = True

    __rows, __cols = self.world.size
    for __j in range(__rows):
      for __i in range(__cols):
        if self.world.map[(__j, __i)] == 1:  # Q(Terminal, A) = 0
          self._parameters['q'][__j][__i] = np.zeros(shape=(len(self.actions,)))

  def _lookup_action(self, action_idx):
    return self.actions[action_idx]

  def _choose_action(self, state):
    """
    Chooses an action (index) either randomly or maximizing the reward in a given state
    :param state: current state where the action will be taken
    :return: index of action to be taken
    """
    __rnd = st.uniform.rvs()
    if __rnd < self.epsilon:
      __action = np.random.choice(np.arange(0, len(self.actions)), size=1)[0]
    else:
      __action = np.argmax(self._lookup_q(state))
    return __action

  def _difference(self, qs: List[np.ndarray], epoch):
    __diff_p = np.abs(np.subtract(qs[epoch-1], qs[epoch-2]))
    __q_max = np.max(np.abs(self.q))
    if __q_max > 0:
      __diff = np.max(__diff_p) / __q_max
    else:
      __diff = 1
    return __diff

  def _lookup_q(self, state, action:int=None):
    if action is not None:
      return self.q[state[0]][state[1]][action]
    else:
      return self.q[state[0]][state[1]]

  def _set_q(self, state, action, value):
    self._parameters['q'][state[0]][state[1]][action] = value

  def _update_q(self, state, next_state, action, next_action):
    """
    Performs
    \begin{equation}
      Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha [R_t + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t,A_t)]$
    \end{equation}
    :param state: current state
    :param next_state: next state
    :param action: current action idx
    :param next_action: next action idx
    """
    __curr = self._lookup_q(state, action)  # Q(S_t,A_t)
    _, __reward = self.agent.checkAction(state, self._lookup_action(action))
    __next = __reward + self.gamma * self._lookup_q(next_state, next_action)  # R_t + \gamma Q(S_{t+1}, A_{t+1})
    __new_value = __curr + self.alpha * (__next - __curr)  # Whole equation
    self._set_q(state, action, __new_value)

  def run(self):
    return NotImplementedError('Method not implemented in abstract class')

  @DeprecationWarning
  def solve(self):
    """
    Deprecated: use Sarsa.getPolicy and call printPolicy instead
    
    Returns the best path starting from agent's current state
    :return: Tuple(List, Bool) --> (path_array, path_found?)
    """
    __epsilon = self.epsilon
    self._parameters['epsilon'] = 0  # always choose the best action
    __starter_state = self.agent.state
    
    __convert = lambda s: (s[0], s[1])  # converts state to tuple so it can be hashed
    __path = [__starter_state.tolist()]
    __path_set = set()
    __path_set.add(__convert(__starter_state))
    
    __found = True
    while True:
      __action = self._choose_action(self.agent.state)
      __prev_state = self.agent.state
      __next_state, __reward = self.agent.executeAction(self._lookup_action(__action))
      __path.append(__next_state.tolist())
      if __convert(__next_state) in __path_set:  # state was already visited
          logging.warning(f'Agent traveling through visited state {__prev_state}. Exiting with no path.')
          __found = False
          break
      __path_set.add(__convert(__next_state))  # add next state to the path
      if __reward == 1:
        break
    self._parameters['epsilon'] = __epsilon  # Restore epsilon value
    self.agent.state = __starter_state
    return __path, __found

  def print_q(self):
    __q = self.q.copy()
    __rows, __cols = self.world.size
    print('[')
    for __j in range(__rows):
      print(f'  Row {__j} [')
      for __i in range(__cols):
        print(f'    Col {__i}' + str(__q[__j][__i]))
      print('  ]')
    print(']')

  def get_policy(self):
    __policy = np.zeros(shape=self.world.size)
    for __i in range(self.world.size[0]):
      for __j in range(self.world.size[1]):
        __policy[__i][__j] = np.argmax(self.q[__i][__j])
    return __policy

# %%
# Solution: SARSA code

class Sarsa(ReinforcedLearning):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def run(self):
    """
    Runs SARSA algorithm
    """
    
    if self.config['ready'] is False:
      logging.critical('Sarsa is not configured. Run sarsa.configure')
      return
    __max_iterations = self.config['max_iterations']
    __max_steps = self.config['max_steps']
    __theta = self.config['theta']
    __qs = self._parameters['qs']  # Q history --> current: starter Q
    __epoch = 0
    while True:
      # self.agent.state = __starter_state  # Set S_t
      __state = self.agent.state
      __action = self._choose_action(__state)  # Choose A_t
      # print(f'Starter action: {__action}')
      __c = 0
      __stop = False
      while __stop is False:
        # print(f'Action: {self._lookup_action(__action)} @ idx.{__action}')
        __next_state, __reward = self.agent.checkAction(__state, self._lookup_action(__action))  # Check next state and reward with chosen action --> S_{t+1}, R_t
        # print(f'Next state: {__next_state}')
        __next_action = self._choose_action(__next_state)  # Choose A_{t+1}
        self._update_q(state=__state, next_state=__next_state, action=__action, next_action=__next_action)  # Update Q(S_t, A_t) with the update equation

        if __reward == 1:  # Terminal state reached --> exit
          break

        # __nextState, reward = self.agent.checkAction(__state, self._lookup_action(__action))  # Take S_{t} = S_{t+1} <-- Update current state with the next state
        __state = __next_state  # Take S_{t} = S_{t+1}
        __action = __next_action  # Take A_{t} = A_{t+1} <-- Update current action with the next action

        __stop = self.world.map[(__state[0], __state[1])] == 1 # or\  # Terminal state reached
                      # self.world.map[(__state[0], __state[1])] == -1  # Either terminal state or obstacle reached
        if __max_steps is not None:
          __stop = __stop or __c >= __max_steps  # If we've reached max steps
        __c += 1
      __qs.append(self.q.copy())
      if __theta is not None and __theta > 0:
        if self._difference(qs=__qs, epoch=__epoch) < __theta:  # Convergence reached
          break
      __epoch += 1
      if __max_iterations is not None:  # We can choose to iterate until convergence
        if __epoch >= __max_iterations:
          break
    # self.agent.state = __starter_state  # Restore agent's position
    # print('SARSA finished')
    return __qs

# %%
# Solution QLearning code

class QLearning(ReinforcedLearning):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def _choose_next_action(self, state):
    """
    Chooses an action (index) maximizing the reward in a given state
    :param state: current state where the action will be taken
    :return: index of action to be taken
    """
    return np.argmax(self._lookup_q(state))

  def run(self):
    """
    Runs QLearning algorithm
    """
    
    if self.config['ready'] is False:
      logging.critical('QLearning is not configured. Run qlearning.configure')
      return
    __max_iterations = self.config['max_iterations']
    __max_steps = self.config['max_steps']
    __theta = self.config['theta']
    __qs = self._parameters['qs']  # Q history --> current: starter Q
    __epoch = 0
    while True:
      __state = self.agent.state  # Set S_t
      __c = 0
      __stop = False
      while __stop is False:
        __action = self._choose_action(__state)  # Choose A_t following Q(S,A) policy
        # print(f'Action: {self._lookup_action(__action)} @ idx.{__action}')
        __next_state, __reward = self.agent.checkAction(__state, self._lookup_action(__action))  # Check next state and reward with chosen action --> S_{t+1}, R_t
        # print(f'Next state: {__next_state}')
        __next_action = self._choose_next_action(__state)  # Choose A_{t+1} <-- argmax(Q(S_{t+1}, a))
        self._update_q(state=__state, next_state=__next_state, action=__action, next_action=__next_action)  # Update Q(S_t, A_t) with the update equation
        if __reward == 1:  # Terminal state reached --> exit
          break
        __state = __next_state
        __stop = self.world.map[(__state[0], __state[1])] == 1 # or\  # Terminal state reached
                      # self.world.map[(__state[0], __state[1])] == -1  # Either terminal state or obstacle reached
        if __max_steps is not None:
          __stop = __stop or __c >= __max_steps  # If we've reached max steps
        __c += 1
      __qs.append(self.q.copy())
      if __theta is not None and __theta > 0:
        if self._difference(qs=__qs, epoch=__epoch) < __theta:  # Convergence reached
          break
      __epoch += 1
      if __max_iterations is not None:  # We can choose to iterate until convergence
        if __epoch >= __max_iterations:
          break
    # print('QLearning finished')
    return __qs

# %% [markdown]
# ## Tests

# %%
w = w1p
start = (0,0)
agent = Agent(w, start)
actions = np.array([(-1,0), (1,0),(0,-1), (0,1)])  # up, down, left, right

print("\n\nRUNNING SARSA \n\n")

sarsa = Sarsa(
  world=w,
  agent=agent,
  actions=actions,
  alpha=.01,
  gamma=.9,
  epsilon=.9
)

# max_iterations = None --> run until theta converges
# max_steps = None --> run until state reached is terminal or fallen into obstacle
# theta = [None | 0] --> run given number of max_iterations independently of theta
# init_value --> starting value of cells except terminal states
# WARNING: do not set max_iterations and theta None at the same time as it won't finish
sarsa.configure(max_iterations=None, max_steps=10**4, theta=1e-12, init_value=0)
sarsa.run()

policy = sarsa.get_policy()
printPolicy(sarsa.world, policy)

pass

# %%
printMap(sarsa.world)
sarsa.print_q()

# %%
# We can try to run the algorithm multiple times varying epsilon factor
# max_attemps = 10
# for i in range(0, max_attemps):
#   __epsilon = np.float32(1 - (i / max_attemps + 1e-3))
#   print(f'Run: [{i+1}/{max_attemps}] @ Epsilon: {__epsilon}')
#   sarsa.set_epsilon(__epsilon)
#   sarsa.run()
#   path, path_found = sarsa.solve()
#   # print(f'Path: {path} @ Found: {path_found}')
#   if path_found:
#     break
# #
# if path_found is False:
#   print(f'Extra @ Epsilon: {.5}')
#   sarsa.set_epsilon(.5)
#   sarsa.run()

# if path[1] is False:
#   sarsa.set_epsilon(.1)
#   path = sarsa.solve()

# %%
# w = w1p
# start = (0,0)
# agent = Agent(w, start)
# actions = np.array([(0,1), (1,0),(-1,0), (-1,0)])

# qlearning = QLearning(
#   world=w,
#   agent=agent,
#   actions=actions,
#   alpha=.01,
#   gamma=.9,
#   epsilon=.55
# )

# qlearning.configure(max_iterations=None, max_steps=10**3, theta=1e-6, init_value=0)
# qlearning.run()

# pass

# %%
# policy = qlearning.get_policy()
# printPolicy(qlearning.world, policy)

# %%
# printMap(qlearning.world)
# qlearning.print_q()
# path = qlearning.solve()
# print(path)
# pass

# %%



