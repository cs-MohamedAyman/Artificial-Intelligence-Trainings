# Deep Q-Learning - Lunar Lander

In this assignment, you will train an agent to land a lunar lander safely on a landing pad on the surface of the moon.


# Outline
- [ 1 - Import Packages <img align="Right" src="./images/lunar_lander.gif" width = 60% >](#1)
- [ 2 - Hyperparameters](#2)
- [ 3 - The Lunar Lander Environment](#3)
  - [ 3.1 Action Space](#3.1)
  - [ 3.2 Observation Space](#3.2)
  - [ 3.3 Rewards](#3.3)
  - [ 3.4 Episode Termination](#3.4)
- [ 4 - Load the Environment](#4)
- [ 5 - Interacting with the Gym Environment](#5)
    - [ 5.1 Exploring the Environment's Dynamics](#5.1)
- [ 6 - Deep Q-Learning](#6)
  - [ 6.1 Target Network](#6.1)
    - [ Exercise 1](#ex01)
  - [ 6.2 Experience Replay](#6.2)
- [ 7 - Deep Q-Learning Algorithm with Experience Replay](#7)
  - [ Exercise 2](#ex02)
- [ 8 - Update the Network Weights](#8)
- [ 9 - Train the Agent](#9)
- [ 10 - See the Trained Agent In Action](#10)
- [ 11 - Congratulations!](#11)
- [ 12 - References](#12)


<a name="1"></a>
## 1 - Import Packages

We'll make use of the following packages:
- `numpy` is a package for scientific computing in python.
- `deque` will be our data structure for our memory buffer.
- `namedtuple` will be used to store the experience tuples.
- The `gym` toolkit is a collection of environments that can be used to test reinforcement learning algorithms. We should note that in this notebook we are using `gym` version `0.24.0`.
- `PIL.Image` and `pyvirtualdisplay` are needed to render the Lunar Lander environment.
- We will use several modules from the `tensorflow.keras` framework for building deep learning models.
- `utils` is a module that contains helper functions for this assignment. You do not need to modify the code in this file.

Run the cell below to import all the necessary packages.


```python
import time
from collections import deque, namedtuple

import gym
import numpy as np
import PIL.Image
import tensorflow as tf
import utils

from pyvirtualdisplay import Display
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
```


```python
# Set up a virtual display to render the Lunar Lander environment.
Display(visible=0, size=(840, 480)).start();

# Set the random seed for TensorFlow
tf.random.set_seed(utils.SEED)
```

<a name="2"></a>
## 2 - Hyperparameters

Run the cell below to set the hyperparameters.


```python
MEMORY_SIZE = 100_000     # size of memory buffer
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps
```

<a name="3"></a>
## 3 - The Lunar Lander Environment

In this notebook we will be using [OpenAI's Gym Library](https://gymlibrary.ml/). The Gym library provides a wide variety of environments for reinforcement learning. To put it simply, an environment represents a problem or task to be solved. In this notebook, we will try to solve the Lunar Lander environment using reinforcement learning.

The goal of the Lunar Lander environment is to land the lunar lander safely on the landing pad on the surface of the moon. The landing pad is designated by two flag poles and it is always at coordinates `(0,0)` but the lander is also allowed to land outside of the landing pad. The lander starts at the top center of the environment with a random initial force applied to its center of mass and has infinite fuel. The environment is considered solved if you get `200` points.

<br>
<br>
<figure>
  <img src = "images/lunar_lander.gif" width = 40%>
      <figcaption style = "text-align: center; font-style: italic">Fig 1. Lunar Lander Environment.</figcaption>
</figure>



<a name="3.1"></a>
### 3.1 Action Space

The agent has four discrete actions available:

* Do nothing.
* Fire right engine.
* Fire main engine.
* Fire left engine.

Each action has a corresponding numerical value:

```python
Do nothing = 0
Fire right engine = 1
Fire main engine = 2
Fire left engine = 3
```

<a name="3.2"></a>
### 3.2 Observation Space

The agent's observation space consists of a state vector with 8 variables:

* Its $(x,y)$ coordinates. The landing pad is always at coordinates $(0,0)$.
* Its linear velocities $(\dot x,\dot y)$.
* Its angle $\theta$.
* Its angular velocity $\dot \theta$.
* Two booleans, $l$ and $r$, that represent whether each leg is in contact with the ground or not.

<a name="3.3"></a>
### 3.3 Rewards

The Lunar Lander environment has the following reward system:

* Landing on the landing pad and coming to rest is about 100-140 points.
* If the lander moves away from the landing pad, it loses reward.
* If the lander crashes, it receives -100 points.
* If the lander comes to rest, it receives +100 points.
* Each leg with ground contact is +10 points.
* Firing the main engine is -0.3 points each frame.
* Firing the side engine is -0.03 points each frame.

<a name="3.4"></a>
### 3.4 Episode Termination

An episode ends (i.e the environment enters a terminal state) if:

* The lunar lander crashes (i.e if the body of the lunar lander comes in contact with the surface of the moon).

* The lander's $x$-coordinate is greater than 1.

You can check out the [Open AI Gym documentation](https://gymlibrary.ml/environments/box2d/lunar_lander/) for a full description of the environment.

<a name="4"></a>
## 4 - Load the Environment

We start by loading the `LunarLander-v2` environment from the `gym` library by using the `.make()` method. `LunarLander-v2` is the latest version of the Lunar Lander environment and you can read about its version history in the [Open AI Gym documentation](https://gymlibrary.ml/environments/box2d/lunar_lander/#version-history).


```python
env = gym.make('LunarLander-v2')
```

Once we load the environment we use the `.reset()` method to reset the environment to the initial state. The lander starts at the top center of the environment and we can render the first frame of the environment by using the `.render()` method.


```python
env.reset()
PIL.Image.fromarray(env.render(mode='rgb_array'))
```




![png](output_10_0.png)



In order to build our neural network later on we need to know the size of the state vector and the number of valid actions. We can get this information from our environment by using the `.observation_space.shape` and `action_space.n` methods, respectively.


```python
state_size = env.observation_space.shape
num_actions = env.action_space.n

print('State Shape:', state_size)
print('Number of actions:', num_actions)
```

    State Shape: (8,)
    Number of actions: 4


<a name="5"></a>
## 5 - Interacting with the Gym Environment

The Gym library implements the standard “agent-environment loop” formalism:

<br>
<center>
<video src = "./videos/rl_formalism.m4v" width="840" height="480" controls autoplay loop poster="./images/rl_formalism.png"> </video>
<figcaption style = "text-align:center; font-style:italic">Fig 2. Agent-environment Loop Formalism.</figcaption>
</center>
<br>

In the standard “agent-environment loop” formalism, an agent interacts with the environment in discrete time steps $t=0,1,2,...$. At each time step $t$, the agent uses a policy $\pi$ to select an action $A_t$ based on its observation of the environment's state $S_t$. The agent receives a numerical reward $R_t$ and on the next time step, moves to a new state $S_{t+1}$.

<a name="5.1"></a>
### 5.1 Exploring the Environment's Dynamics

In Open AI's Gym environments, we use the `.step()` method to run a single time step of the environment's dynamics. In the version of `gym` that we are using the `.step()` method accepts an action and returns four values:

* `observation` (**object**): an environment-specific object representing your observation of the environment. In the Lunar Lander environment this corresponds to a numpy array containing the positions and velocities of the lander as described in section [3.2 Observation Space](#3.2).


* `reward` (**float**): amount of reward returned as a result of taking the given action. In the Lunar Lander environment this corresponds to a float of type `numpy.float64` as described in section [3.3 Rewards](#3.3).


* `done` (**boolean**): When done is `True`, it indicates the episode has terminated and it’s time to reset the environment.


* `info` (**dictionary**): diagnostic information useful for debugging. We won't be using this variable in this notebook but it is shown here for completeness.

To begin an episode, we need to reset the environment to an initial state. We do this by using the `.reset()` method.


```python
# Reset the environment and get the initial state.
initial_state = env.reset()
```

Once the environment is reset, the agent can start taking actions in the environment by using the `.step()` method. Note that the agent can only take one action per time step.

In the cell below you can select different actions and see how the returned values change depending on the action taken. Remember that in this environment the agent has four discrete actions available and we specify them in code by using their corresponding numerical value:

```python
Do nothing = 0
Fire right engine = 1
Fire main engine = 2
Fire left engine = 3
```


```python
# Select an action
action = 0

# Run a single time step of the environment's dynamics with the given action.
next_state, reward, done, info = env.step(action)

with np.printoptions(formatter={'float': '{:.3f}'.format}):
    print("Initial State:", initial_state)
    print("Action:", action)
    print("Next State:", next_state)
    print("Reward Received:", reward)
    print("Episode Terminated:", done)
    print("Info:", info)
```

    Initial State: [0.002 1.422 0.194 0.506 -0.002 -0.044 0.000 0.000]
    Action: 0
    Next State: [0.004 1.433 0.194 0.480 -0.004 -0.044 0.000 0.000]
    Reward Received: 1.1043263227541047
    Episode Terminated: False
    Info: {}


In practice, when we train the agent we use a loop to allow the agent to take many consecutive actions during an episode.

<a name="6"></a>
## 6 - Deep Q-Learning

In cases where both the state and action space are discrete we can estimate the action-value function iteratively by using the Bellman equation:

$$
Q_{i+1}(s,a) = R + \gamma \max_{a'}Q_i(s',a')
$$

This iterative method converges to the optimal action-value function $Q^*(s,a)$ as $i\to\infty$. This means that the agent just needs to gradually explore the state-action space and keep updating the estimate of $Q(s,a)$ until it converges to the optimal action-value function $Q^*(s,a)$. However, in cases where the state space is continuous it becomes practically impossible to explore the entire state-action space. Consequently, this also makes it practically impossible to gradually estimate $Q(s,a)$ until it converges to $Q^*(s,a)$.

In the Deep $Q$-Learning, we solve this problem by using a neural network to estimate the action-value function $Q(s,a)\approx Q^*(s,a)$. We call this neural network a $Q$-Network and it can be trained by adjusting its weights at each iteration to minimize the mean-squared error in the Bellman equation.

Unfortunately, using neural networks in reinforcement learning to estimate action-value functions has proven to be highly unstable. Luckily, there's a couple of techniques that can be employed to avoid instabilities. These techniques consist of using a ***Target Network*** and ***Experience Replay***. We will explore these two techniques in the following sections.

<a name="6.1"></a>
### 6.1 Target Network

We can train the $Q$-Network by adjusting it's weights at each iteration to minimize the mean-squared error in the Bellman equation, where the target values are given by:

$$
y = R + \gamma \max_{a'}Q(s',a';w)
$$

where $w$ are the weights of the $Q$-Network. This means that we are adjusting the weights $w$ at each iteration to minimize the following error:

$$
\overbrace{\underbrace{R + \gamma \max_{a'}Q(s',a'; w)}_{\rm {y~target}} - Q(s,a;w)}^{\rm {Error}}
$$

Notice that this forms a problem because the $y$ target is changing on every iteration. Having a constantly moving target can lead to oscillations and instabilities. To avoid this, we can create
a separate neural network for generating the $y$ targets. We call this separate neural network the **target $\hat Q$-Network** and it will have the same architecture as the original $Q$-Network. By using the target $\hat Q$-Network, the above error becomes:

$$
\overbrace{\underbrace{R + \gamma \max_{a'}\hat{Q}(s',a'; w^-)}_{\rm {y~target}} - Q(s,a;w)}^{\rm {Error}}
$$

where $w^-$ and $w$ are the weights the target $\hat Q$-Network and $Q$-Network, respectively.

In practice, we will use the following algorithm: every $C$ time steps we will use the $\hat Q$-Network to generate the $y$ targets and update the weights of the target $\hat Q$-Network using the weights of the $Q$-Network. We will update the weights $w^-$ of the the target $\hat Q$-Network using a **soft update**. This means that we will update the weights $w^-$ using the following rule:

$$
w^-\leftarrow \tau w + (1 - \tau) w^-
$$

where $\tau\ll 1$. By using the soft update, we are ensuring that the target values, $y$, change slowly, which greatly improves the stability of our learning algorithm.

<a name="ex01"></a>
### Exercise 1

In this exercise you will create the $Q$ and target $\hat Q$ networks and set the optimizer. Remember that the Deep $Q$-Network (DQN) is a neural network that approximates the action-value function $Q(s,a)\approx Q^*(s,a)$. It does this by learning how to map states to $Q$ values.

To solve the Lunar Lander environment, we are going to employ a DQN with the following architecture:

* An `Input` layer that takes `state_size` as input.

* A `Dense` layer with `64` units and a `relu` activation function.

* A `Dense` layer with `64` units and a `relu` activation function.

* A `Dense` layer with `num_actions` units and a `linear` activation function. This will be the output layer of our network.


In the cell below you should create the $Q$-Network and the target $\hat Q$-Network using the model architecture described above. Remember that both the $Q$-Network and the target $\hat Q$-Network have the same architecture.

Lastly, you should set `Adam` as the optimizer with a learning rate equal to `ALPHA`. Recall that `ALPHA` was defined in the [Hyperparameters](#2) section. We should note that for this exercise you should use the already imported packages:
```python
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
```


```python
# UNQ_C1
# GRADED CELL

# Create the Q-Network
q_network = Sequential([
    Input(shape=state_size),
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=num_actions, activation='linear'),
    ])

# Create the target Q^-Network
target_q_network = Sequential([
    Input(shape=state_size),
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=num_actions, activation='linear'),
    ])

optimizer = Adam(learning_rate=ALPHA)
```


```python
# UNIT TEST
from public_tests import *

test_network(q_network)
test_network(target_q_network)
test_optimizer(optimizer, ALPHA)
```

    [92mAll tests passed!
    [92mAll tests passed!
    [92mAll tests passed!



  <summary><font size="3" color="darkgreen"><b>Click for hints</b></font></summary>

```python
# Create the Q-Network
q_network = Sequential([
    Input(shape=state_size),
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=num_actions, activation='linear'),
    ])

# Create the target Q^-Network
target_q_network = Sequential([
    Input(shape=state_size),
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=num_actions, activation='linear'),
    ])

optimizer = Adam(learning_rate=ALPHA)
```

<a name="6.2"></a>
### 6.2 Experience Replay

When an agent interacts with the environment, the states, actions, and rewards the agent experiences are sequential by nature. If the agent tries to learn from these consecutive experiences it can run into problems due to the strong correlations between them. To avoid this, we employ a technique known as **Experience Replay** to generate uncorrelated experiences for training our agent. Experience replay consists of storing the agent's experiences (i.e the states, actions, and rewards the agent receives) in a memory buffer and then sampling a random mini-batch of experiences from the buffer to do the learning. The experience tuples $(S_t, A_t, R_t, S_{t+1})$ will be added to the memory buffer at each time step as the agent interacts with the environment.

For convenience, we will store the experiences as named tuples.


```python
# Store experiences as named tuples
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
```

By using experience replay we avoid problematic correlations, oscillations and instabilities. In addition, experience replay also allows the agent to potentially use the same experience in multiple weight updates, which increases data efficiency.

<a name="7"></a>
## 7 - Deep Q-Learning Algorithm with Experience Replay

Now that we know all the techniques that we are going to use, we can put them togther to arrive at the Deep Q-Learning Algorithm With Experience Replay.
<br>
<br>
<figure>
  <img src = "images/deep_q_algorithm.png" width = 90% style = "border: thin silver solid; padding: 0px">
      <figcaption style = "text-align: center; font-style: italic">Fig 3. Deep Q-Learning with Experience Replay.</figcaption>
</figure>

<a name="ex02"></a>
### Exercise 2

In this exercise you will implement line ***12*** of the algorithm outlined in *Fig 3* above and you will also compute the loss between the $y$ targets and the $Q(s,a)$ values. In the cell below, complete the `compute_loss` function by setting the $y$ targets equal to:

$$
\begin{equation}
    y_j =
    \begin{cases}
      R_j & \text{if episode terminates at step  } j+1\\
      R_j + \gamma \max_{a'}\hat{Q}(s_{j+1},a') & \text{otherwise}\\
    \end{cases}
\end{equation}
$$

Here are a couple of things to note:

* The `compute_loss` function takes in a mini-batch of experience tuples. This mini-batch of experience tuples is unpacked to extract the `states`, `actions`, `rewards`, `next_states`, and `done_vals`. You should keep in mind that these variables are *TensorFlow Tensors* whose size will depend on the mini-batch size. For example, if the mini-batch size is `64` then both `rewards` and `done_vals` will be TensorFlow Tensors with `64` elements.


* Using `if/else` statements to set the $y$ targets will not work when the variables are tensors with many elements. However, notice that you can use the `done_vals` to implement the above in a single line of code. To do this, recall that the `done` variable is a Boolean variable that takes the value `True` when an episode terminates at step $j+1$ and it is `False` otherwise. Taking into account that a Boolean value of `True` has the numerical value of `1` and a Boolean value of `False` has the numerical value of `0`, you can use the factor `(1 - done_vals)` to implement the above in a single line of code. Here's a hint: notice that `(1 - done_vals)` has a value of `0` when `done_vals` is `True` and a value of `1` when `done_vals` is `False`.

Lastly, compute the loss by calculating the Mean-Squared Error (`MSE`) between the `y_targets` and the `q_values`. To calculate the mean-squared error you should use the already imported package `MSE`:
```python
from tensorflow.keras.losses import MSE
```


```python
# UNQ_C2
# GRADED FUNCTION: calculate_loss

def compute_loss(experiences, gamma, q_network, target_q_network):
    """
    Calculates the loss.

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
      q_network: (tf.keras.Sequential) Keras model for predicting the q_values
      target_q_network: (tf.keras.Sequential) Karas model for predicting the targets

    Returns:
      loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
            the y targets and the Q(s,a) values.
    """

    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences

    # Compute max Q^(s,a)
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)

    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    ### START CODE HERE ###
    y_targets = rewards + (gamma * max_qsa * (1 - done_vals))
    ### END CODE HERE ###

    # Get the q_values
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))

    # Compute the loss
    ### START CODE HERE ###
    loss = MSE(y_targets, q_values)
    ### END CODE HERE ###

    return loss
```


```python
# UNIT TEST
test_compute_loss(compute_loss)
```

    [92mAll tests passed!



  <summary><font size="3" color="darkgreen"><b>Click for hints</b></font></summary>

```python
def compute_loss(experiences, gamma, q_network, target_q_network):
    """
    Calculates the loss.

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
      q_network: (tf.keras.Sequential) Keras model for predicting the q_values
      target_q_network: (tf.keras.Sequential) Karas model for predicting the targets

    Returns:
      loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
            the y targets and the Q(s,a) values.
    """


    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences

    # Compute max Q^(s,a)
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)

    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    y_targets = rewards + (gamma * max_qsa * (1 - done_vals))

    # Get the q_values
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))

    # Calculate the loss
    loss = MSE(y_targets, q_values)

    return loss

```


<a name="8"></a>
## 8 - Update the Network Weights

We will use the `agent_learn` function below to implement lines ***12 -14*** of the algorithm outlined in [Fig 3](#7). The `agent_learn` function will update the weights of the $Q$ and target $\hat Q$ networks using a custom training loop. Because we are using a custom training loop we need to retrieve the gradients via a `tf.GradientTape` instance, and then call `optimizer.apply_gradients()` to update the weights of our $Q$-Network. Note that we are also using the `@tf.function` decorator to increase performance. Without this decorator our training will take twice as long. If you would like to know more about how to increase performance with `@tf.function` take a look at the [TensorFlow documentation](https://tensorflow.org/guide/function).

The last line of this function updates the weights of the target $\hat Q$-Network using a [soft update](#6.1). If you want to know how this is implemented in code we encourage you to take a look at the `utils.update_target_network` function in the `utils` module.


```python
@tf.function
def agent_learn(experiences, gamma):
    """
    Updates the weights of the Q networks.

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.

    """

    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)

    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, q_network.trainable_variables)

    # Update the weights of the q_network.
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    # update the weights of target q_network
    utils.update_target_network(q_network, target_q_network)
```

<a name="9"></a>
## 9 - Train the Agent

We are now ready to train our agent to solve the Lunar Lander environment. In the cell below we will implement the algorithm in [Fig 3](#7) line by line (please note that we have included the same algorithm below for easy reference. This will prevent you from scrolling up and down the notebook):

* **Line 1**: We initialize the `memory_buffer` with a capacity of $N =$ `MEMORY_SIZE`. Notice that we are using a `deque` as the data structure for our `memory_buffer`.


* **Line 2**: We skip this line since we already initialized the `q_network` in [Exercise 1](#ex01).


* **Line 3**: We initialize the `target_q_network` by setting its weights to be equal to those of the `q_network`.


* **Line 4**: We start the outer loop. Notice that we have set $M =$ `num_episodes = 2000`. This number is reasonable because the agent should be able to solve the Lunar Lander environment in less than `2000` episodes using this notebook's default parameters.


* **Line 5**: We use the `.reset()` method to reset the environment to the initial state and get the initial state.


* **Line 6**: We start the inner loop. Notice that we have set $T =$ `max_num_timesteps = 1000`. This means that the episode will automatically terminate if the episode hasn't terminated after `1000` time steps.


* **Line 7**: The agent observes the current `state` and chooses an `action` using an $\epsilon$-greedy policy. Our agent starts out using a value of $\epsilon =$ `epsilon = 1` which yields an $\epsilon$-greedy policy that is equivalent to the equiprobable random policy. This means that at the beginning of our training, the agent is just going to take random actions regardless of the observed `state`. As training progresses we will decrease the value of $\epsilon$ slowly towards a minimum value using a given $\epsilon$-decay rate. We want this minimum value to be close to zero because a value of $\epsilon = 0$ will yield an $\epsilon$-greedy policy that is equivalent to the greedy policy. This means that towards the end of training, the agent will lean towards selecting the `action` that it believes (based on its past experiences) will maximize $Q(s,a)$. We will set the minimum $\epsilon$ value to be `0.01` and not exactly 0 because we always want to keep a little bit of exploration during training. If you want to know how this is implemented in code we encourage you to take a look at the `utils.get_action` function in the `utils` module.


* **Line 8**: We use the `.step()` method to take the given `action` in the environment and get the `reward` and the `next_state`.


* **Line 9**: We store the `experience(state, action, reward, next_state, done)` tuple in our `memory_buffer`. Notice that we also store the `done` variable so that we can keep track of when an episode terminates. This allowed us to set the $y$ targets in [Exercise 2](#ex02).


* **Line 10**: We check if the conditions are met to perform a learning update. We do this by using our custom `utils.check_update_conditions` function. This function checks if $C =$ `NUM_STEPS_FOR_UPDATE = 4` time steps have occured and if our `memory_buffer` has enough experience tuples to fill a mini-batch. For example, if the mini-batch size is `64`, then our `memory_buffer` should have at least `64` experience tuples in order to pass the latter condition. If the conditions are met, then the `utils.check_update_conditions` function will return a value of `True`, otherwise it will return a value of `False`.


* **Lines 11 - 14**: If the `update` variable is `True` then we perform a learning update. The learning update consists of sampling a random mini-batch of experience tuples from our `memory_buffer`, setting the $y$ targets, performing gradient descent, and updating the weights of the networks. We will use the `agent_learn` function we defined in [Section 8](#8) to perform the latter 3.


* **Line 15**: At the end of each iteration of the inner loop we set `next_state` as our new `state` so that the loop can start again from this new state. In addition, we check if the episode has reached a terminal state (i.e we check if `done = True`). If a terminal state has been reached, then we break out of the inner loop.


* **Line 16**: At the end of each iteration of the outer loop we update the value of $\epsilon$, and check if the environment has been solved. We consider that the environment has been solved if the agent receives an average of `200` points in the last `100` episodes. If the environment has not been solved we continue the outer loop and start a new episode.

Finally, we wanted to note that we have included some extra variables to keep track of the total number of points the agent received in each episode. This will help us determine if the agent has solved the environment and it will also allow us to see how our agent performed during training. We also use the `time` module to measure how long the training takes.

<br>
<br>
<figure>
  <img src = "images/deep_q_algorithm.png" width = 90% style = "border: thin silver solid; padding: 0px">
      <figcaption style = "text-align: center; font-style: italic">Fig 4. Deep Q-Learning with Experience Replay.</figcaption>
</figure>
<br>

**Note:** With this notebook's default parameters, the following cell takes between 10 to 15 minutes to run.


```python
start = time.time()

num_episodes = 2000
max_num_timesteps = 1000

total_point_history = []

num_p_av = 100    # number of total points to use for averaging
epsilon = 1.0     # initial ε value for ε-greedy policy

# Create a memory buffer D with capacity N
memory_buffer = deque(maxlen=MEMORY_SIZE)

# Set the target network weights equal to the Q-Network weights
target_q_network.set_weights(q_network.get_weights())

for i in range(num_episodes):

    # Reset the environment to the initial state and get the initial state
    state = env.reset()
    total_points = 0

    for t in range(max_num_timesteps):

        # From the current state S choose an action A using an ε-greedy policy
        state_qn = np.expand_dims(state, axis=0)  # state needs to be the right shape for the q_network
        q_values = q_network(state_qn)
        action = utils.get_action(q_values, epsilon)

        # Take action A and receive reward R and the next state S'
        next_state, reward, done, _ = env.step(action)

        # Store experience tuple (S,A,R,S') in the memory buffer.
        # We store the done variable as well for convenience.
        memory_buffer.append(experience(state, action, reward, next_state, done))

        # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
        update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)

        if update:
            # Sample random mini-batch of experience tuples (S,A,R,S') from D
            experiences = utils.get_experiences(memory_buffer)

            # Set the y targets, perform a gradient descent step,
            # and update the network weights.
            agent_learn(experiences, GAMMA)

        state = next_state.copy()
        total_points += reward

        if done:
            break

    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-num_p_av:])

    # Update the ε value
    epsilon = utils.get_new_eps(epsilon)

    print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

    if (i+1) % num_p_av == 0:
        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

    # We will consider that the environment is solved if we get an
    # average of 200 points in the last 100 episodes.
    if av_latest_points >= 200.0:
        print(f"\n\nEnvironment solved in {i+1} episodes!")
        q_network.save('lunar_lander_model.h5')
        break

tot_time = time.time() - start

print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")
```

    Episode 100 | Total point average of the last 100 episodes: -150.85
    Episode 200 | Total point average of the last 100 episodes: -106.11
    Episode 300 | Total point average of the last 100 episodes: -77.256
    Episode 400 | Total point average of the last 100 episodes: -25.01
    Episode 500 | Total point average of the last 100 episodes: 159.91
    Episode 534 | Total point average of the last 100 episodes: 201.37

    Environment solved in 534 episodes!

    Total Runtime: 715.21 s (11.92 min)


We can plot the point history to see how our agent improved during training.


```python
# Plot the point history
utils.plot_history(total_point_history)
```


![png](output_37_0.png)


<a name="10"></a>
## 10 - See the Trained Agent In Action

Now that we have trained our agent, we can see it in action. We will use the `utils.create_video` function to create a video of our agent interacting with the environment using the trained $Q$-Network. The `utils.create_video` function uses the `imageio` library to create the video. This library produces some warnings that can be distracting, so, to suppress these warnings we run the code below.


```python
# Suppress warnings from imageio
import logging
logging.getLogger().setLevel(logging.ERROR)
```

In the cell below we create a video of our agent interacting with the Lunar Lander environment using the trained `q_network`. The video is saved to the `videos` folder with the given `filename`. We use the `utils.embed_mp4` function to embed the video in the Jupyter Notebook so that we can see it here directly without having to download it.

We should note that since the lunar lander starts with a random initial force applied to its center of mass, every time you run the cell below you will see a different video. If the agent was trained properly, it should be able to land the lunar lander in the landing pad every time, regardless of the initial force applied to its center of mass.


```python
filename = "./videos/lunar_lander.mp4"

utils.create_video(filename, env, q_network)
utils.embed_mp4(filename)
```





<video width="840" height="480" controls>
<source src="data:video/mp4;base64,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AAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACAAAAAACAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAgAAAAAAgAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAABgAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAgAABAAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEA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" type="video/mp4">
Your browser does not support the video tag.
</video>



<a name="11"></a>
## 11 - Congratulations!

You have successfully used Deep Q-Learning with Experience Replay to train an agent to land a lunar lander safely on a landing pad on the surface of the moon. Congratulations!

<a name="12"></a>
## 12 - References

If you would like to learn more about Deep Q-Learning, we recommend you check out the following papers.


* [Human-level Control Through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)


* [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf)


* [Playing Atari with Deep Reinforcement Learning](https://cs.toronto.edu/~vmnih/docs/dqn.pdf)


```python

```
