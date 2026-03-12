


import time
import gymnasium as gym
from traffic_environment import TrafficEnv
import rl_planners
import numpy as np

# define rewards function
rewards = {"state": 0}

# initialize the environment
env = TrafficEnv(rewards=rewards, max_steps=100)

# choose RL algorithm
rl_algo = "Value Iteration"

# initialize the agent
if rl_algo == "Value Iteration":
    agent = rl_planners.ValueIterationPlanner(env)
elif rl_algo == "Policy Iteration":
    agent = rl_planners.PolicyIterationPlanner(env)

# reset environment
observation, info = env.reset(seed=42), {}
np.random.seed(42)
env.action_space.seed(42)

# initialize counters
step_count = 0
total_reward = 0

# performance metrics
total_ns_cars = 0
total_ew_cars = 0

max_ns = 0
max_ew = 0

threshold = 10
congestion_events = 0

light_switches = 0

# light states
RED, GREEN = 0, 1

# initial light state
ns, ew, prev_light = tuple(observation)

terminated = False
truncated = False

while not terminated and not truncated:

    step_count += 1

    # choose action
    action = agent.choose_action(observation)

    # environment step
    observation, reward, terminated, truncated, info = env.step(action)

    # accumulate reward
    total_reward += reward

    # unpack state
    ns, ew, light = tuple(observation)
    light_color = "GREEN" if light == GREEN else "RED"

    # track average waiting cars
    total_ns_cars += ns
    total_ew_cars += ew

    # track maximum queue
    max_ns = max(max_ns, ns)
    max_ew = max(max_ew, ew)

    # congestion events
    if ns > threshold or ew > threshold:
        congestion_events += 1

    # traffic light switches
    if light != prev_light:
        light_switches += 1
    prev_light = light

    # print step information
    print(f"Step: {step_count}, NS Cars: {ns}, EW Cars: {ew}, Light NS: {light_color}, Reward: {reward}")

# close environment
env.render(close=True)

# compute averages
avg_ns = total_ns_cars / step_count
avg_ew = total_ew_cars / step_count

# performance evaluation
print("\n=== PERFORMANCE EVALUATION ===")
print(f"Total Steps: {step_count}")
print(f"Total Reward: {total_reward}")
print(f"Average NS Cars Waiting: {avg_ns:.2f}")
print(f"Average EW Cars Waiting: {avg_ew:.2f}")
print(f"Max NS Queue Length: {max_ns}")
print(f"Max EW Queue Length: {max_ew}")
print(f"Congestion Events (> {threshold} cars): {congestion_events}")
print(f"Traffic Light Switches: {light_switches}")