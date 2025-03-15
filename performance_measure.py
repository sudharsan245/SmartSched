import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from hpc_env import HPCSchedulingEnv

# Load Models
dqn_model = DQN.load("dqn_hpc_scheduler")
ppo_model = PPO.load("ppo_hpc_scheduler")

env = HPCSchedulingEnv()

metrics = {"DQN": [], "PPO": []}

# Run tests for both models
for model_name, model in [("DQN", dqn_model), ("PPO", ppo_model)]:
    env.reset()
    waiting_times, turnaround_times = [], []

    for _ in range(100):  # Run for 100 scheduling decisions
        action, _states = model.predict(env.reset(), deterministic=True)
        obs, reward, done, info = env.step(action)

        waiting_times.append(info["waiting_time"])
        turnaround_times.append(info["turnaround_time"])

    metrics[model_name] = {"Waiting Time": np.mean(waiting_times), "Turnaround Time": np.mean(turnaround_times)}

# Plot Results
x_labels = ["Waiting Time", "Turnaround Time"]
dqn_values = [metrics["DQN"]["Waiting Time"], metrics["DQN"]["Turnaround Time"]]
ppo_values = [metrics["PPO"]["Waiting Time"], metrics["PPO"]["Turnaround Time"]]

x = np.arange(len(x_labels))
width = 0.3

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, dqn_values, width, label="DQN")
rects2 = ax.bar(x + width/2, ppo_values, width, label="PPO")

ax.set_ylabel("Time (Seconds)")
ax.set_title("Comparison of Waiting Time & Turnaround Time")
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.legend()

plt.show()
