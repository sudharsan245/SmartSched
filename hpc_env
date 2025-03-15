import gym
from gym import spaces
import numpy as np

class HPCSchedulingEnv(gym.Env):
    def __init__(self, queue_size=5, max_cpus=16):
        super(HPCSchedulingEnv, self).__init__()

        self.queue_size = queue_size
        self.max_cpus = max_cpus
        
        # Observation: [CPU Load, Avg Job Demand, Avg Job Priority, Avg Job Time]
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        # Action: Select a job (DQN) or allocate CPU (PPO, MARL)
        self.action_space = spaces.Discrete(self.queue_size)

        self.reset()

    def reset(self):
        self.cpu_load = np.random.uniform(0.2, 0.8)
        self.job_queue = np.random.uniform(0.1, 1.0, size=(self.queue_size,))
        self.job_priority = np.random.choice([0, 1], size=(self.queue_size,))
        self.job_time = np.random.uniform(0.5, 5.0, size=(self.queue_size,))

        # Track job arrival and execution times
        self.job_arrival_time = np.zeros(self.queue_size)
        self.execution_start_time = np.full(self.queue_size, -1.0)
        self.job_completion_time = np.full(self.queue_size, -1.0)
        self.current_time = 0

        return np.array([self.cpu_load, np.mean(self.job_queue), np.mean(self.job_priority), np.mean(self.job_time)])

    def step(self, action):
        selected_job = action
        allocated_cpus = self.job_queue[selected_job] * self.max_cpus
        execution_time = self.job_time[selected_job] / (allocated_cpus / 4.0)

        # If job starts execution, record start time
        if self.execution_start_time[selected_job] == -1:
            self.execution_start_time[selected_job] = self.current_time

        # Update CPU Load
        self.cpu_load = min(1.0, self.cpu_load + (allocated_cpus / self.max_cpus))

        # Track completion time
        self.current_time += execution_time
        self.job_completion_time[selected_job] = self.current_time

        # Compute waiting time & turnaround time
        waiting_time = self.execution_start_time[selected_job] - self.job_arrival_time[selected_job]
        turnaround_time = self.job_completion_time[selected_job] - self.job_arrival_time[selected_job]

        # Reward: Minimize execution time & avoid CPU underutilization
        reward = -execution_time if self.cpu_load < 0.9 else -execution_time / 2

        # Generate new jobs
        self.job_queue = np.random.uniform(0.1, 1.0, size=(self.queue_size,))
        self.job_priority = np.random.choice([0, 1], size=(self.queue_size,))
        self.job_time = np.random.uniform(0.5, 5.0, size=(self.queue_size,))

        return np.array([self.cpu_load, np.mean(self.job_queue), np.mean(self.job_priority), np.mean(self.job_time)]), reward, False, {"waiting_time": waiting_time, "turnaround_time": turnaround_time}
