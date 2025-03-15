#ppo training
from stable_baselines3 import PPO
env = HPCSchedulingEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo_hpc_scheduler")

#marl training
from stable_baselines3 import PPO
from pettingzoo.utils import parallel_to_aec
from hpc_marl_env import MultiAgentHPCEnv

env = MultiAgentHPCEnv(num_agents=3)
aec_env = parallel_to_aec(env)

models = {agent: PPO("MlpPolicy", aec_env, verbose=1) for agent in env.agents}
for agent in env.agents:
    models[agent].learn(total_timesteps=100000)
    models[agent].save(f"ppo_hpc_{agent}")
