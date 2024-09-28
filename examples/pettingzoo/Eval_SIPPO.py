import utils
from meltingpot import substrate
from stable_baselines3.common import vec_env
from Soc_Inf_ppo import Soc_Inf_ppo
from buffers import Eval_Buffer
import numpy as np
import matplotlib.pyplot as plt


import supersuit as ss


def main():
  model_path = "./results/sb3/harvest_open_ppo_paramsharing/SocInfPPO_58/model"

  # Config
  env_name = "boat_race__eight_races"
  env_config = substrate.get_config(env_name)
  env = utils.parallel_env(env_config)
  rollout_len = 1000
  total_timesteps = 10000
  num_agents = env.max_num_agents
  render_mode = "human"
  data_suffix = "_boat_no_inf.png"

  # Training
  num_cpus = 1  # number of cpus
  num_envs = 1  # number of parallel multi-agent environments
  # number of frames to stack together; use >4 to avoid automatic
  # VecTransposeImage
  num_frames = 4

  env = utils.parallel_env(
      max_cycles=rollout_len, env_config=env_config, render_mode=render_mode
  )
  env = ss.observation_lambda_v0(env, lambda x, _: x["RGB"], lambda s: s["RGB"])
  env = ss.frame_stack_v1(
      env, num_frames
  )  # stacks environments, so agents work in multiple env simultaniously
  # actions, observations, rewards and other env-agent-specific vars become
  # vectors
  env = ss.pettingzoo_env_to_vec_env_v1(env)
  env = ss.concat_vec_envs_v1(
      env,
      num_vec_envs=num_envs,
      num_cpus=num_cpus,
      base_class="stable_baselines3",
  )
  env = vec_env.VecMonitor(env)
  env = vec_env.VecTransposeImage(env, True)

  model = Soc_Inf_ppo.load(model_path, env=env)
  num_actions = model.action_space.n
  buffer = Eval_Buffer(num_agents, total_timesteps, num_actions)

  actions, whole_actions, rewards, inf_rews = model.eval(
      buffer, total_timesteps
  )

  x = np.linspace(0, total_timesteps, total_timesteps, endpoint=False)

  for agent in range(num_agents):
    plt.plot(x, rewards[:, agent], label=f"agent_{agent}")
  plt.legend()
  plt.savefig("rewards" + data_suffix)
  plt.show()
  plt.close()

  for agent in range(num_agents):
    plt.plot(x, inf_rews[:, agent], label=f"agent_{agent}")
  plt.legend()
  plt.savefig("inf_rews" + data_suffix)
  plt.show()
  plt.close()

  fig, axs = plt.subplots(num_agents)

  x = np.linspace(0, num_actions, num_actions, endpoint=False)

  for agent in range(num_agents):
    axs[agent].bar(x, actions[agent], label=f"agent_{agent}")
  fig.savefig("actions" + data_suffix)
  plt.close(fig)

  plt.bar(x, whole_actions)
  plt.savefig("summed_acts" + data_suffix)

  # write feature extractor for collab cooking


if __name__ == "__main__":
  main()
