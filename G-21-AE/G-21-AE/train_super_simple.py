import gym
import torch
import pfrl
import numpy
import modules
import env_manager
import numpy as np
from Auto_Encoder import Auto_Encoder


class TrainManager():

    def __init__(self,
                 env,  # 环境
                 episodes=1000,  # 轮次数量
                 batch_size=32,  # 每一批次的数量
                 num_steps=4,  # 进行学习的频次
                 memory_size=2000,  # 经验回放池的容量
                 replay_start_size=200,  # 开始回放的次数
                 update_target_steps=200,  # 同步参数的次数
                 lr=0.001,  # 学习率
                 gamma=0.9,  # 收益衰减率
                 # epsilon=0.1,  # 探索与利用中的探索概率
                 ):
        n_act = env.num_nodes
        n_obs = env.obs.shape[1]

        self.env = env
        self.episodes = episodes

        explorer = pfrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=0.1, end_epsilon=0.01, decay_steps=15000,
                                                           random_action_func=np.random.choice(n_act))
        q_func = modules.MLP(n_obs, n_act)
        # q_func = Auto_Encoder(n_obs)
        optimizer = torch.optim.AdamW(q_func.parameters(), lr=lr)
        rb = pfrl.replay_buffers.ReplayBuffer(capacity=memory_size, num_steps=num_steps)

        self.agent = pfrl.agents.DoubleDQN(
            q_function=q_func,
            optimizer=optimizer,
            explorer=explorer,
            replay_buffer=rb,
            minibatch_size=batch_size,
            replay_start_size=replay_start_size,
            target_update_interval=update_target_steps,
            gamma=gamma,
            # phi=lambda x: x.astype(numpy.float32, copy=False),

        )

    def train(self):
        pfrl.experiments.train_agent_with_evaluation(
            agent=self.agent,
            env=self.env,
            steps=50000,  # Train the agent for 20000 steps
            eval_n_steps=None,  # We evaluate for episodes, not time
            eval_n_episodes=10,  # 10 episodes are sampled for each evaluation
            train_max_episode_len=500,  # Maximum length of each episode
            eval_interval=1000,  # Evaluate the agent after every 1000 steps
            outdir='result',  # Save everything to 'result' directory
        )


if __name__ == '__main__':
    list_file_graph = ['g_27']
    em = env_manager.EnvManager(list_file_graph=list_file_graph)
    env = em.list_env[0]
    tm = TrainManager(env)
    tm.train()
