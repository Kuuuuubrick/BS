import agents, MLP, replay_buffers, explorers
import torch
import MLP
import env_manager
import numpy as np
from Auto_Encoder import Auto_Encoder
from draw import draw_mean_reward_variation
from draw import draw_reward_variation
import csv


class TrainManager():

    def __init__(self,
                 env,  # 环境
                 episodes=500,  # 轮次数量
                 batch_size=8,  # 每一批次的数量
                 num_steps=4,  # 进行学习的频次
                 memory_size=2000,  # 经验回放池的容量
                 replay_start_size=16,  # 开始回放的次数
                 update_target_steps=20,  # 同步参数的次数
                 lr=0.1,  # 学习率
                 gamma=0.9,  # 收益衰减率
                 e_greed=0.2,  # 探索与利用中的探索概率
                 e_gredd_decay=1e-4,  # 探索与利用中探索概率的衰减步长
                 use_gpu=False,
                 ):
        self.use_gpu = use_gpu
        n_act = env.num_nodes
        n_obs = env.obs.shape[1]

        self.env = env
        self.episodes = episodes

        explorer = explorers.EpsilonGreedy(n_act, e_greed, e_gredd_decay)
        if use_gpu:
            # q_func = Auto_Encoder(obs_size=n_obs).cuda()
            q_func = MLP.MLP(n_obs, n_act).cuda()
        else:
            q_func = MLP.MLP(n_obs, n_act)
            # q_func = Auto_Encoder(obs_size=n_obs)
        optimizer = torch.optim.AdamW(q_func.parameters(), lr=lr)
        rb = replay_buffers.ReplayBuffer(memory_size, num_steps)

        self.agent = agents.DQNAgent(
            q_func=q_func,
            optimizer=optimizer,
            explorer=explorer,
            replay_buffer=rb,
            batch_size=batch_size,
            replay_start_size=replay_start_size,
            update_target_steps=update_target_steps,
            n_act=n_act,
            gamma=gamma,
            use_gpu=use_gpu)

    def train_episode(self, index_episode):
        total_reward = 0
        obs = self.env.reset()
        step = 0
        while True:
            action = self.agent.act(obs)
            self.agent.list_pre_actions.append(action)
            next_obs, reward, done = self.env.step(action)
            total_reward += reward
            self.agent.learn(obs, action, reward, next_obs, done)
            obs = next_obs
            if index_episode % 2 == 0:
                with open("result.csv", "a", encoding='utf8', newline='') as f:
                    writer = csv.writer(f)
                    row_result = [index_episode, action, step, reward, total_reward]
                    writer.writerow(row_result)
                    f.close()
            step += 1
            if done:
                self.agent.list_pre_actions = []
                break
        return total_reward

    def test_episode(self):
        total_reward = 0
        obs = self.env.reset()
        while True:
            action = self.agent.predict(obs)
            next_obs, reward, done = self.env.step(action)
            total_reward += reward
            obs = next_obs
            if done: break
        return total_reward

    def train(self):
        fileHeader = ["Episode", "index_removal", "step", 'reward', 'total_reward']
        with open("result.csv", "w", encoding='utf8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fileHeader)
            f.close()
        list_episodes = []
        list_test_episodes = []
        list_rewards = []
        list_test_rewards = []
        for e in range(self.episodes):
            ep_reward = self.train_episode(e)
            list_episodes.append(e + 1)
            list_rewards.append(ep_reward)
            if (e + 1) % 10 == 0:
                sub_list = list_rewards[e - 9:]
                temp = np.mean(sub_list)
                list_test_episodes.append(e)
                list_test_rewards.append(temp)
        draw_reward_variation(list_episodes, list_rewards)
        draw_mean_reward_variation(list_test_episodes, list_test_rewards)


if __name__ == '__main__':
    list_file_graph = ['g_27']
    use_gpu = torch.cuda.is_available()
    em = env_manager.EnvManager(list_file_graph=list_file_graph, use_gpu=use_gpu)
    env = em.list_env[0]
    tm = TrainManager(env, use_gpu=use_gpu)
    tm.train()
