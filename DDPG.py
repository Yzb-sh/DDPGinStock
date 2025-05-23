import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorLSTM(nn.Module):
    """带LSTM的Actor网络"""
    def __init__(self, state_dim, action_dim, max_action, lstm_hidden=128):
        super(ActorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=lstm_hidden, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action
        
        # 正交初始化LSTM参数
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, state_seq):
        # state_seq形状: (batch_size, seq_len, state_dim)
        lstm_out, _ = self.lstm(state_seq)
        last_hidden = lstm_out[:, -1, :]  # 取最后一个时间步
        a = F.relu(self.fc1(last_hidden))
        a = F.relu(self.fc2(a))
        return self.max_action * torch.tanh(self.fc3(a))

class Actor(nn.Module):
    """原始全连接Actor网络"""
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        return self.max_action * torch.tanh(self.fc3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = F.relu(self.fc1(sa))
        q = F.relu(self.fc2(q))
        return self.fc3(q)

class SequenceReplayBuffer:
    """支持序列采样的经验回放缓冲区"""
    def __init__(self, state_dim, action_dim, seq_len=10, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.seq_len = seq_len
        
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, reward, next_state, done):
        # 如果state是序列，取最后一个状态
        if isinstance(state, np.ndarray) and len(state.shape) > 1 and state.shape[0] == self.seq_len:
            state = state[-1]  # 取序列的最后一个状态
        
        # 如果next_state是序列，取最后一个状态
        if isinstance(next_state, np.ndarray) and len(next_state.shape) > 1 and next_state.shape[0] == self.seq_len:
            next_state = next_state[-1]  # 取序列的最后一个状态
            
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(self.seq_len, self.size, batch_size)
        
        state_seq = np.zeros((batch_size, self.seq_len, self.state.shape[1]))
        next_state_seq = np.zeros_like(state_seq)
        
        for i, j in enumerate(idx):
            start_idx = j - self.seq_len
            state_seq[i] = self.state[start_idx:j]
            next_state_seq[i] = self.next_state[start_idx:j]

        return (
            torch.FloatTensor(state_seq),
            torch.FloatTensor(self.action[idx]),
            torch.FloatTensor(self.reward[idx]),
            torch.FloatTensor(next_state_seq),
            torch.FloatTensor(self.done[idx])
        )

class DDPG:
    def __init__(self, state_dim, action_dim, max_action, use_lstm=False, seq_len=10):
        self.use_lstm = use_lstm
        
        # 初始化Actor
        if use_lstm:
            self.actor = ActorLSTM(state_dim, action_dim, max_action).to(device)
        else:
            self.actor = Actor(state_dim, action_dim, max_action).to(device)
            
        self.actor_target = type(self.actor)(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        
        # Critic保持不变
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # 经验回放缓冲区
        self.replay_buffer = SequenceReplayBuffer(state_dim, action_dim, seq_len) if use_lstm else ReplayBuffer(state_dim, action_dim)
        self.max_action = max_action
        self.train_steps = 0

    def select_action(self, state):
        if self.use_lstm:
            # 如果是LSTM模型，我们需要确保输入是(1, seq_len, state_dim)的形状
            # 但在训练初始阶段，我们可能只有一个状态向量
            state_np = np.array(state)
            if len(state_np.shape) == 1:
                # 如果只是一个状态向量，创建一个假的序列（全部重复该状态）
                seq = np.tile(state_np, (self.replay_buffer.seq_len, 1))
                state = torch.FloatTensor(seq).unsqueeze(0).to(device)  # 变成(1, seq_len, state_dim)
            elif len(state_np.shape) == 2:
                # 已经是序列形式，但可能需要添加batch维度
                state = torch.FloatTensor(state_np).unsqueeze(0).to(device)
            else:
                # 已经是完整形状
                state = torch.FloatTensor(state_np).to(device)
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch_size=128, discount=0.99, tau=0.005):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        next_state = next_state.to(device)
        done = done.to(device)

        # 处理Critic输入
        if self.use_lstm:
            next_state_flat = next_state[:, -1, :]  # 取序列最后一个状态
            state_flat = state[:, -1, :]
        else:
            next_state_flat = next_state
            state_flat = state

        # 计算target Q
        with torch.no_grad():
            # 确保传递给actor_target的参数形状正确
            target_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state_flat, target_action)
            target_Q = reward + (1 - done) * discount * target_Q

        # 更新Critic
        current_Q = self.critic(state_flat, action)
        critic_loss = F.mse_loss(current_Q, target_Q)
        
        # 使用梯度裁剪和更稳定的优化
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # 更新Actor (延迟更新，每2次更新critic更新一次actor)
        if self.train_steps % 2 == 0:
            # 计算actor损失
            actor_loss = -self.critic(state_flat, self.actor(state)).mean()
            
            # 应用梯度裁剪和更稳定的优化
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # 软更新target网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
        self.train_steps += 1

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.state[idx]),
            torch.FloatTensor(self.action[idx]),
            torch.FloatTensor(self.reward[idx]),
            torch.FloatTensor(self.next_state[idx]),
            torch.FloatTensor(self.done[idx])
        )

