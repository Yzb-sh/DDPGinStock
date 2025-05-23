import random
import numpy as np
import os
import data
import torch
import DDPG
import stockEnv
import pandas as pd
import time

INITIAL_ACCOUNT_BALANCE = 50e4
NUM_EPISODES = 399
BATCH_SIZE = 128

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建必要的文件夹
def create_directories():
    directories = [
        "./results", 
        "./models", 
        "./models/reward-1", 
        "./models/reward-2", 
        "./models/reward-3", 
        "./models/reward-4", 
        "./models/reward-5", 
        "./tmp/ddpg", 
        "./data"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def train(code, reward_type, seed, model_type='ddpg', seq_len=10):
    """
    训练DDPG模型
    
    参数:
        code (str): 股票代码
        reward_type (str): 奖励函数类型 (1-5)
        seed (int): 随机种子
        model_type (str): 模型类型 ('ddpg' 或 'ddpg_lstm')
        seq_len (int): 使用LSTM时的序列长度
    """
    # 创建保存模型的文件夹
    model_dir = f'./models/reward-{reward_type}/{code}'
    os.makedirs(model_dir, exist_ok=True)
    
    # 加载数据
    path = data.getData(code=code)
    datas = pd.read_csv(path)
    datas = data.cleanData(datas)
    datas = data.computeData(datas)
    division = len(datas) - 750
    train_datas = datas[30:division]
    train_datas = train_datas.reset_index(drop=True)
    print(f"训练数据集大小: {len(train_datas)}")

    # 设置环境和DDPG算法
    env = stockEnv.StockTradingEnv(datas, train_datas)
    env.seed(seed)
    env.set_lstm_mode(model_type == 'ddpg_lstm')  # 设置环境的LSTM模式
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.train_set()
    state_dim = env.observation_space.shape[0]
    action_dim = 2
    max_action = 1.0
    ddpg = DDPG.DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        use_lstm=(model_type == 'ddpg_lstm'),
        seq_len=seq_len
        )

    max_reward = -1e9
    max_model_profit = -float('inf')
    best_model_state = None
    
    # 增加探索机制和学习率调整
    exploration_noise = 0.3  # 初始探索噪声
    exploration_decay = 0.99  # 探索噪声衰减率
    min_exploration = 0.05   # 最小探索噪声

    # 训练
    print(f"开始训练, 股票代码: {code}, 奖励函数: {reward_type}, 种子: {seed}")
    for episode in range(NUM_EPISODES):
        state, done = env.reset(), False
        total_reward = 0.0
        max_money = INITIAL_ACCOUNT_BALANCE
        min_money = INITIAL_ACCOUNT_BALANCE
        money = INITIAL_ACCOUNT_BALANCE
        steps = 0

        # 动态调整探索噪声
        current_noise = max(min_exploration, exploration_noise * (exploration_decay ** episode))

        while not done:
            # 基础动作加上噪声以增加探索
            base_action = ddpg.select_action(state)
            noisy_action = base_action + np.random.normal(0, current_noise, size=base_action.shape)
            noisy_action = np.clip(noisy_action, -1, 1)  # 裁剪到合法范围
            
            # 选择奖励函数
            if reward_type == '3':
                next_state, reward, done, money = env.step3(noisy_action)
            elif reward_type == '2':
                next_state, reward, done, money = env.step2(noisy_action)
            elif reward_type == '4':
                next_state, reward, done, money = env.step4(noisy_action)
            elif reward_type == '5':
                next_state, reward, done, money = env.step5(noisy_action)
            else:
                next_state, reward, done, money = env.step1(noisy_action)
                
            ddpg.replay_buffer.add(state, noisy_action, reward, next_state, done)

            state = next_state
            total_reward += reward
            max_money = max(money, max_money)
            min_money = min(money, min_money)
            steps += 1

            # 每步都进行多次训练，增强学习效果
            if ddpg.replay_buffer.size > BATCH_SIZE:
                for _ in range(2):  # 每步进行多次训练
                    ddpg.train(BATCH_SIZE)

        # 计算本次episode的盈利情况
        profit_percentage = (money - INITIAL_ACCOUNT_BALANCE) / INITIAL_ACCOUNT_BALANCE * 100
        
        # 根据总奖励保存最佳模型
        if total_reward > max_reward:
            torch.save(ddpg, f'{model_dir}/best_reward.pt')
            max_reward = total_reward
        
        # 根据利润百分比保存最佳模型
        if profit_percentage > max_model_profit:
            torch.save(ddpg, f'{model_dir}/best.pt')
            max_model_profit = profit_percentage
            
        # 打印本次episode的训练结果
        print(f"Episode: {episode + 1}/{NUM_EPISODES}, Steps: {steps}, Noise: {current_noise:.4f}, Total Reward: {total_reward:.2f}, Profit: {profit_percentage:.2f}%")

    # 保存最终模型
    torch.save(ddpg, f'{model_dir}/{seed}.pt')
    print(f"训练完成，模型已保存到 {model_dir}/{seed}.pt")
    print(f"最大奖励: {max_reward:.2f}, 最大利润率: {max_model_profit:.2f}%")

if __name__ == "__main__":
    create_directories()
    # 示例用法
    # train('600016', '3', 1234)
