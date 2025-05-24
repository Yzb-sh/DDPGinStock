import random
import numpy as np
import os
import data
import torch
import DDPG
import stockEnv
import pandas as pd
import time
import matplotlib.pyplot as plt

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
        # 为每种奖励函数类型创建ddpg和ddpg_lstm文件夹
        "./models/reward-1/ddpg",
        "./models/reward-1/ddpg_lstm",
        "./models/reward-2/ddpg",
        "./models/reward-2/ddpg_lstm", 
        "./models/reward-3/ddpg",
        "./models/reward-3/ddpg_lstm",
        "./models/reward-4/ddpg",
        "./models/reward-4/ddpg_lstm",
        "./models/reward-5/ddpg",
        "./models/reward-5/ddpg_lstm",
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
    # 创建保存模型的文件夹，按模型类型分开保存
    model_dir = f'./models/reward-{reward_type}/{model_type}/{code}'
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"模型保存路径: {model_dir}")
    print(f"模型类型: {model_type}")
    
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
    
    # 记录训练过程数据用于绘图
    episode_rewards = []
    episode_profits = []
    episode_numbers = []
    
    # 增加探索机制和学习率调整
    exploration_noise = 0.3  # 初始探索噪声
    exploration_decay = 0.99  # 探索噪声衰减率
    min_exploration = 0.05   # 最小探索噪声

    # 训练
    print(f"开始训练, 股票代码: {code}, 奖励函数: {reward_type}, 种子: {seed}, 模型类型: {model_type}")
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
        
        # 记录训练数据
        episode_numbers.append(episode + 1)
        episode_rewards.append(total_reward)
        episode_profits.append(profit_percentage)
        
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
    
    # 绘制并保存训练曲线
    plot_training_curves(episode_numbers, episode_rewards, episode_profits, 
                         code, model_type, reward_type, seed, model_dir)

def plot_training_curves(episodes, rewards, profits, code, model_type, reward_type, seed, save_dir):
    """
    绘制并保存训练过程中的reward和profit曲线
    
    参数:
        episodes (list): episode编号列表
        rewards (list): 每个episode的总奖励列表
        profits (list): 每个episode的利润百分比列表
        code (str): 股票代码
        model_type (str): 模型类型
        reward_type (str): 奖励函数类型
        seed (int): 随机种子
        save_dir (str): 保存目录
    """
    # 设置图表参数 - 使用默认字体避免字体找不到的问题
    try:
        # 尝试设置中文字体，如果失败就使用默认字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans', 'Liberation Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    except:
        # 如果字体设置失败，使用默认设置
        plt.rcParams.update(plt.rcParamsDefault)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制奖励曲线
    ax1.plot(episodes, rewards, 'b-', linewidth=1.5, alpha=0.7)
    ax1.set_title(f'Training Reward Curve - {code} ({model_type})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 添加趋势线（移动平均）
    if len(rewards) > 10:
        window = min(20, len(rewards)//5)  # 窗口大小
        moving_avg_reward = []
        for i in range(len(rewards)):
            start = max(0, i-window+1)
            moving_avg_reward.append(np.mean(rewards[start:i+1]))
        ax1.plot(episodes, moving_avg_reward, 'r-', linewidth=2, label=f'Moving Average (window={window})')
        ax1.legend()
    
    # 绘制利润曲线
    ax2.plot(episodes, profits, 'g-', linewidth=1.5, alpha=0.7)
    ax2.set_title(f'Training Profit Curve - {code} ({model_type})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Profit (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)  # 零线
    
    # 添加趋势线（移动平均）
    if len(profits) > 10:
        window = min(20, len(profits)//5)  # 窗口大小
        moving_avg_profit = []
        for i in range(len(profits)):
            start = max(0, i-window+1)
            moving_avg_profit.append(np.mean(profits[start:i+1]))
        ax2.plot(episodes, moving_avg_profit, 'orange', linewidth=2, label=f'Moving Average (window={window})')
        ax2.legend()
    
    # 添加统计信息
    final_reward = rewards[-1]
    final_profit = profits[-1]
    max_reward = max(rewards)
    max_profit = max(profits)
    min_profit = min(profits)
    
    # 在图表上添加文本信息
    info_text = (
        f"Final: Reward={final_reward:.2f}, Profit={final_profit:.2f}%\n"
        f"Max: Reward={max_reward:.2f}, Profit={max_profit:.2f}%\n"
        f"Min Profit: {min_profit:.2f}%\n"
        f"Reward Type: {reward_type}, Seed: {seed}"
    )
    
    fig.text(0.02, 0.02, info_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # 为文本信息留出空间
    
    # 保存图表
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{code}_{model_type}_reward{reward_type}_training_curves_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)
    
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    except Exception as e:
        print(f"保存图表时出错: {e}")
        # 尝试使用更简单的保存方式
        plt.savefig(save_path, dpi=150)
        print(f"训练曲线已保存到: {save_path} (简化版)")
    
    # 可选：显示图表（注释掉避免阻塞）
    # plt.show()
    
    # 关闭图表释放内存
    plt.close()
    
    # 保存训练数据为CSV文件
    training_data = pd.DataFrame({
        'Episode': episodes,
        'Total_Reward': rewards,
        'Profit_Percentage': profits
    })
    
    csv_filename = f"{code}_{model_type}_reward{reward_type}_training_data_{timestamp}.csv"
    csv_path = os.path.join(save_dir, csv_filename)
    training_data.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"训练数据已保存到: {csv_path}")

if __name__ == "__main__":
    create_directories()
    # 示例用法
    # train('600016', '3', 1234)
