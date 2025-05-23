import numpy as np
import os
import data
import torch
import DDPG
import stockEnv
import pandas as pd
import matplotlib.pyplot as plt
from torch.serialization import add_safe_globals

# 添加所有 DDPG 相关类到安全全局变量列表
add_safe_globals([
    DDPG.DDPG,
    DDPG.Actor,
    DDPG.ActorLSTM,
    DDPG.Critic,
    DDPG.ReplayBuffer,
    DDPG.SequenceReplayBuffer
])

INITIAL_ACCOUNT_BALANCE = 50e4

def get_data(code):
    """
    获取并处理股票数据
    
    参数:
        code (str): 股票代码
    
    返回:
        tuple: (所有数据, 测试数据)
    """
    path = f"./data/{code}.csv"
    if not os.path.exists(path):
        path = data.getData(code=code)
    
    datas = pd.read_csv(path)
    datas = data.cleanData(datas)
    datas = data.computeData(datas)
    division = len(datas) - 750
    test_datas = datas[division:-1]
    test_datas = test_datas.reset_index(drop=True)
    return datas, test_datas

def load_model(model_path):
    """
    加载训练好的模型
    
    参数:
        model_path (str): 模型文件路径
    
    返回:
        DDPG: 加载的DDPG模型
    """
    try:
        # 首先尝试默认加载方式
        return torch.load(model_path)
    except Exception as e:
        # 如果出现weights_only相关错误，使用weights_only=False
        if "weights_only" in str(e) or "WeightsUnpickler" in str(e):
            print("警告: 使用weights_only=False加载模型文件。请确保模型文件来源可信。")
            return torch.load(model_path, weights_only=False)
        else:
            raise e

def test_model(code, ddpg, reward_type=3, verbose=True, save_results=True):
    """
    测试DDPG模型在股票交易中的表现
    
    参数:
        code (str): 股票代码
        ddpg (DDPG): 训练好的DDPG模型
        reward_type (int): 使用的奖励函数类型
        verbose (bool): 是否打印详细信息
        save_results (bool): 是否保存测试结果到文件
    
    返回:
        tuple: (基准收益率列表, 模型收益率列表, 日期列表, 交易信息列表)
    """
    # 分割数据
    datas, test_datas = get_data(code)
    
    # 设置环境
    env = stockEnv.StockTradingEnv(datas, test_datas)
    env.seed(1234)
    env.test_set()
    
    # 计算基准收益 (买入并持有策略)
    basic_profit = []
    date_list = []
    info_list = []
    
    for i in range(len(test_datas) - 1):
        basic_profit.append(test_datas['close'][i + 1] / test_datas['close'][0] - 1)
        date_list.append(test_datas['date'][i+1])
    
    # 使用模型测试
    state = env.reset()
    done = False
    max_money = INITIAL_ACCOUNT_BALANCE
    min_money = INITIAL_ACCOUNT_BALANCE
    money = INITIAL_ACCOUNT_BALANCE
    model_profit = []
    daily_returns = []  # 存储每日收益率用于夏普比率计算
    
    # 开始回测
    previous_money = INITIAL_ACCOUNT_BALANCE
    for i in range(len(test_datas) - 1):
        # 根据状态得到动作
        action = ddpg.select_action(state)

        # 执行动作
        if reward_type == 1:
            next_state, reward, done, money = env.step1(action)
        elif reward_type == 2:
            next_state, reward, done, money = env.step2(action)
        elif reward_type == 3:
            next_state, reward, done, money = env.step3(action)
        elif reward_type == 4:
            next_state, reward, done, money = env.step4(action)
        elif reward_type == 5:
            next_state, reward, done, money = env.step5(action)
        else:
            next_state, reward, done, money = env.step3(action)  # 默认使用step3
            
        # 计算当日收益率
        daily_return = (money - previous_money) / previous_money
        daily_returns.append(daily_return)
        previous_money = money
            
        model_profit.append(money / INITIAL_ACCOUNT_BALANCE - 1)
        max_money = max(money, max_money)
        min_money = min(money, min_money)
        
        # 记录交易信息
        current_step, open_action, close_action, shares, balance, total_value, percent = env.render()
        percent_str = f"{percent * 100:.2f}%"
        
        # 添加更多详细信息
        current_price = test_datas['close'].iloc[i]
        info_list.append({
            '日期': test_datas['date'].iloc[i],
            '当前价格': f"{current_price:.2f}",
            '开盘操作': open_action, 
            '收盘操作': close_action, 
            '持股数': shares, 
            '可用余额': f"{balance:.2f}", 
            '总资产': f"{total_value:.2f}", 
            '收益率': percent_str,
            '基准收益率': f"{basic_profit[i] * 100:.2f}%",
            '动作值': f"[{action[0]:.4f}]",
            '日收益率': f"{daily_return * 100:.4f}%"
        })
        
        # 状态更新
        state = next_state
    
    # 计算最终收益
    final_profit = (money / INITIAL_ACCOUNT_BALANCE - 1) * 100
    max_profit = (max_money / INITIAL_ACCOUNT_BALANCE - 1) * 100
    max_drawdown = (1 - min_money / INITIAL_ACCOUNT_BALANCE) * 100
    
    # 对比基准收益
    benchmark_final = basic_profit[-1] * 100
    outperform = final_profit - benchmark_final
    
    # 计算夏普比率
    if len(daily_returns) > 0:
        daily_return_mean = np.mean(daily_returns)
        daily_return_std = np.std(daily_returns, ddof=1)  # 使用样本标准差
        
        # 年化收益率和波动率 (假设252个交易日)
        annualized_return = (1 + daily_return_mean) ** 252 - 1
        annualized_volatility = daily_return_std * np.sqrt(252)
        
        # 夏普比率 (假设无风险利率为3%)
        risk_free_rate = 0.03
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        
        # 也计算基于最终收益的简化夏普比率
        simple_sharpe = (final_profit / 100 - risk_free_rate) / (daily_return_std * np.sqrt(252)) if daily_return_std > 0 else 0
    else:
        annualized_return = 0
        annualized_volatility = 0
        sharpe_ratio = 0
        simple_sharpe = 0
    
    if verbose:
        print(f"股票代码: {code}")
        print(f"最大收益: {max_profit:.2f}%")
        print(f"最大回撤: {max_drawdown:.2f}%")
        print(f"最终收益: {final_profit:.2f}%")
        print(f"基准收益: {benchmark_final:.2f}%")
        print(f"超额收益: {outperform:.2f}%")
        print(f"年化收益率: {annualized_return * 100:.2f}%")
        print(f"年化波动率: {annualized_volatility * 100:.2f}%")
        print(f"夏普比率: {sharpe_ratio:.4f}")
        print(f"简化夏普比率: {simple_sharpe:.4f}")
    
    # 保存测试结果到文件
    if save_results:
        save_test_results(code, info_list, final_profit, max_profit, max_drawdown, 
                         benchmark_final, outperform, reward_type, sharpe_ratio, 
                         annualized_return, annualized_volatility, simple_sharpe)
    
    return basic_profit, model_profit, date_list, info_list

def save_test_results(code, info_list, final_profit, max_profit, max_drawdown, 
                     benchmark_final, outperform, reward_type, sharpe_ratio=0, 
                     annualized_return=0, annualized_volatility=0, simple_sharpe=0):
    """
    保存测试结果到文件
    
    参数:
        code (str): 股票代码
        info_list (list): 详细交易信息列表
        final_profit (float): 最终收益率
        max_profit (float): 最大收益率
        max_drawdown (float): 最大回撤
        benchmark_final (float): 基准最终收益率
        outperform (float): 超额收益
        reward_type (int): 奖励函数类型
        sharpe_ratio (float): 夏普比率
        annualized_return (float): 年化收益率
        annualized_volatility (float): 年化波动率
        simple_sharpe (float): 简化夏普比率
    """
    import datetime
    
    # 确保results目录存在
    if not os.path.exists('./results'):
        os.makedirs('./results')
    
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存详细交易记录为CSV
    df = pd.DataFrame(info_list)
    csv_filename = f"./results/{code}_test_detail_{timestamp}.csv"
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    
    # 保存汇总信息为文本文件
    summary_filename = f"./results/{code}_test_summary_{timestamp}.txt"
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write(f"DDPG股票交易模型测试报告\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"测试时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"股票代码: {code}\n")
        f.write(f"奖励函数类型: {reward_type}\n")
        f.write(f"测试期间: {len(info_list)}个交易日\n")
        f.write(f"\n业绩指标:\n")
        f.write(f"-" * 30 + "\n")
        f.write(f"最终收益率: {final_profit:.2f}%\n")
        f.write(f"最大收益率: {max_profit:.2f}%\n")
        f.write(f"最大回撤: {max_drawdown:.2f}%\n")
        f.write(f"基准收益率(买入持有): {benchmark_final:.2f}%\n")
        f.write(f"超额收益: {outperform:.2f}%\n")
        f.write(f"年化收益率: {annualized_return * 100:.2f}%\n")
        f.write(f"\n风险指标:\n")
        f.write(f"-" * 30 + "\n")
        f.write(f"年化波动率: {annualized_volatility * 100:.2f}%\n")
        f.write(f"夏普比率: {sharpe_ratio:.4f}\n")
        f.write(f"简化夏普比率: {simple_sharpe:.4f}\n")
        
        # 夏普比率说明
        f.write(f"\n夏普比率说明:\n")
        f.write(f"-" * 30 + "\n")
        f.write(f"夏普比率 = (年化收益率 - 无风险利率) / 年化波动率\n")
        f.write(f"无风险利率假设: 3%\n")
        f.write(f"年化基础: 252个交易日\n")
        f.write(f"夏普比率评价标准:\n")
        f.write(f"  > 2.0: 非常优秀\n")
        f.write(f"  1.0-2.0: 优秀\n")
        f.write(f"  0.5-1.0: 良好\n")
        f.write(f"  0-0.5: 一般\n")
        f.write(f"  < 0: 较差\n")
        
        f.write(f"\n交易统计:\n")
        f.write(f"-" * 30 + "\n")
        
        # 统计交易次数
        buy_actions = sum(1 for info in info_list if '买入' in info['开盘操作'] or '买入' in info['收盘操作'])
        sell_actions = sum(1 for info in info_list if '卖出' in info['开盘操作'] or '卖出' in info['收盘操作'])
        
        f.write(f"买入操作次数: {buy_actions}\n")
        f.write(f"卖出操作次数: {sell_actions}\n")
        f.write(f"总交易次数: {buy_actions + sell_actions}\n")
        
        f.write(f"\n详细交易记录已保存至: {csv_filename}\n")
    
    print(f"\n测试结果已保存:")
    print(f"详细记录: {csv_filename}")
    print(f"汇总报告: {summary_filename}")

def plot_results(code, basic_profit, model_profit, date_list, save_path=None):
    """
    绘制测试结果图表
    
    参数:
        code (str): 股票代码
        basic_profit (list): 基准收益率列表
        model_profit (list): 模型收益率列表
        date_list (list): 日期列表
        save_path (str, optional): 图表保存路径
    """
    # 转换日期格式
    dates = pd.to_datetime(date_list)
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    plt.plot(dates, basic_profit, label='Buy & Hold Strategy', color='blue')
    plt.plot(dates, model_profit, label='DDPG Strategy', color='red')
    
    # 添加标签和标题
    plt.xlabel('Date')
    plt.ylabel('Return Rate')
    plt.title(f'DDPG Trading Strategy vs Buy & Hold - Stock {code}')
    plt.legend()
    plt.grid(True)
    
    # 格式化x轴日期
    plt.gcf().autofmt_xdate()
    
    # 保存或显示图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # 示例用法
    code = '600016'
    model_path = f"./models/reward-3/{code}/best.pt"
    
    if os.path.exists(model_path):
        ddpg = load_model(model_path)
        basic_profit, model_profit, date_list, info_list = test_model(code, ddpg)
        plot_results(code, basic_profit, model_profit, date_list, f"./results/{code}_test_result.png")
    else:
        print(f"模型文件 {model_path} 不存在，请先训练模型")


