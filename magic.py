import numpy as np
import os
import data
import pandas as pd
from datetime import datetime

def get_state_vector(code, balance, shares, initial_fund, end_date=None, seq_len=10):
    """
    获取当前股票的状态向量，用于DDPG模型预测
    
    参数:
        code (str): 股票代码
        balance (float): 当前余额
        shares (int): 当前持股数
        initial_fund (float): 初始资金
        end_date (str, optional): 结束日期。默认为None表示使用当前日期
        seq_len (int): 状态序列长度，用于LSTM模型
    
    返回:
        numpy.ndarray: 状态向量或状态序列
    """
    # 设置日期范围
    if end_date is None:
        end_date = str(datetime.now().date())
    
    # 获取股票数据
    path = data.getData(code=code, start_date='2000-1-1', end_date=end_date)
    datas = pd.read_csv(path)
    datas = data.cleanData(datas)
    datas = data.computeData(datas)
    
    # 计算指标范围
    max_open = max(datas['open'])
    min_open = min(datas['open'])
    range_open = max_open - min_open
    
    max_high = max(datas['high'])
    min_high = min(datas['high'])
    range_high = max_high - min_high
    
    max_low = max(datas['low'])
    min_low = min(datas['low'])
    range_low = max_low - min_low
    
    max_close = max(datas['close'])
    min_close = min(datas['close'])
    range_close = max_close - min_close
    
    max_preclose = max(datas['preclose'])
    min_preclose = min(datas['preclose'])
    range_preclose = max_preclose - min_preclose
    
    max_volume = max(datas['volume'])
    min_volume = min(datas['volume'])
    range_volume = max_volume - min_volume
    
    max_turn = max(datas['turn'])
    min_turn = min(datas['turn'])
    range_turn = max_turn - min_turn
    
    max_pctChg = max(datas['pctChg'])
    min_pctChg = min(datas['pctChg'])
    range_pctChg = max_pctChg - min_pctChg
    
    max_peTTM = max(datas['peTTM'])
    min_peTTM = min(datas['peTTM'])
    range_peTTM = max_peTTM - min_peTTM
    
    max_MACD = max(datas['MACD'])
    min_MACD = min(datas['MACD'])
    range_MACD = max_MACD - min_MACD
    
    max_RSI = max(datas['RSI30'])
    min_RSI = min(datas['RSI30'])
    range_RSI = max_RSI - min_RSI
    
    max_CCI = max(datas['CCI30'])
    min_CCI = min(datas['CCI30'])
    range_CCI = max_CCI - min_CCI
    
    max_BOLLub = max(datas['BOLLub30'])
    min_BOLLub = min(datas['BOLLub30'])
    range_BOLLub = max_BOLLub - min_BOLLub
    
    max_BOLLlb = max(datas['BOLLlb30'])
    min_BOLLlb = min(datas['BOLLlb30'])
    range_BOLLlb = max_BOLLlb - min_BOLLlb
    
    # 构建状态向量函数
    def create_state(idx):
        return np.array([
            (datas['open'].iloc[idx] - min_open) / range_open,
            (datas['high'].iloc[idx] - min_high) / range_high,
            (datas['low'].iloc[idx] - min_low) / range_low,
            (datas['close'].iloc[idx] - min_close) / range_close,
            (datas['preclose'].iloc[idx] - min_preclose) / range_preclose,
            (datas['volume'].iloc[idx] - min_volume) / range_volume,
            (datas['turn'].iloc[idx] - min_turn) / range_turn,
            (datas['pctChg'].iloc[idx] - min_pctChg) / range_pctChg,
            (datas['peTTM'].iloc[idx] - min_peTTM) / range_peTTM,
            balance / (50 * initial_fund),
            shares / (50 * initial_fund / min_low),
            (shares * datas['close'].iloc[idx]) / (shares * datas['close'].iloc[idx] + balance),
            (datas['MACD'].iloc[idx] - min_MACD) / range_MACD,
            (datas['RSI30'].iloc[idx] - min_RSI) / range_RSI,
            (datas['CCI30'].iloc[idx] - min_CCI) / range_CCI,
            (datas['BOLLub30'].iloc[idx] - min_BOLLub) / range_BOLLub,
            (datas['BOLLlb30'].iloc[idx] - min_BOLLlb) / range_BOLLlb,
        ])
    
    # 如果提供了end_date，还返回最后一天的开盘价、收盘价和日期
    if end_date != str(datetime.now().date()):
        last_state = create_state(-1)
        return last_state, datas['open'].iloc[-1], datas['close'].iloc[-1], datas['date'].iloc[-1]
    
    # 获取最近seq_len个状态，构建序列
    obs_seq = []
    last_n_days = min(seq_len, len(datas))
    
    for i in range(last_n_days):
        idx = -last_n_days + i
        obs_seq.append(create_state(idx))
    
    # 如果数据不足seq_len天，用最早的状态填充
    if len(obs_seq) < seq_len:
        first_state = obs_seq[0]
        while len(obs_seq) < seq_len:
            obs_seq.insert(0, first_state)
    
    return np.array(obs_seq)


def get_trading_suggestion(code, balance, shares, initial_fund, model, model_type='ddpg', seq_len=10):
    """
    根据模型给出交易建议
    
    参数:
        code (str): 股票代码
        balance (float): 当前余额
        shares (int): 当前持股数
        initial_fund (float): 初始资金
        model: 训练好的DDPG模型
        model_type (str): 模型类型 ('ddpg' 或 'ddpg_lstm')
        seq_len (int): LSTM序列长度，仅当model_type='ddpg_lstm'时有效
    
    返回:
        tuple: (开盘建议, 收盘建议)
    """
    # 获取当前状态和最新价格
    path = data.getData(code=code, end_date=str(datetime.now().date()))
    datas = pd.read_csv(path)
    datas = data.cleanData(datas)
    datas = data.computeData(datas)
    latest_price = datas['close'].iloc[-1]
    latest_date = datas['date'].iloc[-1]
    
    print(f"数据日期: {latest_date}")
    print(f"股价 (前复权): {latest_price:.2f}元")
    
    # 直接使用已获取的数据计算状态向量，避免重复调用data.getData()
    # 计算指标范围
    max_open = max(datas['open'])
    min_open = min(datas['open'])
    range_open = max_open - min_open
    
    max_high = max(datas['high'])
    min_high = min(datas['high'])
    range_high = max_high - min_high
    
    max_low = max(datas['low'])
    min_low = min(datas['low'])
    range_low = max_low - min_low
    
    max_close = max(datas['close'])
    min_close = min(datas['close'])
    range_close = max_close - min_close
    
    max_preclose = max(datas['preclose'])
    min_preclose = min(datas['preclose'])
    range_preclose = max_preclose - min_preclose
    
    max_volume = max(datas['volume'])
    min_volume = min(datas['volume'])
    range_volume = max_volume - min_volume
    
    max_turn = max(datas['turn'])
    min_turn = min(datas['turn'])
    range_turn = max_turn - min_turn
    
    max_pctChg = max(datas['pctChg'])
    min_pctChg = min(datas['pctChg'])
    range_pctChg = max_pctChg - min_pctChg
    
    max_peTTM = max(datas['peTTM'])
    min_peTTM = min(datas['peTTM'])
    range_peTTM = max_peTTM - min_peTTM
    
    max_MACD = max(datas['MACD'])
    min_MACD = min(datas['MACD'])
    range_MACD = max_MACD - min_MACD
    
    max_RSI = max(datas['RSI30'])
    min_RSI = min(datas['RSI30'])
    range_RSI = max_RSI - min_RSI
    
    max_CCI = max(datas['CCI30'])
    min_CCI = min(datas['CCI30'])
    range_CCI = max_CCI - min_CCI
    
    max_BOLLub = max(datas['BOLLub30'])
    min_BOLLub = min(datas['BOLLub30'])
    range_BOLLub = max_BOLLub - min_BOLLub
    
    max_BOLLlb = max(datas['BOLLlb30'])
    min_BOLLlb = min(datas['BOLLlb30'])
    range_BOLLlb = max_BOLLlb - min_BOLLlb
    
    # 构建状态向量函数
    def create_state(idx):
        return np.array([
            (datas['open'].iloc[idx] - min_open) / range_open,
            (datas['high'].iloc[idx] - min_high) / range_high,
            (datas['low'].iloc[idx] - min_low) / range_low,
            (datas['close'].iloc[idx] - min_close) / range_close,
            (datas['preclose'].iloc[idx] - min_preclose) / range_preclose,
            (datas['volume'].iloc[idx] - min_volume) / range_volume,
            (datas['turn'].iloc[idx] - min_turn) / range_turn,
            (datas['pctChg'].iloc[idx] - min_pctChg) / range_pctChg,
            (datas['peTTM'].iloc[idx] - min_peTTM) / range_peTTM,
            balance / (50 * initial_fund),
            shares / (50 * initial_fund / min_low),
            (shares * datas['close'].iloc[idx]) / (shares * datas['close'].iloc[idx] + balance),
            (datas['MACD'].iloc[idx] - min_MACD) / range_MACD,
            (datas['RSI30'].iloc[idx] - min_RSI) / range_RSI,
            (datas['CCI30'].iloc[idx] - min_CCI) / range_CCI,
            (datas['BOLLub30'].iloc[idx] - min_BOLLub) / range_BOLLub,
            (datas['BOLLlb30'].iloc[idx] - min_BOLLlb) / range_BOLLlb,
        ])
    
    # 根据模型类型获取状态向量
    if model_type == 'ddpg_lstm':
        # LSTM模型需要序列状态
        obs_seq = []
        last_n_days = min(seq_len, len(datas))
        
        for i in range(last_n_days):
            idx = -last_n_days + i
            obs_seq.append(create_state(idx))
        
        # 如果数据不足seq_len天，用最早的状态填充
        if len(obs_seq) < seq_len:
            first_state = obs_seq[0]
            while len(obs_seq) < seq_len:
                obs_seq.insert(0, first_state)
        
        state = np.array(obs_seq)
        print(f"LSTM状态序列维度: {state.shape}")
    else:
        # 普通DDPG模型只需要单个状态向量
        state = create_state(-1)  # 取最后一个状态
        print(f"DDPG状态向量维度: {state.shape}")
    
    # 使用模型预测动作
    action = model.select_action(state)
    print(f"模型预测动作: [{action[0]:.4f}, {action[1]:.4f}]")
    
    # 解释动作
    open_action = "开盘时观望"
    close_action = "收盘时观望"
    
    # 卖出动作 (action[0])
    if action[0] > 0 and shares > 0:
        sell_ratio = min(action[0], 1.0)  # 限制在0-1之间
        sell_shares = int(shares * sell_ratio // 100) * 100  # 向下取整到100的倍数
        if sell_shares > 0:
            estimated_income = sell_shares * latest_price * 0.997  # 考虑手续费
            open_action = f"开盘时卖出{sell_shares}股，预计收入{estimated_income:.2f}元"
    
    # 买入动作 (action[1])
    if action[1] > 0 and balance > 0:
        buy_ratio = min(action[1], 1.0)  # 限制在0-1之间
        buy_amount = balance * buy_ratio
        # 考虑手续费和涨停限制
        max_shares = int(buy_amount / (latest_price * 1.103) // 100) * 100  # 考虑10%涨停+手续费
        if max_shares > 0 and buy_amount >= latest_price * 100:  # 至少能买1手
            close_action = f"收盘时花费{buy_amount:.2f}元买入约{max_shares}股"
    
    return open_action, close_action

if __name__ == "__main__":
    # 测试代码
    import torch
    
    code = '601318'
    balance = 100000
    shares = 1000
    initial_fund = 200000
    
    # 测试获取状态向量
    state = get_state_vector(code, balance, shares, initial_fund)
    print(f"状态向量维度: {state.shape}")
    
    # 加载模型进行测试
    model_path = f"./models/reward-3/{code}/best.pt"
    if os.path.exists(model_path):
        model = torch.load(model_path)
        open_action, close_action = get_trading_suggestion(code, balance, shares, initial_fund, model)
        print(f"开盘建议: {open_action}")
        print(f"收盘建议: {close_action}")
    else:
        print(f"模型文件 {model_path} 不存在，请先训练模型")
