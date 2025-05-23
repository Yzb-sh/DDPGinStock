import random
import numpy as np
import gym
from gym import spaces
import pandas as pd

# 全局常量
MAX_PREDICT_RATE = 50  # 最大预测收益率倍数
INITIAL_ACCOUNT_BALANCE = 50e4  # 初始资金 50万元


class StockTradingEnv(gym.Env):
    """
    股票交易环境，用于强化学习的训练和测试
    
    该环境实现了一个股票交易系统，支持每日开盘卖出和收盘买入的操作策略。
    状态空间包含了股票的各种技术指标和当前账户信息。
    动作空间为连续值，表示开盘卖出比例和收盘买入比例。
    
    Attributes:
        action_space: 动作空间，包含开盘卖出和收盘买入的比例
        observation_space: 状态空间，包含17个归一化的特征
        service_charge: 交易手续费率
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, all_data, datas):
        """
        初始化交易环境
        
        Args:
            all_data (DataFrame): 全部股票数据，用于计算指标范围
            datas (DataFrame): 训练/测试用的股票数据
        """
        super(StockTradingEnv, self).__init__()

        # 定义动作空间：[开盘卖出比例, 收盘买入比例]，取值范围均为[-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        # 定义状态空间：17个特征，均归一化到[0, 1]区间
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(17,), dtype=np.float32)

        # 保存数据
        self.all_data = all_data
        self.datas = datas
        
        # 设置默认交易手续费率
        self.service_charge = 0.003

        # 初始化账户状态
        self.reset_account_state()
        
        # 计算数据的归一化范围
        self._calculate_normalization_ranges()

    def _calculate_normalization_ranges(self):
        """计算所有特征的归一化范围"""
        # 股票价格和交易量相关指标
        self.max_open = max(self.all_data['open'])
        self.min_open = min(self.all_data['open'])
        self.range_open = self.max_open - self.min_open
        
        self.max_high = max(self.all_data['high'])
        self.min_high = min(self.all_data['high'])
        self.range_high = self.max_high - self.min_high
        
        self.max_low = max(self.all_data['low'])
        self.min_low = min(self.all_data['low'])
        self.range_low = self.max_low - self.min_low
        
        self.max_close = max(self.all_data['close'])
        self.min_close = min(self.all_data['close'])
        self.range_close = self.max_close - self.min_close
        
        self.max_preclose = max(self.all_data['preclose'])
        self.min_preclose = min(self.all_data['preclose'])
        self.range_preclose = self.max_preclose - self.min_preclose
        
        self.max_volume = max(self.all_data['volume'])
        self.min_volume = min(self.all_data['volume'])
        self.range_volume = self.max_volume - self.min_volume
        
        # 交易和财务指标
        self.max_turn = max(self.all_data['turn'])
        self.min_turn = min(self.all_data['turn'])
        self.range_turn = self.max_turn - self.min_turn
        
        self.max_pctChg = max(self.all_data['pctChg'])
        self.min_pctChg = min(self.all_data['pctChg'])
        self.range_pctChg = self.max_pctChg - self.min_pctChg
        
        self.max_peTTM = max(self.all_data['peTTM'])
        self.min_peTTM = min(self.all_data['peTTM'])
        self.range_peTTM = self.max_peTTM - self.min_peTTM
        
        # 技术指标
        self.max_MACD = max(self.all_data['MACD'])
        self.min_MACD = min(self.all_data['MACD'])
        self.range_MACD = self.max_MACD - self.min_MACD
        
        self.max_RSI = max(self.all_data['RSI30'])
        self.min_RSI = min(self.all_data['RSI30'])
        self.range_RSI = self.max_RSI - self.min_RSI
        
        self.max_CCI = max(self.all_data['CCI30'])
        self.min_CCI = min(self.all_data['CCI30'])
        self.range_CCI = self.max_CCI - self.min_CCI
        
        self.max_BOLLub = max(self.all_data['BOLLub30'])
        self.min_BOLLub = min(self.all_data['BOLLub30'])
        self.range_BOLLub = self.max_BOLLub - self.min_BOLLub
        
        self.max_BOLLlb = max(self.all_data['BOLLlb30'])
        self.min_BOLLlb = min(self.all_data['BOLLlb30'])
        self.range_BOLLlb = self.max_BOLLlb - self.min_BOLLlb

    def reset_account_state(self):
        """重置账户状态为初始值"""
        # 资金和持股状态
        self.total_value_before = INITIAL_ACCOUNT_BALANCE  # 前一日总资产
        self.total_value = INITIAL_ACCOUNT_BALANCE  # 总资产
        self.balance = INITIAL_ACCOUNT_BALANCE  # 可用余额
        self.last_sell_totalvalue = INITIAL_ACCOUNT_BALANCE  # 上次卖出时的总资产
        self.shares = 0  # 持股数
        
        # 追踪变量
        self.current_step = 0  # 当前步数
        self.max_value = INITIAL_ACCOUNT_BALANCE  # 历史最大资产
        self.min_value = INITIAL_ACCOUNT_BALANCE  # 历史最小资产
        
        # 交易动作相关
        self.a_open = 0  # 开盘时的action值
        self.a_close = 0  # 收盘时的action值
        self.open_sold = 0  # 开盘时卖出的股数
        self.close_buy = 0  # 收盘时买入的股数
        
        # 统计指标
        self.shares_sold = 0  # 累计卖出股数
        self.shares_trade = 0  # 累计交易股数
        self.values_trade = 0  # 累计交易额
        self.cnt_buy = 0  # 买入次数
        self.cnt_sell = 0  # 卖出次数

    def seed(self, seed):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)

    def train_set(self):
        """设置为训练模式，使用较高的手续费率"""
        self.service_charge = 0.003
        
    def test_set(self):
        """设置为测试模式，使用较低的手续费率"""
        self.service_charge = 0.001

    def _next_observation(self):
        """
        获取当前的观察状态
        
        Returns:
            numpy.ndarray: 包含17个特征的状态向量
        """
        # 获取当前数据行
        current_data = self.datas.loc[self.current_step]
        
        # 构建状态向量
        obs = np.array([
            # 价格特征
            (current_data['open'] - self.min_open) / self.range_open,
            (current_data['high'] - self.min_high) / self.range_high,
            (current_data['low'] - self.min_low) / self.range_low,
            (current_data['close'] - self.min_close) / self.range_close,
            (current_data['preclose'] - self.min_preclose) / self.range_preclose,
            
            # 成交量和技术指标
            (current_data['volume'] - self.min_volume) / self.range_volume,
            (current_data['turn'] - self.min_turn) / self.range_turn,
            (current_data['pctChg'] - self.min_pctChg) / self.range_pctChg,
            (current_data['peTTM'] - self.min_peTTM) / self.range_peTTM,
            
            # 账户状态
            self.balance / (MAX_PREDICT_RATE * INITIAL_ACCOUNT_BALANCE),
            self.shares / (MAX_PREDICT_RATE * INITIAL_ACCOUNT_BALANCE / self.min_low),
            (self.total_value - self.balance) / self.total_value,
            
            # 更多技术指标
            (current_data['MACD'] - self.min_MACD) / self.range_MACD,
            (current_data['RSI30'] - self.min_RSI) / self.range_RSI,
            (current_data['CCI30'] - self.min_CCI) / self.range_CCI,
            (current_data['BOLLub30'] - self.min_BOLLub) / self.range_BOLLub,
            (current_data['BOLLlb30'] - self.min_BOLLlb) / self.range_BOLLlb,
        ])

        return obs

    def _take_action(self, action):
        """
        执行交易动作
        
        Args:
            action (numpy.ndarray): 包含两个元素的数组，分别表示开盘卖出比例和收盘买入比例
        """
        # 确保动作有效
        if np.isnan(action.any()):
            print("警告：动作包含NaN值")
            action = np.zeros_like(action)
            
        # 获取当前价格
        current_step_data = self.datas.loc[self.current_step]
        open_price = current_step_data["open"]
        close_price = current_step_data["close"]
        
        # 重置交易记录
        self.a_open = 0
        self.a_close = 0
        self.open_sold = 0
        self.close_buy = 0
        
        # 开盘时卖出操作
        if action[0] > 0:  # 卖出股票
            self.cnt_sell += 1
            self.a_open = action[0]  # 记录卖出动作值
            
            # 计算要卖出的股数（向下取整到100的倍数）
            max_shares = int(self.shares * action[0] // 100) * 100
            
            if max_shares == 0:
                self.cnt_sell -= 1
            else:
                # 更新账户状态
                sell_amount = max_shares * open_price * (1 - self.service_charge)
                self.balance += sell_amount
                self.shares -= max_shares
                self.open_sold = max_shares
                
                # 更新统计数据
                self.shares_sold = max_shares
                self.shares_trade += max_shares
                self.values_trade += max_shares * open_price * (1 + self.service_charge)
        
        # 收盘时买入操作
        if action[1] > 0:  # 买入股票
            self.cnt_buy += 1
            self.a_close = action[1]  # 记录买入动作值
            
            # 计算要买入的股数（向下取整到100的倍数）
            max_money = self.balance * action[1]
            max_shares = int(max_money / (close_price * (1 + self.service_charge)) // 100) * 100
            
            if max_shares == 0:
                self.cnt_buy -= 1
            else:
                # 更新账户状态
                buy_cost = max_shares * close_price * (1 + self.service_charge)
                self.balance -= buy_cost
                self.shares += max_shares
                self.close_buy = max_shares
                
                # 更新统计数据
                self.shares_sold = -max_shares  # 负值表示买入
                self.shares_trade += max_shares
                self.values_trade += max_shares * close_price * (1 + self.service_charge)
        
        # 更新资产总值
        self.total_value_before = self.total_value
        self.total_value = self.balance + self.shares * close_price
        
        # 更新历史最大/最小资产值
        self.max_value = max(self.total_value, self.max_value)
        self.min_value = min(self.total_value, self.min_value)

    def reset(self, new_df=None):
        """
        重置环境状态
        
        Args:
            new_df (DataFrame, optional): 新的数据集。默认为None，表示使用当前数据集。
            
        Returns:
            numpy.ndarray: 初始观察状态
        """
        # 重置账户状态
        self.reset_account_state()
        
        # 如果提供了新数据，则使用新数据
        if new_df is not None:
            self.datas = pd.read_csv(new_df)
            
        # 获取初始观察状态
        initial_obs = self._next_observation()
        
        # 初始化状态历史队列，用于LSTM模型
        # 只有在使用LSTM模式时才初始化状态历史
        if hasattr(self, 'use_lstm') and self.use_lstm:
            self.state_history = [initial_obs] * 10  # 初始状态重复10次
            
        return initial_obs

    def set_lstm_mode(self, use_lstm):
        """设置是否使用LSTM模式"""
        self.use_lstm = use_lstm

    def step(self, action, reward_type=3):
        """
        执行一步交易并返回结果
        
        Args:
            action (numpy.ndarray): 包含两个元素的数组，分别表示开盘卖出比例和收盘买入比例
            reward_type (int): 使用的奖励计算方式，取值1-5
            
        Returns:
            tuple: (新状态, 奖励, 是否结束, 当前总资产)
        """
        # 执行动作
        self._take_action(action)
        self.current_step += 1
        
        # 检查是否结束
        done = self._check_done()
        
        # 根据不同的奖励函数计算奖励
        reward_functions = {
            1: self._calculate_reward_1,
            2: self._calculate_reward_2,
            3: self._calculate_reward_3,
            4: self._calculate_reward_4,
            5: self._calculate_reward_5
        }
        
        reward = reward_functions.get(reward_type, self._calculate_reward_3)()
        
        # 获取新的观察状态
        obs = self._next_observation()
        
        # 如果启用LSTM模式，更新状态历史队列并返回序列状态
        if hasattr(self, 'use_lstm') and self.use_lstm and hasattr(self, 'state_history'):
            self.state_history.pop(0)  # 移除最旧的状态
            self.state_history.append(obs)  # 添加新状态
            # 返回完整序列状态，用于LSTM模型
            return np.array(self.state_history), reward, done, self.total_value
        
        # 普通模式直接返回单个状态
        return obs, reward, done, self.total_value
    
    def step1(self, action):
        """使用奖励函数1的step方法"""
        return self.step(action, reward_type=1)
    
    def step2(self, action):
        """使用奖励函数2的step方法"""
        return self.step(action, reward_type=2)
    
    def step3(self, action):
        """使用奖励函数3的step方法"""
        return self.step(action, reward_type=3)
    
    def step4(self, action):
        """使用奖励函数4的step方法"""
        return self.step(action, reward_type=4)
    
    def step5(self, action):
        """使用奖励函数5的step方法"""
        return self.step(action, reward_type=5)

    def _check_done(self):
        """
        检查是否达到结束条件
        
        Returns:
            bool: 如果达到结束条件则为True，否则为False
        """
        # 资金不足
        if self.balance < 0:
            print('资金不足，交易结束')
            return True
            
        # 达到最大收益率
        if self.total_value >= INITIAL_ACCOUNT_BALANCE * MAX_PREDICT_RATE:
            print(f"在第{self.current_step}步达到了{MAX_PREDICT_RATE}倍的收益率!")
            print(f"共买入{self.cnt_buy}次, 卖出{self.cnt_sell}次")
            return True
            
        # 达到数据末尾
        if self.current_step >= len(self.datas) - 2:
            print(f"总资产为{self.total_value}")
            print(f"共买入{self.cnt_buy}次, 卖出{self.cnt_sell}次")
            return True
            
        return False

    def _calculate_reward_1(self):
        """
        奖励函数1：直接使用资产变化作为奖励
        
        Returns:
            float: 计算得到的奖励值
        """
        return self.total_value - self.total_value_before

    def _calculate_reward_2(self):
        """
        奖励函数2：使用收益率作为奖励，亏损时有惩罚
        
        Returns:
            float: 计算得到的奖励值
        """
        # 环境结束时使用总收益率作为奖励
        done = self.current_step >= len(self.datas) - 2 or self.balance < 0 or self.total_value >= INITIAL_ACCOUNT_BALANCE * MAX_PREDICT_RATE
        if done:
            return (self.total_value - INITIAL_ACCOUNT_BALANCE) / INITIAL_ACCOUNT_BALANCE
        
        # 正常交易时，盈利给予正奖励，亏损给予固定惩罚
        profit = (self.total_value - INITIAL_ACCOUNT_BALANCE) / INITIAL_ACCOUNT_BALANCE
        return profit if profit > 0 else -0.1

    def _calculate_reward_3(self):
        """
        奖励函数3：基于交易动作和价格变化的复杂奖励
        
        Returns:
            float: 计算得到的奖励值
        """
        # 环境结束时使用总收益率作为奖励
        done = self.current_step >= len(self.datas) - 2 or self.balance < 0 or self.total_value >= INITIAL_ACCOUNT_BALANCE * MAX_PREDICT_RATE
        if done:
            return (self.total_value - INITIAL_ACCOUNT_BALANCE) / INITIAL_ACCOUNT_BALANCE * 100 * self.max_high
        
        reward = 0
        # 开盘卖出的奖励
        if self.a_open > 0:
            # 根据卖出后的资产变化计算奖励
            reward += (self.datas['open'][self.current_step - 1] * self.shares + self.balance - self.last_sell_totalvalue) / \
                     self.last_sell_totalvalue * 100 * self.datas['open'][self.current_step - 1]
            self.last_sell_totalvalue = self.datas['open'][self.current_step - 1] * self.shares + self.balance
        elif self.a_open == 0:
            # 观望时的奖励基于价格变化
            reward += 0.3 * 100 * (self.datas['open'][self.current_step] - self.datas['open'][self.current_step - 1])
        
        # 收盘买入的奖励
        if self.a_close > 0:
            reward += self.a_close * 100 * (self.datas['open'][self.current_step] - self.datas['close'][self.current_step - 1])
        elif self.a_close == 0:
            reward += 0.3 * 100 * (self.datas['close'][self.current_step - 1] - self.datas['open'][self.current_step])
            
        return reward

    def _calculate_reward_4(self):
        """
        奖励函数4：基于开盘价和收盘价差异的奖励
        
        Returns:
            float: 计算得到的奖励值
        """
        price_c_t = self.datas['close'][self.current_step - 1]
        price_o_next_t = self.datas['open'][self.current_step]
        price_o_t = self.datas['open'][self.current_step - 1]
        
        # 收盘买入的奖励基于第二天开盘价与当天收盘价的差值
        reward_close = (price_o_next_t * 0.997 - price_c_t) * 100 * self.a_close
        
        # 开盘卖出的奖励基于价格走势
        if self.a_open > 0:
            reward_open = (price_o_t * 0.997 - price_c_t) * 100 * self.a_open
        else:
            reward_open = (price_o_next_t * 0.997 - price_o_t) * 100 * self.a_open
            
        return reward_open + reward_close

    def _calculate_reward_5(self):
        """
        奖励函数5：另一种基于价格差异的奖励
        
        Returns:
            float: 计算得到的奖励值
        """
        price_c_t = self.datas['close'][self.current_step - 1]
        price_o_next_t = self.datas['open'][self.current_step]
        price_o_t = self.datas['open'][self.current_step - 1]
        
        # 收盘买入的奖励
        reward_close = (price_o_next_t * 0.997 - price_c_t) * 100 * self.a_close
        
        # 开盘卖出的奖励
        reward_open = (price_o_t * 0.997 - price_c_t) * 100 * self.a_open
            
        return reward_open + reward_close

    def render(self, mode='human'):
        """
        渲染当前环境状态
        
        Args:
            mode (str): 渲染模式，目前仅支持'human'
            
        Returns:
            tuple: 当前步数、开盘操作、收盘操作、持股数、可用余额、总资产、收益率
        """
        # 计算收益
        profit = self.total_value - INITIAL_ACCOUNT_BALANCE
        percent = profit / INITIAL_ACCOUNT_BALANCE
        
        # 打印环境信息
        print('-' * 30)
        print(f'步数: {self.current_step}')
        print(f'可用资金: {self.balance}')
        
        # 打印开盘操作
        if self.open_sold > 0:
            open_action = f'开盘时卖出{self.open_sold}股'
        else:
            open_action = '开盘时不操作'
        print(open_action)
        
        # 打印收盘操作
        if self.close_buy > 0:
            close_action = f'收盘时买入{self.close_buy}股'
        else:
            close_action = '收盘时不操作'
        print(close_action)
        
        # 打印账户状态
        print(f'持股数: {self.shares}')
        print(f'总市值: {self.total_value}')
        print(f'盈利: {profit}')
        print(f'盈利率: {percent}')
        
        return self.current_step, open_action, close_action, self.shares, self.balance, self.total_value, percent
