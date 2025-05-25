import baostock as bs
import pandas as pd
import os
import time
import sys
import io
from datetime import datetime

def getData(code='000581', start_date='2000-1-1', end_date=str(datetime.now().date())):
    """
    从baostock获取股票历史数据
    
    参数:
        code (str): 股票代码
        start_date (str): 开始日期，格式为'yyyy-mm-dd'
        end_date (str): 结束日期，格式为'yyyy-mm-dd'
    
    返回:
        str: 保存的文件路径
    """
    # 临时重定向stdout来抑制baostock的输出
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        print("正在连接数据源...", file=old_stdout)
        # 登录系统
        lg = bs.login()
        
        print(f"正在获取股票 {code} 的历史数据...", file=old_stdout)
        # 获取沪深A股历史K线数据
        rs = bs.query_history_k_data_plus(
            f"sh.{code}",
            "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM",
            start_date=start_date, 
            end_date=end_date,
            frequency="d", 
            adjustflag="2"
        )

        # 处理结果集
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)

        # 确保目录存在
        os.makedirs("./data", exist_ok=True)
        
        # 保存结果集到csv文件
        path = f"./data/{code}.csv"
        result.to_csv(path, index=False)

        # 登出系统
        bs.logout()
        print("数据获取完成", file=old_stdout)
        
    finally:
        # 恢复stdout
        sys.stdout = old_stdout

    # 返回文件路径
    return path


def cleanData(datas):
    """
    清理数据，处理异常值
    
    参数:
        datas (DataFrame): 需要清理的数据
    
    返回:
        DataFrame: 清理后的数据
    """
    # 异常值处理
    datas = datas.ffill()  # 用前一个非缺失值填充
    datas = datas.bfill()  # 用后一个非缺失值填充
    
    # 去除volume为0的行
    datas = datas[datas['volume'] != '0']
    datas = datas.reset_index(drop=True)
    
    # 转换数据类型
    numeric_columns = ['open', 'high', 'low', 'close', 'preclose', 'volume', 
                       'amount', 'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']
    for col in numeric_columns:
        if col in datas.columns:
            datas[col] = pd.to_numeric(datas[col], errors='coerce')
    
    return datas


def computeData(datas):
    """
    计算技术指标
    
    参数:
        datas (DataFrame): 原始数据
    
    返回:
        DataFrame: 添加了技术指标的数据
    """
    days = len(datas)
    
    # 计算MA30 (30日移动平均线)
    datas['MA30'] = 0.0
    for i in range(days):
        if i == 0:
            datas.loc[i, 'MA30'] = datas.loc[i, 'close']
        elif i < 30:
            datas.loc[i, 'MA30'] = datas.loc[0:i, 'close'].mean()
        else:
            datas.loc[i, 'MA30'] = datas.loc[i-30:i, 'close'].mean()
    
    # 计算MA60 (60日移动平均线)
    datas['MA60'] = 0.0
    for i in range(days):
        if i == 0:
            datas.loc[i, 'MA60'] = datas.loc[i, 'close']
        elif i < 60:
            datas.loc[i, 'MA60'] = datas.loc[0:i, 'close'].mean()
        else:
            datas.loc[i, 'MA60'] = datas.loc[i-60:i, 'close'].mean()
    
    # 计算MACD
    short_ema = datas['close'].ewm(span=12).mean()
    long_ema = datas['close'].ewm(span=26).mean()
    datas['DIFF'] = short_ema - long_ema
    datas['DEA'] = datas['DIFF'].ewm(span=9).mean()
    datas['MACD'] = 2 * (datas['DIFF'] - datas['DEA'])
    
    # 计算TP (典型价格)
    datas['TP'] = (datas['high'] + datas['low'] + datas['close']) / 3
    
    # 计算MD30 (近30日移动偏差)
    datas['MD30'] = 0.0
    for i in range(days):
        if i >= 60:  # 确保有足够的数据计算MA30
            total = 0
            for j in range(30):
                total += abs(datas.loc[i-j, 'MA30'] - datas.loc[i-j, 'close'])
            datas.loc[i, 'MD30'] = total / 30
    
    # 计算Sigma30 (30日标准差)
    datas['Sigma30'] = 0.0
    for i in range(days):
        if i >= 30:
            datas.loc[i, 'Sigma30'] = datas.loc[i-30:i, 'close'].std()
    
    # 计算CCI30 (30日商品渠道指数)
    datas['CCI30'] = 0.0
    for i in range(days):
        if i >= 30 and datas.loc[i, 'Sigma30'] != 0:
            datas.loc[i, 'CCI30'] = (datas.loc[i, 'TP'] - datas.loc[i, 'MA30']) / (datas.loc[i, 'Sigma30'] * 0.015)
    
    # 计算BOLL上轨 (布林带上轨)
    datas['BOLLub30'] = 0.0
    for i in range(days):
        if i >= 30:
            datas.loc[i, 'BOLLub30'] = datas.loc[i, 'MA30'] + 2 * datas.loc[i, 'Sigma30']
    
    # 计算BOLL下轨 (布林带下轨)
    datas['BOLLlb30'] = 0.0
    for i in range(days):
        if i >= 30:
            datas.loc[i, 'BOLLlb30'] = datas.loc[i, 'MA30'] - 2 * datas.loc[i, 'Sigma30']
    
    # 计算RS30 (相对强弱指标)
    datas['RS30'] = 0.0
    for i in range(days):
        if i >= 30:
            sum_up = 0
            sum_down = 0
            for j in range(30):
                close_price = datas.loc[i-j-1, 'close']
                open_price = datas.loc[i-j-1, 'open']
                if close_price > open_price:
                    # 上涨
                    sum_up += close_price
                elif close_price < open_price:
                    # 下跌
                    sum_down += close_price
            # 避免除以0的情况
            if sum_down != 0:
                datas.loc[i, 'RS30'] = sum_up / sum_down
            else:
                datas.loc[i, 'RS30'] = 100  # 如果没有下跌，设置一个较大的值
    
    # 计算RSI30 (30日相对强弱指数)
    datas['RSI30'] = 0.0
    for i in range(days):
        if i >= 30:
            datas.loc[i, 'RSI30'] = 100 - (100 / (1 + datas.loc[i, 'RS30']))
    
    return datas


def zscoreData(datas):
    """
    对数据进行z-score标准化
    
    参数:
        datas (DataFrame): 原始数据
    
    返回:
        DataFrame: 标准化后的数据
    """
    # 将datas数据进行z-score标准化
    date_code = datas.iloc[:, 0:2]  # 保留日期和代码列
    numeric_data = datas.iloc[:, 2:]  # 只对数值列进行标准化
    
    # 按列进行z-score标准化
    z_score = (numeric_data - numeric_data.mean()) / numeric_data.std()
    
    # 合并回原始DataFrame
    standardized_data = pd.concat([date_code, z_score], axis=1)
    
    return standardized_data
