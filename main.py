# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import argparse
import os
import torch
import train
import test
import magic

def print_help():
    """打印帮助信息"""
    print("DDPG股票交易模型 - 使用说明")
    print("=" * 50)
    print("训练模型: python main.py train --code 股票代码 --reward 奖励函数类型 --seed 随机种子")
    print("测试模型: python main.py test --code 股票代码 --model 模型路径 --reward 奖励函数类型")
    print("获取交易建议: python main.py suggest --code 股票代码 --model 模型路径 --balance 可用余额 --shares 持股数 --initial 初始资金")
    print("=" * 50)
    print("参数说明:")
    print("  --code: 股票代码，如600016")
    print("  --reward: 奖励函数类型，取值范围1-5，默认为3")
    print("  --seed: 训练随机种子，默认为1234")
    print("  --model: 模型路径，默认为./models/reward-3/股票代码/best.pt")
    print("  --balance: 当前可用余额")
    print("  --shares: 当前持股数")
    print("  --initial: 初始资金")
    print("  --save: 是否保存测试结果图表，默认为True")
    print("=" * 50)
    print("示例:")
    print("  python main.py train --code 600016 --reward 3 --seed 1234")
    print("  python main.py test --code 600016")
    print("  python main.py suggest --code 600016 --balance 100000 --shares 1000 --initial 200000")

def main():
    """主函数，处理命令行参数并执行相应操作"""
    parser = argparse.ArgumentParser(description='DDPG股票交易模型')
    parser.add_argument('action', choices=['train', 'test', 'suggest', 'help'], 
                        help='要执行的操作: train(训练), test(测试), suggest(交易建议), help(帮助)')
    parser.add_argument('--code', type=str, help='股票代码')
    parser.add_argument('--reward', type=str, default='3', help='奖励函数类型(1-5)')
    parser.add_argument('--seed', type=int, default=1234, help='随机种子')
    parser.add_argument('--model', type=str, help='模型路径')
    parser.add_argument('--balance', type=float, help='当前可用余额')
    parser.add_argument('--shares', type=int, help='当前持股数')
    parser.add_argument('--initial', type=float, help='初始资金')
    parser.add_argument('--save', type=bool, default=True, help='是否保存测试结果图表')
    parser.add_argument('--model_type', choices=['ddpg', 'ddpg_lstm'], default='ddpg', 
                    help='模型类型: ddpg（默认）或ddpg_lstm')
    parser.add_argument('--seq_len', type=int, default=10, 
                    help='LSTM序列长度（仅当model_type=ddpg_lstm时生效）')
    
    args = parser.parse_args()
    
    # 创建必要的文件夹
    train.create_directories()
    
    if args.action == 'help':
        print_help()
        return
    
    if args.code is None:
        print("错误: 必须指定股票代码")
        print_help()
        return
    
    if args.action == 'train':
        print(f"开始训练模型, 股票代码: {args.code}, 奖励函数: {args.reward}, 随机种子: {args.seed}")
        train.train(args.code, args.reward, args.seed, args.model_type, args.seq_len)
        
    elif args.action == 'test':
        # 确定模型路径
        model_path = args.model if args.model else f"./models/reward-{args.reward}/{args.code}/best.pt"
        
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 {model_path} 不存在")
            return
        
        print(f"开始测试模型, 股票代码: {args.code}, 模型路径: {model_path}")
        ddpg = test.load_model(model_path)
        basic_profit, model_profit, date_list, info_list = test.test_model(args.code, ddpg, int(args.reward))
        
        if args.save:
            save_path = f"./results/{args.code}_test_result.png"
            test.plot_results(args.code, basic_profit, model_profit, date_list, save_path)
        else:
            test.plot_results(args.code, basic_profit, model_profit, date_list)
        
    elif args.action == 'suggest':
        if args.balance is None or args.shares is None or args.initial is None:
            print("错误: 获取交易建议需要指定余额(--balance)、持股数(--shares)和初始资金(--initial)")
            return
        
        # 确定模型路径
        model_path = args.model if args.model else f"./models/reward-{args.reward}/{args.code}/best.pt"
        
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 {model_path} 不存在")
            return
        
        print(f"获取交易建议, 股票代码: {args.code}")
        model = torch.load(model_path)
        open_action, close_action = magic.get_trading_suggestion(
            args.code, args.balance, args.shares, args.initial, model
        )
        
        print("\n交易建议:")
        print(f"  {open_action}")
        print(f"  {close_action}")

if __name__ == "__main__":
    main()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
