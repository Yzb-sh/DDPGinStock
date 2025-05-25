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
    print("训练模型: python main.py train --code 股票代码 --reward 奖励函数类型 --model_type 模型类型")
    print("测试模型: python main.py test --code 股票代码 --model_type 模型类型 --reward 奖励函数类型")
    print("获取交易建议: python main.py suggest --code 股票代码 --model_type 模型类型 --balance 可用余额 --shares 持股数 --initial 初始资金")
    print("=" * 50)
    print("参数说明:")
    print("  --code: 股票代码，如600016")
    print("  --reward: 奖励函数类型，取值范围1-5，默认为3")
    print("  --seed: 训练随机种子，默认为1234")
    print("  --model_type: 模型类型，ddpg或ddpg_lstm，默认为ddpg")
    print("  --seq_len: LSTM序列长度，默认为10（仅ddpg_lstm有效）")
    print("  --model: 模型路径，可手动指定或自动生成")
    print("  --balance: 当前可用余额")
    print("  --shares: 当前持股数")
    print("  --initial: 初始资金")
    print("  --save: 是否保存测试结果图表，默认为True")
    print("=" * 50)
    print("示例:")
    print("  # 训练DDPG+LSTM模型")
    print("  python main.py train --code 600016 --reward 3 --model_type ddpg_lstm")
    print("  # 测试模型")
    print("  python main.py test --code 600016 --model_type ddpg_lstm")
    print("  # 获取交易建议")
    print("  python main.py suggest --code 600016 --model_type ddpg_lstm --balance 10000 --shares 0 --initial 10000")

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
        # 确定模型路径，根据模型类型构建正确的路径
        if args.model:
            model_path = args.model
            # 从模型路径推断模型类型
            if 'ddpg_lstm' in model_path:
                model_type = 'ddpg_lstm'
            else:
                model_type = 'ddpg'
        else:
            model_type = args.model_type  # 使用用户指定的模型类型
            model_path = f"./models/reward-{args.reward}/{model_type}/{args.code}/best.pt"
        
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 {model_path} 不存在")
            print(f"请确认模型类型 ({model_type}) 和奖励函数类型 ({args.reward}) 是否正确")
            return
        
        print(f"开始测试模型, 股票代码: {args.code}, 模型路径: {model_path}, 模型类型: {model_type}")
        
        # 加载模型
        ddpg = test.load_model(model_path)
        
        # 根据模型类型设置测试环境 (需要修改test_model函数)
        basic_profit, model_profit, date_list, info_list = test.test_model(
            args.code, ddpg, int(args.reward), model_type=model_type, seq_len=args.seq_len
        )
        
        if args.save:
            save_path = f"./results/{args.code}_test_result_{model_type}.png"
            test.plot_results(args.code, basic_profit, model_profit, date_list, save_path)
        else:
            test.plot_results(args.code, basic_profit, model_profit, date_list)
        
    elif args.action == 'suggest':
        if args.balance is None or args.shares is None or args.initial is None:
            print("错误: 获取交易建议需要指定余额(--balance)、持股数(--shares)和初始资金(--initial)")
            return
        
        # 确定模型路径和模型类型
        if args.model:
            model_path = args.model
            # 从模型路径推断模型类型
            if 'ddpg_lstm' in model_path:
                model_type = 'ddpg_lstm'
            else:
                model_type = 'ddpg'
        else:
            model_type = args.model_type  # 使用用户指定的模型类型
            model_path = f"./models/reward-{args.reward}/{model_type}/{args.code}/best.pt"
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 {model_path} 不存在")
            print(f"请确认以下信息:")
            print(f"  - 股票代码: {args.code}")
            print(f"  - 模型类型: {model_type}")
            print(f"  - 奖励函数类型: {args.reward}")
            print(f"  - 期望的模型路径: {model_path}")
            print(f"\n正在搜索可用的模型文件...")
            
            # 列出可用的模型文件
            found_models = []
            models_base_dir = "./models"
            
            if os.path.exists(models_base_dir):
                # 遍历所有奖励函数类型
                for reward_dir in os.listdir(models_base_dir):
                    reward_path = os.path.join(models_base_dir, reward_dir)
                    if os.path.isdir(reward_path) and reward_dir.startswith('reward-'):
                        # 遍历所有模型类型
                        for mt in ['ddpg', 'ddpg_lstm']:
                            model_type_path = os.path.join(reward_path, mt)
                            if os.path.exists(model_type_path):
                                # 遍历所有股票代码
                                for stock_dir in os.listdir(model_type_path):
                                    stock_path = os.path.join(model_type_path, stock_dir)
                                    if os.path.isdir(stock_path):
                                        model_file = os.path.join(stock_path, "best.pt")
                                        if os.path.exists(model_file):
                                            # 转换为相对路径
                                            rel_path = os.path.relpath(model_file)
                                            found_models.append({
                                                'path': rel_path,
                                                'stock': stock_dir,
                                                'model_type': mt,
                                                'reward': reward_dir.split('-')[1]
                                            })
            
            if found_models:
                print(f"\n找到 {len(found_models)} 个可用的模型文件:")
                print("-" * 80)
                print(f"{'股票代码':<10} {'模型类型':<12} {'奖励函数':<8} {'模型路径'}")
                print("-" * 80)
                for model in found_models:
                    print(f"{model['stock']:<10} {model['model_type']:<12} {model['reward']:<8} {model['path']}")
                print("-" * 80)
                
                # 给出建议
                matching_stock = [m for m in found_models if m['stock'] == args.code]
                if matching_stock:
                    print(f"\n针对股票 {args.code} 的可用模型:")
                    for model in matching_stock:
                        print(f"  python main.py suggest --code {args.code} --model_type {model['model_type']} --reward {model['reward']} --balance {args.balance} --shares {args.shares} --initial {args.initial}")
                else:
                    print(f"\n未找到股票 {args.code} 的训练模型，请先训练模型:")
                    print(f"  python main.py train --code {args.code} --model_type {model_type} --reward {args.reward}")
            else:
                print(f"  未找到任何模型文件")
                print(f"  请先训练模型: python main.py train --code {args.code} --model_type {model_type} --reward {args.reward}")
            return
        
        print(f"获取交易建议, 股票代码: {args.code}, 模型类型: {model_type}")
        
        # 使用test模块的load_model函数加载模型，它已经处理了PyTorch兼容性问题
        try:
            model = test.load_model(model_path)
            print(f"成功加载模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            return
        
        # 获取交易建议，传递模型类型信息
        try:
            open_action, close_action = magic.get_trading_suggestion(
                args.code, args.balance, args.shares, args.initial, model, model_type, args.seq_len
            )
            
            print("\n=== 交易建议 ===")
            print(f"模型类型: {model_type}")
            print(f"奖励函数: {args.reward}")
            print(f"当前余额: {args.balance:,.2f}元")
            print(f"当前持股: {args.shares}股")
            print(f"初始资金: {args.initial:,.2f}元")
            print(f"建议:")
            print(f"  {open_action}")
            print(f"  {close_action}")
            print("=" * 20)
            
        except Exception as e:
            print(f"获取交易建议失败: {e}")
            return

if __name__ == "__main__":
    main()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
