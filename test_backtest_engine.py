#!/usr/bin/env python3
"""
测试回测引擎
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append('src')

from strategies.grid_strategy import create_grid_strategy
from core.backtest_engine import BacktestEngine, BacktestConfig

def main():
    """测试主函数"""
    print("开始测试回测引擎...")

    try:
        # 读取解析的配置
        config_df = pd.read_csv('parsed_grid_configs.csv')
        if config_df.empty:
            print("错误: 找不到解析的配置文件")
            return

        sample_config = config_df.iloc[0]
        print(f"使用配置: {sample_config['stock_name']} ({sample_config['stock_code']})")
        print(f"基准价格: {sample_config['base_price']}")
        print(f"卖出百分比: {sample_config['sell_percentage']}%")
        print(f"买入百分比: {sample_config['buy_percentage']}%")

    except Exception as e:
        print(f"读取配置失败: {e}")
        return

    # 创建策略配置
    strategy_config = {
        'base_price': sample_config['base_price'],
        'sell_percentage': sample_config['sell_percentage'],
        'buy_percentage': sample_config['buy_percentage'],
        'position_size': 1000,
        'position_size_type': 'shares',
        'grid_count': 6
    }

    # 创建网格策略
    strategy = create_grid_strategy('basic_grid', strategy_config)

    # 创建回测配置
    backtest_config = BacktestConfig(
        initial_cash=100000.0,  # 10万初始资金
        commission_rate=0.0003,  # 万分之三手续费
        slippage_rate=0.001,     # 千分之一滑点
        min_trade_unit=100       # 最小交易单位100股
    )

    # 创建价格数据 - 使用更合理的波动
    base_price = sample_config['base_price']
    dates = pd.date_range('2023-01-01', periods=120, freq='D')

    # 创建有趋势的锯齿形价格，确保能触发网格交易
    prices = []
    current_price = base_price

    for i in range(120):
        # 创建更有规律的波动来测试网格
        cycle = i % 20
        if cycle < 10:  # 前10天下跌阶段
            change = -0.015  # 1.5%下跌
        else:  # 后10天上涨阶段
            change = 0.012   # 1.2%上涨

        # 添加随机噪声
        noise = np.random.normal(0, 0.008)  # 0.8%随机波动
        total_change = change + noise

        current_price *= (1 + total_change)
        current_price = max(current_price, base_price * 0.75)  # 价格下限
        current_price = min(current_price, base_price * 1.3)  # 价格上限
        prices.append(current_price)

    price_data = pd.DataFrame({
        'date': dates,
        'close': np.array(prices),
        'open': np.array(prices) * np.random.uniform(0.99, 1.01, 120),
        'high': np.array(prices) * np.random.uniform(1.00, 1.02, 120),
        'low': np.array(prices) * np.random.uniform(0.98, 1.00, 120)
    })

    print(f"价格数据: {len(price_data)}条记录")
    print(f"价格范围: {min(prices):.3f} - {max(prices):.3f}")
    print(f"价格波动率: {np.std(prices) / np.mean(prices):.2%}")

    # 创建回测引擎
    engine = BacktestEngine(backtest_config)

    # 运行回测
    print(f"\n开始回测...")
    results = engine.run_backtest(strategy, price_data, sample_config['stock_code'])

    # 显示结果
    print(f"\n{'='*60}")
    print(f"回测结果 - {sample_config['stock_name']}")
    print(f"{'='*60}")

    print(f"\n基本指标:")
    print(f"   初始资金: ¥{results['initial_value']:,.2f}")
    print(f"   最终价值: ¥{results['final_value']:,.2f}")
    print(f"   总收益率: {results['total_return']:+.2%}")
    print(f"   年化收益率: {results['annual_return']:+.2%}")
    print(f"   最大回撤: {results['max_drawdown']:+.2%}")
    print(f"   夏普比率: {results['sharpe_ratio']:.2f}")

    print(f"\n交易统计:")
    print(f"   总交易次数: {results['total_trades']}次")
    print(f"   胜率: {results['win_rate']:.2%}")
    print(f"   已实现盈亏: ¥{results['realized_pnl']:+,.2f}")
    print(f"   总手续费: ¥{results['total_commission']:,.2f}")
    print(f"   总滑点成本: ¥{results['total_slippage']:,.2f}")

    print(f"\n持仓信息:")
    print(f"   最终持仓: {results['final_position']:,}股")
    print(f"   持仓成本: ¥{results['final_cost']:,.2f}")

    if 'daily_values' in results and not results['daily_values'].empty:
        df_daily = results['daily_values']
        print(f"\n📅 每日统计:")
        print(f"   交易天数: {results['trading_days']}天")
        print(f"   平均日收益率: {df_daily['daily_return'].mean():+.4f}")
        print(f"   日收益率标准差: {df_daily['daily_return'].std():.4f}")
        print(f"   最高净值: ¥{df_daily['total_value'].max():,.2f}")
        print(f"   最低净值: ¥{df_daily['total_value'].min():,.2f}")

    # 显示交易历史
    trade_history = engine.get_trade_history()
    if not trade_history.empty:
        print(f"\n📋 交易历史 (最近10笔):")
        print(trade_history.tail(10)[['timestamp', 'side', 'quantity', 'price', 'amount']].to_string(index=False))

    # 测试不同策略的对比
    print(f"\n{'='*60}")
    print(f"策略对比测试")
    print(f"{'='*60}")

    strategies_to_test = ['basic_grid', 'dynamic_grid', 'martingale_grid']
    comparison_results = {}

    for strategy_name in strategies_to_test:
        try:
            print(f"\n测试策略: {strategy_name}")

            test_strategy = create_grid_strategy(strategy_name, strategy_config.copy())
            test_engine = BacktestEngine(backtest_config)

            test_results = test_engine.run_backtest(test_strategy, price_data, sample_config['stock_code'])

            comparison_results[strategy_name] = test_results

            print(f"   总收益率: {test_results['total_return']:+.2%}")
            print(f"   最大回撤: {test_results['max_drawdown']:+.2%}")
            print(f"   交易次数: {test_results['total_trades']}次")
            print(f"   夏普比率: {test_results['sharpe_ratio']:.2f}")

        except Exception as e:
            print(f"   测试失败: {e}")

    # 策略对比总结
    if comparison_results:
        print(f"\n🏆 策略对比总结:")
        print(f"{'策略':<15} {'收益率':<10} {'最大回撤':<10} {'夏普比率':<10} {'交易次数':<10}")
        print("-" * 65)

        for strategy_name, results in comparison_results.items():
            print(f"{strategy_name:<15} {results['total_return']:+.2%} {results['max_drawdown']:+.2%} "
                  f"{results['sharpe_ratio']:<10.2f} {results['total_trades']:<10}")

    print(f"\n回测引擎测试完成！")

if __name__ == "__main__":
    main()