#!/usr/bin/env python3
"""
调试网格价格计算
"""

import sys
import pandas as pd
import numpy as np
sys.path.append('src')

from strategies.grid_strategy import create_grid_strategy

def debug_grid_levels():
    """调试网格价格计算"""
    # 使用解析的配置
    config_df = pd.read_csv('parsed_grid_configs.csv')
    sample_config = config_df.iloc[0]

    print(f"股票: {sample_config['stock_name']} ({sample_config['stock_code']})")
    print(f"基准价格: {sample_config['base_price']}")
    print(f"卖出百分比: {sample_config['sell_percentage']}%")
    print(f"买入百分比: {sample_config['buy_percentage']}%")

    # 策略配置
    strategy_configs = {
        'base_price': sample_config['base_price'],
        'sell_percentage': sample_config['sell_percentage'],
        'buy_percentage': sample_config['buy_percentage'],
        'position_size': 1000,
        'position_size_type': 'shares',
        'grid_count': 8
    }

    # 创建基础网格策略
    strategy = create_grid_strategy('basic_grid', strategy_configs)

    print(f"\n网格价格计算:")
    print(f"网格数量: {len(strategy.grid_levels)}")

    buy_levels = [level for level in strategy.grid_levels if level.level_type == 'buy']
    sell_levels = [level for level in strategy.grid_levels if level.level_type == 'sell']

    print(f"\n买入网格价格 (共{len(buy_levels)}个):")
    for i, level in enumerate(buy_levels):
        distance_pct = (strategy.base_price - level.price) / strategy.base_price * 100
        print(f"  {i+1}. {level.price:.3f}元 (下跌{distance_pct:.2f}%)")

    print(f"\n卖出网格价格 (共{len(sell_levels)}个):")
    for i, level in enumerate(sell_levels):
        distance_pct = (level.price - strategy.base_price) / strategy.base_price * 100
        print(f"  {i+1}. {level.price:.3f}元 (上涨{distance_pct:.2f}%)")

    # 模拟价格数据
    base_price = sample_config['base_price']
    dates = pd.date_range('2023-01-01', periods=50, freq='D')

    # 创建更大幅度的价格波动来触发网格
    prices = []
    current_price = base_price

    for i in range(50):
        # 创建锯齿形波动，确保触及网格
        if i % 10 < 5:  # 前5天下跌
            change = -0.08  # 8%下跌
        else:  # 后5天上涨
            change = 0.06   # 6%上涨

        current_price *= (1 + change)
        current_price = max(current_price, base_price * 0.6)  # 价格下限
        current_price = min(current_price, base_price * 1.4)  # 价格上限
        prices.append(current_price)

    price_data = pd.DataFrame({
        'date': dates,
        'close': np.array(prices),
        'open': np.array(prices),
        'high': np.array(prices) * 1.01,
        'low': np.array(prices) * 0.99
    })

    print(f"\n模拟价格范围: {min(prices):.3f} - {max(prices):.3f}")

    # 生成信号
    signals = strategy.generate_signals(price_data)

    print(f"\n生成的交易信号: {len(signals)}个")

    if signals:
        buy_signals = [s for s in signals if s.signal_type.value == 'buy']
        sell_signals = [s for s in signals if s.signal_type.value == 'sell']

        print(f"买入信号: {len(buy_signals)}个")
        print(f"卖出信号: {len(sell_signals)}个")

        print(f"\n详细信号:")
        for i, signal in enumerate(signals):
            print(f"  {i+1}. {signal.timestamp.strftime('%Y-%m-%d')}: "
                  f"{signal.signal_type.value.upper()} {signal.quantity}股 @ {signal.price:.3f}")

        # 检查网格价格重复
        buy_prices = sorted([level.price for level in buy_levels])
        print(f"\n买入网格价格列表: {[f'{p:.3f}' for p in buy_prices]}")
        print(f"是否有重复价格: {len(buy_prices) != len(set(buy_prices))}")

    else:
        print("没有生成任何信号，尝试调整价格波动...")

        # 手动创建价格数据来测试
        test_prices = []
        # 从基准价开始，依次测试每个网格价格
        test_prices.extend([base_price * 0.9, base_price * 0.85, base_price * 0.8])  # 测试买入
        test_prices.extend([base_price * 1.05, base_price * 1.1, base_price * 1.15])  # 测试卖出

        test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=len(test_prices), freq='D'),
            'close': test_prices,
            'open': test_prices,
            'high': test_prices * 1.01,
            'low': test_prices * 0.99
        })

        print(f"测试价格: {test_prices}")

        # 重新生成信号
        signals = strategy.generate_signals(test_data)
        print(f"手动测试生成的信号: {len(signals)}个")

        for signal in signals:
            print(f"  {signal.timestamp.strftime('%Y-%m-%d')}: "
                  f"{signal.signal_type.value.upper()} {signal.quantity}股 @ {signal.price:.3f}")


if __name__ == "__main__":
    debug_grid_levels()