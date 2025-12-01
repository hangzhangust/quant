#!/usr/bin/env python3
"""
调试信号生成逻辑
"""

import sys
import pandas as pd
import numpy as np
sys.path.append('src')

from strategies.grid_strategy import create_grid_strategy, SignalType

def debug_signal_logic():
    """调试信号生成逻辑"""
    # 使用简单配置
    config = {
        'base_price': 1.408,
        'sell_percentage': 5.0,
        'buy_percentage': 10.0,
        'position_size': 1000,
        'position_size_type': 'shares',
        'grid_count': 4  # 减少网格数量便于调试
    }

    strategy = create_grid_strategy('basic_grid', config)

    print(f"基准价格: {config['base_price']}")
    print(f"网格数量: {len(strategy.grid_levels)}")

    buy_levels = [level for level in strategy.grid_levels if level.level_type == 'buy']
    sell_levels = [level for level in strategy.grid_levels if level.level_type == 'sell']

    print(f"\n买入网格:")
    for i, level in enumerate(buy_levels):
        print(f"  {i+1}. 价格: {level.price:.3f}, 触发状态: {level.triggered}")

    print(f"\n卖出网格:")
    for i, level in enumerate(sell_levels):
        print(f"  {i+1}. 价格: {level.price:.3f}, 触发状态: {level.triggered}")

    # 手动测试一个价格点
    test_price = 1.295  # 这个价格应该触发1.302的买入网格
    print(f"\n测试价格: {test_price}")

    print(f"应该触发的买入网格检查:")
    for i, level in enumerate(buy_levels):
        should_trigger = test_price <= level.price
        print(f"  网格{i+1} ({level.price:.3f}): {should_trigger}")

    # 创建简单数据测试
    test_data = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01']),
        'close': [test_price],
        'open': [test_price],
        'high': [test_price * 1.01],
        'low': [test_price * 0.99]
    })

    print(f"\n生成信号前状态:")
    for i, level in enumerate(buy_levels):
        print(f"  买入网格{i+1}: 触发={level.triggered}")

    # 生成信号
    signals = strategy.generate_signals(test_data)

    print(f"\n生成的信号数量: {len(signals)}")
    for signal in signals:
        print(f"  {signal.signal_type.value.upper()} {signal.quantity} @ {signal.price:.3f}: {signal.reason}")

    print(f"\n生成信号后状态:")
    for i, level in enumerate(buy_levels):
        print(f"  买入网格{i+1}: 触发={level.triggered}")

    # 再次测试相同价格（不应该再次触发）
    print(f"\n再次测试相同价格 {test_price}:")
    signals2 = strategy.generate_signals(test_data)

    print(f"生成的信号数量: {len(signals2)}")
    for signal in signals2:
        print(f"  {signal.signal_type.value.upper()} {signal.quantity} @ {signal.price:.3f}")

if __name__ == "__main__":
    debug_signal_logic()