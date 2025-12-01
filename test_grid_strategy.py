#!/usr/bin/env python3
"""
测试网格交易策略
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append('src')

from strategies.grid_strategy import create_grid_strategy, SignalType

def main():
    """测试主函数"""
    print("开始测试网格交易策略...")

    # 从解析的配置中读取一个示例
    try:
        config_df = pd.read_csv('parsed_grid_configs.csv')
        if config_df.empty:
            print("错误: 找不到解析的配置文件")
            return

        # 使用第一个配置进行测试
        sample_config = config_df.iloc[0]
        print(f"使用配置: {sample_config['stock_name']} ({sample_config['stock_code']})")

    except Exception as e:
        print(f"读取配置失败: {e}")
        # 使用默认配置
        sample_config = {
            'stock_name': '科创50',
            'stock_code': '159682',
            'base_price': 1.408,
            'sell_percentage': 5.0,
            'buy_percentage': 10.0,
            'position_size': 1000
        }

    # 准备策略配置
    strategy_configs = {
        'base_price': sample_config['base_price'],
        'sell_percentage': sample_config['sell_percentage'],
        'buy_percentage': sample_config['buy_percentage'],
        'position_size': 1000,
        'position_size_type': 'shares',
        'grid_count': 8
    }

    # 创建模拟价格数据（基于实际价格波动）
    base_price = sample_config['base_price']
    dates = pd.date_range('2023-01-01', periods=200, freq='D')

    # 模拟价格走势（包含趋势和波动）
    trend = np.linspace(0, 0.1, 200)  # 轻微上涨趋势
    noise = np.random.normal(0, 0.03, 200)  # 波动
    prices = base_price * (1 + trend + noise)
    prices = np.maximum(prices, base_price * 0.7)  # 设置价格下限

    price_data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'open': prices * np.random.uniform(0.99, 1.01, 200),
        'high': prices * np.random.uniform(1.00, 1.02, 200),
        'low': prices * np.random.uniform(0.98, 1.00, 200)
    })

    print(f"价格数据: {len(price_data)}条记录")
    print(f"价格范围: {prices.min():.3f} - {prices.max():.3f}")
    print(f"基准价格: {base_price:.3f}")

    # 测试不同策略
    strategies = ['basic_grid', 'dynamic_grid', 'martingale_grid']

    for strategy_name in strategies:
        print(f"\n{'='*50}")
        print(f"测试策略: {strategy_name}")
        print(f"{'='*50}")

        try:
            # 创建策略
            strategy = create_grid_strategy(strategy_name, strategy_configs.copy())

            # 生成信号
            signals = strategy.generate_signals(price_data)

            print(f"生成信号数量: {len(signals)}")

            if signals:
                # 统计信号
                buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
                sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]

                print(f"买入信号: {len(buy_signals)}个")
                print(f"卖出信号: {len(sell_signals)}个")

                # 计算总交易量
                total_buy_volume = sum(s.quantity for s in buy_signals)
                total_sell_volume = sum(s.quantity for s in sell_signals)

                print(f"总买入量: {total_buy_volume:,}股")
                print(f"总卖出量: {total_sell_volume:,}股")

                # 显示前几个信号
                print(f"\n前5个信号:")
                for i, signal in enumerate(signals[:5]):
                    print(f"  {i+1}. {signal.timestamp.strftime('%Y-%m-%d')}: "
                          f"{signal.signal_type.value.upper()} "
                          f"{signal.quantity:,}股 @ {signal.price:.3f} - {signal.reason}")

                # 计算最终持仓
                final_position = strategy.current_position
                final_cost = strategy.total_cost
                avg_cost = final_cost / final_position if final_position > 0 else 0
                current_price = price_data['close'].iloc[-1]
                pnl = strategy._get_current_pnl(current_price)

                print(f"\n最终状态:")
                print(f"  持仓: {final_position:,}股")
                print(f"  平均成本: {avg_cost:.3f}元/股")
                print(f"  当前价格: {current_price:.3f}元/股")
                print(f"  总成本: {final_cost:,.2f}元")
                print(f"  浮动盈亏: {pnl:+,.2f}元")
                print(f"  盈亏率: {pnl/final_cost*100 if final_cost > 0 else 0:+.2f}%")

            else:
                print("未生成任何交易信号")

        except Exception as e:
            print(f"策略测试失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*50}")
    print("所有策略测试完成")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()