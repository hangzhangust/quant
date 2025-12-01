#!/usr/bin/env python3
"""
完整的网格交易回测系统演示
Complete Grid Trading Backtest System Demo
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append('src')

def demo_complete_system():
    """演示完整系统功能"""
    print("="*80)
    print("网格交易策略回测系统 - 完整演示")
    print("="*80)

    try:
        # 1. 数据解析演示
        print("\n1. 网格配置数据解析")
        print("-" * 40)

        from data.grid_config_parser import GridConfigParser
        parser = GridConfigParser()

        # 解析Table.xls
        df = parser.parse_excel_file('Table.xls')
        print(f"解析结果: {len(df)}个有效配置")

        # 显示统计信息
        stats = parser.get_summary_statistics(df)
        print(f"平均基准价格: {stats['avg_base_price']:.3f}")
        print(f"平均卖出百分比: {stats['avg_sell_percentage']:.2f}%")
        print(f"平均买入百分比: {stats['avg_buy_percentage']:.2f}%")

        # 2. 市场数据获取演示
        print("\n2. 市场数据获取")
        print("-" * 40)

        from data.market_data_fetcher import MarketDataFetcher
        fetcher = MarketDataFetcher()

        # 获取第一个ETF的数据
        symbol = df.iloc[0]['stock_code']
        print(f"获取 {symbol} 的历史数据...")

        market_data = fetcher.fetch_etf_data(symbol)
        if not market_data.empty:
            print(f"成功获取 {len(market_data)} 条历史数据")
            print(f"时间范围: {market_data['date'].min().date()} 至 {market_data['date'].max().date()}")
            print(f"价格范围: {market_data['close'].min():.3f} - {market_data['close'].max():.3f}")

            # 数据质量报告
            quality_report = fetcher.get_data_quality_report(market_data)
            print(f"数据完整性: {quality_report['data_completeness']:.1f}%")
        else:
            print("获取市场数据失败")
            return

        # 3. 网格策略演示
        print("\n3. 网格交易策略")
        print("-" * 40)

        from strategies.grid_strategy import create_grid_strategy

        # 使用第一个配置创建策略
        config_row = df.iloc[0]
        strategy_config = {
            'base_price': config_row['base_price'],
            'sell_percentage': config_row['sell_percentage'],
            'buy_percentage': config_row['buy_percentage'],
            'position_size': 1000,
            'grid_count': 6
        }

        print(f"策略配置: {config_row['stock_name']}")
        print(f"基准价格: {strategy_config['base_price']}")
        print(f"网格间距: 卖出{strategy_config['sell_percentage']}%, 买入{strategy_config['buy_percentage']}%")

        # 测试不同策略
        strategy_types = ['basic_grid', 'dynamic_grid']
        strategies = {}

        for strategy_type in strategy_types:
            try:
                strategy = create_grid_strategy(strategy_type, strategy_config.copy())
                strategies[strategy_type] = strategy
                print(f"创建 {strategy_type} 策略成功")
            except Exception as e:
                print(f"创建 {strategy_type} 策略失败: {e}")

        # 4. 回测引擎演示
        print("\n4. 回测引擎")
        print("-" * 40)

        from core.backtest_engine import BacktestEngine, BacktestConfig

        backtest_config = BacktestConfig(
            initial_cash=100000.0,
            commission_rate=0.0003,
            slippage_rate=0.001
        )

        print(f"回测配置:")
        print(f"   初始资金: {backtest_config.initial_cash:,.2f}")
        print(f"   手续费率: {backtest_config.commission_rate:.4f}")
        print(f"   滑点率: {backtest_config.slippage_rate:.4f}")

        # 使用最近的100天数据进行回测
        test_data = market_data.tail(100).copy()
        print(f"回测数据: {len(test_data)}条记录")

        # 5. 执行回测
        print("\n5. 回测结果")
        print("-" * 40)

        backtest_results = {}

        for strategy_name, strategy in strategies.items():
            try:
                engine = BacktestEngine(backtest_config)
                results = engine.run_backtest(strategy, test_data, symbol)
                backtest_results[strategy_name] = results

                print(f"\n{strategy_name} 策略结果:")
                print(f"   总收益率: {results['total_return']:+.2%}")
                print(f"   最大回撤: {results['max_drawdown']:+.2%}")
                print(f"   夏普比率: {results['sharpe_ratio']:.2f}")
                print(f"   交易次数: {results['total_trades']}")
                print(f"   胜率: {results['win_rate']:.2%}")

            except Exception as e:
                print(f"{strategy_name} 策略回测失败: {e}")

        # 6. 策略对比
        print("\n6. 策略对比分析")
        print("-" * 40)

        if backtest_results:
            print(f"{'策略':<15} {'收益率':<10} {'最大回撤':<10} {'夏普比率':<10} {'交易次数':<10}")
            print("-" * 70)

            for strategy_name, results in backtest_results.items():
                print(f"{strategy_name:<15} {results['total_return']:+.2%} {results['max_drawdown']:+.2%} "
                      f"{results['sharpe_ratio']:<10.2f} {results['total_trades']:<10}")

            # 找出最佳策略
            best_strategy = max(backtest_results.items(), key=lambda x: x[1]['sharpe_ratio'])
            print(f"\n最佳策略 (按夏普比率): {best_strategy[0]}")
            print(f"夏普比率: {best_strategy[1]['sharpe_ratio']:.2f}")
            print(f"总收益率: {best_strategy[1]['total_return']:+.2%}")

        # 7. 参数分析建议
        print("\n7. 参数分析建议")
        print("-" * 40)

        analyze_grid_parameters(df, market_data)

        print("\n" + "="*80)
        print("系统演示完成！")
        print("="*80)

        # 保存结果
        save_demo_results(df, backtest_results)

    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def analyze_grid_parameters(config_df, market_data):
    """分析网格参数并提供建议"""
    print("基于历史数据的参数分析:")

    # 分析当前参数
    avg_sell = config_df['sell_percentage'].mean()
    avg_buy = config_df['buy_percentage'].mean()

    print(f"1. 当前设置:")
    print(f"   平均卖出网格: {avg_sell:.2f}%")
    print(f"   平均买入网格: {avg_buy:.2f}%")
    print(f"   买卖比例: 1:{avg_buy/avg_sell:.2f}")

    # 计算历史波动率
    returns = market_data['close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)  # 年化波动率
    daily_volatility = returns.std()

    print(f"\n2. 市场特征分析:")
    print(f"   年化波动率: {volatility:.2%}")
    print(f"   日均波动率: {daily_volatility:.2%}")

    # 参数建议
    print(f"\n3. 参数优化建议:")

    if daily_volatility > 0.03:  # 高波动
        suggested_sell = max(avg_sell * 0.8, 2.0)
        suggested_buy = max(avg_buy * 0.8, 3.0)
        print(f"   高波动市场，建议缩小网格间距")
        print(f"   建议卖出网格: {suggested_sell:.2f}%")
        print(f"   建议买入网格: {suggested_buy:.2f}%")
    elif daily_volatility < 0.015:  # 低波动
        suggested_sell = min(avg_sell * 1.2, 8.0)
        suggested_buy = min(avg_buy * 1.2, 10.0)
        print(f"   低波动市场，建议扩大网格间距")
        print(f"   建议卖出网格: {suggested_sell:.2f}%")
        print(f"   建议买入网格: {suggested_buy:.2f}%")
    else:  # 适中波动
        print(f"   当前网格间距设置合理")

    # 买卖平衡建议
    sell_buy_ratio = avg_buy / avg_sell
    if sell_buy_ratio > 2.0:
        print(f"   买入网格过大，建议调整为更平衡的设置")
    elif sell_buy_ratio < 1.5:
        print(f"   买入网格可能过小，建议适当增大")

def save_demo_results(config_df, backtest_results):
    """保存演示结果"""
    try:
        # 保存配置分析
        config_df.to_csv('demo_configs_analysis.csv', index=False, encoding='utf-8-sig')

        # 保存回测结果
        if backtest_results:
            results_summary = []
            for strategy_name, results in backtest_results.items():
                results_summary.append({
                    'strategy': strategy_name,
                    'total_return': results['total_return'],
                    'annual_return': results['annual_return'],
                    'max_drawdown': results['max_drawdown'],
                    'sharpe_ratio': results['sharpe_ratio'],
                    'total_trades': results['total_trades'],
                    'win_rate': results['win_rate'],
                    'realized_pnl': results['realized_pnl']
                })

            results_df = pd.DataFrame(results_summary)
            results_df.to_csv('demo_backtest_results.csv', index=False, encoding='utf-8-sig')

        print(f"\n结果已保存:")
        print(f"   demo_configs_analysis.csv - 配置分析")
        print(f"   demo_backtest_results.csv - 回测结果")

    except Exception as e:
        print(f"保存结果失败: {e}")

if __name__ == "__main__":
    demo_complete_system()