#!/usr/bin/env python3
"""
Beta计算调试脚本
专门用于调试Beta系数计算过程
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from analysis.metrics_calculator import MetricsCalculator
from data.market_data_fetcher import MarketDataFetcher

# 设置详细日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_beta_calculation():
    """测试Beta计算的详细过程"""
    print("=" * 60)
    print("Beta计算详细调试")
    print("=" * 60)

    try:
        # 1. 获取基准数据
        print("\n1. 获取基准数据...")
        fetcher = MarketDataFetcher()
        benchmark_data = fetcher.fetch_benchmark_data('000300', '20240101', '20241231')

        if benchmark_data.empty:
            print("❌ 基准数据获取失败")
            return False

        print(f"✅ 基准数据获取成功: {len(benchmark_data)} 条记录")
        print(f"   日期范围: {benchmark_data['date'].min()} 到 {benchmark_data['date'].max()}")
        print(f"   价格范围: {benchmark_data['close'].min():.2f} - {benchmark_data['close'].max():.2f}")

        # 2. 创建模拟的回测数据
        print("\n2. 创建模拟回测数据...")

        # 使用不同的随机种子创建与基准有不同相关性的数据
        np.random.seed(123)  # 不同的种子

        dates = pd.bdate_range(start='2024-01-01', end='2024-12-31')
        n_days = len(dates)

        # 创建具有特定Beta特征的价格序列
        # 基准收益率
        benchmark_returns = benchmark_data['close'].pct_change().dropna()

        # 创建目标Beta为1.5的资产收益率
        target_beta = 1.5
        asset_returns = target_beta * benchmark_returns.values + np.random.normal(0, 0.01, len(benchmark_returns))

        # 转换为价格序列
        asset_prices = [100.0]
        for ret in asset_returns[1:]:
            asset_prices.append(asset_prices[-1] * (1 + ret))

        # 确保长度匹配
        min_length = min(len(dates), len(asset_prices))
        dates = dates[:min_length]
        asset_prices = asset_prices[:min_length]

        # 创建正确的daily_values DataFrame格式
        daily_values = pd.DataFrame({
            'date': dates,
            'total_value': asset_prices
        })

        print(f"✅ 模拟数据创建成功: {len(daily_values)} 条记录")
        print(f"   目标Beta: {target_beta}")
        print(f"   价格范围: {daily_values['total_value'].min():.2f} - {daily_values['total_value'].max():.2f}")

        # 3. 创建回测结果
        print("\n3. 创建回测结果结构...")
        backtest_results = {
            'initial_capital': 100000,
            'final_capital': daily_values['total_value'].iloc[-1] * 1000,  # 假设1000股
            'trades': [],
            'daily_values': daily_values
        }

        # 4. 创建MetricsCalculator并计算Beta
        print("\n4. 创建MetricsCalculator...")
        metrics_calc = MetricsCalculator(benchmark_data)

        # 5. 手动计算Beta进行对比
        print("\n5. 手动计算Beta进行验证...")

        # 计算基准收益率
        benchmark_returns_calc = benchmark_data['close'].pct_change().dropna()
        print(f"   基准收益率统计: 均值={benchmark_returns_calc.mean():.6f}, 标准差={benchmark_returns_calc.std():.6f}")

        # 计算资产收益率
        asset_returns_calc = daily_values['total_value'].pct_change().dropna()
        print(f"   资产收益率统计: 均值={asset_returns_calc.mean():.6f}, 标准差={asset_returns_calc.std():.6f}")

        # 对齐数据长度
        min_len = min(len(asset_returns_calc), len(benchmark_returns_calc))
        asset_returns_aligned = asset_returns_calc.iloc[-min_len:]
        benchmark_returns_aligned = benchmark_returns_calc.iloc[-min_len:]

        print(f"   对齐后数据长度: {len(asset_returns_aligned)}")

        # 计算协方差和方差
        covariance = np.cov(asset_returns_aligned, benchmark_returns_aligned)[0, 1]
        variance = np.var(benchmark_returns_aligned)

        if variance > 0:
            manual_beta = covariance / variance
            print(f"   手动计算Beta: {manual_beta:.6f}")
            print(f"   协方差: {covariance:.8f}")
            print(f"   基准方差: {variance:.8f}")
        else:
            print("   ❌ 基准方差为0，无法计算Beta")
            return False

        # 6. 使用MetricsCalculator计算Beta
        print("\n6. 使用MetricsCalculator计算Beta...")
        risk_metrics = metrics_calc._calculate_risk_metrics(backtest_results, pd.DataFrame())

        calculator_beta = risk_metrics.get('beta', 1.0)
        print(f"   MetricsCalculator Beta: {calculator_beta:.6f}")

        # 7. 对比结果
        print("\n7. 结果对比...")
        print(f"   目标Beta: {target_beta:.6f}")
        print(f"   手动计算Beta: {manual_beta:.6f}")
        print(f"   计算器Beta: {calculator_beta:.6f}")

        # 判断是否修复
        beta_diff = abs(calculator_beta - 1.0)
        if beta_diff < 0.01:
            print(f"   ❌ Beta仍接近1.0 (差异: {beta_diff:.6f})")
            return False
        else:
            print(f"   ✅ Beta计算修复成功! (差异: {beta_diff:.6f})")
            return True

    except Exception as e:
        print(f"❌ 调试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_beta_calculation()
    print("\n" + "=" * 60)
    if success:
        print("🎉 Beta计算调试成功!")
    else:
        print("❌ Beta计算仍有问题")
    print("=" * 60)