#!/usr/bin/env python3
"""
测试市场数据获取器
"""

import sys
import pandas as pd
from pathlib import Path
sys.path.append('src')

# 直接导入，避免__init__.py依赖问题
from data.market_data_fetcher import MarketDataFetcher

def main():
    """测试主函数"""
    print("开始测试市场数据获取器...")

    # 创建获取器实例
    fetcher = MarketDataFetcher()

    # 测试获取单个ETF数据
    test_symbols = ["159682", "159380"]  # 科创50ETF, A500ETF

    for symbol in test_symbols:
        try:
            print(f"\n正在获取ETF数据: {symbol}")
            data = fetcher.fetch_etf_data(symbol)

            if not data.empty:
                print(f"成功获取 {len(data)} 条数据")
                print("数据预览:")
                print(data.head()[['date', 'open', 'high', 'low', 'close']])

                # 生成数据质量报告
                report = fetcher.get_data_quality_report(data)
                print("\n数据质量报告:")
                for key, value in report.items():
                    if key not in ['missing_values']:  # 跳过详细缺失值信息
                        print(f"   {key}: {value}")

            else:
                print(f"未获取到数据: {symbol}")

        except Exception as e:
            print(f"获取数据失败 {symbol}: {e}")
            import traceback
            traceback.print_exc()

    print("\n测试完成!")

if __name__ == "__main__":
    main()