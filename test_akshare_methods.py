#!/usr/bin/env python3
"""
测试AKShare不同的ETF数据获取方法
"""

import sys
import pandas as pd
sys.path.append('src')

import akshare as ak

def test_akshare_etf_methods():
    """测试AKShare的ETF数据获取方法"""

    test_symbol = "159682"  # 科创50ETF
    start_date = "20241101"
    end_date = "20241201"

    print(f"测试ETF代码: {test_symbol}")
    print(f"时间范围: {start_date} - {end_date}")
    print()

    # 方法1: fund_etf_hist_em
    print("=== 方法1: fund_etf_hist_em ===")
    try:
        data1 = ak.fund_etf_hist_em(symbol=test_symbol, period="daily", start_date=start_date, end_date=end_date, adjust="hfq")
        if data1 is not None and not data1.empty:
            print(f"[OK] 获取到 {len(data1)} 条数据")
            print(f"列名: {data1.columns.tolist()}")
            print("数据预览:")
            print(data1.head(3))
        else:
            print("[ERROR] 未获取到数据")
    except Exception as e:
        print(f"[ERROR] 方法1失败: {e}")

    print()

    # 方法2: fund_etf_hist_sina
    print("=== 方法2: fund_etf_hist_sina ===")
    try:
        # 转换代码格式
        sina_symbol = f"sz{test_symbol}"  # 深圳ETF
        print(f"新浪代码格式: {sina_symbol}")

        data2 = ak.fund_etf_hist_sina(symbol=sina_symbol)
        if data2 is not None and not data2.empty:
            print(f"[OK] 获取到 {len(data2)} 条数据")
            print(f"列名: {data2.columns.tolist()}")
            print("数据预览:")
            print(data2.head(3))
        else:
            print("[ERROR] 未获取到数据")
    except Exception as e:
        print(f"[ERROR] 方法2失败: {e}")

    print()

    # 方法3: stock_zh_a_hist (通用股票历史数据)
    print("=== 方法3: stock_zh_a_hist ===")
    try:
        data3 = ak.stock_zh_a_hist(symbol=test_symbol, period="daily", start_date=start_date, end_date=end_date, adjust="hfq")
        if data3 is not None and not data3.empty:
            print(f"[OK] 获取到 {len(data3)} 条数据")
            print(f"列名: {data3.columns.tolist()}")
            print("数据预览:")
            print(data3.head(3))
        else:
            print("[ERROR] 未获取到数据")
    except Exception as e:
        print(f"[ERROR] 方法3失败: {e}")

    print()

    # 测试另一个ETF代码
    test_symbol2 = "159380"  # A500ETF
    print(f"测试另一个ETF代码: {test_symbol2}")

    try:
        data4 = ak.fund_etf_hist_em(symbol=test_symbol2, period="daily", start_date=start_date, end_date=end_date, adjust="hfq")
        if data4 is not None and not data4.empty:
            print(f"[OK] 获取到 {len(data4)} 条数据")
        else:
            print("[ERROR] 未获取到数据")
    except Exception as e:
        print(f"[ERROR] 失败: {e}")

def test_standardize_data_format():
    """测试数据格式标准化"""
    print("\n=== 测试数据格式标准化 ===")

    try:
        # 获取示例数据
        data = ak.fund_etf_hist_em(symbol="159682", period="daily", start_date="20241101", end_date="20241201", adjust="hfq")

        if data is not None and not data.empty:
            print("原始数据列名:", data.columns.tolist())
            print("原始数据预览:")
            print(data.head(3))

            # 标准化数据格式
            standardized_data = pd.DataFrame()

            # 列名映射（需要根据实际的AKShare列名调整）
            column_mapping = {
                '日期': 'date',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'amount'
            }

            # 应用列名映射
            for old_name, new_name in column_mapping.items():
                if old_name in data.columns:
                    standardized_data[new_name] = data[old_name]

            print("\n标准化后列名:", standardized_data.columns.tolist())
            print("标准化后数据预览:")
            print(standardized_data.head(3))

        else:
            print("无法获取数据进行标准化测试")

    except Exception as e:
        print(f"标准化测试失败: {e}")

if __name__ == "__main__":
    test_akshare_etf_methods()
    test_standardize_data_format()