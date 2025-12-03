#!/usr/bin/env python3
"""
数据源连接调试脚本
"""

import sys
import logging
from pathlib import Path
sys.path.append('src')

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_config():
    """测试配置系统"""
    print("=== 测试配置系统 ===")

    try:
        from src.config.personal_config import get_personal_config
        config = get_personal_config()

        print("[OK] 配置系统加载成功")
        print(f"配置摘要:\n{config.get_config_summary()}")

        # 验证配置
        validation = config.validate_config()
        print(f"\n配置验证: {'通过' if validation['valid'] else '失败'}")

        if validation['errors']:
            print(f"错误: {validation['errors']}")
        if validation['warnings']:
            print(f"警告: {validation['warnings']}")

        return config

    except Exception as e:
        print(f"[ERROR] 配置系统失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_jqdata_connection():
    """测试jqdatasdk连接"""
    print("\n=== 测试jqdatasdk连接 ===")

    try:
        import jqdatasdk
        print("✅ jqdatasdk模块导入成功")

        # 测试认证
        from src.config.personal_config import get_personal_config
        config = get_personal_config()

        if config.is_data_source_enabled('jqdata'):
            credentials = config.get_api_credentials('jqdata')
            username = credentials.get('username')
            password = credentials.get('password')

            if username and password:
                print(f"尝试认证用户: {username}")
                jqdatasdk.auth(username, password)

                # 测试获取数据
                test_symbol = "000001.XSHE"  # 平安银行
                data = jqdatasdk.get_price(test_symbol, count=5, frequency='daily')

                if data is not None and not data.empty:
                    print(f"✅ jqdatasdk连接成功，获取到 {len(data)} 条数据")
                    print("数据预览:")
                    print(data.head())
                    return True
                else:
                    print("❌ jqdatasdk获取数据失败")
                    return False
            else:
                print("❌ jqdatasdk凭证不完整")
                return False
        else:
            print("❌ jqdatasdk未启用")
            return False

    except ImportError:
        print("❌ jqdatasdk模块未安装")
        return False
    except Exception as e:
        print(f"❌ jqdatasdk连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_akshare_connection():
    """测试AKShare连接"""
    print("\n=== 测试AKShare连接 ===")

    try:
        import akshare as ak
        print("✅ AKShare模块导入成功")

        # 测试获取ETF数据
        test_symbol = "159682"  # 科创50ETF

        # 尝试基金历史数据
        try:
            data = ak.fund_etf_hist_em(symbol=test_symbol, period="daily", start_date="20240101", end_date="20241201", adjust="hfq")
            if data is not None and not data.empty:
                print(f"✅ AKShare连接成功，获取到 {len(data)} 条数据")
                print("数据预览:")
                print(data.head())
                return True
            else:
                print("❌ AKShare获取数据为空")
                return False
        except Exception as e1:
            print(f"AKShare fund_etf_hist_em失败: {e1}")

            # 尝试股票数据
            try:
                data = ak.stock_zh_a_hist(symbol=test_symbol, period="daily", start_date="20240101", end_date="20241201", adjust="hfq")
                if data is not None and not data.empty:
                    print(f"✅ AKShare股票数据连接成功，获取到 {len(data)} 条数据")
                    print("数据预览:")
                    print(data.head())
                    return True
                else:
                    print("❌ AKShare获取股票数据为空")
                    return False
            except Exception as e2:
                print(f"AKShare股票数据失败: {e2}")
                return False

    except ImportError:
        print("❌ AKShare模块未安装")
        return False
    except Exception as e:
        print(f"❌ AKShare连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_market_data_fetcher():
    """测试MarketDataFetcher"""
    print("\n=== 测试MarketDataFetcher ===")

    try:
        from src.data.market_data_fetcher import MarketDataFetcher

        fetcher = MarketDataFetcher()
        print("✅ MarketDataFetcher初始化成功")

        # 测试ETF数据获取
        test_symbol = "159682"
        print(f"\n测试获取ETF数据: {test_symbol}")

        data = fetcher.fetch_etf_data(test_symbol, start_date="20241101", end_date="20241201")

        if data is not None and not data.empty:
            print(f"✅ 成功获取 {len(data)} 条数据")
            print("数据预览:")
            print(data.head())

            # 数据质量报告
            report = fetcher.get_data_quality_report(data)
            print(f"\n数据质量报告:")
            for key, value in report.items():
                if key != 'missing_values':
                    print(f"  {key}: {value}")

            return True
        else:
            print("❌ 未获取到数据")
            return False

    except Exception as e:
        print(f"❌ MarketDataFetcher测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("开始数据源连接调试...")

    # 测试配置
    config = test_config()
    if not config:
        print("\n❌ 配置系统失败，停止调试")
        return

    # 测试各个数据源
    results = {}

    # 测试jqdatasdk
    results['jqdata'] = test_jqdata_connection()

    # 测试AKShare
    results['akshare'] = test_akshare_connection()

    # 测试MarketDataFetcher
    results['market_data_fetcher'] = test_market_data_fetcher()

    # 总结
    print("\n=== 调试结果总结 ===")
    for source, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{source}: {status}")

    success_count = sum(results.values())
    total_count = len(results)
    print(f"\n总体结果: {success_count}/{total_count} 成功")

    if success_count == 0:
        print("🚨 所有数据源都失败了，请检查网络连接和API配置")
    elif success_count < total_count:
        print("⚠️ 部分数据源失败，建议检查失败的数据源配置")
    else:
        print("🎉 所有数据源都正常工作")

if __name__ == "__main__":
    main()