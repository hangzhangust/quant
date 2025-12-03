#!/usr/bin/env python3
"""
数据源连接调试脚本 - 简化版
"""

import sys
import logging
from pathlib import Path
sys.path.append('src')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
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
        print(f"启用的数据源: {config.get_enabled_data_sources()}")
        print(f"jqdata启用状态: {config.is_data_source_enabled('jqdata')}")
        print(f"akshare启用状态: {config.is_data_source_enabled('akshare')}")

        # 验证配置
        validation = config.validate_config()
        print(f"配置验证: {'通过' if validation['valid'] else '失败'}")

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
        print("[OK] jqdatasdk模块导入成功")

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
                    print(f"[OK] jqdatasdk连接成功，获取到 {len(data)} 条数据")
                    return True
                else:
                    print("[ERROR] jqdatasdk获取数据失败")
                    return False
            else:
                print("[ERROR] jqdatasdk凭证不完整")
                return False
        else:
            print("[INFO] jqdatasdk未启用")
            return False

    except ImportError:
        print("[ERROR] jqdatasdk模块未安装")
        return False
    except Exception as e:
        print(f"[ERROR] jqdatasdk连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_akshare_connection():
    """测试AKShare连接"""
    print("\n=== 测试AKShare连接 ===")

    try:
        import akshare as ak
        print("[OK] AKShare模块导入成功")

        # 测试获取ETF数据
        test_symbol = "159682"  # 科创50ETF

        # 尝试基金历史数据
        try:
            data = ak.fund_etf_hist_em(symbol=test_symbol, period="daily", start_date="20241101", end_date="20241201", adjust="hfq")
            if data is not None and not data.empty:
                print(f"[OK] AKShare连接成功，获取到 {len(data)} 条数据")
                print("数据列名:", data.columns.tolist())
                return True
            else:
                print("[ERROR] AKShare获取数据为空")
                return False
        except Exception as e1:
            print(f"AKShare fund_etf_hist_em失败: {e1}")
            return False

    except ImportError:
        print("[ERROR] AKShare模块未安装")
        return False
    except Exception as e:
        print(f"[ERROR] AKShare连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("开始数据源连接调试...")

    # 测试配置
    config = test_config()
    if not config:
        print("\n配置系统失败，停止调试")
        return

    # 测试各个数据源
    results = {}

    # 测试jqdatasdk
    results['jqdata'] = test_jqdata_connection()

    # 测试AKShare
    results['akshare'] = test_akshare_connection()

    # 总结
    print("\n=== 调试结果总结 ===")
    for source, success in results.items():
        status = "[OK]" if success else "[ERROR]"
        print(f"{source}: {status}")

    success_count = sum(results.values())
    total_count = len(results)
    print(f"\n总体结果: {success_count}/{total_count} 成功")

if __name__ == "__main__":
    main()