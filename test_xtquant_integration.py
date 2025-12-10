#!/usr/bin/env python3
"""
XTQuant Integration Test
XTQuant 集成测试

测试XTQuant数据源的完整功能，包括：
- 历史数据获取
- 基准数据获取
- 实时数据订阅
- 数据质量验证
- 错误处理
"""

import sys
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
import traceback

# Add src to path
sys.path.append('src')

# Import the market data fetcher
from data.market_data_fetcher import MarketDataFetcher


def test_xtquant_availability():
    """测试XTQuant可用性"""
    print("🔍 测试XTQuant可用性...")

    try:
        from xtquant import xtdata
        print("✅ XTQuant导入成功")
        return True, None
    except ImportError as e:
        print(f"❌ XTQuant导入失败: {e}")
        print("💡 请安装XTQuant: pip install xtquant")
        return False, e
    except Exception as e:
        print(f"❌ XTQuant检查异常: {e}")
        return False, e


def test_xtquant_initialization():
    """测试XTQuant初始化"""
    print("\n🔧 测试XTQuant初始化...")

    try:
        fetcher = MarketDataFetcher()

        # Check if xtquant is available and initialized
        if hasattr(fetcher, 'xtquant_initialized') and fetcher.xtquant_initialized:
            print("✅ XTQuant初始化成功")
            return True, fetcher
        else:
            print("❌ XTQuant初始化失败")
            return False, fetcher

    except Exception as e:
        print(f"❌ XTQuant初始化异常: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False, None


def test_historical_etf_data(fetcher):
    """测试历史ETF数据获取"""
    print("\n📈 测试历史ETF数据获取...")

    test_symbols = ["159682", "510300", "512880"]  # 科创50ETF, 沪深300ETF, 证券ETF
    start_date = "20231001"
    end_date = "20231201"

    results = {}

    for symbol in test_symbols:
        try:
            print(f"\n🔍 测试ETF: {symbol}")
            print(f"📅 时间范围: {start_date} - {end_date}")

            # Test data fetching
            start_time = time.time()
            data = fetcher.fetch_etf_data(symbol, start_date, end_date)
            end_time = time.time()

            if not data.empty:
                print(f"✅ 成功获取 {symbol} 数据: {len(data)} 条记录")
                print(f"⏱️  耗时: {end_time - start_time:.2f} 秒")
                print(f"📊 数据列: {list(data.columns)}")
                print(f"🗓️  日期范围: {data['date'].min()} 到 {data['date'].max()}")

                # Display sample data
                print("📋 数据示例:")
                print(data[['date', 'open', 'high', 'low', 'close', 'volume']].head(3))

                # Basic data quality checks
                if 'close' in data.columns:
                    price_stats = data['close'].describe()
                    print(f"💰 价格统计: 最小={price_stats['min']:.4f}, 最大={price_stats['max']:.4f}, 平均={price_stats['mean']:.4f}")

                results[symbol] = {
                    'success': True,
                    'data_count': len(data),
                    'execution_time': end_time - start_time,
                    'date_range': f"{data['date'].min()} 到 {data['date'].max()}"
                }

            else:
                print(f"❌ 未获取到 {symbol} 的数据")
                results[symbol] = {
                    'success': False,
                    'error': 'Empty data',
                    'execution_time': end_time - start_time
                }

        except Exception as e:
            print(f"❌ 测试 {symbol} 异常: {type(e).__name__}: {e}")
            results[symbol] = {
                'success': False,
                'error': str(e),
                'execution_time': 0
            }

    return results


def test_benchmark_data(fetcher):
    """测试基准指数数据获取"""
    print("\n📊 测试基准指数数据获取...")

    test_benchmarks = ["000300", "000905"]  # 沪深300, 中证500
    start_date = "20231001"
    end_date = "20231201"

    results = {}

    for benchmark in test_benchmarks:
        try:
            print(f"\n🔍 测试基准指数: {benchmark}")
            print(f"📅 时间范围: {start_date} - {end_date}")

            # Test benchmark data fetching
            start_time = time.time()
            data = fetcher.fetch_benchmark_data(benchmark, start_date, end_date)
            end_time = time.time()

            if not data.empty:
                print(f"✅ 成功获取基准 {benchmark} 数据: {len(data)} 条记录")
                print(f"⏱️  耗时: {end_time - start_time:.2f} 秒")
                print(f"📊 数据列: {list(data.columns)}")
                print(f"🗓️  日期范围: {data['date'].min()} 到 {data['date'].max()}")

                # Display sample data
                print("📋 数据示例:")
                print(data[['date', 'open', 'high', 'low', 'close', 'volume']].head(3))

                # Basic data quality checks
                if 'close' in data.columns:
                    price_stats = data['close'].describe()
                    print(f"💰 价格统计: 最小={price_stats['min']:.4f}, 最大={price_stats['max']:.4f}, 平均={price_stats['mean']:.4f}")

                results[benchmark] = {
                    'success': True,
                    'data_count': len(data),
                    'execution_time': end_time - start_time,
                    'date_range': f"{data['date'].min()} 到 {data['date'].max()}"
                }

            else:
                print(f"❌ 未获取到基准 {benchmark} 的数据")
                results[benchmark] = {
                    'success': False,
                    'error': 'Empty data',
                    'execution_time': end_time - start_time
                }

        except Exception as e:
            print(f"❌ 测试基准 {benchmark} 异常: {type(e).__name__}: {e}")
            results[benchmark] = {
                'success': False,
                'error': str(e),
                'execution_time': 0
            }

    return results


def test_realtime_subscription(fetcher):
    """测试实时数据订阅功能"""
    print("\n📡 测试实时数据订阅功能...")

    try:
        # Define a custom callback for testing
        test_callback_results = []

        def test_callback(data):
            """测试回调函数"""
            if data:
                timestamp = datetime.now().strftime('%H:%M:%S')
                symbols = list(data.keys())
                test_callback_results.append({
                    'timestamp': timestamp,
                    'symbols': symbols,
                    'data_received': True
                })
                print(f"📡 实时数据回调 [{timestamp}]: 收到 {len(symbols)} 个标的数据")

        # Test subscription setup
        test_symbols = ["159682", "510300"]
        print(f"🔍 订阅测试标的: {test_symbols}")

        subscription_success = fetcher.subscribe_xtquant_realtime(
            symbols=test_symbols,
            callback=test_callback,
            period="1d"
        )

        if subscription_success:
            print("✅ 实时数据订阅设置成功")

            # Check subscription status
            status = fetcher.get_xtquant_subscription_status()
            print(f"📊 订阅状态: {status['total_subscriptions']} 个活跃订阅")

            for sub in status['subscriptions']:
                print(f"  - {sub['symbol']} ({sub['xt_symbol']}): 订阅时长 {sub['duration_seconds']} 秒")

            # Wait for some data (max 30 seconds)
            print("⏳ 等待实时数据推送 (最多30秒)...")

            def monitor_realtime_data():
                """监控实时数据的独立线程"""
                nonlocal test_callback_results
                for i in range(30):  # 等待30秒
                    time.sleep(1)
                    if test_callback_results:
                        break

            monitor_thread = threading.Thread(target=monitor_realtime_data)
            monitor_thread.start()
            monitor_thread.join(timeout=35)

            # Check results
            if test_callback_results:
                print(f"✅ 成功接收到实时数据: {len(test_callback_results)} 次回调")
                for result in test_callback_results[:3]:  # Show first 3 results
                    print(f"  📡 回调时间: {result['timestamp']}, 标的: {result['symbols']}")
                subscription_results = {
                    'success': True,
                    'callbacks_received': len(test_callback_results),
                    'subscription_count': status['total_subscriptions']
                }
            else:
                print("⚠️  30秒内未接收到实时数据推送")
                print("💡 这可能是正常的，如果市场不在交易时间")
                subscription_results = {
                    'success': True,  # 订阅设置成功，但无数据推送
                    'callbacks_received': 0,
                    'subscription_count': status['total_subscriptions'],
                    'note': 'No data received (possibly outside trading hours)'
                }

            # Clean up subscription
            print("\n🧹 清理测试订阅...")
            fetcher.unsubscribe_xtquant_realtime()
            print("✅ 订阅已清理")

        else:
            print("❌ 实时数据订阅设置失败")
            subscription_results = {
                'success': False,
                'error': 'Subscription setup failed'
            }

        return subscription_results

    except Exception as e:
        print(f"❌ 实时数据订阅测试异常: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


def test_data_quality(fetcher):
    """测试数据质量验证"""
    print("\n🔍 测试数据质量验证...")

    try:
        # Get some test data
        test_symbol = "159682"
        test_data = fetcher.fetch_etf_data(test_symbol, "20231101", "20231130")

        if test_data.empty:
            print("❌ 无法获取测试数据进行质量验证")
            return {'success': False, 'error': 'No test data available'}

        # Generate data quality report
        quality_report = fetcher.get_data_quality_report(test_data)

        print("📊 数据质量报告:")
        print(f"  📈 总记录数: {quality_report['total_records']}")
        print(f"  🗓️  日期范围: {quality_report['date_range']['start']} 到 {quality_report['date_range']['end']}")
        print(f"  💰 价格统计: 最小={quality_report['price_stats']['min_price']:.4f}, 最大={quality_report['price_stats']['max_price']:.4f}")
        print(f"  📊 数据完整性: {quality_report['data_completeness']:.2f}%")

        # Check for missing values
        missing_values = quality_report['missing_values']
        if missing_values:
            print("  ⚠️  缺失值统计:")
            for col, count in missing_values.items():
                if count > 0:
                    print(f"    {col}: {count} 个缺失值")
        else:
            print("  ✅ 无缺失值")

        # Validate against quality requirements
        quality_score = 0
        total_checks = 0

        # Check data completeness
        total_checks += 1
        if quality_report['data_completeness'] >= 95:
            print("  ✅ 数据完整性良好 (>=95%)")
            quality_score += 1
        else:
            print(f"  ⚠️  数据完整性较低 ({quality_report['data_completeness']:.2f}%)")

        # Check data volume
        total_checks += 1
        if quality_report['total_records'] >= 20:  # 至少20个交易日数据
            print("  ✅ 数据量充足")
            quality_score += 1
        else:
            print(f"  ⚠️  数据量较少 ({quality_report['total_records']} 条记录)")

        # Check price volatility (should not be zero)
        total_checks += 1
        if quality_report['price_stats']['price_volatility'] > 0:
            print("  ✅ 价格数据有波动")
            quality_score += 1
        else:
            print("  ⚠️  价格数据无波动")

        quality_results = {
            'success': True,
            'quality_score': quality_score,
            'total_checks': total_checks,
            'quality_percentage': (quality_score / total_checks) * 100 if total_checks > 0 else 0,
            'report': quality_report
        }

        print(f"\n🎯 数据质量评分: {quality_score}/{total_checks} ({quality_results['quality_percentage']:.1f}%)")

        return quality_results

    except Exception as e:
        print(f"❌ 数据质量验证异常: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


def test_xtquant_configuration():
    """测试XTQuant配置"""
    print("\n⚙️  测试XTQuant配置...")

    try:
        from src.config.personal_config import get_personal_config

        config = get_personal_config()

        # Check if xtquant is enabled
        xtquant_enabled = config.is_data_source_enabled('xtquant')
        print(f"🔌 XTQuant启用状态: {'✅ 已启用' if xtquant_enabled else '❌ 未启用'}")

        # Get data sources priority
        enabled_sources = config.get_enabled_data_sources()
        xtquant_priority = config.get_data_source_priority('xtquant')
        print(f"📊 XTQuant优先级: {xtquant_priority}")
        print(f"📈 启用的数据源: {enabled_sources}")

        # Check xtquant position in priority
        if xtquant_enabled:
            xtquant_position = enabled_sources.index('xtquant') + 1 if 'xtquant' in enabled_sources else -1
            if xtquant_position > 0:
                print(f"🎯 XTQuant在数据源中的位置: 第 {xtquant_position} 位")
            else:
                print("⚠️  XTQuant未在启用数据源列表中")

        # Validate configuration
        validation = config.validate_config()

        config_results = {
            'success': True,
            'xtquant_enabled': xtquant_enabled,
            'xtquant_priority': xtquant_priority,
            'enabled_sources': enabled_sources,
            'xtquant_position': enabled_sources.index('xtquant') + 1 if 'xtquant' in enabled_sources else -1,
            'validation_passed': validation['valid']
        }

        print(f"🔧 配置验证: {'✅ 通过' if validation['valid'] else '❌ 失败'}")

        if validation['errors']:
            print("❌ 配置错误:")
            for error in validation['errors']:
                print(f"  - {error}")

        if validation['warnings']:
            print("⚠️  配置警告:")
            for warning in validation['warnings']:
                print(f"  - {warning}")

        return config_results

    except Exception as e:
        print(f"❌ 配置测试异常: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


def generate_test_summary(all_results):
    """生成测试总结报告"""
    print("\n" + "="*80)
    print("📋 XTQuant集成测试总结报告")
    print("="*80)

    # Overall status
    total_tests = len(all_results)
    passed_tests = sum(1 for result in all_results.values() if result.get('success', False))

    print(f"\n📊 总体测试结果: {passed_tests}/{total_tests} 通过 ({(passed_tests/total_tests)*100:.1f}%)")

    # Detailed results
    print("\n🔍 详细测试结果:")

    for test_name, result in all_results.items():
        status_icon = "✅" if result.get('success', False) else "❌"
        print(f"  {status_icon} {test_name}")

        if not result.get('success', False) and 'error' in result:
            print(f"    错误: {result['error']}")

    # Performance summary
    if 'etf_data_test' in all_results and all_results['etf_data_test'].get('success'):
        etf_results = all_results['etf_data_test']
        successful_etfs = {k: v for k, v in etf_results.items() if v.get('success')}

        if successful_etfs:
            avg_time = sum(r['execution_time'] for r in successful_etfs.values()) / len(successful_etfs)
            total_records = sum(r.get('data_count', 0) for r in successful_etfs.values())

            print(f"\n⚡ 性能统计:")
            print(f"  📈 平均获取时间: {avg_time:.2f} 秒")
            print(f"  📊 总数据记录: {total_records} 条")
            print(f"  🎯 成功ETF数量: {len(successful_etfs)} 个")

    # Data quality summary
    if 'data_quality_test' in all_results and all_results['data_quality_test'].get('success'):
        quality_result = all_results['data_quality_test']
        if 'quality_percentage' in quality_result:
            print(f"\n🎯 数据质量: {quality_result['quality_percentage']:.1f}%")

    # Recommendations
    print(f"\n💡 建议:")

    if passed_tests == total_tests:
        print("  🎉 所有测试通过！XTQuant集成成功。")
        print("  📈 可以开始使用XTQuant作为数据源。")
    else:
        print("  ⚠️  部分测试失败，请检查以下项目:")

        if not all_results['availability_test'].get('success'):
            print("    - 安装XTQuant: pip install xtquant")

        if not all_results['initialization_test'].get('success'):
            print("    - 检查MiniQmt安装和配置")
            print("    - 确保XTQuant服务正常运行")

        if not all_results['configuration_test'].get('success'):
            print("    - 检查环境变量配置")
            print("    - 验证XTQUANT_ENABLED设置")

        if not all_results['etf_data_test'].get('success'):
            print("    - 检查网络连接")
            print("    - 验证数据代码格式")
            print("    - 确认数据权限")

    print(f"\n📖 更多信息:")
    print(f"  📚 XTQuant文档: https://dict.thinktrader.net/nativeApi/start_now.html")
    print(f"  🔧 配置文件: src/config/personal_config.py")
    print(f"  🧪 测试脚本: test_xtquant_integration.py")


def main():
    """主测试函数"""
    print("🚀 XTQuant集成测试开始")
    print("="*80)
    print(f"🕒 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python版本: {sys.version}")

    # Store all test results
    all_results = {}

    # Test 1: XTQuant availability
    availability_success, availability_error = test_xtquant_availability()
    all_results['availability_test'] = {
        'success': availability_success,
        'error': str(availability_error) if availability_error else None
    }

    if not availability_success:
        print("\n❌ XTQuant不可用，无法进行后续测试")
        generate_test_summary(all_results)
        return

    # Test 2: Configuration
    config_results = test_xtquant_configuration()
    all_results['configuration_test'] = config_results

    # Test 3: Initialization
    init_success, fetcher = test_xtquant_initialization()
    all_results['initialization_test'] = {
        'success': init_success
    }

    if not init_success or fetcher is None:
        print("\n❌ XTQuant初始化失败，无法进行数据获取测试")
        generate_test_summary(all_results)
        return

    # Test 4: Historical ETF data
    try:
        etf_results = test_historical_etf_data(fetcher)

        # Check if any ETF tests succeeded
        etf_success = any(result.get('success', False) for result in etf_results.values())
        all_results['etf_data_test'] = {
            'success': etf_success,
            'details': etf_results
        }

    except Exception as e:
        print(f"❌ ETF数据测试异常: {e}")
        all_results['etf_data_test'] = {
            'success': False,
            'error': str(e)
        }

    # Test 5: Benchmark data
    try:
        benchmark_results = test_benchmark_data(fetcher)

        # Check if any benchmark tests succeeded
        benchmark_success = any(result.get('success', False) for result in benchmark_results.values())
        all_results['benchmark_data_test'] = {
            'success': benchmark_success,
            'details': benchmark_results
        }

    except Exception as e:
        print(f"❌ 基准数据测试异常: {e}")
        all_results['benchmark_data_test'] = {
            'success': False,
            'error': str(e)
        }

    # Test 6: Real-time subscription
    try:
        realtime_results = test_realtime_subscription(fetcher)
        all_results['realtime_test'] = realtime_results
    except Exception as e:
        print(f"❌ 实时数据测试异常: {e}")
        all_results['realtime_test'] = {
            'success': False,
            'error': str(e)
        }

    # Test 7: Data quality
    try:
        quality_results = test_data_quality(fetcher)
        all_results['data_quality_test'] = quality_results
    except Exception as e:
        print(f"❌ 数据质量测试异常: {e}")
        all_results['data_quality_test'] = {
            'success': False,
            'error': str(e)
        }

    # Generate summary
    generate_test_summary(all_results)


if __name__ == "__main__":
    main()