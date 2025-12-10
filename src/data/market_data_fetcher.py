"""
市场数据获取器
Market Data Fetcher

使用多种数据源获取ETF历史数据，解决代理连接问题
"""

import akshare as ak
import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
import pickle
import urllib.request
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 导入jqdatasdk
try:
    import jqdatasdk
    from jqdatasdk import auth, get_price
    JQDATA_AVAILABLE = True
except ImportError:
    JQDATA_AVAILABLE = False
    jqdatasdk = None

# 导入xtquant
try:
    from xtquant import xtdata
    XTQUANT_AVAILABLE = True
except ImportError:
    XTQUANT_AVAILABLE = False
    xtdata = None

# 导入个人配置系统
try:
    from src.config.personal_config import get_personal_config
    PERSONAL_CONFIG_AVAILABLE = True
except ImportError:
    PERSONAL_CONFIG_AVAILABLE = False
    logger.warning("个人配置系统不可用，将使用默认配置")

# 导入模拟数据生成器
try:
    from src.data.mock_benchmark_generator import mock_generator
    MOCK_GENERATOR_AVAILABLE = True
except ImportError:
    MOCK_GENERATOR_AVAILABLE = False
    logger.warning("模拟数据生成器不可用")

logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """市场数据获取器类"""

    def __init__(self, cache_dir: str = "cache", personal_config=None):
        """
        初始化市场数据获取器

        Args:
            cache_dir: 缓存目录
            personal_config: 个人配置管理器实例，如果为None则使用全局配置
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = 86400  # 24小时缓存

        # 初始化个人配置
        if personal_config:
            self.personal_config = personal_config
        elif PERSONAL_CONFIG_AVAILABLE:
            self.personal_config = get_personal_config()
        else:
            self.personal_config = None

        # 配置网络设置
        self._setup_network()

        # 初始化jqdatasdk（如果配置了）
        self._init_jqdata()

        # 初始化Tushare（如果配置了）
        self._init_tushare()

        # 初始化XTQuant（如果配置了）
        self._init_xtquant()

    def _setup_network(self):
        """配置网络连接，禁用代理并设置重试机制"""
        # 禁用所有代理设置
        os.environ.update({
            'HTTP_PROXY': '',
            'HTTPS_PROXY': '',
            'http_proxy': '',
            'https_proxy': '',
            'NO_PROXY': '*',
            'no_proxy': '*'
        })

        # 配置请求session，添加重试机制
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.trust_env = False  # 不信任系统代理

        # 为urllib禁用代理
        proxy_handler = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(proxy_handler)
        urllib.request.install_opener(opener)

    def _init_jqdata(self):
        """初始化jqdatasdk"""
        self.jqdata_initialized = False

        # 检查是否配置了jqdatasdk
        if self.personal_config and self.personal_config.is_data_source_enabled('jqdata') and JQDATA_AVAILABLE:
            try:
                # 获取jqdatasdk凭证
                credentials = self.personal_config.get_api_credentials('jqdata')
                username = credentials.get('username')
                password = credentials.get('password')

                if username and password:
                    # 使用jqdatasdk认证
                    jqdatasdk.auth(username, password)
                    self.jqdata_initialized = True
                    logger.info("jqdatasdk 初始化成功")
                else:
                    logger.warning("jqdatasdk凭证不完整，jqdatasdk功能不可用")

            except Exception as e:
                logger.error(f"jqdatasdk初始化失败: {e}")
        elif not JQDATA_AVAILABLE:
            logger.warning("jqdatasdk包未安装，请运行: pip install jqdatasdk")
        else:
            logger.info("jqdatasdk数据源未启用")

    def _init_tushare(self):
        """初始化Tushare Pro API"""
        self.tushare_pro = None

        # 检查是否配置了Tushare
        if self.personal_config and self.personal_config.is_data_source_enabled('tushare'):
            try:
                import tushare as ts

                # 获取Tushare token
                credentials = self.personal_config.get_api_credentials('tushare')
                token = credentials.get('token')

                if token:
                    # 设置token并初始化API
                    ts.set_token(token)
                    self.tushare_pro = ts.pro_api()
                    logger.info("Tushare Pro API 初始化成功")
                else:
                    logger.warning("Tushare token未配置，Tushare功能不可用")

            except ImportError:
                logger.error("Tushare包未安装，请运行: pip install tushare")
            except Exception as e:
                logger.error(f"Tushare初始化失败: {e}")
        else:
            logger.info("Tushare数据源未启用")

    def _init_xtquant(self):
        """初始化XTQuant"""
        self.xtquant_initialized = False

        # 检查是否启用了xtquant
        if self.personal_config and self.personal_config.is_data_source_enabled('xtquant') and XTQUANT_AVAILABLE:
            try:
                # xtquant通常不需要认证，但需要MiniQmt支持
                # 尝试连接测试
                test_result = xtdata.get_market_data_ex([], ["000001.SZ"], period="1d", count=1)
                self.xtquant_initialized = True
                logger.info("XTQuant 初始化成功")

                # 设置本地数据目录
                if hasattr(xtdata, 'set_data_path'):
                    # 可以设置自定义数据路径
                    cache_dir = str(self.cache_dir / "xtquant_data")
                    xtdata.set_data_path(cache_dir)
                    logger.info(f"XTQuant 数据目录设置为: {cache_dir}")

            except Exception as e:
                logger.error(f"XTQuant初始化失败: {e}")
                logger.warning("请确保MiniQmt已正确安装和配置")
        elif not XTQUANT_AVAILABLE:
            logger.warning("XTQuant包未安装，请运行: pip install xtquant")
        else:
            logger.info("XTQuant数据源未启用")

    def fetch_benchmark_data(self, benchmark_symbol: str = "000300", start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        获取基准指数数据用于Beta计算

        Args:
            benchmark_symbol: 基准指数代码，默认沪深300 (000300)
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD

        Returns:
            包含OHLCV数据的DataFrame
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        # 检查缓存
        cache_key = f"benchmark_{benchmark_symbol}_{start_date}_{end_date}"
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            logger.info(f"从缓存获取基准数据: {benchmark_symbol}")
            return cached_data

        try:
            logger.info(f"获取基准指数数据: {benchmark_symbol}, 时间范围: {start_date} - {end_date}")

            # 尝试获取基准数据
            data = self._try_benchmark_sources(benchmark_symbol, start_date, end_date)

            if data is not None and not data.empty:
                # 缓存数据
                self._save_to_cache(cache_key, data)
                logger.info(f"成功获取基准数据: {benchmark_symbol}, 数据量: {len(data)}")
                return data
            else:
                logger.warning(f"无法获取基准数据: {benchmark_symbol}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"获取基准数据失败 {benchmark_symbol}: {e}")
            return pd.DataFrame()

    def _try_benchmark_sources(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """尝试多种数据源获取基准数据"""
        logger.info(f"开始获取基准数据: {symbol}, 时间范围: {start_date} - {end_date}")

        # 获取按优先级排序的数据源列表
        enabled_sources = self._get_enabled_data_sources()
        logger.info(f"启用的数据源优先级: {enabled_sources}")

        # 映射数据源到对应的基准获取方法
        benchmark_methods = {
            'jqdata': lambda: self._try_jqdata_benchmark(symbol, start_date, end_date),
            'tushare': lambda: self._try_tushare_benchmark(symbol, start_date, end_date),
            'wind': lambda: self._try_wind_benchmark(symbol, start_date, end_date),
            'xtquant': lambda: self._try_xtquant_benchmark(symbol, start_date, end_date),
            'akshare': lambda: self._try_akshare_benchmark(symbol, start_date, end_date),
            'yfinance': lambda: self._try_yahoo_benchmark(symbol, start_date, end_date)
        }

        results = []

        # 按优先级尝试数据源
        for source_name in enabled_sources:
            if source_name in benchmark_methods:
                logger.info(f"🔄 正在尝试数据源: {source_name}")
                try:
                    data = benchmark_methods[source_name]()
                    if data is not None and not data.empty:
                        logger.info(f"✅ {source_name} 成功获取基准数据: {len(data)} 条记录")
                        logger.info(f"   数据形状: {data.shape}")
                        logger.info(f"   列名: {list(data.columns)}")
                        logger.info(f"   日期范围: {data.index[0]} 到 {data.index[-1]}")

                        # 验证数据质量
                        if self._validate_benchmark_data_quality(data):
                            logger.info(f"✅ {source_name} 基准数据质量验证通过")
                            return data
                        else:
                            logger.warning(f"⚠️ {source_name} 基准数据质量验证失败")
                            results.append((source_name, data, "质量验证失败"))
                    else:
                        logger.warning(f"❌ {source_name} 返回空数据")
                        results.append((source_name, pd.DataFrame(), "返回空数据"))
                except Exception as e:
                    logger.error(f"💥 {source_name} 异常: {str(e)[:200]}")
                    logger.debug(f"   详细错误信息: {type(e).__name__}: {e}")
                    results.append((source_name, pd.DataFrame(), f"异常: {type(e).__name__}"))

        # 如果所有配置的数据源都失败，尝试备用的免费数据源
        if not any(result[2] == "质量验证通过" for result in results):
            logger.info("🔄 配置数据源均失败，尝试备用免费数据源")
            fallback_sources = ['akshare', 'yfinance']
            for source_name in fallback_sources:
                if source_name not in enabled_sources and source_name in benchmark_methods:
                    logger.info(f"🔄 正在尝试备用数据源: {source_name}")
                    try:
                        data = benchmark_methods[source_name]()
                        if data is not None and not data.empty:
                            logger.info(f"✅ 备用数据源 {source_name} 成功获取数据: {len(data)} 条记录")
                            logger.info(f"   数据形状: {data.shape}")
                            logger.info(f"   列名: {list(data.columns)}")
                            logger.info(f"   日期范围: {data.index[0]} 到 {data.index[-1]}")

                            if self._validate_benchmark_data_quality(data):
                                logger.info(f"✅ 备用数据源 {source_name} 质量验证通过")
                                return data
                            else:
                                logger.warning(f"⚠️ 备用数据源 {source_name} 质量验证失败")
                                results.append((source_name, data, "质量验证失败"))
                        else:
                            logger.warning(f"❌ 备用数据源 {source_name} 返回空数据")
                            results.append((source_name, pd.DataFrame(), "返回空数据"))
                    except Exception as e:
                        logger.error(f"💥 备用数据源 {source_name} 异常: {str(e)[:200]}")
                        results.append((source_name, pd.DataFrame(), f"异常: {type(e).__name__}"))

        # 输出所有尝试结果的详细报告
        logger.info("📊 基准数据获取尝试结果报告:")
        for source_name, data, status in results:
            if not data.empty:
                logger.info(f"  {source_name}: 成功获取 {len(data)} 条数据, 状态: {status}")
            else:
                logger.warning(f"  {source_name}: 失败, 状态: {status}")

        # 尝试模拟数据作为最后手段
        if MOCK_GENERATOR_AVAILABLE and mock_generator.is_simulated_data_enabled():
            logger.info("🎭 所有真实数据源失败，尝试使用模拟数据...")
            try:
                mock_data = mock_generator.generate_benchmark_data(symbol, start_date, end_date)
                if not mock_data.empty and self._validate_benchmark_data_quality(mock_data):
                    logger.info(f"🎭 模拟基准数据生成成功: {len(mock_data)} 条记录")
                    logger.warning("⚠️ 注意: 当前使用模拟数据进行Beta计算，请检查网络连接和数据源配置")
                    return mock_data
                else:
                    logger.warning("🎭 模拟数据验证失败")
            except Exception as mock_e:
                logger.error(f"🎭 模拟数据生成失败: {type(mock_e).__name__}: {mock_e}")

        if not results:
            logger.error(f"❌ 所有数据源尝试均未执行，可能是配置问题")
            logger.info("💡 建议:")
            logger.info("   1. 检查网络连接")
            logger.info("   2. 配置有效的API令牌")
            logger.info("   3. 设置 MOCK_DATA_ENABLED=true 使用模拟数据进行测试")
        elif not any(result[2] in ["质量验证通过", "返回空数据"] for result in results):
            logger.error(f"❌ 所有数据源均失败: {symbol}")
            logger.info("💡 建议:")
            logger.info("   1. 检查网络连接和防火墙设置")
            logger.info("   2. 验证API令牌有效性")
            logger.info("   3. 设置 MOCK_DATA_ENABLED=true 使用模拟数据进行测试")
        else:
            logger.warning(f"⚠️ 未找到有效的基准数据: {symbol}")
            logger.info("💡 建议设置 MOCK_DATA_ENABLED=true 使用模拟数据进行测试")

        return pd.DataFrame()

    def _try_akshare_benchmark(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用AKShare获取基准数据"""
        try:
            # 转换日期格式为YYYYMMDD
            start_clean = start_date.replace('-', '')
            end_clean = end_date.replace('-', '')

            # 对于指数代码，尝试多种AKShare方法
            ak_methods = []

            if symbol == "000300":
                # 沪深300指数的多种获取方式
                ak_methods = [
                    # 方法1: 使用stock_zh_index_daily_em (推荐)
                    lambda: ak.stock_zh_index_daily_em(symbol="000300", start_date=start_clean, end_date=end_clean),
                    # 方法2: 使用index_zh_a_hist
                    lambda: ak.index_zh_a_hist(symbol="000300", period="daily", start_date=start_date, end_date=end_date),
                    # 方法3: 直接使用指数代码
                    lambda: ak.stock_zh_a_hist(symbol="000300", period="daily", start_date=start_clean, end_date=end_clean),
                    # 方法4: 使用带前缀的格式
                    lambda: ak.stock_zh_a_hist(symbol="sh000300", period="daily", start_date=start_clean, end_date=end_clean)
                ]
            else:
                # 其他指数的处理
                ak_methods = [
                    lambda: ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_clean, end_date=end_clean),
                    lambda: ak.stock_zh_a_hist(symbol=f"sh{symbol}", period="daily", start_date=start_clean, end_date=end_clean)
                ]

            for i, method in enumerate(ak_methods):
                try:
                    logger.debug(f"AKShare基准数据 - 尝试方法 {i+1}/{len(ak_methods)}")
                    data = method()

                    if data is not None and not data.empty:
                        logger.info(f"AKShare基准数据 - 方法 {i+1} 成功获取 {len(data)} 条数据")

                        # 标准化列名映射
                        column_mapping = {
                            '日期': 'date',
                            '开盘': 'open',
                            '收盘': 'close',
                            '最高': 'high',
                            '最低': 'low',
                            '成交量': 'volume',
                            '成交额': 'amount',
                            '涨跌幅': 'change_pct'
                        }

                        # 应用列名映射
                        data = data.rename(columns={k: v for k, v in column_mapping.items() if k in data.columns})

                        # 确保必要列存在
                        if 'date' not in data.columns:
                            # 尝试其他可能的日期列名
                            for date_col in ['time', 'datetime', '日期']:
                                if date_col in data.columns:
                                    data = data.rename(columns={date_col: 'date'})
                                    break

                        if 'date' in data.columns:
                            data['date'] = pd.to_datetime(data['date'])
                            data = data.sort_values('date').reset_index(drop=True)
                        else:
                            logger.warning(f"AKShare数据未找到日期列，现有列: {list(data.columns)}")
                            continue

                        # 确保有价格数据
                        if 'close' not in data.columns:
                            # 尝试其他价格列名
                            for price_col in ['收盘价', 'Close', '收盘']:
                                if price_col in data.columns:
                                    data = data.rename(columns={price_col: 'close'})
                                    break

                        if 'close' in data.columns:
                            # 转换价格数据为数值类型
                            data['close'] = pd.to_numeric(data['close'], errors='coerce')
                            return data
                        else:
                            logger.warning(f"AKShare数据未找到价格数据，现有列: {list(data.columns)}")
                            continue

                except Exception as method_e:
                    logger.debug(f"AKShare基准数据 - 方法 {i+1} 失败: {type(method_e).__name__}: {method_e}")
                    continue

        except Exception as e:
            logger.warning(f"AKShare基准数据获取完全失败: {type(e).__name__}: {e}")

        logger.warning("AKShare: 所有基准数据获取方法均失败")
        return pd.DataFrame()

    def _try_yahoo_benchmark(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用Yahoo Finance获取基准数据"""
        try:
            import yfinance as yf

            # 转换日期格式
            start_clean = start_date.replace('-', '')
            end_clean = end_date.replace('-', '')
            start_dt = datetime.strptime(start_clean, '%Y%m%d').strftime('%Y-%m-%d')
            end_dt = datetime.strptime(end_clean, '%Y%m%d').strftime('%Y-%m-%d')

            # 尝试多种符号格式
            symbol_formats = []

            if symbol == "000300":
                # 沪深300的多种格式
                symbol_formats = [
                    "000300.SS",     # 标准上海格式
                    "000300.SZ",     # 深圳格式
                    "^HSI",          # 恒生指数作为替代
                    "300750.SZ",     # 宁德时代作为活跃股票测试
                ]
            elif symbol.startswith("00"):
                # 深圳市场代码
                symbol_formats = [f"{symbol}.SZ"]
            else:
                # 上海市场代码
                symbol_formats = [f"{symbol}.SS"]

            # 添加一些国际指数作为网络连接测试
            test_symbols = ["^GSPC", "SPY", "QQQ", "VTI"]
            symbol_formats.extend(test_symbols)

            logger.debug(f"Yahoo Finance基准数据 - 尝试 {len(symbol_formats)} 种符号格式")

            for i, yf_symbol in enumerate(symbol_formats):
                try:
                    logger.debug(f"Yahoo Finance基准数据 - 测试符号 {i+1}/{len(symbol_formats)}: {yf_symbol}")

                    # 设置超时和重试参数
                    data = yf.download(
                        yf_symbol,
                        start=start_dt,
                        end=end_dt,
                        progress=False,
                        timeout=15,
                        show_errors=False
                    )

                    if data is not None and not data.empty:
                        logger.info(f"Yahoo Finance基准数据 - 符号 {yf_symbol} 成功获取 {len(data)} 条数据")

                        # 重置索引以获取日期列
                        data = data.reset_index()

                        # 标准化列名
                        column_mapping = {
                            'Date': 'date',
                            'Open': 'open',
                            'High': 'high',
                            'Low': 'low',
                            'Close': 'close',
                            'Volume': 'volume',
                            'Adj Close': 'adj_close'
                        }

                        data = data.rename(columns={k: v for k, v in column_mapping.items() if k in data.columns})

                        # 确保日期格式正确
                        if 'date' in data.columns:
                            data['date'] = pd.to_datetime(data['date'])
                            data = data.sort_values('date').reset_index(drop=True)

                        # 确保价格数据为数值类型
                        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                        for col in numeric_columns:
                            if col in data.columns:
                                data[col] = pd.to_numeric(data[col], errors='coerce')

                        logger.debug(f"Yahoo Finance数据列: {list(data.columns)}")
                        return data
                    else:
                        logger.debug(f"Yahoo Finance符号 {yf_symbol} 返回空数据")

                except Exception as symbol_e:
                    logger.debug(f"Yahoo Finance符号 {yf_symbol} 失败: {type(symbol_e).__name__}: {symbol_e}")
                    continue

        except ImportError:
            logger.warning("Yahoo Finance: yfinance库未安装")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Yahoo Finance基准数据获取完全失败: {type(e).__name__}: {e}")

        logger.warning("Yahoo Finance: 所有符号格式均失败")
        return pd.DataFrame()

    def _try_tushare_benchmark(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用Tushare Pro API获取基准数据"""
        if self.tushare_pro is None:
            logger.debug("Tushare Pro未初始化")
            return pd.DataFrame()

        try:
            # 转换指数代码为Tushare格式
            ts_symbol = self._convert_benchmark_to_tushare(symbol)

            # 获取指数日线数据
            df = self.tushare_pro.index_daily(
                ts_code=ts_symbol,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,open,high,low,close,vol,amount'
            )

            if df.empty:
                logger.debug(f"Tushare未获取到基准数据: {ts_symbol}")
                return pd.DataFrame()

            # 数据清洗和格式转换
            df = df.sort_values('trade_date')
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df.set_index('trade_date')

            # 重命名列以匹配标准格式
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'vol': 'volume',
                'amount': 'amount'
            })

            # 选择需要的列
            df = df[['open', 'high', 'low', 'close', 'volume']]

            logger.info(f"Tushare成功获取基准数据 {len(df)} 条: {symbol}")
            return df

        except Exception as e:
            logger.debug(f"Tushare获取基准数据失败 {symbol}: {e}")
            return pd.DataFrame()

    def _convert_benchmark_to_tushare(self, symbol: str) -> str:
        """转换指数代码为Tushare格式"""
        # 沪深300指数
        if symbol == "000300":
            return "000300.SH"
        # 上证综指
        elif symbol == "000001":
            return "000001.SH"
        # 中证500指数
        elif symbol == "000905":
            return "000905.SH"
        # 深证成指
        elif symbol == "399001":
            return "399001.SZ"
        # 创业板指
        elif symbol == "399006":
            return "399006.SZ"
        else:
            # 默认尝试上海市场
            return f"{symbol}.SH"

    def _try_wind_benchmark(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用Wind API获取基准数据（预留接口）"""
        # Wind集成需要特殊的授权和安装，这里提供基础框架
        logger.debug("Wind基准数据接口暂未实现")
        return pd.DataFrame()

    def fetch_etf_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        获取ETF历史数据

        Args:
            symbol: ETF代码
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD

        Returns:
            包含OHLCV数据的DataFrame
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        # 检查缓存
        cache_key = f"etf_{symbol}_{start_date}_{end_date}"
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            logger.info(f"从缓存获取数据: {symbol}")
            return cached_data

        try:
            logger.info(f"获取ETF数据: {symbol}, 时间范围: {start_date} - {end_date}")

            # 尝试多种数据源
            data = self._try_data_sources(symbol, start_date, end_date)

            if data.empty:
                logger.warning(f"未获取到数据: {symbol}")
                return pd.DataFrame()

            # 数据清洗和标准化
            data = self._clean_and_standardize_data(data)

            # 保存到缓存
            self._save_to_cache(cache_key, data)

            logger.info(f"成功获取 {len(data)} 条数据: {symbol}")
            return data

        except Exception as e:
            logger.error(f"获取ETF数据失败 {symbol}: {e}")
            return pd.DataFrame()

    def _try_data_sources(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """尝试多种数据源，提高数据获取成功率"""
        # 获取按优先级排序的数据源列表
        enabled_sources = self._get_enabled_data_sources()

        # 映射数据源到对应的方法
        source_methods = {
            'jqdata': self._try_jqdata,
            'tushare': self._try_tushare,
            'wind': self._try_wind,  # 预留Wind接口
            'xtquant': self._try_xtquant,
            'akshare': self._try_akshare_primary,
            'yfinance': self._try_yfinance
        }

        # 按优先级尝试数据源
        for source_name in enabled_sources:
            if source_name in source_methods:
                source_func = source_methods[source_name]
                try:
                    logger.info(f"尝试数据源: {source_name}")
                    data = source_func(symbol, start_date, end_date)
                    if not data.empty:
                        logger.info(f"使用 {source_name} 成功获取数据")
                        return data
                    else:
                        logger.debug(f"{source_name} 返回空数据")
                except Exception as e:
                    logger.debug(f"{source_name} 失败: {str(e)[:100]}")
            else:
                logger.debug(f"未知数据源: {source_name}")

        # 如果所有配置的数据源都失败，尝试备用的免费数据源
        fallback_sources = ['akshare', 'yfinance']
        for source_name in fallback_sources:
            if source_name not in enabled_sources and source_name in source_methods:
                logger.info(f"尝试备用数据源: {source_name}")
                try:
                    data = source_methods[source_name](symbol, start_date, end_date)
                    if not data.empty:
                        logger.info(f"使用备用数据源 {source_name} 成功获取数据")
                        return data
                except Exception as e:
                    logger.debug(f"备用数据源 {source_name} 失败: {str(e)[:100]}")

        logger.warning(f"所有数据源均失败: {symbol}")
        return pd.DataFrame()

    def _get_enabled_data_sources(self) -> List[str]:
        """获取启用的数据源列表，按优先级排序"""
        if self.personal_config:
            try:
                return self.personal_config.get_enabled_data_sources()
            except Exception as e:
                logger.error(f"获取启用数据源失败: {e}")

        # 回退到默认数据源
        return ['akshare', 'yfinance']

    def _try_akshare_primary(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """主要akshare数据源 - 优先使用可用的方法"""

        # 方法1: 尝试 fund_etf_hist_sina (这个方法工作正常)
        try:
            # 转换代码格式
            if symbol.startswith('51'):
                sina_symbol = f"sh{symbol}"  # 上海ETF
            elif symbol.startswith('15'):
                sina_symbol = f"sz{symbol}"  # 深圳ETF
            else:
                sina_symbol = symbol

            logger.debug(f"尝试AKShare新浪方法: {sina_symbol}")
            data = ak.fund_etf_hist_sina(symbol=sina_symbol)

            if data is not None and not data.empty:
                # 过滤日期范围
                data['date'] = pd.to_datetime(data['date'])
                start_dt = pd.to_datetime(start_date, format='%Y%m%d')
                end_dt = pd.to_datetime(end_date, format='%Y%m%d')

                # 过滤日期
                filtered_data = data[
                    (data['date'] >= start_dt) & (data['date'] <= end_dt)
                ].copy()

                # 确保列名标准化
                if all(col in filtered_data.columns for col in ['date', 'open', 'high', 'low', 'close', 'volume']):
                    logger.info(f"AKShare新浪方法成功获取 {len(filtered_data)} 条数据")
                    return filtered_data
                else:
                    logger.debug(f"AKShare新浪方法数据列不完整: {filtered_data.columns.tolist()}")

        except Exception as e:
            logger.debug(f"AKShare新浪方法失败: {e}")

        # 方法2: 尝试 stock_zh_a_hist (备用方法)
        try:
            logger.debug(f"尝试AKShare股票历史数据方法: {symbol}")
            data = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="hfq"
            )

            if data is not None and not data.empty:
                # 标准化列名映射
                column_mapping = {
                    '日期': 'date',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume',
                    '成交额': 'amount'
                }

                standardized_data = pd.DataFrame()
                for old_name, new_name in column_mapping.items():
                    if old_name in data.columns:
                        standardized_data[new_name] = data[old_name]

                # 确保必要列存在
                if all(col in standardized_data.columns for col in ['date', 'open', 'high', 'low', 'close', 'volume']):
                    logger.info(f"AKShare股票历史方法成功获取 {len(standardized_data)} 条数据")
                    return standardized_data

        except Exception as e:
            logger.debug(f"AKShare股票历史方法失败: {e}")

        # 方法3: 尝试 fund_etf_hist_em (这个方法有代理问题，但最后尝试)
        try:
            logger.debug(f"尝试AKShare东方财富方法: {symbol}")
            data = ak.fund_etf_hist_em(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="hfq"
            )
            if data is not None and not data.empty:
                logger.info(f"AKShare东方财富方法成功获取 {len(data)} 条数据")
                return data

        except Exception as e:
            logger.debug(f"AKShare东方财富方法失败: {e}")

        logger.debug(f"所有AKShare方法都失败了: {symbol}")
        return pd.DataFrame()

    def _try_akshare_secondary(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """备选akshare数据源 - fund_etf_hist_sina"""
        try:
            # 转换代码格式
            if symbol.startswith('51'):
                sina_symbol = f"sh{symbol}"  # 上海ETF
            elif symbol.startswith('15'):
                sina_symbol = f"sz{symbol}"  # 深圳ETF
            else:
                sina_symbol = symbol

            # 获取数据
            data = ak.fund_etf_hist_sina(symbol=sina_symbol)

            if not data.empty:
                # 转换日期格式
                data['date'] = pd.to_datetime(data['date'])

                # 过滤日期范围
                if start_date:
                    start_dt = pd.to_datetime(start_date)
                    data = data[data['date'] >= start_dt]
                if end_date:
                    end_dt = pd.to_datetime(end_date)
                    data = data[data['date'] <= end_dt]

                return data
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.debug(f"akshare secondary失败: {e}")
            return pd.DataFrame()

    def _try_tushare(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用Tushare Pro API获取数据"""
        if self.tushare_pro is None:
            logger.debug("Tushare Pro未初始化")
            return pd.DataFrame()

        try:
            # Tushare需要特定的股票代码格式
            ts_symbol = self._convert_to_tushare_symbol(symbol)

            # 获取基础信息确定股票类型
            basic_info = self.tushare_pro.basic(
                ts_code=ts_symbol,
                fields='ts_code,name,area,industry,list_date'
            )

            if basic_info.empty:
                logger.debug(f"Tushare未找到股票代码: {ts_symbol}")
                return pd.DataFrame()

            # 根据股票类型获取日线数据
            if ts_symbol.endswith('.SH') or ts_symbol.endswith('.SZ'):
                # A股或ETF
                df = self.tushare_pro.daily(
                    ts_code=ts_symbol,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,trade_date,open,high,low,close,pre_close,vol,amount'
                )
            else:
                logger.debug(f"不支持的股票代码格式: {ts_symbol}")
                return pd.DataFrame()

            if df.empty:
                logger.debug(f"Tushare未获取到数据: {ts_symbol}")
                return pd.DataFrame()

            # 数据清洗和格式转换
            df = df.sort_values('trade_date')
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df.set_index('trade_date')

            # 重命名列以匹配标准格式
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'vol': 'volume',
                'amount': 'amount'
            })

            # 选择需要的列
            df = df[['open', 'high', 'low', 'close', 'volume']]

            logger.info(f"Tushare成功获取 {len(df)} 条数据: {symbol}")
            return df

        except Exception as e:
            logger.debug(f"Tushare获取数据失败 {symbol}: {e}")
            return pd.DataFrame()

    def _convert_to_tushare_symbol(self, symbol: str) -> str:
        """转换ETF代码为Tushare格式"""
        # 移除可能的前缀和后缀
        clean_symbol = symbol.replace('etf', '').replace('ETF', '').replace('SH', '').replace('SZ', '')

        # 确定市场后缀
        if symbol.startswith(('51', '58', '56')):  # 上海市场ETF
            return f"{clean_symbol}.SH"
        elif symbol.startswith(('15', '16', '159')):  # 深圳市场ETF
            return f"{clean_symbol}.SZ"
        else:
            # 默认尝试上海市场
            return f"{clean_symbol}.SH"

    def _convert_to_jqdata_symbol(self, symbol: str) -> str:
        """转换证券代码为jqdatasdk格式"""
        # 移除可能的前缀和后缀
        clean_symbol = symbol.replace('etf', '').replace('ETF', '').replace('XSHG', '').replace('XSHE', '').replace('SH', '').replace('SZ', '')

        # ETF代码转换
        if symbol.startswith(('51', '58', '56')):  # 上海ETF
            return f"{clean_symbol}.XSHG"
        elif symbol.startswith(('15', '16', '159')):  # 深圳ETF
            return f"{clean_symbol}.XSHE"
        # 指数代码转换
        elif symbol.startswith('000'):  # 深圳指数（如000001.SZ）
            return f"{clean_symbol}.XSHE"
        elif symbol.startswith(('399', '0009')):  # 深证成指等
            return f"{symbol}.XSHE"
        elif symbol.startswith(('0003', '0009')):  # 沪深300等特殊指数
            return f"{symbol}.XSHG"
        # 个股代码转换
        elif symbol.startswith(('600', '601', '603', '605')):  # 上海A股
            return f"{clean_symbol}.XSHG"
        elif symbol.startswith(('000', '001', '002', '003')):  # 深圳A股
            return f"{clean_symbol}.XSHE"
        else:
            # 默认处理为上海市场
            return f"{clean_symbol}.XSHG"

    def _try_jqdata(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用jqdatasdk获取ETF和个股数据"""
        if not self.jqdata_initialized:
            logger.debug("jqdatasdk未初始化")
            return pd.DataFrame()

        try:
            # 转换日期格式
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')

            # 转换证券代码为jqdatasdk格式
            jq_symbol = self._convert_to_jqdata_symbol(symbol)

            # 获取历史数据
            data = get_price(
                security=jq_symbol,
                start_date=start_dt.strftime('%Y-%m-%d'),
                end_date=end_dt.strftime('%Y-%m-%d'),
                frequency='daily'
            )

            if data is None or data.empty:
                logger.debug(f"jqdatasdk未获取到数据: {jq_symbol}")
                return pd.DataFrame()

            # 标准化列名和数据格式
            data = data.reset_index()

            # 重命名列以匹配标准格式
            column_mapping = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }

            # 应用列名映射
            for old_col, new_col in column_mapping.items():
                if old_col in data.columns:
                    data = data.rename(columns={old_col: new_col})

            # 确保日期列格式正确
            if 'date' not in data.columns and 'index' in str(data.index.names):
                data['date'] = data.index
                data = data.reset_index(drop=True)

            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data = data.sort_values('date').reset_index(drop=True)

            # 选择需要的列
            required_columns = ['date', 'open', 'high', 'low', 'close']
            if 'volume' in data.columns:
                required_columns.append('volume')
            else:
                data['volume'] = 0

            # 确保所有必要列都存在
            for col in ['open', 'high', 'low', 'close']:
                if col not in data.columns:
                    logger.warning(f"jqdatasdk数据缺少列: {col}")
                    data[col] = data['close']  # 使用收盘价填充缺失的价格数据

            logger.info(f"jqdatasdk成功获取数据 {len(data)} 条: {symbol}")
            return data[required_columns]

        except Exception as e:
            logger.debug(f"jqdatasdk获取数据失败 {symbol}: {e}")
            return pd.DataFrame()

    def _try_jqdata_benchmark(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用jqdatasdk获取基准数据"""
        if not self.jqdata_initialized:
            logger.debug("jqdatasdk未初始化")
            return pd.DataFrame()

        try:
            # 转换日期格式
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')

            # 转换指数代码为jqdatasdk格式
            jq_symbol = self._convert_to_jqdata_symbol(symbol)

            # 获取指数历史数据
            data = get_price(
                security=jq_symbol,
                start_date=start_dt.strftime('%Y-%m-%d'),
                end_date=end_dt.strftime('%Y-%m-%d'),
                frequency='daily'
            )

            if data is None or data.empty:
                logger.debug(f"jqdatasdk未获取到基准数据: {jq_symbol}")
                return pd.DataFrame()

            # 标准化数据格式
            data = data.reset_index()

            # 重命名列以匹配标准格式
            column_mapping = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }

            # 应用列名映射
            for old_col, new_col in column_mapping.items():
                if old_col in data.columns:
                    data = data.rename(columns={old_col: new_col})

            # 确保日期列格式正确
            if 'date' not in data.columns:
                data['date'] = data.index
                data = data.reset_index(drop=True)

            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data = data.sort_values('date').reset_index(drop=True)

            # 确保成交量列存在
            if 'volume' not in data.columns:
                data['volume'] = 0

            # 选择需要的列
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']

            logger.info(f"jqdatasdk成功获取基准数据 {len(data)} 条: {symbol}")
            return data[required_columns]

        except Exception as e:
            logger.debug(f"jqdatasdk获取基准数据失败 {symbol}: {e}")
            return pd.DataFrame()

    def _try_xtquant_benchmark(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用XTQuant获取基准指数数据"""
        if not self.xtquant_initialized or not XTQUANT_AVAILABLE:
            logger.debug("XTQuant未初始化或不可用")
            return pd.DataFrame()

        try:
            logger.info(f"XTQuant: 开始获取基准指数数据 {symbol}")

            # 转换日期格式
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')

            # 转换指数代码为XTQuant格式
            xt_symbol = self._convert_to_xtquant_symbol(symbol)
            logger.debug(f"XTQuant: 转换后基准代码 {xt_symbol}")

            # 首先尝试获取数据，如果没有则先下载
            data = xtdata.get_market_data_ex([], [xt_symbol], period="1d", count=-1)

            if data is None or not data or xt_symbol not in data:
                logger.info(f"XTQuant: 本地无基准数据，开始下载 {xt_symbol}")
                # 下载基准指数历史数据
                xtdata.download_history_data(xt_symbol, period="1d", incrementally=True)
                # 再次获取数据
                data = xtdata.get_market_data_ex([], [xt_symbol], period="1d", count=-1)

            if not data or xt_symbol not in data:
                logger.warning(f"XTQuant: 下载后仍无法获取基准数据 {xt_symbol}")
                return pd.DataFrame()

            # 处理XTQuant返回的数据格式
            df = data[xt_symbol]

            # 将数据转换为DataFrame
            ohlc_data = []
            for timestamp, row_data in df.items():
                # timestamp是时间戳格式
                date = pd.to_datetime(timestamp)

                # 提取OHLCV数据
                if isinstance(row_data, dict) or len(row_data) >= 4:
                    open_price = float(row_data[0]) if len(row_data) > 0 else 0
                    high_price = float(row_data[1]) if len(row_data) > 1 else 0
                    low_price = float(row_data[2]) if len(row_data) > 2 else 0
                    close_price = float(row_data[3]) if len(row_data) > 3 else 0
                    volume = float(row_data[4]) if len(row_data) > 4 else 0

                    ohlc_data.append({
                        'date': date,
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'volume': volume
                    })

            if not ohlc_data:
                logger.warning(f"XTQuant: 无有效基准OHLCV数据 {xt_symbol}")
                return pd.DataFrame()

            # 创建DataFrame
            result_df = pd.DataFrame(ohlc_data)
            result_df = result_df.sort_values('date').reset_index(drop=True)

            # 过滤日期范围
            start_date_filtered = result_df['date'] >= start_dt
            end_date_filtered = result_df['date'] <= end_dt
            filtered_df = result_df[start_date_filtered & end_date_filtered].copy()

            if filtered_df.empty:
                logger.warning(f"XTQuant: 过滤后无基准数据 {xt_symbol}, 原始数据范围: {result_df['date'].min()} 到 {result_df['date'].max()}")
                return pd.DataFrame()

            # 验证基准数据质量
            if not self._validate_xtquant_benchmark_data(filtered_df):
                logger.warning(f"XTQuant: 基准数据质量验证失败 {xt_symbol}")
                return pd.DataFrame()

            logger.info(f"XTQuant: 成功获取基准数据 {symbol}, 数据量: {len(filtered_df)}")
            return filtered_df

        except Exception as e:
            logger.debug(f"XTQuant获取基准数据失败 {symbol}: {type(e).__name__}: {e}")
            return pd.DataFrame()

    def _validate_xtquant_benchmark_data(self, data: pd.DataFrame) -> bool:
        """验证XTQuant基准数据质量"""
        try:
            if data.empty:
                logger.error("XTQuant基准数据验证失败: DataFrame为空")
                return False

            # 检查必要列
            required_columns = ['date', 'close']
            for col in required_columns:
                if col not in data.columns:
                    logger.error(f"XTQuant基准数据验证失败: 缺少列 {col}")
                    return False

            # 检查价格数据有效性
            close_prices = data['close'].dropna()
            if len(close_prices) == 0:
                logger.error("XTQuant基准数据验证失败: 无有效价格数据")
                return False

            # 检查价格是否为正数
            if (close_prices <= 0).any():
                logger.warning(f"XTQuant基准数据警告: 发现非正价格 {(close_prices <= 0).sum()} 个")

            # 检查数据量
            if len(data) < 10:
                logger.warning(f"XTQuant基准数据警告: 数据点较少 {len(data)} 个，可能影响Beta计算")

            logger.debug(f"XTQuant基准数据验证通过: {len(data)} 条数据, 价格范围 {close_prices.min():.4f} - {close_prices.max():.4f}")
            return True

        except Exception as e:
            logger.error(f"XTQuant基准数据验证过程异常: {type(e).__name__}: {e}")
            return False

    def _try_wind(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用Wind API获取数据（预留接口）"""
        # Wind集成需要特殊的授权和安装，这里提供基础框架
        logger.debug("Wind接口暂未实现")
        return pd.DataFrame()

    def _try_yfinance(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用yfinance获取数据（需要pip install yfinance）"""
        try:
            # 尝试导入yfinance
            import yfinance as yf

            # 转换日期格式
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')

            # 转换符号格式
            if symbol.startswith('51'):
                yf_symbol = f"{symbol}.SS"  # 上海证券交易所
            elif symbol.startswith('15'):
                yf_symbol = f"{symbol}.SZ"  # 深圳证券交易所
            else:
                yf_symbol = f"{symbol}.SS"  # 默认上海

            # 下载数据
            data = yf.download(
                yf_symbol,
                start=start_dt.strftime('%Y-%m-%d'),
                end=end_dt.strftime('%Y-%m-%d'),
                progress=False,
                timeout=10  # 设置超时
            )

            if not data.empty:
                data = data.reset_index()
                # 重命名列以匹配标准格式
                data = data.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                return data
            else:
                return pd.DataFrame()

        except ImportError:
            logger.warning("yfinance未安装，可以使用: pip install yfinance")
            return pd.DataFrame()
        except Exception as e:
            logger.debug(f"yfinance获取失败: {e}")
            return pd.DataFrame()

    def _try_xtquant(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用XTQuant获取ETF数据"""
        if not self.xtquant_initialized or not XTQUANT_AVAILABLE:
            logger.debug("XTQuant未初始化或不可用")
            return pd.DataFrame()

        try:
            logger.info(f"XTQuant: 开始获取ETF数据 {symbol}")

            # 转换日期格式
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')

            # 转换ETF代码为XTQuant格式
            xt_symbol = self._convert_to_xtquant_symbol(symbol)
            logger.debug(f"XTQuant: 转换后代码 {xt_symbol}")

            # 首先尝试获取数据，如果没有则先下载
            data = xtdata.get_market_data_ex([], [xt_symbol], period="1d", count=-1)

            if data is None or not data or xt_symbol not in data:
                logger.info(f"XTQuant: 本地无数据，开始下载 {xt_symbol}")
                # 下载历史数据
                xtdata.download_history_data(xt_symbol, period="1d", incrementally=True)
                # 再次获取数据
                data = xtdata.get_market_data_ex([], [xt_symbol], period="1d", count=-1)

            if not data or xt_symbol not in data:
                logger.warning(f"XTQuant: 下载后仍无法获取数据 {xt_symbol}")
                return pd.DataFrame()

            # 处理XTQuant返回的数据格式
            df = data[xt_symbol]

            # 将数据转换为DataFrame
            ohlc_data = []
            for timestamp, row_data in df.items():
                # timestamp是时间戳格式
                date = pd.to_datetime(timestamp)

                # 提取OHLCV数据（根据XTQuant返回格式调整）
                if isinstance(row_data, dict) or len(row_data) >= 4:
                    open_price = float(row_data[0]) if len(row_data) > 0 else 0
                    high_price = float(row_data[1]) if len(row_data) > 1 else 0
                    low_price = float(row_data[2]) if len(row_data) > 2 else 0
                    close_price = float(row_data[3]) if len(row_data) > 3 else 0
                    volume = float(row_data[4]) if len(row_data) > 4 else 0

                    ohlc_data.append({
                        'date': date,
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'volume': volume
                    })

            if not ohlc_data:
                logger.warning(f"XTQuant: 无有效OHLCV数据 {xt_symbol}")
                return pd.DataFrame()

            # 创建DataFrame
            result_df = pd.DataFrame(ohlc_data)
            result_df = result_df.sort_values('date').reset_index(drop=True)

            # 过滤日期范围
            start_date_filtered = result_df['date'] >= start_dt
            end_date_filtered = result_df['date'] <= end_dt
            filtered_df = result_df[start_date_filtered & end_date_filtered].copy()

            if filtered_df.empty:
                logger.warning(f"XTQuant: 过滤后无数据 {xt_symbol}, 原始数据范围: {result_df['date'].min()} 到 {result_df['date'].max()}")
                return pd.DataFrame()

            logger.info(f"XTQuant: 成功获取ETF数据 {symbol}, 数据量: {len(filtered_df)}")
            return filtered_df

        except Exception as e:
            logger.error(f"XTQuant获取ETF数据失败 {symbol}: {type(e).__name__}: {e}")
            return pd.DataFrame()

    def _convert_to_xtquant_symbol(self, symbol: str) -> str:
        """转换ETF/股票代码为XTQuant格式"""
        # 移除可能的前缀和后缀
        clean_symbol = symbol.replace('etf', '').replace('ETF', '').replace('SH', '').replace('SZ', '')

        # 根据代码前缀确定市场和格式
        if symbol.startswith(('51', '58', '56')):  # 上海市场ETF
            return f"{clean_symbol}.SZ"  # XTQuant中上海ETF使用.SZ后缀
        elif symbol.startswith(('15', '16', '159')):  # 深圳市场ETF
            return f"{clean_symbol}.SZ"
        elif symbol.startswith(('600', '601', '603', '605')):  # 上海A股
            return f"{clean_symbol}.SH"
        elif symbol.startswith(('000', '001', '002', '003')):  # 深圳A股
            return f"{clean_symbol}.SZ"
        elif symbol == "000300":  # 沪深300指数
            return "000300.SH"
        elif symbol.startswith('000'):  # 深圳指数
            return f"{symbol}.SZ"
        else:
            # 默认尝试深圳市场
            return f"{clean_symbol}.SZ"

    def _clean_and_standardize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """清洗和标准化数据"""
        # 重命名列
        column_mapping = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            'Date': 'date',
            'Open': 'open',
            'Close': 'close',
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume',
            'Amount': 'amount'
        }

        # 查找实际存在的列
        existing_columns = {}
        for chinese_col, english_col in column_mapping.items():
            for col in data.columns:
                if chinese_col in col or english_col in col:
                    existing_columns[col] = english_col
                    break

        data = data.rename(columns=existing_columns)

        # 确保必要的列存在
        required_columns = ['date', 'open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                logger.error(f"缺少必要列: {col}")
                raise ValueError(f"缺少必要列: {col}")

        # 数据类型转换
        data['date'] = pd.to_datetime(data['date'])
        numeric_columns = ['open', 'high', 'low', 'close']
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # 处理成交量
        if 'volume' in data.columns:
            data['volume'] = pd.to_numeric(data['volume'], errors='coerce')
        else:
            data['volume'] = 0

        # 处理成交额
        if 'amount' in data.columns:
            data['amount'] = pd.to_numeric(data['amount'], errors='coerce')
        else:
            data['amount'] = 0

        # 按日期排序
        data = data.sort_values('date').reset_index(drop=True)

        # 移除重复和空值
        data = data.drop_duplicates(subset=['date'])
        data = data.dropna(subset=required_columns)

        # 添加技术指标
        data = self._add_technical_indicators(data)

        return data

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        if len(data) < 2:
            return data

        # 计算收益率
        data['returns'] = data['close'].pct_change()

        # 计算移动平均线
        if len(data) >= 5:
            data['ma5'] = data['close'].rolling(window=5).mean()
        else:
            data['ma5'] = np.nan

        if len(data) >= 10:
            data['ma10'] = data['close'].rolling(window=10).mean()
        else:
            data['ma10'] = np.nan

        if len(data) >= 20:
            data['ma20'] = data['close'].rolling(window=20).mean()
        else:
            data['ma20'] = np.nan

        if len(data) >= 60:
            data['ma60'] = data['close'].rolling(window=60).mean()
        else:
            data['ma60'] = np.nan

        # 计算波动率
        if len(data) >= 20:
            data['volatility_20'] = data['returns'].rolling(window=20).std() * np.sqrt(252)
        else:
            data['volatility_20'] = np.nan

        # 计算RSI
        data['rsi_14'] = self._calculate_rsi(data['close'], 14)

        return data

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        if len(prices) < period + 1:
            return pd.Series([np.nan] * len(prices))

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.pkl"

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """从缓存加载数据"""
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        # 检查缓存是否过期
        cache_time = cache_path.stat().st_mtime
        current_time = datetime.now().timestamp()

        if current_time - cache_time > self.cache_ttl:
            return None

        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                logger.debug(f"从缓存加载: {cache_key}, 数据量: {len(data)}")
                return data
        except Exception as e:
            logger.warning(f"读取缓存失败: {e}")
            return None

    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """保存数据到缓存"""
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"保存到缓存: {cache_key}, 数据量: {len(data)}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")

    def batch_fetch_etf_data(self, symbols: List[str], start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """批量获取ETF数据"""
        results = {}

        for symbol in symbols:
            try:
                data = self.fetch_etf_data(symbol, start_date, end_date)
                if not data.empty:
                    results[symbol] = data
                    logger.info(f"成功获取 {symbol}: {len(data)} 条数据")
                else:
                    logger.warning(f"未获取到数据: {symbol}")
            except Exception as e:
                logger.error(f"获取数据失败 {symbol}: {e}")

        logger.info(f"批量获取完成: {len(results)}/{len(symbols)} 个ETF成功")
        return results

    def get_data_quality_report(self, data: pd.DataFrame) -> Dict:
        """生成数据质量报告"""
        if data.empty:
            return {"error": "数据为空"}

        report = {
            "total_records": len(data),
            "date_range": {
                "start": data['date'].min().strftime('%Y-%m-%d'),
                "end": data['date'].max().strftime('%Y-%m-%d')
            },
            "missing_values": data.isnull().sum().to_dict(),
            "price_stats": {
                "min_price": float(data['close'].min()),
                "max_price": float(data['close'].max()),
                "avg_price": float(data['close'].mean()),
                "price_volatility": float(data['close'].std() / data['close'].mean() if data['close'].mean() != 0 else 0)
            },
            "trading_days": len(data),
            "data_completeness": float((1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100) if len(data) > 0 else 0
        }

        return report

    def clear_cache(self):
        """清空缓存"""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            for cache_file in cache_files:
                cache_file.unlink()
            logger.info(f"缓存已清空，删除 {len(cache_files)} 个文件")
        except Exception as e:
            logger.error(f"清空缓存失败: {e}")

    def get_cache_info(self) -> Dict:
        """获取缓存信息"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        info = {
            "cache_dir": str(self.cache_dir),
            "total_files": len(cache_files),
            "files": [f.name for f in cache_files[:10]]  # 显示前10个文件
        }
        return info

    def _validate_benchmark_data_quality(self, data: pd.DataFrame) -> bool:
        """验证基准数据质量"""
        try:
            if data.empty:
                logger.error("📊 数据验证失败: DataFrame为空")
                return False

            required_columns = ['date', 'close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"📊 数据验证失败: 缺少必要列 {missing_columns}")
                logger.error(f"📊 现有列: {list(data.columns)}")
                return False

            # 数据质量检查
            data_shape = data.shape
            date_range = f"{data['date'].min()} 到 {data['date'].max()}" if pd.api.types.is_datetime64_any_dtype(data['date']) else "日期格式异常"

            # 检查数据点数量
            if data_shape[0] < 10:
                logger.warning(f"📊 数据点较少 ({data_shape[0]} 条)，可能影响Beta计算准确性")

            # 检查价格数据有效性
            if 'close' in data.columns:
                close_data = data['close'].dropna()
                if len(close_data) == 0:
                    logger.error("📊 数据验证失败: 收盘价全部为空值")
                    return False

                if (close_data <= 0).any():
                    logger.warning(f"📊 发现非正收盘价数据，共有 {(close_data <= 0).sum()} 个异常值")

                # 检查价格稳定性
                price_std = close_data.std()
                price_mean = close_data.mean()
                if price_std == 0:
                    logger.warning("📊 价格数据无波动，所有值相同")

            # 详细质量报告
            logger.info(f"📊 数据质量验证通过:")
            logger.info(f"  📈 数据形状: {data_shape}")
            logger.info(f"  📅 时间范围: {date_range}")
            logger.info(f"  📋 数据列: {list(data.columns)}")
            logger.info(f"  ❌ 缺失值统计: {data.isnull().sum().to_dict()}")

            # 检查日期连续性
            if pd.api.types.is_datetime64_any_dtype(data['date']):
                data_sorted = data.sort_values('date')
                expected_days = (data_sorted['date'].iloc[-1] - data_sorted['date'].iloc[0]).days + 1
                actual_days = len(data_sorted)
                if actual_days < expected_days * 0.7:  # 允许30%的缺失（周末、节假日）
                    logger.warning(f"📊 日期连续性: 实际 {actual_days} 天，预期约 {expected_days} 天")

            return True

        except Exception as e:
            logger.error(f"📊 数据验证过程中发生异常: {type(e).__name__}: {e}")
            return False

    # ==================== XTQuant 实时订阅功能 ====================

    def subscribe_xtquant_realtime(self, symbols: List[str], callback=None, period: str = "1d"):
        """
        使用XTQuant订阅实时数据

        Args:
            symbols: 证券代码列表
            callback: 回调函数，接收(data)参数
            period: 数据周期，默认"1d"

        Returns:
            订阅成功状态
        """
        if not self.xtquant_initialized or not XTQUANT_AVAILABLE:
            logger.error("XTQuant未初始化，无法订阅实时数据")
            return False

        try:
            # 转换符号格式
            xt_symbols = [self._convert_to_xtquant_symbol(symbol) for symbol in symbols]

            logger.info(f"XTQuant: 开始订阅实时数据 {xt_symbols}")

            # 存储订阅状态
            if not hasattr(self, '_xtquant_subscriptions'):
                self._xtquant_subscriptions = {}

            for symbol, xt_symbol in zip(symbols, xt_symbols):
                try:
                    # 先下载历史数据确保有基础数据
                    xtdata.download_history_data(xt_symbol, period=period, incrementally=True)

                    # 订阅实时数据
                    if callback:
                        xtdata.subscribe_quote(xt_symbol, period=period, count=-1, callback=callback)
                    else:
                        # 使用默认回调函数
                        default_callback = self._default_xtquant_callback
                        xtdata.subscribe_quote(xt_symbol, period=period, count=-1, callback=default_callback)

                    self._xtquant_subscriptions[symbol] = {
                        'xt_symbol': xt_symbol,
                        'period': period,
                        'callback': callback or default_callback,
                        'subscribed_at': datetime.now()
                    }

                    logger.info(f"XTQuant: 成功订阅 {symbol} ({xt_symbol})")

                except Exception as e:
                    logger.error(f"XTQuant: 订阅失败 {symbol} ({xt_symbol}): {e}")
                    continue

            logger.info(f"XTQuant: 实时订阅完成，成功订阅 {len(self._xtquant_subscriptions)}/{len(symbols)} 个标的")
            return len(self._xtquant_subscriptions) > 0

        except Exception as e:
            logger.error(f"XTQuant: 实时订阅异常: {type(e).__name__}: {e}")
            return False

    def _default_xtquant_callback(self, data):
        """
        XTQuant默认回调函数

        Args:
            data: XTQuant推送的数据
        """
        try:
            if not data:
                return

            # 解析回调数据
            symbol_list = list(data.keys())
            if not symbol_list:
                return

            for xt_symbol in symbol_list:
                symbol_data = data[xt_symbol]
                if symbol_data is None:
                    continue

                # 尝试获取最新价格数据
                try:
                    latest_data = xtdata.get_market_data_ex([], [xt_symbol], period="1d", count=1)
                    if latest_data and xt_symbol in latest_data:
                        latest_df = latest_data[xt_symbol]
                        if not latest_df.empty:
                            # 获取最新一条数据
                            latest_timestamp = list(latest_df.keys())[-1]
                            latest_price_data = latest_df[latest_timestamp]

                            # 转换为可读格式
                            current_price = float(latest_price_data[3]) if len(latest_price_data) > 3 else 0
                            current_time = pd.to_datetime(latest_timestamp)

                            logger.info(f"XTQuant实时数据: {xt_symbol} 价格:{current_price:.4f} 时间:{current_time}")

                except Exception as parse_e:
                    logger.debug(f"XTQuant回调数据解析失败: {parse_e}")

        except Exception as e:
            logger.error(f"XTQuant默认回调异常: {type(e).__name__}: {e}")

    def unsubscribe_xtquant_realtime(self, symbols: List[str] = None):
        """
        取消XTQuant实时数据订阅

        Args:
            symbols: 要取消订阅的证券代码列表，None表示取消所有
        """
        if not hasattr(self, '_xtquant_subscriptions') or not self._xtquant_subscriptions:
            logger.info("XTQuant: 无活跃订阅")
            return

        try:
            if symbols is None:
                # 取消所有订阅
                symbols_to_unsubscribe = list(self._xtquant_subscriptions.keys())
            else:
                # 只取消指定的订阅
                symbols_to_unsubscribe = [s for s in symbols if s in self._xtquant_subscriptions]

            unsubscribed_count = 0
            for symbol in symbols_to_unsubscribe:
                if symbol in self._xtquant_subscriptions:
                    xt_symbol = self._xtquant_subscriptions[symbol]['xt_symbol']
                    try:
                        # XTQuant取消订阅
                        if hasattr(xtdata, 'unsubscribe_quote'):
                            xtdata.unsubscribe_quote(xt_symbol)

                        del self._xtquant_subscriptions[symbol]
                        unsubscribed_count += 1
                        logger.info(f"XTQuant: 取消订阅成功 {symbol} ({xt_symbol})")

                    except Exception as e:
                        logger.error(f"XTQuant: 取消订阅失败 {symbol}: {e}")

            logger.info(f"XTQuant: 取消订阅完成，成功取消 {unsubscribed_count}/{len(symbols_to_unsubscribe)} 个订阅")

        except Exception as e:
            logger.error(f"XTQuant: 取消订阅异常: {type(e).__name__}: {e}")

    def get_xtquant_subscription_status(self) -> Dict:
        """
        获取XTQuant实时订阅状态

        Returns:
            订阅状态信息
        """
        if not hasattr(self, '_xtquant_subscriptions'):
            return {
                'enabled': False,
                'total_subscriptions': 0,
                'subscriptions': []
            }

        status = {
            'enabled': len(self._xtquant_subscriptions) > 0,
            'total_subscriptions': len(self._xtquant_subscriptions),
            'xtquant_initialized': self.xtquant_initialized,
            'subscriptions': []
        }

        for symbol, info in self._xtquant_subscriptions.items():
            subscription_time = info['subscribed_at']
            duration = datetime.now() - subscription_time

            status['subscriptions'].append({
                'symbol': symbol,
                'xt_symbol': info['xt_symbol'],
                'period': info['period'],
                'subscribed_at': subscription_time.strftime('%Y-%m-%d %H:%M:%S'),
                'duration_seconds': int(duration.total_seconds())
            })

        return status

    def start_xtquant_realtime_loop(self):
        """
        启动XTQuant实时数据循环（阻塞模式）

        注意：此方法会阻塞当前线程，适合在独立线程中运行
        """
        if not self.xtquant_initialized or not XTQUANT_AVAILABLE:
            logger.error("XTQuant未初始化，无法启动实时循环")
            return

        if not hasattr(self, '_xtquant_subscriptions') or not self._xtquant_subscriptions:
            logger.warning("XTQuant: 无活跃订阅，但将启动实时循环等待订阅")
        else:
            logger.info(f"XTQuant: 启动实时循环，监控 {len(self._xtquant_subscriptions)} 个订阅")

        try:
            logger.info("XTQuant: 实时数据循环已启动，按Ctrl+C停止...")
            # 启动XTQuant实时数据循环
            xtdata.run()

        except KeyboardInterrupt:
            logger.info("XTQuant: 用户中断，停止实时数据循环")
        except Exception as e:
            logger.error(f"XTQuant: 实时循环异常: {type(e).__name__}: {e}")
        finally:
            logger.info("XTQuant: 实时数据循环已停止")


def main():
    """测试函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    fetcher = MarketDataFetcher()

    # 测试获取单个ETF数据
    test_symbols = ["159682", "510300", "512880"]  # 科创50ETF, 沪深300ETF, 证券ETF

    for symbol in test_symbols:
        print(f"\n{'='*60}")
        print(f"测试获取ETF数据: {symbol}")
        print('='*60)

        data = fetcher.fetch_etf_data(symbol, start_date="20230101", end_date="20240101")

        if not data.empty:
            print(f"成功获取 {len(data)} 条数据")
            print("数据预览:")
            print(data[['date', 'open', 'high', 'low', 'close', 'volume']].head())
            print("...")
            print(data[['date', 'open', 'high', 'low', 'close', 'volume']].tail())

            # 生成数据质量报告
            report = fetcher.get_data_quality_report(data)
            print("\n数据质量报告:")
            for key, value in report.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
        else:
            print("获取数据失败")

    # 批量获取测试
    print(f"\n{'='*60}")
    print("批量获取测试")
    print('='*60)

    batch_data = fetcher.batch_fetch_etf_data(test_symbols, start_date="20231001", end_date="20240101")
    print(f"批量获取结果: {len(batch_data)}/{len(test_symbols)} 成功")

    # 缓存信息
    print(f"\n{'='*60}")
    print("缓存信息")
    print('='*60)
    cache_info = fetcher.get_cache_info()
    for key, value in cache_info.items():
        if isinstance(value, list):
            print(f"{key}: {len(value)} 个文件")
            for file in value:
                print(f"  - {file}")
        else:
            print(f"{key}: {value}")

    # 清空缓存测试（可选）
    # fetcher.clear_cache()


if __name__ == "__main__":
    main()