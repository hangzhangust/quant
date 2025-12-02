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

# 导入个人配置系统
try:
    from src.config.personal_config import get_personal_config
    PERSONAL_CONFIG_AVAILABLE = True
except ImportError:
    PERSONAL_CONFIG_AVAILABLE = False
    logger.warning("个人配置系统不可用，将使用默认配置")

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

        # 初始化Tushare（如果配置了）
        self._init_tushare()

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
        # 获取按优先级排序的数据源列表
        enabled_sources = self._get_enabled_data_sources()

        # 映射数据源到对应的基准获取方法
        benchmark_methods = {
            'tushare': lambda: self._try_tushare_benchmark(symbol, start_date, end_date),
            'wind': lambda: self._try_wind_benchmark(symbol, start_date, end_date),
            'akshare': lambda: self._try_akshare_benchmark(symbol, start_date, end_date),
            'yfinance': lambda: self._try_yahoo_benchmark(symbol, start_date, end_date)
        }

        # 按优先级尝试数据源
        for source_name in enabled_sources:
            if source_name in benchmark_methods:
                try:
                    logger.info(f"尝试基准数据源: {source_name}")
                    data = benchmark_methods[source_name]()
                    if data is not None and not data.empty:
                        logger.info(f"使用 {source_name} 成功获取基准数据")
                        return data
                    else:
                        logger.debug(f"{source_name} 基准数据返回空")
                except Exception as e:
                    logger.debug(f"{source_name} 基准数据源失败: {e}")

        # 如果所有配置的数据源都失败，尝试备用的免费数据源
        fallback_sources = ['akshare', 'yfinance']
        for source_name in fallback_sources:
            if source_name not in enabled_sources and source_name in benchmark_methods:
                try:
                    logger.info(f"尝试备用基准数据源: {source_name}")
                    data = benchmark_methods[source_name]()
                    if data is not None and not data.empty:
                        logger.info(f"使用备用基准数据源 {source_name} 成功获取数据")
                        return data
                except Exception as e:
                    logger.debug(f"备用基准数据源 {source_name} 失败: {e}")

        logger.warning(f"所有基准数据源均失败: {symbol}")
        return pd.DataFrame()

    def _try_akshare_benchmark(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用AKShare获取基准数据"""
        # 对于指数代码，需要转换为akshare支持的格式
        # 沪深300指数
        if symbol == "000300":
            symbol_map = {
                "000300": "sh000300",
                "300": "sh000300"
            }
        else:
            symbol_map = {symbol: f"sh{symbol}"}

        for ak_symbol in symbol_map.values():
            try:
                # 尝试获取指数数据
                data = ak.stock_zh_a_hist(symbol=ak_symbol, period="daily",
                                          start_date=start_date, end_date=end_date)

                if data is not None and not data.empty:
                    # 标准化列名
                    data = data.rename(columns={
                        '日期': 'date',
                        '开盘': 'open',
                        '收盘': 'close',
                        '最高': 'high',
                        '最低': 'low',
                        '成交量': 'volume'
                    })

                    # 确保日期格式正确
                    if 'date' in data.columns:
                        data['date'] = pd.to_datetime(data['date'])
                        data = data.set_index('date')

                    return data

            except Exception as e:
                logger.debug(f"AKShare基准数据获取失败 {ak_symbol}: {e}")
                continue

        return None

    def _try_yahoo_benchmark(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用Yahoo Finance获取基准数据"""
        try:
            import yfinance as yf

            # 转换为Yahoo Finance格式
            yf_symbol = f"{symbol}.SS"

            # 创建Ticker对象
            ticker = yf.Ticker(yf_symbol)

            # 获取历史数据
            data = ticker.history(start=start_date, end=end_date)

            if data is not None and not data.empty:
                # 标准化列名
                data.columns = [col.lower() for col in data.columns]
                return data

        except Exception as e:
            logger.debug(f"Yahoo Finance基准数据获取失败 {symbol}: {e}")
            return None

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
            'tushare': self._try_tushare,
            'wind': self._try_wind,  # 预留Wind接口
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
        """主要akshare数据源 - fund_etf_hist_em"""
        try:
            data = ak.fund_etf_hist_em(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="hfq"
            )
            return data
        except Exception as e:
            logger.debug(f"akshare primary失败: {e}")
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