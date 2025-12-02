"""
模拟基准数据生成器
用于在网络连接问题时生成模拟的基准数据进行Beta计算测试
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class MockBenchmarkGenerator:
    """模拟基准数据生成器"""

    def __init__(self):
        self.benchmark_configs = {
            '000300': {
                'name': '沪深300指数',
                'initial_price': 3000.0,
                'volatility': 0.018,
                'trend': 0.0002,  # 每日趋势
                'symbol': 'CSI300'
            },
            '000001': {
                'name': '上证综指',
                'initial_price': 3200.0,
                'volatility': 0.016,
                'trend': 0.00015,
                'symbol': 'SHCI'
            },
            '399001': {
                'name': '深证成指',
                'initial_price': 12000.0,
                'volatility': 0.020,
                'trend': 0.00025,
                'symbol': 'SZCI'
            }
        }

    def generate_benchmark_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        生成模拟的基准数据

        Args:
            symbol: 基准代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)

        Returns:
            pd.DataFrame: 标准格式的基准数据
        """
        try:
            # 获取基准配置
            if symbol not in self.benchmark_configs:
                logger.warning(f"未知基准代码 {symbol}，使用默认配置")
                config = {
                    'name': f'基准{symbol}',
                    'initial_price': 1000.0,
                    'volatility': 0.018,
                    'trend': 0.0002,
                    'symbol': symbol
                }
            else:
                config = self.benchmark_configs[symbol]

            # 解析日期
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')

            # 生成日期序列（仅包含工作日）
            dates = pd.bdate_range(start=start_dt, end=end_dt)

            if len(dates) == 0:
                logger.error(f"指定日期范围内无工作日: {start_date} - {end_date}")
                return pd.DataFrame()

            # 生成价格数据
            n_days = len(dates)
            prices = [config['initial_price']]

            # 使用几何布朗运动模型生成价格
            for i in range(1, n_days):
                daily_return = np.random.normal(config['trend'], config['volatility'])
                new_price = prices[-1] * (1 + daily_return)
                prices.append(new_price)

            # 创建DataFrame
            data = pd.DataFrame({
                'date': dates,
                'close': prices,
                'open': [p * np.random.uniform(0.998, 1.002) for p in prices],  # 开盘价略有变化
                'high': [p * np.random.uniform(1.0, 1.025) for p in prices],   # 高价
                'low': [p * np.random.uniform(0.975, 1.0) for p in prices],    # 低价
                'volume': [np.random.randint(1000000, 5000000) for _ in range(n_days)]  # 成交量
            })

            # 确保价格关系合理
            for i in range(len(data)):
                close = data.loc[i, 'close']
                data.loc[i, 'high'] = max(data.loc[i, 'high'], close)
                data.loc[i, 'low'] = min(data.loc[i, 'low'], close)
                data.loc[i, 'open'] = np.clip(data.loc[i, 'open'], data.loc[i, 'low'], data.loc[i, 'high'])

            # 添加一些技术指标
            data['returns'] = data['close'].pct_change()
            data['ma5'] = data['close'].rolling(window=5, min_periods=1).mean()
            data['ma20'] = data['close'].rolling(window=20, min_periods=1).mean() if n_days >= 20 else data['close']

            # 计算波动率
            if len(data) >= 10:
                data['volatility'] = data['returns'].rolling(window=10, min_periods=1).std() * np.sqrt(252)
            else:
                data['volatility'] = config['volatility']

            logger.info(f"🎭 生成模拟基准数据 {config['name']}: {len(data)} 条记录")
            logger.info(f"   价格范围: {data['close'].min():.2f} - {data['close'].max():.2f}")
            logger.info(f"   平均收益率: {data['returns'].mean():.4f}")
            logger.info(f"   波动率: {data['volatility'].iloc[-1]:.4f}")

            return data

        except Exception as e:
            logger.error(f"生成模拟基准数据失败: {type(e).__name__}: {e}")
            return pd.DataFrame()

    def is_simulated_data_enabled(self) -> bool:
        """检查是否启用模拟数据"""
        import os
        from dotenv import load_dotenv
        load_dotenv()

        return os.getenv('MOCK_DATA_ENABLED', 'false').lower() == 'true'

    def get_benchmark_info(self, symbol: str) -> Dict:
        """获取基准信息"""
        return self.benchmark_configs.get(symbol, {
            'name': f'基准{symbol}',
            'symbol': symbol,
            'initial_price': 1000.0,
            'volatility': 0.018,
            'trend': 0.0002
        })

# 全局实例
mock_generator = MockBenchmarkGenerator()