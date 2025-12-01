"""
策略参数优化模块
Strategy Parameter Optimizer

提供针对不同网格策略的参数优化功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.model_selection import ParameterGrid
from scipy.optimize import minimize
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import time

from src.core.backtest_engine import BacktestEngine, BacktestConfig
from src.strategies.grid_strategy import create_grid_strategy
from src.analysis.metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """策略参数优化器"""

    def __init__(self, max_workers: int = 4):
        """
        初始化策略优化器

        Args:
            max_workers: 最大并行工作线程数
        """
        self.max_workers = max_workers
        self.metrics_calculator = MetricsCalculator()

    def get_strategy_param_space(self, strategy_type: str) -> Dict[str, List[Any]]:
        """
        获取策略参数搜索空间

        Args:
            strategy_type: 策略类型

        Returns:
            参数搜索空间字典
        """
        if strategy_type == 'basic_grid':
            return {
                'grid_count': [6, 8, 10, 12, 16, 20],
                'position_size': [500, 1000, 1500, 2000, 2500],
                'buy_percentage': [0.5, 0.8, 1.0, 1.5, 2.0],
                'sell_percentage': [0.3, 0.5, 0.8, 1.0, 1.5]
            }

        elif strategy_type == 'dynamic_grid':
            return {
                'grid_count': [6, 8, 10, 12, 16],
                'position_size': [500, 1000, 1500, 2000],
                'buy_percentage': [0.5, 0.8, 1.0, 1.5],
                'sell_percentage': [0.3, 0.5, 0.8, 1.0],
                'volatility_window': [10, 15, 20, 25, 30],
                'volatility_threshold': [0.01, 0.015, 0.02, 0.025, 0.03]
            }

        elif strategy_type == 'martingale_grid':
            return {
                'grid_count': [6, 8, 10, 12],
                'position_size': [500, 800, 1000, 1200],
                'buy_percentage': [0.5, 0.8, 1.0, 1.2, 1.5],
                'sell_percentage': [0.3, 0.5, 0.8, 1.0],
                'martingale_factor': [1.5, 2.0, 2.5],
                'max_martingale_levels': [3, 4, 5, 6]
            }

        else:
            raise ValueError(f"未知的策略类型: {strategy_type}")

    def optimize_strategy_params(self, strategy_type: str, base_config: Dict,
                               market_data: pd.DataFrame, symbol: str,
                               optimization_method: str = 'grid_search',
                               max_iterations: int = 50) -> Dict[str, Any]:
        """
        优化策略参数

        Args:
            strategy_type: 策略类型
            base_config: 基础配置
            market_data: 市场数据
            symbol: ETF代码
            optimization_method: 优化方法 ('grid_search', 'random_search', 'bayesian')
            max_iterations: 最大迭代次数

        Returns:
            优化结果
        """
        try:
            logger.info(f"开始优化 {strategy_type} 策略参数，ETF: {symbol}")

            if optimization_method == 'grid_search':
                return self._grid_search_optimization(
                    strategy_type, base_config, market_data, symbol
                )
            elif optimization_method == 'random_search':
                return self._random_search_optimization(
                    strategy_type, base_config, market_data, symbol, max_iterations
                )
            else:
                return self._grid_search_optimization(
                    strategy_type, base_config, market_data, symbol
                )

        except Exception as e:
            logger.error(f"策略参数优化失败: {e}")
            return self._get_optimization_error_result(symbol, strategy_type, str(e))

    def _grid_search_optimization(self, strategy_type: str, base_config: Dict,
                                 market_data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """网格搜索优化"""
        param_space = self.get_strategy_param_space(strategy_type)
        param_grid = list(ParameterGrid(param_space))

        logger.info(f"网格搜索空间大小: {len(param_grid)}")

        best_params = None
        best_score = -float('inf')
        best_metrics = None
        results_summary = []

        start_time = time.time()

        # 并行处理参数组合
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for params in param_grid:
                future = executor.submit(
                    self._evaluate_params, strategy_type, base_config,
                    params, market_data, symbol
                )
                futures.append((future, params))

            for future, params in futures:
                try:
                    score, metrics = future.result()
                    results_summary.append({
                        'params': params,
                        'score': score,
                        'metrics': metrics
                    })

                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_metrics = metrics

                except Exception as e:
                    logger.warning(f"参数评估失败 {params}: {e}")

        execution_time = time.time() - start_time

        return {
            'success': True,
            'symbol': symbol,
            'strategy_type': strategy_type,
            'best_params': best_params,
            'best_score': best_score,
            'best_metrics': best_metrics,
            'total_combinations': len(param_grid),
            'execution_time': execution_time,
            'optimization_method': 'grid_search',
            'results_summary': results_summary
        }

    def _random_search_optimization(self, strategy_type: str, base_config: Dict,
                                   market_data: pd.DataFrame, symbol: str,
                                   max_iterations: int) -> Dict[str, Any]:
        """随机搜索优化"""
        param_space = self.get_strategy_param_space(strategy_type)
        results_summary = []

        best_params = None
        best_score = -float('inf')
        best_metrics = None

        start_time = time.time()

        for iteration in range(max_iterations):
            # 随机采样参数
            params = {}
            for param_name, param_values in param_space.items():
                params[param_name] = np.random.choice(param_values)

            try:
                score, metrics = self._evaluate_params(
                    strategy_type, base_config, params, market_data, symbol
                )

                results_summary.append({
                    'params': params,
                    'score': score,
                    'metrics': metrics
                })

                if score > best_score:
                    best_score = score
                    best_params = params
                    best_metrics = metrics

            except Exception as e:
                logger.warning(f"随机搜索第{iteration}次迭代失败: {e}")

        execution_time = time.time() - start_time

        return {
            'success': True,
            'symbol': symbol,
            'strategy_type': strategy_type,
            'best_params': best_params,
            'best_score': best_score,
            'best_metrics': best_metrics,
            'total_iterations': max_iterations,
            'execution_time': execution_time,
            'optimization_method': 'random_search',
            'results_summary': results_summary
        }

    def _evaluate_params(self, strategy_type: str, base_config: Dict,
                        params: Dict, market_data: pd.DataFrame, symbol: str) -> Tuple[float, Dict]:
        """
        评估参数组合

        Args:
            strategy_type: 策略类型
            base_config: 基础配置
            params: 要评估的参数
            market_data: 市场数据
            symbol: ETF代码

        Returns:
            (评分, 指标字典)
        """
        # 合并配置
        config = base_config.copy()
        config.update(params)

        # 创建策略
        strategy = create_grid_strategy(strategy_type, config)

        # 创建回测引擎
        backtest_config = BacktestConfig(
            initial_cash=100000.0,
            commission_rate=0.0003,
            slippage_rate=0.001
        )
        engine = BacktestEngine(backtest_config)

        # 运行回测
        backtest_results = engine.run_backtest(strategy, market_data, symbol)

        # 计算指标
        metrics = self.metrics_calculator.calculate_comprehensive_metrics(
            backtest_results, market_data
        )

        # 计算综合评分（夏普比率为主，结合其他指标）
        basic_metrics = metrics.get('basic_metrics', {})
        sharpe_ratio = basic_metrics.get('sharpe_ratio', 0)
        total_return = basic_metrics.get('total_return', 0)
        max_drawdown = basic_metrics.get('max_drawdown', 1)

        # 综合评分：夏普比率 * 0.5 + 年化收益率 * 0.3 - 最大回撤 * 0.2
        annual_return = basic_metrics.get('annual_return', 0)
        score = sharpe_ratio * 0.5 + annual_return * 0.3 - max_drawdown * 0.2

        return score, metrics

    def _get_optimization_error_result(self, symbol: str, strategy_type: str, error_msg: str) -> Dict:
        """获取优化错误结果"""
        return {
            'success': False,
            'symbol': symbol,
            'strategy_type': strategy_type,
            'error': error_msg,
            'best_params': None,
            'best_score': -float('inf'),
            'best_metrics': None
        }

    def optimize_multiple_strategies(self, strategy_types: List[str], base_config: Dict,
                                   market_data: pd.DataFrame, symbol: str,
                                   optimization_method: str = 'grid_search') -> Dict[str, Any]:
        """
        优化多种策略参数

        Args:
            strategy_types: 策略类型列表
            base_config: 基础配置
            market_data: 市场数据
            symbol: ETF代码
            optimization_method: 优化方法

        Returns:
            多策略优化结果
        """
        logger.info(f"开始多策略参数优化，ETF: {symbol}，策略: {strategy_types}")

        results = {}
        start_time = time.time()

        for strategy_type in strategy_types:
            try:
                result = self.optimize_strategy_params(
                    strategy_type, base_config, market_data, symbol, optimization_method
                )
                results[strategy_type] = result
            except Exception as e:
                logger.error(f"策略 {strategy_type} 优化失败: {e}")
                results[strategy_type] = self._get_optimization_error_result(
                    symbol, strategy_type, str(e)
                )

        execution_time = time.time() - start_time

        # 找出最佳策略
        best_strategy = None
        best_score = -float('inf')

        for strategy_type, result in results.items():
            if result.get('success', False):
                score = result.get('best_score', -float('inf'))
                if score > best_score:
                    best_score = score
                    best_strategy = strategy_type

        return {
            'symbol': symbol,
            'strategy_types': strategy_types,
            'individual_results': results,
            'best_strategy': best_strategy,
            'best_score': best_score,
            'total_execution_time': execution_time,
            'optimization_method': optimization_method
        }

    def get_optimization_recommendations(self, optimization_results: Dict) -> Dict[str, Any]:
        """
        基于优化结果生成建议

        Args:
            optimization_results: 优化结果

        Returns:
            优化建议
        """
        recommendations = []

        if not optimization_results.get('success', False):
            return {'error': '优化结果无效'}

        best_params = optimization_results.get('best_params', {})
        strategy_type = optimization_results.get('strategy_type')
        best_metrics = optimization_results.get('best_metrics', {})

        # 根据策略类型生成具体建议
        if strategy_type == 'basic_grid':
            grid_count = best_params.get('grid_count', 10)
            if grid_count <= 8:
                recommendations.append("网格数量较少，适合波动较小的ETF")
            elif grid_count >= 16:
                recommendations.append("网格数量较多，适合波动较大的ETF，但要注意交易成本")

            position_size = best_params.get('position_size', 1000)
            if position_size >= 2000:
                recommendations.append("单笔交易金额较大，请确保资金充足")

        elif strategy_type == 'dynamic_grid':
            vol_threshold = best_params.get('volatility_threshold', 0.02)
            if vol_threshold >= 0.025:
                recommendations.append("波动率阈值较高，网格调整较为保守")
            elif vol_threshold <= 0.015:
                recommendations.append("波动率阈值较低，网格调整较为敏感")

        elif strategy_type == 'martingale_grid':
            martingale_factor = best_params.get('martingale_factor', 2.0)
            max_levels = best_params.get('max_martingale_levels', 5)
            if martingale_factor >= 2.5:
                recommendations.append("马丁格尔系数较高，风险较大，请谨慎使用")
            if max_levels >= 6:
                recommendations.append("最大马丁格尔层数较多，需要充足的资金支持")

        # 风险提示
        basic_metrics = best_metrics.get('basic_metrics', {})
        max_drawdown = basic_metrics.get('max_drawdown', 0)
        if max_drawdown > 0.25:
            recommendations.append("最大回撤超过25%，建议降低仓位或调整参数")

        return {
            'strategy_type': strategy_type,
            'best_params': best_params,
            'recommendations': recommendations,
            'risk_assessment': self._assess_optimization_risk(best_metrics)
        }

    def _assess_optimization_risk(self, metrics: Dict) -> str:
        """评估优化结果的风险"""
        try:
            basic_metrics = metrics.get('basic_metrics', {})
            max_drawdown = basic_metrics.get('max_drawdown', 0)
            sharpe_ratio = basic_metrics.get('sharpe_ratio', 0)

            if max_drawdown > 0.3:
                return "高风险"
            elif max_drawdown > 0.2 or sharpe_ratio < 0.5:
                return "中等风险"
            else:
                return "低风险"

        except:
            return "无法评估"


# 使用示例和测试函数
if __name__ == "__main__":
    # 这里可以添加测试代码
    pass