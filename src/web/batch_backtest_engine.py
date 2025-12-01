"""
批量回测引擎
Batch Backtest Engine

负责并行处理多个ETF的网格交易回测，统一收集结果和优化建议
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import time

from src.data.grid_config_parser import GridConfigParser
from src.data.market_data_fetcher import MarketDataFetcher
from src.strategies.grid_strategy import create_grid_strategy
from src.core.backtest_engine import BacktestEngine, BacktestConfig
from src.analysis.metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)


class BatchBacktestEngine:
    """批量回测引擎"""

    def __init__(self, max_workers: int = 4):
        """
        初始化批量回测引擎

        Args:
            max_workers: 最大并行工作线程数
        """
        self.max_workers = max_workers
        self.parser = GridConfigParser()
        self.data_fetcher = MarketDataFetcher()
        self.metrics_calculator = MetricsCalculator()

    def run_batch_backtest(self, config_data: pd.DataFrame,
                          progress_callback=None, strategy_types: List[str] = None) -> Dict:
        """
        运行批量回测

        Args:
            config_data: ETF配置数据
            progress_callback: 进度回调函数
            strategy_types: 要比较的策略类型列表，默认为 ['basic_grid']

        Returns:
            批量回测结果
        """
        if strategy_types is None:
            strategy_types = ['basic_grid']

        logger.info(f"开始批量回测，共{len(config_data)}个ETF，策略类型: {strategy_types}")

        start_time = time.time()
        results = {}

        try:
            # 获取所有ETF代码
            symbols = config_data['stock_code'].tolist()
            configs = config_data.to_dict('records')

            # 并行处理ETF回测
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有回测任务
                if len(strategy_types) == 1:
                    # 单策略模式（原有逻辑）
                    future_to_symbol = {
                        executor.submit(self._run_single_backtest, config, strategy_types[0]): config['stock_code']
                        for config in configs
                    }
                else:
                    # 多策略比较模式
                    future_to_symbol = {}
                    for config in configs:
                        for strategy_type in strategy_types:
                            future = executor.submit(
                                self._run_single_backtest, config, strategy_type
                            )
                            future_key = (config['stock_code'], strategy_type)
                            future_to_symbol[future] = future_key

                # 收集结果
                completed = 0
                total = len(future_to_symbol)

                for future in concurrent.futures.as_completed(future_to_symbol):
                    key = future_to_symbol[future]

                    try:
                        result = future.result()

                        if len(strategy_types) == 1:
                            # 单策略模式
                            symbol = key
                            results[symbol] = result
                            status_msg = f"完成 {symbol}"
                        else:
                            # 多策略模式
                            symbol, strategy_type = key
                            if symbol not in results:
                                results[symbol] = {'success': False, 'strategies': {}}

                            results[symbol]['strategies'][strategy_type] = result
                            results[symbol]['success'] = True
                            status_msg = f"完成 {symbol} - {strategy_type}"

                        completed += 1

                        if progress_callback:
                            progress_callback(completed, total, status_msg)

                    except Exception as e:
                        logger.error(f"回测失败 {key}: {e}")

                        if len(strategy_types) == 1:
                            results[key] = self._get_error_result(key, str(e))
                        else:
                            symbol, strategy_type = key
                            if symbol not in results:
                                results[symbol] = {'success': False, 'strategies': {}}

                            results[symbol]['strategies'][strategy_type] = self._get_error_result(
                                f"{symbol}_{strategy_type}", str(e)
                            )

                        completed += 1

                        if progress_callback:
                            progress_callback(completed, total, f"失败 {key}")

            # 生成汇总统计
            if len(strategy_types) == 1:
                summary_stats = self._generate_summary_stats(results)
            else:
                summary_stats = self._generate_strategy_comparison_stats(results, strategy_types)

            execution_time = time.time() - start_time
            logger.info(f"批量回测完成，耗时 {execution_time:.2f} 秒")

            return {
                'individual_results': results,
                'summary_stats': summary_stats,
                'execution_time': execution_time,
                'total_etfs': len(config_data),
                'successful_etfs': len([r for r in results.values() if r.get('success', False)]),
                'failed_etfs': len([r for r in results.values() if not r.get('success', False)])
            }

        except Exception as e:
            logger.error(f"批量回测失败: {e}")
            return {
                'individual_results': {},
                'summary_stats': {},
                'execution_time': time.time() - start_time,
                'error': str(e)
            }

    def _run_single_backtest(self, config: Dict, strategy_type: str = 'basic_grid') -> Dict:
        """
        运行单个ETF的回测

        Args:
            config: ETF配置

        Returns:
            单个ETF的回测结果
        """
        symbol = config.get('stock_code')
        stock_name = config.get('stock_name', 'Unknown')

        try:
            logger.debug(f"开始回测 {stock_name} ({symbol})")

            # 1. 获取市场数据
            market_data = self.data_fetcher.fetch_etf_data(symbol)
            if market_data.empty:
                return self._get_error_result(symbol, f"无法获取 {symbol} 的市场数据")

            # 2. 创建策略配置
            strategy_config = {
                'base_price': config['base_price'],
                'sell_percentage': config['sell_percentage'],
                'buy_percentage': config['buy_percentage'],
                'position_size': config.get('buy_position_size', 1000),
                'grid_count': 8,
                'position_size_type': 'shares'
            }

            # 3. 创建回测配置
            backtest_config = BacktestConfig(
                initial_cash=100000.0,
                commission_rate=0.0003,
                slippage_rate=0.001
            )

            # 4. 创建网格策略和回测引擎
            strategy = create_grid_strategy(strategy_type, strategy_config)
            engine = BacktestEngine(backtest_config)

            # 5. 运行回测
            backtest_results = engine.run_backtest(strategy, market_data, symbol)

            # 6. 计算完整指标
            metrics = self.metrics_calculator.calculate_comprehensive_metrics(
                backtest_results, market_data
            )

            # 7. 生成优化建议
            optimization_suggestions = self._generate_optimization_suggestions(
                config, metrics, market_data
            )

            # 8. 风险评级
            risk_rating = self._calculate_risk_rating(metrics)

            result = {
                'success': True,
                'symbol': symbol,
                'stock_name': stock_name,
                'config': config,
                'backtest_results': backtest_results,
                'metrics': metrics,
                'optimization_suggestions': optimization_suggestions,
                'risk_rating': risk_rating,
                'data_quality': self._assess_data_quality(market_data),
                'backtest_time': datetime.now().isoformat()
            }

            logger.debug(f"完成回测 {stock_name} ({symbol})")
            return result

        except Exception as e:
            logger.error(f"ETF {symbol} 回测异常: {e}")
            return self._get_error_result(symbol, str(e))

    def _generate_optimization_suggestions(self, config: Dict, metrics: Dict,
                                         market_data: pd.DataFrame) -> List[str]:
        """生成优化建议"""
        suggestions = []

        try:
            basic_metrics = metrics.get('basic_metrics', {})
            trading_metrics = metrics.get('trading_metrics', {})
            risk_metrics = metrics.get('risk_metrics', {})

            current_sell = config.get('sell_percentage', 5.0)
            current_buy = config.get('buy_percentage', 10.0)
            sell_buy_ratio = current_buy / current_sell

            # 参数优化建议
            if sell_buy_ratio > 2.5:
                suggestions.append(f"建议降低买入网格间距，当前买入/卖出比例为1:{sell_buy_ratio:.1f}，建议调整至1:1.5-2.0")
            elif sell_buy_ratio < 1.2:
                suggestions.append(f"建议增加买入网格间距，当前买入/卖出比例为1:{sell_buy_ratio:.1f}，建议调整至1:1.5-2.0")

            # 基于波动率的建议
            volatility = risk_metrics.get('volatility', 0.15)
            if volatility > 0.25:  # 高波动
                suggestions.append(f"该ETF波动率较高({volatility:.1%})，建议缩小网格间距至{current_sell*0.8:.1f}%/{current_buy*0.8:.1f}%")
            elif volatility < 0.10:  # 低波动
                suggestions.append(f"该ETF波动率较低({volatility:.1%})，建议扩大网格间距至{current_sell*1.2:.1f}%/{current_buy*1.2:.1f}%")

            # 夏普比率建议
            sharpe_ratio = basic_metrics.get('sharpe_ratio', 0)
            if sharpe_ratio < 0.5:
                suggestions.append("夏普比率较低，建议优化网格参数或考虑其他策略")
            elif sharpe_ratio > 2.0:
                suggestions.append("当前参数表现优秀，可以继续使用")

            # 交易频率建议
            trading_frequency = trading_metrics.get('trading_frequency', 0)
            if trading_frequency > 0.1:  # 交易过于频繁
                suggestions.append("交易频率过高，建议扩大网格间距以降低交易成本")
            elif trading_frequency < 0.01:  # 交易过少
                suggestions.append("交易频率过低，建议缩小网格间距以提高资金利用率")

            # 最大回撤建议
            max_drawdown = basic_metrics.get('max_drawdown', 0)
            if max_drawdown < -0.2:
                suggestions.append(f"最大回撤较大({max_drawdown:.1%})，建议降低仓位或增加网格层次")

            # 胜率建议
            win_rate = trading_metrics.get('win_rate', 0)
            if win_rate < 0.5:
                suggestions.append("胜率较低，建议调整网格间距或等待更好的入场时机")

            if not suggestions:
                suggestions.append("当前网格参数设置较为合理，建议继续观察市场表现")

            return suggestions

        except Exception as e:
            logger.error(f"生成优化建议失败: {e}")
            return ["无法生成优化建议"]

    def _calculate_risk_rating(self, metrics: Dict) -> str:
        """计算风险评级"""
        try:
            basic_metrics = metrics.get('basic_metrics', {})
            risk_metrics = metrics.get('risk_metrics', {})

            sharpe_ratio = basic_metrics.get('sharpe_ratio', 0)
            max_drawdown = abs(basic_metrics.get('max_drawdown', 0))
            volatility = risk_metrics.get('volatility', 0.15)

            # 风险评分 (0-100, 越低风险越小)
            risk_score = 0

            # 夏普比率评分 (0-30分)
            if sharpe_ratio > 2:
                risk_score += 30
            elif sharpe_ratio > 1:
                risk_score += 20
            elif sharpe_ratio > 0.5:
                risk_score += 10

            # 最大回撤评分 (0-40分)
            if max_drawdown < 0.05:
                risk_score += 40
            elif max_drawdown < 0.1:
                risk_score += 30
            elif max_drawdown < 0.15:
                risk_score += 20
            elif max_drawdown < 0.2:
                risk_score += 10

            # 波动率评分 (0-30分)
            if volatility < 0.1:
                risk_score += 30
            elif volatility < 0.15:
                risk_score += 20
            elif volatility < 0.2:
                risk_score += 10

            # 风险评级
            if risk_score >= 80:
                return "低风险 AAAAA"
            elif risk_score >= 60:
                return "中等风险 AAAA"
            elif risk_score >= 40:
                return "中高风险 AAA"
            elif risk_score >= 20:
                return "高风险 AA"
            else:
                return "极高风险 A"

        except Exception as e:
            logger.error(f"计算风险评级失败: {e}")
            return "未知风险 ⭐"

    def _assess_data_quality(self, market_data: pd.DataFrame) -> Dict:
        """评估数据质量"""
        try:
            if market_data.empty:
                return {'quality_score': 0, 'issues': ['无数据']}

            issues = []
            score = 100

            # 检查数据完整性
            missing_data = market_data.isnull().sum().sum()
            if missing_data > 0:
                score -= missing_data * 2
                issues.append(f'缺失数据: {missing_data}个')

            # 检查数据长度
            if len(market_data) < 100:
                score -= 20
                issues.append(f'数据长度不足: {len(market_data)}条')

            # 检查价格合理性
            price_changes = market_data['close'].pct_change()
            extreme_changes = (abs(price_changes) > 0.1).sum()
            if extreme_changes > len(market_data) * 0.01:  # 超过1%的异常变化
                score -= extreme_changes * 5
                issues.append(f'异常价格变化: {extreme_changes}次')

            # 检查连续性
            expected_dates = pd.date_range(market_data['date'].min(),
                                         market_data['date'].max(),
                                         freq='D')
            if len(expected_dates) != len(market_data):
                missing_days = len(expected_dates) - len(market_data)
                score -= missing_days * 2
                issues.append(f'缺失交易日: {missing_days}天')

            return {
                'quality_score': max(0, score),
                'issues': issues,
                'data_points': len(market_data),
                'date_range': f"{market_data['date'].min().date()} 至 {market_data['date'].max().date()}"
            }

        except Exception as e:
            logger.error(f"评估数据质量失败: {e}")
            return {'quality_score': 0, 'issues': ['评估失败']}

    def _generate_summary_stats(self, results: Dict) -> Dict:
        """生成汇总统计"""
        try:
            successful_results = [r for r in results.values() if r.get('success', False)]

            if not successful_results:
                return {}

            # 收集所有指标
            all_metrics = []
            for result in successful_results:
                metrics = result.get('metrics', {})
                flattened_metrics = self.metrics_calculator._flatten_metrics(metrics)
                flattened_metrics['symbol'] = result.get('symbol')
                flattened_metrics['stock_name'] = result.get('stock_name')
                all_metrics.append(flattened_metrics)

            df_metrics = pd.DataFrame(all_metrics)

            summary = {
                'total_etfs': len(results),
                'successful_etfs': len(successful_results),
                'success_rate': len(successful_results) / len(results),
                'avg_total_return': df_metrics['total_return'].mean(),
                'avg_annual_return': df_metrics['annual_return'].mean(),
                'avg_max_drawdown': df_metrics['max_drawdown'].mean(),
                'avg_sharpe_ratio': df_metrics['sharpe_ratio'].mean(),
                'avg_win_rate': df_metrics['win_rate'].mean(),
                'best_performer': self._find_best_performer(df_metrics),
                'worst_performer': self._find_worst_performer(df_metrics),
                'risk_distribution': self._calculate_risk_distribution(df_metrics)
            }

            return summary

        except Exception as e:
            logger.error(f"生成汇总统计失败: {e}")
            return {}

    def _find_best_performer(self, df_metrics: pd.DataFrame) -> Dict:
        """找出表现最好的ETF"""
        try:
            best_idx = df_metrics['sharpe_ratio'].idxmax()
            return {
                'symbol': df_metrics.loc[best_idx, 'symbol'],
                'stock_name': df_metrics.loc[best_idx, 'stock_name'],
                'sharpe_ratio': df_metrics.loc[best_idx, 'sharpe_ratio'],
                'total_return': df_metrics.loc[best_idx, 'total_return']
            }
        except Exception as e:
            logger.error(f"找出最佳表现者失败: {e}")
            return {}

    def _find_worst_performer(self, df_metrics: pd.DataFrame) -> Dict:
        """找出表现最差的ETF"""
        try:
            worst_idx = df_metrics['sharpe_ratio'].idxmin()
            return {
                'symbol': df_metrics.loc[worst_idx, 'symbol'],
                'stock_name': df_metrics.loc[worst_idx, 'stock_name'],
                'sharpe_ratio': df_metrics.loc[worst_idx, 'sharpe_ratio'],
                'total_return': df_metrics.loc[worst_idx, 'total_return']
            }
        except Exception as e:
            logger.error(f"找出最差表现者失败: {e}")
            return {}

    def _calculate_risk_distribution(self, df_metrics: pd.DataFrame) -> Dict:
        """计算风险分布"""
        try:
            risk_counts = {
                '低风险': 0,
                '中等风险': 0,
                '中高风险': 0,
                '高风险': 0,
                '极高风险': 0
            }

            # 根据最大回撤分类风险
            for _, row in df_metrics.iterrows():
                max_dd = abs(row.get('max_drawdown', 0))
                if max_dd < 0.05:
                    risk_counts['低风险'] += 1
                elif max_dd < 0.1:
                    risk_counts['中等风险'] += 1
                elif max_dd < 0.15:
                    risk_counts['中高风险'] += 1
                elif max_dd < 0.2:
                    risk_counts['高风险'] += 1
                else:
                    risk_counts['极高风险'] += 1

            total = sum(risk_counts.values())
            if total > 0:
                return {k: v/total for k, v in risk_counts.items()}
            else:
                return risk_counts

        except Exception as e:
            logger.error(f"计算风险分布失败: {e}")
            return {}

    def _get_error_result(self, symbol: str, error_message: str) -> Dict:
        """获取错误结果"""
        return {
            'success': False,
            'symbol': symbol,
            'error': error_message,
            'backtest_time': datetime.now().isoformat()
        }

    def export_results_to_csv(self, results: Dict, output_path: str):
        """导出结果到CSV"""
        try:
            if not results or 'individual_results' not in results:
                logger.error("没有结果数据可导出")
                return

            individual_results = results['individual_results']
            successful_results = [r for r in individual_results.values() if r.get('success', False)]

            if not successful_results:
                logger.error("没有成功的结果可导出")
                return

            # 准备导出数据
            export_data = []
            for result in successful_results:
                config = result.get('config', {})
                metrics = result.get('metrics', {})
                basic_metrics = metrics.get('basic_metrics', {})
                trading_metrics = metrics.get('trading_metrics', {})
                risk_metrics = metrics.get('risk_metrics', {})

                row = {
                    'ETF名称': result.get('stock_name', ''),
                    'ETF代码': result.get('symbol', ''),
                    '当前设置': f"卖出{config.get('sell_percentage', 0):.1f}%/买入{config.get('buy_percentage', 0):.1f}%",
                    '总收益率': basic_metrics.get('total_return', 0),
                    '年化收益率': basic_metrics.get('annual_return', 0),
                    '最大回撤': basic_metrics.get('max_drawdown', 0),
                    '夏普比率': basic_metrics.get('sharpe_ratio', 0),
                    '胜率': trading_metrics.get('win_rate', 0),
                    '交易次数': trading_metrics.get('total_trades', 0),
                    '平均持仓天数': trading_metrics.get('avg_holding_period', 0),
                    '总交易成本': trading_metrics.get('total_commission', 0),
                    'VaR(95%)': risk_metrics.get('var_95', 0),
                    '年化波动率': risk_metrics.get('volatility', 0),
                    '最大连续亏损': risk_metrics.get('max_consecutive_losses', 0),
                    '贝塔系数': risk_metrics.get('beta', 1.0),
                    '优化建议': '; '.join(result.get('optimization_suggestions', [])),
                    '风险评级': result.get('risk_rating', '未知'),
                    '数据质量评分': result.get('data_quality', {}).get('quality_score', 0)
                }
                export_data.append(row)

            # 导出到CSV
            df_export = pd.DataFrame(export_data)
            df_export.to_csv(output_path, index=False, encoding='utf-8-sig')

            logger.info(f"结果已导出到: {output_path}")

        except Exception as e:
            logger.error(f"导出结果失败: {e}")

    def _generate_strategy_comparison_stats(self, results: Dict, strategy_types: List[str]) -> Dict:
        """
        生成策略比较统计

        Args:
            results: 多策略回测结果
            strategy_types: 策略类型列表

        Returns:
            策略比较统计信息
        """
        try:
            successful_results = [r for r in results.values() if r.get('success', False)]

            if not successful_results:
                return {}

            # 收集每种策略的指标
            strategy_metrics = {}
            for strategy_type in strategy_types:
                strategy_metrics[strategy_type] = []

            for result in successful_results:
                strategies = result.get('strategies', {})
                for strategy_type, strategy_result in strategies.items():
                    if strategy_result.get('success', False):
                        metrics = strategy_result.get('metrics', {})
                        flattened_metrics = self.metrics_calculator._flatten_metrics(metrics)
                        flattened_metrics['symbol'] = result.get('symbol')
                        flattened_metrics['strategy_type'] = strategy_type
                        strategy_metrics[strategy_type].append(flattened_metrics)

            # 为每种策略生成统计
            strategy_stats = {}
            for strategy_type in strategy_types:
                if strategy_metrics[strategy_type]:
                    df_strategy = pd.DataFrame(strategy_metrics[strategy_type])
                    strategy_stats[strategy_type] = {
                        'count': len(df_strategy),
                        'avg_total_return': df_strategy['total_return'].mean(),
                        'avg_annual_return': df_strategy['annual_return'].mean(),
                        'avg_max_drawdown': df_strategy['max_drawdown'].mean(),
                        'avg_sharpe_ratio': df_strategy['sharpe_ratio'].mean(),
                        'avg_win_rate': df_strategy['win_rate'].mean(),
                        'best_performer': self._find_best_performer(df_strategy),
                        'worst_performer': self._find_worst_performer(df_strategy)
                    }

            # 找出每个ETF的最佳策略
            etf_best_strategies = {}
            for result in successful_results:
                symbol = result.get('symbol')
                strategies = result.get('strategies', {})
                best_strategy = None
                best_sharpe = -float('inf')

                for strategy_type, strategy_result in strategies.items():
                    if strategy_result.get('success', False):
                        metrics = strategy_result.get('metrics', {})
                        basic_metrics = metrics.get('basic_metrics', {})
                        sharpe = basic_metrics.get('sharpe_ratio', 0)

                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_strategy = strategy_type

                if best_strategy:
                    etf_best_strategies[symbol] = {
                        'strategy': best_strategy,
                        'sharpe_ratio': best_sharpe
                    }

            # 统计各策略被推荐为最佳策略的次数
            strategy_recommendations = {}
            for symbol, best_info in etf_best_strategies.items():
                strategy = best_info['strategy']
                strategy_recommendations[strategy] = strategy_recommendations.get(strategy, 0) + 1

            return {
                'total_etfs': len(results),
                'successful_etfs': len(successful_results),
                'success_rate': len(successful_results) / len(results),
                'strategy_types': strategy_types,
                'strategy_stats': strategy_stats,
                'etf_best_strategies': etf_best_strategies,
                'strategy_recommendations': strategy_recommendations,
                'overall_best_strategy': max(strategy_recommendations.items(), key=lambda x: x[1])[0] if strategy_recommendations else None
            }

        except Exception as e:
            logger.error(f"生成策略比较统计失败: {e}")
            return {}

    def get_strategy_comparison_for_app(self, results: Dict) -> Dict:
        """
        为应用层准备策略比较数据

        Args:
            results: 批量回测结果

        Returns:
            应用层友好的策略比较数据
        """
        try:
            summary_stats = results.get('summary_stats', {})
            individual_results = results.get('individual_results', {})

            # 准备表格数据
            table_data = []
            for symbol, result in individual_results.items():
                if not result.get('success', False):
                    continue

                strategies = result.get('strategies', {})
                for strategy_type, strategy_result in strategies.items():
                    if strategy_result.get('success', False):
                        metrics = strategy_result.get('metrics', {})
                        basic_metrics = metrics.get('basic_metrics', {})
                        trading_metrics = metrics.get('trading_metrics', {})

                        row = {
                            'stock_code': symbol,
                            'stock_name': strategy_result.get('stock_name', 'Unknown'),
                            'strategy_type': strategy_type,
                            'total_return': basic_metrics.get('total_return', 0),
                            'annual_return': basic_metrics.get('annual_return', 0),
                            'max_drawdown': basic_metrics.get('max_drawdown', 0),
                            'sharpe_ratio': basic_metrics.get('sharpe_ratio', 0),
                            'win_rate': trading_metrics.get('win_rate', 0),
                            'total_trades': trading_metrics.get('total_trades', 0)
                        }
                        table_data.append(row)

            return {
                'summary_stats': summary_stats,
                'table_data': table_data,
                'strategy_comparison_available': 'strategy_stats' in summary_stats
            }

        except Exception as e:
            logger.error(f"准备策略比较数据失败: {e}")
            return {'error': str(e)}


def main():
    """测试函数"""
    # 创建测试数据
    config_data = pd.DataFrame([
        {
            'stock_code': '159682',
            'stock_name': '科创50',
            'base_price': 1.408,
            'sell_percentage': 5.0,
            'buy_percentage': 10.0,
            'buy_position_size': 1000
        }
    ])

    # 创建批量回测引擎
    engine = BatchBacktestEngine(max_workers=2)

    # 定义进度回调
    def progress_callback(completed, total, message):
        print(f"进度: {completed}/{total} - {message}")

    # 运行批量回测
    results = engine.run_batch_backtest(config_data, progress_callback)

    # 输出结果
    print("批量回测结果:")
    print(f"总ETF数: {results['total_etfs']}")
    print(f"成功数: {results['successful_etfs']}")
    print(f"失败数: {results['failed_etfs']}")
    print(f"执行时间: {results['execution_time']:.2f}秒")

    # 导出结果
    engine.export_results_to_csv(results, 'batch_backtest_results.csv')


if __name__ == "__main__":
    main()
