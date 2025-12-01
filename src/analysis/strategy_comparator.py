"""
策略分析比较器
Strategy Analyzer and Comparator

提供策略性能比较、排名和推荐功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class StrategyComparator:
    """策略分析比较器"""

    def __init__(self):
        """初始化策略比较器"""
        self.strategy_display_names = {
            'basic_grid': '基础网格策略',
            'dynamic_grid': '动态网格策略',
            'martingale_grid': '马丁格尔网格策略'
        }

        self.metric_weights = {
            'total_return': 0.25,
            'sharpe_ratio': 0.30,
            'max_drawdown': 0.20,
            'win_rate': 0.15,
            'trading_frequency': 0.10
        }

    def compare_strategies(self, backtest_results: Dict) -> Dict[str, Any]:
        """
        比较多种策略性能

        Args:
            backtest_results: 批量回测结果

        Returns:
            策略比较结果
        """
        try:
            # 检查是否为多策略比较结果
            summary_stats = backtest_results.get('summary_stats', {})

            if 'strategy_stats' not in summary_stats:
                return self._single_strategy_analysis(backtest_results)

            return self._multi_strategy_analysis(backtest_results)

        except Exception as e:
            logger.error(f"策略比较失败: {e}")
            return {'error': str(e), 'success': False}

    def _single_strategy_analysis(self, backtest_results: Dict) -> Dict[str, Any]:
        """单策略分析"""
        individual_results = backtest_results.get('individual_results', {})

        analysis_data = []
        for symbol, result in individual_results.items():
            if result.get('success', False):
                metrics = result.get('metrics', {})
                basic_metrics = metrics.get('basic_metrics', {})
                trading_metrics = metrics.get('trading_metrics', {})

                analysis_data.append({
                    'symbol': symbol,
                    'stock_name': result.get('stock_name', 'Unknown'),
                    'total_return': basic_metrics.get('total_return', 0),
                    'annual_return': basic_metrics.get('annual_return', 0),
                    'max_drawdown': basic_metrics.get('max_drawdown', 0),
                    'sharpe_ratio': basic_metrics.get('sharpe_ratio', 0),
                    'win_rate': trading_metrics.get('win_rate', 0),
                    'total_trades': trading_metrics.get('total_trades', 0),
                    'strategy_type': 'basic_grid'
                })

        df = pd.DataFrame(analysis_data)

        return {
            'comparison_type': 'single_strategy',
            'strategy_data': analysis_data,
            'summary_stats': self._calculate_strategy_summary(df, 'basic_grid'),
            'best_performers': self._find_best_performers_by_metrics(df),
            'risk_analysis': self._analyze_risk_distribution(df)
        }

    def _multi_strategy_analysis(self, backtest_results: Dict) -> Dict[str, Any]:
        """多策略分析"""
        individual_results = backtest_results.get('individual_results', {})
        summary_stats = backtest_results.get('summary_stats', {})

        # 整理多策略数据
        strategy_data = []
        etf_comparisons = {}

        for symbol, result in individual_results.items():
            if not result.get('success', False):
                continue

            strategies = result.get('strategies', {})
            etf_strategy_data = {}

            for strategy_type, strategy_result in strategies.items():
                if strategy_result.get('success', False):
                    metrics = strategy_result.get('metrics', {})
                    basic_metrics = metrics.get('basic_metrics', {})
                    trading_metrics = metrics.get('trading_metrics', {})

                    data_row = {
                        'symbol': symbol,
                        'stock_name': strategy_result.get('stock_name', 'Unknown'),
                        'strategy_type': strategy_type,
                        'total_return': basic_metrics.get('total_return', 0),
                        'annual_return': basic_metrics.get('annual_return', 0),
                        'max_drawdown': basic_metrics.get('max_drawdown', 0),
                        'sharpe_ratio': basic_metrics.get('sharpe_ratio', 0),
                        'win_rate': trading_metrics.get('win_rate', 0),
                        'total_trades': trading_metrics.get('total_trades', 0),
                        'strategy_score': self._calculate_strategy_score(basic_metrics, trading_metrics)
                    }

                    strategy_data.append(data_row)
                    etf_strategy_data[strategy_type] = data_row

            # 为每个ETF找出最佳策略
            if etf_strategy_data:
                best_strategy = max(etf_strategy_data.items(), key=lambda x: x[1]['strategy_score'])
                etf_comparisons[symbol] = {
                    'best_strategy': best_strategy[0],
                    'best_score': best_strategy[1]['strategy_score'],
                    'all_strategies': etf_strategy_data
                }

        df = pd.DataFrame(strategy_data)

        return {
            'comparison_type': 'multi_strategy',
            'strategy_data': strategy_data,
            'etf_comparisons': etf_comparisons,
            'strategy_summary': self._calculate_multi_strategy_summary(df),
            'best_performers_by_strategy': self._find_best_performers_by_strategy(df),
            'strategy_rankings': self._rank_strategies(df),
            'recommendations': self._generate_strategy_recommendations(df, etf_comparisons),
            'risk_analysis': self._analyze_strategy_risks(df),
            'market_condition_analysis': self._analyze_market_conditions(df)
        }

    def _calculate_strategy_score(self, basic_metrics: Dict, trading_metrics: Dict) -> float:
        """计算策略综合评分"""
        total_return = basic_metrics.get('total_return', 0)
        sharpe_ratio = basic_metrics.get('sharpe_ratio', 0)
        max_drawdown = basic_metrics.get('max_drawdown', 1)
        win_rate = trading_metrics.get('win_rate', 0)
        total_trades = trading_metrics.get('total_trades', 0)

        # 标准化交易频率 (0-1之间)
        trading_score = min(total_trades / 100, 1.0)

        # 综合评分
        score = (
            total_return * self.metric_weights['total_return'] +
            sharpe_ratio * self.metric_weights['sharpe_ratio'] -
            max_drawdown * self.metric_weights['max_drawdown'] +
            win_rate * self.metric_weights['win_rate'] +
            trading_score * self.metric_weights['trading_frequency']
        )

        return score

    def _calculate_strategy_summary(self, df: pd.DataFrame, strategy_type: str) -> Dict:
        """计算策略汇总统计"""
        if df.empty:
            return {}

        return {
            'strategy_type': strategy_type,
            'total_etfs': len(df),
            'avg_total_return': df['total_return'].mean(),
            'avg_annual_return': df['annual_return'].mean(),
            'avg_max_drawdown': df['max_drawdown'].mean(),
            'avg_sharpe_ratio': df['sharpe_ratio'].mean(),
            'avg_win_rate': df['win_rate'].mean(),
            'positive_return_rate': (df['total_return'] > 0).mean(),
            'high_return_rate': (df['total_return'] > 0.15).mean(),
            'return_std': df['total_return'].std(),
            'sharpe_std': df['sharpe_ratio'].std()
        }

    def _calculate_multi_strategy_summary(self, df: pd.DataFrame) -> Dict:
        """计算多策略汇总统计"""
        summary = {}

        for strategy_type in df['strategy_type'].unique():
            strategy_df = df[df['strategy_type'] == strategy_type]
            summary[strategy_type] = self._calculate_strategy_summary(strategy_df, strategy_type)

        return summary

    def _find_best_performers_by_metrics(self, df: pd.DataFrame) -> Dict:
        """找出各项指标的最佳表现者"""
        if df.empty:
            return {}

        best_performers = {}

        metrics = ['total_return', 'sharpe_ratio', 'win_rate']
        for metric in metrics:
            best_idx = df[metric].idxmax()
            best_row = df.loc[best_idx]

            best_performers[f'best_{metric}'] = {
                'symbol': best_row['symbol'],
                'stock_name': best_row['stock_name'],
                'strategy_type': best_row['strategy_type'],
                'value': best_row[metric]
            }

        # 最大回撤最小值
        best_idx = df['max_drawdown'].idxmin()
        best_row = df.loc[best_idx]
        best_performers['lowest_max_drawdown'] = {
            'symbol': best_row['symbol'],
            'stock_name': best_row['stock_name'],
            'strategy_type': best_row['strategy_type'],
            'value': best_row['max_drawdown']
        }

        return best_performers

    def _find_best_performers_by_strategy(self, df: pd.DataFrame) -> Dict:
        """找出每种策略的最佳表现者"""
        if df.empty:
            return {}

        best_performers = {}

        for strategy_type in df['strategy_type'].unique():
            strategy_df = df[df['strategy_type'] == strategy_type]

            if not strategy_df.empty:
                # 找出夏普比率最高的
                best_idx = strategy_df['sharpe_ratio'].idxmax()
                best_row = strategy_df.loc[best_idx]

                best_performers[strategy_type] = {
                    'symbol': best_row['symbol'],
                    'stock_name': best_row['stock_name'],
                    'total_return': best_row['total_return'],
                    'sharpe_ratio': best_row['sharpe_ratio'],
                    'max_drawdown': best_row['max_drawdown'],
                    'win_rate': best_row['win_rate']
                }

        return best_performers

    def _rank_strategies(self, df: pd.DataFrame) -> Dict:
        """策略排名"""
        if df.empty:
            return {}

        # 计算每种策略的平均得分
        strategy_ranking = {}
        for strategy_type in df['strategy_type'].unique():
            strategy_df = df[df['strategy_type'] == strategy_type]
            avg_score = strategy_df['strategy_score'].mean()

            strategy_ranking[strategy_type] = {
                'display_name': self.strategy_display_names.get(strategy_type, strategy_type),
                'avg_score': avg_score,
                'avg_total_return': strategy_df['total_return'].mean(),
                'avg_sharpe_ratio': strategy_df['sharpe_ratio'].mean(),
                'avg_max_drawdown': strategy_df['max_drawdown'].mean(),
                'avg_win_rate': strategy_df['win_rate'].mean(),
                'sample_size': len(strategy_df)
            }

        # 按平均得分排序
        ranked_strategies = sorted(
            strategy_ranking.items(),
            key=lambda x: x[1]['avg_score'],
            reverse=True
        )

        return {
            'ranked_strategies': dict(ranked_strategies),
            'best_strategy': ranked_strategies[0][0] if ranked_strategies else None,
            'ranking_order': [item[0] for item in ranked_strategies]
        }

    def _generate_strategy_recommendations(self, df: pd.DataFrame, etf_comparisons: Dict) -> Dict:
        """生成策略推荐"""
        recommendations = []

        # 基于整体表现推荐
        strategy_ranking = self._rank_strategies(df)
        if 'best_strategy' in strategy_ranking and strategy_ranking['best_strategy']:
            best_strategy = strategy_ranking['best_strategy']
            best_strategy_name = self.strategy_display_names.get(best_strategy, best_strategy)
            recommendations.append({
                'type': 'overall_best',
                'title': '最佳整体策略',
                'content': f'根据综合评分，{best_strategy_name}表现最佳',
                'strategy': best_strategy
            })

        # 基于风险偏好推荐
        risk_analysis = self._analyze_strategy_risks(df)
        if 'lowest_risk_strategy' in risk_analysis:
            low_risk_strategy = risk_analysis['lowest_risk_strategy']
            low_risk_name = self.strategy_display_names.get(low_risk_strategy, low_risk_strategy)
            recommendations.append({
                'type': 'low_risk',
                'title': '低风险策略推荐',
                'content': f'对于风险厌恶型投资者，推荐使用{low_risk_name}',
                'strategy': low_risk_strategy
            })

        # 基于市场条件推荐
        market_analysis = self._analyze_market_conditions(df)
        if 'high_volatility_recommendation' in market_analysis:
            high_vol_rec = market_analysis['high_volatility_recommendation']
            recommendations.append({
                'type': 'market_condition',
                'title': '高波动市场推荐',
                'content': high_vol_rec['content'],
                'strategy': high_vol_rec['strategy']
            })

        # ETF特定推荐
        etf_specific = []
        for symbol, comparison in etf_comparisons.items():
            best_strategy = comparison['best_strategy']
            strategy_name = self.strategy_display_names.get(best_strategy, best_strategy)

            etf_specific.append({
                'symbol': symbol,
                'recommended_strategy': best_strategy,
                'strategy_name': strategy_name,
                'score': comparison['best_score']
            })

        return {
            'general_recommendations': recommendations,
            'etf_specific_recommendations': etf_specific
        }

    def _analyze_strategy_risks(self, df: pd.DataFrame) -> Dict:
        """分析策略风险"""
        if df.empty:
            return {}

        risk_analysis = {}

        for strategy_type in df['strategy_type'].unique():
            strategy_df = df[df['strategy_type'] == strategy_type]

            risk_metrics = {
                'avg_max_drawdown': strategy_df['max_drawdown'].mean(),
                'max_drawdown_std': strategy_df['max_drawdown'].std(),
                'negative_return_rate': (strategy_df['total_return'] < 0).mean(),
                'extreme_loss_rate': (strategy_df['max_drawdown'] > 0.3).mean()
            }

            # 综合风险评分 (0-1, 越低越好)
            risk_score = (
                risk_metrics['avg_max_drawdown'] * 0.4 +
                risk_metrics['max_drawdown_std'] * 0.2 +
                risk_metrics['negative_return_rate'] * 0.3 +
                risk_metrics['extreme_loss_rate'] * 0.1
            )

            risk_analysis[strategy_type] = {
                'metrics': risk_metrics,
                'risk_score': risk_score,
                'risk_level': self._get_risk_level(risk_score)
            }

        # 找出最低风险策略
        if risk_analysis:
            lowest_risk = min(risk_analysis.items(), key=lambda x: x[1]['risk_score'])
            risk_analysis['lowest_risk_strategy'] = lowest_risk[0]
            risk_analysis['lowest_risk_score'] = lowest_risk[1]['risk_score']

        return risk_analysis

    def _get_risk_level(self, risk_score: float) -> str:
        """根据风险评分返回风险等级"""
        if risk_score < 0.1:
            return "低风险"
        elif risk_score < 0.2:
            return "中等风险"
        else:
            return "高风险"

    def _analyze_market_conditions(self, df: pd.DataFrame) -> Dict:
        """分析市场条件对策略的影响"""
        if df.empty:
            return {}

        # 基于策略表现推断市场条件
        dynamic_performance = df[df['strategy_type'] == 'dynamic_grid']
        martingale_performance = df[df['strategy_type'] == 'martingale_grid']

        analysis = {}

        # 如果动态策略表现好，说明市场波动较大
        if not dynamic_performance.empty:
            dynamic_score = dynamic_performance['sharpe_ratio'].mean()
            basic_score = df[df['strategy_type'] == 'basic_grid']['sharpe_ratio'].mean() if not df[df['strategy_type'] == 'basic_grid'].empty else 0

            if dynamic_score > basic_score * 1.2:
                analysis['high_volatility_recommendation'] = {
                    'content': '动态网格策略在当前市场条件下表现优异，建议在波动较大的市场中使用',
                    'strategy': 'dynamic_grid'
                }

        # 如果马丁格尔策略表现好，说明市场有较强的均值回归特性
        if not martingale_performance.empty:
            martingale_score = martingale_performance['total_return'].mean()
            basic_score = df[df['strategy_type'] == 'basic_grid']['total_return'].mean() if not df[df['strategy_type'] == 'basic_grid'].empty else 0

            if martingale_score > basic_score * 1.1:
                analysis['mean_reversion_recommendation'] = {
                    'content': '马丁格尔策略表现良好，说明市场具有均值回归特性，可考虑在震荡市中使用',
                    'strategy': 'martingale_grid'
                }

        return analysis

    def _analyze_risk_distribution(self, df: pd.DataFrame) -> Dict:
        """分析风险分布"""
        if df.empty:
            return {}

        return {
            'return_distribution': {
                'mean': df['total_return'].mean(),
                'std': df['total_return'].std(),
                'min': df['total_return'].min(),
                'max': df['total_return'].max(),
                'percentiles': {
                    '25th': df['total_return'].quantile(0.25),
                    'median': df['total_return'].median(),
                    '75th': df['total_return'].quantile(0.75)
                }
            },
            'drawdown_distribution': {
                'mean': df['max_drawdown'].mean(),
                'std': df['max_drawdown'].std(),
                'worst_case': df['max_drawdown'].max()
            }
        }

    def export_comparison_report(self, comparison_results: Dict, output_path: str = None) -> str:
        """
        导出比较报告

        Args:
            comparison_results: 比较结果
            output_path: 输出路径，如果为None则生成默认路径

        Returns:
            报告文件路径
        """
        try:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"strategy_comparison_report_{timestamp}.json"

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, ensure_ascii=False, indent=2, default=str)

            logger.info(f"策略比较报告已导出至: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"导出比较报告失败: {e}")
            return ""

    def get_display_data_for_dashboard(self, comparison_results: Dict) -> Dict:
        """
        获取用于仪表板显示的数据

        Args:
            comparison_results: 比较结果

        Returns:
            仪表板显示数据
        """
        try:
            display_data = {
                'summary_stats': comparison_results.get('summary_stats', {}),
                'best_performers': comparison_results.get('best_performers_by_strategy', {}),
                'recommendations': comparison_results.get('recommendations', {}),
                'strategy_rankings': comparison_results.get('strategy_rankings', {}),
                'risk_analysis': comparison_results.get('risk_analysis', {})
            }

            # 添加图表数据
            if 'strategy_data' in comparison_results:
                df = pd.DataFrame(comparison_results['strategy_data'])
                display_data['chart_data'] = {
                    'strategy_performance': df.groupby('strategy_type').agg({
                        'total_return': 'mean',
                        'sharpe_ratio': 'mean',
                        'max_drawdown': 'mean',
                        'win_rate': 'mean'
                    }).to_dict('index'),
                    'performance_distribution': df[['strategy_type', 'total_return', 'sharpe_ratio']].to_dict('records')
                }

            return display_data

        except Exception as e:
            logger.error(f"准备仪表板数据失败: {e}")
            return {'error': str(e)}