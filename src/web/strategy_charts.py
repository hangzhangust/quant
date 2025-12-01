"""
策略对比可视化图表模块
Strategy Comparison Charts

提供多种策略对比的可视化图表功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

from src.utils.font_config import get_bilingual_labels, get_font_config

logger = logging.getLogger(__name__)


class StrategyChartGenerator:
    """策略图表生成器"""

    def __init__(self):
        """初始化图表生成器"""
        self.font_config = get_font_config()
        self.labels = get_bilingual_labels()

        # 设置matplotlib和seaborn样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def create_strategy_radar_chart(self, comparison_data: Dict) -> plt.Figure:
        """
        创建策略雷达图

        Args:
            comparison_data: 策略比较数据

        Returns:
            matplotlib图表对象
        """
        try:
            strategy_summary = comparison_data.get('strategy_summary', {})
            if not strategy_summary:
                raise ValueError("策略汇总数据为空")

            # 提取各项指标
            strategies = list(strategy_summary.keys())
            metrics = ['avg_total_return', 'avg_sharpe_ratio', 'avg_win_rate']

            # 标准化指标到0-1范围
            normalized_data = {}
            for strategy in strategies:
                data = strategy_summary[strategy]
                normalized_data[strategy] = [
                    max(0, data['avg_total_return']),  # 总收益率
                    max(0, data['avg_sharpe_ratio']),   # 夏普比率
                    data['avg_win_rate']               # 胜率
                ]

            # 创建雷达图
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

            # 角度设置
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形

            # 颜色设置
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

            # 绘制每个策略
            for i, strategy in enumerate(strategies):
                values = normalized_data[strategy]
                values += values[:1]  # 闭合图形

                ax.plot(angles, values, 'o-', linewidth=2,
                       label=self.labels.get(strategy, strategy),
                       color=colors[i % len(colors)])
                ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

            # 设置标签
            metric_labels = [
                self.labels['total_return'],
                self.labels['sharpe_ratio'],
                self.labels['win_rate']
            ]
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels)
            ax.set_ylim(0, 1)
            ax.set_title(self.labels['strategy_comparison'], size=16, pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)

            return fig

        except Exception as e:
            logger.error(f"创建雷达图失败: {e}")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, f"雷达图创建失败: {str(e)}", ha='center', va='center', transform=ax.transAxes)
            return fig

    def create_performance_comparison_chart(self, comparison_data: Dict) -> plt.Figure:
        """
        创建策略性能对比图

        Args:
            comparison_data: 策略比较数据

        Returns:
            matplotlib图表对象
        """
        try:
            strategy_data = comparison_data.get('strategy_data', [])
            if not strategy_data:
                raise ValueError("策略数据为空")

            df = pd.DataFrame(strategy_data)

            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(self.labels['strategy_comparison'], fontsize=16)

            # 1. 总收益率对比箱线图
            ax1 = axes[0, 0]
            sns.boxplot(data=df, x='strategy_type', y='total_return', ax=ax1)
            ax1.set_title(f"{self.labels['total_return']} Distribution")
            ax1.set_ylabel(self.labels['total_return'])
            ax1.set_xlabel("Strategy Type")

            # 2. 夏普比率对比箱线图
            ax2 = axes[0, 1]
            sns.boxplot(data=df, x='strategy_type', y='sharpe_ratio', ax=ax2)
            ax2.set_title(f"{self.labels['sharpe_ratio']} Distribution")
            ax2.set_ylabel(self.labels['sharpe_ratio'])
            ax2.set_xlabel("Strategy Type")

            # 3. 最大回撤对比箱线图
            ax3 = axes[1, 0]
            sns.boxplot(data=df, x='strategy_type', y='max_drawdown', ax=ax3)
            ax3.set_title(f"{self.labels['max_drawdown']} Distribution")
            ax3.set_ylabel(self.labels['max_drawdown'])
            ax3.set_xlabel("Strategy Type")

            # 4. 胜率对比箱线图
            ax4 = axes[1, 1]
            sns.boxplot(data=df, x='strategy_type', y='win_rate', ax=ax4)
            ax4.set_title(f"{self.labels['win_rate']} Distribution")
            ax4.set_ylabel(self.labels['win_rate'])
            ax4.set_xlabel("Strategy Type")

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"创建性能对比图失败: {e}")
            fig, ax = plt.subplots(figsize=(15, 12))
            ax.text(0.5, 0.5, f"性能对比图创建失败: {str(e)}", ha='center', va='center', transform=ax.transAxes)
            return fig

    def create_risk_return_scatter_plot(self, comparison_data: Dict) -> plt.Figure:
        """
        创建风险收益散点图

        Args:
            comparison_data: 策略比较数据

        Returns:
            matplotlib图表对象
        """
        try:
            strategy_data = comparison_data.get('strategy_data', [])
            if not strategy_data:
                raise ValueError("策略数据为空")

            df = pd.DataFrame(strategy_data)

            fig, ax = plt.subplots(figsize=(12, 8))

            # 按策略类型分组绘制
            strategies = df['strategy_type'].unique()
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

            for i, strategy in enumerate(strategies):
                strategy_df = df[df['strategy_type'] == strategy]
                ax.scatter(strategy_df['max_drawdown'], strategy_df['total_return'],
                          alpha=0.7, s=80, c=colors[i % len(colors)],
                          label=self.labels.get(strategy, strategy),
                          edgecolors='black', linewidth=0.5)

            # 设置图表属性
            ax.set_xlabel(self.labels['max_drawdown'])
            ax.set_ylabel(self.labels['total_return'])
            ax.set_title(self.labels['risk_return_distribution'])
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 添加理想区域标注
            ax.axhspan(0.15, 0.3, alpha=0.1, color='green', label='High Return Zone')
            ax.axvspan(0, 0.15, alpha=0.1, color='blue', label='Low Risk Zone')

            return fig

        except Exception as e:
            logger.error(f"创建风险收益散点图失败: {e}")
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, f"风险收益散点图创建失败: {str(e)}", ha='center', va='center', transform=ax.transAxes)
            return fig

    def create_cumulative_return_comparison(self, comparison_data: Dict) -> plt.Figure:
        """
        创建累积收益对比图

        Args:
            comparison_data: 策略比较数据

        Returns:
            matplotlib图表对象
        """
        try:
            etf_comparisons = comparison_data.get('etf_comparisons', {})
            if not etf_comparisons:
                raise ValueError("ETF比较数据为空")

            # 准备数据
            strategies = {}
            for symbol, comparison in etf_comparisons.items():
                for strategy_type, data in comparison['all_strategies'].items():
                    if strategy_type not in strategies:
                        strategies[strategy_type] = []
                    strategies[strategy_type].append(data['total_return'])

            # 创建图表
            fig, ax = plt.subplots(figsize=(14, 8))

            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            returns_data = []
            strategy_labels = []

            for i, (strategy_type, returns) in enumerate(strategies.items()):
                returns_data.append(returns)
                strategy_labels.append(self.labels.get(strategy_type, strategy_type))

            # 绘制箱线图
            bp = ax.boxplot(returns_data, labels=strategy_labels, patch_artist=True)

            # 设置颜色
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_title(f"{self.labels['cumulative_return']} {self.labels['strategy_comparison']}", fontsize=16)
            ax.set_ylabel(self.labels['total_return'])
            ax.grid(True, alpha=0.3)

            # 添加平均值线
            for i, returns in enumerate(returns_data):
                mean_val = np.mean(returns)
                ax.plot([i+1], [mean_val], marker='o', color='red', markersize=8)

            return fig

        except Exception as e:
            logger.error(f"创建累积收益对比图失败: {e}")
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.text(0.5, 0.5, f"累积收益对比图创建失败: {str(e)}", ha='center', va='center', transform=ax.transAxes)
            return fig

    def create_interactive_strategy_dashboard(self, comparison_data: Dict) -> go.Figure:
        """
        创建交互式策略仪表板

        Args:
            comparison_data: 策略比较数据

        Returns:
            plotly图表对象
        """
        try:
            strategy_data = comparison_data.get('strategy_data', [])
            if not strategy_data:
                raise ValueError("策略数据为空")

            df = pd.DataFrame(strategy_data)

            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    self.labels['total_return'],
                    self.labels['sharpe_ratio'],
                    self.labels['max_drawdown'],
                    self.labels['win_rate']
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )

            # 策略颜色映射
            strategies = df['strategy_type'].unique()
            colors = px.colors.qualitative.Set1[:len(strategies)]

            # 1. 总收益率分布
            for i, strategy in enumerate(strategies):
                strategy_df = df[df['strategy_type'] == strategy]
                fig.add_trace(
                    go.Histogram(
                        x=strategy_df['total_return'],
                        name=strategy,
                        opacity=0.7,
                        marker_color=colors[i]
                    ),
                    row=1, col=1
                )

            # 2. 夏普比率分布
            for i, strategy in enumerate(strategies):
                strategy_df = df[df['strategy_type'] == strategy]
                fig.add_trace(
                    go.Box(
                        y=strategy_df['sharpe_ratio'],
                        name=strategy + "_sharpe",
                        marker_color=colors[i]
                    ),
                    row=1, col=2
                )

            # 3. 最大回撤分布
            for i, strategy in enumerate(strategies):
                strategy_df = df[df['strategy_type'] == strategy]
                fig.add_trace(
                    go.Violin(
                        y=strategy_df['max_drawdown'],
                        name=strategy + "_drawdown",
                        box_visible=True,
                        meanline_visible=True,
                        marker_color=colors[i]
                    ),
                    row=2, col=1
                )

            # 4. 胜率散点图
            for i, strategy in enumerate(strategies):
                strategy_df = df[df['strategy_type'] == strategy]
                fig.add_trace(
                    go.Scatter(
                        x=strategy_df['total_return'],
                        y=strategy_df['win_rate'],
                        mode='markers',
                        name=strategy + "_scatter",
                        marker=dict(color=colors[i], size=8),
                        text=strategy_df['symbol'],
                        textposition="top center"
                    ),
                    row=2, col=2
                )

            # 更新布局
            fig.update_layout(
                title=self.labels['strategy_comparison'] + " - Interactive Dashboard",
                height=800,
                showlegend=True
            )

            return fig

        except Exception as e:
            logger.error(f"创建交互式仪表板失败: {e}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"交互式仪表板创建失败: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

    def create_strategy_recommendation_chart(self, comparison_data: Dict) -> plt.Figure:
        """
        创建策略推荐图表

        Args:
            comparison_data: 策略比较数据

        Returns:
            matplotlib图表对象
        """
        try:
            recommendations = comparison_data.get('recommendations', {})
            etf_specific = recommendations.get('etf_specific_recommendations', [])

            if not etf_specific:
                raise ValueError("ETF推荐数据为空")

            # 统计每种策略被推荐的次数
            strategy_counts = {}
            for rec in etf_specific:
                strategy = rec['recommended_strategy']
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

            # 创建饼图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # 1. 策略推荐分布饼图
            strategies = list(strategy_counts.keys())
            counts = list(strategy_counts.values())
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

            labels = [self.labels.get(strategy, strategy) for strategy in strategies]

            ax1.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors[:len(strategies)])
            ax1.set_title('策略推荐分布')

            # 2. 推荐评分条形图
            strategy_scores = {}
            for rec in etf_specific:
                strategy = rec['recommended_strategy']
                if strategy not in strategy_scores:
                    strategy_scores[strategy] = []
                strategy_scores[strategy].append(rec['score'])

            avg_scores = {strategy: np.mean(scores) for strategy, scores in strategy_scores.items()}
            sorted_strategies = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

            strategies_sorted = [item[0] for item in sorted_strategies]
            scores_sorted = [item[1] for item in sorted_strategies]
            labels_sorted = [self.labels.get(strategy, strategy) for strategy in strategies_sorted]

            bars = ax2.bar(labels_sorted, scores_sorted, color=colors[:len(strategies_sorted)])
            ax2.set_title('策略平均推荐评分')
            ax2.set_ylabel('评分')
            ax2.tick_params(axis='x', rotation=45)

            # 添加数值标签
            for bar, score in zip(bars, scores_sorted):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.3f}', ha='center', va='bottom')

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"创建策略推荐图表失败: {e}")
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.text(0.5, 0.5, f"策略推荐图表创建失败: {str(e)}", ha='center', va='center', transform=ax.transAxes)
            return fig

    def save_charts_to_file(self, fig: plt.Figure, filename: str, dpi: int = 300) -> str:
        """
        保存图表到文件

        Args:
            fig: matplotlib图表对象
            filename: 文件名
            dpi: 分辨率

        Returns:
            保存的文件路径
        """
        try:
            filepath = f"charts/{filename}"
            import os
            os.makedirs("charts", exist_ok=True)

            fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
            logger.info(f"图表已保存至: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"保存图表失败: {e}")
            return ""

    def create_all_comparison_charts(self, comparison_data: Dict) -> Dict[str, plt.Figure]:
        """
        创建所有比较图表

        Args:
            comparison_data: 策略比较数据

        Returns:
            包含所有图表的字典
        """
        charts = {}

        try:
            charts['radar'] = self.create_strategy_radar_chart(comparison_data)
            charts['performance'] = self.create_performance_comparison_chart(comparison_data)
            charts['risk_return'] = self.create_risk_return_scatter_plot(comparison_data)
            charts['cumulative_return'] = self.create_cumulative_return_comparison(comparison_data)
            charts['recommendations'] = self.create_strategy_recommendation_chart(comparison_data)

            # 交互式图表
            charts['interactive_dashboard'] = self.create_interactive_strategy_dashboard(comparison_data)

            logger.info("所有策略比较图表创建完成")
            return charts

        except Exception as e:
            logger.error(f"创建比较图表失败: {e}")
            return charts