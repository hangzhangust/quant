"""
结果展示模块
Results Display Module

负责格式化和展示回测结果，包括表格生成、数据可视化和导出功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
import streamlit as st

logger = logging.getLogger(__name__)


class ResultsDisplay:
    """结果展示管理器"""

    def __init__(self):
        """初始化结果展示管理器"""
        self.color_mapping = {
            'positive': '#28a745',  # 绿色
            'negative': '#dc3545',  # 红色
            'neutral': '#6c757d',   # 灰色
            'warning': '#ffc107',   # 黄色
            'info': '#17a2b8'      # 蓝色
        }

    def create_results_dataframe(self, batch_results: Dict) -> pd.DataFrame:
        """
        创建结果表格数据框 - 支持单策略和多策略模式

        Args:
            batch_results: 批量回测结果

        Returns:
            格式化的结果DataFrame
        """
        try:
            # 检测结果类型
            is_multi_strategy = self._detect_multi_strategy(batch_results)

            if is_multi_strategy:
                return self._create_multi_strategy_dataframe(batch_results)
            else:
                return self._create_single_strategy_dataframe(batch_results)
        except Exception as e:
            logger.error(f"创建结果DataFrame失败: {e}")
            return self._create_empty_dataframe()

    def _format_numeric_value(self, value):
        """格式化数值显示 - 返回数值类型以便DataFrame样式化"""
        try:
            if pd.isna(value) or value == 0:
                return 0.0  # 返回数值而不是字符串

            if isinstance(value, (int, float)):
                return float(value)  # 确保返回数值类型

            # 如果是字符串，尝试转换为数值
            if isinstance(value, str):
                if value == '-' or value == '':
                    return 0.0
                try:
                    return float(value)
                except ValueError:
                    return 0.0

            return 0.0

        except Exception:
            return 0.0

    def _format_optimization_suggestions(self, suggestions: List[str]) -> str:
        """格式化优化建议"""
        try:
            if not suggestions:
                return '无需优化'

            # 只显示前2个建议，用分号分隔
            limited_suggestions = suggestions[:2]
            return ' | '.join(limited_suggestions)

        except Exception:
            return '格式化失败'

    def _create_empty_dataframe(self) -> pd.DataFrame:
        """创建空DataFrame"""
        columns = [
            'ETF名称', 'ETF代码', '当前设置', '总收益率', '年化收益率', '最大回撤',
            '夏普比率', '胜率', '交易次数', '平均持仓天数', '总交易成本',
            'VaR(95%)', '年化波动率', '最大连续亏损', '贝塔系数',
            '优化建议', '风险评级', '数据质量', '最后更新', '策略类型'
        ]
        return pd.DataFrame(columns=columns)

    def _detect_multi_strategy(self, batch_results: Dict) -> bool:
        """检测是否为多策略结果"""
        try:
            individual_results = batch_results.get('individual_results', {})

            # 检查是否有结果包含strategies字段
            for result in individual_results.values():
                if result.get('strategies') and isinstance(result['strategies'], dict):
                    return True

            # 检查summary_stats中是否有strategy_stats
            summary_stats = batch_results.get('summary_stats', {})
            if 'strategy_stats' in summary_stats:
                return True

            return False
        except Exception:
            return False

    def _create_single_strategy_dataframe(self, batch_results: Dict) -> pd.DataFrame:
        """创建单策略DataFrame - 保持原有逻辑"""
        try:
            individual_results = batch_results.get('individual_results', {})
            successful_results = [r for r in individual_results.values() if r.get('success', False)]

            if not successful_results:
                return self._create_empty_dataframe()

            data_rows = []

            for result in successful_results:
                row_data = self._extract_strategy_data(result, None)
                data_rows.append(row_data)

            df = pd.DataFrame(data_rows)
            return self._format_dataframe(df)

        except Exception as e:
            logger.error(f"创建单策略DataFrame失败: {e}")
            return self._create_empty_dataframe()

    def _create_multi_strategy_dataframe(self, batch_results: Dict) -> pd.DataFrame:
        """创建多策略DataFrame - 扁平化处理"""
        try:
            individual_results = batch_results.get('individual_results', {})
            data_rows = []

            for symbol, result in individual_results.items():
                if not result.get('success', False):
                    continue

                strategies = result.get('strategies', {})

                # 如果没有strategies字段，但有直接的数据，当作单策略处理
                if not strategies and result.get('metrics'):
                    row_data = self._extract_strategy_data(result, None)
                    data_rows.append(row_data)
                else:
                    # 处理多策略数据
                    for strategy_type, strategy_result in strategies.items():
                        if strategy_result.get('success', False):
                            row_data = self._extract_strategy_data(strategy_result, strategy_type)
                            data_rows.append(row_data)

            if not data_rows:
                return self._create_empty_dataframe()

            df = pd.DataFrame(data_rows)
            return self._format_dataframe(df)

        except Exception as e:
            logger.error(f"创建多策略DataFrame失败: {e}")
            return self._create_empty_dataframe()

    def _extract_strategy_data(self, result: Dict, strategy_type: Optional[str]) -> Dict:
        """提取策略数据"""
        try:
            config = result.get('config', {})
            metrics = result.get('metrics', {})
            basic_metrics = metrics.get('basic_metrics', {})
            trading_metrics = metrics.get('trading_metrics', {})
            risk_metrics = metrics.get('risk_metrics', {})

            row = {
                'ETF名称': result.get('stock_name', ''),
                'ETF代码': result.get('symbol', ''),
                '策略类型': self._format_strategy_type(strategy_type),
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
                '优化建议': self._format_optimization_suggestions(result.get('optimization_suggestions', [])),
                '风险评级': result.get('risk_rating', '未知'),
                '数据质量': result.get('data_quality', {}).get('quality_score', 0),
                '最后更新': result.get('backtest_time', '')
            }
            return row

        except Exception as e:
            logger.error(f"提取策略数据失败: {e}")
            return self._get_empty_row(strategy_type)

    def _format_strategy_type(self, strategy_type: Optional[str]) -> str:
        """格式化策略类型显示"""
        if not strategy_type:
            return "默认策略"

        strategy_names = {
            'basic_grid': '基础网格策略',
            'dynamic_grid': '动态网格策略',
            'martingale_grid': '马丁格尔网格策略'
        }
        return strategy_names.get(strategy_type, strategy_type)

    def _get_empty_row(self, strategy_type: Optional[str]) -> Dict:
        """获取空数据行"""
        return {
            'ETF名称': '',
            'ETF代码': '',
            '策略类型': self._format_strategy_type(strategy_type),
            '当前设置': '',
            '总收益率': 0,
            '年化收益率': 0,
            '最大回撤': 0,
            '夏普比率': 0,
            '胜率': 0,
            '交易次数': 0,
            '平均持仓天数': 0,
            '总交易成本': 0,
            'VaR(95%)': 0,
            '年化波动率': 0,
            '最大连续亏损': 0,
            '贝塔系数': 1.0,
            '优化建议': '无数据',
            '风险评级': '未知',
            '数据质量': 0,
            '最后更新': ''
        }

    def _format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """格式化DataFrame数值列"""
        try:
            # 如果是单策略，移除策略类型列
            if '策略类型' in df.columns and df['策略类型'].nunique() == 1 and df['策略类型'].iloc[0] == "默认策略":
                df = df.drop('策略类型', axis=1)

            # 格式化数值列
            numeric_columns = ['总收益率', '年化收益率', '最大回撤', '夏普比率', '胜率',
                              '平均持仓天数', '总交易成本', 'VaR(95%)', '年化波动率',
                              '最大连续亏损', '贝塔系数', '数据质量']

            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].apply(self._format_numeric_value)

            return df

        except Exception as e:
            logger.error(f"格式化DataFrame失败: {e}")
            return df

    def get_available_strategies(self, batch_results: Dict) -> List[str]:
        """获取可用策略列表"""
        try:
            strategies = set()
            individual_results = batch_results.get('individual_results', {})

            for result in individual_results.values():
                if result.get('strategies') and isinstance(result['strategies'], dict):
                    strategies.update(result['strategies'].keys())

            return sorted(list(strategies)) if strategies else []
        except Exception:
            return []

    def display_summary_statistics(self, batch_results: Dict):
        """显示汇总统计信息"""
        try:
            summary_stats = batch_results.get('summary_stats', {})

            if not summary_stats:
                st.warning("无统计数据可显示")
                return

            # 创建统计卡片
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "总ETF数",
                    summary_stats.get('total_etfs', 0),
                    delta=None
                )

            with col2:
                success_count = summary_stats.get('successful_etfs', 0)
                st.metric(
                    "成功数",
                    success_count,
                    delta=f"{success_count/summary_stats.get('total_etfs', 1)*100:.1f}%"
                )

            with col3:
                avg_return = summary_stats.get('avg_total_return', 0)
                st.metric(
                    "平均收益率",
                    f"{avg_return:+.2%}",
                    delta=f"{avg_return:+.2%}"
                )

            with col4:
                avg_sharpe = summary_stats.get('avg_sharpe_ratio', 0)
                st.metric(
                    "平均夏普比率",
                    f"{avg_sharpe:.2f}",
                    delta=None
                )

            # 最佳和最差表现者
            best_performer = summary_stats.get('best_performer', {})
            worst_performer = summary_stats.get('worst_performer', {})

            if best_performer and worst_performer:
                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🏆 最佳表现")
                    st.write(f"**{best_performer.get('stock_name', '')}**")
                    st.write(f"夏普比率: `{best_performer.get('sharpe_ratio', 0):.2f}`")
                    st.write(f"总收益率: `{best_performer.get('total_return', 0):+.2%}`")

                with col2:
                    st.markdown("### ⚠️ 需要关注")
                    st.write(f"**{worst_performer.get('stock_name', '')}**")
                    st.write(f"夏普比率: `{worst_performer.get('sharpe_ratio', 0):.2f}`")
                    st.write(f"总收益率: `{worst_performer.get('total_return', 0):+.2%}`")

        except Exception as e:
            logger.error(f"显示汇总统计失败: {e}")
            st.error("显示汇总统计时出错")

    def display_results_table(self, df: pd.DataFrame):
        """显示结果表格"""
        try:
            if df.empty:
                st.warning("没有可显示的结果数据")
                return

            # 定义列配置
            column_config = {
                'ETF名称': 'ETF名称',
                'ETF代码': '代码',
                '当前设置': '网格设置',
                '总收益率': '总收益率',
                '年化收益率': '年化收益率',
                '最大回撤': '最大回撤',
                '夏普比率': '夏普比率',
                '胜率': '胜率',
                '交易次数': '交易次数',
                '平均持仓天数': '平均持仓(天)',
                '总交易成本': '交易成本',
                'VaR(95%)': 'VaR(95%)',
                '年化波动率': '波动率',
                '最大连续亏损': '最大连亏',
                '贝塔系数': '贝塔系数',
                '优化建议': '优化建议',
                '风险评级': '风险评级',
                '数据质量': '数据质量'
            }

            # 定义显示格式 - 处理零值的显示
            def format_with_zero_handling(fmt_str):
                def formatter(x):
                    if pd.isna(x) or x == 0:
                        return '-'
                    return fmt_str.format(x)
                return formatter

            format_config = {
                '总收益率': format_with_zero_handling('{:+.2%}'),
                '年化收益率': format_with_zero_handling('{:+.2%}'),
                '最大回撤': format_with_zero_handling('{:+.2%}'),
                '胜率': format_with_zero_handling('{:.1%}'),
                'VaR(95%)': format_with_zero_handling('{:.4f}'),
                '年化波动率': format_with_zero_handling('{:.2%}'),
                '数据质量': format_with_zero_handling('{:.0f}')
            }

            # 添加颜色样式
            def color_negative_red(val):
                if pd.isna(val) or val == 0:
                    return ''
                color = 'red' if val < 0 else 'green'
                return f'color: {color}'

            def color_sharpe(val):
                if pd.isna(val) or val == 0:
                    return ''
                if val > 1.5:
                    return 'color: green'
                elif val > 0.5:
                    return 'color: orange'
                else:
                    return 'color: red'

            def color_return(val):
                if pd.isna(val) or val == 0:
                    return ''
                color = 'green' if val > 0 else 'red'
                return f'color: {color}'

            # 应用样式
            styled_df = df.style.format(format_config) \
                .background_gradient(cmap='RdYlGn', subset=['总收益率', '年化收益率']) \
                .applymap(color_sharpe, subset=['夏普比率']) \
                .applymap(color_return, subset=['总收益率', '年化收益率']) \
                .applymap(color_negative_red, subset=['最大回撤'])

            # 显示表格
            st.dataframe(styled_df, use_container_width=True, column_config=column_config)

        except Exception as e:
            logger.error(f"显示结果表格失败: {e}")
            st.error("显示结果表格时出错")

    def display_detailed_analysis(self, selected_etf: str, batch_results: Dict):
        """显示详细分析"""
        try:
            individual_results = batch_results.get('individual_results', {})

            # 查找选中的ETF结果
            selected_result = None
            for symbol, result in individual_results.items():
                if result.get('success') and result.get('symbol') == selected_etf:
                    selected_result = result
                    break

            if not selected_result:
                st.warning(f"未找到 {selected_etf} 的详细分析数据")
                return

            # 创建标签页
            tab1, tab2, tab3 = st.tabs(["📊 性能指标", "💡 优化建议", "📈 详细数据"])

            with tab1:
                self._display_performance_metrics(selected_result)

            with tab2:
                self._display_optimization_details(selected_result)

            with tab3:
                self._display_detailed_data(selected_result)

        except Exception as e:
            logger.error(f"显示详细分析失败: {e}")
            st.error("显示详细分析时出错")

    def _display_performance_metrics(self, result: Dict):
        """显示性能指标"""
        try:
            metrics = result.get('metrics', {})
            basic_metrics = metrics.get('basic_metrics', {})
            trading_metrics = metrics.get('trading_metrics', {})
            risk_metrics = metrics.get('risk_metrics', {})

            # 基础指标
            st.subheader("基础指标")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("总收益率", f"{basic_metrics.get('total_return', 0):+.2%}")
                st.metric("年化收益率", f"{basic_metrics.get('annual_return', 0):+.2%}")

            with col2:
                st.metric("最大回撤", f"{basic_metrics.get('max_drawdown', 0):+.2%}")
                st.metric("夏普比率", f"{basic_metrics.get('sharpe_ratio', 0):.2f}")

            with col3:
                st.metric("索提诺比率", f"{basic_metrics.get('sortino_ratio', 0):.2f}")
                st.metric("收益回撤比", f"{basic_metrics.get('calmar_ratio', 0):.2f}")

            # 交易指标
            st.subheader("交易指标")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("总交易次数", trading_metrics.get('total_trades', 0))
                st.metric("胜率", f"{trading_metrics.get('win_rate', 0):.1%}")
                st.metric("盈亏比", f"{trading_metrics.get('profit_loss_ratio', 0):.2f}")

            with col2:
                st.metric("平均持仓天数", f"{trading_metrics.get('avg_holding_period', 0):.1f}")
                st.metric("最大连胜", trading_metrics.get('max_consecutive_wins', 0))
                st.metric("最大连亏", trading_metrics.get('max_consecutive_losses', 0))

            # 风险指标
            st.subheader("风险指标")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("年化波动率", f"{risk_metrics.get('volatility', 0):.2%}")
                st.metric("VaR(95%)", f"{risk_metrics.get('var_95', 0):.4f}")
                st.metric("CVaR(95%)", f"{risk_metrics.get('cvar_95', 0):.4f}")

            with col2:
                st.metric("贝塔系数", f"{risk_metrics.get('beta', 1.0):.2f}")
                st.metric("下行风险", f"{risk_metrics.get('downside_deviation', 0):.2f}")
                st.metric("跟踪误差", f"{risk_metrics.get('tracking_error', 0):.2%}")

        except Exception as e:
            logger.error(f"显示性能指标失败: {e}")
            st.error("显示性能指标时出错")

    def _display_optimization_details(self, result: Dict):
        """显示优化建议详情"""
        try:
            suggestions = result.get('optimization_suggestions', [])
            risk_rating = result.get('risk_rating', '未知')

            st.subheader("风险评估")
            st.write(f"**风险评级**: {risk_rating}")

            st.subheader("优化建议")
            if suggestions:
                # suggestions现在是List[str]格式，直接显示字符串建议
                for i, suggestion in enumerate(suggestions, 1):
                    with st.expander(f"建议 {i}: {suggestion[:50]}..."):
                        st.write(suggestion)
            else:
                st.info("当前参数设置较为合理，无需特别优化")

        except Exception as e:
            logger.error(f"显示优化建议失败: {e}")
            st.error("显示优化建议时出错")

    def _display_detailed_data(self, result: Dict):
        """显示详细数据"""
        try:
            # 数据质量
            data_quality = result.get('data_quality', {})
            st.subheader("数据质量评估")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("数据质量评分", f"{data_quality.get('quality_score', 0)}/100")
                st.metric("数据点数", data_quality.get('data_points', 0))

            with col2:
                st.metric("数据范围", data_quality.get('date_range', ''))

            if 'issues' in data_quality and data_quality['issues']:
                st.write("**数据问题**:")
                for issue in data_quality['issues']:
                    st.write(f"• {issue}")

            # 交易记录
            backtest_results = result.get('backtest_results', {})
            if 'trades' in backtest_results and backtest_results['trades']:
                st.subheader("交易记录")
                trades_data = []
                for trade in backtest_results['trades']:
                    trades_data.append({
                        '时间': trade.timestamp.strftime('%Y-%m-%d %H:%M'),
                        '方向': trade.side.value.upper(),
                        '数量': trade.quantity,
                        '价格': f"{trade.price:.3f}",
                        '金额': f"{trade.amount:,.2f}",
                        '手续费': f"{trade.commission:.2f}",
                        '滑点': f"{trade.slippage:.4f}"
                    })

                df_trades = pd.DataFrame(trades_data)
                st.dataframe(df_trades, use_container_width=True)

            # 日净值曲线（如果有数据）
            if 'daily_values' in backtest_results and not backtest_results['daily_values'].empty:
                st.subheader("净值曲线")
                daily_values = backtest_results['daily_values']

                # 转换数据格式以适应plotly
                chart_data = pd.DataFrame({
                    'date': daily_values['date'],
                    '总净值': daily_values['total_value'],
                    '现金': daily_values['cash'],
                    '持仓市值': daily_values['market_value'],
                    '浮动盈亏': daily_values['unrealized_pnl']
                })

                st.subheader("净值变化趋势")
                st.line_chart(chart_data, x='date', y=['总净值', '现金', '持仓市值'])

        except Exception as e:
            logger.error(f"显示详细数据失败: {e}")
            st.error("显示详细数据时出错")

    def export_to_csv(self, df: pd.DataFrame, filename: str = None) -> bytes:
        """导出为CSV"""
        try:
            if filename is None:
                filename = f"grid_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            # 生成CSV内容
            csv_content = df.to_csv(index=False, encoding='utf-8-sig')

            return csv_content.encode('utf-8-sig')

        except Exception as e:
            logger.error(f"导出CSV失败: {e}")
            return b''

    def export_optimization_report(self, batch_results: Dict, filename: str = None) -> str:
        """导出优化报告"""
        try:
            if filename is None:
                filename = f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

            report_lines = [f"网格交易优化报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]
            report_lines.append("=" * 50 + "\n\n")

            # 汇总统计
            summary_stats = batch_results.get('summary_stats', {})
            report_lines.append("=== 汇总统计 ===\n")
            report_lines.append(f"总ETF数: {summary_stats.get('total_etfs', 0)}\n")
            report_lines.append(f"成功数: {summary_stats.get('successful_etfs', 0)}\n")
            report_lines.append(f"成功率: {summary_stats.get('success_rate', 0):.1%}\n")
            report_lines.append(f"平均收益率: {summary_stats.get('avg_total_return', 0):+.2%}\n")
            report_lines.append(f"平均夏普比率: {summary_stats.get('avg_sharpe_ratio', 0):.2f}\n")
            report_lines.append(f"平均最大回撤: {summary_stats.get('avg_max_drawdown', 0):+.2%}\n\n")

            # 个别ETF详情
            individual_results = batch_results.get('individual_results', {})
            successful_results = [r for r in individual_results.values() if r.get('success', False)]

            for result in successful_results:
                report_lines.append(f"=== {result.get('stock_name', '')} ({result.get('symbol', '')}) ===\n")

                # 基本信息
                config = result.get('config', {})
                report_lines.append(f"网格设置: 卖出{config.get('sell_percentage', 0):.1f}% / 买入{config.get('buy_percentage', 0):.1f}%\n")

                # 优化建议
                suggestions = result.get('optimization_suggestions', [])
                if suggestions:
                    report_lines.append("优化建议:\n")
                    for suggestion in suggestions:
                        report_lines.append(f"• {suggestion}\n")
                report_lines.append("\n")

            return '\n'.join(report_lines)

        except Exception as e:
            logger.error(f"导出优化报告失败: {e}")
            return "导出失败"


def main():
    """测试函数"""
    # 创建测试数据
    test_data = {
        'ETF名称': ['测试ETF1', '测试ETF2'],
        'ETF代码': ['TEST001', 'TEST002'],
        '总收益率': [0.05, -0.02],
        '夏普比率': [1.2, 0.8],
        '优化建议': ['建议A', '建议B']
    }

    df = pd.DataFrame(test_data)

    # 创建展示管理器
    display = ResultsDisplay()

    # 测试DataFrame创建
    result_df = display.create_results_dataframe({})
    print("测试DataFrame创建完成")
    print(f"列数: {len(result_df.columns)}")
    print(f"行数: {len(result_df)}")


if __name__ == "__main__":
    main()