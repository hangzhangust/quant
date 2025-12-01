"""
网格交易策略回测系统 - Streamlit Web应用
Grid Trading Strategy Backtest System - Web Interface

上传Table.xls文件，自动解析网格配置并进行批量回测分析
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import time
import sys
import os
from datetime import datetime
from typing import Dict, List

# 添加src路径
sys.path.append('src')

# 导入必要模块
try:
    from src.web.batch_backtest_engine import BatchBacktestEngine
    from src.web.results_display import ResultsDisplay
    from src.web.strategy_charts import StrategyChartGenerator
    from src.analysis.strategy_comparator import StrategyComparator
    from src.analysis.strategy_optimizer import StrategyOptimizer
    from src.utils.font_config import setup_chinese_fonts, get_bilingual_labels
except ImportError as e:
    st.error(f"❌ 模块导入失败: {str(e)}")
    st.error("请确保所有必需的模块文件已创建")
    st.stop()

# 导入matplotlib用于统计分析
try:
    import matplotlib.pyplot as plt
    # 设置中文字体
    setup_chinese_fonts()
except ImportError:
    st.error("❌ matplotlib未安装，请运行: pip install matplotlib")
    st.stop()

# 页面配置
st.set_page_config(
    page_title="网格交易策略回测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    border: 1px solid #e1e5e9;
    padding: 10px;
    border-radius: 5px;
    text-align: center;
}
.stDataFrame {
    width: 100%;
}
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.success-message {
    background-color: #d4edda;
    color: #155724;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
.warning-message {
    background-color: #fff3cd;
    color: #856404;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def main():
    """主应用函数"""
    # 创建侧边栏
    with st.sidebar:
        st.markdown("## 📈 网格交易回测系统")

        st.markdown("---")

        # 文件上传区域
        st.markdown("### 📁 文件上传")
        uploaded_file = st.file_uploader(
            "上传 Table.xls 文件",
            type=['xls', 'xlsx'],
            help="请上传包含网格交易条件的Table.xls文件"
        )

        if uploaded_file:
            st.success(f"已上传文件: {uploaded_file.name}")

        st.markdown("---")

        # 回测配置
        st.markdown("### ⚙️ 回测配置")

        # 时间范围选择
        time_range = st.selectbox(
            "回测时间范围",
            options=[
                "最近3年",
                "最近2年",
                "最近1年",
                "最近6个月"
            ],
            index=0
        )

        # 并行线程数
        max_workers = st.slider(
            "并行处理线程数",
            min_value=1,
            max_value=8,
            value=3,
            help="线程数越多处理越快，但消耗更多系统资源"
        )

        st.markdown("---")

        # 策略比较选项
        st.markdown("### 🔄 策略比较")

        # 初始化所有变量的默认值
        strategy_types = ['basic_grid']  # 默认策略类型
        enable_optimization = False     # 默认不启用优化
        optimization_method = 'grid_search'  # 默认优化方法
        max_iterations = 50            # 默认迭代次数

        enable_strategy_comparison = st.checkbox(
            "启用多策略比较",
            value=False,
            help="同时运行多种策略进行性能对比"
        )

        if enable_strategy_comparison:
            strategy_types = st.multiselect(
                "选择要比较的策略",
                options=['basic_grid', 'dynamic_grid', 'martingale_grid'],
                default=['basic_grid', 'dynamic_grid'],
                help="选择要同时回测的网格策略类型",
                format_func=lambda x: {
                    'basic_grid': '基础网格策略',
                    'dynamic_grid': '动态网格策略',
                    'martingale_grid': '马丁格尔网格策略'
                }[x]
            )

            enable_optimization = st.checkbox(
                "启用参数优化",
                value=False,
                help="为每种策略进行参数优化"
            )

            if enable_optimization:
                optimization_method = st.selectbox(
                    "优化方法",
                    options=['grid_search', 'random_search'],
                    index=0,
                    format_func=lambda x: '网格搜索' if x == 'grid_search' else '随机搜索'
                )
                max_iterations = st.slider(
                    "最大迭代次数",
                    min_value=10,
                    max_value=100,
                    value=50,
                    help="参数优化的最大迭代次数"
                )
        else:
            strategy_types = ['basic_grid']
            enable_optimization = False
            optimization_method = 'grid_search'
            max_iterations = 50

        st.markdown("---")

        # 使用说明
        st.markdown("### 📋 使用说明")
        st.markdown("""
        1. **上传文件**: 上传包含ETF网格配置的Table.xls文件
        2. **开始回测**: 点击"开始批量回测"按钮
        3. **查看结果**: 系统会显示所有ETF的回测结果
        4. **详细分析**: 选择特定ETF查看详细分析
        5. **导出报告**: 下载CSV格式的结果和优化建议
        """)

    # 主界面
    st.markdown('<div class="main-header">📊 网格交易策略回测分析系统</div>', unsafe_allow_html=True)

    # 初始化session state
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = None
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'selected_etf' not in st.session_state:
        st.session_state.selected_etf = None

    # 文件上传处理
    if uploaded_file is not None:
        # 解析上传的文件
        with st.spinner("正在解析文件..."):
            try:
                # 使用网格配置解析器处理
                from src.data.grid_config_parser import GridConfigParser
                parser = GridConfigParser()
                config_df = parser.parse_excel_file(uploaded_file)

                if config_df.empty:
                    st.error("❌ 文件解析失败，请检查文件格式是否正确")
                    return

                st.success(f"✅ 成功解析 {len(config_df)} 个ETF配置")

                # 显示解析统计
                stats = parser.get_summary_statistics(config_df)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("有效配置", stats.get('total_configs', 0))
                with col2:
                    st.metric("平均基准价", f"{stats.get('avg_base_price', 0):.3f}")
                with col3:
                    st.metric("平均卖出网格", f"{stats.get('avg_sell_percentage', 0):.2f}%")

                # 存储解析结果
                st.session_state.config_df = config_df

                # 显示预览
                st.markdown("### 📋 配置预览")
                preview_cols = ['stock_name', 'stock_code', 'base_price', 'sell_percentage', 'buy_percentage']
                st.dataframe(config_df[preview_cols].head(), use_container_width=True)

            except Exception as e:
                st.error(f"❌ 文件解析出错: {str(e)}")
                return

        # 批量回测按钮
        if st.button("🚀 开始批量回测", type="primary", use_container_width=True):
            if 'config_df' in st.session_state:
                run_batch_backtest(
                    st.session_state.config_df,
                    max_workers,
                    enable_strategy_comparison,
                    strategy_types,
                    enable_optimization,
                    optimization_method,
                    max_iterations
                )

    # 显示结果
    if st.session_state.batch_results:
        display_results(st.session_state.batch_results)
    else:
        # 显示欢迎信息
        st.markdown("""
        ## 🎯 使用指南

        欢开始使用网格交易回测系统，请：

        1. **准备文件**: 确保您的Table.xls文件包含以下列：
           - ETF名称和代码
           - 基准价格
           - 网格设置（卖出/买入百分比）
           - 委托数量配置

        2. **上传文件**: 在左侧边栏上传您的Table.xls文件

        3. **开始回测**: 点击"开始批量回测"按钮

        4. **分析结果**: 系统将自动为每个ETF生成：
           - 完整的回测指标
           - 风险评级和优化建议
           - 详细的性能分析
        """)

        # 显示系统特性
        st.markdown("### ✨ 系统特性")

        features = [
            "📊 **完整指标体系**: 总收益率、年化收益率、最大回撤、夏普比率等",
            "🎯 **智能优化建议**: 基于历史数据的参数优化建议",
            "⚡ **并行处理**: 多线程并行处理，提高回测效率",
            "📈 **实时进度**: 显示处理进度和状态更新",
            "💾 **数据导出**: 支持CSV格式结果导出",
            "🔍 **详细分析**: 逐个ETF的深入分析",
            "⚖️ **风险评级**: 自动风险评级和管理建议"
        ]

        for feature in features:
            st.markdown(f"- {feature}")

        # 显示示例数据
        st.markdown("### 📊 支持的ETF示例")

        sample_etfs = [
            "科创50ETF (159682)", "沪深300ETF (510300)", "芯片ETF (159995)",
            "消费ETF (159928)", "恒生ETF (513600)", "创业板ETF (159915)"
        ]

        col1, col2, col3 = st.columns(3)
        for i, etf in enumerate(sample_etfs):
            col = [col1, col2, col3][i % 3]
            with col:
                st.info(f"{etf}")

def run_batch_backtest(config_df: pd.DataFrame, max_workers: int,
                       enable_strategy_comparison: bool = False,
                       strategy_types: List[str] = None,
                       enable_optimization: bool = False,
                       optimization_method: str = 'grid_search',
                       max_iterations: int = 50):
    """运行批量回测"""
    # 创建进度条
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 创建批量回测引擎
    engine = BatchBacktestEngine(max_workers=max_workers)

    # 进度回调函数
    def progress_callback(completed, total, message):
        progress = completed / total
        progress_bar.progress(progress)
        status_text.text(f"处理进度: {completed}/{total} - {message}")

    try:
        # 确定策略类型
        if not strategy_types:
            strategy_types = ['basic_grid']

        # 运行批量回测
        st.session_state.batch_results = engine.run_batch_backtest(
            config_df,
            progress_callback,
            strategy_types if enable_strategy_comparison else ['basic_grid']
        )

        # 生成结果表格
        display_engine = ResultsDisplay()
        st.session_state.results_df = display_engine.create_results_dataframe(st.session_state.batch_results)

        # 完成提示
        progress_bar.progress(1.0)
        status_text.text("✅ 批量回测完成！")

        # 显示成功消息
        summary = st.session_state.batch_results
        success_count = summary.get('successful_etfs', 0)
        total_count = summary.get('total_etfs', 0)
        execution_time = summary.get('execution_time', 0)

        st.success(f"🎉 回测完成！成功处理 {success_count}/{total_count} 个ETF，耗时 {execution_time:.2f} 秒")

    except Exception as e:
        st.error(f"❌ 批量回测失败: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def display_results(batch_results: Dict):
    """显示回测结果"""
    try:
        # 创建结果展示器
        display = ResultsDisplay()

        # 创建结果DataFrame
        results_df = display.create_results_dataframe(batch_results)
        st.session_state.results_df = results_df

        if results_df.empty:
            st.warning("⚠️ 没有可显示的回测结果")
            return

        # 显示汇总统计
        display.display_summary_statistics(batch_results)

        st.markdown("---")

        # 检查是否为多策略比较结果
        summary_stats = batch_results.get('summary_stats', {})
        is_multi_strategy = 'strategy_stats' in summary_stats

        # 动态创建标签页
        if is_multi_strategy:
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 结果表格",
                "📈 统计分析",
                "🔄 策略比较",
                "📥 导出报告"
            ])
        else:
            tab1, tab2, tab3 = st.tabs(["📊 结果表格", "📈 统计分析", "📥 导出报告"])

        with tab1:
            # 策略选择器（仅在多策略模式下显示）
            if is_multi_strategy:
                # 获取可用策略类型
                available_strategies = display.get_available_strategies(batch_results)

                if available_strategies:
                    # 初始化session_state
                    if 'selected_strategies' not in st.session_state:
                        st.session_state.selected_strategies = available_strategies

                    # 策略选择器
                    selected_strategies = st.multiselect(
                        "选择要显示的策略",
                        options=available_strategies,
                        default=st.session_state.selected_strategies,
                        format_func=lambda x: {
                            'basic_grid': '基础网格策略',
                            'dynamic_grid': '动态网格策略',
                            'martingale_grid': '马丁格尔网格策略'
                        }.get(x, x),
                        help="选择要同时显示的网格策略类型"
                    )

                    # 更新session_state
                    st.session_state.selected_strategies = selected_strategies

                    # 根据策略选择过滤结果
                    if selected_strategies:
                        # 检查是否有策略类型列
                        if '策略类型' in results_df.columns:
                            # 使用策略显示名称进行过滤
                            strategy_filter_map = {
                                'basic_grid': '基础网格策略',
                                'dynamic_grid': '动态网格策略',
                                'martingale_grid': '马丁格尔网格策略'
                            }
                            display_names = [strategy_filter_map[s] for s in selected_strategies]
                            filtered_df = results_df[results_df['策略类型'].isin(display_names)]
                        else:
                            filtered_df = results_df
                    else:
                        st.warning("⚠️ 请至少选择一个策略类型")
                        filtered_df = results_df
                else:
                    st.warning("⚠️ 未检测到策略数据")
                    filtered_df = results_df
            else:
                # 单策略模式，使用原始结果
                filtered_df = results_df

            # ETF选择器
            if not filtered_df.empty:
                etf_names = filtered_df['ETF名称'].unique().tolist()
                if etf_names:
                    selected_etf_name = st.selectbox("选择ETF查看详细分析", options=etf_names)

                    # 找到对应的ETF代码
                    selected_etf = filtered_df[filtered_df['ETF名称'] == selected_etf_name]['ETF代码'].iloc[0]
                    st.session_state.selected_etf = selected_etf

                    # 显示结果表格
                    st.subheader("📊 回测结果表格")
                    display.display_results_table(filtered_df)

                    # 详细分析
                    if st.session_state.selected_etf:
                        st.markdown("---")
                        display.display_detailed_analysis(st.session_state.selected_etf, batch_results)
                else:
                    st.warning("⚠️ 没有可用的ETF数据")
            else:
                st.warning("⚠️ 没有符合条件的结果数据")

        with tab2:
            st.subheader("📈 统计分析")
            display_statistical_analysis(batch_results)

        # 策略比较标签页（仅在多策略模式下显示）
        if is_multi_strategy:
            with tab3:
                st.subheader("🔄 策略比较分析")
                display_strategy_comparison_analysis(batch_results)

        # 导出报告标签页
        export_tab = tab4 if is_multi_strategy else tab3
        with export_tab:
            st.subheader("📥 导出报告")
            display_export_options(batch_results, results_df)

    except Exception as e:
        st.error(f"❌ 显示结果失败: {str(e)}")

def display_statistical_analysis(batch_results: Dict):
    """显示统计分析"""
    try:
        summary_stats = batch_results.get('summary_stats', {})

        if not summary_stats:
            st.warning("⚠️ 无统计数据可显示")
            return

        # 收集性能数据
        individual_results = batch_results.get('individual_results', {})
        successful_results = [r for r in individual_results.values() if r.get('success', False)]

        if not successful_results:
            st.warning("⚠️ 没有成功的结果可用于分析")
            return

        # 提取性能指标
        performance_data = []
        for result in successful_results:
            metrics = result.get('metrics', {})
            basic_metrics = metrics.get('basic_metrics', {})
            trading_metrics = metrics.get('trading_metrics', {})

            performance_data.append({
                'ETF名称': result.get('stock_name', ''),
                '总收益率': basic_metrics.get('total_return', 0),
                '夏普比率': basic_metrics.get('sharpe_ratio', 0),
                '最大回撤': abs(basic_metrics.get('max_drawdown', 0)),
                '胜率': trading_metrics.get('win_rate', 0),
                '交易次数': trading_metrics.get('total_trades', 0)
            })

        df_performance = pd.DataFrame(performance_data)

        # 获取双语标签
        labels = get_bilingual_labels()

        # 分布统计
        st.subheader("📊 " + labels['return_distribution'])
        col1, col2 = st.columns(2)

        with col1:
            # 收益率分布直方图
            fig, ax = plt.subplots()
            ax.hist(df_performance['总收益率'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel(labels['total_return'])
            ax.set_ylabel(labels['etf_count'])
            ax.set_title(labels['return_distribution'])
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with col2:
            # 风险收益散点图
            fig, ax = plt.subplots()
            scatter = ax.scatter(df_performance['最大回撤'], df_performance['总收益率'],
                              alpha=0.6, s=50, c=df_performance['夏普比率'], cmap='viridis')
            ax.set_xlabel(labels['max_drawdown'])
            ax.set_ylabel(labels['total_return'])
            ax.set_title(labels['risk_return_distribution'])
            plt.colorbar(scatter, ax=ax, label=labels['sharpe_ratio'])
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        # 相关性分析
        st.subheader("🔍 相关性分析")

        correlation_data = df_performance[['总收益率', '夏普比率', '最大回撤', '胜率', '交易次数']].corr()

        st.write("**指标相关性矩阵**")
        st.dataframe(correlation_data.style.background_gradient(cmap='RdYlBu'), use_container_width=True)

        # 排行榜
        st.subheader("🏆 排行榜")

        # 按不同指标排序
        sort_options = ['总收益率', '夏普比率', '胜率']
        sort_by = st.selectbox("排序指标", options=sort_options, index=1)

        if sort_by == '总收益率':
            sorted_df = df_performance.sort_values('总收益率', ascending=False)
        elif sort_by == '夏普比率':
            sorted_df = df_performance.sort_values('夏普比率', ascending=False)
        else:
            sorted_df = df_performance.sort_values('胜率', ascending=False)

        st.dataframe(sorted_df.style.background_gradient(cmap='RdYlGn', subset=[sort_by]), use_container_width=True)

    except Exception as e:
        st.error(f"❌ 统计分析失败: {str(e)}")

def display_export_options(batch_results: Dict, results_df: pd.DataFrame):
    """显示导出选项"""
    try:
        st.subheader("💾 数据导出")

        # CSV导出
        st.markdown("#### 📊 CSV结果导出")

        if st.button("📥 导出结果CSV"):
            try:
                # 使用ResultsDisplay的导出功能
                display = ResultsDisplay()
                csv_content = display.export_to_csv(results_df)

                # 提供下载
                st.download_button(
                    label="下载回测结果.csv",
                    data=csv_content,
                    file_name=f"grid_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

                st.success("✅ CSV文件已准备下载")

            except Exception as e:
                st.error(f"❌ CSV导出失败: {str(e)}")

        # 优化报告导出
        st.markdown("#### 📝 优化建议报告")

        if st.button("📄 导出优化建议"):
            try:
                # 使用ResultsDisplay的导出功能
                display = ResultsDisplay()
                report_content = display.export_optimization_report(batch_results)

                # 提供下载
                st.download_button(
                    label="下载优化建议.txt",
                    data=report_content,
                    file_name=f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

                st.success("✅ 优化报告已准备下载")

            except Exception as e:
                st.error(f"❌ 报告导出失败: {str(e)}")

        # 数据概览
        st.markdown("#### 📊 数据概览")

        # 显示数据统计
        col1, col2, col3, col4 = st.columns(4)

        summary = batch_results.get('summary_stats', {})

        with col1:
            st.metric("处理ETF数", summary.get('total_etfs', 0))
        with col2:
            st.metric("成功率", f"{summary.get('success_rate', 0):.1%}")
        with col3:
            st.metric("平均收益率", f"{summary.get('avg_total_return', 0):+.2%}")
        with col4:
            st.metric("平均夏普比率", f"{summary.get('avg_sharpe_ratio', 0):.2f}")

        # 详细统计
        if 'individual_results' in batch_results:
            individual_results = batch_results['individual_results']

            # 统计不同风险等级的ETF数量
            risk_ratings = {}
            for result in individual_results.values():
                if result.get('success'):
                    rating = result.get('risk_rating', '未知')
                    risk_ratings[rating] = risk_ratings.get(rating, 0) + 1

            if risk_ratings:
                st.markdown("**风险评级分布**")
                risk_df = pd.DataFrame(list(risk_ratings.items()),
                                      columns=['风险等级', 'ETF数量'])
                st.dataframe(risk_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 导出选项显示失败: {str(e)}")

def display_strategy_comparison_analysis(batch_results: Dict):
    """显示策略比较分析"""
    try:
        # 创建策略比较器
        comparator = StrategyComparator()
        chart_generator = StrategyChartGenerator()

        # 生成策略比较数据
        comparison_data = comparator.compare_strategies(batch_results)

        if 'error' in comparison_data:
            st.error(f"❌ 策略比较分析失败: {comparison_data['error']}")
            return

        # 获取双语标签
        labels = get_bilingual_labels()

        # 📊 策略汇总统计
        st.subheader("📊 策略汇总统计")

        if 'strategy_summary' in comparison_data:
            summary_data = comparison_data['strategy_summary']

            # 创建策略对比表格
            summary_rows = []
            for strategy_type, stats in summary_data.items():
                strategy_name = labels.get(strategy_type, strategy_type)
                summary_rows.append({
                    '策略类型': strategy_name,
                    '平均总收益率': f"{stats['avg_total_return']:.2%}",
                    '平均年化收益率': f"{stats['avg_annual_return']:.2%}",
                    '平均最大回撤': f"{stats['avg_max_drawdown']:.2%}",
                    '平均夏普比率': f"{stats['avg_sharpe_ratio']:.3f}",
                    '平均胜率': f"{stats['avg_win_rate']:.2%}",
                    '正收益率比例': f"{stats['positive_return_rate']:.2%}"
                })

            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True)

        # 🏆 最佳策略表现
        st.subheader("🏆 最佳策略表现")

        if 'best_performers_by_strategy' in comparison_data:
            best_performers = comparison_data['best_performers_by_strategy']

            col1, col2 = st.columns(2)

            with col1:
                for strategy_type, performer in best_performers.items():
                    strategy_name = labels.get(strategy_type, strategy_type)
                    st.markdown(f"""
                    **{strategy_name} 最佳表现:**
                    - ETF: {performer['stock_name']} ({performer['symbol']})
                    - 总收益率: {performer['total_return']:.2%}
                    - 夏普比率: {performer['sharpe_ratio']:.3f}
                    - 最大回撤: {performer['max_drawdown']:.2%}
                    - 胜率: {performer['win_rate']:.2%}
                    """)

        # 📈 策略排名
        st.subheader("📈 策略排名")

        if 'strategy_rankings' in comparison_data:
            rankings = comparison_data['strategy_rankings']

            if 'ranked_strategies' in rankings:
                ranked_data = []
                for i, (strategy_type, ranking_data) in enumerate(rankings['ranked_strategies'].items(), 1):
                    strategy_name = labels.get(strategy_type, strategy_type)
                    ranked_data.append({
                        '排名': i,
                        '策略类型': strategy_name,
                        '综合评分': f"{ranking_data['avg_score']:.3f}",
                        '平均收益率': f"{ranking_data['avg_total_return']:.2%}",
                        '平均夏普比率': f"{ranking_data['avg_sharpe_ratio']:.3f}",
                        '样本数量': ranking_data['sample_size']
                    })

                ranking_df = pd.DataFrame(ranked_data)
                st.dataframe(ranking_df, use_container_width=True)

        # 🎯 策略推荐
        st.subheader("🎯 策略推荐")

        if 'recommendations' in comparison_data:
            recommendations = comparison_data['recommendations']

            # 通用推荐
            if 'general_recommendations' in recommendations:
                st.markdown("##### 💡 通用推荐")
                for rec in recommendations['general_recommendations']:
                    st.markdown(f"- **{rec['title']}**: {rec['content']}")

            # ETF特定推荐
            if 'etf_specific_recommendations' in recommendations:
                st.markdown("##### 📋 ETF特定推荐")

                etf_rec_data = []
                for rec in recommendations['etf_specific_recommendations']:
                    etf_rec_data.append({
                        'ETF代码': rec['symbol'],
                        '推荐策略': labels.get(rec['recommended_strategy'], rec['recommended_strategy']),
                        '推荐评分': f"{rec['score']:.3f}"
                    })

                if etf_rec_data:
                    etf_rec_df = pd.DataFrame(etf_rec_data)
                    st.dataframe(etf_rec_df, use_container_width=True)

        # 📊 可视化图表
        st.subheader("📊 可视化分析")

        # 创建图表选项
        chart_type = st.selectbox(
            "选择图表类型",
            options=["雷达图", "性能对比图", "风险收益散点图", "策略推荐图表"],
            format_func=lambda x: {
                "雷达图": "策略雷达图",
                "性能对比图": "性能指标对比",
                "风险收益散点图": "风险收益分析",
                "策略推荐图表": "策略推荐分布"
            }.get(x, x)
        )

        # 生成并显示选定的图表
        try:
            if chart_type == "雷达图":
                fig = chart_generator.create_strategy_radar_chart(comparison_data)
                st.pyplot(fig)
            elif chart_type == "性能对比图":
                fig = chart_generator.create_performance_comparison_chart(comparison_data)
                st.pyplot(fig)
            elif chart_type == "风险收益散点图":
                fig = chart_generator.create_risk_return_scatter_plot(comparison_data)
                st.pyplot(fig)
            elif chart_type == "策略推荐图表":
                fig = chart_generator.create_strategy_recommendation_chart(comparison_data)
                st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ 图表生成失败: {str(e)}")

        # ⚠️ 风险分析
        st.subheader("⚠️ 风险分析")

        if 'risk_analysis' in comparison_data:
            risk_analysis = comparison_data['risk_analysis']

            risk_data = []
            for strategy_type, risk_info in risk_analysis.items():
                if strategy_type in ['lowest_risk_strategy', 'lowest_risk_score']:
                    continue

                strategy_name = labels.get(strategy_type, strategy_type)
                risk_data.append({
                    '策略类型': strategy_name,
                    '风险评分': f"{risk_info['risk_score']:.3f}",
                    '风险等级': risk_info['risk_level'],
                    '平均最大回撤': f"{risk_info['metrics']['avg_max_drawdown']:.2%}",
                    '负收益比例': f"{risk_info['metrics']['negative_return_rate']:.2%}"
                })

            if risk_data:
                risk_df = pd.DataFrame(risk_data)
                st.dataframe(risk_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 策略比较分析失败: {str(e)}")
        logger.error(f"策略比较分析错误: {e}")


def main_app():
    """主应用入口函数"""
    main()

if __name__ == "__main__":
    main_app()