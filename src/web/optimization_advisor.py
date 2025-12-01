"""
优化建议生成器
Optimization Advisor

基于回测结果生成智能的网格参数优化建议
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class OptimizationAdvisor:
    """优化建议生成器"""

    def __init__(self):
        """初始化优化建议生成器"""
        self.parameter_ranges = {
            'sell_percentage': (1.0, 15.0),  # 卖出网格间距范围
            'buy_percentage': (2.0, 20.0),   # 买入网格间距范围
            'grid_count': (4, 20),           # 网格数量范围
            'position_size_ratio': (0.5, 2.0) # 仓位比例范围
        }

    def generate_comprehensive_advice(self, config: Dict, metrics: Dict,
                                      market_data: pd.DataFrame) -> Dict:
        """
        生成综合优化建议

        Args:
            config: ETF配置
            metrics: 回测指标
            market_data: 市场数据

        Returns:
            优化建议字典
        """
        try:
            advice = {
                'parameter_optimization': self._generate_parameter_optimization(config, metrics, market_data),
                'strategy_adjustment': self._generate_strategy_adjustment(metrics, market_data),
                'risk_management': self._generate_risk_management_advice(config, metrics),
                'market_timing': self._generate_market_timing_advice(metrics, market_data),
                'overall_recommendation': self._generate_overall_recommendation(metrics),
                'optimization_priority': self._determine_optimization_priority(config, metrics)
            }

            logger.info(f"为 {config.get('stock_name', 'Unknown')} 生成综合优化建议")
            return advice

        except Exception as e:
            logger.error(f"生成综合优化建议失败: {e}")
            return self._get_default_advice()

    def _generate_parameter_optimization(self, config: Dict, metrics: Dict,
                                         market_data: pd.DataFrame) -> List[Dict]:
        """生成参数优化建议"""
        suggestions = []

        try:
            basic_metrics = metrics.get('basic_metrics', {})
            trading_metrics = metrics.get('trading_metrics', {})
            risk_metrics = metrics.get('risk_metrics', {})

            current_sell = config.get('sell_percentage', 5.0)
            current_buy = config.get('buy_percentage', 10.0)

            # 网格间距优化
            sell_suggestion = self._optimize_sell_grid(current_sell, metrics, market_data)
            if sell_suggestion:
                suggestions.append(sell_suggestion)

            buy_suggestion = self._optimize_buy_grid(current_buy, metrics, market_data)
            if buy_suggestion:
                suggestions.append(buy_suggestion)

            # 买卖比例优化
            ratio_suggestion = self._optimize_buy_sell_ratio(current_sell, current_buy, metrics)
            if ratio_suggestion:
                suggestions.append(ratio_suggestion)

            # 网格数量优化
            grid_count_suggestion = self._optimize_grid_count(config, metrics, market_data)
            if grid_count_suggestion:
                suggestions.append(grid_count_suggestion)

            # 仓位大小优化
            position_suggestion = self._optimize_position_size(config, metrics)
            if position_suggestion:
                suggestions.append(position_suggestion)

            return suggestions

        except Exception as e:
            logger.error(f"生成参数优化建议失败: {e}")
            return []

    def _optimize_sell_grid(self, current_sell: float, metrics: Dict,
                          market_data: pd.DataFrame) -> Dict:
        """优化卖出网格间距"""
        try:
            basic_metrics = metrics.get('basic_metrics', {})
            risk_metrics = metrics.get('risk_metrics', {})

            volatility = risk_metrics.get('volatility', 0.15)
            sharpe_ratio = basic_metrics.get('sharpe_ratio', 0)

            # 基于波动率调整
            if volatility > 0.25:  # 高波动
                recommended_sell = min(current_sell * 0.7, 3.0)
                reason = f"高波动市场({volatility:.1%})，建议缩小卖出网格间距至{recommended_sell:.1f}%以提高交易频率"
            elif volatility < 0.10:  # 低波动
                recommended_sell = max(current_sell * 1.3, 6.0)
                reason = f"低波动市场({volatility:.1%})，建议扩大卖出网格间距至{recommended_sell:.1f}%以降低交易成本"
            elif sharpe_ratio < 0.5:  # 表现不佳
                recommended_sell = max(current_sell * 1.2, 4.0)
                reason = f"当前表现不佳(夏普比率{sharpe_ratio:.2f})，建议调整卖出网格间距至{recommended_sell:.1f}%"
            else:
                return None

            return {
                'type': 'sell_grid_optimization',
                'current_value': current_sell,
                'recommended_value': recommended_sell,
                'reason': reason,
                'priority': 'high' if abs(recommended_sell - current_sell) / current_sell > 0.2 else 'medium',
                'expected_improvement': f"预计提升夏普比率10-20%"
            }

        except Exception as e:
            logger.error(f"优化卖出网格失败: {e}")
            return None

    def _optimize_buy_grid(self, current_buy: float, metrics: Dict,
                         market_data: pd.DataFrame) -> Dict:
        """优化买入网格间距"""
        try:
            trading_metrics = metrics.get('trading_metrics', {})
            basic_metrics = metrics.get('basic_metrics', {})

            win_rate = trading_metrics.get('win_rate', 0.5)
            max_drawdown = abs(basic_metrics.get('max_drawdown', 0))

            # 基于胜率和回撤调整
            if win_rate < 0.4:  # 胜率低
                recommended_buy = min(current_buy * 0.8, 6.0)
                reason = f"胜率较低({win_rate:.1%})，建议缩小买入网格间距至{recommended_buy:.1f}%以提高买入成功率"
            elif max_drawdown > 0.15:  # 回撤大
                recommended_buy = max(current_buy * 1.3, 12.0)
                reason = f"最大回撤较大({max_drawdown:.1%})，建议扩大买入网格间距至{recommended_buy:.1f}%以控制风险"
            else:
                return None

            return {
                'type': 'buy_grid_optimization',
                'current_value': current_buy,
                'recommended_value': recommended_buy,
                'reason': reason,
                'priority': 'high',
                'expected_improvement': f"预计降低最大回撤5-10%"
            }

        except Exception as e:
            logger.error(f"优化买入网格失败: {e}")
            return None

    def _optimize_buy_sell_ratio(self, current_sell: float, current_buy: float,
                               metrics: Dict) -> Dict:
        """优化买卖比例"""
        try:
            current_ratio = current_buy / current_sell
            trading_metrics = metrics.get('trading_metrics', {})

            win_rate = trading_metrics.get('win_rate', 0.5)
            avg_holding_period = trading_metrics.get('avg_holding_period', 0)

            # 理想比例范围: 1:1.5 到 1:2.0
            if current_ratio > 2.5:  # 买入网格过大
                recommended_buy = current_sell * 2.0
                reason = f"当前买入/卖出比例(1:{current_ratio:.1f})过高，建议调整至1:2.0以提高资金效率"
            elif current_ratio < 1.2:  # 买入网格过小
                recommended_buy = current_sell * 1.5
                reason = f"当前买入/卖出比例(1:{current_ratio:.1f})过低，建议调整至1:1.5以平衡风险收益"
            else:
                return None

            return {
                'type': 'buy_sell_ratio_optimization',
                'current_ratio': current_ratio,
                'recommended_ratio': recommended_buy / current_sell,
                'recommended_buy': recommended_buy,
                'reason': reason,
                'priority': 'medium'
            }

        except Exception as e:
            logger.error(f"优化买卖比例失败: {e}")
            return None

    def _optimize_grid_count(self, config: Dict, metrics: Dict,
                            market_data: pd.DataFrame) -> Dict:
        """优化网格数量"""
        try:
            base_price = config.get('base_price', 1.0)
            price_range = market_data['close'].max() - market_data['close'].min()
            price_volatility = price_range / base_price

            basic_metrics = metrics.get('basic_metrics', {})
            trading_metrics = metrics.get('trading_metrics', {})

            trading_frequency = trading_metrics.get('trading_frequency', 0)

            # 基于价格波动和交易频率调整
            if price_volatility > 0.3 and trading_frequency < 0.02:  # 高波动但交易少
                recommended_grid_count = min(15, 12)
                reason = f"高波动环境但交易频率低，建议增加网格数量至{recommended_grid_count}个以提高交易机会"
            elif price_volatility < 0.1 and trading_frequency > 0.05:  # 低波动但交易频繁
                recommended_grid_count = max(6, 8)
                reason = f"低波动环境但交易频繁，建议减少网格数量至{recommended_grid_count}个以降低交易成本"
            else:
                return None

            return {
                'type': 'grid_count_optimization',
                'current_value': config.get('grid_count', 8),
                'recommended_value': recommended_grid_count,
                'reason': reason,
                'priority': 'low'
            }

        except Exception as e:
            logger.error(f"优化网格数量失败: {e}")
            return None

    def _optimize_position_size(self, config: Dict, metrics: Dict) -> Dict:
        """优化仓位大小"""
        try:
            current_position = config.get('buy_position_size', 1000)
            base_price = config.get('base_price', 1.0)

            trading_metrics = metrics.get('trading_metrics', {})
            basic_metrics = metrics.get('basic_metrics', {})

            total_return = basic_metrics.get('total_return', 0)
            max_drawdown = abs(basic_metrics.get('max_drawdown', 0))

            # 基于收益和回撤调整
            if max_drawdown > 0.1:  # 回撤较大
                recommended_position = max(current_position * 0.7, 500)
                reason = f"最大回撤较大({max_drawdown:.1%})，建议降低单次仓位至{recommended_position}股以控制风险"
            elif total_return > 0.2 and max_drawdown < 0.05:  # 收益好且风险低
                recommended_position = min(current_position * 1.3, 2000)
                reason = f"表现优秀且回撤可控，建议增加单次仓位至{recommended_position}股以提高收益"
            else:
                return None

            return {
                'type': 'position_size_optimization',
                'current_value': current_position,
                'recommended_value': recommended_position,
                'reason': reason,
                'priority': 'medium'
            }

        except Exception as e:
            logger.error(f"优化仓位大小失败: {e}")
            return None

    def _generate_strategy_adjustment(self, metrics: Dict,
                                        market_data: pd.DataFrame) -> List[Dict]:
        """生成策略调整建议"""
        suggestions = []

        try:
            basic_metrics = metrics.get('basic_metrics', {})
            sharpe_ratio = basic_metrics.get('sharpe_ratio', 0)
            max_drawdown = basic_metrics.get('max_drawdown', 0)

            # 策略类型建议
            if sharpe_ratio < 0.3:
                suggestions.append({
                    'type': 'strategy_change',
                    'recommendation': 'dynamic_grid',
                    'reason': '当前策略表现较差，建议尝试动态网格策略以适应市场变化',
                    'priority': 'high'
                })
            elif sharpe_ratio > 1.5 and max_drawdown < 0.05:
                suggestions.append({
                    'type': 'strategy_optimization',
                    'recommendation': 'martingale_grid',
                    'reason': '当前策略表现优秀，可以考虑马丁格尔策略以进一步优化收益',
                    'priority': 'low'
                })

            # 市场适应性建议
            volatility = market_data['close'].pct_change().std()
            if volatility > 0.03:
                suggestions.append({
                    'type': 'market_adaptation',
                    'recommendation': 'increase_frequency',
                    'reason': '高波动市场，建议增加网格调整频率',
                    'priority': 'medium'
                })

            return suggestions

        except Exception as e:
            logger.error(f"生成策略调整建议失败: {e}")
            return []

    def _generate_risk_management_advice(self, config: Dict, metrics: Dict) -> List[Dict]:
        """生成风险管理建议"""
        suggestions = []

        try:
            basic_metrics = metrics.get('basic_metrics', {})
            trading_metrics = metrics.get('trading_metrics', {})

            max_drawdown = abs(basic_metrics.get('max_drawdown', 0))
            win_rate = trading_metrics.get('win_rate', 0)

            # 止损建议
            if max_drawdown > 0.2:
                suggestions.append({
                    'type': 'risk_control',
                    'recommendation': 'set_stop_loss',
                    'reason': f"最大回撤过大({max_drawdown:.1%})，建议设置-15%止损以控制风险",
                    'priority': 'high'
                })

            # 仓位控制建议
            if win_rate < 0.3:
                suggestions.append({
                    'type': 'position_control',
                    'recommendation': 'reduce_position',
                    'reason': f"胜率过低({win_rate:.1%})，建议降低单次仓位至50%",
                    'priority': 'high'
                })

            # 多样化建议
            if max_drawdown > 0.15:
                suggestions.append({
                    'type': 'diversification',
                    'recommendation': 'multiple_etfs',
                    'reason': f"单一ETF风险过高，建议分散投资多个ETF以降低整体风险",
                    'priority': 'medium'
                })

            return suggestions

        except Exception as e:
            logger.error(f"生成风险管理建议失败: {e}")
            return []

    def _generate_market_timing_advice(self, metrics: Dict,
                                        market_data: pd.DataFrame) -> List[Dict]:
        """生成市场时机建议"""
        suggestions = []

        try:
            # 计算当前市场状态
            current_price = market_data['close'].iloc[-1]
            ma20 = market_data['close'].rolling(20).mean().iloc[-1]
            ma60 = market_data['close'].rolling(60).mean().iloc[-1]

            # 趋势分析
            if current_price > ma20 and current_price > ma60:
                suggestions.append({
                    'type': 'market_timing',
                    'recommendation': 'bull_market_strategy',
                    'reason': "当前处于上升趋势，建议继续执行网格策略",
                    'priority': 'low'
                })
            elif current_price < ma20 and current_price < ma60:
                suggestions.append({
                    'type': 'market_timing',
                    'recommendation': 'bear_market_adjustment',
                    'reason': "当前处于下降趋势，建议保守设置或暂停网格交易",
                    'priority': 'high'
                })

            return suggestions

        except Exception as e:
            logger.error(f"生成市场时机建议失败: {e}")
            return []

    def _generate_overall_recommendation(self, metrics: Dict) -> str:
        """生成总体建议"""
        try:
            basic_metrics = metrics.get('basic_metrics', {})

            sharpe_ratio = basic_metrics.get('sharpe_ratio', 0)
            total_return = basic_metrics.get('total_return', 0)
            max_drawdown = abs(basic_metrics.get('max_drawdown', 0))

            if sharpe_ratio > 1.0 and total_return > 0.1 and max_drawdown < 0.1:
                return "优秀：当前策略表现良好，建议继续使用并微调优化"
            elif sharpe_ratio > 0.5 and max_drawdown < 0.15:
                return "良好：策略表现稳定，建议进行参数优化以提升收益"
            elif total_return > 0:
                return "一般：策略有一定收益但风险较高，建议重点优化风险控制"
            else:
                return "需要改进：当前策略表现不佳，建议大幅调整参数或更换策略"

        except Exception as e:
            logger.error(f"生成总体建议失败: {e}")
            return "无法评估策略表现"

    def _determine_optimization_priority(self, config: Dict, metrics: Dict) -> List[str]:
        """确定优化优先级"""
        priorities = []

        try:
            basic_metrics = metrics.get('basic_metrics', {})
            trading_metrics = metrics.get('trading_metrics', {})

            max_drawdown = abs(basic_metrics.get('max_drawdown', 0))
            win_rate = trading_metrics.get('win_rate', 0)
            sharpe_ratio = basic_metrics.get('sharpe_ratio', 0)

            # 风险控制优先
            if max_drawdown > 0.15:
                priorities.append("风险控制：降低最大回撤")
            if win_rate < 0.4:
                priorities.append("胜率提升：优化买入时机")

            # 收益优化优先
            if sharpe_ratio < 0.5:
                priorities.append("收益优化：调整参数提升夏普比率")

            # 效率优化优先
            if trading_metrics.get('trading_frequency', 0) > 0.1:
                priorities.append("成本控制：降低交易频率")

            if not priorities:
                priorities.append("保持现状：当前参数设置合理")

        except Exception as e:
            logger.error(f"确定优化优先级失败: {e}")
            priorities.append("无法确定优先级")

        return priorities

    def _get_default_advice(self) -> Dict:
        """获取默认建议"""
        return {
            'parameter_optimization': [],
            'strategy_adjustment': [],
            'risk_management': [],
            'market_timing': [],
            'overall_recommendation': "无法生成优化建议",
            'optimization_priority': []
        }

    def format_optimization_report(self, advice: Dict, config: Dict) -> str:
        """格式化优化报告"""
        try:
            report = f"=== {config.get('stock_name', 'Unknown')} 优化建议报告 ===\n\n"

            # 总体建议
            report += f"【总体评估】\n{advice.get('overall_recommendation', '')}\n\n"

            # 优化优先级
            report += "【优化优先级】\n"
            for i, priority in enumerate(advice.get('optimization_priority', []), 1):
                report += f"{i}. {priority}\n"
            report += "\n"

            # 参数优化建议
            param_suggestions = advice.get('parameter_optimization', [])
            if param_suggestions:
                report += "【参数优化建议】\n"
                for suggestion in param_suggestions:
                    report += f"- {suggestion.get('reason', '')}\n"
                    report += f"  建议: {suggestion.get('recommended_value', '')} (当前: {suggestion.get('current_value', '')})\n"
                    report += f"  预期改善: {suggestion.get('expected_improvement', '')}\n\n"

            # 其他建议
            other_suggestions = (advice.get('strategy_adjustment', []) +
                                 advice.get('risk_management', []) +
                                 advice.get('market_timing', []))

            if other_suggestions:
                report += "【其他建议】\n"
                for suggestion in other_suggestions:
                    report += f"- {suggestion.get('reason', '')}\n\n"

            report += "【执行建议】\n"
            report += "1. 优先执行高优先级优化项目\n"
            report += "2. 建议先进行小批量测试\n"
            report += "3. 监控优化效果并持续调整\n"

            return report

        except Exception as e:
            logger.error(f"格式化优化报告失败: {e}")
            return "无法生成优化报告"


def main():
    """测试函数"""
    # 创建测试数据
    config = {
        'stock_name': '测试ETF',
        'stock_code': 'TEST001',
        'base_price': 1.0,
        'sell_percentage': 5.0,
        'buy_percentage': 10.0,
        'buy_position_size': 1000,
        'grid_count': 8
    }

    metrics = {
        'basic_metrics': {
            'total_return': 0.15,
            'annual_return': 0.12,
            'max_drawdown': -0.08,
            'sharpe_ratio': 0.8
        },
        'trading_metrics': {
            'win_rate': 0.6,
            'trading_frequency': 0.03,
            'avg_holding_period': 15,
            'total_trades': 25
        },
        'risk_metrics': {
            'volatility': 0.18,
            'var_95': -0.02
        }
    }

    # 创建测试市场数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    market_data = pd.DataFrame({
        'close': np.random.normal(1.0, 0.02, 100)
    })
    market_data['date'] = dates

    # 创建优化建议生成器
    advisor = OptimizationAdvisor()
    advice = advisor.generate_comprehensive_advice(config, metrics, market_data)

    # 输出建议
    print("优化建议:")
    print(advisor.get('overall_recommendation', ''))
    print("\n优化优先级:")
    for priority in advice.get('optimization_priority', []):
        print(f"- {priority}")

    print("\n格式化报告:")
    print(advisor.format_optimization_report(advice, config))


if __name__ == "__main__":
    main()