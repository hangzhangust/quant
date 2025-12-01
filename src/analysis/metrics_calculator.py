"""
完整指标计算器
Comprehensive Metrics Calculator

计算网格交易回测的完整指标体系，包括基础指标、交易指标和风险指标
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """完整指标计算器"""

    def __init__(self, benchmark_data: pd.DataFrame = None):
        """
        初始化指标计算器

        Args:
            benchmark_data: 基准数据（用于计算贝塔系数）
        """
        self.benchmark_data = benchmark_data
        self.risk_free_rate = 0.03  # 无风险利率3%

    def calculate_comprehensive_metrics(self, backtest_results: Dict,
                                      price_data: pd.DataFrame) -> Dict:
        """
        计算完整的回测指标

        Args:
            backtest_results: 回测结果
            price_data: 价格数据

        Returns:
            包含所有指标的字典
        """
        try:
            metrics = {
                'basic_metrics': self._calculate_basic_metrics(backtest_results, price_data),
                'trading_metrics': self._calculate_trading_metrics(backtest_results),
                'risk_metrics': self._calculate_risk_metrics(backtest_results, price_data),
                'efficiency_metrics': self._calculate_efficiency_metrics(backtest_results, price_data)
            }

            logger.info(f"成功计算完整指标，包含{len(self._flatten_metrics(metrics))}个指标")
            return metrics

        except Exception as e:
            logger.error(f"计算完整指标失败: {e}")
            return self._get_empty_metrics()

    def _calculate_basic_metrics(self, backtest_results: Dict, price_data: pd.DataFrame) -> Dict:
        """计算基础指标"""
        try:
            initial_value = backtest_results.get('initial_value', 100000)
            final_value = backtest_results.get('final_value', initial_value)
            trading_days = backtest_results.get('trading_days', len(price_data))

            # 总收益率
            total_return = (final_value - initial_value) / initial_value

            # 年化收益率
            if trading_days > 0:
                annual_return = (1 + total_return) ** (252 / trading_days) - 1
            else:
                annual_return = 0

            # 最大回撤
            if 'daily_values' in backtest_results and not backtest_results['daily_values'].empty:
                daily_values = backtest_results['daily_values']
                daily_values['cumulative_max'] = daily_values['total_value'].cummax()
                daily_values['drawdown'] = (daily_values['total_value'] - daily_values['cumulative_max']) / daily_values['cumulative_max']
                max_drawdown = daily_values['drawdown'].min()
            else:
                max_drawdown = 0

            # 夏普比率
            if 'daily_values' in backtest_results and not backtest_results['daily_values'].empty:
                daily_returns = backtest_results['daily_values']['total_value'].pct_change().dropna()
                if daily_returns.std() > 0:
                    sharpe_ratio = (annual_return - self.risk_free_rate) / (daily_returns.std() * np.sqrt(252))
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0

            # 索提诺比率（只考虑下行波动）
            if 'daily_values' in backtest_results and not backtest_results['daily_values'].empty:
                daily_returns = backtest_results['daily_values']['total_value'].pct_change().dropna()
                downside_returns = daily_returns[daily_returns < 0]
                if len(downside_returns) > 0 and downside_returns.std() > 0:
                    sortino_ratio = (annual_return - self.risk_free_rate) / (downside_returns.std() * np.sqrt(252))
                else:
                    sortino_ratio = 0
            else:
                sortino_ratio = 0

            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': annual_return / abs(max_drawdown) if max_drawdown != 0 else 0,
                'total_pnl': final_value - initial_value,
                'final_value': final_value
            }

        except Exception as e:
            logger.error(f"计算基础指标失败: {e}")
            return {}

    def _calculate_trading_metrics(self, backtest_results: Dict) -> Dict:
        """计算交易指标"""
        try:
            trades = backtest_results.get('trades', [])
            signals = backtest_results.get('signals', [])

            if not trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'profit_loss_ratio': 0,
                    'avg_trade_pnl': 0,
                    'avg_holding_period': 0,
                    'total_commission': 0,
                    'total_slippage': 0
                }

            # 分类买卖交易
            buy_trades = [t for t in trades if t.side.value == 'buy']
            sell_trades = [t for t in trades if t.side.value == 'sell']

            total_trades = min(len(buy_trades), len(sell_trades))
            total_commission = sum(t.commission for t in trades)
            total_slippage = sum(t.slippage for t in trades)

            # 计算每笔交易的盈亏
            trade_pnls = []
            holding_periods = []

            for i in range(min(len(buy_trades), len(sell_trades))):
                buy_trade = buy_trades[i]
                sell_trade = sell_trades[i]

                pnl = (sell_trade.price - buy_trade.price) * sell_trade.quantity - buy_trade.commission - sell_trade.commission
                trade_pnls.append(pnl)

                # 计算持仓时间
                holding_period = (sell_trade.timestamp - buy_trade.timestamp).days
                holding_periods.append(holding_period)

            # 胜率和盈亏比
            profitable_trades = [pnl for pnl in trade_pnls if pnl > 0]
            losing_trades = [pnl for pnl in trade_pnls if pnl < 0]

            win_rate = len(profitable_trades) / len(trade_pnls) if trade_pnls else 0
            avg_trade_pnl = np.mean(trade_pnls) if trade_pnls else 0

            if losing_trades:
                avg_loss = np.mean(losing_trades)
                profit_loss_ratio = -np.mean(profitable_trades) / avg_loss if avg_loss != 0 else 0
            else:
                profit_loss_ratio = float('inf') if profitable_trades else 0

            avg_holding_period = np.mean(holding_periods) if holding_periods else 0

            return {
                'total_trades': total_trades,
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'win_rate': win_rate,
                'profit_loss_ratio': profit_loss_ratio,
                'avg_trade_pnl': avg_trade_pnl,
                'avg_holding_period': avg_holding_period,
                'total_commission': total_commission,
                'total_slippage': total_slippage,
                'max_consecutive_wins': self._calculate_max_consecutive(trade_pnls, True),
                'max_consecutive_losses': self._calculate_max_consecutive(trade_pnls, False)
            }

        except Exception as e:
            logger.error(f"计算交易指标失败: {e}")
            return {}

    def _calculate_risk_metrics(self, backtest_results: Dict, price_data: pd.DataFrame) -> Dict:
        """计算风险指标"""
        try:
            if 'daily_values' not in backtest_results or backtest_results['daily_values'].empty:
                return self._get_empty_risk_metrics()

            daily_values = backtest_results['daily_values']
            daily_returns = daily_values['total_value'].pct_change().dropna()

            if daily_returns.empty:
                return self._get_empty_risk_metrics()

            # 波动率
            volatility = daily_returns.std() * np.sqrt(252)

            # VaR (Value at Risk) - 95%置信水平
            var_95 = daily_returns.quantile(0.05)

            # CVaR (Conditional Value at Risk) - 期望短缺
            cvar_95 = daily_returns[daily_returns <= var_95].mean() if var_95 is not None else 0

            # 最大连续亏损天数
            consecutive_losses = self._calculate_max_consecutive_losses(daily_returns)

            # 贝塔系数
            beta = self._calculate_beta(daily_returns)

            # 信息比率（相对基准的超额收益）
            information_ratio = self._calculate_information_ratio(daily_returns)

            # 下行风险
            downside_deviation = self._calculate_downside_deviation(daily_returns)

            return {
                'volatility': volatility,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'max_consecutive_losses': consecutive_losses,
                'beta': beta,
                'information_ratio': information_ratio,
                'downside_deviation': downside_deviation,
                'tracking_error': self._calculate_tracking_error(daily_returns),
                'upside_capture': self._calculate_capture_ratio(daily_returns, True),
                'downside_capture': self._calculate_capture_ratio(daily_returns, False)
            }

        except Exception as e:
            logger.error(f"计算风险指标失败: {e}")
            return self._get_empty_risk_metrics()

    def _calculate_efficiency_metrics(self, backtest_results: Dict, price_data: pd.DataFrame) -> Dict:
        """计算效率指标"""
        try:
            initial_value = backtest_results.get('initial_value', 100000)
            final_value = backtest_results.get('final_value', initial_value)
            trading_days = backtest_results.get('trading_days', len(price_data))

            total_return = (final_value - initial_value) / initial_value
            max_drawdown = backtest_results.get('max_drawdown', 0)

            # 收益回撤比
            return_drawdown_ratio = -total_return / max_drawdown if max_drawdown != 0 else 0

            # 交易频率
            total_trades = backtest_results.get('total_trades', 0)
            trading_frequency = total_trades / trading_days if trading_days > 0 else 0

            # 资金利用率
            trades = backtest_results.get('trades', [])
            if trades:
                avg_position_value = np.mean([t.amount for t in trades])
                avg_utilization = avg_position_value / initial_value
            else:
                avg_utilization = 0

            # 成本收益比
            total_commission = backtest_results.get('total_commission', 0)
            total_slippage = backtest_results.get('total_slippage', 0)
            total_cost = total_commission + total_slippage
            cost_benefit_ratio = total_cost / abs(final_value - initial_value) if final_value != initial_value else 0

            return {
                'return_drawdown_ratio': return_drawdown_ratio,
                'trading_frequency': trading_frequency,
                'avg_utilization': avg_utilization,
                'cost_benefit_ratio': cost_benefit_ratio,
                'profit_per_trade': (final_value - initial_value - total_cost) / total_trades if total_trades > 0 else 0
            }

        except Exception as e:
            logger.error(f"计算效率指标失败: {e}")
            return {}

    def _calculate_beta(self, daily_returns: pd.Series) -> float:
        """计算贝塔系数"""
        try:
            if self.benchmark_data is None or self.benchmark_data.empty:
                return 1.0  # 默认贝塔系数

            # 计算基准收益率
            benchmark_returns = self.benchmark_data['close'].pct_change().dropna()

            # 对齐数据
            if len(daily_returns) != len(benchmark_returns):
                min_length = min(len(daily_returns), len(benchmark_returns))
                daily_returns = daily_returns.iloc[-min_length:]
                benchmark_returns = benchmark_returns.iloc[-min_length:]

            if len(daily_returns) < 10 or len(benchmark_returns) < 10:
                return 1.0

            # 计算协方差和方差
            covariance = np.cov(daily_returns, benchmark_returns)[0, 1]
            variance = np.var(benchmark_returns)

            if variance == 0:
                return 1.0

            return covariance / variance

        except Exception as e:
            logger.error(f"计算贝塔系数失败: {e}")
            return 1.0

    def _calculate_information_ratio(self, daily_returns: pd.Series) -> float:
        """计算信息比率"""
        try:
            if self.benchmark_data is None or self.benchmark_data.empty:
                return 0.0

            benchmark_returns = self.benchmark_data['close'].pct_change().dropna()

            # 对齐数据
            if len(daily_returns) != len(benchmark_returns):
                min_length = min(len(daily_returns), len(benchmark_returns))
                daily_returns = daily_returns.iloc[-min_length:]
                benchmark_returns = benchmark_returns.iloc[-min_length:]

            excess_returns = daily_returns - benchmark_returns

            if len(excess_returns) < 10 or excess_returns.std() == 0:
                return 0.0

            return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        except Exception as e:
            logger.error(f"计算信息比率失败: {e}")
            return 0.0

    def _calculate_downside_deviation(self, daily_returns: pd.Series) -> float:
        """计算下行偏差"""
        try:
            negative_returns = daily_returns[daily_returns < 0]
            if len(negative_returns) < 2:
                return 0.0

            return negative_returns.std() * np.sqrt(252)

        except Exception as e:
            logger.error(f"计算下行偏差失败: {e}")
            return 0.0

    def _calculate_tracking_error(self, daily_returns: pd.Series) -> float:
        """计算跟踪误差"""
        try:
            if self.benchmark_data is None or self.benchmark_data.empty:
                return 0.0

            benchmark_returns = self.benchmark_data['close'].pct_change().dropna()

            # 对齐数据
            if len(daily_returns) != len(benchmark_returns):
                min_length = min(len(daily_returns), len(benchmark_returns))
                daily_returns = daily_returns.iloc[-min_length:]
                benchmark_returns = benchmark_returns.iloc[-min_length:]

            excess_returns = daily_returns - benchmark_returns
            return excess_returns.std() * np.sqrt(252)

        except Exception as e:
            logger.error(f"计算跟踪误差失败: {e}")
            return 0.0

    def _calculate_capture_ratio(self, daily_returns: pd.Series, upside: bool = True) -> float:
        """计算捕获比率"""
        try:
            if self.benchmark_data is None or self.benchmark_data.empty:
                return 1.0

            benchmark_returns = self.benchmark_data['close'].pct_change().dropna()

            # 对齐数据
            if len(daily_returns) != len(benchmark_returns):
                min_length = min(len(daily_returns), len(benchmark_returns))
                daily_returns = daily_returns.iloc[-min_length:]
                benchmark_returns = benchmark_returns.iloc[-min_length:]

            if upside:
                # 上行捕获比率
                strategy_up = daily_returns[benchmark_returns > 0].mean()
                benchmark_up = benchmark_returns[benchmark_returns > 0].mean()
            else:
                # 下行捕获比率
                strategy_down = daily_returns[benchmark_returns < 0].mean()
                benchmark_down = benchmark_returns[benchmark_returns < 0].mean()
                return strategy_down / benchmark_down if benchmark_down != 0 else 0

            return strategy_up / benchmark_up if benchmark_up != 0 else 1.0

        except Exception as e:
            logger.error(f"计算捕获比率失败: {e}")
            return 1.0

    def _calculate_max_consecutive(self, values: List[float], wins: bool = True) -> int:
        """计算最大连续盈亏次数"""
        try:
            if not values:
                return 0

            max_consecutive = 0
            current_consecutive = 0

            for value in values:
                if wins and value > 0:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                elif not wins and value < 0:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0

            return max_consecutive

        except Exception as e:
            logger.error(f"计算最大连续次数失败: {e}")
            return 0

    def _calculate_max_consecutive_losses(self, daily_returns: pd.Series) -> int:
        """计算最大连续亏损天数"""
        try:
            if daily_returns.empty:
                return 0

            max_consecutive = 0
            current_consecutive = 0

            for return_val in daily_returns:
                if return_val < 0:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0

            return max_consecutive

        except Exception as e:
            logger.error(f"计算最大连续亏损天数失败: {e}")
            return 0

    def _get_empty_metrics(self) -> Dict:
        """获取空指标字典"""
        return {
            'basic_metrics': {},
            'trading_metrics': {},
            'risk_metrics': {},
            'efficiency_metrics': {}
        }

    def _get_empty_risk_metrics(self) -> Dict:
        """获取空风险指标字典"""
        return {
            'volatility': 0,
            'var_95': 0,
            'cvar_95': 0,
            'max_consecutive_losses': 0,
            'beta': 1.0,
            'information_ratio': 0,
            'downside_deviation': 0,
            'tracking_error': 0,
            'upside_capture': 1.0,
            'downside_capture': 1.0
        }

    def _flatten_metrics(self, metrics: Dict) -> Dict:
        """展平指标字典"""
        flattened = {}
        for category, category_metrics in metrics.items():
            if isinstance(category_metrics, dict):
                for key, value in category_metrics.items():
                    flattened[key] = value
            else:
                flattened[category] = category_metrics
        return flattened

    def get_performance_summary(self, metrics: Dict) -> str:
        """获取性能摘要字符串"""
        try:
            basic = metrics.get('basic_metrics', {})
            trading = metrics.get('trading_metrics', {})
            risk = metrics.get('risk_metrics', {})

            summary = f"""
性能摘要:
- 总收益率: {basic.get('total_return', 0):+.2%}
- 年化收益率: {basic.get('annual_return', 0):+.2%}
- 最大回撤: {basic.get('max_drawdown', 0):+.2%}
- 夏普比率: {basic.get('sharpe_ratio', 0):.2f}
- 总交易次数: {trading.get('total_trades', 0)}
- 胜率: {trading.get('win_rate', 0):.2%}
- 年化波动率: {risk.get('volatility', 0):.2%}
- VaR(95%): {risk.get('var_95', 0):.4f}
            """
            return summary.strip()

        except Exception as e:
            logger.error(f"生成性能摘要失败: {e}")
            return "无法生成性能摘要"


def main():
    """测试函数"""
    # 创建测试数据
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    daily_values = pd.DataFrame({
        'date': dates,
        'total_value': 100000 * (1 + np.random.normal(0.001, 0.02, 252).cumsum())
    })

    # 创建测试交易记录
    from core.backtest_engine import Trade, OrderSide
    trades = [
        Trade(
            timestamp=datetime.now() - timedelta(days=10),
            symbol='TEST',
            side=OrderSide.BUY,
            quantity=1000,
            price=10.0,
            amount=10000,
            commission=3.0,
            slippage=0.01
        ),
        Trade(
            timestamp=datetime.now() - timedelta(days=5),
            symbol='TEST',
            side=OrderSide.SELL,
            quantity=1000,
            price=10.5,
            amount=10500,
            commission=3.15,
            slippage=0.01
        )
    ]

    backtest_results = {
        'initial_value': 100000,
        'final_value': 105000,
        'daily_values': daily_values,
        'trades': trades,
        'trading_days': 252
    }

    price_data = pd.DataFrame({
        'date': dates,
        'close': np.random.normal(100, 5, 252)
    })

    # 测试指标计算
    calculator = MetricsCalculator(price_data)
    metrics = calculator.calculate_comprehensive_metrics(backtest_results, price_data)

    print("指标计算测试:")
    print(calculator.get_performance_summary(metrics))

if __name__ == "__main__":
    main()