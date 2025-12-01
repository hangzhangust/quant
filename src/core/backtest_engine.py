"""
回测引擎核心
Backtest Engine Core

负责执行网格交易策略的回测
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

from src.strategies.grid_strategy import GridStrategy, Signal, SignalType
from src.data.grid_config_parser import GridConfigParser

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Trade:
    """交易记录"""
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    amount: float
    commission: float
    slippage: float


@dataclass
class Position:
    """持仓记录"""
    symbol: str
    quantity: int
    avg_price: float
    total_cost: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class Portfolio:
    """投资组合"""
    cash: float = field(default=100000.0)  # 初始资金10万
    positions: Dict[str, Position] = field(default_factory=dict)
    total_value: float = 0.0
    total_pnl: float = 0.0


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_cash: float = 100000.0
    commission_rate: float = 0.0003  # 万分之三手续费
    slippage_rate: float = 0.001     # 千分之一滑点
    min_trade_unit: int = 100        # 最小交易单位
    position_limit: float = 0.8      # 最大仓位限制


class BacktestEngine:
    """回测引擎"""

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.portfolio = Portfolio(cash=self.config.initial_cash)
        self.trades: List[Trade] = []
        self.daily_values: List[Dict] = []
        self.signals: List[Signal] = []

    def run_backtest(self, strategy: GridStrategy, price_data: pd.DataFrame,
                    symbol: str) -> Dict:
        """运行单策略回测"""
        logger.info(f"开始回测 {symbol}, 数据长度: {len(price_data)}")

        # 重置状态
        self._reset_state()
        self.symbol = symbol

        # 生成交易信号
        signals = strategy.generate_signals(price_data)
        self.signals.extend(signals)

        # 执行回测
        for idx, row in price_data.iterrows():
            current_date = row['date']
            current_price = row['close']

            # 检查当日的交易信号
            daily_signals = [s for s in signals if s.timestamp.date() == current_date.date()]

            # 执行当日信号
            for signal in daily_signals:
                self._execute_signal(signal, current_price)

            # 更新持仓市值
            self._update_portfolio_value(current_price, current_date)

        # 计算最终结果
        results = self._calculate_results(price_data)
        logger.info(f"回测完成，总收益率: {results['total_return']:.2%}")

        return results

    def run_portfolio_backtest(self, strategies: Dict[str, GridStrategy],
                             price_data: Dict[str, pd.DataFrame]) -> Dict:
        """运行组合回测"""
        logger.info(f"开始组合回测，包含 {len(strategies)} 个策略")

        # 重置状态
        self._reset_state()

        # 获取所有交易日期
        all_dates = set()
        for data in price_data.values():
            all_dates.update(data['date'].dt.date)
        all_dates = sorted(list(all_dates))

        # 按日期执行回测
        for date in all_dates:
            for symbol, strategy in strategies.items():
                if symbol not in price_data:
                    continue

                # 获取当日价格数据
                symbol_data = price_data[symbol]
                daily_data = symbol_data[symbol_data['date'].dt.date == date]

                if daily_data.empty:
                    continue

                current_price = daily_data['close'].iloc[0]

                # 检查当日的交易信号
                daily_signals = [s for s in strategy.signals if s.timestamp.date() == date]

                # 执行当日信号
                for signal in daily_signals:
                    self._execute_signal(signal, current_price, symbol)

                # 更新持仓市值
                self._update_portfolio_value_multi(current_price, date)

        # 计算最终结果
        results = self._calculate_portfolio_results(price_data)
        logger.info(f"组合回测完成，总收益率: {results['total_return']:.2%}")

        return results

    def _reset_state(self):
        """重置回测状态"""
        self.portfolio = Portfolio(cash=self.config.initial_cash)
        self.trades = []
        self.daily_values = []
        self.signals = []

    def _execute_signal(self, signal: Signal, current_price: float, symbol: str = None):
        """执行交易信号"""
        if symbol is None:
            symbol = getattr(self, 'symbol', 'UNKNOWN')

        # 应用滑点
        execution_price = self._apply_slippage(current_price, signal.signal_type)

        # 计算交易金额和手续费
        trade_amount = signal.quantity * execution_price
        commission = trade_amount * self.config.commission_rate

        if signal.signal_type == SignalType.BUY:
            self._execute_buy(symbol, signal.quantity, execution_price, commission)
        else:  # SELL
            self._execute_sell(symbol, signal.quantity, execution_price, commission)

        # 记录交易
        trade = Trade(
            timestamp=signal.timestamp,
            symbol=symbol,
            side=OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL,
            quantity=signal.quantity,
            price=execution_price,
            amount=trade_amount,
            commission=commission,
            slippage=abs(execution_price - current_price)
        )

        self.trades.append(trade)
        logger.debug(f"执行交易: {trade.side.value} {trade.quantity} {symbol} @ {trade.price:.3f}")

    def _execute_buy(self, symbol: str, quantity: int, price: float, commission: float):
        """执行买入"""
        total_cost = quantity * price + commission

        # 检查资金充足性
        if self.portfolio.cash < total_cost:
            logger.warning(f"资金不足，无法买入 {symbol}: 需要 {total_cost:.2f}, 可用 {self.portfolio.cash:.2f}")
            return

        # 更新现金
        self.portfolio.cash -= total_cost

        # 更新持仓
        if symbol not in self.portfolio.positions:
            self.portfolio.positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                avg_price=0.0,
                total_cost=0.0
            )

        position = self.portfolio.positions[symbol]
        old_quantity = position.quantity
        old_cost = position.total_cost

        position.quantity += quantity
        position.total_cost += quantity * price
        position.avg_price = position.total_cost / position.quantity if position.quantity > 0 else 0

    def _execute_sell(self, symbol: str, quantity: int, price: float, commission: float):
        """执行卖出"""
        if symbol not in self.portfolio.positions:
            logger.warning(f"无持仓，无法卖出 {symbol}")
            return

        position = self.portfolio.positions[symbol]

        if position.quantity < quantity:
            logger.warning(f"持仓不足，无法卖出 {symbol}: 需要 {quantity}, 可用 {position.quantity}")
            quantity = position.quantity  # 卖出全部

        if quantity <= 0:
            return

        # 更新持仓
        position.quantity -= quantity
        sold_cost = quantity * position.avg_price
        position.total_cost -= sold_cost

        # 更新现金
        proceeds = quantity * price - commission
        self.portfolio.cash += proceeds

        # 如果持仓为0，删除持仓记录
        if position.quantity == 0:
            del self.portfolio.positions[symbol]

    def _apply_slippage(self, price: float, signal_type: SignalType) -> float:
        """应用滑点"""
        if signal_type == SignalType.BUY:
            return price * (1 + self.config.slippage_rate)
        else:  # SELL
            return price * (1 - self.config.slippage_rate)

    def _update_portfolio_value(self, current_price: float, date: datetime):
        """更新投资组合价值（单股票）"""
        symbol = getattr(self, 'symbol', 'UNKNOWN')

        # 计算持仓市值
        total_market_value = 0.0
        total_cost = 0.0

        if symbol in self.portfolio.positions:
            position = self.portfolio.positions[symbol]
            position.market_value = position.quantity * current_price
            position.unrealized_pnl = position.market_value - position.total_cost
            total_market_value = position.market_value
            total_cost = position.total_cost

        # 计算总价值
        self.portfolio.total_value = self.portfolio.cash + total_market_value
        self.portfolio.total_pnl = self.portfolio.total_value - self.config.initial_cash

        # 记录每日价值
        self.daily_values.append({
            'date': date,
            'cash': self.portfolio.cash,
            'market_value': total_market_value,
            'total_value': self.portfolio.total_value,
            'total_pnl': self.portfolio.total_pnl,
            'position_quantity': self.portfolio.positions[symbol].quantity if symbol in self.portfolio.positions else 0,
            'position_cost': total_cost,
            'unrealized_pnl': self.portfolio.positions[symbol].unrealized_pnl if symbol in self.portfolio.positions else 0.0
        })

    def _update_portfolio_value_multi(self, current_price: float, date: datetime):
        """更新投资组合价值（多股票）"""
        # 对于组合回测，这里需要更复杂的逻辑
        # 暂时跳过具体实现
        pass

    def _calculate_results(self, price_data: pd.DataFrame) -> Dict:
        """计算回测结果"""
        if not self.daily_values:
            return {}

        df_values = pd.DataFrame(self.daily_values)

        # 基本统计
        initial_value = self.config.initial_cash
        final_value = self.portfolio.total_value
        total_return = (final_value - initial_value) / initial_value

        # 计算日收益率
        df_values['daily_return'] = df_values['total_value'].pct_change()

        # 年化收益率
        trading_days = len(df_values)
        annual_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0

        # 最大回撤
        df_values['cumulative_max'] = df_values['total_value'].cummax()
        df_values['drawdown'] = (df_values['total_value'] - df_values['cumulative_max']) / df_values['cumulative_max']
        max_drawdown = df_values['drawdown'].min()

        # 夏普比率
        daily_returns = df_values['daily_return'].dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

        # 交易统计
        buy_trades = [t for t in self.trades if t.side == OrderSide.BUY]
        sell_trades = [t for t in self.trades if t.side == OrderSide.SELL]

        # 胜率计算
        profitable_trades = 0
        total_trades = min(len(buy_trades), len(sell_trades))

        # 计算已实现盈亏
        realized_pnl = 0.0
        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_trade = buy_trades[i]
            sell_trade = sell_trades[i]
            pnl = (sell_trade.price - buy_trade.price) * sell_trade.quantity - buy_trade.commission - sell_trade.commission
            realized_pnl += pnl
            if pnl > 0:
                profitable_trades += 1

        win_rate = profitable_trades / total_trades if total_trades > 0 else 0

        # 持仓信息
        final_position = 0
        final_cost = 0.0
        if hasattr(self, 'symbol') and self.symbol in self.portfolio.positions:
            position = self.portfolio.positions[self.symbol]
            final_position = position.quantity
            final_cost = position.total_cost

        results = {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'realized_pnl': realized_pnl,
            'total_commission': sum(t.commission for t in self.trades),
            'total_slippage': sum(t.slippage * t.quantity for t in self.trades),
            'initial_value': initial_value,
            'final_value': final_value,
            'trading_days': trading_days,
            'final_position': final_position,
            'final_cost': final_cost,
            'daily_values': df_values,
            'trades': self.trades,
            'signals': self.signals
        }

        return results

    def _calculate_portfolio_results(self, price_data: Dict[str, pd.DataFrame]) -> Dict:
        """计算组合回测结果"""
        # 暂时返回基本结果
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'total_trades': len(self.trades),
            'win_rate': 0.0,
            'realized_pnl': 0.0,
            'total_commission': sum(t.commission for t in self.trades),
            'initial_value': self.config.initial_cash,
            'final_value': self.portfolio.total_value,
            'trades': self.trades
        }

    def get_trade_history(self) -> pd.DataFrame:
        """获取交易历史"""
        if not self.trades:
            return pd.DataFrame()

        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'quantity': trade.quantity,
                'price': trade.price,
                'amount': trade.amount,
                'commission': trade.commission,
                'slippage': trade.slippage
            })

        return pd.DataFrame(trades_data)

    def get_daily_performance(self) -> pd.DataFrame:
        """获取每日业绩"""
        if not self.daily_values:
            return pd.DataFrame()

        return pd.DataFrame(self.daily_values)


def main():
    """测试函数"""
    from ..strategies.grid_strategy import create_grid_strategy

    # 创建测试策略
    config = {
        'base_price': 1.408,
        'sell_percentage': 5.0,
        'buy_percentage': 10.0,
        'position_size': 1000,
        'grid_count': 8
    }

    strategy = create_grid_strategy('basic_grid', config)

    # 创建测试数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = np.random.normal(1.408, 0.05, 100)

    price_data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'open': prices,
        'high': prices * 1.02,
        'low': prices * 0.98
    })

    # 运行回测
    engine = BacktestEngine()
    results = engine.run_backtest(strategy, price_data, '159682')

    print("回测结果:")
    print(f"总收益率: {results['total_return']:.2%}")
    print(f"年化收益率: {results['annual_return']:.2%}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"交易次数: {results['total_trades']}")
    print(f"胜率: {results['win_rate']:.2%}")


if __name__ == "__main__":
    main()