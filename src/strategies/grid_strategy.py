"""
网格交易策略实现
Grid Trading Strategy Implementation

包含三种网格策略：基础网格、动态网格、马丁格尔网格
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, NamedTuple
import pandas as pd
import numpy as np
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """交易信号类型"""
    BUY = "buy"
    SELL = "sell"


class Signal(NamedTuple):
    """交易信号"""
    timestamp: datetime
    signal_type: SignalType
    price: float
    quantity: int
    reason: str


class GridLevel:
    """网格价位"""
    def __init__(self, price: float, level_type: str, quantity: int = 0):
        self.price = price
        self.level_type = level_type  # 'buy' or 'sell'
        self.quantity = quantity
        self.triggered = False
        self.trigger_time: Optional[datetime] = None


class GridStrategy(ABC):
    """网格策略基类"""

    def __init__(self, config: Dict):
        self.config = config
        self.base_price = config['base_price']
        self.sell_percentage = config['sell_percentage'] / 100
        self.buy_percentage = config['buy_percentage'] / 100
        self.position_size = config.get('position_size', 10000)
        self.position_size_type = config.get('position_size_type', 'shares')
        self.max_position_ratio = config.get('max_position_ratio', 0.8)

        # 网格价位列表
        self.grid_levels: List[GridLevel] = []
        self.current_position = 0
        self.total_cost = 0.0
        self.last_signal_price = 0.0

        # 交易记录
        self.signals: List[Signal] = []
        self.positions: List[Dict] = []

    @abstractmethod
    def generate_signals(self, price_data: pd.DataFrame) -> List[Signal]:
        """生成交易信号"""
        pass

    @abstractmethod
    def _calculate_grid_levels(self, base_price: float) -> List[GridLevel]:
        """计算网格价位"""
        pass

    def _calculate_position_size(self, price: float, signal_type: SignalType) -> int:
        """计算交易数量"""
        if self.position_size_type == 'shares':
            return self.position_size
        else:  # amount
            return int(self.position_size / price)

    def _should_trigger_signal(self, current_price: float, grid_level: GridLevel,
                              signal_type: SignalType) -> bool:
        """判断是否应该触发信号"""
        if grid_level.triggered:
            return False

        if signal_type == SignalType.BUY:
            return current_price <= grid_level.price
        else:  # SELL
            return current_price >= grid_level.price

    def _get_current_position_value(self, current_price: float) -> float:
        """计算当前持仓价值"""
        return self.current_position * current_price

    def _get_current_pnl(self, current_price: float) -> float:
        """计算当前盈亏"""
        if self.current_position == 0:
            return 0.0
        current_value = self.current_position * current_price
        return current_value - self.total_cost

    def reset(self):
        """重置策略状态"""
        self.grid_levels = []
        self.current_position = 0
        self.total_cost = 0.0
        self.signals = []
        self.positions = []
        self.last_signal_price = 0.0


class BasicGridStrategy(GridStrategy):
    """基础网格策略"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.grid_count = config.get('grid_count', 10)
        self.grid_levels = self._calculate_grid_levels(self.base_price)

    def _calculate_grid_levels(self, base_price: float) -> List[GridLevel]:
        """计算基础网格价位"""
        levels = []

        # 计算买入网格（向下）- 使用等间距
        for i in range(1, self.grid_count // 2 + 1):
            buy_price = base_price * (1 - self.buy_percentage * i / (self.grid_count // 2))
            levels.append(GridLevel(buy_price, 'buy', self.position_size))

        # 计算卖出网格（向上）- 使用等间距
        for i in range(1, self.grid_count // 2 + 1):
            sell_price = base_price * (1 + self.sell_percentage * i / (self.grid_count // 2))
            levels.append(GridLevel(sell_price, 'sell', self.position_size))

        # 按价格排序
        levels.sort(key=lambda x: x.price)
        return levels

    def generate_signals(self, price_data: pd.DataFrame) -> List[Signal]:
        """生成基础网格交易信号"""
        signals = []

        for idx, row in price_data.iterrows():
            current_price = row['close']
            current_time = row['date']

            # 检查买入信号
            buy_levels = [level for level in self.grid_levels
                         if level.level_type == 'buy' and not level.triggered]
            for level in buy_levels:
                if self._should_trigger_signal(current_price, level, SignalType.BUY):
                    quantity = self._calculate_position_size(current_price, SignalType.BUY)

                    signal = Signal(
                        timestamp=current_time,
                        signal_type=SignalType.BUY,
                        price=current_price,
                        quantity=quantity,
                        reason=f"触发买入网格 {level.price:.3f}"
                    )

                    signals.append(signal)
                    level.triggered = True
                    level.trigger_time = current_time

                    # 更新持仓
                    self.current_position += quantity
                    self.total_cost += quantity * current_price
                    self.last_signal_price = current_price

            # 检查卖出信号（只有持仓时才卖出）
            if self.current_position > 0:
                sell_levels = [level for level in self.grid_levels
                              if level.level_type == 'sell' and not level.triggered]
                for level in sell_levels:
                    if self._should_trigger_signal(current_price, level, SignalType.SELL):
                        quantity = min(self._calculate_position_size(current_price, SignalType.SELL),
                                     self.current_position)

                        signal = Signal(
                            timestamp=current_time,
                            signal_type=SignalType.SELL,
                            price=current_price,
                            quantity=quantity,
                            reason=f"触发卖出网格 {level.price:.3f}"
                        )

                        signals.append(signal)
                        level.triggered = True
                        level.trigger_time = current_time

                        # 更新持仓
                        avg_cost = self.total_cost / self.current_position if self.current_position > 0 else 0
                        self.total_cost -= quantity * avg_cost
                        self.current_position -= quantity
                        self.last_signal_price = current_price

        self.signals.extend(signals)
        return signals


class DynamicGridStrategy(GridStrategy):
    """动态网格策略"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.grid_count = config.get('grid_count', 10)
        self.volatility_window = config.get('volatility_window', 20)
        self.volatility_threshold = config.get('volatility_threshold', 0.02)
        self.grid_levels = []
        self.last_rebalance_time = None

    def generate_signals(self, price_data: pd.DataFrame) -> List[Signal]:
        """生成动态网格交易信号"""
        signals = []

        for idx, row in price_data.iterrows():
            current_price = row['close']
            current_time = row['date']

            # 定期重新计算网格
            if self._should_rebalance(price_data.iloc[:idx+1], current_time):
                self._recalculate_grid_levels(price_data.iloc[:idx+1])

            # 生成信号逻辑与基础网格相同
            signals.extend(self._generate_grid_signals(current_price, current_time))

        self.signals.extend(signals)
        return signals

    def _should_rebalance(self, historical_data: pd.DataFrame, current_time: datetime) -> bool:
        """判断是否需要重新平衡"""
        if self.last_rebalance_time is None:
            self.last_rebalance_time = current_time
            return True

        # 计算最近波动率
        if len(historical_data) >= self.volatility_window:
            returns = historical_data['close'].pct_change().tail(self.volatility_window)
            current_volatility = returns.std()

            # 如果波动率变化超过阈值，重新平衡
            if current_volatility > self.volatility_threshold:
                return True

        return False

    def _recalculate_grid_levels(self, historical_data: pd.DataFrame):
        """重新计算网格价位"""
        current_price = historical_data['close'].iloc[-1]

        # 根据波动率调整网格数量
        if len(historical_data) >= self.volatility_window:
            returns = historical_data['close'].pct_change().tail(self.volatility_window)
            current_volatility = returns.std()

            # 高波动率时增加网格数量
            if current_volatility > self.volatility_threshold:
                adjusted_grid_count = min(self.grid_count * 2, 20)
            else:
                adjusted_grid_count = max(self.grid_count // 2, 5)
        else:
            adjusted_grid_count = self.grid_count

        self.grid_levels = self._calculate_grid_levels(current_price, adjusted_grid_count)

    def _calculate_grid_levels(self, base_price: float, grid_count: int = None) -> List[GridLevel]:
        """计算动态网格价位"""
        if grid_count is None:
            grid_count = self.grid_count

        levels = []

        # 根据当前市场状况调整间距
        for i in range(1, grid_count // 2 + 1):
            buy_price = base_price * (1 - self.buy_percentage * i)
            levels.append(GridLevel(buy_price, 'buy', self.position_size))

        for i in range(1, grid_count // 2 + 1):
            sell_price = base_price * (1 + self.sell_percentage * i)
            levels.append(GridLevel(sell_price, 'sell', self.position_size))

        levels.sort(key=lambda x: x.price)
        return levels

    def _generate_grid_signals(self, current_price: float, current_time: datetime) -> List[Signal]:
        """生成网格信号"""
        signals = []

        # 检查买入信号
        buy_levels = [level for level in self.grid_levels
                     if level.level_type == 'buy' and not level.triggered]
        for level in buy_levels:
            if self._should_trigger_signal(current_price, level, SignalType.BUY):
                quantity = self._calculate_position_size(current_price, SignalType.BUY)

                signal = Signal(
                    timestamp=current_time,
                    signal_type=SignalType.BUY,
                    price=current_price,
                    quantity=quantity,
                    reason=f"动态网格买入 {level.price:.3f}"
                )

                signals.append(signal)
                level.triggered = True
                level.trigger_time = current_time

                self.current_position += quantity
                self.total_cost += quantity * current_price
                self.last_signal_price = current_price

        # 检查卖出信号
        if self.current_position > 0:
            sell_levels = [level for level in self.grid_levels
                          if level.level_type == 'sell' and not level.triggered]
            for level in sell_levels:
                if self._should_trigger_signal(current_price, level, SignalType.SELL):
                    quantity = min(self._calculate_position_size(current_price, SignalType.SELL),
                                 self.current_position)

                    signal = Signal(
                        timestamp=current_time,
                        signal_type=SignalType.SELL,
                        price=current_price,
                        quantity=quantity,
                        reason=f"动态网格卖出 {level.price:.3f}"
                    )

                    signals.append(signal)
                    level.triggered = True
                    level.trigger_time = current_time

                    avg_cost = self.total_cost / self.current_position if self.current_position > 0 else 0
                    self.total_cost -= quantity * avg_cost
                    self.current_position -= quantity
                    self.last_signal_price = current_price

        return signals


class MartingaleGridStrategy(GridStrategy):
    """马丁格尔网格策略"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.grid_count = config.get('grid_count', 10)
        self.martingale_factor = config.get('martingale_factor', 2.0)
        self.max_martingale_levels = config.get('max_martingale_levels', 5)
        self.current_martingale_level = 0
        self.consecutive_losses = 0
        self.grid_levels = self._calculate_grid_levels(self.base_price)

    def _calculate_grid_levels(self, base_price: float) -> List[GridLevel]:
        """计算马丁格尔网格价位"""
        levels = []

        # 计算买入网格（递增间距）
        for i in range(1, self.grid_count // 2 + 1):
            buy_price = base_price * (1 - self.buy_percentage * i)
            # 马丁格尔：越跌越买，买入量递增
            quantity = self.position_size * min(self.martingale_factor ** (i-1),
                                              self.martingale_factor ** self.max_martingale_levels)
            levels.append(GridLevel(buy_price, 'buy', int(quantity)))

        # 计算卖出网格
        for i in range(1, self.grid_count // 2 + 1):
            sell_price = base_price * (1 + self.sell_percentage * i)
            levels.append(GridLevel(sell_price, 'sell', self.position_size))

        levels.sort(key=lambda x: x.price)
        return levels

    def generate_signals(self, price_data: pd.DataFrame) -> List[Signal]:
        """生成马丁格尔网格交易信号"""
        signals = []

        for idx, row in price_data.iterrows():
            current_price = row['close']
            current_time = row['date']

            # 检查买入信号
            buy_levels = [level for level in self.grid_levels
                         if level.level_type == 'buy' and not level.triggered]
            for level in buy_levels:
                if self._should_trigger_signal(current_price, level, SignalType.BUY):
                    signal = Signal(
                        timestamp=current_time,
                        signal_type=SignalType.BUY,
                        price=current_price,
                        quantity=level.quantity,
                        reason=f"马丁格尔买入L{self.current_martingale_level + 1}"
                    )

                    signals.append(signal)
                    level.triggered = True
                    level.trigger_time = current_time

                    self.current_position += level.quantity
                    self.total_cost += level.quantity * current_price
                    self.current_martingale_level += 1
                    self.consecutive_losses += 1
                    self.last_signal_price = current_price

            # 检查卖出信号
            if self.current_position > 0:
                sell_levels = [level for level in self.grid_levels
                              if level.level_type == 'sell' and not level.triggered]
                for level in sell_levels:
                    if self._should_trigger_signal(current_price, level, SignalType.SELL):
                        quantity = min(level.quantity, self.current_position)

                        signal = Signal(
                            timestamp=current_time,
                            signal_type=SignalType.SELL,
                            price=current_price,
                            quantity=quantity,
                            reason=f"马丁格尔卖出"
                        )

                        signals.append(signal)
                        level.triggered = True
                        level.trigger_time = current_time

                        avg_cost = self.total_cost / self.current_position if self.current_position > 0 else 0
                        self.total_cost -= quantity * avg_cost
                        self.current_position -= quantity

                        # 重置马丁格尔计数器
                        if current_price > self.last_signal_price:
                            self.current_martingale_level = 0
                            self.consecutive_losses = 0

                        self.last_signal_price = current_price

        self.signals.extend(signals)
        return signals


def create_grid_strategy(strategy_type: str, config: Dict) -> GridStrategy:
    """工厂函数：创建网格策略实例"""
    strategies = {
        'basic_grid': BasicGridStrategy,
        'dynamic_grid': DynamicGridStrategy,
        'martingale_grid': MartingaleGridStrategy
    }

    if strategy_type not in strategies:
        raise ValueError(f"不支持的策略类型: {strategy_type}")

    return strategies[strategy_type](config)


def main():
    """测试函数"""
    # 测试配置
    config = {
        'base_price': 1.408,
        'sell_percentage': 5.0,
        'buy_percentage': 10.0,
        'position_size': 1000,
        'position_size_type': 'shares',
        'grid_count': 8
    }

    # 创建基础网格策略
    strategy = create_grid_strategy('basic_grid', config)

    # 创建模拟价格数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = np.random.normal(1.408, 0.05, 100)

    price_data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'open': prices,
        'high': prices * 1.02,
        'low': prices * 0.98
    })

    # 生成信号
    signals = strategy.generate_signals(price_data)

    print(f"策略测试完成，生成 {len(signals)} 个交易信号")
    for signal in signals[:5]:  # 显示前5个信号
        print(f"{signal.timestamp}: {signal.signal_type.value} {signal.quantity}@{signal.price:.3f}")


if __name__ == "__main__":
    main()