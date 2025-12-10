# XTQuant 数据源集成文档

## 概述

本文档介绍XTQuant数据源在量化交易系统中的集成和使用方法。XTQuant是一个高质量的免费数据源，提供中国市场的实时和历史数据，支持ETF、股票、指数等多种金融产品。

## 功能特性

### 📈 历史数据获取
- **自动下载**: 首次访问时自动下载历史数据到本地
- **增量更新**: 支持增量数据更新，提高效率
- **多周期支持**: 支持日线数据获取
- **数据缓存**: 本地缓存机制，减少重复下载

### 📡 实时数据订阅
- **实时推送**: 支持实时市场数据推送
- **回调机制**: 自定义回调函数处理实时数据
- **多标的订阅**: 同时订阅多个标的的实时数据
- **订阅管理**: 完整的订阅状态管理和清理功能

### 🎯 系统集成
- **优先级配置**: 配置为优先级9的高质量免费数据源
- **无缝切换**: 与其他数据源无缝集成，自动切换
- **错误处理**: 完善的错误处理和重试机制
- **数据标准化**: 统一的数据格式和质量验证

## 安装和配置

### 1. 安装XTQuant

```bash
pip install xtquant
```

### 2. MiniQmt环境要求

XTQuant需要MiniQmt客户端支持：

1. **下载MiniQmt**: 从迅投官网下载MiniQmt客户端
2. **安装配置**: 按照MiniQmt安装指南完成配置
3. **启动服务**: 确保MiniQmt服务正常运行

### 3. 环境变量配置

在`.env`文件中添加以下配置：

```bash
# XTQuant配置
XTQUANT_ENABLED=true                    # 启用XTQuant数据源
# 可选配置
XTQUANT_CACHE_DIR=./xtquant_data       # 本地数据缓存目录
XTQUANT_AUTO_DOWNLOAD=true             # 自动下载历史数据
XTQUANT_REALTIME_ENABLED=true          # 启用实时数据功能
```

### 4. 数据源优先级配置

XTQuant默认配置为优先级9，在数据源优先级中的位置：

```
1. jqdatasdk (优先级 1)    - 付费，最高质量
2. tushare (优先级 2)       - 付费
3. wind (优先级 3)          - 付费
4. xtquant (优先级 9)       - 免费，高质量 ✨
5. akshare (优先级 10)      - 免费
6. yfinance (优先级 11)     - 免费
```

## 使用方法

### 历史数据获取

#### ETF数据获取

```python
from src.data.market_data_fetcher import MarketDataFetcher

# 初始化数据获取器
fetcher = MarketDataFetcher()

# 获取ETF历史数据
symbol = "159682"  # 科创50ETF
start_date = "20231001"
end_date = "20231201"

data = fetcher.fetch_etf_data(symbol, start_date, end_date)

if not data.empty:
    print(f"获取到 {len(data)} 条数据")
    print(data.head())
```

#### 基准指数数据获取

```python
# 获取沪深300基准数据
benchmark_data = fetcher.fetch_benchmark_data(
    benchmark_symbol="000300",
    start_date="20231001",
    end_date="20231201"
)
```

#### 批量数据获取

```python
# 批量获取多个ETF数据
symbols = ["159682", "510300", "512880"]
batch_data = fetcher.batch_fetch_etf_data(symbols, start_date, end_date)

for symbol, data in batch_data.items():
    print(f"{symbol}: {len(data)} 条数据")
```

### 实时数据订阅

#### 基本实时订阅

```python
import threading
import time

# 定义回调函数
def my_callback(data):
    if data:
        for symbol, symbol_data in data.items():
            print(f"收到 {symbol} 的实时数据")

# 订阅实时数据
symbols = ["159682", "510300"]
success = fetcher.subscribe_xtquant_realtime(
    symbols=symbols,
    callback=my_callback,
    period="1d"
)

if success:
    print("实时数据订阅成功")

    # 启动实时数据循环（在新线程中运行）
    def run_realtime():
        fetcher.start_xtquant_realtime_loop()

    realtime_thread = threading.Thread(target=run_realtime)
    realtime_thread.start()

    # 主线程继续其他工作
    # ...

    # 清理订阅
    fetcher.unsubscribe_xtquant_realtime()
```

#### 订阅状态管理

```python
# 获取订阅状态
status = fetcher.get_xtquant_subscription_status()
print(f"活跃订阅数: {status['total_subscriptions']}")

for sub in status['subscriptions']:
    print(f"订阅: {sub['symbol']}, 持续时间: {sub['duration_seconds']}秒")

# 取消特定订阅
fetcher.unsubscribe_xtquant_realtime(symbols=["159682"])

# 取消所有订阅
fetcher.unsubscribe_xtquant_realtime()
```

### 自定义回调函数

```python
def advanced_callback(data):
    """高级回调函数示例"""
    if not data:
        return

    current_time = datetime.now().strftime('%H:%M:%S')

    for xt_symbol in data.keys():
        try:
            # 获取最新完整数据
            latest_data = fetcher.xtdata.get_market_data_ex(
                [], [xt_symbol], period="1d", count=1
            )

            if latest_data and xt_symbol in latest_data:
                latest_df = latest_data[xt_symbol]
                if not latest_df.empty:
                    timestamp = list(latest_df.keys())[-1]
                    price_data = latest_df[timestamp]
                    current_price = float(price_data[3])

                    print(f"[{current_time}] {xt_symbol}: {current_price}")

                    # 自定义处理逻辑
                    # 例如：发送通知、保存到数据库、触发交易信号等

        except Exception as e:
            print(f"处理 {xt_symbol} 数据时出错: {e}")
```

## 数据格式

### 输出数据格式

XTQuant返回的数据已经标准化为统一格式：

```python
# 标准OHLCV格式
{
    'date': pd.Timestamp('2023-10-01'),
    'open': 1.2345,
    'high': 1.2456,
    'low': 1.2234,
    'close': 1.2401,
    'volume': 1234567,
    'returns': 0.0123,      # 收益率
    'ma5': 1.2389,          # 5日均线
    'ma10': 1.2356,         # 10日均线
    'ma20': 1.2321,         # 20日均线
    'ma60': 1.2289,         # 60日均线
    'volatility_20': 0.156, # 20日波动率
    'rsi_14': 58.3          # RSI指标
}
```

### 代码转换规则

| 原始代码 | XTQuant格式 | 说明 |
|---------|-------------|------|
| 159682 | 159682.SZ | 科创50ETF |
| 510300 | 510300.SZ | 沪深300ETF |
| 000300 | 000300.SH | 沪深300指数 |
| 000001 | 000001.SZ | 平安银行 |

## 测试和验证

### 运行集成测试

```bash
# 运行完整的XTQuant集成测试
python test_xtquant_integration.py
```

测试内容包括：
1. **可用性测试**: 检查XTQuant安装和导入
2. **配置测试**: 验证配置文件设置
3. **初始化测试**: 检查XTQuant初始化状态
4. **历史数据测试**: 测试ETF和基准数据获取
5. **实时数据测试**: 测试实时订阅功能
6. **数据质量测试**: 验证数据质量和完整性

### 数据质量检查

```python
# 生成数据质量报告
data = fetcher.fetch_etf_data("159682", "20231001", "20231201")
quality_report = fetcher.get_data_quality_report(data)

print(f"数据完整性: {quality_report['data_completeness']:.2f}%")
print(f"总记录数: {quality_report['total_records']}")
print(f"价格统计: {quality_report['price_stats']}")
```

## 性能优化

### 本地缓存策略

1. **首次访问**: 自动下载并缓存历史数据
2. **增量更新**: 只下载新增数据，节省带宽
3. **智能缓存**: 24小时TTL缓存机制
4. **存储优化**: 压缩存储，减少磁盘占用

### 网络优化

1. **连接复用**: 复用MiniQmt连接
2. **批量请求**: 支持批量数据获取
3. **超时控制**: 合理的超时设置
4. **重试机制**: 自动重试失败请求

## 故障排除

### 常见问题

#### 1. XTQuant初始化失败

**错误信息**: `XTQuant初始化失败`

**解决方案**:
- 检查MiniQmt是否正确安装和运行
- 确认网络连接正常
- 验证XTQuant版本兼容性

```python
# 检查XTQuant状态
from xtquant import xtdata
try:
    test_result = xtdata.get_market_data_ex([], ["000001.SZ"], period="1d", count=1)
    print("XTQuant连接正常")
except Exception as e:
    print(f"XTQuant连接失败: {e}")
```

#### 2. 数据获取为空

**错误信息**: `未获取到数据`

**解决方案**:
- 检查证券代码格式是否正确
- 确认查询时间范围内有交易数据
- 验证网络和防火墙设置

#### 3. 实时数据无推送

**可能原因**:
- 市场不在交易时间
- 订阅的标的代码错误
- MiniQmt实时数据服务异常

**解决方案**:
- 在交易时间内测试
- 验证订阅标的代码
- 检查MiniQmt服务状态

### 调试模式

启用详细日志进行调试：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 初始化数据获取器时会输出详细日志
fetcher = MarketDataFetcher()
```

## 最佳实践

### 1. 数据获取策略

```python
# 推荐：使用批量接口获取多个标的
symbols = ["159682", "510300", "512880"]
batch_data = fetcher.batch_fetch_etf_data(symbols, start_date, end_date)

# 避免：循环调用单个接口
for symbol in symbols:
    data = fetcher.fetch_etf_data(symbol, start_date, end_date)  # 效率低
```

### 2. 实时数据处理

```python
# 推荐：使用独立线程处理实时数据
def run_realtime_service():
    fetcher.subscribe_xtquant_realtime(symbols, callback=process_data)
    fetcher.start_xtquant_realtime_loop()

realtime_thread = threading.Thread(target=run_realtime_service, daemon=True)
realtime_thread.start()
```

### 3. 错误处理

```python
# 推荐：完善的错误处理
try:
    data = fetcher.fetch_etf_data(symbol, start_date, end_date)
    if data.empty:
        logger.warning(f"获取到空数据: {symbol}")
        # 尝试其他数据源或使用备用策略
    else:
        # 正常处理数据
        process_data(data)
except Exception as e:
    logger.error(f"数据获取异常: {symbol}, {e}")
    # 异常处理逻辑
```

### 4. 资源管理

```python
# 推荐：及时清理资源
try:
    # 使用XTQuant数据
    pass
finally:
    # 清理订阅
    if hasattr(fetcher, 'unsubscribe_xtquant_realtime'):
        fetcher.unsubscribe_xtquant_realtime()
```

## API参考

### MarketDataFetcher类

#### 主要方法

| 方法名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `fetch_etf_data()` | symbol, start_date, end_date | pd.DataFrame | 获取ETF历史数据 |
| `fetch_benchmark_data()` | symbol, start_date, end_date | pd.DataFrame | 获取基准指数数据 |
| `batch_fetch_etf_data()` | symbols, start_date, end_date | Dict[str, DataFrame] | 批量获取ETF数据 |
| `subscribe_xtquant_realtime()` | symbols, callback, period | bool | 订阅实时数据 |
| `unsubscribe_xtquant_realtime()` | symbols | None | 取消实时订阅 |
| `get_xtquant_subscription_status()` | None | Dict | 获取订阅状态 |
| `start_xtquant_realtime_loop()` | None | None | 启动实时循环 |
| `get_data_quality_report()` | data | Dict | 生成数据质量报告 |

#### 配置属性

| 属性名 | 类型 | 说明 |
|--------|------|------|
| `xtquant_initialized` | bool | XTQuant初始化状态 |
| `_xtquant_subscriptions` | Dict | 当前活跃订阅 |

## 更新日志

### v1.0.0 (2025-01-10)
- ✅ 完成XTQuant数据源集成
- ✅ 支持历史数据获取和自动下载
- ✅ 实现实时数据订阅功能
- ✅ 添加完整的数据质量验证
- ✅ 集成到现有数据源优先级系统
- ✅ 提供完整的测试套件

## 支持和反馈

### 文档资源
- [XTQuant官方文档](https://dict.thinktrader.net/nativeApi/start_now.html)
- [MiniQmt安装指南](https://www.xtquant.com/)

### 技术支持
- 📧 邮箱: [技术支持邮箱]
- 💬 官方QQ群: [群号]
- 📱 微信群: [群二维码]

### 问题反馈
如果遇到问题，请提供以下信息：
1. 错误信息和堆栈跟踪
2. 使用的XTQuant版本
3. MiniQmt版本和配置
4. 复现步骤和测试代码

---

**文档版本**: 1.0.0
**最后更新**: 2025年01月10日
**维护者**: 量化交易系统开发团队