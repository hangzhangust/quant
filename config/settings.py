"""
网格交易策略回测系统配置文件
Grid Trading Strategy Backtest System Configuration
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List

# 项目配置
PROJECT_NAME = "网格交易策略回测系统"
VERSION = "1.0.0"
AUTHOR = "Claude Assistant"

# 数据配置
DATA_DIR = "data"
CACHE_DIR = "cache"
RESULTS_DIR = "results"

# 时间配置
DEFAULT_START_DATE = (datetime.now() - timedelta(days=3*365)).strftime('%Y%m%d')
DEFAULT_END_DATE = datetime.now().strftime('%Y%m%d')

# 交易配置
DEFAULT_COMMISSION_RATE = 0.0003  # 万分之三手续费
DEFAULT_SLIPPAGE_RATE = 0.001    # 千分之一滑点
MIN_TRADE_UNIT = 100             # 最小交易单位

# 网格策略默认参数
DEFAULT_GRID_CONFIG = {
    "base_strategy": "basic_grid",  # basic_grid, dynamic_grid, martingale_grid
    "grid_spacing": 0.03,           # 3%网格间距
    "grid_count": 10,               # 网格数量
    "position_size_type": "fixed",  # fixed, percentage
    "position_size": 10000,         # 固定股数或金额
    "rebalance_threshold": 0.02,    # 2%再平衡阈值
    "max_position_ratio": 0.8,      # 最大仓位比例
}

# 策略比较配置
STRATEGY_COMPARISON_CONFIG = {
    "enabled_strategies": ["basic_grid", "dynamic_grid", "martingale_grid"],
    "strategy_display_names": {
        "basic_grid": "基础网格策略",
        "dynamic_grid": "动态网格策略",
        "martingale_grid": "马丁格尔网格策略"
    },
    "comparison_metrics": ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"],
    "ranking_method": "weighted_score",
    "metric_weights": {
        "total_return": 0.30,
        "sharpe_ratio": 0.35,
        "max_drawdown": 0.20,
        "win_rate": 0.15
    },
    "default_chart_types": ["radar", "performance_comparison", "risk_return_scatter"],
    "enable_parameter_optimization": False,
    "default_optimization_method": "grid_search"
}

# 多语言配置
MULTILINGUAL_CONFIG = {
    "enabled": True,
    "default_language": "zh",
    "fallback_language": "en",
    "display_mode": "bilingual",  # monolingual, bilingual, toggle
    "bilingual_labels": {
        "total_return": "总收益率",
        "annual_return": "年化收益率",
        "max_drawdown": "最大回撤",
        "sharpe_ratio": "夏普比率",
        "win_rate": "胜率",
        "etf_count": "ETF数量",
        "return_distribution": "收益率分布",
        "risk_return_distribution": "风险收益分布",
        "cumulative_return": "累积收益",
        "volatility": "波动率",
        "strategy_comparison": "策略对比",
        "basic_grid": "基础网格",
        "dynamic_grid": "动态网格",
        "martingale_grid": "马丁格尔网格"
    }
}

# 性能分析配置
PERFORMANCE_METRICS = [
    "total_return",
    "annual_return",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "win_rate",
    "profit_loss_ratio",
    "trade_count",
    "avg_holding_period",
    "var_95",
    "cvar_95"
]

# 策略参数优化配置
STRATEGY_OPTIMIZATION_CONFIG = {
    "basic_grid": {
        "param_space": {
            "grid_count": [6, 8, 10, 12, 16, 20],
            "position_size": [500, 1000, 1500, 2000, 2500],
            "buy_percentage": [0.5, 0.8, 1.0, 1.5, 2.0],
            "sell_percentage": [0.3, 0.5, 0.8, 1.0, 1.5]
        },
        "objective_weights": {
            "total_return": 0.25,
            "sharpe_ratio": 0.35,
            "max_drawdown": 0.20,
            "win_rate": 0.15,
            "trading_frequency": 0.05
        }
    },
    "dynamic_grid": {
        "param_space": {
            "grid_count": [6, 8, 10, 12, 16],
            "position_size": [500, 1000, 1500, 2000],
            "buy_percentage": [0.5, 0.8, 1.0, 1.5],
            "sell_percentage": [0.3, 0.5, 0.8, 1.0],
            "volatility_window": [10, 15, 20, 25, 30],
            "volatility_threshold": [0.01, 0.015, 0.02, 0.025, 0.03]
        },
        "objective_weights": {
            "total_return": 0.20,
            "sharpe_ratio": 0.40,
            "max_drawdown": 0.20,
            "win_rate": 0.15,
            "trading_frequency": 0.05
        }
    },
    "martingale_grid": {
        "param_space": {
            "grid_count": [6, 8, 10, 12],
            "position_size": [500, 800, 1000, 1200],
            "buy_percentage": [0.5, 0.8, 1.0, 1.2, 1.5],
            "sell_percentage": [0.3, 0.5, 0.8, 1.0],
            "martingale_factor": [1.5, 2.0, 2.5],
            "max_martingale_levels": [3, 4, 5, 6]
        },
        "objective_weights": {
            "total_return": 0.30,
            "sharpe_ratio": 0.25,
            "max_drawdown": 0.30,
            "win_rate": 0.10,
            "trading_frequency": 0.05
        },
        "risk_constraints": {
            "max_martingale_factor": 3.0,
            "max_martingale_levels": 8,
            "min_risk_free_ratio": 0.15
        }
    },
    "default_settings": {
        "optimization_method": "grid_search",  # grid_search, random_search, bayesian
        "max_iterations": 50,
        "max_workers": 4,
        "enable_parallel": True,
        "cache_results": True,
        "validation_split": 0.2
    }
}

# 优化配置
OPTIMIZATION_CONFIG = {
    "method": "local_search",  # local_search, bayesian, genetic
    "n_trials": 100,
    "param_ranges": {
        "grid_spacing": (0.01, 0.10),  # 1%-10%
        "grid_count": (5, 20),
        "position_size": (5000, 50000)
    },
    "objective": "sharpe_ratio",
    "constraints": {
        "max_drawdown": 0.20,
        "min_trade_count": 20
    }
}

# 可视化配置
CHART_CONFIG = {
    "figure_size": (12, 8),
    "style": "seaborn-v0_8",
    "color_palette": "Set2",
    "interactive": True,
    "font_config": {
        "chinese_fonts": ["Microsoft YaHei", "SimHei", "PingFang SC"],
        "english_fonts": ["Arial", "Calibri", "Verdana"],
        "fallback_fonts": ["DejaVu Sans", "Liberation Sans"],
        "default_font_size": 10,
        "title_font_size": 12,
        "label_font_size": 10,
        "tick_font_size": 9,
        "legend_font_size": 9
    },
    "bilingual_support": {
        "enabled": True,
        "display_format": "chinese (english)",
        "force_english_fallback": True
    }
}

# Web应用配置
WEB_CONFIG = {
    "title": "网格交易策略回测系统",
    "icon": "📈",
    "layout": "wide",
    "theme": "light",
    "cache_ttl": 3600  # 1小时缓存
}

# 风险控制配置
RISK_CONFIG = {
    "max_single_loss": 0.05,     # 单笔最大亏损5%
    "max_drawdown_limit": 0.25,  # 最大回撤限制25%
    "position_concentration": 0.3, # 单个ETF最大仓位30%
    "correlation_threshold": 0.7   # 相关性阈值
}

# ETF代码映射
ETF_CODE_MAP = {
    "科创50": "159682",
    "A500ETF": "159380",
    "沪深ETF": "159985",
    "平安9999": "159937",
    "恒生ETF": "513730",
    "沪深300ETF": "513130",
    "芯片ETF": "159599",
    "50ETF": "530050",
    "恒生互联网ETF": "159691",
    "恒生科技ETF": "513030",
    "消费ETF": "513650",
    "科创100": "588000",
    "双创50": "159781",
    "科创板50": "588080",
    "创业50": "159915",
    "银行ETF": "512800"
}

# 日志配置
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/grid_backtest.log",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# 缓存配置
CACHE_CONFIG = {
    "enable": True,
    "ttl": 86400,  # 24小时
    "max_size": 1000  # 最大缓存条目数
}

# 数据源配置
DATA_SOURCE_CONFIG = {
    "premium_sources": ["tushare", "wind"],
    "free_sources": ["akshare", "yfinance"],
    "fallback_strategy": "auto",
    "cache_premium_data": True,
    "retry_attempts": 3,
    "timeout_seconds": 30,
    "rate_limiting": {
        "tushare": {"requests_per_minute": 200},
        "akshare": {"requests_per_minute": 60},
        "yfinance": {"requests_per_minute": 120}
    }
}

# Beta计算配置
BENCHMARK_CONFIG = {
    "default_benchmark": "000300",
    "auto_fetch_benchmark": True,
    "cache_benchmark_data": True,
    "benchmark_data_period": 3,  # years
    "alternative_benchmarks": {
        "000001": "上证综指",
        "000300": "沪深300指数",
        "000905": "中证500指数",
        "399001": "深证成指",
        "399006": "创业板指"
    },
    "market_benchmarks": {
        "SH": "000001",  # 上海市场基准
        "SZ": "399001",  # 深圳市场基准
        "HS": "000300"   # 沪深市场基准（默认）
    }
}

# 个人配置系统
PERSONAL_CONFIG = {
    "env_file": ".env",
    "auto_reload": True,
    "validation_on_startup": True,
    "fallback_to_free_sources": True
}