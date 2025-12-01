"""
字体配置模块
提供中英文字体支持和matplotlib配置
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import warnings
import os
from typing import Tuple, Optional


class FontConfig:
    """字体配置类，负责处理中英文字体显示"""

    def __init__(self):
        self.system = platform.system()
        self.chinese_font = None
        self.english_font = None
        self._setup_fonts()

    def _setup_fonts(self):
        """设置系统中英文字体"""
        try:
            self._setup_chinese_font()
            self._setup_english_font()
            self._configure_matplotlib()
        except Exception as e:
            warnings.warn(f"字体配置失败: {e}")
            self._use_fallback_fonts()

    def _setup_chinese_font(self):
        """设置中文字体 - 简化版本"""
        # Windows系统优先字体顺序 - 优先使用SimHei
        if self.system == "Windows":
            chinese_fonts = ["SimHei", "Microsoft YaHei", "SimSun", "KaiTi"]
        elif self.system == "Darwin":  # macOS
            chinese_fonts = ["PingFang SC", "Hiragino Sans GB", "SimHei", "STXihei"]
        else:  # Linux
            chinese_fonts = ["WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]

        # 查找第一个可用的中文字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        self.chinese_font = None
        for font_name in chinese_fonts:
            if font_name in available_fonts:
                self.chinese_font = font_name
                break

        # 如果没找到，使用手动路径
        if not self.chinese_font and self.system == "Windows":
            self.chinese_font = self._find_chinese_font_manually()

    def _setup_english_font(self):
        """设置英文字体"""
        english_fonts = [
            "Arial",
            "Calibri",
            "Verdana",
            "Helvetica",
            "Times New Roman",
            "DejaVu Sans"
        ]

        available_fonts = [f.name for f in fm.fontManager.ttflist]

        for font_name in english_fonts:
            if font_name in available_fonts:
                self.english_font = font_name
                break

    def _find_chinese_font_manually(self) -> Optional[str]:
        """手动查找中文字体文件"""
        font_paths = []

        if self.system == "Windows":
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",  # Microsoft YaHei
                "C:/Windows/Fonts/simhei.ttf", # SimHei
                "C:/Windows/Fonts/simsun.ttc", # SimSun
            ]
        elif self.system == "Darwin":
            font_paths = [
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/Hiragino Sans GB.ttc",
                "/System/Library/Fonts/STHeiti Light.ttc",
            ]

        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font_prop = fm.FontProperties(fname=font_path)
                    return font_prop.get_name()
                except:
                    continue

        return None

    def _configure_matplotlib(self):
        """配置matplotlib字体设置"""
        # 强制设置中文字体
        if self.chinese_font:
            plt.rcParams['font.sans-serif'] = [self.chinese_font, 'SimHei', 'Microsoft YaHei', 'SimSun']
            plt.rcParams['axes.unicode_minus'] = False
        else:
            # 回退方案
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False

        # 设置字体大小
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9

    def _use_fallback_fonts(self):
        """使用回退字体"""
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def get_chinese_font_properties(self) -> Optional[fm.FontProperties]:
        """获取中文字体属性"""
        if self.chinese_font:
            return fm.FontProperties(family=self.chinese_font)
        return None

    def get_bilingual_label(self, chinese: str, english: str) -> str:
        """生成双语标签"""
        if self.chinese_font:
            return f"{chinese} ({english})"
        else:
            # 如果没有中文字体，只返回英文
            return english

    def test_font_display(self) -> Tuple[bool, str]:
        """测试字体显示效果"""
        test_text = self.get_bilingual_label("测试", "Test")

        try:
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, test_text, ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

            # 保存到临时文件测试
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                plt.savefig(tmp.name, dpi=50, bbox_inches='tight')
                plt.close(fig)

                # 检查文件是否存在且有内容
                if os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
                    os.unlink(tmp.name)
                    return True, "字体配置成功"
                else:
                    return False, "字体测试失败"

        except Exception as e:
            return False, f"字体测试异常: {e}"


# 全局字体配置实例
_font_config = None

def get_font_config() -> FontConfig:
    """获取全局字体配置实例"""
    global _font_config
    if _font_config is None:
        _font_config = FontConfig()
    return _font_config

def setup_chinese_fonts():
    """设置中文字体的便捷函数"""
    config = get_font_config()
    success, message = config.test_font_display()

    if not success:
        warnings.warn(f"中文字体设置可能有问题: {message}")

    return success

def get_bilingual_labels():
    """获取常用的双语标签字典"""
    config = get_font_config()

    return {
        'total_return': config.get_bilingual_label("总收益率", "Total Return"),
        'max_drawdown': config.get_bilingual_label("最大回撤", "Max Drawdown"),
        'sharpe_ratio': config.get_bilingual_label("夏普比率", "Sharpe Ratio"),
        'win_rate': config.get_bilingual_label("胜率", "Win Rate"),
        'etf_count': config.get_bilingual_label("ETF数量", "ETF Count"),
        'return_distribution': config.get_bilingual_label("收益率分布", "Return Distribution"),
        'risk_return_distribution': config.get_bilingual_label("风险收益分布", "Risk-Return Distribution"),
        'cumulative_return': config.get_bilingual_label("累积收益", "Cumulative Return"),
        'volatility': config.get_bilingual_label("波动率", "Volatility"),
        'strategy_comparison': config.get_bilingual_label("策略对比", "Strategy Comparison"),
        'basic_grid': config.get_bilingual_label("基础网格", "Basic Grid"),
        'dynamic_grid': config.get_bilingual_label("动态网格", "Dynamic Grid"),
        'martingale_grid': config.get_bilingual_label("马丁格尔网格", "Martingale Grid")
    }

# 初始化字体配置
if __name__ == "__main__":
    # 测试字体配置
    setup_chinese_fonts()
    config = get_font_config()
    success, message = config.test_font_display()
    print(f"字体配置结果: {message}")

    # 显示双语标签示例
    labels = get_bilingual_labels()
    for key, label in labels.items():
        print(f"{key}: {label}")