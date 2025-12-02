"""
个人配置管理模块
Personal Configuration Manager

管理用户的API令牌、数据源配置等敏感信息
所有配置通过环境变量管理，确保敏感信息不被提交到Git
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """数据源配置"""
    name: str
    enabled: bool
    credentials: Dict[str, str]
    priority: int  # 优先级，数字越小优先级越高


class PersonalConfigManager:
    """
    个人配置管理器

    负责管理用户的个人配置，包括：
    - API令牌和凭证
    - 数据源启用状态
    - 其他用户偏好设置
    """

    def __init__(self, env_file: str = ".env"):
        """
        初始化个人配置管理器

        Args:
            env_file: 环境变量文件路径
        """
        self.env_file = env_file
        self.config = {}

        # 加载环境变量
        self._load_env_config()

        logger.info("个人配置管理器初始化完成")

    def _load_env_config(self) -> None:
        """从环境变量加载配置"""
        try:
            # 尝试加载.env文件
            if os.path.exists(self.env_file):
                load_dotenv(self.env_file)
                logger.info(f"已加载环境变量文件: {self.env_file}")
            else:
                logger.warning(f"环境变量文件不存在: {self.env_file}")

            # 加载数据源配置
            self.config['data_sources'] = self._load_data_sources_config()

            # 加载其他配置
            self.config['benchmark'] = os.getenv('DEFAULT_BENCHMARK', '000300')
            self.config['cache_enabled'] = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'

        except Exception as e:
            logger.error(f"加载个人配置失败: {e}")
            self.config = self._get_default_config()

    def _load_data_sources_config(self) -> Dict[str, DataSourceConfig]:
        """加载数据源配置"""
        data_sources = {}

        # Tushare Pro配置
        data_sources['tushare'] = DataSourceConfig(
            name='tushare',
            enabled=os.getenv('TUSHARE_ENABLED', 'false').lower() == 'true',
            credentials={
                'token': os.getenv('TUSHARE_TOKEN', '')
            },
            priority=1  # 最高优先级
        )

        # Wind配置
        data_sources['wind'] = DataSourceConfig(
            name='wind',
            enabled=os.getenv('WIND_ENABLED', 'false').lower() == 'true',
            credentials={
                'username': os.getenv('WIND_USERNAME', ''),
                'password': os.getenv('WIND_PASSWORD', '')
            },
            priority=2
        )

        # AKShare配置（免费数据源）
        data_sources['akshare'] = DataSourceConfig(
            name='akshare',
            enabled=os.getenv('AKSHARE_ENABLED', 'true').lower() == 'true',  # 默认启用
            credentials={},
            priority=10  # 较低优先级
        )

        # Yahoo Finance配置（免费数据源）
        data_sources['yfinance'] = DataSourceConfig(
            name='yfinance',
            enabled=os.getenv('YFINANCE_ENABLED', 'true').lower() == 'true',  # 默认启用
            credentials={},
            priority=11  # 最低优先级
        )

        return data_sources

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'data_sources': {
                'tushare': DataSourceConfig('tushare', False, {}, 1),
                'wind': DataSourceConfig('wind', False, {}, 2),
                'akshare': DataSourceConfig('akshare', True, {}, 10),
                'yfinance': DataSourceConfig('yfinance', True, {}, 11)
            },
            'benchmark': '000300',
            'cache_enabled': True
        }

    def get_api_credentials(self, source: str) -> Dict[str, str]:
        """
        安全获取API凭证

        Args:
            source: 数据源名称 (tushare, wind等)

        Returns:
            API凭证字典
        """
        try:
            if source not in self.config['data_sources']:
                logger.warning(f"未知的数据源: {source}")
                return {}

            data_source = self.config['data_sources'][source]

            if not data_source.enabled:
                logger.info(f"数据源 {source} 未启用")
                return {}

            # 验证凭证完整性
            credentials = data_source.credentials.copy()

            if source == 'tushare':
                if not credentials.get('token'):
                    logger.warning(f"Tushare token未配置")
                    return {}
            elif source == 'wind':
                if not credentials.get('username') or not credentials.get('password'):
                    logger.warning(f"Wind凭证不完整")
                    return {}

            logger.info(f"成功获取 {source} API凭证")
            return credentials

        except Exception as e:
            logger.error(f"获取 {source} API凭证失败: {e}")
            return {}

    def is_data_source_enabled(self, source: str) -> bool:
        """
        检查数据源是否启用

        Args:
            source: 数据源名称

        Returns:
            是否启用
        """
        try:
            if source not in self.config['data_sources']:
                return False

            return self.config['data_sources'][source].enabled

        except Exception as e:
            logger.error(f"检查数据源状态失败: {e}")
            return False

    def get_enabled_data_sources(self) -> list:
        """
        获取所有启用的数据源，按优先级排序

        Returns:
            启用的数据源列表（按优先级排序）
        """
        try:
            enabled_sources = [
                (source, config) for source, config in self.config['data_sources'].items()
                if config.enabled
            ]

            # 按优先级排序（数字越小优先级越高）
            enabled_sources.sort(key=lambda x: x[1].priority)

            return [source for source, _ in enabled_sources]

        except Exception as e:
            logger.error(f"获取启用数据源列表失败: {e}")
            return ['akshare', 'yfinance']  # 默认免费数据源

    def get_data_source_priority(self, source: str) -> int:
        """
        获取数据源优先级

        Args:
            source: 数据源名称

        Returns:
            优先级数值
        """
        try:
            if source not in self.config['data_sources']:
                return 999  # 未知数据源优先级最低

            return self.config['data_sources'][source].priority

        except Exception as e:
            logger.error(f"获取数据源优先级失败: {e}")
            return 999

    def get_default_benchmark(self) -> str:
        """
        获取默认基准指数

        Returns:
            基准指数代码
        """
        try:
            return self.config.get('benchmark', '000300')
        except Exception as e:
            logger.error(f"获取默认基准失败: {e}")
            return '000300'

    def is_cache_enabled(self) -> bool:
        """
        检查缓存是否启用

        Returns:
            缓存启用状态
        """
        try:
            return self.config.get('cache_enabled', True)
        except Exception as e:
            logger.error(f"获取缓存状态失败: {e}")
            return True

    def validate_config(self) -> Dict[str, Any]:
        """
        验证配置完整性和有效性

        Returns:
            验证结果字典
        """
        try:
            validation_result = {
                'valid': True,
                'warnings': [],
                'errors': [],
                'enabled_sources': [],
                'premium_sources': [],
                'free_sources': []
            }

            # 检查数据源配置
            for source, config in self.config['data_sources'].items():
                if config.enabled:
                    validation_result['enabled_sources'].append(source)

                    # 分类数据源
                    if source in ['tushare', 'wind']:
                        validation_result['premium_sources'].append(source)

                        # 检查付费数据源凭证
                        if source == 'tushare' and not config.credentials.get('token'):
                            validation_result['errors'].append(f"Tushare token未配置")
                            validation_result['valid'] = False
                        elif source == 'wind':
                            if not config.credentials.get('username') or not config.credentials.get('password'):
                                validation_result['errors'].append(f"Wind凭证不完整")
                                validation_result['valid'] = False
                    else:
                        validation_result['free_sources'].append(source)

            # 检查是否有可用的数据源
            if not validation_result['enabled_sources']:
                validation_result['errors'].append("没有启用的数据源")
                validation_result['valid'] = False

            # 检查基准配置
            benchmark = self.get_default_benchmark()
            if not benchmark:
                validation_result['warnings'].append("默认基准指数未配置")

            # 信息性提示
            if not validation_result['premium_sources']:
                validation_result['warnings'].append("未启用付费数据源，仅使用免费数据源")

            logger.info(f"配置验证完成: {'通过' if validation_result['valid'] else '失败'}")
            return validation_result

        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return {
                'valid': False,
                'warnings': [],
                'errors': [f"验证过程异常: {e}"],
                'enabled_sources': [],
                'premium_sources': [],
                'free_sources': []
            }

    def get_config_summary(self) -> str:
        """
        获取配置摘要信息

        Returns:
            配置摘要字符串
        """
        try:
            validation = self.validate_config()

            summary = f"""
个人配置摘要:
- 启用的数据源: {', '.join(validation['enabled_sources'])}
- 付费数据源: {', '.join(validation['premium_sources']) if validation['premium_sources'] else '无'}
- 免费数据源: {', '.join(validation['free_sources']) if validation['free_sources'] else '无'}
- 默认基准: {self.get_default_benchmark()}
- 缓存状态: {'启用' if self.is_cache_enabled() else '禁用'}
- 配置状态: {'有效' if validation['valid'] else '无效'}
"""

            if validation['warnings']:
                summary += f"- 警告: {'; '.join(validation['warnings'])}\n"

            if validation['errors']:
                summary += f"- 错误: {'; '.join(validation['errors'])}\n"

            return summary.strip()

        except Exception as e:
            logger.error(f"生成配置摘要失败: {e}")
            return "配置摘要生成失败"

    def reload_config(self) -> None:
        """重新加载配置"""
        logger.info("重新加载个人配置...")
        self._load_env_config()
        logger.info("个人配置重新加载完成")


# 全局配置管理器实例
_config_manager = None


def get_personal_config() -> PersonalConfigManager:
    """
    获取全局个人配置管理器实例

    Returns:
        PersonalConfigManager实例
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = PersonalConfigManager()
    return _config_manager


def reload_personal_config() -> None:
    """重新加载全局个人配置"""
    global _config_manager
    if _config_manager is not None:
        _config_manager.reload_config()
    else:
        _config_manager = PersonalConfigManager()


if __name__ == "__main__":
    # 测试配置管理器
    logging.basicConfig(level=logging.INFO)

    config_manager = PersonalConfigManager()

    print("=== 个人配置管理器测试 ===")
    print(config_manager.get_config_summary())

    # 测试数据源状态
    print(f"\n启用的数据源: {config_manager.get_enabled_data_sources()}")
    print(f"Tushare启用状态: {config_manager.is_data_source_enabled('tushare')}")
    print(f"默认基准: {config_manager.get_default_benchmark()}")

    # 验证配置
    validation = config_manager.validate_config()
    print(f"\n配置验证: {'通过' if validation['valid'] else '失败'}")
    if validation['warnings']:
        print(f"警告: {validation['warnings']}")
    if validation['errors']:
        print(f"错误: {validation['errors']}")