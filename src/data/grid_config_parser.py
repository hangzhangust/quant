"""
网格配置解析器
Grid Configuration Parser

用于解析Table.xls中的网格交易条件单配置
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

from config.settings import ETF_CODE_MAP

logger = logging.getLogger(__name__)


class GridConfigParser:
    """网格配置解析器类"""

    def __init__(self):
        self.etf_code_map = ETF_CODE_MAP
        self.pattern_cache = {}

    def _detect_file_type(self, file_source):
        """
        检测输入文件类型

        Args:
            file_source: 文件路径字符串或Streamlit文件对象

        Returns:
            str: 'file_path' 或 'streamlit_file'
        """
        if hasattr(file_source, 'read'):
            # Streamlit上传的文件对象
            return 'streamlit_file'
        elif isinstance(file_source, str):
            # 文件路径字符串
            return 'file_path'
        else:
            raise ValueError(f"不支持的文件源类型: {type(file_source)}")

    def _read_streamlit_file(self, streamlit_file):
        """
        读取Streamlit上传的文件对象

        Args:
            streamlit_file: Streamlit文件上传对象

        Returns:
            pd.DataFrame: 解析后的数据
        """
        # 获取文件名
        filename = streamlit_file.name

        # 根据文件扩展名判断文件类型
        if filename.endswith(('.xls', '.xlsx')):
            # 尝试作为Excel文件读取
            try:
                df = pd.read_excel(streamlit_file, engine='openpyxl')
                logger.info(f"成功读取Streamlit Excel文件: {filename}, 共{len(df)}条记录")
                return df
            except Exception as e:
                # 如果Excel解析失败，尝试作为文本文件读取
                logger.warning(f"Excel解析失败，尝试文本解析: {e}")

        # 尝试作为文本文件读取（处理实际上是文本的.xls文件）
        encodings = ['gb2312', 'gbk', 'utf-8', 'latin-1']
        streamlit_file.seek(0)  # 重置文件指针

        for encoding in encodings:
            try:
                streamlit_file.seek(0)  # 重置文件指针
                content = streamlit_file.read()
                # 尝试解码内容
                if isinstance(content, bytes):
                    content = content.decode(encoding)

                # 使用StringIO处理文本内容
                from io import StringIO
                df = pd.read_csv(StringIO(content), sep='\t')
                logger.info(f"成功使用{encoding}编码读取Streamlit文本文件: {filename}, 共{len(df)}条记录")
                return df
            except (UnicodeDecodeError, Exception) as e:
                logger.debug(f"编码{encoding}读取失败: {e}")
                continue

        raise ValueError(f"无法用任何方式读取Streamlit文件: {filename}")

    def parse_excel_file(self, file_source) -> pd.DataFrame:
        """
        解析Table.xls文件，支持文件路径和Streamlit文件对象

        Args:
            file_source: Excel文件路径字符串或Streamlit文件上传对象

        Returns:
            解析后的DataFrame，包含标准化的网格配置
        """
        try:
            # 检测文件源类型
            file_type = self._detect_file_type(file_source)

            if file_type == 'streamlit_file':
                # 处理Streamlit上传的文件对象
                df = self._read_streamlit_file(file_source)
                logger.info(f"成功读取Streamlit文件，共{len(df)}条记录")
            else:
                # 处理文件路径（保持原有逻辑）
                if file_source.endswith(('.xls', '.xlsx')):
                    # Excel文件格式，但先尝试Excel，失败后尝试文本格式
                    excel_success = False
                    try:
                        df = pd.read_excel(file_source, engine='openpyxl')
                        excel_success = True
                        logger.info(f"成功读取Excel文件: {file_source}, 共{len(df)}条记录")
                    except Exception as e:
                        # 如果openpyxl失败，尝试xlrd
                        try:
                            df = pd.read_excel(file_source, engine='xlrd')
                            excel_success = True
                            logger.info(f"成功读取Excel文件(xlrd): {file_source}, 共{len(df)}条记录")
                        except Exception as e2:
                            logger.warning(f"Excel解析失败，尝试文本格式: {file_source}")

                    # 如果Excel解析都失败，尝试作为文本文件处理（针对伪.xls文件）
                    if not excel_success:
                        encodings = ['gb2312', 'gbk', 'utf-8', 'latin-1']
                        df = None
                        used_encoding = None

                        for encoding in encodings:
                            try:
                                df = pd.read_csv(file_source, sep='\t', encoding=encoding)
                                used_encoding = encoding
                                logger.info(f"成功使用{encoding}编码读取文本文件: {file_source}, 共{len(df)}条记录")
                                break
                            except UnicodeDecodeError:
                                continue
                            except Exception as e:
                                logger.debug(f"编码{encoding}读取失败: {e}")
                                continue

                        if df is None:
                            raise ValueError(f"无法用任何方式读取文件: {file_source}")
                else:
                    # CSV文件格式，尝试多种编码
                    encodings = ['gb2312', 'gbk', 'utf-8', 'latin-1']
                    df = None
                    used_encoding = None

                    for encoding in encodings:
                        try:
                            df = pd.read_csv(file_source, sep='\t', encoding=encoding)
                            used_encoding = encoding
                            logger.info(f"成功使用{encoding}编码读取文件: {file_source}, 共{len(df)}条记录")
                            break
                        except UnicodeDecodeError:
                            continue

                    if df is None:
                        raise ValueError(f"无法用任何编码读取文件: {file_source}")

            # 数据清洗和标准化
            df_clean = self._clean_data(df)

            # 解析网格配置
            df_parsed = self._parse_grid_configurations(df_clean)

            # 验证数据完整性
            df_validated = self._validate_configurations(df_parsed)

            logger.info(f"解析完成，有效配置共{len(df_validated)}条")
            return df_validated

        except Exception as e:
            logger.error(f"解析Excel文件失败: {e}")
            raise

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        # 移除空行
        df = df.dropna(how='all')

        # 根据列名判断数据格式
        columns = df.columns.tolist()

        # 检查是否是简化格式（包含ETF名称、ETF代码等列）
        if any('ETF' in str(col) for col in columns):
            # 简化格式：直接映射列名
            column_mapping = {}

            # 查找对应的列
            for col in columns:
                col_lower = str(col).lower()
                if 'etf' in col_lower and '名称' in col_lower:
                    column_mapping[col] = 'stock_name'
                elif 'etf' in col_lower and '代码' in col_lower:
                    column_mapping[col] = 'stock_code'
                elif '基准' in col_lower or 'base' in col_lower:
                    column_mapping[col] = 'base_price'
                elif '卖出' in col_lower and '网格' in col_lower:
                    column_mapping[col] = 'sell_percentage'
                elif '买入' in col_lower and '网格' in col_lower:
                    column_mapping[col] = 'buy_percentage'
                elif '委托' in col_lower and '数量' in col_lower:
                    column_mapping[col] = 'position_size'
                elif '状态' in col_lower:
                    column_mapping[col] = 'status'
                elif '网格设置' in col_lower:
                    column_mapping[col] = 'grid_settings'

            # 重命名列
            df = df.rename(columns=column_mapping)

        else:
            # 原始复杂格式：按位置映射列名
            # 实际列名位置: 0序号, 1空白, 2状态, 3条件类型, 4股票代码, 5股票名称, 6网格设置, 7委托方式, 8委托时间, 9有效期, 10交易方式, 11账户类型
            if len(df.columns) >= 11:
                columns_to_extract = {
                    df.columns[2]: 'status',        # 状态
                    df.columns[4]: 'stock_code_filter',  # 股票代码
                    df.columns[5]: 'stock_name',    # 股票名称
                    df.columns[6]: 'grid_settings', # 网格设置
                    df.columns[7]: 'order_method',  # 委托方式
                    df.columns[8]: 'order_time',    # 委托时间
                    df.columns[9]: 'validity_period', # 有效期
                    df.columns[10]: 'trade_method', # 交易方式
                }

                if len(df.columns) >= 12:
                    columns_to_extract[df.columns[11]] = 'account_type'  # 账户类型

                # 重命名列
                df = df.rename(columns=columns_to_extract)

        # 返回清洗后的DataFrame
        return df

    def _parse_grid_configurations(self, df: pd.DataFrame) -> pd.DataFrame:
        """解析网格配置"""
        parsed_configs = []

        for idx, row in df.iterrows():
            try:
                config = self._parse_single_configuration(row)
                if config:
                    parsed_configs.append(config)
            except Exception as e:
                logger.warning(f"解析第{idx+1}行配置失败: {e}")
                continue

        return pd.DataFrame(parsed_configs)

    def _parse_single_configuration(self, row: pd.Series) -> Optional[Dict]:
        """解析单条网格配置"""

        # 检查数据格式并提取相应字段
        if 'stock_code' in row and 'stock_name' in row:
            # 简化格式
            stock_code = row.get('stock_code', '')
            stock_name = row.get('stock_name', '')
            base_price = float(row.get('base_price', 1.0))
            sell_percentage = float(row.get('sell_percentage', 2.0))
            buy_percentage = float(row.get('buy_percentage', 4.0))
            position_size = int(row.get('position_size', 1000))

            return {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'base_price': base_price,
                'sell_percentage': sell_percentage,
                'buy_percentage': buy_percentage,
                'buy_position_size': position_size,
                'sell_position_size': position_size,
                'status': row.get('status', '有效'),
                'grid_count': 8
            }

        else:
            # 原始复杂格式
            # 提取股票代码
            stock_code = self._extract_stock_code(row.get('stock_code_filter', ''))
            stock_name = row.get('stock_name', '')

            if not stock_code and stock_name in self.etf_code_map:
                stock_code = self.etf_code_map[stock_name]

            if not stock_code:
                logger.warning(f"无法提取股票代码: {stock_name}")
                return None

            # 解析网格设置
            grid_settings = row.get('grid_settings', '')
            grid_params = self._parse_grid_settings(grid_settings)

            if not grid_params:
                logger.warning(f"无法解析网格设置: {grid_settings}")
                return None

            # 解析委托方式
            order_method = row.get('order_method', '')
            order_params = self._parse_order_method(order_method)

            # 构建配置字典
            return {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'status': row.get('status', '激活'),
                'base_price': grid_params['base_price'],
                'sell_percentage': grid_params['sell_percentage'],
                'buy_percentage': grid_params['buy_percentage'],
                'buy_position_size': order_params.get('buy_position_size', 0),
                'sell_position_size': order_params.get('sell_position_size', 0),
                'position_size_type': order_params.get('position_size_type', 'shares'),
                'order_time': row.get('order_time', ''),
                'validity_period': row.get('validity_period', ''),
                'trade_method': row.get('trade_method', '系统自动交易'),
                'account_type': row.get('account_type', '普通'),
                'raw_grid_settings': grid_settings,
                'raw_order_method': order_method
            }

    def _extract_stock_code(self, stock_code_filter: str) -> str:
        """从条件单股票代码字段提取股票代码"""
        if not stock_code_filter:
            return ""

        # 匹配格式如: '>= "159682"' 或直接是股票代码
        patterns = [
            r'["\']?(\d{6})["\']?',  # 提取6位数字
            r'(\d{6})',              # 直接提取6位数字
        ]

        for pattern in patterns:
            match = re.search(pattern, stock_code_filter)
            if match:
                return match.group(1)

        return ""

    def _parse_grid_settings(self, grid_settings: str) -> Optional[Dict]:
        """解析网格设置字符串"""
        if not grid_settings:
            return None

        # 直接使用字符编码匹配，避免编码问题
        # 根据实际字符: 以1.408元为基准价，每涨5.00%卖出，每跌-10.00%买入
        # 实际格式: 1.408元为基准价，每涨5.00%卖出，每跌-10.00%买入

        try:
            # 查找数字模式
            # 使用更简单的模式：找到数字和百分比
            import re

            # 匹配基准价格 (数字开头到元)
            base_price_match = re.search(r'(\d+\.?\d*)', grid_settings)
            if not base_price_match:
                return None
            base_price = float(base_price_match.group(1))

            # 查找所有百分比数值
            percentage_matches = re.findall(r'([-\d.]+)%', grid_settings)
            if len(percentage_matches) < 2:
                return None

            # 第一个百分比是卖出，第二个是买入
            sell_percentage = abs(float(percentage_matches[0]))
            buy_percentage = abs(float(percentage_matches[1]))

            logger.debug(f"解析成功: 基准价={base_price}, 卖出={sell_percentage}%, 买入={buy_percentage}%")

            return {
                'base_price': base_price,
                'sell_percentage': sell_percentage,
                'buy_percentage': buy_percentage
            }

        except (ValueError, AttributeError, IndexError) as e:
            logger.error(f"解析网格参数失败: {e}, 字符串: {grid_settings}")
            return None

    def _parse_order_method(self, order_method: str) -> Dict:
        """解析委托方式"""
        if not order_method:
            return {}

        params = {
            'buy_position_size': 0,
            'sell_position_size': 0,
            'position_size_type': 'shares'
        }

        # 提取买入数量 - 改进的正则模式
        buy_patterns = [
            r'买入\s*(\d+)\s*份',
            r'买入\s*(\d+)\s*股',
            r'买入\s*([\d.]+)\s*元',
            r'买入(\d+)股',
            r'买入([\d.]+)元',
            r'买入(\d+)份'
        ]

        for pattern in buy_patterns:
            match = re.search(pattern, order_method)
            if match:
                params['buy_position_size'] = self._convert_to_number(match.group(1))
                if '元' in pattern:
                    params['position_size_type'] = 'amount'
                break

        # 提取卖出数量 - 改进的正则模式
        sell_patterns = [
            r'卖出\s*(\d+)\s*份',
            r'卖出\s*(\d+)\s*股',
            r'卖出\s*([\d.]+)\s*元',
            r'卖出(\d+)股',
            r'卖出([\d.]+)元',
            r'卖出(\d+)份'
        ]

        for pattern in sell_patterns:
            match = re.search(pattern, order_method)
            if match:
                params['sell_position_size'] = self._convert_to_number(match.group(1))
                if '元' in pattern:
                    params['position_size_type'] = 'amount'
                break

        return params

    def _convert_to_number(self, value_str: str) -> float:
        """将字符串转换为数字"""
        try:
            # 处理千分位分隔符
            value_str = value_str.replace(',', '')
            return float(value_str)
        except ValueError:
            return 0.0

    def _validate_configurations(self, df: pd.DataFrame) -> pd.DataFrame:
        """验证配置的完整性和有效性"""
        valid_configs = []

        for idx, row in df.iterrows():
            is_valid = True
            validation_errors = []

            # 验证必要字段
            if not row.get('stock_code'):
                is_valid = False
                validation_errors.append("股票代码缺失")

            if row.get('base_price', 0) <= 0:
                is_valid = False
                validation_errors.append("基准价格无效")

            if row.get('sell_percentage', 0) <= 0:
                is_valid = False
                validation_errors.append("卖出百分比无效")

            if row.get('buy_percentage', 0) <= 0:
                is_valid = False
                validation_errors.append("买入百分比无效")

            # 添加验证结果
            row = row.copy()
            row['is_valid'] = is_valid
            row['validation_errors'] = ';'.join(validation_errors) if validation_errors else ''

            if is_valid:
                valid_configs.append(row)
            else:
                logger.warning(f"配置验证失败 {row.get('stock_name', 'Unknown')}: {validation_errors}")

        return pd.DataFrame(valid_configs)

    def get_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """获取配置统计信息"""
        if df.empty:
            return {}

        stats = {
            'total_configs': len(df),
            'valid_configs': len(df[df['is_valid'] == True]),
            'unique_stocks': df['stock_code'].nunique(),
            'avg_base_price': df['base_price'].mean(),
            'avg_sell_percentage': df['sell_percentage'].mean(),
            'avg_buy_percentage': df['buy_percentage'].mean(),
            'price_range': {
                'min': df['base_price'].min(),
                'max': df['base_price'].max()
            },
            'sell_percentage_range': {
                'min': df['sell_percentage'].min(),
                'max': df['sell_percentage'].max()
            },
            'buy_percentage_range': {
                'min': df['buy_percentage'].min(),
                'max': df['buy_percentage'].max()
            }
        }

        return stats

    def export_parsed_config(self, df: pd.DataFrame, output_path: str):
        """导出解析后的配置"""
        try:
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"配置已导出到: {output_path}")
        except Exception as e:
            logger.error(f"导出配置失败: {e}")
            raise


def main():
    """测试函数"""
    parser = GridConfigParser()

    # 测试文件路径
    test_file = "Table.xls"

    if Path(test_file).exists():
        try:
            df = parser.parse_excel_file(test_file)
            print(f"解析成功，共{len(df)}条有效配置")

            # 显示统计信息
            stats = parser.get_summary_statistics(df)
            print("配置统计信息:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

            # 显示前几条配置
            print("\n前5条配置:")
            print(df.head())

            # 导出解析结果
            parser.export_parsed_config(df, "parsed_grid_configs.csv")

        except Exception as e:
            print(f"测试失败: {e}")
    else:
        print(f"测试文件不存在: {test_file}")


if __name__ == "__main__":
    main()