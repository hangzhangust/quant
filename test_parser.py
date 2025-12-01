#!/usr/bin/env python3
"""
测试网格配置解析器
"""

import sys
import pandas as pd
from pathlib import Path

# 直接导入解析器，绕过__init__.py的依赖问题
sys.path.append('src')
from data.grid_config_parser import GridConfigParser

def main():
    """测试主函数"""
    print("开始测试网格配置解析器...")

    # 检查Table.xls文件是否存在
    table_file = Path("Table.xls")
    if not table_file.exists():
        print(f"错误: 找不到Table.xls文件，当前目录: {Path.cwd()}")
        print("请确保Table.xls文件在当前目录下")
        return

    # 创建解析器实例
    parser = GridConfigParser()

    try:
        # 解析Excel文件
        print(f"正在解析文件: {table_file}")
        df = parser.parse_excel_file(str(table_file))

        if df.empty:
            print("警告: 未找到有效的网格配置")
            return

        print(f"解析成功！")
        print(f"总配置数量: {len(df)}")
        print(f"有效配置数量: {len(df[df['is_valid'] == True])}")

        # 显示统计信息
        stats = parser.get_summary_statistics(df)
        print("\n配置统计信息:")
        for key, value in stats.items():
            print(f"   {key}: {value}")

        # 显示前5条有效配置
        valid_configs = df[df['is_valid'] == True]
        if not valid_configs.empty:
            print("\n前5条有效配置:")
            display_columns = ['stock_name', 'stock_code', 'base_price', 'sell_percentage', 'buy_percentage']
            for col in display_columns:
                if col in valid_configs.columns:
                    print(f"   {col}: {valid_configs[col].head().tolist()}")

        # 导出到CSV
        output_file = "parsed_grid_configs.csv"
        parser.export_parsed_config(df, output_file)
        print(f"\n配置已导出到: {output_file}")

        # 分析网格参数分布
        print("\n网格参数分析:")
        if 'sell_percentage' in df.columns:
            print(f"   卖出百分比范围: {df['sell_percentage'].min():.2f}% - {df['sell_percentage'].max():.2f}%")
            print(f"   平均卖出百分比: {df['sell_percentage'].mean():.2f}%")

        if 'buy_percentage' in df.columns:
            print(f"   买入百分比范围: {df['buy_percentage'].min():.2f}% - {df['buy_percentage'].max():.2f}%")
            print(f"   平均买入百分比: {df['buy_percentage'].mean():.2f}%")

        # 价格区间分析
        if 'base_price' in df.columns:
            print(f"   基准价格范围: {df['base_price'].min():.3f} - {df['base_price'].max():.3f}")
            print(f"   平均基准价格: {df['base_price'].mean():.3f}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()