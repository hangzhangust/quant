#!/usr/bin/env python3
"""
测试网格参数解析模式
"""

import re

def test_grid_patterns():
    # 测试数据
    test_strings = [
        "以1.408元为基准价，每涨5.00%卖出，每跌-10.00%买入",
        "以1.408ԪΪ��׼�ۣ�ÿ����5.00%������ÿ�µ�-10.00%����",  # 实际文件格式
        "以1.237ԪΪ��׼�ۣ�ÿ����5.00%������ÿ�µ�-5.00%����",
        "以2.032ԪΪ��׼�ۣ�ÿ����5.00%������ÿ�µ�-10.00%����",
    ]

    patterns = [
        r'以(\d+\.?\d*)元为基准价，每涨([\d.-]+)%卖出，每跌([\d.-]+)%买入',
        r'以(\d+\.?\d*)ԪΪ��׼�ۣ�ÿ����([\d.-]+)%������ÿ�µ�([\d.-]+)%����',
        r'(\d+\.?\d*)Ԫ.*����([\d.]+)%.*ÿ�µ�([\d.-]+)%',
        r'(\d+\.?\d*).*涨([\d.]+)%.*跌([\d.]+)%'
    ]

    for test_str in test_strings:
        print(f"\n测试字符串: {test_str}")

        for i, pattern_str in enumerate(patterns):
            try:
                pattern = re.compile(pattern_str)
                match = pattern.search(test_str)

                if match:
                    print(f"  模式 {i+1} 匹配成功:")
                    print(f"    基准价: {match.group(1)}")
                    print(f"    卖出: {match.group(2)}%")
                    print(f"    买入: {match.group(3)}%")
                    break
                else:
                    print(f"  模式 {i+1} 未匹配")
            except Exception as e:
                print(f"  模式 {i+1} 错误: {e}")

if __name__ == "__main__":
    test_grid_patterns()