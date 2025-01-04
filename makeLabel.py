import pandas as pd
import numpy as np
from tqdm import tqdm


def preprocess_data(data, price_diff_threshold=0.01):
    """
    数据预处理：覆盖异常点。
    :param data: K线数据 (DataFrame)
    :param price_diff_threshold: 异常价差阈值 (0.01表示1%)
    :return: 预处理后的数据
    """
    prices = data['Close'].values
    for i in range(1, len(prices) - 1):
        prev_price = prices[i - 1]
        next_price = prices[i + 1]
        current_price = prices[i]
        if abs(current_price - prev_price) / prev_price > price_diff_threshold and abs(current_price - next_price) / next_price > price_diff_threshold:
            prices[i] = (prev_price + next_price) / 2  # 用前后两条数据的均值覆盖
    data['Close'] = prices
    return data


def calculate_labels(data, price_diff_threshold=0.002):
    """
    根据算法计算峰值和谷值的标签。
    :param data: K线数据 (DataFrame)
    :param price_diff_threshold: 峰值和谷值的价差阈值
    :return: 标签数组
    """
    close_prices = data['Close'].values
    labels = np.zeros(len(data), dtype=int)  # 初始化标签为0

    comparison_idx = 0  # 初始对比数据索引
    current_phase = "none"  # 当前阶段
    temporary_peak = None  # 临时峰值
    temporary_valley = None  # 临时谷值

    for i in tqdm(range(1, len(close_prices)), desc="计算标签"):
        current_price = close_prices[i]
        comparison_price = close_prices[comparison_idx]
        slope = (current_price - comparison_price) / comparison_price

        if slope > 0:  # 正斜率，可能是峰值
            # 谷值期对峰值记录
            if temporary_peak is None or slope > (close_prices[temporary_peak] - comparison_price) / comparison_price:
                temporary_peak = i

            # 谷值期状态反转为峰值期
            if close_prices[temporary_peak] - comparison_price > comparison_price * price_diff_threshold:
                current_phase = "peak"
                labels[comparison_idx + 1:temporary_peak + 1] = 1  # 标记为1
                comparison_idx = temporary_peak  # 更新对比数据为正式峰值
                temporary_valley = None  # 清空未超过阈值的临时谷值

            # 峰值期上限被刷新
            if current_phase == "peak":
                labels[comparison_idx + 1:temporary_peak + 1] = 1  # 标记为1
                comparison_idx = temporary_peak  # 更新对比数据为正式峰值
                temporary_valley = None  # 清空未超过阈值的临时谷值


        elif slope < 0:  # 负斜率，可能是谷值
            # 峰值期对谷值记录
            if temporary_valley is None or slope < (close_prices[temporary_valley] - comparison_price) / comparison_price:
                temporary_valley = i
            
            # 峰值期状态反转为谷值期
            if comparison_price - close_prices[temporary_valley] > comparison_price * price_diff_threshold:
                current_phase = "valley"
                labels[comparison_idx + 1:temporary_valley + 1] = 0  # 标记为-1
                comparison_idx = temporary_valley  # 更新对比数据为正式谷值
                temporary_peak = None  # 清空未超过阈值的临时峰值

            # 谷值期下限被刷新
            if current_phase == "valley":
                labels[comparison_idx + 1:temporary_valley + 1] = 0  # 标记为0
                comparison_idx = temporary_valley  # 更新对比数据为正式谷值
                temporary_peak = None  # 清空未超过阈值的临时峰值
            
            

    return labels


def process_data(data, price_diff_threshold=0.002):
    """
    处理完整的K线数据并计算标签。
    :param data: K线数据 (DataFrame)
    :param price_diff_threshold: 峰值和谷值的价差阈值
    :return: 带有标签的DataFrame
    """
    print(f"[INFO] 数据加载成功，共有 {len(data)} 条记录。")

    # 数据预处理
    data = preprocess_data(data, price_diff_threshold=0.01)
    print("[INFO] 数据预处理完成。")

    # 计算标签
    labels = calculate_labels(data, price_diff_threshold=price_diff_threshold)
    print("[INFO] 标签计算完成。")

    # 将标签加入到数据中
    data['Label'] = labels
    return data


# 主程序
if __name__ == "__main__":
    # 读取数据
    data = pd.read_csv('ETH_USDT_1m_binance_data.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # 处理数据
    labeled_data = process_data(data, price_diff_threshold=0.002)

    # 保存结果到CSV
    labeled_data.to_csv('labeled_kline_data.csv', index=False)
    print("[INFO] 数据已保存到 'labeled_kline_data.csv'")
