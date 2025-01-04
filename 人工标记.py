import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

# 加载大规模的K线数据
def load_large_csv(csv_path, chunk_size=100):
    """
    分批加载CSV文件，逐块加载并返回DataFrame。
    :param csv_path: CSV文件路径
    :param chunk_size: 每次加载的数据量
    :return: 返回一个迭代器，逐块返回数据
    """
    return pd.read_csv(csv_path, chunksize=chunk_size)

# 初始化用户标记数据
labels = []

# 每个窗口的数据处理
def process_window(window_data):
    """
    处理每个窗口的数据，绘制图表并允许用户标记。
    :param window_data: 当前窗口的数据
    :return: 返回当前窗口的标记
    """
    # 创建图表并绘制收盘价
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(window_data['Date'], window_data['Close'], label='Close Price', color='blue')

    # 标记逻辑：通过点击选择峰值（1）和谷值（-1）
    def on_click(event):
        # 获取点击的点的索引
        clicked_idx = np.abs(window_data['Date'] - event.xdata).argmin()
        
        # 获取点击的价格
        clicked_price = window_data['Close'].iloc[clicked_idx]
        
        # 标记逻辑：通过点击选择峰值（1）和谷值（-1）
        label = input(f"Click at index {clicked_idx}, price {clicked_price}. Enter '1' for Peak, '-1' for Valley, '0' for none: ")
        
        if label in ['1', '-1']:
            labels.append({
                'Index': clicked_idx,
                'Date': window_data['Date'].iloc[clicked_idx],
                'Price': clicked_price,
                'Label': int(label)
            })
            
            # 打印反馈
            print(f"Marked: {window_data['Date'].iloc[clicked_idx]} as {'Peak' if label == '1' else 'Valley'}")

        # 更新图表的标记
        ax.scatter(window_data['Date'].iloc[clicked_idx], clicked_price, color='red' if label == '1' else 'green', marker='o')

    # 连接点击事件
    fig.canvas.mpl_connect('button_press_event', on_click)

    # 绘制图表
    ax.set_title('Interactive K-Line Chart with Peaks and Valleys')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.legend()
    plt.show()

    return labels

# 保存标记到CSV文件
def save_labels_to_csv(labels, filename='interactive_labels.csv'):
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(filename, index=False)
    print(f"Labels have been saved to '{filename}'.")

# 主程序：逐窗口加载数据并标记
def main(csv_path):
    chunk_size = 100  # 每次加载100条数据
    data_chunks = load_large_csv(csv_path, chunk_size)
    
    for i, chunk in enumerate(data_chunks):
        print(f"Processing chunk {i + 1}...")
        
        # 将Date列转换为datetime格式
        chunk['Date'] = pd.to_datetime(chunk['Date'])

        # 处理当前窗口数据并标记
        process_window(chunk)

        # 每处理完一个窗口就保存标记
        save_labels_to_csv(labels)

if __name__ == '__main__':
    # 设置CSV文件路径
    csv_path = 'kline_data.csv'  # 替换为你的CSV文件路径
    main(csv_path)
