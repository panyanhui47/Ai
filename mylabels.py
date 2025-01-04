import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
import matplotlib.dates as mdates
from datetime import datetime

# 1. 读取 CSV 数据
def load_data(csv_file):
    # 读取 CSV 数据
    df = pd.read_csv(csv_file)
    
    # 确保 timestamp 是 datetime 类型
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# 2. 绘制 K 线图
def plot_candlestick(df, start_idx, end_idx):
    fig, ax = plt.subplots(figsize=(10, 6))
    df_window = df.iloc[start_idx:end_idx]

    # 只绘制收盘价
    ax.plot(df_window['timestamp'], df_window['Close'], label='Close', color='blue', lw=1)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f'K-Line from {df_window["timestamp"].iloc[0]} to {df_window["timestamp"].iloc[-1]}')

    # 测试交互功能
    mplcursors.cursor(hover=True)
    return fig, ax, df_window

# 3. 在图上添加标记并更新 label 列
def add_marker_and_update_label(fig, ax, df_window, df, label_column='label'):
    cursor = mplcursors.cursor(ax, hover=True, highlight=True)  # 使用highlight=True确保高亮显示
    
    # 在图形上点击位置时，添加标记
    cursor.connect('add', lambda sel: on_click(sel, df_window, df, label_column))

# 4. 响应点击事件，标记并更新 label
def on_click(sel, df_window, df, label_column='label'):
    try:
        # 获取点击位置的索引并确保是整数
        idx = int(sel.index)
        
        # 确保通过索引获取的数据有效
        timestamp = df_window['timestamp'].iloc[idx]
        open_price = df_window['Open'].iloc[idx]
        close_price = df_window['Close'].iloc[idx]
        
        # 输出选择的数据
        print(f"标记: {timestamp} | 开盘: {open_price} | 收盘: {close_price}")

        # 让用户输入标签
        label = input(f"请输入标签 {timestamp}: ")

        # 更新数据框中的标签列
        timestamp = pd.to_datetime(timestamp)
        df.loc[df['timestamp'] == timestamp, label_column] = label

    except Exception as e:
        print(f"点击事件发生错误: {e}")

# 5. 保存更新后的 CSV 文件
def save_to_csv(df, output_file):
    df.to_csv(output_file, index=False)
    print(f"Updated data saved to {output_file}")

# 6. 主函数
def main(csv_file, start_idx, end_idx, output_file):
    df = load_data(csv_file)
    fig, ax, df_window = plot_candlestick(df, start_idx, end_idx)
    add_marker_and_update_label(fig, ax, df_window, df)
    
    # 展示图形
    plt.show()
    
    # 保存更新后的 CSV 文件
    save_to_csv(df, output_file)
if __name__ == "__main__":
    # 示例：假设 CSV 文件路径为 'data.csv'，标记窗口为 0-100，输出文件为 'data_with_labels.csv'
    input_file = '预测/binance_data.csv'
    output_file = 'data_with_labels.csv'
    main(input_file, 0, 100, output_file)