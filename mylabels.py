import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib.widgets import Cursor

# 读取本地 CSV 文件
df = pd.read_csv('预测/binance_data.csv', parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)

# 如果 CSV 数据没有列名，请确保它包含必要的 K 线列：'open', 'high', 'low', 'close', 'volume'
# 假设 CSV 文件中有这些列：'timestamp', 'open', 'high', 'low', 'close', 'volume'

# 用于保存人工标记的字典
marks = {}

# 函数：在图上标记点击的 K 线
def on_click(event):
    global marks
    # 获取点击的位置（即 K 线的索引）
    if event.inaxes:
        x, y = event.xdata, event.ydata
        # 获取最近的 K 线
        idx = int(round(x))  # 将 X 坐标（时间轴位置）转换为最近的行索引
        if idx < len(df):
            # 标记输入框
            label = input(f"为时间点 {df.index[idx]} 输入标签（1 或 2）：")
            if label in ['1', '2']:
                marks[idx] = int(label)
                print(f"为 {df.index[idx]} 标记为 {label}")
                # 更新标记到 DataFrame
                df.loc[df.index[idx], 'label'] = label
            else:
                print("标签必须是 1 或 2。")
        
        # 显示更新后的数据
        print(marks)

# 绘制 K 线图
fig, axes = mpf.plot(df, type='candle', style='charles', returnfig=True)
ax = axes[0]

# 创建交互式光标（便于定位）
cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

# 连接点击事件
fig.canvas.mpl_connect('button_press_event', on_click)

# 显示图形
plt.show()

# 最终保存带有标记的 K 线数据到 CSV 文件
df.to_csv('kline_with_labels.csv', index=True)
