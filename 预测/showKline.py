import dash
from dash import dcc, html
import pandas as pd
import plotly.graph_objs as go

# 创建 Dash 应用
app = dash.Dash(__name__)

# 读取CSV文件
labeled_data = pd.read_csv('labeled_kline_data.csv')
labeled_data['timestamp'] = pd.to_datetime(labeled_data['timestamp'])

# 定义默认窗口大小和初始起点
default_window_size = 100
initial_start = 0

# 绘制图表函数
def create_figure(data):
    fig = go.Figure()

    # 添加收盘价的折线图
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue', width=2)))

    # 标记接近峰值的点（标签为 1）
    peaks = data[data['Label'] == 1]
    fig.add_trace(go.Scatter(x=peaks['timestamp'], y=peaks['Close'], mode='markers', name='Peak (1)', marker=dict(color='red', symbol='triangle-up', size=10)))

    # 标记接近谷值的点（标签为 -1）
    valleys = data[data['Label'] == 0]
    fig.add_trace(go.Scatter(x=valleys['timestamp'], y=valleys['Close'], mode='markers', name='Valley (0)', marker=dict(color='green', symbol='triangle-down', size=10)))

    # 设置图表布局
    fig.update_layout(
        title='K-line Data with Peaks and Valleys',
        xaxis_title='Timestamp',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,  # 禁用Plotly自带滑动条
        showlegend=True
    )
    return fig

# 应用布局
app.layout = html.Div([
    html.H1('K-line Data with Peaks and Valleys'),

    # 图表组件
    dcc.Graph(id='kline-graph'),

    # 滑块组件
    dcc.Slider(
        id='start-slider',
        min=0,
        max=len(labeled_data) - 1,
        step=1,
        value=initial_start,
        marks={i: str(labeled_data['timestamp'].iloc[i]) for i in range(0, len(labeled_data), max(1, len(labeled_data) // 10))}  # 每10%标记一次
    ),

    # 下拉菜单，用于选择窗口大小
    html.Label("Select Window Size:"),
    dcc.Dropdown(
        id='window-size-dropdown',
        options=[
            {'label': '50', 'value': 50},
            {'label': '100', 'value': 100},
            {'label': '200', 'value': 200},
            {'label': '500', 'value': 500},
            {'label': '1000', 'value': 1000}
        ],
        value=default_window_size,  # 默认窗口大小
        style={'width': '200px'}
    )
])

# 回调函数：根据起点和窗口大小更新图表数据
@app.callback(
    dash.dependencies.Output('kline-graph', 'figure'),
    [dash.dependencies.Input('start-slider', 'value'),
     dash.dependencies.Input('window-size-dropdown', 'value')]
)
def update_graph(start_idx, window_size):
    # 计算数据的结束位置
    end_idx = min(len(labeled_data) - 1, start_idx + window_size)

    # 获取新的数据段
    data_segment = labeled_data.iloc[start_idx:end_idx]
    return create_figure(data_segment)

# 回调函数：当调整窗口大小时，动态更新滑块范围（起点保持不变）
@app.callback(
    dash.dependencies.Output('start-slider', 'max'),
    [dash.dependencies.Input('window-size-dropdown', 'value')]
)
def update_slider_max(window_size):
    # 滑块的最大值需要减去窗口大小，以防止起点超过范围
    return len(labeled_data) - window_size

# 运行应用
if __name__ == '__main__':
    app.run_server(debug=True)
