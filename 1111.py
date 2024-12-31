import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import backtrader as bt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from binance.client import Client
# from binance.ws.threaded_stream import BinanceSocketManager
import time

# ====== Binance 配置 ======
API_KEY = 'your_api_key'
API_SECRET = 'your_api_secret'
client = Client(API_KEY, API_SECRET)

# ====== 全局变量 ======
live_data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# ====== 获取历史数据 ======
def get_binance_data(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1MINUTE, limit=100):
    """
    从 Binance 获取历史数据。
    """
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    data = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
    data['close_time'] = pd.to_datetime(data['close_time'], unit='ms')
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    data[numeric_cols] = data[numeric_cols].astype(float)
    return data[['open_time', 'open', 'high', 'low', 'close', 'volume']]

# ====== 技术指标计算 ======
def calculate_technical_indicators(data):
    """
    根据行情数据计算多种技术指标。
    """
    data['EMA_20'] = ta.ema(data['close'], length=20)
    data['RSI_14'] = ta.rsi(data['close'], length=14)
    macd = ta.macd(data['close'], fast=12, slow=26, signal=9)
    data['MACD'] = macd['MACD_12_26_9']
    data['MACD_signal'] = macd['MACDs_12_26_9']
    data['ATR'] = ta.atr(data['high'], data['low'], data['close'], length=14)
    bb = ta.bbands(data['close'], length=20, std=2)
    data['Bollinger_Upper'] = bb['BBU_20_2.0']
    data['Bollinger_Lower'] = bb['BBL_20_2.0']
    data['Momentum'] = ta.mom(data['close'], length=10)
    return data

# ====== 特征工程 ======
def create_features(data):
    """
    创建特征，包含价格变化和技术指标特征。
    """
    data['Price_Change'] = data['close'].pct_change()
    data['Future_Price'] = data['close'].shift(-1)
    data.dropna(inplace=True)
    data['Target'] = (data['Future_Price'] > data['close']).astype(int)
    features = ['EMA_20', 'RSI_14', 'MACD', 'MACD_signal', 'ATR', 
                'Bollinger_Upper', 'Bollinger_Lower', 'Momentum', 'Price_Change']
    return data, features

# ====== 模型训练 ======
def train_model(data, features):
    """
    训练随机森林模型用于价格方向预测。
    """
    X = data[features]
    y = data['Target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.2f}")
    
    joblib.dump(model, 'price_prediction_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    return model, scaler

# ====== 信号生成 ======
def generate_signals(data, model, scaler, features):
    """
    使用训练好的模型生成买卖信号。
    """
    X = data[features]
    X_scaled = scaler.transform(X)
    data['Prediction'] = model.predict(X_scaled)
    data['Signal'] = data['Prediction'].apply(lambda x: "Buy" if x == 1 else "Sell")
    return data

# ====== 信号平滑 ======
def smooth_signals(data, window=3):
    """
    对交易信号进行平滑处理。
    """
    data['Signal_Num'] = data['Signal'].map({'Buy': 1, 'Sell': -1})
    data['Smoothed_Signal_Num'] = data['Signal_Num'].rolling(window=window, min_periods=1).mean()
    data['Smoothed_Signal'] = data['Smoothed_Signal_Num'].apply(lambda x: "Buy" if x > 0 else "Sell")
    return data

# ====== 回测系统 ======
class SignalStrategy(bt.Strategy):
    def __init__(self):
        self.signal = self.datas[0].close

    def next(self):
        if self.signal[0] == "Buy" and not self.position:
            self.buy(size=1)
        elif self.signal[0] == "Sell" and self.position:
            self.sell(size=1)

def run_backtest(data):
    # 确保时间范围和列正确
    if 'timestamp' not in data.columns:
        raise ValueError("数据缺少 timestamp 列！")

    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce').dt.floor('s')  # 修复 FutureWarning
    data = data.tail(1000)  # 限制数据行数

    # 检查必要列
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"缺少必要列: {col}")

    # 创建 backtrader 数据源
    data_feed = bt.feeds.PandasData(dataname=data, datetime='timestamp', open='open', 
                                    high='high', low='low', close='close', volume='volume')

    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)
    cerebro.addstrategy(SignalStrategy)
    cerebro.broker.setcash(100000.0)
    cerebro.run()

    # 格式化 matplotlib 绘图
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
    cerebro.plot(style='candlestick', numfigs=1)




# ====== 实时交易 ======
def process_message(msg):
    global live_data
    if msg['e'] == 'kline':
        kline = msg['k']
        row = {
            'timestamp': pd.to_datetime(kline['t'], unit='ms'),
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
        }
        live_data = pd.concat([live_data, pd.DataFrame([row])]).tail(200)
        live_data.set_index('timestamp', inplace=True)

def execute_trade(signal, symbol):
    """
    根据信号执行交易。
    """
    try:
        if signal == "Buy":
            client.create_order(symbol=symbol.upper(), side='BUY', type='MARKET', quantity=0.001)
        elif signal == "Sell":
            client.create_order(symbol=symbol.upper(), side='SELL', type='MARKET', quantity=0.001)
    except Exception as e:
        print(f"交易失败: {e}")

# def run_realtime_trading(symbol):
#     """
#     实时监听市场数据并生成信号。
#     """
#     bm = BinanceSocketManager(client)
#     conn_key = bm.start_kline_socket(symbol, process_message, interval=Client.KLINE_INTERVAL_1MINUTE)
#     bm.start()

#     while True:
#         if not live_data.empty:
#             data = calculate_technical_indicators(live_data)
#             data = generate_signals(data, model, scaler, features)
#             data = smooth_signals(data)
#             latest_signal = data['Smoothed_Signal'].iloc[-1]
#             execute_trade(latest_signal, symbol)
#         time.sleep(60)

# ====== 主程序 ======
if __name__ == "__main__":
    # 从 Binance 获取历史数据
    data = get_binance_data()
    data = calculate_technical_indicators(data)
    data, features = create_features(data)

    # 加载或训练模型
    try:
        model = joblib.load('price_prediction_model.pkl')
        scaler = joblib.load('scaler.pkl')
    except FileNotFoundError:
        print("模型文件不存在，开始训练模型...")
        model, scaler = train_model(data, features)

    # 生成信号并平滑
    data = generate_signals(data, model, scaler, features)
    data = smooth_signals(data)

    # 检查和补全必要列
    if 'timestamp' not in data.columns:
        data['timestamp'] = pd.date_range(start='2024-12-01', periods=len(data), freq='1min')
    else:
        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce').dt.floor('s')
    if 'open' not in data.columns:
        data['open'] = data['close'] * 0.99
    if 'high' not in data.columns:
        data['high'] = data['close'] * 1.01
    if 'low' not in data.columns:
        data['low'] = data['close'] * 0.98
    if 'volume' not in data.columns:
        data['volume'] = 100

    # 验证数据结构
    print("检查数据结构：")
    print(data.head())
    print(data.dtypes)

    # 运行回测
    run_backtest(data)
    # 启用实时交易（如果需要）
    # run_realtime_trading('btcusdt')
