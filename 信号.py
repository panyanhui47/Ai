import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
from binance.client import Client
import time, uuid, logging

# ====== Binance 配置 ======
API_KEY = 'your_api_key'
API_SECRET = 'your_api_secret'
client = Client(API_KEY, API_SECRET)

# ====== 配置选项 ======
MODEL_TYPE = "XGBoost"  # 支持 "XGBoost" 或 "RandomForest"
initial_cash = 100000.0  # 初始资金
cash = initial_cash
holdings = 0.0  # 持仓数量
trade_log = []  # 交易日志
current_buy_id = None  # 当前持仓的买入单号
profit_threshold = -1000  # 盈利阈值，触发再训练
trade_fee_rate = 0.001  # 手续费率 0.1%
loop_interval = 60  # 每轮间隔时间（秒）
RSI_B = 40
RSI_S = 60
stop_loss = 0.02  # 止损比例
take_profit = 0.05  # 止盈比例
cooldown_period = 5  # 冷却时间（分钟）
last_trade_time = 0

# ====== 初始化日志模块 ======
logging.basicConfig(
    filename='trade_log.log',  # 保存日志的文件
    filemode='a',  # 追加模式
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s [%(levelname)s] %(message)s',  # 日志格式
    datefmt='%Y-%m-%d %H:%M:%S'  # 时间格式
)
logger = logging.getLogger()

# ====== 获取历史数据 ======
def get_binance_data(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1MINUTE, limit=100):
    logger.info("Fetching data from Binance...")
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    data = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    data[numeric_cols] = data[numeric_cols].astype(float)
    return data[['open_time', 'open', 'high', 'low', 'close', 'volume']]

# ====== 技术指标计算 ======
def calculate_technical_indicators(data):
    logger.info("Calculating technical indicators...")
    # 确保数据按时间排序
    data = data.sort_values(by='open_time')

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
    # 添加 VWAP，确保索引为 datetime
    data.set_index('open_time', inplace=True)
    data['VWAP'] = ta.vwap(data['high'], data['low'], data['close'], data['volume'])
    data.reset_index(inplace=True)
    # data['VWAP'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
    data['CCI'] = ta.cci(data['high'], data['low'], data['close'], length=20)
    return data

# ====== 特征工程 ======
def create_features(data):
    logger.info("Creating features for model...")
    data['Price_Change'] = data['close'].pct_change()
    data['Future_Price'] = data['close'].shift(-1)
    data.dropna(inplace=True)
    data['Target'] = (data['Future_Price'] > data['close']).astype(int)

    # 添加技术指标
    # data['VWAP'] = ta.vwap(data['high'], data['low'], data['close'], data['volume'])
    # data['CCI'] = ta.cci(data['high'], data['low'], data['close'], length=20)
    # 确保训练和预测使用一致的特征
    features = ['EMA_20', 'RSI_14', 'MACD', 'MACD_signal', 'ATR',
                'Bollinger_Upper', 'Bollinger_Lower', 'Momentum', 'VWAP', 'CCI', 'Price_Change']
    return data, features

# ====== 模型训练 ======
def train_model(data, features):
    logger.info("Training model...")
    X = data[features]
    y = data['Target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    if MODEL_TYPE == "XGBoost":
        model = xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=5, 
            learning_rate=0.1, 
            random_state=42, 
            eval_metric='logloss')
    elif MODEL_TYPE == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42)
    else:
        raise ValueError(f"不支持的模型类型: {MODEL_TYPE}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model ({MODEL_TYPE}) 准确率: {accuracy:.2f}")

    joblib.dump(model, f'{MODEL_TYPE.lower()}_price_prediction_model.pkl')
    joblib.dump(scaler, f'{MODEL_TYPE.lower()}_scaler.pkl')
    return model, scaler

# ====== 信号生成 ======
def generate_signals(data, model, scaler, features):
    logger.info("Generating signals...")
    
    # 确保只使用训练时的特征
    X = data[features]
    X_scaled = scaler.transform(X)
    data['Prediction'] = model.predict(X_scaled)
    data['Signal'] = "Hold"
    data.loc[data['Prediction'] == 1, 'Signal'] = "Buy"
    data.loc[data['Prediction'] == 0, 'Signal'] = "Sell"
    return data
# ====== 增加信号过滤条件 ======
def filter_signals(data):
    data['Strong_Signal'] = "Hold"
    data.loc[(data['Signal'] == "Buy") & (data['RSI_14'] < RSI_B), 'Strong_Signal'] = "Buy"
    data.loc[(data['Signal'] == "Sell") & (data['RSI_14'] > RSI_S), 'Strong_Signal'] = "Sell"
    return data
# ====== 模拟交易 ======
def simulate_trading(data):
    global cash, holdings, trade_log, current_buy_id, last_trade_time
    entry_price = 0  # 记录买入价格

    for i in range(1, len(data)):
        signal = data['Strong_Signal'].iloc[i]
        price = data['close'].iloc[i]
        current_time = time.time()

        if current_time - last_trade_time < cooldown_period * 60:
            continue

        if signal == "Buy" and cash > 0:
            trade_cost = cash * trade_fee_rate
            holdings = (cash - trade_cost) / price
            entry_price = price
            cash = 0
            current_buy_id = str(uuid.uuid4())  # 生成唯一买入单 UUID
            trade_log.append({
                "单号": current_buy_id,
                "交易类型": "买入",
                "价格": price,
                "手续费": trade_cost,
                "持仓数量": holdings
            })

        elif signal == "Sell" and holdings > 0:
            sell_value = holdings * price
            trade_cost = sell_value * trade_fee_rate
            cash = sell_value - trade_cost
            profit = (price - entry_price) * holdings - trade_cost
            trade_log.append({
                "单号": current_buy_id,
                "交易类型": "卖出",
                "价格": price,
                "手续费": trade_cost,
                "卖出金额": sell_value,
                "本单获利": profit
            })
            holdings = 0
            current_buy_id = None
            last_trade_time = current_time

    final_value = cash + (holdings * data['close'].iloc[-1])
    profit = final_value - initial_cash
    print(f"[INFO] 当前总资产: {final_value:.2f}, 当前收益: {profit:.2f}")
    return profit

# ====== 动态再训练 ======
def retrain_model(data, features):
    print("[INFO] 触发模型再训练...")
    return train_model(data, features)



# ====== 显示交易日志 ======
def display_trade_log(trade_log):
    if not trade_log:
        logger.info("无交易记录.")
        return

    for log in trade_log:
        log_entry = " | ".join(f"{key}: {value}" for key, value in log.items())
        logger.info(log_entry)


# ====== 主循环 ======
if __name__ == "__main__":
    try:
        # 加载或训练模型
        model_filename = f'{MODEL_TYPE.lower()}_price_prediction_model.pkl'
        scaler_filename = f'{MODEL_TYPE.lower()}_scaler.pkl'
        try:
            model = joblib.load(model_filename)
            scaler = joblib.load(scaler_filename)
            logger.info(f"Loaded existing model: {model_filename}")
        except FileNotFoundError:
            logger.warning("Model not found. Training a new model...")
            data = get_binance_data()
            data = calculate_technical_indicators(data)
            data, features = create_features(data)
            model, scaler = train_model(data, features)

        # 主循环
        while True:
            logger.info("Fetching latest data...")
            data = get_binance_data()
            data = calculate_technical_indicators(data)
            data, features = create_features(data)

            # 生成信号
            data = generate_signals(data, model, scaler, features)
            # data = filter_signals(data)
            
            # 模拟交易并记录日志
            profit = simulate_trading(data)
            # 打印交易日志
            display_trade_log(trade_log)

            # 动态再训练
            if profit < profit_threshold:
                model, scaler = retrain_model(data, features)

            # 等待下一轮
            logger.info(f"Waiting for {loop_interval} seconds...")
            time.sleep(loop_interval)

    except KeyboardInterrupt:
        logger.info("Program terminated manually.")
