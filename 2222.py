import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
from binance.client import Client
import time, uuid, logging

# ====== Binance Configuration ======
API_KEY = 'your_api_key'
API_SECRET = 'your_api_secret'
client = Client(API_KEY, API_SECRET)

# ====== Options ======
symbol = "XRPUSDT"
MODEL_TYPE = "XGBoost"
initial_cash = 100000.0
cash = initial_cash
holdings = 0.0
trade_log = []  # 记录交易日志
buy_queue = []  # 记录未结算买单
profit_threshold = -1000
trade_fee_rate = 0.001  # 手续费为 0.1%
loop_interval = 60  # 脚本循环间隔
kline_interval = Client.KLINE_INTERVAL_1MINUTE  # K 线监控间隔

# 风控策略
max_buy_percentage = 0.01  # 每次买入不得超过总资金的 1%
profit_rate = 0.001 # 保证盈利率
max_trades_per_cycle = 5  # 每次循环最多买入 1 单

# ====== Logging Configuration ======
logging.basicConfig(
    filename='trade_log.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

# ====== Helper Functions ======
def calculate_rsi(series, period):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

# ====== Fetch Data ======
def get_binance_data(symbol=symbol, interval=kline_interval, limit=100):
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

# ====== Calculate Technical Indicators ======
def calculate_technical_indicators(data, weights):
    data['EMA_20'] = data['close'].ewm(span=20, adjust=False).mean()
    data['RSI_14'] = calculate_rsi(data['close'], 14)
    data['ATR'] = calculate_atr(data['high'], data['low'], data['close'], 14)
    macd_line = data['close'].ewm(span=12, adjust=False).mean() - data['close'].ewm(span=26, adjust=False).mean()
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    data['MACD'] = macd_line - signal_line
    data['Weighted_MACD'] = data['MACD'] * weights.get('MACD', 1)
    data['Weighted_RSI'] = data['RSI_14'] * weights.get('RSI', 1)
    return data

# ====== Feature Engineering ======
def create_features(data, trade_log):
    data['Price_Change'] = data['close'].pct_change()
    data['Future_Price'] = data['close'].shift(-1)
    data.dropna(inplace=True)
    data['Target'] = (data['Future_Price'] > data['close']).astype(int)
    features = ['EMA_20', 'RSI_14', 'ATR', 'Price_Change', 'Weighted_MACD', 'Weighted_RSI']
    return data, features

# ====== Train Model ======
def train_model(data, features, trade_log=None):
    """
    Train a machine learning model with optional trade log feedback and return the model, scaler, and accuracy.

    Args:
        data (pd.DataFrame): Dataset containing features and target.
        features (list): List of feature names for training.
        trade_log (list, optional): List of trade logs for model feedback.

    Returns:
        model: Trained machine learning model.
        scaler: StandardScaler object for feature scaling.
        accuracy (float): Model accuracy on the test dataset.
    """
    logger.info("Starting model training...")
    start_time = time.time()

    # Prepare data
    X = data[features]
    y = data['Target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Incorporate feedback from trade_log
    if trade_log:
        logger.info("Incorporating trade log feedback into training...")
        profit_feedback = np.zeros(len(y))
        for trade in trade_log:
            if trade["Type"] == "Sell":
                sell_time = trade["Time"]
                profit = trade.get("Profit", 0)
                # Match trade time with data and adjust weights
                idx = data.index[data['open_time'] == sell_time].tolist()
                if idx:
                    profit_feedback[idx[0]] = profit
        y = np.clip(y + profit_feedback, 0, 1)  # Adjust y with feedback

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Select and train the model
    if MODEL_TYPE == "XGBoost":
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, eval_metric='logloss')
    elif MODEL_TYPE == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model Accuracy: {accuracy:.2f}")

    # Save model and scaler
    model_filename = f'{MODEL_TYPE.lower()}_price_prediction_model.pkl'
    scaler_filename = f'{MODEL_TYPE.lower()}_scaler.pkl'
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    logger.info(f"Model saved to {model_filename}, Scaler saved to {scaler_filename}")

    # Log training time
    end_time = time.time()
    training_duration = end_time - start_time
    logger.info(f"Model training completed in {training_duration:.2f} seconds")

    return model, scaler, accuracy

# ====== Generate Single Signal ======
def generate_signal(data, model, scaler, features):
    X = data[features].iloc[-1:]
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    return "Buy" if prediction == 1 else "Sell"

# ====== Simulate Trading ======
def simulate_trading(data, signal):
    """
    Simulates trading based on the signal.
    Returns:
        portfolio_value (float): Total value of the portfolio including holdings and cash.
        profit_loss (float): Profit or loss from the current trade.
    """
    global cash, holdings, trade_log, buy_queue
    price = data['close'].iloc[-1]
    timestamp = data['open_time'].iloc[-1]
    profit_loss = 0  # Default profit/loss is 0 for non-trading cases.

    # Handle Buy Signal
    if signal == "Buy" and cash > 0 and len(buy_queue) < max_trades_per_cycle:
        # Enforce risk control: ensure price difference meets criteria
        if buy_queue and abs(price - buy_queue[-1]["Buy Price"]) <= trade_fee_rate:
            logger.warning(f"Buy skipped due to price difference constraint. Current Price: {price}")
            return cash + sum([trade["Holdings"] * price for trade in buy_queue]), profit_loss

        buy_amount = min(cash * max_buy_percentage, cash)
        cost = buy_amount * trade_fee_rate
        holdings = (buy_amount - cost) / price
        cash -= buy_amount
        buy_id = str(uuid.uuid4())
        buy_trade = {
            "Type": "Buy",
            "ID": buy_id,
            "Buy Price": price,
            "Buy Amount": buy_amount,
            "Buy Fee": cost,
            "Holdings": holdings,
            "Time": timestamp
        }
        buy_queue.append(buy_trade)
        trade_log.append(buy_trade)
        logger.info(f"Executed Buy: {buy_trade}")

    # Handle Sell Signal
    elif signal == "Sell" and buy_queue:
        # Find the buy trade with the lowest buy price
        buy_trade = min(buy_queue, key=lambda x: x["Buy Price"])
        expected_profit = price - buy_trade["Buy Price"]

        # Check if the profit condition is met
        if expected_profit > profit_rate * buy_trade["Buy Price"]:
            buy_queue.remove(buy_trade)
            sell_value = buy_trade["Holdings"] * price
            cost = sell_value * trade_fee_rate
            profit = sell_value - cost - buy_trade["Buy Amount"]
            profit_loss = profit  # Record the profit/loss of this trade.
            sell_trade = {
                "Type": "Sell",
                "ID": buy_trade["ID"],
                "Sell Price": price,
                "Sell Amount": sell_value,
                "Sell Fee": cost,
                "Profit": profit,
                "Time": timestamp
            }
            trade_log.append(sell_trade)
            logger.info(f"Executed Sell: {sell_trade}")

            # Update cash and clear holdings
            cash += sell_value - cost

    # Calculate total holdings from all buy_queue entries
    total_holdings_value = sum([trade["Holdings"] * price for trade in buy_queue])
    portfolio_value = cash + total_holdings_value

    return portfolio_value, profit_loss

# ====== Main Loop ======
if __name__ == "__main__":
    try:
        model, scaler = None, None
        weights = {"MACD": 1, "RSI": 1, "ATR": 1}  # 初始技术指标权重
        accuracy = 0  # 模型准确率
        consecutive_losses = 0  # 连续亏损次数

        while True:
            # 获取数据并计算技术指标
            data = get_binance_data()
            data = calculate_technical_indicators(data, weights)
            data, features = create_features(data, trade_log)

            # 初始训练或重新训练
            if model is None or accuracy < 0.70 or consecutive_losses > 3:
                logger.info("Training or Retraining the model...")
                model, scaler, accuracy = train_model(data, features, trade_log)

                # 根据准确率动态调整循环间隔和K线监控间隔
                if accuracy > 0.85:
                    loop_interval = 60
                    kline_interval = Client.KLINE_INTERVAL_1MINUTE
                elif accuracy > 0.75:
                    loop_interval = 300
                    kline_interval = Client.KLINE_INTERVAL_5MINUTE
                else:
                    loop_interval = 900
                    kline_interval = Client.KLINE_INTERVAL_15MINUTE

                # 调整技术指标权重
                weights["MACD"] *= (1 + accuracy / 10)
                weights["RSI"] *= (1 - accuracy / 20)
                logger.info(f"Adjusted Weights: {weights}")
            
            # 生成交易信号
            signal = generate_signal(data, model, scaler, features)

            # 执行交易并记录
            portfolio_value, profit_loss = simulate_trading(data, signal)
            logger.info(f"Portfolio Value: {portfolio_value:.2f}, Profit/Loss: {profit_loss:.2f}")

            # 根据交易收益更新状态
            if profit_loss < 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0

            # 短暂休息后重新进入循环
            time.sleep(loop_interval)

    except KeyboardInterrupt:
        logger.info("Program terminated.")
