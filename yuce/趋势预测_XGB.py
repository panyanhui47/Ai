import pandas as pd
import numpy as np
# import cupy as cp
import talib
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier, HistGradientBoostingClassifier, VotingClassifier
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
# from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

import ccxt, time, os, joblib
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter
# from xgboost import XGBClassifier
# import xgboost as xgb
# import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, TerminateOnNaN, EarlyStopping
import tensorflow as tf
import logging
import tensorflow.keras.backend as K

K.clear_session()  # 清理 Keras 会话，释放 GPU 内存

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# 获取ETH/USD历史数据
def get_binance_data(symbol='ETH/USDT', timeframe='1m', days=7, save_path='binance_data.csv'):
    """
    获取配置化的历史数据，并增量保存到本地（可配置交易对、时间框架和时间范围）

    Args:
        symbol (str): 交易对（如 'ETH/USDT'）
        timeframe (str): 时间框架（如 '1m', '5m', '1h'）
        days (int): 向前获取的天数
        save_path (str): 保存数据的本地文件路径

    Returns:
        pd.DataFrame: 包含历史K线数据的DataFrame
    """
    print(f"1、[INFO] 开始获取 {symbol} 数据，时间框架：{timeframe}，天数：{days}……")
    
    # 使用ccxt库连接Binance
    binance = ccxt.binance()
    
    # 读取已有的本地数据（如果存在）
    if os.path.exists(save_path):
        print(f"[INFO] 检测到本地数据文件：{save_path}，尝试加载已有数据……")
        local_df = pd.read_csv(save_path, parse_dates=['timestamp'], index_col='timestamp')
        print(f"[INFO] 本地数据加载成功，共计 {len(local_df)} 条。")
        last_timestamp = int(local_df.index[-1].timestamp() * 1000)  # 本地数据的最后时间戳
    else:
        print(f"[INFO] 未找到本地数据文件，将从头开始获取……")
        local_df = None
        last_timestamp = int((pd.Timestamp.now() - pd.Timedelta(days=days)).timestamp() * 1000)

    # 计算 since 时间戳（以毫秒为单位）
    since = last_timestamp
    # 将 since 转换为人类可读格式
    human_readable_since = pd.to_datetime(since, unit='ms')
    print(f"Since (timestamp): {since}")
    print(f"Since (human-readable): {human_readable_since}")

    limit = 1000  # 每次请求最多获取1000条数据
    all_data = []

    while True:
        # 请求数据
        ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        
        if not ohlcv:
            break  # 如果没有更多数据，退出循环
        
        all_data += ohlcv  # 添加到总数据中
        since = ohlcv[-1][0] + 1  # 更新 since 为最后一条数据时间戳

        readable_since = pd.to_datetime(since, unit='ms')
        print(f"[INFO] {readable_since} 获取到 {len(ohlcv)} 条数据，总数据量：{len(all_data)} 条")
        
        # 防止 API 请求过快被限速
        time.sleep(1)

        # 如果获取的数据不足1000条，说明没有更多数据
        if len(ohlcv) < limit:
            break

    # 如果没有新数据，直接返回本地数据
    if not all_data:
        print(f"[INFO] 没有新数据需要更新，返回本地数据。")
        return local_df

    # 转换新数据为DataFrame
    new_df = pd.DataFrame(all_data, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
    new_df.set_index('timestamp', inplace=True)
    
    # 合并新数据和本地数据（去重）
    if local_df is not None:
        combined_df = pd.concat([local_df, new_df])
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]  # 去重
    else:
        combined_df = new_df

    # 保存到本地
    combined_df.to_csv(save_path)
    print(f"[INFO] 数据已保存到本地文件：{save_path}，总数据量：{len(combined_df)} 条。")

    return combined_df


# 获取新闻数据并分析情感
def get_news_sentiment():
    print(f"3、[INFO] 获取新闻数据……")
    # 使用NewsAPI连接获取新闻数据
    newsapi = NewsApiClient(api_key='3a7a664be7a543d6934e9fc06cd1385f')  # 替换为您的API Key
    all_articles = newsapi.get_everything(q='Ethereum', language='en', sort_by='publishedAt', page_size=100)
    
    # 提取新闻标题
    articles = all_articles['articles']
    headlines = [article['title'] for article in articles]
    timestamps = [article['publishedAt'] for article in articles]
    
    # 初始化VADER情感分析工具
    analyzer = SentimentIntensityAnalyzer()
    
    # 对新闻标题进行情感分析
    sentiments = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
    
    # 将新闻情感分析结果和时间戳一起转换为DataFrame
    sentiment_df = pd.DataFrame({
        'Sentiment': sentiments,
        'publishedAt': pd.to_datetime(timestamps)  # 将时间戳转换为Datetime
    })

    # 设置时间戳为索引
    sentiment_df.set_index('publishedAt', inplace=True)
    # 去除时区信息
    sentiment_df.index = sentiment_df.index.tz_localize(None)
    # 确保索引为 datetime 类型
    sentiment_df.index = pd.to_datetime(sentiment_df.index)
    print(f"3.1、[INFO] 纯新闻数据……")
    print(sentiment_df.tail())
    return sentiment_df

def add_news_sentiment_feature(df):
    print(f"4、[INFO] 合并新闻数据……")
    
    sentiment_df = get_news_sentiment()
    
    # 确保市场数据的时间戳是 Datetime 类型，且去除时区信息
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)
    # 对 sentiment_df 进行按时间戳升序排序
    sentiment_df = sentiment_df.sort_index()
    
    # 合并情感数据，使用 merge_asof 按时间戳进行近似匹配
    merged_df = pd.merge_asof(df, sentiment_df, left_index=True, right_index=True, direction='backward')

    # 添加平滑窗口
    print(f"4.2、[INFO] 对情感分数进行平滑处理……")
    # 3小时平滑
    merged_df['Sentiment_Smoothed_3h'] = merged_df['Sentiment'].rolling(window=3, min_periods=1).mean()
    
    # 6小时平滑
    merged_df['Sentiment_Smoothed_6h'] = merged_df['Sentiment'].rolling(window=6, min_periods=1).mean()

    # 打印合并后的 df
    print(f"4.1、[INFO] 打印合并后的数据……")
    print(merged_df.tail())
    
    return merged_df


# 计算技术指标
def calculate_technical_indicators(df,timeframe):
    """
    根据时间框架动态计算技术指标
    Args:
        df (DataFrame): 包含价格数据的 DataFrame
        timeframe (str): 时间框架 (如 '1m', '5m', '1h', '1d')
    Returns:
        DataFrame: 添加了技术指标的 DataFrame
    """
    print(f"2、[INFO] 技术指标数据……")
    # 时间框架对应的分钟数
    timeframe_minutes = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }
    # 确保时间框架合法
    if timeframe not in timeframe_minutes:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Supported values: {list(timeframe_minutes.keys())}")
    # 获取时间框架对应的分钟数
    multiplier = timeframe_minutes[timeframe]

    # 确保 'Close' 列被转换为一维 numpy 数组
    close_prices = np.array(df['Close']).flatten()
    high_prices = np.array(df['High']).flatten()
    low_prices = np.array(df['Low']).flatten()
    volume = np.array(df['Volume']).flatten()  # 增加成交量

    # 动态调整技术指标的时间周期
    ma_short_period = 50 // multiplier  # MA50 动态调整
    ma_long_period = 200 // multiplier  # MA200 动态调整
    adx_period = 14 // multiplier       # ADX 动态调整
    bb_period = 20 // multiplier        # 布林带动态调整
    atr_period = 14 // multiplier       # ATR 动态调整
    rsi_period = 14 // multiplier       # RSI 动态调整

    # 确保周期不为 0（特别是时间框架较大的情况下）
    ma_short_period = max(ma_short_period, 1)
    ma_long_period = max(ma_long_period, 1)
    adx_period = max(adx_period, 1)
    bb_period = max(bb_period, 1)
    atr_period = max(atr_period, 1)
    rsi_period = max(rsi_period, 1)
    # 计算技术指标
    df['MA50'] = talib.SMA(close_prices, timeperiod=ma_short_period)
    df['MA200'] = talib.SMA(close_prices, timeperiod=ma_long_period)
    df['ADX'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=adx_period)
    df['UpperBand'], df['MiddleBand'], df['LowerBand'] = talib.BBANDS(close_prices, timeperiod=bb_period)
    df['ATR'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=atr_period)
    df['RSI'] = talib.RSI(close_prices, timeperiod=rsi_period)

    # 将成交量加入数据框作为特征
    df['Volume'] = volume

    # 检查是否有多级索引
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)  # 删除 Price 层级
    
    print(f"2.1、[INFO] 打印计算后的技术指标数据……")
    print(df.tail())
    
    return df

# 特征和标签生成
def generate_labels_and_features(df, window_size=20):
    """
    生成标签和特征数据
    :param df: 包含所有数据的DataFrame
    :param window_size: 滚动窗口的大小
    :return: 特征数据和标签数据
    """
    # 计算价格变化并生成标签
    df['PriceChange'] = df['Close'].pct_change(-1)  # 计算价格变化
    # df['Label'] = np.where(df['PriceChange'] > 0, 1, 0)  # 将标签改为 0 或 1
    df['Label'] = np.where(df['PriceChange'] > 0.001, 1, np.where(df['PriceChange'] < -0.001, 2, 0))  # 保持 1, 0, 2



    # 去除NaN数据（特别是由于pct_change产生的NaN）
    df = df.dropna(subset=['PriceChange', 'Label'])

    # 过滤掉无关列
    features = df.drop(columns=['Label', 'PriceChange'], errors='ignore')  # 确保去掉无关列

    # 获取滚动窗口数据
    def get_rolling_window_data(data, window_size):
        X = []
        for i in range(len(data) - window_size):
            if i + window_size <= len(data):  # 确保不越界
                X.append(data[i:i + window_size])  # 按照窗口大小提取数据
        return np.array(X)

    # 获取特征数据和标签
    X = get_rolling_window_data(features.values, window_size)
    y = df['Label'].iloc[window_size:].values  # 标签需要从第 window_size 个数据开始

    logger.info(f"特征数据形状: {X.shape}")
    logger.info(f"标签数据形状: {y.shape}")

    # 确保特征和标签的样本数一致
    if X.shape[0] != len(y):
        logger.error(f"特征样本数 ({X.shape[0]}) 和标签样本数 ({len(y)}) 不匹配")
        raise ValueError("特征和标签的样本数量不匹配")

    print(f"标签的唯一值: {np.unique(y)}")

    # 检查并去除可能的无效值
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        logger.warning("特征数据中存在NaN或Inf: \n {}".format(X))
        X = np.nan_to_num(X)  # 用 0 填充 NaN 或 Inf
        logger.warning("用 0 填充特征数据: \n {}".format(X))

    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        logger.warning("标签数据中存在NaN或Inf: \n {}".format(y))
        y = np.nan_to_num(y)  # 用 0 填充 NaN 或 Inf
        logger.warning("用 0 填充标签数据: \n {}".format(y))


    return X, y

def create_data_generator(X, y, batch_size=32):
    """
    创建数据生成器，用于按批次加载数据
    :param X: 特征数据
    :param y: 标签数据
    :param batch_size: 每个批次的大小
    :return: 数据生成器
    """
    # 转换为 TensorFlow 张量
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.int32)

    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices((X_tensor, y_tensor))
    dataset = dataset.batch(batch_size)  # 设置批次大小
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)  # 自动优化数据加载
    return dataset

def train_model(df, window_size=20, epochs=5000, batch_size=16):
    """
    执行初步训练的LSTM流程
    :param df: 包含特征和标签的数据框
    :param window_size: 滚动窗口大小，使用过去多少时间步的数据作为输入
    :param epochs: 训练的轮次
    :param batch_size: 每个批次的样本数
    """
    logger.info("执行初步训练的LSTM流程")

    # 准备数据
    X, y = generate_labels_and_features(df, window_size)
    # 调试标签生成
    print(f"生成的标签值: {y[:10]}")  # 输出前10个标签，检查是否有无效值
    
    # 检查可用的物理设备（GPU）
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logger.info("GPU设备已启用，内存增长已设置")

    # 创建数据生成器
    dataset = create_data_generator(X, y, batch_size)

    # 标签平衡处理：计算类别权重
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(class_weights))

    # 初始化LSTM模型
    logger.info("初始化LSTM模型...")
    model = Sequential()
    model.add(Input(shape=(window_size, X.shape[2])))  # 输入形状根据特征数量
    model.add(LSTM(100, activation='tanh', return_sequences=True))  # 使用tanh激活函数
    model.add(Dropout(0.3))  # 添加Dropout层来防止过拟合
    model.add(LSTM(50, activation='tanh', return_sequences=False))  # 再加一层 LSTM
    model.add(Dropout(0.3))  # 继续增加 Dropout
    model.add(Dense(3, activation='softmax'))  # 使用softmax激活函数输出三分类结果
    optimizer = Adam(learning_rate=0.0001, beta_1=0.85, beta_2=0.99, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    logger.info("LSTM模型初始化完成")

    # 定义学习率调度器
    lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, min_lr=1e-6, verbose=1)
    # 验证集的准确率没有改善，可以提前停止训练，避免过拟合
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # 初步训练模型
    logger.info("开始初步训练...")
    try:
        model.fit(dataset, epochs=epochs, batch_size=batch_size, verbose=1, 
        class_weight=class_weight_dict, callbacks=[lr_scheduler, early_stopping])
    except Exception as e:
        logger.error(f"训练出错: {e}")
        logger.debug(f"X: {X}")
        logger.debug(f"y: {y}")
    logger.info("初步训练完成")




# 保存模型
def save_model(model, path='model.pkl'):
    joblib.dump(model, path)
    print(f"[INFO] 模型已保存至 {path}")

# 加载模型
def load_model(path='model.pkl'):
    model = joblib.load(path)
    print(f"[INFO] 模型已加载")
    return model

# 模拟预测
def simulate_prediction(model, df):
    print(f"5、[INFO] 模拟实时预测……")
    features = df[['MA50', 'MA200', 'ADX', 'UpperBand', 'LowerBand', 'ATR', 'RSI', 'Volume', 'Sentiment_Smoothed_6h']].iloc[-1]
    prediction = model.predict(features.values.reshape(1, -1))[0]
    action = "Buy" if prediction == 1 else "Sell" if prediction == -1 else "Hold"
    print(f"Predicted Action: {action}")
    return action

# 综合分析
def comprehensive_analysis(df, model):
    # 预测时确保输入数据包含特征名称
    latest_data = pd.DataFrame(
        df.iloc[-1][['MA50', 'MA200', 'ADX', 'UpperBand', 'LowerBand', 'ATR', 'RSI', 'Volume', 'Sentiment_Smoothed_6h']].values.reshape(1, -1),
        columns=['MA50', 'MA200', 'ADX', 'UpperBand', 'LowerBand', 'ATR', 'RSI', 'Volume', 'Sentiment_Smoothed_6h']
    )
    
    # 预测未来趋势
    prediction = model.predict(latest_data)[0]
    
    # 根据技术指标和机器学习结果综合分析
    ma_signal = 'Buy' if df['MA50'].iloc[-1] > df['MA200'].iloc[-1] else 'Sell'
    rsi_signal = 'Overbought' if df['RSI'].iloc[-1] > 70 else ('Oversold' if df['RSI'].iloc[-1] < 30 else 'Neutral')
    
    trend = 'Uptrend' if prediction == 1 else ('Downtrend' if prediction == -1 else 'Sideways')
    suggestion = 'Buy' if prediction == 1 else ('Sell' if prediction == -1 else 'Wait')
    
    print(f"Current Trend: {trend}")
    print(f"MA Signal: {ma_signal}")
    print(f"RSI Signal: {rsi_signal}")
    print(f"Suggested Action: {suggestion}")

    # 可视化
    # plt.figure(figsize=(10, 6))
    # plt.plot(df['Close'], label='Close Price')
    # plt.plot(df['MA50'], label='50-Day MA')
    # plt.plot(df['MA200'], label='200-Day MA')
    # plt.title('ETH/USD Price and Moving Averages')
    # plt.legend()
    # plt.show()



def backtest_model(model, df, initial_balance=10000, transaction_fee=0.001):
    """
    批量处理提高回测速度
    """
    print(f"6、[INFO] 开始回测模型……")
    balance = initial_balance
    position = 0  # 持仓状态，0 表示空仓，正值表示持仓
    performance = []

    # 提取所有特征并批量预测
    features = df[['MA50', 'MA200', 'ADX', 'UpperBand', 'LowerBand', 'ATR', 'RSI', 'Volume', 'Sentiment_Smoothed_6h']].fillna(0)
    predictions = model.predict(features.values)  # 一次性预测所有数据

    # 遍历预测结果进行回测
    for i in range(1, len(df)):
        prediction = predictions[i]
        current_price = df['Close'].iloc[i]

        # 买入信号
        if prediction == 1 and position == 0:
            position = balance / current_price
            balance -= balance * transaction_fee
            performance.append({'action': 'Buy', 'price': current_price, 'balance': balance, 'timestamp': df.index[i]})

        # 卖出信号
        elif prediction == -1 and position > 0:
            balance += position * current_price
            balance -= balance * transaction_fee
            position = 0
            performance.append({'action': 'Sell', 'price': current_price, 'balance': balance, 'timestamp': df.index[i]})

    # 清仓
    if position > 0:
        balance += position * df['Close'].iloc[-1]
        performance.append({'action': 'End Sell', 'price': df['Close'].iloc[-1], 'balance': balance, 'timestamp': df.index[-1]})

    final_balance = round(balance, 2)
    print(f"[INFO] 回测完成，最终资金: ${final_balance}")
    performance_df = pd.DataFrame(performance)
    return final_balance, performance_df


def simulate_realtime_prediction(model, df, initial_balance=10000, transaction_fee=0.001):
    """
    使用实时数据模拟预测并执行决策
    Args:
        model: 训练好的机器学习模型
        df: 实时数据（特征计算完成后的数据）
        initial_balance: 初始资金
        transaction_fee: 每笔交易的手续费比例 (默认 0.1%)
    """
    print(f"7、[INFO] 开始实时预测……")
    balance = initial_balance
    position = 0  # 持仓状态

    # 遍历实时数据
    for i in range(len(df)):
        # 提取实时数据的特征
        features = df[['MA50', 'MA200', 'ADX', 'UpperBand', 'LowerBand', 'ATR', 'RSI', 'Volume', 'Sentiment_Smoothed_6h']].iloc[i]
        features = features.fillna(0)

        # 模型预测
        prediction = model.predict(features.values.reshape(1, -1))[0]

        # 当前价格
        current_price = df['Close'].iloc[i]

        # 决策逻辑
        if prediction == 1 and position == 0:  # 买入信号
            position = balance / current_price
            balance -= balance * transaction_fee
            print(f"[ACTION] 买入: {current_price}, 当前余额: {balance:.2f}")

        elif prediction == -1 and position > 0:  # 卖出信号
            balance += position * current_price
            balance -= balance * transaction_fee
            position = 0
            print(f"[ACTION] 卖出: {current_price}, 当前余额: {balance:.2f}")

    # 强制清仓
    if position > 0:
        balance += position * df['Close'].iloc[-1]
        print(f"[INFO] 模拟结束，最终余额: {balance:.2f}")

def evaluate_performance(performance_df, initial_balance):
    """
    评估模型回测的表现
    Args:
        performance_df: 包含每笔交易的数据
        initial_balance: 初始资金
    Returns:
        metrics: 包含关键指标的字典
    """
    print(f"8、[INFO] 评估回测表现……")

    # 计算收益率
    final_balance = performance_df['balance'].iloc[-1]
    total_return = (final_balance - initial_balance) / initial_balance

    # 计算胜率
    wins = performance_df[performance_df['action'] == 'Sell']['balance'].diff().dropna() > 0
    win_rate = wins.mean()

    # 计算最大回撤
    max_drawdown = (performance_df['balance'].cummax() - performance_df['balance']).max()

    # 计算夏普比率
    daily_returns = performance_df['balance'].pct_change().dropna()
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)  # 年化夏普比率

    metrics = {
        'Total Return': f"{total_return:.2%}",
        'Win Rate': f"{win_rate:.2%}",
        'Max Drawdown': f"${max_drawdown:.2f}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}"
    }
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return metrics

def backtest_and_retrain(data, model, initial_balance=10000, window_size=500, step_size=100):
    """
    分段式回测与再训练函数
    """
    print("[INFO] 开始分段式回测与再训练……")
    balance = initial_balance
    performance_records = []

    for start_idx in range(0, len(data) - window_size, step_size):
        end_idx = start_idx + window_size
        train_data = data.iloc[start_idx:end_idx]

        # 特征和标签
        features_train, labels_train = generate_features_and_labels(train_data)

        # 检查训练数据是否足够
        if features_train.empty or labels_train.empty:
            print(f"[WARN] Not enough data to train at window [{start_idx}:{end_idx}]. Skipping...")
            continue

        # 模型再训练
        model = train_predict_model(features_train, labels_train)

        # 模拟交易
        test_data = data.iloc[end_idx:end_idx + step_size]
        features_test, labels_test = generate_features_and_labels(test_data)

        if not features_test.empty:
            prediction = model.predict(features_test.values)
            action = "Buy" if prediction[-1] == 1 else "Sell"
            print(f"[INFO] Window [{start_idx}:{end_idx}]: Predicted Action: {action}")

            # 更新余额逻辑
            if action == "Buy":
                balance *= 1.01  # 模拟 1% 收益
            elif action == "Sell":
                balance *= 0.99  # 模拟 1% 损失

        performance_records.append({
            "start_idx": start_idx,
            "end_idx": end_idx,
            "balance": balance
        })

    performance_df = pd.DataFrame(performance_records)
    return balance, performance_df



def rolling_window_backtest_and_retrain(df, window_size=1000, retrain_interval=100, initial_balance=10000):
    """
    滚动窗口方法实现回测和再训练
    Args:
        df: 历史数据
        window_size: 每次训练的时间窗口大小
        retrain_interval: 每次滚动的步长
        initial_balance: 初始资金
    
    Returns:
        final_balance: 回测后的最终资金
        performance_df: 每次回测的结果
    """
    print(f"[INFO] 滚动窗口回测与再训练开始……")
    balance = initial_balance
    position = 0
    performance = []

    for start_idx in range(0, len(df) - window_size, retrain_interval):
        # 定义训练窗口和回测窗口
        train_window = df.iloc[start_idx:start_idx + window_size]
        test_window = df.iloc[start_idx + window_size:start_idx + window_size + retrain_interval]

        # 特征和标签生成
        features_train, labels_train = generate_features_and_labels(train_window)
        model = train_predict_model(features_train, labels_train)

        # 回测当前测试窗口
        for i in range(len(test_window)):
            features_test = test_window[['MA50', 'MA200', 'ADX', 'UpperBand', 'LowerBand', 'ATR', 'RSI', 'Volume', 'Sentiment_Smoothed_6h']].iloc[i]
            features_test = features_test.fillna(0)

            # 模型预测
            prediction = model.predict(features_test.values.reshape(1, -1))[0]
            current_price = test_window['Close'].iloc[i]

            # 买入信号
            if prediction == 1 and position == 0:
                position = balance / current_price
                balance -= balance * 0.001
                performance.append({'action': 'Buy', 'price': current_price, 'balance': balance, 'timestamp': test_window.index[i]})

            # 卖出信号
            elif prediction == -1 and position > 0:
                balance += position * current_price
                balance -= balance * 0.001
                position = 0
                performance.append({'action': 'Sell', 'price': current_price, 'balance': balance, 'timestamp': test_window.index[i]})

    # 清仓
    if position > 0:
        balance += position * df['Close'].iloc[-1]
        performance.append({'action': 'End Sell', 'price': df['Close'].iloc[-1], 'balance': balance, 'timestamp': df.index[-1]})

    final_balance = round(balance, 2)
    print(f"[INFO] 滚动窗口回测完成，最终资金: ${final_balance}")
    performance_df = pd.DataFrame(performance)
    return final_balance, performance_df


# 运行脚本
if __name__ == "__main__":
    symbol = 'ETH/USDT'
    timeframe = '1m'  # 时间框架 (如 '1m', '5m', '1h', '1d')
    days = 30  # 数据跨度 (天)
    save_path = 'ETH_USDT_1m_binance_data.csv'

    # 获取数据
    df = get_binance_data(symbol,timeframe,days,save_path)

    
    # 计算技术指标
    df = calculate_technical_indicators(df, timeframe)
    df = add_news_sentiment_feature(df)
    
    # 生成特征和标签
    # feature, labels = generate_labels_and_features(df)
    
    # 训练模型并进行预测
    # model = train_predict_model(feature, labels)
    train_model(df, window_size=20, epochs=5000, batch_size=64)
    
    # 保存模型
    # save_model(model)

    # 模拟实时预测
    # simulate_prediction(model, df)
    # 综合分析
    # comprehensive_analysis(df, model)
    
    # 回测模型

    # print("回测")
    # final_balance, performance_df = backtest_model(model, df)
    # print("评估")
    # evaluate_performance(performance_df, initial_balance=10000)

    # 模拟实时预测
    # print("实时预测")
    # simulate_realtime_prediction(model, df.tail(100))  # 用最后 100 条数据模拟实时预测

    # # 执行分段式回测与再训练
    # # final_balance, performance_df = backtest_and_retrain(
    # #     model=None,  # 初始模型可以为空，由函数内部生成
    # #     df=df,
    # #     initial_balance=10000,
    # #     retrain_window=500
    # # )


    # # # 评估回测表现
    # # evaluate_performance(performance_df, initial_balance=10000)

    # # 或者使用滚动窗口
    # final_balance, performance_df = rolling_window_backtest_and_retrain(df, window_size=1000, retrain_interval=200)

