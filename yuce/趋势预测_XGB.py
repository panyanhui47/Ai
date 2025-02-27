import pandas as pd
import numpy as np
# import cupy as cp
import talib
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier, HistGradientBoostingClassifier, VotingClassifier
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
# from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
# from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, TerminateOnNaN, EarlyStopping, ModelCheckpoint, Callback, TensorBoard
import tensorflow as tf
import logging
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Precision, Recall
import keras_tuner as kt
from functools import partial

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

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
    # 6小时平滑
    merged_df['Sentiment_Smoothed_6h'] = merged_df['Sentiment'].rolling(window=6, min_periods=1).mean()

    # 移除原始的 Sentiment 列
    merged_df.drop(columns=['Sentiment'], inplace=True)

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

    # 仅对技术指标列保留两位小数，其他列保持不变
    df['MA50'] = df['MA50'].round(4)
    df['MA200'] = df['MA200'].round(4)
    df['ADX'] = df['ADX'].round(4)
    df['UpperBand'] = df['UpperBand'].round(4)
    df['MiddleBand'] = df['MiddleBand'].round(4)
    df['LowerBand'] = df['LowerBand'].round(4)
    df['ATR'] = df['ATR'].round(4)
    df['RSI'] = df['RSI'].round(4)
    
    print(f"2.1、[INFO] 打印计算后的技术指标数据……")
    print(df.tail())
    
    return df

# 特征和标签生成
def generate_labels_and_features(df, window_size=20, train_ratio=0.8):
    """
    生成标签和特征数据
    :param df: 包含所有数据的DataFrame
    :param window_size: 滚动窗口的大小
    :param train_ratio: 训练集的比例（默认80%训练，20%验证）
    :return: 训练集特征、训练集标签、验证集特征和验证集标签
    """
    # # 计算价格变化并生成标签
    df['PriceChange'] = df['Close'].pct_change(1)  # 计算当前时间步和前一个时间步的价格变化（百分比变化）
    # df['Label'] = np.where(df['PriceChange'] > 0.001, 1, np.where(df['PriceChange'] < -0.001, 2, 0))  # 保持 1, 0, 2
    
    # 检查 PriceChange 中的 NaN
    logger.info(f"PriceChange 列中 NaN 值的数量: {df['PriceChange'].isna().sum()}")

    # 去除NaN数据（特别是由于pct_change产生的NaN）
    df = df.dropna(subset=['PriceChange', 'Label'])

    # 检查数据是否仍然存在NaN值
    logger.info(f"数据去除NaN后的大小: {df.shape}")
    
     # 删除头部 200 行: 这是因为你需要等待MA200的值才能开始训练
    df = df.iloc[200:]
 
    # 筛选出包含 NaN 值的行
    nan_rows = df[df.isna().any(axis=1)]
    logger.info(f"筛选出包含 NaN 值的行: {nan_rows}")
    # nan_rows.to_csv('nan_rows.csv') # 如果需要可以保存 NaN 行
    logger.info(f"df数据的头几行:\n{df.head()}")
    logger.info(f"df数据的尾几行:\n{df.tail()}")
    # 过滤掉无关列
    features = df.drop(columns=['Label', 'PriceChange'], errors='ignore')  # 确保去掉无关列

    logger.info(f"特征数据的头几行:\n{features.head()}")
    logger.info(f"特征数据的尾几行:\n{features.tail()}")
    logger.info(f"特征数据的NaN数量: {np.sum(np.isnan(features.values))}")

    # =================================================================================
    # 1. 标准化 K线数据 (Open, High, Low, Close, Volume)
    kline_columns = ['Open', 'High', 'Low', 'Close', 'Volume']  # K线数据的列
    kline_data = features[kline_columns]

    # 使用 StandardScaler 对 K线数据进行标准化
    scaler_kline = StandardScaler()
    features[kline_columns] = scaler_kline.fit_transform(kline_data)

    # 2. 对技术指标进行 Min-Max 缩放
    indicator_columns = ['MA50', 'MA200', 'ADX', 'UpperBand', 'MiddleBand', 'LowerBand', 'ATR', 'RSI']
    indicator_data = features[indicator_columns]

    # 使用 MinMaxScaler 对技术指标数据进行缩放
    scaler_indicator = MinMaxScaler(feature_range=(0, 1))
    features[indicator_columns] = scaler_indicator.fit_transform(indicator_data)

    # 将 features 转换为 Pandas DataFrame 并输出前几行
    features_scaled = pd.DataFrame(features)  # 将标准化后的特征数据转换为 DataFrame
    # 删除 timestamp 列
    # features_scaled = features_scaled.drop(columns=['timestamp'], errors='ignore')  # 后期可取时间周期作为特征，而不是时间本身，但索引列默认是不传入模型的
    # =================================================================================

    logger.info(f"特征数据【处理后】的头几行:\n{features_scaled.head()}")
    logger.info(f"特征数据【处理后】的尾几行:\n{features_scaled.tail()}")
    
    logger.info(f"特征数据【处理后】的NaN数量: {np.sum(np.isnan(features_scaled.values))}")

    # 获取滚动窗口数据
    def get_rolling_window_data(data, labels, window_size, train_ratio=0.8):
        X_train, y_train, X_val, y_val = [], [], [], []
        
        # 假设原始数据是按时间序列顺序排好
        for i in range(len(data) - window_size):
            # 获取当前窗口的特征数据，排除标签列
            X_window = data[i:i + window_size, :]
            y_window = labels[i + window_size]  # 获取当前窗口对应的标签

            # 跟踪输出头两个和末尾两个窗口的数据
            if i < 2 or i >= len(data) - window_size - 2:
                X_window_flat = X_window.flatten()  # 将窗口数据展平成一维数组
                print(f"窗口 {i} 的特征数据: {', '.join(map(str, X_window_flat))}, 标签数据: {y_window}")

            # 切分训练集和验证集
            if i < len(data) * train_ratio:  # 判断是训练集还是验证集
                X_train.append(X_window)  # 将完整的窗口添加到训练集
                y_train.append(y_window)  # 标签对应的值也添加到训练集
            else:
                X_val.append(X_window)  # 将完整的窗口添加到验证集
                y_val.append(y_window)  # 标签对应的值也添加到验证集

        return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)

    # 获取特征数据和标签
    X_train, y_train, X_val, y_val = get_rolling_window_data(features_scaled.values, df['Label'].values, window_size, train_ratio)

    logger.info(f"训练特征数据形状: {X_train.shape}")
    logger.info(f"训练标签数据形状: {y_train.shape}")
    logger.info(f"验证特征数据形状: {X_val.shape}")
    logger.info(f"验证标签数据形状: {y_val.shape}")

    # 确保特征和标签的样本数一致
    if X_train.shape[0] != len(y_train) or X_val.shape[0] != len(y_val):
        logger.error(f"特征样本数与标签样本数不匹配")
        raise ValueError("特征和标签的样本数量不匹配")

    # 检查并处理训练数据中的 NaN 值
    if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
        logger.warning("训练数据 X_train 包含 NaN 值，正在处理...")

        # 输出 X_train 和 y_train 中 NaN 的数量
        logger.info(f"训练数据 X_train 中 NaN 值的数量: {np.sum(np.isnan(X_train))}")
        logger.info(f"训练数据 y_train 中 NaN 值的数量: {np.sum(np.isnan(y_train))}")

        # 对 X_train 中的 NaN 值进行线性插值
        if np.any(np.isnan(X_train)):
            logger.warning("X_train 中包含 NaN 值，正在用 0 填充...")
            X_train = np.nan_to_num(X_train, nan=0)  # 使用 0 填充 NaN

        # 对 y_train 中的 NaN 值进行填充，使用 0 填充
        if np.any(np.isnan(y_train)):
            logger.warning("y_train 中包含 NaN 值，正在用 0 填充...")
            y_train = np.nan_to_num(y_train, nan=0)  # 使用 0 填充 NaN

    # 检查并处理验证数据中的 NaN 值
    if np.any(np.isnan(X_val)) or np.any(np.isnan(y_val)):
        logger.warning("验证数据 X_val 包含 NaN 值，正在处理...")

        # 输出 X_val 和 y_val 中 NaN 的数量
        logger.info(f"验证数据 X_val 中 NaN 值的数量: {np.sum(np.isnan(X_val))}")
        logger.info(f"验证数据 y_val 中 NaN 值的数量: {np.sum(np.isnan(y_val))}")

        # 对 X_val 中的 NaN 值进行线性插值
        if np.any(np.isnan(X_val)):
            logger.warning("X_val 中包含 NaN 值，正在用 0 填充...")
            X_val = np.nan_to_num(X_val, nan=0)  # 使用 0 填充 NaN

        # 对 y_val 中的 NaN 值进行填充，使用 0 填充
        if np.any(np.isnan(y_val)):
            logger.warning("y_val 中包含 NaN 值，正在用 0 填充...")
            y_val = np.nan_to_num(y_val, nan=0)  # 使用 0 填充 NaN

    return X_train, y_train, X_val, y_val

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

# 精确率计算
def precision(y_true, y_pred):
    y_true = K.cast(y_true, 'int32')
    y_pred = K.cast(K.argmax(y_pred, axis=-1), 'int32')  # 获取最大概率的类别
    true_positives = K.sum(K.cast(y_true * y_pred, 'float32'))
    predicted_positives = K.sum(K.cast(y_pred, 'float32'))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# 召回率计算
def recall(y_true, y_pred):
    y_true = K.cast(y_true, 'int32')
    y_pred = K.cast(K.argmax(y_pred, axis=-1), 'int32')  # 获取最大概率的类别
    true_positives = K.sum(K.cast(y_true * y_pred, 'float32'))
    possible_positives = K.sum(K.cast(y_true, 'float32'))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# F1 分数计算
def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r + K.epsilon())

# 为每个类别计算精确率、召回率、F1 分数
def class_metrics(y_true, y_pred, class_weight=None, num_classes=2):
    """
    计算分类的精确率、召回率和 F1 分数（宏平均和加权平均）。
    :param y_true: 真实标签，形状为 (batch_size, )，整数编码
    :param y_pred: 模型预测的概率分布，形状为 (batch_size, num_classes)
    :param class_weight: 类别权重的字典，形如 {0: weight_0, 1: weight_1, ...}
    :param num_classes: 类别数量
    :return: 一个字典，包含宏平均和加权平均的精确率、召回率和 F1 分数
    """
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # 计算类别分布（如果提供了 class_weight）
    label_counts = K.sum(K.one_hot(y_true, num_classes), axis=0)
    total_labels = K.sum(label_counts)
    weights = (
        [class_weight.get(i, 1.0) for i in range(num_classes)]
        if class_weight is not None
        else [1.0] * num_classes
    )

    for i in range(num_classes):
        # 当前类别的真实标签和预测标签
        true_class = K.cast(K.equal(y_true, i), "float32")
        predicted_class = K.cast(K.equal(K.argmax(y_pred, axis=-1), i), "float32")

        # 计算精确率、召回率和 F1 分数
        precision_class = K.sum(true_class * predicted_class) / (K.sum(predicted_class) + K.epsilon())
        recall_class = K.sum(true_class * predicted_class) / (K.sum(true_class) + K.epsilon())
        f1_class = 2 * precision_class * recall_class / (precision_class + recall_class + K.epsilon())

        precision_scores.append(precision_class)
        recall_scores.append(recall_class)
        f1_scores.append(f1_class)

        # 打印每个类别的指标
        logger.info(
            f"标签 {i}: 精确率 = {precision_class.numpy():.4f}, 召回率 = {recall_class.numpy():.4f}, F1 分数 = {f1_class.numpy():.4f}"
        )

    # 计算宏平均
    macro_precision = K.mean(K.stack(precision_scores))
    macro_recall = K.mean(K.stack(recall_scores))
    macro_f1 = K.mean(K.stack(f1_scores))

    # 计算加权平均
    weighted_precision = K.sum(
        K.stack(precision_scores) * (label_counts / total_labels)
    )
    weighted_recall = K.sum(
        K.stack(recall_scores) * (label_counts / total_labels)
    )
    weighted_f1 = K.sum(
        K.stack(f1_scores) * (label_counts / total_labels)
    )

    # 返回结果字典
    return {
        "macro_precision": macro_precision.numpy(),
        "macro_recall": macro_recall.numpy(),
        "macro_f1": macro_f1.numpy(),
        "weighted_precision": weighted_precision.numpy(),
        "weighted_recall": weighted_recall.numpy(),
        "weighted_f1": weighted_f1.numpy(),
    }

# 自定义回调类来计算每个 epoch 的 class_metrics
class ClassMetricsCallback(Callback):
    def __init__(self, val_data=None, class_weight=None):
        super().__init__()
        self.val_data = val_data  # 显示传入的验证数据
        self.class_weight = class_weight # 显示传入权重字典

    def on_epoch_end(self, epoch, logs=None):
        # 获取模型的预测值
        if self.val_data:
            X_val, y_val = self.val_data  # 获取验证数据
            y_pred = self.model.predict(X_val)  # 使用验证数据进行预测
            y_true = y_val
        else:
            # 如果验证数据不可用，抛出异常并退出
            raise ValueError(f"验证数据不可用 for epoch {epoch + 1}. Training stopped.")
        # 计算指标
        metrics = class_metrics(y_true, y_pred, class_weight=self.class_weight)
        
        # 打印每个类的宏平均和加权平均指标
        logger.info(f"Epoch {epoch}:")
        logger.info(f"宏精确率: {metrics['macro_precision']}")
        logger.info(f"宏召回率: {metrics['macro_recall']}")
        logger.info(f"宏F1分数: {metrics['macro_f1']}")
        logger.info(f"加权精确率: {metrics['weighted_precision']}")
        logger.info(f"加权召回率: {metrics['weighted_recall']}")
        logger.info(f"加权F1分数: {metrics['weighted_f1']}")

# 损失函数focal_loss
def focal_loss_with_class_weight(gamma=2., alpha=0.25, class_weight=None):
    """
    Focal Loss with class weighting for multi-class classification.
    :param gamma: focusing parameter, usually set to 2
    :param alpha: balancing factor, typically between 0 and 1
    :param class_weight: dictionary containing class weights, e.g. {0: 0.389, 1: 4.6, 2: 4.6}
    :return: loss function
    """
    def focal_loss_fixed(y_true, y_pred):
        print('y_true shape:', y_true.shape)
        print('y_pred shape:', y_pred.shape)

        # Ensure y_true is float32 to match y_pred's type
        y_true = K.cast(y_true, dtype='float32')  # Ensure y_true is float32
        
        # Clip predictions to prevent log(0)
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        
        # Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred)
        
        # Apply class weight
        if class_weight is not None:
            # Convert class_weight dict to tensor (shape: [num_classes])
            class_weight_tensor = tf.constant([class_weight.get(i, 1.0) for i in range(len(class_weight))], dtype=tf.float32)
            
            # For each sample, get the class index (using argmax) and the corresponding weight
            sample_weights = K.gather(class_weight_tensor, K.argmax(y_true, axis=-1))  # Shape: (batch_size,)
            sample_weights = K.expand_dims(sample_weights, axis=-1)  # Expand to shape (batch_size, 1) to match y_true's shape
        else:
            sample_weights = 1.0  # Default, no class weight applied
            sample_weights = K.expand_dims(sample_weights, axis=-1)  # Expand to shape (batch_size, 1)
        
        # Calculate the loss with focal term
        focal_loss = K.pow(1 - y_pred, gamma) * cross_entropy
        
        # Apply the sample weight to the focal loss
        weighted_loss = sample_weights * focal_loss
        
        return K.sum(weighted_loss, axis=-1)  # Sum over the classes (for each sample)

    return focal_loss_fixed


def build_model(hp, X, window_size, class_weight=None):
    """
    构建并返回LSTM模型，包含超参数调优
    :param hp: KerasTuner的超参数对象
    :param X: 特征数据，用于获取输入数据形状
    :param window_size: 滚动窗口大小
    :return: 编译后的Keras模型
    """
    model = Sequential()
    # 输入层
    model.add(Input(shape=(window_size, X.shape[2])))  # 输入形状为 (window_size, 特征数量)

    # 第一个Bidirectional LSTM层，超参数调优
    model.add(Bidirectional(LSTM(
        units=hp.Int('units', min_value=32, max_value=128, step=32),
        activation='tanh',
        return_sequences=True
    )))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))

    # 第二个Bidirectional LSTM层，超参数调优
    model.add(Bidirectional(LSTM(
        units=hp.Int('units_2', min_value=32, max_value=128, step=32),
        activation='tanh',
        return_sequences=False
    )))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))

    # 全连接Dense层，超参数调优
    model.add(Dense(
        hp.Int('dense_units', min_value=32, max_value=128, step=32),
        activation='relu'
    ))

    # 输出层（假设是三分类问题）
    model.add(Dense(3, activation='softmax'))

    # 编译模型，使用Adam优化器，并调节学习率
    optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log'))

    # 使用 Focal Loss with Class Weight 作为损失函数
    # loss_fn = focal_loss_with_class_weight(
    #     gamma=hp.Float('gamma', min_value=1.0, max_value=5.0, step=0.1),
    #     alpha=hp.Float('alpha', min_value=0.1, max_value=0.8, step=0.1),
    #     class_weight=class_weight
    # )

    model.compile(optimizer=optimizer, 
                    loss='sparse_categorical_crossentropy', # 用于多分类问题
                    # loss=loss_fn,
                    metrics=['accuracy'])

    return model

def restore_model_from_checkpoint(model, checkpoint_dir):
    """

    从断训点恢复模型和优化器的状态
    :param model: 当前模型
    :param checkpoint_dir: 断训点的保存目录
    :return: 恢复后的模型
    """
    checkpoint = tf.train.Checkpoint(model=model)  # 创建checkpoint对象
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)  # 获取最新的检查点路径
    if latest_checkpoint:
        logger.info(f"恢复模型从：{latest_checkpoint}")
        checkpoint.restore(latest_checkpoint)  # 恢复模型和优化器状态
        logger.info("模型和优化器状态已恢复")
    else:
        logger.info("未找到检查点，模型从头开始训练")
    
    return model

def train_model_with_tuning(df, window_size=20, train_ratio=0.8, epochs=5000, batch_size=16):
    """
    使用KerasTuner进行LSTM超参数自动调优
    :param df: 包含特征和标签的数据框
    :param window_size: 滚动窗口大小，使用过去多少时间步的数据作为输入
    :param train_ratio: 80% 用于训练，20% 用于验证
    :param epochs: 训练的轮次
    :param batch_size: 每个批次的样本数
    """
    logger.info("开始执行LSTM自动调参...")

    # 从超参数对象中获取 window_size 的值
    # window_size = hp.Int('window_size', min_value=10, max_value=100, step=10)

    # 准备数据
    X_train, y_train, X_val, y_val = generate_labels_and_features(df, window_size, train_ratio)

    # 计算类别权重，处理类别不平衡
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    
    # 打印每个类的权重
    logger.info(f"每个标签的权重:, {class_weights}")
    class_weight_dict = dict(enumerate(class_weights))

    # class_weight_dict[0] = 0.0  # 将标签为0的权重设为0，表示忽略这些样本
    logger.info(f"权重字典:, {class_weight_dict}")
    
    # 创建数据生成器（如果需要）
    # dataset = create_data_generator(X_train, y_train, batch_size)

    # 使用 `partial` 来传递 `X_train` 和 `window_size` 到 `build_model`
    build_model_partial = partial(build_model, X=X_train, 
                                    window_size=window_size, 
                                    class_weight=class_weight_dict)

    # 使用KerasTuner进行调参
    tuner = kt.Hyperband(
        build_model_partial,  # 使用部分参数化的模型构建函数
        objective='val_loss',  # 目标是验证集准确率
        max_epochs=epochs,
        factor=2,
        hyperband_iterations=3,
        directory='my_dir',  # 存储调参结果的目录
        project_name='LSTM_tuning'
    )
    
    # 定义学习率调度器和早期停止回调
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # # 添加模型检查点回调
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

    # # 添加断训检查点回调
    checkpoint_dir = 'checkpoint'
    # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_dir + "/model_checkpoint-{epoch}", 
    #     save_weights_only=True,
    #     save_freq='epoch',  # 每个 epoch 保存一次
    #     verbose=1
    # )
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "model_checkpoint-{epoch}.h5"), 
        save_weights_only=True,
        save_freq='epoch',  # 每个 epoch 保存一次
        verbose=1
    )
    # 创建TensorBoard回调
    log_dir = "logs"  # TensorBoard日志目录
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


    # 恢复之前的超参数搜索状态（如果有）
    if os.path.exists('my_dir/LSTM_tuning'):
        logger.info("恢复上次的超参数搜索状态...")
        tuner.reload()  # 恢复搜索进度

    # 进行超参数调优
    logger.info("开始超参数调优...")
    try:
        tuner.search(X_train, y_train, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    validation_data=(X_val, y_val), 
                    class_weight=class_weight_dict, 
                    callbacks=[ClassMetricsCallback(val_data=(X_val, y_val),
                                                    class_weight=class_weight_dict),
                                lr_scheduler, 
                                early_stopping, 
                                checkpoint_callback,
                                tensorboard_callback
                                ]
                    )

        # 检查是否有最佳模型
        best_trials = tuner.oracle.get_best_trials(num_trials=1)
        if not best_trials:
            raise ValueError("未找到最佳试验，超参数调优可能未能正常完成。")
        
        # 获取最佳模型
        best_hp = best_trials[0].hyperparameters
        best_model = build_model_partial(best_hp)  # 使用最佳超参数组合来构建模型
        
        logger.info("最优模型训练完成")
    
    except Exception as e:
        logger.error(f"超参数调优或模型训练失败：{e}")
        # 可以返回一个默认模型或执行回退操作
        best_model = None

    
    # 获取最佳模型
    best_model = tuner.get_best_models(num_models=1)[0]
    logger.info("最优模型训练完成")

    # 保存最终模型
    best_model.save('final_model.h5')
    
    return best_model



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
    # save_path = 'binance_data.csv'

    # 获取数据
    # df = get_binance_data(symbol,timeframe,days,save_path)
    df = pd.read_csv("labeled_kline_data.csv")
    df.set_index('timestamp', inplace=True)

    
    # 计算技术指标
    df = calculate_technical_indicators(df, timeframe)
    # df = add_news_sentiment_feature(df)
    
    # 生成特征和标签
    # feature, labels = generate_labels_and_features(df)
    
    # 训练模型并进行预测
    # model = train_predict_model(feature, labels)
    model = train_model_with_tuning(df, window_size=20, epochs=5000, batch_size=64)
    
    # 保存模型
    save_model(model)

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

