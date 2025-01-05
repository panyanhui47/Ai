import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
import talib
from getdata import get_news_sentiment

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

def add_news_sentiment_feature(df):
    print(f"4、[INFO] 合并新闻数据……")

    # 获取新闻情感数据
    sentiment_df = get_news_sentiment()

    # 检查时间范围
    print("[DEBUG] 市场数据时间范围:", df.index.min(), "至", df.index.max())
    print("[DEBUG] 新闻数据时间范围:", sentiment_df.index.min(), "至", sentiment_df.index.max())

    # 确保索引已排序
    df = df.sort_index()
    sentiment_df = sentiment_df.sort_index()

    # 合并情感数据
    try:
        merged_df = pd.merge_asof(df, sentiment_df, left_index=True, right_index=True, direction='backward')
    except ValueError as e:
        print(f"[ERROR] 合并新闻数据时失败: {e}")
        return df  # 返回原始数据框，跳过情感分数处理

    # 检查合并结果
    print("[DEBUG] 合并后数据非空值统计:")
    print(merged_df[['Sentiment']].notna().sum())

    # 平滑处理情感数据
    if 'Sentiment' not in merged_df.columns or merged_df['Sentiment'].isna().all():
        print("[WARN] 无法生成情感分数，设置默认值 0")
        merged_df['Sentiment_Smoothed_6h'] = 0
    else:
        merged_df['Sentiment_Smoothed_3h'] = merged_df['Sentiment'].rolling(window=3, min_periods=1).mean()
        merged_df['Sentiment_Smoothed_6h'] = merged_df['Sentiment'].rolling(window=6, min_periods=1).mean()

    # 返回合并后的数据框
    return merged_df



# 特征和标签生成
def generate_features_and_labels(df):
    print(f"5、[INFO] 特征和标签生成……")
    print(f"[DEBUG] 数据框初始大小：{df.shape}")

    # 检查每列缺失值比例
    required_columns = ['MA50', 'MA200', 'ADX', 'UpperBand', 'LowerBand', 'ATR', 'RSI', 'Volume']
    if 'Sentiment_Smoothed_6h' in df.columns and df['Sentiment_Smoothed_6h'].notna().all():
        required_columns.append('Sentiment_Smoothed_6h')
    else:
        print("[WARN] 情感分数列不可用，将跳过该特征")

    # 填充缺失值
    df = df.ffill().bfill()

    # 丢弃 NaN 行
    print(f"Before dropna: {df.shape}")
    df = df.dropna(subset=required_columns)
    print(f"After dropna: {df.shape}")

    # 如果数据框为空，跳过当前窗口
    if df.empty:
        print("[ERROR] 数据框在处理后为空，无法生成特征和标签。")
        return None, None

    # 创建标签
    df['PriceChange'] = df['Close'].pct_change().shift(-1)
    df['Label'] = np.where(df['PriceChange'] > 0, 1, -1)

    # 检查标签分布
    label_counts = Counter(df['Label'])
    print(f"[INFO] 标签分布：{label_counts}")
    if len(label_counts) < 2:
        print("[WARN] 标签类别不足，跳过当前窗口。")
        return None, None

    # 提取特征和标签
    features = df[required_columns]
    labels = df['Label']

    # 应用 SMOTE 平衡数据
    try:
        smote = SMOTE(random_state=42, k_neighbors=1)
        features_resampled, labels_resampled = smote.fit_resample(features, labels)
    except ValueError as e:
        print(f"[ERROR] SMOTE 失败：{e}")
        return None, None

    # 检查平衡后的标签分布
    print(f"[INFO] 重新采样后标签分布：{Counter(labels_resampled)}")
    return features_resampled, labels_resampled

