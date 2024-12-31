import pandas as pd
import numpy as np
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import ccxt
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter

# 获取ETH/USD历史数据
def get_eth_data():
    print(f"1、[INFO] 开始获取币安数据……")
    # 使用ccxt库连接Binance
    binance = ccxt.binance()
    
    # 获取ETH/USD的1小时历史数据，时间范围是1年
    ohlcv = binance.fetch_ohlcv('ETH/USDT', timeframe='1h', limit=5 * 365 * 24)  # 5 * 365天的1小时数据
    
    # 将数据转化为DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    
    # 将时间戳转换为日期格式
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # 设置时间戳为索引
    df.set_index('timestamp', inplace=True)
    print(df.head())
    return df

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
def calculate_technical_indicators(df):
    print(f"2、[INFO] 技术指标数据……")
    # 确保 'Close' 列被转换为一维 numpy 数组
    close_prices = np.array(df['Close']).flatten()
    high_prices = np.array(df['High']).flatten()
    low_prices = np.array(df['Low']).flatten()
    volume = np.array(df['Volume']).flatten()  # 增加成交量

    # 计算50日和200日移动平均线
    df['MA50'] = talib.SMA(close_prices, timeperiod=50)
    df['MA200'] = talib.SMA(close_prices, timeperiod=200)
    # 计算ADX
    df['ADX'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
    # 计算布林带
    df['UpperBand'], df['MiddleBand'], df['LowerBand'] = talib.BBANDS(close_prices, timeperiod=20)
    # 计算ATR
    df['ATR'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
    # 计算RSI
    df['RSI'] = talib.RSI(close_prices, timeperiod=14)

    # 将成交量加入数据框作为特征
    df['Volume'] = volume

    # 检查是否有多级索引
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)  # 删除 Price 层级

    print(df.tail())
    # print(df[['MA50', 'MA200', 'ADX', 'UpperBand', 'LowerBand', 'ATR', 'RSI', 'Volume']].tail())
    
    return df

# 特征和标签生成
def generate_features_and_labels(df):
    print(f"5、[INFO] 特征和标签生成……")
    # 检查 'NewsSentiment' 列是否存在
    if 'Sentiment_Smoothed_6h' not in df.columns:
        print(f"5.1、[WARN] 'Sentiment' column not found, skipping related operations.")
        # 如果没有 'NewsSentiment' 列，跳过相关的 dropna 操作
        df = df.dropna(subset=['MA50', 'MA200', 'ADX', 'UpperBand', 'LowerBand', 'ATR', 'RSI', 'Volume'])
    else:
        # 如果存在 'NewsSentiment' 列，进行包含该列的 dropna 操作
        df = df.dropna(subset=['MA50', 'MA200', 'ADX', 'UpperBand', 'LowerBand', 'ATR', 'RSI', 'Volume', 'Sentiment_Smoothed_6h',])

    # 填充缺失数据，确保数据完整性
    df = df.ffill()  # 使用显式的 forward fill（取代 fillna(method='ffill')）

    # 移动平均线需要200行的数据，确保没有NaN值
    df = df.dropna(subset=['MA50', 'MA200', 'ADX', 'UpperBand', 'LowerBand', 'ATR', 'RSI', 'Volume', 'Sentiment_Smoothed_6h'])
    
    # 对情感分数进行归一化处理
    df['Sentiment_Smoothed_6h'] = (df['Sentiment_Smoothed_6h'] - df['Sentiment_Smoothed_6h'].mean()) / df['Sentiment_Smoothed_6h'].std()

    
    # 显式地创建副本，避免警告
    df = df.copy()

    # 计算 PriceChange
    df['PriceChange'] = df['Close'].pct_change().shift(-1)

    # 设置 Label 列
    df['Label'] = np.where(df['PriceChange'] > 0, 1, (np.where(df['PriceChange'] < 0, -1, 0)))

    # 删除含 NaN 的标签行
    labels = df['Label'].dropna()

    # 移除类别 0 样本
    df = df[df['Label'] != 0]
    
    # 对齐 features 和 labels 的索引
    common_index = df.index.intersection(labels.index)
    features = df.loc[common_index, ['MA50', 'MA200', 'ADX', 'UpperBand', 'LowerBand', 'ATR', 'RSI', 'Volume', 'Sentiment_Smoothed_6h']]
    labels = labels.loc[common_index]

    # 检查类别分布
    print("Original label distribution:", Counter(labels))
    # 初始化 SMOTE 并调整 k_neighbors
    smote = SMOTE(random_state=42, k_neighbors=1)

    # 应用 SMOTE
    features_resampled, labels_resampled = smote.fit_resample(features, labels)

    # 检查新的类别分布
    print("Resampled label distribution:", Counter(labels_resampled))

    return features_resampled, labels_resampled

# 训练和预测模型
def train_predict_model(features, labels):
    # 数据拆分：训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # 使用随机森林模型
    rf_model = RandomForestClassifier(random_state=42)
    # 使用GridSearchCV调整超参数
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # 打印最佳参数和最佳得分
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

    # 使用最佳参数预测
    rf_model = grid_search.best_estimator_
    
    # 在测试集上评估模型
    y_pred = rf_model.predict(X_test)
    print(f"随机森林 准确率: {accuracy_score(y_test, y_pred)}")
    print(f"随机森林 Classification Report:\n{classification_report(y_test, y_pred)}")

    # 你还可以绘制模型评估图，如 ROC 曲线
    # 例如，绘制 ROC AUC 曲线：
    y_pred_prob = rf_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label='ROC Curve')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()
    
    return rf_model

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
    plt.figure(figsize=(10, 6))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['MA50'], label='50-Day MA')
    plt.plot(df['MA200'], label='200-Day MA')
    plt.title('ETH/USD Price and Moving Averages')
    plt.legend()
    plt.show()

# 运行脚本
if __name__ == "__main__":
    # 获取数据
    df = get_eth_data()
    
    # 计算技术指标
    df = calculate_technical_indicators(df)
    df = add_news_sentiment_feature(df)
    
    # 生成特征和标签
    features, labels = generate_features_and_labels(df)
    
    # 训练模型并进行预测
    model = train_predict_model(features, labels)
    
    # 综合分析
    comprehensive_analysis(df, model)
