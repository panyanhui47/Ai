import pandas as pd
import ccxt
import os
import time
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# pip install pandas ccxt newsapi vaderSentiment

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
    limit = 1000  # 每次请求最多获取1000条数据
    all_data = []

    while True:
        print("请求数据")
        ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        print(ohlcv)
        if not ohlcv:
            break  # 如果没有更多数据，退出循环
        
        all_data += ohlcv  # 添加到总数据中
        since = ohlcv[-1][0] + 1  # 更新 since 为最后一条数据时间戳

        print(f"[INFO] 获取到 {len(ohlcv)} 条数据，总数据量：{len(all_data)} 条")
        
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
    # if local_df is not None:
    #     combined_df = pd.concat([local_df, new_df])
    #     combined_df = combined_df[~combined_df.index.duplicated(keep='last')]  # 去重
    # else:
    #     combined_df = new_df

    # 如果本地有数据，进行去重
    if local_df is not None:
        # 找到新数据中没有在本地数据中出现的部分

        new_data = new_df[~new_df.index.isin(local_df.index)]
        if not new_data.empty:
            # 追加新数据到本地文件，避免重复
            new_data.to_csv(save_path, mode='a', header=False)
        else:
            print("[INFO] 新数据与本地数据完全一致，无需追加。")
        

        combined_df = pd.concat([local_df, new_df])
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]  # 去重
    else:
        # 如果本地没有数据文件，直接保存新数据
        new_df.to_csv(save_path)

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


def get_binance_order_data(symbol='ETH/USDT', timeframe='1m', days=7, save_path='binance_data_with_order_book.csv', save_interval=30):
    """
    获取Binance数据并保存，去重并按需追加，只写入文件中不存在的数据。
    
    :param symbol: 交易对符号，如 'ETH/USDT'
    :param timeframe: 数据时间框架，如 '1m'
    :param days: 获取的历史数据天数
    :param save_path: 保存CSV文件的路径
    :param save_interval: 每次循环后保存数据的间隔（次数）
    :return: 完整的历史数据
    """
    print(f"1、[INFO] 开始获取 {symbol} 数据，时间框架：{timeframe}，天数：{days}……")
    
    binance = ccxt.binance()
    
    # 如果文件存在，加载现有数据
    if os.path.exists(save_path):
        print(f"[INFO] 检测到本地数据文件：{save_path}，尝试加载已有数据……")
        local_df = pd.read_csv(save_path, parse_dates=['timestamp'], index_col='timestamp')
        print(f"[INFO] 本地数据加载成功，共计 {len(local_df)} 条。")
        
        # 去除重复的时间戳，保留最后一条数据
        local_df = local_df[~local_df.index.duplicated(keep='last')]
        print(f"[INFO] 去重后的数据共计 {len(local_df)} 条。")
        
        # 获取文件中最后一条数据的时间戳，作为下一次请求的起点
        last_timestamp = int(local_df.index[-1].timestamp() * 1000)
    else:
        print(f"[INFO] 未找到本地数据文件，将从头开始获取……")
        local_df = pd.DataFrame()  # 初始化为空数据框
        last_timestamp = int((pd.Timestamp.now() - pd.Timedelta(days=days)).timestamp() * 1000)

    since = last_timestamp
    human_readable_since = pd.to_datetime(since, unit='ms')
    print(f"Since (timestamp): {since}")
    print(f"Since (human-readable): {human_readable_since}")

    limit = 1000
    all_data = []  # 用于存储所有历史数据
    cycle_count = 0
    total_data_count = len(local_df) if not local_df.empty else 0  # 初始化总数据量

    while True:
        try:
            # 获取OHLCV数据
            ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)

            if not ohlcv:
                print("[INFO] 没有更多数据，退出循环")
                break

            for data in ohlcv:
                timestamp = data[0]
                ohlcv_time = pd.to_datetime(timestamp, unit='ms')

                # 检查时间戳是否已存在，避免重复
                if not local_df.empty and ohlcv_time in local_df.index:
                    print(f"[INFO] 时间戳 {ohlcv_time} 已存在，跳过该数据。")
                    continue

                # 仅在最新的OHLCV数据时获取order book数据
                if (pd.Timestamp.now() - ohlcv_time).total_seconds() <= 60:  # 1分钟内
                    order_book = binance.fetch_order_book(symbol)
                    bids = order_book['bids'][:10]
                    asks = order_book['asks'][:10]
                    print("订单簿买单:", bids)
                    print("订单簿卖单:", asks)
                else:
                    bids, asks = [], []  # 如果时间不匹配，赋空值，避免重复请求

                # 收集数据
                all_data.append({
                    'timestamp': ohlcv_time,
                    'Open': data[1],
                    'High': data[2],
                    'Low': data[3],
                    'Close': data[4],
                    'Volume': data[5],
                    'bids': bids,
                    'asks': asks
                })

            cycle_count += 1
            total_data_count += len(ohlcv)  # 更新总数据量

            # 每隔一定周期保存数据
            if cycle_count >= save_interval:
                print(f"[INFO] 已完成 {cycle_count} 次循环，保存当前数据……")
                df = pd.DataFrame(all_data)

                # 展开bids和asks数据
                df_bids = pd.json_normalize(df['bids'].explode())
                df_asks = pd.json_normalize(df['asks'].explode())

                # 合并数据并去重
                df = pd.concat([df.drop(columns=['bids', 'asks']), df_bids.add_prefix('bids_'), df_asks.add_prefix('asks_')], axis=1)
                df.set_index('timestamp', inplace=True)

                # 读取文件中的现有数据
                if os.path.exists(save_path):
                    existing_df = pd.read_csv(save_path, parse_dates=['timestamp'], index_col='timestamp')

                    # 去除重复数据并确保新数据不重复
                    df = pd.concat([existing_df, df])
                    df = df[~df.index.duplicated(keep='last')]  # 去重保留最后一条数据

                    # 过滤掉已经存在的时间戳的数据
                    df = df[~df.index.isin(existing_df.index)]

                # 追加保存数据
                df.to_csv(save_path, mode='a', header=not os.path.exists(save_path))
                print(f"[INFO] 数据已保存到本地文件：{save_path}，新保存数据量：{len(df)} 条。")

                cycle_count = 0  # 重置循环计数
                all_data = []  # 清空数据缓冲区

            # 更新since时间戳
            since = ohlcv[-1][0] + 1
            readable_since = pd.to_datetime(since, unit='ms')
            print(f"[INFO] {readable_since} 获取到 {len(ohlcv)} 条数据，总数据量：{total_data_count} 条")

            time.sleep(2)  # 增加请求间隔

            if len(ohlcv) < limit:
                print("[INFO] 获取的数据不足1000条，说明没有更多数据")
                break

        except Exception as e:
            print(f"[ERROR] 请求失败: {e}")
            time.sleep(5)  # 请求失败后，等待5秒后重试

    # 如果有剩余数据，保存
    if all_data:
        print(f"[INFO] 获取完成，保存剩余数据……")
        df = pd.DataFrame(all_data)
        df_bids = pd.json_normalize(df['bids'].explode())
        df_asks = pd.json_normalize(df['asks'].explode())
        df = pd.concat([df.drop(columns=['bids', 'asks']), df_bids.add_prefix('bids_'), df_asks.add_prefix('asks_')], axis=1)
        df.set_index('timestamp', inplace=True)

        # 检查文件是否存在，并去重保存
        if os.path.exists(save_path):
            existing_df = pd.read_csv(save_path, parse_dates=['timestamp'], index_col='timestamp')

            # 过滤掉已经存在的时间戳的数据
            df = df[~df.index.isin(existing_df.index)]

        df.to_csv(save_path, mode='a', header=not os.path.exists(save_path))
        print(f"[INFO] 数据已保存到本地文件：{save_path}，新保存数据量：{len(df)} 条。")





# 主程序代码
if __name__ == "__main__":
    import argparse
    # 初始化 ArgumentParser
    parser = argparse.ArgumentParser(description="Fetch Binance order data")
    # 添加命令行参数，并设置默认值
    parser.add_argument('--symbol', type=str, default='ETH/USDT', help="Trading pair symbol (default: 'ETH/USDT')")
    parser.add_argument('--interval', type=str, default='1m', help="Interval (default: '1m')")
    parser.add_argument('--limit', type=int, default=1825, help="Limit for data (number of records, default: 1825)")
    parser.add_argument('--param', type=int, default=10, help="Additional parameter (default: 10)")

    # 解析命令行参数
    args = parser.parse_args()
    # 自动生成文件名
    filename = f"{args.symbol.replace('/', '_')}_{args.interval}_binance_data.csv"

    # 调用函数，使用传递的参数和自动生成的文件名
    get_binance_order_data(args.symbol, args.interval, args.limit, filename, args.param)

    # df = get_binance_order_data('ETH/USDT','1m',1825,'ETH_USDT_1m_binance_data.csv',10)