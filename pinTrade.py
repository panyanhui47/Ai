import os
import websocket
from websocket import WebSocketApp
import json
import requests
import hmac
import hashlib
import time
from datetime import datetime
import pandas as pd
import sqlite3
import logging
import signal
import sys
import json
import numpy as np

# 日志配置
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Binance API 配置
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
BASE_URL = "https://api.binance.com"

# 风控参数
PRICE_THRESHOLD = 0.05  # 5%的价格变动触发插针
MAX_SLIPPAGE = 0.01  # 最大滑点 1%
TRADE_SYMBOLS = ["ETHUSDT", "BTCUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"]  # 支持的交易对
TRADE_QUANTITY = 0.01  # 每次交易数量

# 数据存储
market_data = {symbol: [] for symbol in TRADE_SYMBOLS}  # 每个币种单独存储数据
stop_signal_received = False  # 全局标志变量

# 数据库初始化
def init_db():
    conn = sqlite3.connect("pin_trades.db")
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS trades (symbol TEXT, time TEXT, price REAL, volume REAL, result TEXT)"
    )
    conn.commit()
    conn.close()

# 保存交易日志
def log_trade(symbol, event):
    conn = sqlite3.connect("pin_trades.db")
    cursor = conn.cursor()
    # 确保 `event['result']` 为字符串
    result_str = json.dumps(event['result']) if isinstance(event['result'], dict) else str(event['result'])
    cursor.execute(
        "INSERT INTO trades (symbol, time, price, volume, result) VALUES (?, ?, ?, ?, ?)",
        (symbol, event['time'], event['price'], event['volume'], result_str),
    )
    conn.commit()
    conn.close()

# 下单函数
def create_order(symbol, side, quantity, price=None):
    endpoint = "/api/v3/order"
    params = {
        "symbol": symbol,
        "side": side,
        "type": "LIMIT" if price else "MARKET",
        "quantity": quantity,
        "price": price,
        "timeInForce": "GTC" if price else None,
        "timestamp": int(time.time() * 1000),
    }
    params = {k: v for k, v in params.items() if v is not None}
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    params["signature"] = signature

    headers = {"X-MBX-APIKEY": API_KEY}
    response = requests.post(BASE_URL + endpoint, headers=headers, params=params)
    
    if response.status_code == 200:
        logger.info(f"下单成功 ({symbol}) - 响应: {response.json()}")
    else:
        logger.error(f"下单失败 ({symbol}) - 状态码: {response.status_code}, 响应: {response.text}")
    
    return response.json()

# 风控逻辑
def risk_control(order_price, market_price, max_slippage=MAX_SLIPPAGE):
    slippage = abs(order_price - market_price) / market_price
    logger.debug(f"风控检查 - 订单价格: {order_price}, 市场价格: {market_price}, 滑点: {slippage}")
    return slippage <= max_slippage

# 时间转换函数
def convert_to_readable_time(timestamp_ms, local_time=True):
    timestamp_seconds = timestamp_ms / 1000
    if local_time:
        # 转换为本地时间
        return datetime.fromtimestamp(timestamp_seconds).strftime('%Y-%m-%d %H:%M:%S')
    else:
        # 转换为 UTC 时间
        return datetime.utcfromtimestamp(timestamp_seconds).strftime('%Y-%m-%d %H:%M:%S')

# 插针检测函数
def detect_pin_bar(kline, volume_data, k=2, p=0.2, m=2):
    """
    检测下插针，忽略上插针。

    :param kline: 当前 K 线数据 (dict)
    :param volume_data: 最近历史成交量数据 (list of float)
    :param k: 下影线与实体比例的阈值 (float)
    :param p: 实体占总振幅比例的阈值 (float)
    :param m: 当前成交量与平均成交量的倍数阈值 (float)
    :return: 是否为下插针 (bool), 详细检测信息 (dict)
    """
    open_price = float(kline['o'])
    close_price = float(kline['c'])
    high_price = float(kline['h'])
    low_price = float(kline['l'])
    current_volume = float(kline['v'])

    # 计算影线和实体长度
    # upper_shadow = high_price - max(open_price, close_price)
    lower_shadow = min(open_price, close_price) - low_price
    body = abs(close_price - open_price)

    # 防止实体为 0 导致除零错误
    body = max(body, 1e-8)

    # 计算历史成交量均值
    if len(volume_data) > 0:
        average_volume = sum(volume_data) / len(volume_data)
    else:
        average_volume = 0

    # 插针检测条件
    # long_shadow = (upper_shadow > k * body) or (lower_shadow > k * body)
    long_lower_shadow = lower_shadow > k * body  # 下影线是否较长
    small_body = abs(open_price - close_price) < p * (high_price - low_price)  # 实体占振幅比例
    volume_spike = current_volume > m * average_volume # 成交量是否异常放大

    # 返回检测结果和细节
    return long_lower_shadow  and small_body and volume_spike, {
        # "upper_shadow": upper_shadow,
        "lower_shadow": lower_shadow,
        "body": body,
        "long_shadow": long_lower_shadow,
        "small_body": small_body,
        "volume_spike": volume_spike,
        "average_volume": average_volume,
        "current_volume": current_volume,
    }

def detect_downward_trend(recent_prices, threshold=0.01):
    """
    检测最近的价格是否呈现下降趋势。

    :param recent_prices: 最近的价格列表 (list of float)
    :param threshold: 判断趋势的价格变化率阈值，例如 0.01 表示 1% (float)
    :return: 是否呈现下降趋势 (bool), 趋势描述 (str)
    """
    if len(recent_prices) < 2:
        return False, "数据不足"

    # 使用线性回归检测趋势
    x = np.arange(len(recent_prices))
    y = np.array(recent_prices)
    slope, _ = np.polyfit(x, y, 1)  # 拟合斜率

    # 判断斜率是否为负（下降趋势）
    if slope < -threshold:
        return True, f"下降趋势，斜率: {slope:.4f}"
    else:
        return False, f"无明显下降趋势，斜率: {slope:.4f}"

# 实时监控
def on_message(ws, message):
    global market_data
    try:
        data = json.loads(message)

        # 忽略订阅确认消息
        if 'result' in data and data['result'] is None:
            return

        # 检测插针
        if 'k' in data:
            kline = data['k']
            symbol = kline['s']  # 当前交易对
            if symbol not in market_data:
                logger.warning(f"未监控的交易对: {symbol}")
                return
            # 获取最近窗口数据（最多最近 10 条）
            recent_data = market_data[symbol][-10:] if len(market_data[symbol]) > 10 else market_data[symbol]
            recent_prices = [item['price'] for item in recent_data]

            volume_data = [item['volume'] for item in market_data[symbol][-10:]]  # 最近 10 条历史成交量
            is_pin, details = detect_pin_bar(kline, volume_data)
            
            # logger.info(f"插针检测详情 ({symbol}): {details}")
            if is_pin:
                # 检测是否为下跌趋势
                downward_trend, trend_description = detect_downward_trend(recent_prices)

                logger.info(f"\n")
                logger.info(f"插针检测详情 ({symbol}): {details}")
                logger.info(f"检测到插针 ({symbol}) - 价格: {kline['c']}, 成交量: {kline['v']}")
                logger.info(f"趋势判断 ({symbol}): {trend_description}")

                if downward_trend:
                    # 下跌趋势，不下单
                    logger.warning(f"检测到插针 ({symbol}) - 价格: {kline['c']}, 但存在向下趋势，忽略下单。")
                    return  # 跳过下单逻辑

                # 风控检查
                last_price = float(kline['c'])
                if risk_control(last_price, last_price):
                    logger.info(f"风控通过 ({symbol}) - 准备下单: {last_price}")
                    order_result = create_order(symbol, "BUY", TRADE_QUANTITY)
                    log_trade(symbol, {
                        'time': convert_to_readable_time(kline['t'], local_time=True),
                        'price': last_price,
                        'volume': float(kline['v']),
                        'result': order_result,
                    })
                    logger.info("下单结果 (%s): %s", symbol, order_result)
                else:
                    logger.warning(f"风控未通过 ({symbol}) - 下单被拦截")

            # 更新市场数据
            market_data[symbol].append({
                "time": kline['t'],
                "price": float(kline['c']),
                "volume": float(kline['v']),
            })
            if len(market_data[symbol]) > 50:
                market_data[symbol].pop(0)

    except Exception as e:
        logger.error(f"error from callback {on_message}: {e}")

# 错误处理
def on_error(ws, error):
    logger.error("WebSocket错误: %s", error)

# 连接关闭处理
def on_close(ws, close_status_code, close_msg):
    logger.info("WebSocket关闭")

# 打开连接
def on_open(ws):
    payload = {
        "method": "SUBSCRIBE",
        "params": [f"{symbol.lower()}@kline_1s" for symbol in TRADE_SYMBOLS],  # 订阅多个交易对的 1 秒 K 线
        "id": 1,
    }
    ws.send(json.dumps(payload))

# 停止 WebSocket
def stop_websocket(signum, frame):
    global stop_signal_received
    stop_signal_received = True
    logger.info("接收到停止信号，程序正在退出...")
    sys.exit(0)

# 启动 WebSocket
def start_websocket():
    retry_count = 0
    while not stop_signal_received:
        try:
            socket = "wss://stream.binance.com:9443/ws"
            logger.info(f"尝试连接 WebSocket，第 {retry_count + 1} 次...")
            ws = websocket.WebSocketApp(
                socket,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            ws.on_open = on_open
            ws.run_forever()
        except Exception as e:
            retry_count += 1
            logger.error(f"WebSocket连接失败，第 {retry_count} 次重试: {e}")
            time.sleep(5)

# 主程序
if __name__ == "__main__":
    # 注册信号处理器
    signal.signal(signal.SIGINT, stop_websocket)
    signal.signal(signal.SIGTERM, stop_websocket)
    
    init_db()
    logger.info("插针接单系统启动中...")
    start_websocket()
