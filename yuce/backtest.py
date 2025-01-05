import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from multiprocessing import Pool, cpu_count
from getdata import get_binance_data
from featuresLabels import generate_features_and_labels, calculate_technical_indicators, add_news_sentiment_feature
from tranModel import train_predict_model, load_model


def backtest_model_multi_positions(
    model,
    df,
    initial_balance=10000,
    transaction_fee=0.001,
    slippage=0.001,
    stop_loss=None,
    take_profit=None,
    num_positions=10  # 将资金分成几等份
):
    """
    回测函数：支持分批建仓和多个仓位同时持有。

    Args:
        model: 已训练的模型，用于生成买卖信号。
        df: 包含价格和技术指标的历史数据，要求包含特征列和 'Close' 列。
        initial_balance: 初始资金。
        transaction_fee: 每笔交易手续费（比例）。
        slippage: 每次交易的价格滑点（比例）。
        stop_loss: 止损点（相对买入价格的比例，例如 0.05 表示 5% 止损）。
        take_profit: 止盈点（相对买入价格的比例，例如 0.10 表示 10% 止盈）。
        num_positions: 将资金分成几等份，用于分批建仓。

    Returns:
        final_balance: 回测后的最终资金。
        performance_df: 包含每笔交易的详细信息。
    """
    print(f"[INFO] 开始回测模型（支持多仓位）……")
    
    # 初始化资金和仓位
    balance = initial_balance
    available_balance = initial_balance  # 可用资金
    positions = []  # 存储每个仓位信息，列表中每个仓位是一个字典
    performance = []  # 记录每笔交易

    # 提取特征和价格
    features = df[['MA50', 'MA200', 'ADX', 'UpperBand', 'LowerBand', 'ATR', 'RSI', 'Volume']].fillna(0)
    close_prices = df['Close'].values

    # 批量预测买卖信号
    predictions = model.predict(features.values)

    # 每批资金
    funds_per_trade = initial_balance / num_positions

    # 遍历数据生成交易记录
    for i in range(1, len(df)):
        signal = predictions[i]  # 当前预测信号：1=买入，-1=卖出
        current_price = close_prices[i]
        timestamp = df.index[i]

        # 模拟价格滑点
        effective_price = current_price * (1 + slippage if signal == 1 else 1 - slippage)

        # 买入逻辑（分批建仓）
        if signal == 1 and available_balance >= funds_per_trade:  # 有足够可用资金开新仓
            position_size = funds_per_trade / effective_price  # 计算购买量
            available_balance -= funds_per_trade  # 扣除占用资金
            balance -= funds_per_trade * transaction_fee  # 扣除手续费

            # 添加新仓位
            positions.append({'entry_price': effective_price, 'size': position_size, 'timestamp': timestamp})
            performance.append({
                'timestamp': timestamp,
                'action': 'Buy',
                'price': effective_price,
                'balance': balance,
                'available_balance': available_balance,
                'position_size': position_size
            })

        # 卖出逻辑（逐步清仓）
        elif signal == -1 and len(positions) > 0:  # 有持仓时清仓
            # 遍历持仓逐个清理
            for position in positions[:]:  # 遍历副本，避免修改时出错
                # 获取持仓信息
                entry_price = position['entry_price']
                position_size = position['size']

                # 检查是否触发止损或止盈
                if stop_loss is not None and current_price < entry_price * (1 - stop_loss):
                    action = 'Stop Loss'
                elif take_profit is not None and current_price > entry_price * (1 + take_profit):
                    action = 'Take Profit'
                else:
                    action = 'Sell'

                # 计算平仓收益
                balance += position_size * effective_price  # 增加资金
                balance -= balance * transaction_fee  # 扣除手续费
                available_balance += position_size * effective_price  # 恢复可用资金

                # 记录平仓信息
                performance.append({
                    'timestamp': timestamp,
                    'action': action,
                    'price': effective_price,
                    'balance': balance,
                    'available_balance': available_balance,
                    'position_size': position_size
                })

                # 移除该仓位
                positions.remove(position)

    # 强制清仓（回测结束时清仓）
    for position in positions:
        entry_price = position['entry_price']
        position_size = position['size']
        balance += position_size * close_prices[-1]
        performance.append({
            'timestamp': df.index[-1],
            'action': 'End Sell',
            'price': close_prices[-1],
            'balance': balance,
            'available_balance': available_balance,
            'position_size': position_size
        })

    # 计算最终资金
    final_balance = round(balance, 2)
    print(f"[INFO] 回测完成，最终资金: ${final_balance}")

    # 转换为 DataFrame
    performance_df = pd.DataFrame(performance)
    return final_balance, performance_df


def calculate_performance_metrics(trade_history, initial_balance):
    """
    计算回测绩效指标，包括收益率、最大回撤和夏普比率。

    Args:
        trade_history (pd.DataFrame): 包含每笔交易记录的数据。
        initial_balance (float): 初始资金。

    Returns:
        dict: 包括收益率、最大回撤和夏普比率的指标。
    """
    # 计算资金曲线
    equity_curve = trade_history['balance']

    # 总收益率
    total_return = (equity_curve.iloc[-1] - initial_balance) / initial_balance

    # 最大回撤
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min()

    # 夏普比率
    returns = equity_curve.pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)

    return {
        "Total Return": total_return,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio
    }


def rolling_window_backtest_and_retrain(
    df,
    model,
    initial_balance=10000,
    window_size=1000,
    retrain_interval=200,
    transaction_fee=0.001,
    slippage=0.001,
    stop_loss=None,
    take_profit=None,
    num_positions=10
):
    print(f"[INFO] 开始分段回测与再训练……")

    # 初始化资金和绩效记录
    balance = initial_balance
    overall_performance = []
    retrained_models = []

    for start in range(0, len(df) - window_size, retrain_interval):
        # 划分训练窗口和回测窗口
        train_window = df.iloc[start:start + window_size]
        test_window = df.iloc[start + window_size:start + window_size + retrain_interval]

        # 检查训练窗口是否为空
        if train_window.dropna(subset=['MA50', 'MA200', 'ADX', 'UpperBand', 'LowerBand', 'ATR', 'RSI', 'Volume']).shape[0] < 200:
            print("[WARN] Training window does not have enough valid data. Skipping this window.")
            continue

        # 生成特征和标签
        features_train, labels_train = generate_features_and_labels(train_window)
        if features_train is None or labels_train is None:
            print("[WARN] Skipping this window due to insufficient data or label diversity.")
            continue

        # 再训练模型
        retrained_model = train_predict_model(features_train, labels_train)
        retrained_models.append(retrained_model)

        # 回测当前窗口
        final_balance, performance_df = backtest_model_multi_positions(
            retrained_model,
            test_window,
            balance,
            transaction_fee,
            slippage,
            stop_loss,
            take_profit,
            num_positions
        )

        # 更新资金和总体绩效
        balance = final_balance
        overall_performance.append(performance_df)

    # 合并所有交易记录
    performance_df = pd.concat(overall_performance).reset_index(drop=True)

    # 计算绩效指标
    metrics = calculate_performance_metrics(performance_df, initial_balance)

    return balance, performance_df, metrics, retrained_models



def multi_asset_backtest(assets, model, initial_balance=10000):
    """
    多资产组合回测。

    Args:
        assets (dict): 每个资产对应的历史数据字典，格式为 {symbol: df}。
        model: 训练好的模型。
        initial_balance: 每个资产的初始资金。

    Returns:
        results: 包含每个资产的最终资金和绩效指标。
    """
    results = {}

    for symbol, df in assets.items():
        print(f"[INFO] 开始回测资产: {symbol}")
        final_balance, performance_df, metrics, _ = rolling_window_backtest_and_retrain(
            df,
            model,
            initial_balance
        )
        results[symbol] = {
            "Final Balance": final_balance,
            "Metrics": metrics,
            "Performance": performance_df
        }
        print(f"[INFO] {symbol} 回测完成，最终资金: {final_balance}")

    return results


def parallel_backtest(strategy_fn, params_list):
    """
    并行化回测。

    Args:
        strategy_fn: 策略回测函数。
        params_list: 回测参数列表。

    Returns:
        results: 每个策略的回测结果。
    """
    print(f"[INFO] 开始并行化回测，使用 {cpu_count()} 核心 CPU")
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(strategy_fn, params_list)
    return results


# 示例调用
if __name__ == "__main__":
    timeframe = '1m'  # 时间框架 (如 '1m', '5m', '1h', '1d')
    days = 1825  # 数据跨度 (天)
    # 加载数据与模型

    # assets = {
    #     "ETH/USDT": get_binance_data("ETH/USDT",timeframe,days,"ETH_USDT_1m_binance_data.csv")  # 替换为实际数据加载函数
    # }

    ethdf=get_binance_data("ETH/USDT",timeframe,days,"ETH_USDT_1m_binance_data.csv")
    # 1. 计算技术指标
    df = calculate_technical_indicators(ethdf, timeframe)
    # 2. 添加新闻情感分数
    # df = add_news_sentiment_feature(ethdf)

    model = load_model('model.pkl')  # 替换为实际模型加载函数

    # 检查是否包含数据（通常是 DataFrame 或类似格式）
    if hasattr(model, 'df'):
        print("模型中包含数据：")
        print(model.df.head())
    else:
        print("模型中没有包含数据集。")

    # 多资产回测
    # results = multi_asset_backtest(assets, model, initial_balance=10000)
    final_balance, performance_df, metrics, _ = rolling_window_backtest_and_retrain(
        df,
        model,
        10000
    )
    results_usdt = {
        "Final Balance": final_balance,
        "Metrics": metrics,
        "Performance": performance_df
    }
    print(f"[INFO] ETHUSDT 回测完成，最终资金: {final_balance}")


    print(f"ETHUSDT 绩效指标:")
    print(results_usdt["Metrics"])
    print("交易记录:")
    print(results_usdt["Performance"].head())
