import time
import pandas as pd
from binance.client import Client
import matplotlib.pyplot as plt
import numpy as np

# Твой fetch_klines_paged без изменений
def fetch_klines_paged(symbol, interval, total_bars=20000, client=None):
    if client is None:
        client = Client()
    limit = 1000
    data = []
    end_time = int(time.time() * 1000)

    while len(data) < total_bars:
        bars_to_fetch = min(limit, total_bars - len(data))
        klines = client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=bars_to_fetch,
            endTime=end_time
        )
        if not klines:
            break
        data = klines + data
        end_time = klines[0][0] - 1

    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    df[['open','high','low','close']] = df[['open','high','low','close']].astype(float)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('timestamp', inplace=True)  # удобно для plotting
    return df

class MarketStructureTracker:
    def __init__(self):
        self.trend = None
        self.last_hh = None
        self.last_hl = None
        self.last_lh = None
        self.last_ll = None
        self.swings = []  # list of (index, high, low, type or None)
        self.structure = []  # list of {'index':, 'type':, 'price':, 'label':}
        self.current_impulse = None  # {'type': 'bull'/'bear', 'low_price', 'high_price', 'low_index', 'high_index'}

    def add_candle(self, idx, high, low):
        self.swings.append((idx, high, low, None))

        # Проверяем, можно ли подтвердить pivot на предыдущей свече
        if len(self.swings) >= 3:
            # Свечи: -3 (left), -2 (potential pivot), -1 (right = current)
            left = self.swings[-3]
            pivot = self.swings[-2]
            right = self.swings[-1]

            pivot_idx, pivot_high, pivot_low, _ = pivot

            # Swing high?
            if pivot_high > left[1] and pivot_high > right[1]:
                self.swings[-2] = (pivot_idx, pivot_high, pivot_low, 'high')
                self._process_swing(pivot_idx, 'high', pivot_high)

            # Swing low?
            if pivot_low < left[2] and pivot_low < right[2]:
                self.swings[-2] = (pivot_idx, pivot_high, pivot_low, 'low')
                self._process_swing(pivot_idx, 'low', pivot_low)

    def _process_swing(self, idx, swing_type, price):
        label = None

        if swing_type == 'low':
            if self.trend in [None, 'bull']:
                if self.last_hl is None or price > self.last_hl['price']:
                    label = 'HL'
                    self.last_hl = {'index': idx, 'price': price}
                    if self.trend is None:
                        self.trend = 'bull'
                    # Обновляем/создаём бычий импульс
                    if self.current_impulse and self.current_impulse['type'] == 'bull':
                        self.current_impulse['low_price'] = price
                        self.current_impulse['low_index'] = idx
                    else:
                        self.current_impulse = {
                            'type': 'bull',
                            'low_price': price,
                            'high_price': None,
                            'low_index': idx,
                            'high_index': None
                        }
                else:
                    label = 'CHoCH'
                    self.trend = 'bear'
                    self.last_ll = {'index': idx, 'price': price}
                    self.last_hl = self.last_hh = None
                    self.current_impulse = {
                        'type': 'bear',
                        'low_price': price,
                        'high_price': self.last_lh['price'] if self.last_lh else price,
                        'low_index': idx,
                        'high_index': self.last_lh['index'] if self.last_lh else idx
                    }

            elif self.trend == 'bear':
                if self.last_ll is None or price < self.last_ll['price']:
                    label = 'LL'
                    self.last_ll = {'index': idx, 'price': price}
                    if self.current_impulse and self.current_impulse['type'] == 'bear':
                        self.current_impulse['low_price'] = price
                        self.current_impulse['low_index'] = idx
                else:
                    label = 'HL'  # weakening

        elif swing_type == 'high':
            if self.trend in [None, 'bear']:
                if self.last_lh is None or price < self.last_lh['price']:
                    label = 'LH'
                    self.last_lh = {'index': idx, 'price': price}
                    if self.trend is None:
                        self.trend = 'bear'
                    if self.current_impulse and self.current_impulse['type'] == 'bear':
                        self.current_impulse['high_price'] = price
                        self.current_impulse['high_index'] = idx
                    else:
                        self.current_impulse = {
                            'type': 'bear',
                            'high_price': price,
                            'low_price': None,
                            'high_index': idx,
                            'low_index': None
                        }
                else:
                    label = 'CHoCH'
                    self.trend = 'bull'
                    self.last_hh = {'index': idx, 'price': price}
                    self.last_ll = self.last_lh = None
                    self.current_impulse = {
                        'type': 'bull',
                        'high_price': price,
                        'low_price': self.last_hl['price'] if self.last_hl else price,
                        'high_index': idx,
                        'low_index': self.last_hl['index'] if self.last_hl else idx
                    }

            elif self.trend == 'bull':
                if self.last_hh is None or price > self.last_hh['price']:
                    label = 'HH'
                    self.last_hh = {'index': idx, 'price': price}
                    if self.current_impulse and self.current_impulse['type'] == 'bull':
                        self.current_impulse['high_price'] = price
                        self.current_impulse['high_index'] = idx
                else:
                    label = 'LH'  # weakening

        if label:
            self.structure.append({'index': idx, 'type': swing_type, 'price': price, 'label': label})

def calculate_trade_stats(trades):
    if not trades:
        return {
            'total_trades': 0,
            'total_pnl': 0.0,
            'winrate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_win': 0.0,
            'max_loss': 0.0
        }

    total_trades = len(trades)
    wins = 0
    total_pnl = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    wins_list = []
    losses_list = []

    for trade in trades:
        pnl = trade['profit']
        total_pnl += pnl
        if pnl > 0:
            wins += 1
            gross_profit += pnl
            wins_list.append(pnl)
        else:
            gross_loss += abs(pnl)
            losses_list.append(pnl)

    winrate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    avg_win = sum(wins_list) / len(wins_list) if wins_list else 0.0
    avg_loss = sum(losses_list) / len(losses_list) if losses_list else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf') if gross_profit > 0 else 0.0

    return {
        'total_trades': total_trades,
        'total_pnl': round(total_pnl, 2),
        'winrate': round(winrate, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else '∞',
        'max_win': round(max(wins_list), 2) if wins_list else 0.0,
        'max_loss': round(min(losses_list), 2) if losses_list else 0.0
    }

# Обновлённая функция run_backtest с equity_curve + комиссии
def run_backtest(df, commission_rate=0.00025):  # 0.04% round-trip (Binance futures taker ~0.02% per side)
    tracker = MarketStructureTracker()
    trades = []
    current_trade = None
    equity_curve = []  # список (timestamp, cumulative_pnl)
    current_equity = 0.0
    position_size = 1.0  # 1 BTC (можно менять; PnL в USDT)

    for idx, row in df.iterrows():
        i = df.index.get_loc(idx)  # numeric position
        high = row['high']
        low = row['low']
        close = row['close']

        tracker.add_candle(i, high, low)

        # Выход по CHoCH
        if tracker.structure and tracker.structure[-1]['index'] == i and tracker.structure[-1]['label'] == 'CHoCH':
            if current_trade:
                exit_price = close
                gross_profit = (exit_price - current_trade['entry']) * current_trade.get('direction', 1) * position_size
                commission = commission_rate * (current_trade['entry'] + exit_price) * position_size
                net_profit = gross_profit - commission
                current_equity += net_profit
                trades.append({
                    'entry_idx': current_trade['entry_idx'],
                    'entry_time': df.index[current_trade['entry_idx']],
                    'entry_price': current_trade['entry'],
                    'exit_idx': i,
                    'exit_time': idx,
                    'exit_price': exit_price,
                    'gross_profit': round(gross_profit, 2),
                    'commission': round(commission, 2),
                    'profit': round(net_profit, 2),
                    'type': current_trade['type']
                })
                equity_curve.append((idx, current_equity))
                current_trade = None

        # Трейлинг-стоп управление
        if current_trade:
            if current_trade['type'] == 'long':
                current_trade['max_price'] = max(current_trade['max_price'], high)

                if not current_trade['trailing_active'] and close > current_trade['impulse_high']:
                    current_trade['trailing_active'] = True

                current_stop = current_trade['stop']
                if current_trade['trailing_active']:
                    trail_stop = current_trade['max_price'] - current_trade['trailing_distance']
                    current_stop = max(current_stop, trail_stop)

                if low <= current_stop:
                    exit_price = current_stop
                    gross_profit = (exit_price - current_trade['entry']) * position_size
                    commission = commission_rate * (current_trade['entry'] + exit_price) * position_size
                    net_profit = gross_profit - commission
                    current_equity += net_profit
                    trades.append({
                        'entry_idx': current_trade['entry_idx'],
                        'entry_time': df.index[current_trade['entry_idx']],
                        'entry_price': current_trade['entry'],
                        'exit_idx': i,
                        'exit_time': idx,
                        'exit_price': exit_price,
                        'gross_profit': round(gross_profit, 2),
                        'commission': round(commission, 2),
                        'profit': round(net_profit, 2),
                        'type': 'long'
                    })
                    equity_curve.append((idx, current_equity))
                    current_trade = None

            elif current_trade['type'] == 'short':
                current_trade['min_price'] = min(current_trade['min_price'], low)

                if not current_trade['trailing_active'] and close < current_trade['impulse_low']:
                    current_trade['trailing_active'] = True

                current_stop = current_trade['stop']
                if current_trade['trailing_active']:
                    trail_stop = current_trade['min_price'] + current_trade['trailing_distance']
                    current_stop = min(current_stop, trail_stop)

                if high >= current_stop:
                    exit_price = current_stop
                    gross_profit = (current_trade['entry'] - exit_price) * position_size
                    commission = commission_rate * (current_trade['entry'] + exit_price) * position_size
                    net_profit = gross_profit - commission
                    current_equity += net_profit
                    trades.append({
                        'entry_idx': current_trade['entry_idx'],
                        'entry_time': df.index[current_trade['entry_idx']],
                        'entry_price': current_trade['entry'],
                        'exit_idx': i,
                        'exit_time': idx,
                        'exit_price': exit_price,
                        'gross_profit': round(gross_profit, 2),
                        'commission': round(commission, 2),
                        'profit': round(net_profit, 2),
                        'type': 'short'
                    })
                    equity_curve.append((idx, current_equity))
                    current_trade = None

        # Вход
        if not current_trade and tracker.current_impulse:
            impulse = tracker.current_impulse
            if impulse['high_price'] is None or impulse['low_price'] is None:
                continue
            # Только после формирования импульса (i > индексы экстремумов)
            if i <= impulse.get('high_index', -1) or i <= impulse.get('low_index', -1):
                continue

            if impulse['type'] == 'bull':
                fib_05 = impulse['high_price'] - 0.5 * (impulse['high_price'] - impulse['low_price'])
                if low <= fib_05 and close > fib_05:
                    entry_price = close
                    sl = impulse['low_price']
                    trailing_distance = 0.5 * (impulse['high_price'] - entry_price)
                    current_trade = {
                        'type': 'long',
                        'entry': entry_price,
                        'entry_idx': i,
                        'stop': sl,
                        'trailing_distance': trailing_distance,
                        'max_price': high,
                        'impulse_high': impulse['high_price'],
                        'trailing_active': False,
                        'direction': 1
                    }

            elif impulse['type'] == 'bear':
                fib_05 = impulse['low_price'] + 0.5 * (impulse['high_price'] - impulse['low_price'])
                if high >= fib_05 and close < fib_05:
                    entry_price = close
                    sl = impulse['high_price']
                    trailing_distance = 0.5 * (entry_price - impulse['low_price'])
                    current_trade = {
                        'type': 'short',
                        'entry': entry_price,
                        'entry_idx': i,
                        'stop': sl,
                        'trailing_distance': trailing_distance,
                        'min_price': low,
                        'impulse_low': impulse['low_price'],
                        'trailing_active': False,
                        'direction': -1
                    }

    # Закрытие открытой позиции в конце (по последней close)
    if current_trade:
        last_close = df['close'].iloc[-1]
        last_time = df.index[-1]
        gross_profit = (last_close - current_trade['entry']) * current_trade.get('direction', 1) * position_size
        commission = commission_rate * (current_trade['entry'] + last_close) * position_size
        net_profit = gross_profit - commission
        current_equity += net_profit
        trades.append({
            'entry_idx': current_trade['entry_idx'],
            'entry_time': df.index[current_trade['entry_idx']],
            'entry_price': current_trade['entry'],
            'exit_idx': len(df) - 1,
            'exit_time': last_time,
            'exit_price': last_close,
            'gross_profit': round(gross_profit, 2),
            'commission': round(commission, 2),
            'profit': round(net_profit, 2),
            'type': current_trade['type']
        })
        equity_curve.append((last_time, current_equity))

    stats = calculate_trade_stats(trades)
    return trades, stats, equity_curve

# Функция для отрисовки equity curve (обновлена с net PnL)
def plot_equity_curve(equity_curve, title="Equity Curve (Cumulative Net PnL after Fees)"):
    if not equity_curve:
        print("Нет сделок — equity curve пустая")
        return

    dates = [t for t, e in equity_curve]
    equities = [e for t, e in equity_curve]

    plt.figure(figsize=(14, 7))
    plt.plot(dates, equities, color='blue', linewidth=2, label='Cumulative Net PnL')
    plt.fill_between(dates, equities, color='blue', alpha=0.1)

    plt.title(title)
    plt.xlabel('Время')
    plt.ylabel('Накопленный Net PnL (USDT)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
# Запуск и красивый вывод
# Запуск (исправил таймфрейм в print на 1h)
df = fetch_klines_paged("ETHUSDT", "1h", 50000)  # 3000 часовых свечей ~ 4–5 месяцев
trades, stats, equity_curve = run_backtest(df)

print("\n" + "="*60)
print("BACKTEST RESULTS (Market Structure + 0.5 Fib Pullback)")
print("="*60)
print(f"Символ:          ETHUSDT")
print(f"Таймфрейм:       1h")   # исправлено
print(f"Всего баров:     {len(df)}")
print(f"Всего трейдов:   {stats['total_trades']}")
print(f"Общий PnL:       {stats['total_pnl']:+.2f} USDT")
print(f"Winrate:         {stats['winrate']}%")
print(f"Avg Win:         {stats['avg_win']:+.2f}")
print(f"Avg Loss:        {stats['avg_loss']:+.2f}")
print(f"Profit Factor:   {stats['profit_factor']}")
print(f"Max Win:         {stats['max_win']:+.2f}")
print(f"Max Loss:        {stats['max_loss']:+.2f}")
print("="*60)

# Отрисовка equity curve
plot_equity_curve(equity_curve, title=f"Equity Curve — BTCUSDT 1h (3000 bars)")

# Чтобы визуализировать, можно добавить в plot_structure точки входа/выхода
# Например:
# ax.scatter(df.index[trade['entry_idx']], trade['entry_price'], color='blue' if trade['type']=='long' else 'red', marker='^' if trade['type']=='long' else 'v', s=150)