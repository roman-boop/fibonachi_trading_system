# realtime_bingx_trading.py
# Полный скрипт для реал-тайм торговли на BingX (ETH-USDT perpetual)
# Количество: 0.05 ETH
# Таймфрейм: 1h
# Логика: Market Structure + 0.5 Fib pullback + trailing stop

import time
import hmac
import hashlib
import requests
import json
import threading
import logging
import pandas as pd
from datetime import datetime

# ────────────────────────────────────────────────
# Настройка логирования
# ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-5s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ────────────────────────────────────────────────
# Класс BingxClient (можно вынести в отдельный файл bingx_client.py)
# ────────────────────────────────────────────────
class BingxClient:
    BASE_URL = "https://open-api-vst.bingx.com"  # mainnet
    # BASE_URL = "https://open-api-vst.bingx.com"  # testnet (если доступен)

    def __init__(self, api_key: str, api_secret: str, symbol: str = "ETHUSDT"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol.replace("USDT", "-USDT")  # ETHUSDT → ETH-USDT
        self.time_offset = self.get_server_time_offset()

    def _sign(self, query: str) -> str:
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def parseParam(self, paramsMap: dict) -> str:
        sortedKeys = sorted(paramsMap)
        paramsStr = "&".join(f"{k}={paramsMap[k]}" for k in sortedKeys)
        timestamp = str(int(time.time() * 1000) + self.time_offset)
        if paramsStr:
            return f"{paramsStr}&timestamp={timestamp}"
        return f"timestamp={timestamp}"

    def send_request(self, method: str, path: str, params: dict = None):
        if params is None:
            params = {}
        query = self.parseParam(params)
        sign = self._sign(query)
        url = f"{self.BASE_URL}{path}?{query}&signature={sign}"
        headers = {'X-BX-APIKEY': self.api_key}
        response = requests.request(method, url, headers=headers)
        try:
            return response.json()
        except Exception as e:
            logging.error(f"JSON parse error: {e} | Response: {response.text}")
            return None

    def _public_request(self, path: str, params=None):
        if params is None:
            params = {}
        url = f"{self.BASE_URL}{path}"
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_server_time_offset(self):
        url = f"{self.BASE_URL}/openApi/swap/v2/server/time"
        r = requests.get(url)
        data = r.json()
        if data.get("code") == 0:
            server_time = int(data["data"]["serverTime"])
            local_time = int(time.time() * 1000)
            return server_time - local_time
        return 0

    def get_mark_price(self):
        path = "/openApi/swap/v2/quote/premiumIndex"
        params = {'symbol': self.symbol}
        data = self._public_request(path, params)
        if data.get('code') == 0 and data.get('data'):
            if isinstance(data['data'], list) and data['data']:
                return float(data['data'][0].get('markPrice'))
            elif isinstance(data['data'], dict):
                return float(data['data'].get('markPrice'))
        return None

    def get_klines(self, interval="1h", limit=1000):
        path = "/openApi/swap/v2/quote/klines"
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": str(limit),
            'timestamp': int(time.time() * 1000)
        }
        data = self._public_request(path, params)
        if data.get('code') == 0:
            return data.get('data', [])
        logging.error(f"Klines error: {data.get('msg')}")
        return []

    def place_market_order(self, side: str, qty: float, stop: float = None):
        side_param = "BUY" if side == "long" else "SELL"
        positionSide = 'LONG' if side == "long" else 'SHORT'
        params = {
            "symbol": self.symbol,
            "side": side_param,
            "positionSide": positionSide,
            "type": "MARKET",
            "quantity": str(qty),
            "recvWindow": "5000",
            'timestamp': int(time.time() * 1000)
        }
        if stop is not None:
            params["stopLoss"] = json.dumps({
                "type": "STOP_MARKET",
                "stopPrice": str(stop),
                "workingType": "MARK_PRICE"
            })
        return self.send_request("POST", "/openApi/swap/v2/trade/order", params)

    def close_position(self):
        path = "/openApi/swap/v2/trade/position"
        pos_data = self.send_request("GET", path, {"symbol": self.symbol})
        if not pos_data or pos_data.get('code') != 0 or not pos_data.get('data'):
            logging.info("Нет открытых позиций")
            return None

        for pos in pos_data['data']:
            amt = float(pos['positionAmt'])
            if amt == 0:
                continue
            side = "SELL" if amt > 0 else "BUY"
            qty = abs(amt)
            params = {
                "symbol": self.symbol,
                "side": side,
                "positionSide": pos['positionSide'],
                "type": "MARKET",
                "quantity": str(qty),
                "recvWindow": "5000",
                'timestamp': int(time.time() * 1000)
            }
            resp = self.send_request("POST", "/openApi/swap/v2/trade/order", params)
            logging.info(f"Закрыта позиция: {resp}")
            return resp
        return None


# ────────────────────────────────────────────────
# Market Structure Tracker (из предыдущего backtest)
# ────────────────────────────────────────────────
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

# ────────────────────────────────────────────────
# Настройки
# ────────────────────────────────────────────────
API_KEY = ""
API_SECRET = ""

SYMBOL = "ETHUSDT"
QTY = 1
INTERVAL = "1h"
CANDLE_POLL_SEC = 15      # проверка новой свечи
TRAIL_POLL_SEC = 5        # проверка trailing stop

client = BingxClient(API_KEY, API_SECRET, SYMBOL)
tracker = MarketStructureTracker()
df = pd.DataFrame()  # будет заполняться историей + новыми свечами
current_trade = None

def load_initial_history(limit=1400):
    global df
    klines = client.get_klines(interval=INTERVAL, limit=limit)
    if not klines:
        logging.error("Не удалось загрузить историю")
        return

    data = []
    for k in klines:
        # k — это словарь
        data.append({
            'timestamp': pd.to_datetime(int(k['time']), unit='ms'),
            'open': float(k['open']),
            'high': float(k['high']),
            'low': float(k['low']),
            'close': float(k['close']),
            # 'volume': float(k['volume']),  # если нужно — раскомментируй
        })

    df = pd.DataFrame(data).set_index('timestamp')
    df.sort_index(inplace=True)

    # Инициализируем tracker историей
    for i, row in df.iterrows():
        tracker.add_candle(df.index.get_loc(i), row['high'], row['low'])

    logging.info(f"Загружено {len(df)} исторических свечей 1h")


def check_new_candle():
    global df, current_trade
    klines = client.get_klines(interval=INTERVAL, limit=2)
    if not klines or len(klines) < 2:
        return

    latest = klines[-2]
    ts = pd.to_datetime(int(latest['time']), unit='ms')

    if ts not in df.index:
        new_row = pd.Series({
                'open': float(latest['open']),
                'high': float(latest['high']),
                'low': float(latest['low']),
                'close': float(latest['close']),
        }, name=ts)

        df = pd.concat([df, new_row.to_frame().T])
        i = len(df) - 1
        tracker.add_candle(i, new_row['high'], new_row['low'])

        logging.info(f"Новая свеча: {ts} | Close: {new_row['close']}")




        # Проверка CHoCH → закрытие позиции
        if tracker.structure and tracker.structure[-1]['index'] == i and tracker.structure[-1]['label'] == 'CHoCH':
            if current_trade:
                client.close_position()
                logging.info("CHoCH → позиция закрыта")
                current_trade = None

        # Проверка входа
        if not current_trade and tracker.current_impulse:
            impulse = tracker.current_impulse
            if 'high_price' not in impulse or 'low_price' not in impulse:
                return
            if i <= impulse.get('high_index', -1) or i <= impulse.get('low_index', -1):
                return

            mark = client.get_mark_price()
            if mark is None:
                return

            if impulse['type'] == 'bull':
                fib_05 = impulse['high_price'] - 0.5 * (impulse['high_price'] - impulse['low_price'])
                if new_row['low'] <= fib_05 < new_row['close']:
                    sl = impulse['low_price']
                    resp = client.place_market_order("long", QTY, stop=sl)
                    if resp and resp.get('code') == 0:
                        current_trade = {
                            'type': 'long',
                            'entry': mark,
                            'stop': sl,
                            'trailing_distance': 0.5 * (impulse['high_price'] - mark),
                            'max_price': new_row['high'],
                            'impulse_high': impulse['high_price'],
                            'trailing_active': False,
                            'direction': 1
                        }
                        logging.info(f"LONG открыт @ {mark:.2f}, SL {sl:.2f}")

            elif impulse['type'] == 'bear':
                fib_05 = impulse['low_price'] + 0.5 * (impulse['high_price'] - impulse['low_price'])
                if new_row['high'] >= fib_05 > new_row['close']:
                    sl = impulse['high_price']
                    resp = client.place_market_order("short", QTY, stop=sl)
                    if resp and resp.get('code') == 0:
                        current_trade = {
                            'type': 'short',
                            'entry': mark,
                            'stop': sl,
                            'trailing_distance': 0.5 * (mark - impulse['low_price']),
                            'min_price': new_row['low'],
                            'impulse_low': impulse['low_price'],
                            'trailing_active': False,
                            'direction': -1
                        }
                        logging.info(f"SHORT открыт @ {mark:.2f}, SL {sl:.2f}")


def monitor_trailing():
    global current_trade
    while True:
        if current_trade:
            mark = client.get_mark_price()
            if mark is None:
                time.sleep(TRAIL_POLL_SEC)
                continue

            if current_trade['type'] == 'long':
                current_trade['max_price'] = max(current_trade['max_price'], mark)

                if not current_trade['trailing_active'] and mark > current_trade['impulse_high']:
                    current_trade['trailing_active'] = True
                    logging.info("Trailing активирован (long)")

                if current_trade['trailing_active']:
                    trail_stop = current_trade['max_price'] - current_trade['trailing_distance']
                    current_trade['stop'] = max(current_trade['stop'], trail_stop)

                    if mark <= current_trade['stop']:
                        client.close_position()
                        logging.info(f"Trailing hit LONG → закрытие @ {mark:.2f}")
                        current_trade = None

            else:  # short
                current_trade['min_price'] = min(current_trade['min_price'], mark)

                if not current_trade['trailing_active'] and mark < current_trade['impulse_low']:
                    current_trade['trailing_active'] = True
                    logging.info("Trailing активирован (short)")

                if current_trade['trailing_active']:
                    trail_stop = current_trade['min_price'] + current_trade['trailing_distance']
                    current_trade['stop'] = min(current_trade['stop'], trail_stop)

                    if mark >= current_trade['stop']:
                        client.close_position()
                        logging.info(f"Trailing hit SHORT → закрытие @ {mark:.2f}")
                        current_trade = None

        time.sleep(TRAIL_POLL_SEC)


if __name__ == "__main__":
    logging.info("Запуск реал-тайм скрипта BingX (ETH-USDT 1h)")
    load_initial_history()

    # Запускаем мониторинг trailing в отдельном потоке
    threading.Thread(target=monitor_trailing, daemon=True).start()

    while True:
        try:
            check_new_candle()
            time.sleep(CANDLE_POLL_SEC)
        except KeyboardInterrupt:
            logging.info("Остановка по Ctrl+C")
            if current_trade:
                client.close_position()
            break
        except Exception as e:
            logging.error(f"Ошибка в главном цикле: {e}")
            time.sleep(30)