import time
import hmac
import hashlib
import requests
import json
import threading
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BingxClient:
    BASE_URL = "https://open-api.bingx.com"  # mainnet (testnet: https://open-api-vst.bingx.com ? check docs)
    APIURL = BASE_URL  # alias

    def __init__(self, api_key: str, api_secret: str, symbol: str = "ETHUSDT"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = self._to_bingx_symbol(symbol)
        self.time_offset = self.get_server_time_offset()

    def _to_bingx_symbol(self, symbol: str) -> str:
        return symbol.replace("USDT", "-USDT")  # ETHUSDT -> ETH-USDT

    def _sign(self, query: str) -> str:
        return hmac.new(self.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()

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
        url = f"{self.APIURL}{path}?{query}&signature={sign}"
        headers = {'X-BX-APIKEY': self.api_key}
        response = requests.request(method, url, headers=headers)
        try:
            return response.json()
        except Exception as e:
            logging.error(f"JSON parse error: {e}, response: {response.text}")
            return None

    def _public_request(self, path: str, params=None):
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

    def get_mark_price(self, symbol=None):
        path = "/openApi/swap/v2/quote/premiumIndex"
        params = {'symbol': symbol or self.symbol}
        data = self._public_request(path, params)
        if data.get('code') == 0 and 'data' in data:
            if isinstance(data['data'], list) and data['data']:
                return float(data['data'][0].get('markPrice'))
            elif isinstance(data['data'], dict):
                return float(data['data'].get('markPrice'))
        return None

    def get_klines(self, interval="1h", limit=1000, start_time=None, end_time=None):
        path = "/openApi/swap/v2/quote/klines"
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        data = self._public_request(path, params)
        if data.get('code') == 0:
            return data.get('data', [])
        logging.error(f"Klines error: {data}")
        return []

    def place_market_order(self, side: str, qty: float, stop: float = None, tp: float = None):
        side_param = "BUY" if side == "long" else "SELL"
        positionSide = 'LONG' if side == "long" else 'SHORT'
        params = {
            "symbol": self.symbol,
            "side": side_param,
            "positionSide": positionSide,
            "type": "MARKET",
            "quantity": qty,
            "recvWindow": 5000,
        }
        if stop is not None:
            params["stopLoss"] = json.dumps({
                "type": "STOP_MARKET",
                "stopPrice": stop,
                "workingType": "MARK_PRICE"
            })
        if tp is not None:
            params["takeProfit"] = json.dumps({
                "type": "TAKE_PROFIT_MARKET",
                "stopPrice": tp,
                "workingType": "MARK_PRICE"
            })
        return self.send_request("POST", "/openApi/swap/v2/trade/order", params)

    def close_position(self):
        pos = self.get_position()
        if not pos or pos.get('code') != 0 or not pos.get('data'):
            logging.info("No open position")
            return
        for p in pos['data']:
            amt = float(p['positionAmt'])
            if amt != 0:
                pos_side = p['positionSide']
                side = "SELL" if amt > 0 else "BUY"
                qty = abs(amt)
                params = {
                    "symbol": self.symbol,
                    "side": side,
                    "positionSide": pos_side,
                    "type": "MARKET",
                    "quantity": qty,
                    "recvWindow": 5000,
                }
                resp = self.send_request("POST", "/openApi/swap/v2/trade/order", params)
                logging.info(f"Closed position: {resp}")
                return resp
        logging.info("No position to close")

    def get_position(self):
        path = "/openApi/swap/v2/trade/position"
        params = {"symbol": self.symbol}
        return self.send_request("GET", path, params)