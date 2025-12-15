import os
import logging
import time
import traceback
from decimal import Decimal
from datetime import timedelta
from typing import Optional
from uuid import uuid4

import pandas as pd
from dotenv import load_dotenv

from t_tech.invest import (
    Client,
    OrderDirection,
    OrderType,
    InstrumentIdType,
    OrderExecutionReportStatus,
    CandleInterval,
)
from t_tech.invest.services import Services
from t_tech.invest.utils import (
    quotation_to_decimal,
    money_to_decimal,
    now,
)

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

load_dotenv()

class Config:
    """Конфигурация торгового бота"""
    TOKEN = os.environ.get("TINKOFF_INVEST_TOKEN")
    
    # --- НАСТРОЙКИ ИНСТРУМЕНТА (RTS Mini) ---
    TICKER = "RMZ5"
    CLASS_CODE = "SPBFUT" 
    
    # --- СТРАТЕГИЯ ---
    EMA_SHORT = 30
    EMA_LONG = 260
    
    # Скачиваем 1 минуту, торгуем на 15 минутах
    DOWNLOAD_TIMEFRAME = CandleInterval.CANDLE_INTERVAL_1_MIN  # Изменено на 1 минуту
    TRADE_TIMEFRAME_MINUTES = 15
    
    @classmethod
    def validate(cls):
        if not cls.TOKEN:
            raise ValueError("TINKOFF_INVEST_TOKEN не установлен в .env")

class TradingBot:
    def __init__(self):
        Config.validate()
        self.client: Optional[Services] = None
        self.account_id: Optional[str] = None
        self.instrument = None
        
    def run(self):
        """Основной цикл запуска в реальном времени"""
        logger.info(f"Запуск робота по {Config.TICKER} (Data: 5min -> Trade: 15min, EMA: {Config.EMA_SHORT}/{Config.EMA_LONG})")
        
        while True:
            try:
                with Client(Config.TOKEN) as client:
                    self.client = client
                    if not self.account_id:
                        self._setup_account()
                    if not self.instrument:
                        self._setup_instrument()
                    
                    logger.info("--- Анализ рынка ---")
                    
                    signal = self._analyze_market()
                    has_position = self._has_open_position()
                    
                    if signal == 'BUY' and not has_position:
                        logger.info("Сигнал BUY. Входим в позицию.")
                        self._execute_order(OrderDirection.ORDER_DIRECTION_BUY)
                        
                    elif signal == 'SELL' and has_position:
                        logger.info("Сигнал SELL. Закрываем позицию.")
                        self._execute_order(OrderDirection.ORDER_DIRECTION_SELL)
                        
                    else:
                        logger.info(f"Сигнал {signal}. Позиция: {'Есть' if has_position else 'Нет'}. Ждем.")

            except Exception as e:
                logger.error(f"Ошибка в цикле: {e}")
                logger.debug(traceback.format_exc())
            
            # Пауза 60 секунд
            time.sleep(60)

    def _setup_account(self):
        accounts = self.client.users.get_accounts()
        self.account_id = accounts.accounts[0].id
        logger.info(f"Аккаунт ID: {self.account_id}")

    def _setup_instrument(self):
        try:
            item = self.client.instruments.get_instrument_by(
                id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER,
                id=Config.TICKER,
                class_code=Config.CLASS_CODE,
            ).instrument
            self.instrument = item
            logger.info(f"Инструмент: {item.name} (FIGI: {item.figi})")
        except Exception as e:
            logger.critical(f"Не удалось найти инструмент {Config.TICKER}. Ошибка: {e}")
            raise e

    def _has_open_position(self) -> bool:
        """Проверка наличия позиций"""
        positions = self.client.operations.get_positions(account_id=self.account_id)
        
        for p in positions.securities:
            if p.figi == self.instrument.figi and p.balance != 0:
                return True
        if hasattr(positions, 'futures'):
             for f in positions.futures:
                if f.figi == self.instrument.figi and f.balance != 0:
                    return True
        return False

    def _get_candles_dataframe(self, days_back: int = 50) -> pd.DataFrame:
        """Загрузка 5-минутных свечей и ресемплинг в 15-минутные"""
        candles = self.client.get_all_candles(
            instrument_id=self.instrument.uid,
            from_=now() - timedelta(days=days_back),
            to=now(),
            interval=Config.DOWNLOAD_TIMEFRAME, # Скачиваем 5 мин
        )
        
        data = []
        for c in candles:
            data.append({
                'time': c.time,
                'open': float(quotation_to_decimal(c.open)),
                'high': float(quotation_to_decimal(c.high)),
                'low': float(quotation_to_decimal(c.low)),
                'close': float(quotation_to_decimal(c.close)),
                'volume': c.volume
            })
            
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'])
        df['time'] = df['time'].dt.tz_convert('Europe/Moscow').dt.tz_localize(None)
        df.set_index('time', inplace=True)

        # РЕСЕМПЛИНГ: Превращаем 5-минутки в 15-минутки
        # Правила агрегации: Open - первый, High - макс, Low - мин, Close - последний, Volume - сумма
        logic = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # '15T' означает 15 минут. label='left' означает, что свеча 10:00-10:15 будет называться 10:00
        df_resampled = df.resample(f'{Config.TRADE_TIMEFRAME_MINUTES}min', label='left', closed='left').agg(logic)
        
        # Удаляем пустые интервалы (если были пропуски торгов)
        df_resampled.dropna(inplace=True)
        
        return df_resampled

    def _analyze_market(self) -> str:
        # Загружаем данные (они уже будут 15-минутными после ресемплинга)
        df = self._get_candles_dataframe(days_back=30)
        
        if len(df) < Config.EMA_LONG:
            logger.warning(f"Недостаточно данных: {len(df)}")
            return 'HOLD'

        df['ema_short'] = df['close'].ewm(span=Config.EMA_SHORT, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=Config.EMA_LONG, adjust=False).mean()
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        logger.info(f"Close: {current['close']} | EMA{Config.EMA_SHORT}: {current['ema_short']:.2f} | EMA{Config.EMA_LONG}: {current['ema_long']:.2f}")
        
        if prev['ema_short'] <= prev['ema_long'] and current['ema_short'] > current['ema_long']:
            return 'BUY'
            
        if prev['ema_short'] >= prev['ema_long'] and current['ema_short'] < current['ema_long']:
            return 'SELL'
            
        return 'HOLD'

    def _execute_order(self, direction):
        action = "Покупка" if direction == OrderDirection.ORDER_DIRECTION_BUY else "Продажа"
        logger.info(f"{action} 1 контракта {Config.TICKER}...")
        try:
            response = self.client.orders.post_order(
                instrument_id=self.instrument.uid,
                quantity=1,
                account_id=self.account_id,
                order_type=OrderType.ORDER_TYPE_MARKET,
                direction=direction,
                order_id=str(uuid4())
            )
            price = money_to_decimal(response.executed_order_price)
            if price == 0:
                price = money_to_decimal(response.initial_security_price)
            logger.info(f"Цена исполнения (примерно): {price}")
        except Exception as e:
            logger.error(f"Ошибка при исполнении ордера: {e}")

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()