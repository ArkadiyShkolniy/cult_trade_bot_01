import os
import logging
import traceback
from decimal import Decimal
from datetime import timedelta
from typing import Optional, List
from uuid import uuid4

import pandas as pd
from dotenv import load_dotenv

# FIXED: Replaced tinkoff.invest with t_tech.invest
from t_tech.invest import (
    Client,
    OrderDirection,
    OrderType,
    InstrumentIdType,
    OrderExecutionReportStatus,
    StopOrderDirection,
    StopOrderType,
    StopOrderExpirationType,
    StopOrderStatusOption,
    TakeProfitType,
    ExchangeOrderType,
    PostStopOrderRequestTrailingData,
    CandleInterval,
    HistoricCandle,
)
from t_tech.invest.services import Services
from t_tech.invest.schemas import TrailingValueType
from t_tech.invest.utils import (
    quotation_to_decimal,
    decimal_to_quotation,
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
    TICKER = "SBER"
    CLASS_CODE = "TQBR"
    
    # Стратегия
    STRATEGY_TYPE = "EMA"
    EMA_SHORT = 30
    EMA_LONG = 195
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BB_PERIOD = 20
    BB_STD = 2.0
    STOCH_K = 14
    STOCH_D = 3
    PATTERN_TYPE = "candlestick"
    PATTERN_NAMES = [] # Empty list means no patterns selected
    
    TIMEFRAME = CandleInterval.CANDLE_INTERVAL_30_MIN
    
    # Риск-менеджмент
    TRAILING_INDENT = 0.003  # 0.3%
    TRAILING_SPREAD = 0.5    # 0.5%
    TAKE_PROFIT = 0.01       # 1.0%
    
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
        """Основной цикл запуска"""
        try:
            with Client(Config.TOKEN) as client:
                self.client = client
                self._setup_account()
                self._setup_instrument()
                
                # 1. Анализ рынка
                signal = self._analyze_market()
                
                # 2. Исполнение сигналов
                if signal == 'BUY':
                    self._execute_buy_sequence()
                elif signal == 'SELL':
                    logger.info("Сигнал SELL. Автоматическая продажа не настроена (только закрытие по стопам).")
                else:
                    logger.info("Сигнал HOLD. Действий не требуется.")
                    
        except Exception as e:
            logger.error(f"Критическая ошибка: {e}")
            logger.debug(traceback.format_exc())

    def _setup_account(self):
        """Получение данных аккаунта"""
        accounts = self.client.users.get_accounts()
        self.account_id = accounts.accounts[0].id
        logger.info(f"Аккаунт ID: {self.account_id}")

    def _setup_instrument(self):
        """Получение данных инструмента"""
        item = self.client.instruments.get_instrument_by(
            id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER,
            id=Config.TICKER,
            class_code=Config.CLASS_CODE,
        ).instrument
        
        self.instrument = item
        logger.info(f"Инструмент: {item.name} (FIGI: {item.figi})")

    def _get_candles_dataframe(self, days_back: int = 100) -> pd.DataFrame:
        """Загрузка свечей и конвертация в DataFrame"""
        logger.info(f"Загрузка свечей за {days_back} дней...")
        
        candles = self.client.get_all_candles(
            instrument_id=self.instrument.uid,
            from_=now() - timedelta(days=days_back),
            to=now(),
            interval=Config.TIMEFRAME,
        )
        
        data = []
        for c in candles:
            data.append({
                'time': c.time,
                'close': float(quotation_to_decimal(c.close)),
                'open': float(quotation_to_decimal(c.open)),
                'high': float(quotation_to_decimal(c.high)),
                'low': float(quotation_to_decimal(c.low)),
                'volume': c.volume
            })
            
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df

    def _analyze_market(self) -> str:
        """Анализ рынка и генерация сигнала"""
        # Определяем необходимый период истории в зависимости от стратегии
        required_candles = 200 # Default
        
        if Config.STRATEGY_TYPE == "EMA":
            required_candles = Config.EMA_LONG + 10
        elif Config.STRATEGY_TYPE == "RSI":
            required_candles = Config.RSI_PERIOD * 2 + 10
        elif Config.STRATEGY_TYPE == "MACD":
            required_candles = Config.MACD_SLOW + 20
        elif Config.STRATEGY_TYPE == "BB":
            required_candles = Config.BB_PERIOD + 10
        elif Config.STRATEGY_TYPE == "STOCH":
            required_candles = Config.STOCH_K + 10
        elif Config.STRATEGY_TYPE == "PATTERN":
            required_candles = 50 # Паттерны обычно требуют немного свечей
            
        # Загружаем с запасом для расчета индикаторов
        # Конвертируем в дни примерно (для _get_candles_dataframe)
        # Упрощенно считаем 1 день = 24 часа. Для минутных свечей 200 свечей это всего 3 часа.
        # Берем 60 дней для надежности как и раньше, но можно оптимизировать
        df = self._get_candles_dataframe(days_back=60)
        
        if len(df) < required_candles:
            logger.warning(f"Недостаточно данных: {len(df)} < {required_candles}")
            return 'HOLD'

        # Инициализируем оптимизатор для использования его методов расчета
        from cult_test.strategy_optimizer import StrategyOptimizer
        optimizer = StrategyOptimizer(token=Config.TOKEN)
        
        # Расчет сигналов в зависимости от выбранной стратегии
        if Config.STRATEGY_TYPE == "EMA":
            df = optimizer.calculate_ema_signals(df, Config.EMA_SHORT, Config.EMA_LONG)
        elif Config.STRATEGY_TYPE == "RSI":
            df = optimizer.calculate_rsi_signals(df, Config.RSI_PERIOD, Config.RSI_OVERSOLD, Config.RSI_OVERBOUGHT)
        elif Config.STRATEGY_TYPE == "MACD":
            df = optimizer.calculate_macd_signals(df, Config.MACD_FAST, Config.MACD_SLOW, Config.MACD_SIGNAL)
        elif Config.STRATEGY_TYPE == "BB":
            df = optimizer.calculate_bb_signals(df, Config.BB_PERIOD, Config.BB_STD)
        elif Config.STRATEGY_TYPE == "STOCH":
            df = optimizer.calculate_stoch_signals(df, Config.STOCH_K, Config.STOCH_D)
        elif Config.STRATEGY_TYPE == "PATTERN":
            df = optimizer.calculate_pattern_signals(df, Config.PATTERN_TYPE, Config.PATTERN_NAMES)
        else:
            logger.warning(f"Неизвестный тип стратегии: {Config.STRATEGY_TYPE}, используется HOLD")
            return 'HOLD'
            
        current = df.iloc[-1]
        
        # Логирование текущих значений
        if Config.STRATEGY_TYPE == "EMA":
            logger.info(f"EMA({Config.EMA_SHORT}): {current.get('ema_short', 0):.2f}, EMA({Config.EMA_LONG}): {current.get('ema_long', 0):.2f}")
        elif Config.STRATEGY_TYPE == "RSI":
            logger.info(f"RSI({Config.RSI_PERIOD}): {current.get('rsi', 0):.2f}")
        
        if current['signal'] == 1:
            logger.info(f"✅ СИГНАЛ BUY по стратегии {Config.STRATEGY_TYPE}")
            return 'BUY'
        elif current['signal'] == -1:
            logger.info(f"✅ СИГНАЛ SELL по стратегии {Config.STRATEGY_TYPE}")
            return 'SELL'
            
        return 'HOLD'

    def _execute_buy_sequence(self):
        """Выполнение последовательности покупки"""
        
        # 0. Отменяем старые заявки перед входом
        self._cancel_all_orders()
        
        # 1. Рыночная покупка
        logger.info(f"Покупка 1 лота {Config.TICKER}...")
        order_id = str(uuid4())
        response = self.client.orders.post_order(
            instrument_id=self.instrument.uid,
            quantity=1,
            account_id=self.account_id,
            order_type=OrderType.ORDER_TYPE_MARKET,
            direction=OrderDirection.ORDER_DIRECTION_BUY,
            order_id=order_id
        )
        
        status = response.execution_report_status
        logger.info(f"Статус ордера: {OrderExecutionReportStatus(status).name}")
        
        if status not in [
            OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_FILL,
            OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_NEW,
            OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_PARTIALLYFILL
        ]:
            logger.error("Ордер на покупку не исполнен. Стопы не выставляем.")
            return

        executed_price = money_to_decimal(response.executed_order_price)
        logger.info(f"Цена исполнения: {executed_price}")
        
        # 2. Выставление защитных ордеров
        self._place_stop_loss_and_take_profit(executed_price)

    def _place_stop_loss_and_take_profit(self, entry_price: Decimal):
        """Расчет и выставление Trailing Stop и Take Profit"""
        min_step = quotation_to_decimal(self.instrument.min_price_increment)
        
        # --- Trailing Stop ---
        # Stop Price = Цена входа - отступ
        stop_price = entry_price * Decimal(1 - Config.TRAILING_INDENT)
        # Округление до шага цены
        stop_price = round(stop_price / min_step) * min_step
        
        # Цена активации (триггер) чуть выше стопа
        price_buffer = entry_price * Decimal(1.001) 
        price_buffer = round(price_buffer / min_step) * min_step

        try:
            logger.info("Выставляем Trailing Stop...")
            self.client.stop_orders.post_stop_order(
                instrument_id=self.instrument.uid,
                quantity=1,
                price=decimal_to_quotation(price_buffer),
                stop_price=decimal_to_quotation(stop_price),
                direction=StopOrderDirection.STOP_ORDER_DIRECTION_SELL,
                account_id=self.account_id,
                stop_order_type=StopOrderType.STOP_ORDER_TYPE_TAKE_PROFIT, # Trailing в API T-Invest это тип TakeProfit
                expiration_type=StopOrderExpirationType.STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_CANCEL,
                exchange_order_type=ExchangeOrderType.EXCHANGE_ORDER_TYPE_LIMIT,
                take_profit_type=TakeProfitType.TAKE_PROFIT_TYPE_TRAILING,
                trailing_data=PostStopOrderRequestTrailingData(
                    indent=decimal_to_quotation(Decimal(Config.TRAILING_INDENT)),
                    indent_type=TrailingValueType.TRAILING_VALUE_RELATIVE,
                    spread=decimal_to_quotation(Decimal(Config.TRAILING_SPREAD / 100)),
                    spread_type=TrailingValueType.TRAILING_VALUE_RELATIVE,
                ),
                order_id=str(uuid4()),
            )
            logger.info("✅ Trailing Stop выставлен")
        except Exception as e:
            logger.error(f"Ошибка выставления Trailing Stop: {e}")

        # --- Take Profit ---
        tp_price = entry_price * Decimal(1 + Config.TAKE_PROFIT)
        tp_price = round(tp_price / min_step) * min_step
        
        try:
            logger.info(f"Выставляем Take Profit по {tp_price}...")
            self.client.stop_orders.post_stop_order(
                instrument_id=self.instrument.uid,
                quantity=1,
                price=decimal_to_quotation(tp_price),
                stop_price=decimal_to_quotation(tp_price),
                direction=StopOrderDirection.STOP_ORDER_DIRECTION_SELL,
                account_id=self.account_id,
                stop_order_type=StopOrderType.STOP_ORDER_TYPE_TAKE_PROFIT,
                expiration_type=StopOrderExpirationType.STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_CANCEL,
                order_id=str(uuid4()),
            )
            logger.info("✅ Take Profit выставлен")
        except Exception as e:
            logger.error(f"Ошибка выставления Take Profit: {e}")

    def _cancel_all_orders(self):
        """Отмена всех заявок по инструменту"""
        logger.info("Отмена активных заявок...")
        try:
            # Обычные ордера
            orders = self.client.orders.get_orders(account_id=self.account_id).orders
            for o in orders:
                if o.instrument_uid == self.instrument.uid:
                    self.client.orders.cancel_order(account_id=self.account_id, order_id=o.order_id)
                    logger.info(f"Отменен ордер {o.order_id}")

            # Стоп-ордера
            stops = self.client.stop_orders.get_stop_orders(
                account_id=self.account_id, 
                status=StopOrderStatusOption.STOP_ORDER_STATUS_ACTIVE
            ).stop_orders
            
            for s in stops:
                if s.instrument_uid == self.instrument.uid:
                    self.client.stop_orders.cancel_stop_order(account_id=self.account_id, stop_order_id=s.stop_order_id)
                    logger.info(f"Отменен стоп-ордер {s.stop_order_id}")
                    
        except Exception as e:
            logger.warning(f"Ошибка при отмене заявок (не критично): {e}")

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()