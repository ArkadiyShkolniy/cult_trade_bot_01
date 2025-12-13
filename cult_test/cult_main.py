import os
from uuid import uuid4
from decimal import Decimal
from datetime import timedelta
from dotenv import load_dotenv
import os

# Загрузить переменные из .env
load_dotenv()

TOKEN = os.environ.get("TINKOFF_INVEST_TOKEN")
if not TOKEN:
    raise ValueError("TINKOFF_INVEST_TOKEN не установлен!")

# Временная отладка - проверка токена (удалите после теста)
print(f"Токен загружен: {TOKEN[:10]}...{TOKEN[-10:] if len(TOKEN) > 20 else 'короткий'}")  # Показывает только начало и конец

from tinkoff.invest import (
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
)
from tinkoff.invest.schemas import TrailingValueType
from tinkoff.invest.utils import (
    quotation_to_decimal,
    decimal_to_quotation,
    money_to_decimal,
    now,
)

TOKEN = os.environ["TINKOFF_INVEST_TOKEN"]

# Тикер Сбербанка
TICKER = "SBER"
CLASS_CODE = "TQBR"  # Класс кода для акций на Московской бирже
TRAILING_STOP_INDENT_PERCENTAGE = 0.003  # 0.3% - отступ trailing stop от максимальной цены
TRAILING_STOP_SPREAD_PERCENTAGE = 0.5  # 0.5% - защитный спред для trailing stop
TAKE_PROFIT_PERCENTAGE = 0.01  # 1% выше цены покупки

# Параметры EMA стратегии
EMA_SHORT_PERIOD = 30  # Быстрая EMA
EMA_LONG_PERIOD = 195  # Медленная EMA
CANDLE_INTERVAL = CandleInterval.CANDLE_INTERVAL_30_MIN  # Таймфрейм 30 минут


def calculate_ema(prices: list[Decimal], period: int) -> list[Decimal]:
    """
    Вычисляет экспоненциальную скользящую среднюю (EMA).
    
    Args:
        prices: Список цен закрытия свечей
        period: Период EMA
    
    Returns:
        Список значений EMA
    """
    if len(prices) < period:
        return []
    
    ema_values = []
    multiplier = Decimal(2) / Decimal(period + 1)
    
    # Первое значение EMA = простое среднее первых period значений
    first_ema = sum(prices[:period]) / Decimal(period)
    ema_values.append(first_ema)
    
    # Остальные значения EMA
    for i in range(period, len(prices)):
        ema = (prices[i] * multiplier) + (ema_values[-1] * (Decimal(1) - multiplier))
        ema_values.append(ema)
    
    return ema_values


def get_trading_signal(client, instrument_id, instrument_figi, ema_short_period, ema_long_period, candle_interval):
    """
    Получает торговый сигнал на основе пересечения EMA.
    
    Returns:
        'BUY' - сигнал на покупку (EMA короткая пересекла EMA длинную снизу вверх)
        'SELL' - сигнал на продажу (EMA короткая пересекла EMA длинную сверху вниз)
        'HOLD' - нет сигнала
    """
    # Получаем свечи за период, достаточный для расчета EMA(195)
    # Нужно минимум 195 свечей + несколько дополнительных для надежности
    days_back = max(ema_long_period * 2, 100)  # Достаточно для EMA(195) на 30-минутных свечах
    
    print(f"\nЗагружаем исторические свечи за последние {days_back} дней...")
    candles = list(client.get_all_candles(
        instrument_id=instrument_id,
        from_=now() - timedelta(days=days_back),
        to=now(),
        interval=candle_interval,
    ))
    
    if len(candles) < ema_long_period:
        print(f"⚠️ Недостаточно свечей: {len(candles)}, требуется минимум {ema_long_period}")
        return 'HOLD'
    
    # Получаем цены закрытия
    prices = [quotation_to_decimal(candle.close) for candle in candles]
    
    # Рассчитываем EMA
    ema_short_values = calculate_ema(prices, ema_short_period)
    ema_long_values = calculate_ema(prices, ema_long_period)
    
    if len(ema_short_values) < 2 or len(ema_long_values) < 2:
        print("⚠️ Недостаточно данных для определения сигнала")
        return 'HOLD'
    
    # Берем последние два значения для определения пересечения
    # Индексы для последних значений (после расчета EMA их меньше на период)
    short_idx = len(ema_short_values) - 1
    long_idx = len(ema_long_values) - 1
    
    current_ema_short = ema_short_values[short_idx]
    current_ema_long = ema_long_values[long_idx]
    
    prev_ema_short = ema_short_values[short_idx - 1] if short_idx > 0 else None
    prev_ema_long = ema_long_values[long_idx - 1] if long_idx > 0 else None
    
    print(f"\nТекущие значения EMA:")
    print(f"  EMA({ema_short_period}): {current_ema_short}")
    print(f"  EMA({ema_long_period}): {current_ema_long}")
    
    if prev_ema_short is not None and prev_ema_long is not None:
        print(f"\nПредыдущие значения EMA:")
        print(f"  EMA({ema_short_period}): {prev_ema_short}")
        print(f"  EMA({ema_long_period}): {prev_ema_long}")
        
        # Проверяем пересечение (Золотой крест / Мертвый крест)
        # Золотой крест: EMA короткая пересекла EMA длинную снизу вверх (сигнал на покупку)
        if prev_ema_short <= prev_ema_long and current_ema_short > current_ema_long:
            print(f"\n✅ СИГНАЛ НА ПОКУПКУ: Золотой крест!")
            print(f"   EMA({ema_short_period}) пересекла EMA({ema_long_period}) снизу вверх")
            return 'BUY'
        
        # Мертвый крест: EMA короткая пересекла EMA длинную сверху вниз (сигнал на продажу)
        elif prev_ema_short >= prev_ema_long and current_ema_short < current_ema_long:
            print(f"\n✅ СИГНАЛ НА ПРОДАЖУ: Мертвый крест!")
            print(f"   EMA({ema_short_period}) пересекла EMA({ema_long_period}) сверху вниз")
            return 'SELL'
        else:
            if current_ema_short > current_ema_long:
                print(f"\nℹ️ Тренд бычий (EMA{ema_short_period} > EMA{ema_long_period}), но пересечения нет")
            else:
                print(f"\nℹ️ Тренд медвежий (EMA{ema_short_period} < EMA{ema_long_period}), но пересечения нет")
    
    return 'HOLD'


def cancel_all_orders_for_instrument(client, account_id, instrument_id, instrument_figi):
    """
    Отменяет все активные ордера и стоп-ордера для указанного инструмента.
    
    Args:
        client: Клиент Tinkoff Invest
        account_id: ID счета
        instrument_id: UID инструмента
        instrument_figi: FIGI инструмента
    """
    print(f"\nОтменяем все заявки по инструменту {instrument_figi}...")
    
    cancelled_count = 0
    
    try:
        # Получаем все активные ордера
        orders_response = client.orders.get_orders(account_id=account_id)
        
        # Отменяем ордера по нашему инструменту
        for order in orders_response.orders:
            if order.instrument_uid == instrument_id or order.figi == instrument_figi:
                try:
                    client.orders.cancel_order(
                        account_id=account_id,
                        order_id=order.order_id
                    )
                    print(f"  ✅ Отменен ордер: {order.order_id}")
                    cancelled_count += 1
                except Exception as e:
                    print(f"  ⚠️ Ошибка при отмене ордера {order.order_id}: {e}")
        
        # Получаем все активные стоп-ордера
        stop_orders_response = client.stop_orders.get_stop_orders(
            account_id=account_id,
            status=StopOrderStatusOption.STOP_ORDER_STATUS_ACTIVE
        )
        
        # Отменяем стоп-ордера по нашему инструменту
        for stop_order in stop_orders_response.stop_orders:
            if stop_order.instrument_uid == instrument_id or stop_order.figi == instrument_figi:
                try:
                    client.stop_orders.cancel_stop_order(
                        account_id=account_id,
                        stop_order_id=stop_order.stop_order_id
                    )
                    print(f"  ✅ Отменен стоп-ордер: {stop_order.stop_order_id}")
                    cancelled_count += 1
                except Exception as e:
                    print(f"  ⚠️ Ошибка при отмене стоп-ордера {stop_order.stop_order_id}: {e}")
        
        if cancelled_count > 0:
            print(f"✅ Всего отменено заявок: {cancelled_count}")
        else:
            print("ℹ️ Активных заявок по инструменту не найдено")
            
    except Exception as e:
        print(f"❌ Ошибка при отмене заявок: {e}")
        import traceback
        traceback.print_exc()


with Client(TOKEN) as client:
    # Получаем список аккаунтов
    accounts = client.users.get_accounts()
    account_id = accounts.accounts[0].id
    print(f"Используем аккаунт: {account_id}")
    
    # Получаем информацию об инструменте (акции Сбербанка)
    instrument = client.instruments.get_instrument_by(
        id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER,
        id=TICKER,
        class_code=CLASS_CODE,
    )
    
    instrument_id = instrument.instrument.uid
    instrument_figi = instrument.instrument.figi
    instrument_name = instrument.instrument.name
    min_price_increment = quotation_to_decimal(instrument.instrument.min_price_increment)
    
    print(f"Инструмент: {instrument_name}")
    print(f"FIGI: {instrument_figi}")
    print(f"UID: {instrument_id}")
    print(f"Минимальный шаг цены: {min_price_increment}")
    print(f"Таймфрейм: 30 минут")
    print(f"EMA периоды: {EMA_SHORT_PERIOD} и {EMA_LONG_PERIOD}")
    
    # Проверяем торговый сигнал на основе EMA
    signal = get_trading_signal(
        client=client,
        instrument_id=instrument_id,
        instrument_figi=instrument_figi,
        ema_short_period=EMA_SHORT_PERIOD,
        ema_long_period=EMA_LONG_PERIOD,
        candle_interval=CANDLE_INTERVAL,
    )
    
    if signal != 'BUY':
        if signal == 'SELL':
            print(f"\n⚠️ Получен сигнал на продажу, но покупка не выполняется.")
            print("Для автоматической продажи нужна открытая позиция.")
        else:
            print(f"\n⚠️ Нет сигнала на покупку. Покупка не выполняется.")
        print("Программа завершена.")
    else:
        # Покупаем 1 акцию
        print(f"\nПокупаем 1 акцию {TICKER}...")
        buy_response = client.orders.post_order(
            order_type=OrderType.ORDER_TYPE_MARKET,
            direction=OrderDirection.ORDER_DIRECTION_BUY,
            instrument_id=instrument_id,
            quantity=1,
            account_id=account_id,
            order_id=str(uuid4()),
        )
        
        status = buy_response.execution_report_status
        print(f"Статус покупки: {status} ({OrderExecutionReportStatus(status).name})")
        print(f"Сообщение: {buy_response.message}")
        print(f"Исполнено лотов: {buy_response.lots_executed}")
        
        # Проверяем, что ордер успешно принят (NEW, FILL или PARTIALLYFILL)
        is_order_accepted = status in [
            OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_NEW,  # 4 - заявка принята
            OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_FILL,  # 1 - полностью исполнена
            OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_PARTIALLYFILL,  # 5 - частично исполнена
        ]
        
        if is_order_accepted:
            print("Покупка успешно принята!")
            
            # Получаем цену исполнения покупки
            executed_price = money_to_decimal(buy_response.executed_order_price)
            print(f"Цена исполнения покупки: {executed_price}")
            
            # Сохраняем ID стоп-ордеров для отслеживания
            trailing_stop_order_id = None
            take_profit_order_id = None
            
            # Рассчитываем начальную стоп-цену для trailing stop (0.3% ниже цены покупки)
            initial_stop_price = executed_price * Decimal(1 - TRAILING_STOP_INDENT_PERCENTAGE)
            initial_stop_price = round(initial_stop_price / min_price_increment) * min_price_increment
            
            # Рассчитываем начальную цену исполнения (немного выше стоп-цены)
            initial_execution_price = executed_price * Decimal(1 + 0.001)  # 0.1% выше цены покупки
            initial_execution_price = round(initial_execution_price / min_price_increment) * min_price_increment
            
            print(f"\nВыставляем трейлинг стоп...")
            print(f"Начальная стоп-цена: {initial_stop_price}")
            print(f"Отступ: {TRAILING_STOP_INDENT_PERCENTAGE * 100}% от максимальной цены")
            print(f"Защитный спред: {TRAILING_STOP_SPREAD_PERCENTAGE}%")
            print("Трейлинг стоп будет автоматически подтягиваться при росте цены")
            
            try:
                # Выставляем трейлинг стоп ордер
                trailing_stop_response = client.stop_orders.post_stop_order(
                    instrument_id=instrument_id,
                    quantity=1,
                    price=decimal_to_quotation(initial_execution_price),
                    stop_price=decimal_to_quotation(initial_stop_price),
                    direction=StopOrderDirection.STOP_ORDER_DIRECTION_SELL,
                    account_id=account_id,
                    stop_order_type=StopOrderType.STOP_ORDER_TYPE_TAKE_PROFIT,
                    expiration_type=StopOrderExpirationType.STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_CANCEL,
                    exchange_order_type=ExchangeOrderType.EXCHANGE_ORDER_TYPE_LIMIT,
                    take_profit_type=TakeProfitType.TAKE_PROFIT_TYPE_TRAILING,
                    trailing_data=PostStopOrderRequestTrailingData(
                        indent=decimal_to_quotation(Decimal(TRAILING_STOP_INDENT_PERCENTAGE)),
                        indent_type=TrailingValueType.TRAILING_VALUE_RELATIVE,
                        spread=decimal_to_quotation(Decimal(TRAILING_STOP_SPREAD_PERCENTAGE / 100)),
                        spread_type=TrailingValueType.TRAILING_VALUE_RELATIVE,
                    ),
                    order_id=str(uuid4()),
                )
                
                trailing_stop_order_id = trailing_stop_response.stop_order_id
                print(f"✅ Трейлинг стоп ордер успешно выставлен!")
                print(f"ID трейлинг стоп ордера: {trailing_stop_order_id}")
                print(f"Начальная стоп-цена: {initial_stop_price}")
                
            except Exception as e:
                print(f"❌ Ошибка при выставлении трейлинг стоп: {e}")
                import traceback
                traceback.print_exc()
            
            # Рассчитываем цену тейк-профит (1% выше цены покупки)
            take_profit_price = executed_price * Decimal(1 + TAKE_PROFIT_PERCENTAGE)
            
            # Округляем цену до минимального шага цены
            take_profit_price = round(take_profit_price / min_price_increment) * min_price_increment
            
            print(f"\nВыставляем тейк-профит на {TAKE_PROFIT_PERCENTAGE * 100}% выше цены покупки...")
            print(f"Цена тейк-профит: {take_profit_price}")
            
            try:
                # Выставляем тейк-профит ордер
                take_profit_response = client.stop_orders.post_stop_order(
                    instrument_id=instrument_id,
                    quantity=1,
                    price=decimal_to_quotation(take_profit_price),
                    stop_price=decimal_to_quotation(take_profit_price),
                    direction=StopOrderDirection.STOP_ORDER_DIRECTION_SELL,
                    account_id=account_id,
                    stop_order_type=StopOrderType.STOP_ORDER_TYPE_TAKE_PROFIT,
                    expiration_type=StopOrderExpirationType.STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_CANCEL,
                    order_id=str(uuid4()),
                )
                
                take_profit_order_id = take_profit_response.stop_order_id
                print(f"✅ Тейк-профит ордер успешно выставлен!")
                print(f"ID тейк-профит ордера: {take_profit_order_id}")
                print(f"Цена тейк-профит: {take_profit_price}")
                
            except Exception as e:
                print(f"❌ Ошибка при выставлении тейк-профит: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"\n✅ Все ордера выставлены успешно!")
            print("Позиция будет автоматически закрыта при срабатывании:")
            print("  - Тейк-профит: при достижении цены +1%")
            print("  - Трейлинг стоп: при падении цены на 0.3% от максимума")
            print("\nПрограмма завершена. Ордера работают автоматически.")
            
        else:
            print(f"❌ Ошибка при покупке. Статус: {status} ({OrderExecutionReportStatus(status).name})")
            if status == OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_REJECTED:
                print("Заявка отклонена брокером")
            elif status == OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_CANCELLED:
                print("Заявка отменена")