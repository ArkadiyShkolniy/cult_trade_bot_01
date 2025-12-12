import os
import time
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
)
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
WAIT_MINUTES = 5
STOP_LOSS_PERCENTAGE = 0.003  # 0.3% ниже цены покупки
TAKE_PROFIT_PERCENTAGE = 0.01  # 1% выше цены покупки


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
        stop_order_id = None
        take_profit_order_id = None
        
        # Рассчитываем цену стоп-лосс (0.3% ниже цены покупки)
        stop_loss_price = executed_price * Decimal(1 - STOP_LOSS_PERCENTAGE)
        
        # Округляем цену до минимального шага цены
        stop_loss_price = round(stop_loss_price / min_price_increment) * min_price_increment
        
        print(f"\nВыставляем стоп-лосс на {STOP_LOSS_PERCENTAGE * 100}% ниже цены покупки...")
        print(f"Цена стоп-лосс: {stop_loss_price}")
        
        try:
            # Выставляем стоп-лосс ордер
            stop_order_response = client.stop_orders.post_stop_order(
                instrument_id=instrument_id,
                quantity=1,
                stop_price=decimal_to_quotation(stop_loss_price),
                direction=StopOrderDirection.STOP_ORDER_DIRECTION_SELL,
                account_id=account_id,
                stop_order_type=StopOrderType.STOP_ORDER_TYPE_STOP_LOSS,
                expiration_type=StopOrderExpirationType.STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_CANCEL,
                order_id=str(uuid4()),
            )
            
            stop_order_id = stop_order_response.stop_order_id
            print(f"✅ Стоп-лосс ордер успешно выставлен!")
            print(f"ID стоп-ордера: {stop_order_id}")
            print(f"Стоп-цена: {stop_loss_price}")
            
        except Exception as e:
            print(f"❌ Ошибка при выставлении стоп-лосс: {e}")
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
        
        # Ждем 5 минут
        print(f"\nЖдем {WAIT_MINUTES} минут перед продажей...")
        time.sleep(WAIT_MINUTES * 60)
        
        # Проверяем позицию перед продажей
        print("\nПроверяем позицию в портфеле...")
        portfolio = client.operations.get_portfolio(account_id=account_id)
        position_quantity = Decimal(0)
        
        for position in portfolio.positions:
            if position.figi == instrument_figi or position.instrument_uid == instrument_id:
                position_quantity = quotation_to_decimal(position.quantity)
                print(f"Найдена позиция: {position_quantity} лотов")
                break
        
        # Если позиция равна 0 или меньше 1, значит актив продан или сработал стоп/тейк-профит
        if position_quantity < Decimal("1"):
            print(f"\n⚠️ Позиция по инструменту отсутствует или меньше 1 лота ({position_quantity})")
            print("Возможно, сработал стоп-лосс или тейк-профит, или актив был продан")
            
            # Отменяем все заявки по этому инструменту
            cancel_all_orders_for_instrument(client, account_id, instrument_id, instrument_figi)
            
            print("✅ Обработка завершена - позиция закрыта")
        else:
            # Продаем 1 акцию
            print(f"\nПродаем 1 акцию {TICKER}...")
            try:
                sell_response = client.orders.post_order(
                    order_type=OrderType.ORDER_TYPE_MARKET,
                    direction=OrderDirection.ORDER_DIRECTION_SELL,
                    instrument_id=instrument_id,
                    quantity=1,
                    account_id=account_id,
                    order_id=str(uuid4()),
                )
                
                sell_status = sell_response.execution_report_status
                print(f"Статус продажи: {sell_status} ({OrderExecutionReportStatus(sell_status).name})")
                print(f"Сообщение: {sell_response.message}")
                print(f"Исполнено лотов: {sell_response.lots_executed}")
                
                if sell_status in [
                    OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_NEW,
                    OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_FILL,
                    OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_PARTIALLYFILL,
                ]:
                    print("✅ Продажа успешно принята!")
                    
                    # После успешной продажи отменяем все заявки по инструменту
                    cancel_all_orders_for_instrument(client, account_id, instrument_id, instrument_figi)
                    
                else:
                    print(f"❌ Ошибка при продаже. Статус: {sell_status}")
            except Exception as e:
                print(f"❌ Исключение при продаже: {e}")
                import traceback
                traceback.print_exc()
    else:
        print(f"❌ Ошибка при покупке. Статус: {status} ({OrderExecutionReportStatus(status).name})")
        if status == OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_REJECTED:
            print("Заявка отклонена брокером")
        elif status == OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_CANCELLED:
            print("Заявка отменена")