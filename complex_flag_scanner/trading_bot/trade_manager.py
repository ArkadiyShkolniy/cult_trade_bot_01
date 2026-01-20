import json
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from t_tech.invest import Client
from t_tech.invest.utils import quotation_to_decimal

class TradeManager:
    """
    Класс управления сделками: расчет объема, отправка ордеров, учет позиций, сбор данных для ML.
    """
    def __init__(self, token, account_id=None, risk_per_trade=0.01, dry_run=True, debug_mode=True, use_ai_filter=True):
        """
        Args:
            token: API токен
            account_id: ID торгового счета
            risk_per_trade: Риск на сделку (0.01 = 1%)
            dry_run: True = эмуляция торгов
            debug_mode: True = всегда торговать 1 лотом (для отладки)
            use_ai_filter: True = использовать ML модель для фильтрации сделок
        """
        self.token = token
        self.risk_per_trade = risk_per_trade
        self.dry_run = dry_run
        self.debug_mode = debug_mode
        self.account_id = account_id
        self.use_ai_filter = use_ai_filter
        
        # Комиссия брокера (0.04% = 0.0004)
        self.commission_rate = 0.0004
        
        # --- Файловая структура ---
        self.base_dir = Path("trading_bot")
        self.model_path = Path("neural_network/models/trading_model_rf.pkl")
        
        # Активные сделки (JSON)
        self.trades_file = self.base_dir / "trades_active.json"
        
        # История закрытых сделок (JSON)
        self.history_file = self.base_dir / "trades_history.json"
        
        # Данные для обучения ML
        self.training_dir = self.base_dir / "training_data"
        self.snapshots_dir = self.training_dir / "snapshots"
        self.dataset_file = self.training_dir / "dataset_v1.csv"
        
        # Создаем директории
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализация CSV датасета
        if not self.dataset_file.exists():
            with open(self.dataset_file, 'w') as f:
                # Заголовок датасета
                f.write("trade_id,ticker,direction,entry_time,exit_time,entry_price,exit_price,pnl_net,result_type,mae,mfe,hold_time_minutes,stop_loss,take_profit,pattern_score,snapshot_file,ai_probability\n")
        
        # Загрузка состояния
        self.active_trades = self._load_json(self.trades_file, is_dict=True)
        self.closed_trades = self._load_json(self.history_file, is_dict=False)
        
        # Загрузка AI модели
        self.ai_model = None
        if self.use_ai_filter:
            if self.model_path.exists():
                try:
                    self.ai_model = joblib.load(self.model_path)
                    print(f"✅ AI Модель загружена: {self.model_path}")
                except Exception as e:
                    print(f"⚠️ Ошибка загрузки AI модели: {e}")
            else:
                print(f"⚠️ AI Модель не найдена по пути {self.model_path}")

        if not self.dry_run and not self.account_id:
            self._fetch_account_id()

    def _load_json(self, path, is_dict=True):
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except:
                return {} if is_dict else []
        return {} if is_dict else []

    def _save_active_trades(self):
        with open(self.trades_file, 'w') as f:
            json.dump(self.active_trades, f, indent=4, default=str)

    def _save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.closed_trades, f, indent=4, default=str)

    def _fetch_account_id(self):
        """Получает ID первого брокерского счета"""
        try:
            with Client(self.token) as client:
                accounts = client.users.get_accounts()
                self.account_id = accounts.accounts[0].id
                print(f"✅ TradeManager: Используем счет {self.account_id}")
        except Exception as e:
            print(f"❌ Ошибка получения счета: {e}")

    def _get_portfolio_value(self):
        """Получает текущую стоимость портфеля"""
        if self.dry_run:
            return 100000.0  # Виртуальные 100к
        try:
            with Client(self.token) as client:
                portfolio = client.operations.get_portfolio(account_id=self.account_id)
                amount = quotation_to_decimal(portfolio.total_amount_portfolio)
                return float(amount)
        except Exception as e:
            print(f"⚠️ Не удалось получить баланс: {e}")
            return 100000.0

    def _get_lot_size(self, uid):
        """Получает размер лота"""
        if self.dry_run:
            return 1
        try:
            with Client(self.token) as client:
                instrument = client.instruments.get_instrument_by(id_type=1, id=uid).instrument
                return instrument.lot
        except:
            return 1

    def calculate_quantity(self, entry_price, stop_loss, instrument_uid):
        """
        Рассчитывает количество лотов.
        """
        # В режиме отладки - всегда 1 лот
        if self.debug_mode:
            return 1, self._get_lot_size(instrument_uid)

        portfolio_value = self._get_portfolio_value()
        risk_amount = portfolio_value * self.risk_per_trade
        
        loss_per_share = abs(entry_price - stop_loss)
        if loss_per_share == 0: return 0, 1
        
        lot_size = self._get_lot_size(instrument_uid)
        loss_per_lot = loss_per_share * lot_size
        
        if loss_per_lot == 0: return 0, lot_size
        
        quantity = int(risk_amount / loss_per_lot)
        if quantity < 1: quantity = 0
        
        return quantity, lot_size
        
    def _predict_success(self, pattern_info, entry_price, stop_loss, take_profit):
        """
        Использует AI модель для оценки вероятности успеха сделки.
        Возвращает: (is_good: bool, probability: float)
        """
        if not self.ai_model or not pattern_info:
            return True, 0.5 # Если модели нет, пропускаем всех (neutral)
            
        try:
            # 1. Извлечение признаков (Feature Extraction)
            # Должно полностью совпадать с generate_trading_dataset.py / train_trading_model.py
            
            t0 = pattern_info['t0']['price']
            t1 = pattern_info['t1']['price']
            t2 = pattern_info['t2']['price']
            t3 = pattern_info['t3']['price']
            
            # Индексы (для наклона)
            t1_idx = pattern_info['t1']['idx']
            t3_idx = pattern_info['t3']['idx']
            
            # correction_ratio
            pole_height = abs(t1 - t0)
            correction_depth = abs(t2 - t1)
            correction_ratio = correction_depth / pole_height if pole_height != 0 else 0
            
            # slope_channel
            slope_channel = (t3 - t1) / (t3_idx - t1_idx) if (t3_idx - t1_idx) != 0 else 0
            
            # risk_reward_ratio
            rr_ratio = abs(take_profit - entry_price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) != 0 else 0
            
            # Формируем DataFrame (модель ожидает имена колонок)
            features = pd.DataFrame([{
                'correction_ratio': correction_ratio,
                'slope_channel': slope_channel,
                'risk_reward_ratio': rr_ratio
            }])
            
            # 2. Прогноз
            # predict_proba возвращает [[prob_0, prob_1]]
            probability = self.ai_model.predict_proba(features)[0][1]
            
            # Порог принятия решения (например, > 50%)
            # Можно сделать настраиваемым параметром
            is_good = probability > 0.5
            
            return is_good, probability
            
        except Exception as e:
            print(f"⚠️ Ошибка AI прогноза: {e}")
            return True, 0.5

    def open_position(self, ticker, uid, direction, price, stop_loss, take_profit, strategy_desc, df_context=None, pattern_info=None):
        """
        Открывает позицию и сохраняет данные для ML.
        """
        if ticker in self.active_trades:
            return
            
        print(f"\n🔔 СИГНАЛ НА ВХОД: {ticker} ({direction})")
        
        # --- AI ФИЛЬТР ---
        ai_prob = 0.0
        if self.use_ai_filter and self.ai_model:
            is_good, ai_prob = self._predict_success(pattern_info, price, stop_loss, take_profit)
            if not is_good:
                print(f"🤖 AI FILTER: Сделка отклонена. Вероятность успеха {ai_prob:.1%} < 50%")
                return
            else:
                print(f"🤖 AI FILTER: Одобрено! Вероятность успеха {ai_prob:.1%}")
        
        quantity_lots, lot_size = self.calculate_quantity(price, stop_loss, uid)
        
        if quantity_lots == 0:
            print(f"❌ Отмена: 0 лотов (недостаточно капитала или риск велик)")
            return

        # Рассчитываем комиссию
        position_value = price * quantity_lots * lot_size
        commission = position_value * self.commission_rate

        print(f"   Цена: {price}, SL: {stop_loss}, TP: {take_profit}")
        print(f"   Объем: {quantity_lots} лотов (x{lot_size})")
        print(f"   Комиссия входа: {commission:.2f}")
        
        order_id = "SIM_" + datetime.now().strftime("%H%M%S")
        trade_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + ticker
        
        # --- ML: Сохранение контекста (снэпшот) ---
        snapshot_filename = ""
        if df_context is not None and not df_context.empty:
            snapshot_filename = f"{trade_id}.csv"
            try:
                # Сохраняем последние 200 свечей
                df_save = df_context.tail(200).copy()
                df_save.to_csv(self.snapshots_dir / snapshot_filename, index=False)
                
                # Сохраняем паттерн
                if pattern_info:
                    with open(self.snapshots_dir / f"{trade_id}_pattern.json", 'w') as f:
                        json.dump(pattern_info, f, default=str, indent=2)
            except Exception as e:
                print(f"⚠️ Ошибка сохранения снэпшота: {e}")

        if not self.dry_run:
            # TODO: Реальная отправка
            pass

        trade = {
            'id': trade_id,
            'ticker': ticker,
            'uid': uid,
            'direction': direction,
            'entry_time': datetime.now().isoformat(),
            'entry_price': price,
            'quantity_lots': quantity_lots,
            'lot_size': lot_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'commission_entry': commission,
            'status': 'OPEN',
            'strategy_desc': strategy_desc,
            # Метрики ML
            'mae': 0.0,
            'mfe': 0.0,
            'snapshot_file': snapshot_filename,
            'ai_probability': ai_prob
        }
        
        self.active_trades[ticker] = trade
        self._save_active_trades()
        
        action = "🟢 КУПЛЕНО" if direction == 'LONG' else "🔴 ПРОДАНО"
        print(f"✅ {action} {quantity_lots} лотов {ticker}. {strategy_desc}")

    def update_positions(self, current_prices):
        """
        Проверяет выходы и обновляет MFE/MAE.
        """
        to_remove = []
        
        for ticker, trade in self.active_trades.items():
            if ticker not in current_prices:
                continue
                
            current_data = current_prices[ticker]
            current_price = current_data['price']
            
            direction = trade['direction']
            entry_price = trade['entry_price']
            stop_loss = trade['stop_loss']
            take_profit = trade['take_profit']
            
            # --- Обновление MFE/MAE ---
            price_change = current_price - entry_price
            
            # Для шорта прибыль - это падение цены (отрицательный change)
            # Превратим в "пункты прибыли":
            if direction == 'SHORT':
                points_profit = -price_change
            else:
                points_profit = price_change
                
            # MFE (максимальная прибыль в моменте)
            if points_profit > trade.get('mfe', 0):
                trade['mfe'] = points_profit
                
            # MAE (максимальный убыток/просадка в моменте)
            # MAE всегда <= 0 (или tracking drawdown)
            if points_profit < trade.get('mae', 0):
                trade['mae'] = points_profit
            
            close_reason = None
            
            if direction == 'LONG':
                if current_price >= take_profit:
                    close_reason = f"TAKE PROFIT"
                elif current_price <= stop_loss:
                    close_reason = f"STOP LOSS"
            elif direction == 'SHORT':
                if current_price <= take_profit:
                    close_reason = f"TAKE PROFIT"
                elif current_price >= stop_loss:
                    close_reason = f"STOP LOSS"
            
            if close_reason:
                self._close_position(ticker, trade, current_price, close_reason, current_data['time'])
                to_remove.append(ticker)
        
        # Если были обновления MFE/MAE, сохраняем состояние
        if not to_remove and self.active_trades:
            self._save_active_trades()
                
        for t in to_remove:
            del self.active_trades[t]
        
        if to_remove:
            self._save_active_trades()
            self.print_statistics()

    def _close_position(self, ticker, trade, exit_price, reason, exit_time):
        quantity = trade['quantity_lots']
        lot_size = trade['lot_size']
        entry_price = trade['entry_price']
        direction = trade['direction']
        
        # Расчет финансов
        position_value_exit = exit_price * quantity * lot_size
        commission_exit = position_value_exit * self.commission_rate
        total_commission = trade['commission_entry'] + commission_exit
        
        if direction == 'LONG':
            gross_profit = (exit_price - entry_price) * quantity * lot_size
        else: # SHORT
            gross_profit = (entry_price - exit_price) * quantity * lot_size
            
        net_profit = gross_profit - total_commission
        
        print(f"\n⚖️ ЗАКРЫТИЕ ПОЗИЦИИ {ticker} ({direction})")
        print(f"   Причина: {reason}")
        print(f"   Вход: {entry_price:.2f} -> Выход: {exit_price:.2f}")
        print(f"   P&L (грязный): {gross_profit:.2f}")
        print(f"   Комиссия: {total_commission:.2f}")
        print(f"   P&L (чистый): {net_profit:.2f}")
        print(f"   MFE: {trade.get('mfe', 0):.2f} | MAE: {trade.get('mae', 0):.2f}")
        
        if not self.dry_run:
            # TODO: Отправка ордера
            pass

        # Финализация записи
        trade['exit_time'] = str(exit_time)
        trade['exit_price'] = exit_price
        trade['status'] = 'CLOSED'
        trade['close_reason'] = reason
        trade['gross_profit'] = gross_profit
        trade['commission_total'] = total_commission
        trade['net_profit'] = net_profit
        
        if isinstance(self.closed_trades, list):
            self.closed_trades.append(trade)
        else:
            self.closed_trades = [trade]
            
        self._save_history()
        
        # --- ML: Запись в датасет ---
        try:
            entry_dt = datetime.fromisoformat(trade['entry_time'])
            if isinstance(exit_time, str):
                exit_dt = datetime.fromisoformat(exit_time)
            else:
                exit_dt = exit_time
                
            hold_time_minutes = (exit_dt - entry_dt).total_seconds() / 60
            result_type = "WIN" if net_profit > 0 else "LOSS"
            
            with open(self.dataset_file, 'a') as f:
                # trade_id,ticker,direction,entry_time,exit_time,entry_price,exit_price,pnl_net,result_type,mae,mfe,hold_time_minutes,stop_loss,take_profit,pattern_score,snapshot_file,ai_probability
                f.write(f"{trade['id']},{ticker},{direction},{trade['entry_time']},{trade['exit_time']},"
                        f"{entry_price},{exit_price},{net_profit:.2f},{result_type},"
                        f"{trade.get('mae', 0):.2f},{trade.get('mfe', 0):.2f},{hold_time_minutes:.1f},"
                        f"{trade['stop_loss']},{trade['take_profit']},0,{trade['snapshot_file']},"
                        f"{trade.get('ai_probability', 0):.4f}\n")
            print(f"   💾 Данные для ML сохранены в {self.dataset_file}")
            
        except Exception as e:
            print(f"❌ Ошибка сохранения датасета ML: {e}")

    def print_statistics(self):
        """Выводит сводную статистику"""
        if not self.closed_trades or not isinstance(self.closed_trades, list):
            return

        total_trades = len(self.closed_trades)
        total_profit = sum(t['net_profit'] for t in self.closed_trades)
        wins = sum(1 for t in self.closed_trades if t['net_profit'] > 0)
        losses = total_trades - wins
        
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        print("\n" + "="*40)
        print("📊 СТАТИСТИКА ТОРГОВЛИ")
        print(f"   Всего сделок: {total_trades}")
        print(f"   Прибыльных: {wins} ({win_rate:.1f}%) | Убыточных: {losses}")
        print(f"   Общий P&L: {total_profit:.2f} руб.")
        print("="*40 + "\n")