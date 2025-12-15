import streamlit as st
import pandas as pd
import time
import logging
import sys
import os
import plotly.graph_objects as go

# Добавляем корень проекта в PYTHONPATH для корректных импортов
project_root = None

# Стратегия 1: Определяем по расположению файла
if __file__:
    try:
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)
        # Если файл в pages/, то поднимаемся на два уровня выше
        if os.path.basename(current_dir) == 'pages':
            candidate_root = os.path.dirname(os.path.dirname(current_dir))
            if os.path.exists(os.path.join(candidate_root, 'cult_test')):
                project_root = candidate_root
    except Exception:
        pass

# Стратегия 2: Проверяем текущую рабочую директорию
if not project_root:
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, 'cult_test')):
        project_root = cwd

# Стратегия 3: Используем /app (для Docker)
if not project_root:
    if os.path.exists('/app/cult_test'):
        project_root = '/app'

# Стратегия 4: Fallback - используем текущую директорию
if not project_root:
    project_root = os.getcwd()

# Добавляем в PYTHONPATH если еще не добавлен
if project_root and project_root not in sys.path:
    sys.path.insert(0, project_root)

# Настройка страницы
st.set_page_config(page_title="Оптимизация Стратегий", layout="wide", page_icon="🔬")

# Импортируем необходимые модули
try:
    from cult_test.cult_main import Config
    from cult_test.strategy_optimizer import StrategyOptimizer
    from t_tech.invest import CandleInterval
except ImportError as e:
    st.error("❌ Ошибка импорта модулей")
    st.code(f"Ошибка: {e}", language="text")
    st.stop()

# Получаем logger
logger = logging.getLogger(__name__)

# Получаем ticker из session_state или используем значение по умолчанию
ticker = st.session_state.get('ticker', Config.TICKER)

st.title("🔬 Оптимизация Стратегий")

# Устанавливаем флаг активной страницы
st.session_state.active_tab = "optimizer"

# Функция для очистки результатов при изменении параметров
def _clear_optimization_results():
    """Очищает результаты оптимизации при изменении параметров"""
    try:
        if 'optimization_results' in st.session_state:
            del st.session_state.optimization_results
        if 'best_strategy' in st.session_state:
            del st.session_state.best_strategy
        st.session_state.optimization_completed = False
    except Exception:
        pass  # Игнорируем ошибки при очистке

# Основной контейнер с настройками
with st.container(border=True):
    st.subheader("Параметры тестирования")
    
    col_main1, col_main2, col_main3 = st.columns(3)
    
    with col_main1:
        st.markdown("**1. Инструмент и данные**")
        opt_ticker = st.text_input("Тикер", value=ticker, key="opt_ticker")
        
        # Выбор таймфрейма
        timeframe_options = {
            "1 минута": CandleInterval.CANDLE_INTERVAL_1_MIN,
            "5 минут": CandleInterval.CANDLE_INTERVAL_5_MIN,
            "15 минут": CandleInterval.CANDLE_INTERVAL_15_MIN,
            "30 минут": CandleInterval.CANDLE_INTERVAL_30_MIN,
            "1 час": CandleInterval.CANDLE_INTERVAL_HOUR,
            "1 день": CandleInterval.CANDLE_INTERVAL_DAY,
        }
        opt_timeframe_name = st.selectbox(
            "Таймфрейм",
            options=list(timeframe_options.keys()),
            index=0,  # По умолчанию 1 минута
            key="opt_timeframe"
        )
        opt_timeframe = timeframe_options[opt_timeframe_name]
        
        opt_days = st.number_input("Дней истории", min_value=7, max_value=90, value=30, step=1, key="opt_days", help="За какой период загружать данные")

    with col_main2:
        st.markdown("**2. Параметры перебора**")
        
        # Выбор типа стратегии
        strategy_type = st.selectbox(
            "Тип стратегии",
            options=["EMA", "RSI", "MACD", "BB", "STOCH", "PATTERN"],
            index=0,
            key="opt_strategy_type"
        )
        
        # Очищаем результаты при изменении типа стратегии
        current_strategy = st.session_state.get('last_strategy_type')
        if current_strategy is not None and current_strategy != strategy_type:
            _clear_optimization_results()
        st.session_state.last_strategy_type = strategy_type
        
        opt_step = st.number_input("Шаг перебора", min_value=1, max_value=10, value=5, step=1, key="opt_step", help="С каким шагом перебирать параметры (меньше шаг = дольше тест)")
        
        opt_commission = st.number_input(
            "Комиссия (%)", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.05, 
            step=0.01, 
            key="opt_commission",
            help="Комиссия брокера за сделку"
        )

    with col_main3:
        st.markdown("**3. Риск-менеджмент**")
        opt_take_profit = st.number_input(
            "Take Profit (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=Config.TAKE_PROFIT * 100, 
            step=0.1, 
            key="opt_take_profit",
            help="Целевая прибыль"
        ) / 100.0
        
        opt_trailing_indent = st.number_input(
            "Trailing Indent (%)", 
            min_value=0.0, 
            max_value=10.0, 
            value=Config.TRAILING_INDENT * 100, 
            step=0.01, 
            key="opt_trailing_indent",
            help="Отступ трейлинг-стопа"
        ) / 100.0

    st.divider()
    st.markdown(f"**4. Диапазоны параметров для стратегии {strategy_type}**")
    
    col_strat1, col_strat2 = st.columns(2)
    
    with col_strat1:
        # Параметры в зависимости от типа стратегии
        if strategy_type == "EMA":
            opt_ema_short_min = st.number_input("EMA Short (мин)", min_value=3, max_value=50, value=5, step=1, key="opt_short_min")
            opt_ema_short_max = st.number_input("EMA Short (макс)", min_value=5, max_value=100, value=30, step=1, key="opt_short_max")
        elif strategy_type == "RSI":
            opt_rsi_period_min = st.number_input("RSI Period (мин)", min_value=5, max_value=20, value=10, step=1, key="opt_rsi_period_min")
            opt_rsi_period_max = st.number_input("RSI Period (макс)", min_value=10, max_value=30, value=20, step=1, key="opt_rsi_period_max")
        elif strategy_type == "MACD":
            opt_macd_fast_min = st.number_input("MACD Fast (мин)", min_value=5, max_value=15, value=8, step=1, key="opt_macd_fast_min")
            opt_macd_fast_max = st.number_input("MACD Fast (макс)", min_value=10, max_value=20, value=16, step=1, key="opt_macd_fast_max")
        elif strategy_type == "BB":
            opt_bb_period_min = st.number_input("BB Period (мин)", min_value=10, max_value=20, value=15, step=1, key="opt_bb_period_min")
            opt_bb_period_max = st.number_input("BB Period (макс)", min_value=15, max_value=30, value=25, step=1, key="opt_bb_period_max")
        elif strategy_type == "STOCH":
            opt_stoch_k_min = st.number_input("Stochastic K (мин)", min_value=8, max_value=15, value=10, step=1, key="opt_stoch_k_min")
            opt_stoch_k_max = st.number_input("Stochastic K (макс)", min_value=12, max_value=25, value=18, step=1, key="opt_stoch_k_max")
        elif strategy_type == "PATTERN":
            opt_pattern_type = st.selectbox(
                "Тип паттернов",
                options=["candlestick", "chart", "combined"],
                index=0,
                key="opt_pattern_type"
            )
            
    with col_strat2:
        if strategy_type == "EMA":
            opt_ema_long_min = st.number_input("EMA Long (мин)", min_value=20, max_value=200, value=50, step=5, key="opt_long_min")
            opt_ema_long_max = st.number_input("EMA Long (макс)", min_value=50, max_value=500, value=150, step=5, key="opt_long_max")
        elif strategy_type == "MACD":
            opt_macd_slow_min = st.number_input("MACD Slow (мин)", min_value=15, max_value=25, value=20, step=1, key="opt_macd_slow_min")
            opt_macd_slow_max = st.number_input("MACD Slow (макс)", min_value=20, max_value=35, value=30, step=1, key="opt_macd_slow_max")
        elif strategy_type == "PATTERN":
            # Выбор конкретных паттернов
            if opt_pattern_type == "candlestick" or opt_pattern_type == "combined":
                st.markdown("**Свечные паттерны:**")
                opt_pattern_hammer = st.checkbox("Молот (Hammer)", value=True, key="opt_pattern_hammer")
                opt_pattern_inv_hammer = st.checkbox("Перевернутый молот", value=True, key="opt_pattern_inv_hammer")
                opt_pattern_bull_eng = st.checkbox("Бычье поглощение", value=True, key="opt_pattern_bull_eng")
                opt_pattern_bear_eng = st.checkbox("Медвежье поглощение", value=True, key="opt_pattern_bear_eng")
                opt_pattern_shooting = st.checkbox("Звезда (Shooting Star)", value=True, key="opt_pattern_shooting")
                opt_pattern_hanging = st.checkbox("Падающая звезда (Hanging Man)", value=True, key="opt_pattern_hanging")
            
            if opt_pattern_type == "chart" or opt_pattern_type == "combined":
                st.markdown("**Графические паттерны:**")
                opt_pattern_double_bottom = st.checkbox("Двойное дно", value=True, key="opt_pattern_double_bottom")
                opt_pattern_double_top = st.checkbox("Двойная вершина", value=True, key="opt_pattern_double_top")
                opt_pattern_triple_bottom = st.checkbox("Тройное дно", value=True, key="opt_pattern_triple_bottom")
                opt_pattern_triple_top = st.checkbox("Тройная вершина", value=True, key="opt_pattern_triple_top")

# Автоматическая очистка результатов при изменении ключевых параметров
current_params_hash = str({
    'type': strategy_type,
    'days': opt_days,
    'step': opt_step,
    'ticker': opt_ticker,
    'timeframe': opt_timeframe_name,
    'commission': opt_commission,
    'take_profit': opt_take_profit,
    'trailing': opt_trailing_indent
})

if 'last_params_hash' not in st.session_state:
    st.session_state.last_params_hash = current_params_hash

# Если параметры изменились - сбрасываем результаты
if st.session_state.last_params_hash != current_params_hash:
    _clear_optimization_results()
    st.session_state.last_params_hash = current_params_hash

# Защита от повторного запуска оптимизации
if st.session_state.get('optimization_in_progress', False):
    st.session_state.optimization_in_progress = False

optimization_in_progress = False

st.divider()

# Кнопка запуска оптимизации
if st.button("🚀 Запустить оптимизацию", type="primary", use_container_width=True, key="start_optimization", disabled=optimization_in_progress):
    st.session_state.optimization_in_progress = True
    st.session_state.optimization_completed = False
    
    _clear_optimization_results()
    
    with st.spinner("Выполняется оптимизация стратегии... (пожалуйста, не меняйте параметры во время выполнения)"):
        status_text = st.empty()
        error_display = st.empty()
        
        try:
            status_text.markdown("🔄 **Запуск оптимизатора...**")
            
            optimizer = StrategyOptimizer()
            
            status_text.markdown("📥 **Загрузка исторических данных...**")
            
            df = optimizer.load_candle_data(
                days=int(opt_days),
                ticker=opt_ticker,
                class_code=Config.CLASS_CODE,
                interval=opt_timeframe
            )
            
            if df is None or len(df) == 0:
                raise ValueError("Не удалось загрузить данные или данные пусты")
            
            timeframe_display = {
                'CANDLE_INTERVAL_1_MIN': 'минутных',
                'CANDLE_INTERVAL_5_MIN': '5-минутных',
                'CANDLE_INTERVAL_15_MIN': '15-минутных',
                'CANDLE_INTERVAL_30_MIN': '30-минутных',
                'CANDLE_INTERVAL_HOUR': 'часовых',
                'CANDLE_INTERVAL_DAY': 'дневных',
            }.get(opt_timeframe.name if hasattr(opt_timeframe, 'name') else str(opt_timeframe), 'свечей')
            
            status_text.markdown(f"✅ **Загружено {len(df)} {timeframe_display} свечей.** Начало оптимизации...")
            
            commission = float(opt_commission)
            results = None
            
            if strategy_type == "EMA":
                opt_ema_short_min = st.session_state.get('opt_short_min', 5)
                opt_ema_short_max = st.session_state.get('opt_short_max', 30)
                opt_ema_long_min = st.session_state.get('opt_long_min', 50)
                opt_ema_long_max = st.session_state.get('opt_long_max', 150)
                results = optimizer.optimize_strategy(
                    df,
                    strategy_type="EMA",
                    ema_short_range=(int(opt_ema_short_min), int(opt_ema_short_max)),
                    ema_long_range=(int(opt_ema_long_min), int(opt_ema_long_max)),
                    step=int(opt_step),
                    timeframe=opt_timeframe,
                    commission_percent=commission,
                    take_profit=opt_take_profit,
                    trailing_stop=opt_trailing_indent
                )
            elif strategy_type == "RSI":
                opt_rsi_period_min = st.session_state.get('opt_rsi_period_min', 10)
                opt_rsi_period_max = st.session_state.get('opt_rsi_period_max', 20)
                results = optimizer.optimize_strategy(
                    df,
                    strategy_type="RSI",
                    rsi_period_range=(int(opt_rsi_period_min), int(opt_rsi_period_max)),
                    step=int(opt_step),
                    timeframe=opt_timeframe,
                    commission_percent=commission,
                    take_profit=opt_take_profit,
                    trailing_stop=opt_trailing_indent
                )
            elif strategy_type == "MACD":
                opt_macd_fast_min = st.session_state.get('opt_macd_fast_min', 8)
                opt_macd_fast_max = st.session_state.get('opt_macd_fast_max', 16)
                opt_macd_slow_min = st.session_state.get('opt_macd_slow_min', 20)
                opt_macd_slow_max = st.session_state.get('opt_macd_slow_max', 30)
                results = optimizer.optimize_strategy(
                    df,
                    strategy_type="MACD",
                    macd_fast_range=(int(opt_macd_fast_min), int(opt_macd_fast_max)),
                    macd_slow_range=(int(opt_macd_slow_min), int(opt_macd_slow_max)),
                    step=int(opt_step),
                    timeframe=opt_timeframe,
                    commission_percent=commission,
                    take_profit=opt_take_profit,
                    trailing_stop=opt_trailing_indent
                )
            elif strategy_type == "BB":
                opt_bb_period_min = st.session_state.get('opt_bb_period_min', 15)
                opt_bb_period_max = st.session_state.get('opt_bb_period_max', 25)
                results = optimizer.optimize_strategy(
                    df,
                    strategy_type="BB",
                    bb_period_range=(int(opt_bb_period_min), int(opt_bb_period_max)),
                    step=int(opt_step),
                    timeframe=opt_timeframe,
                    commission_percent=commission,
                    take_profit=opt_take_profit,
                    trailing_stop=opt_trailing_indent
                )
            elif strategy_type == "STOCH":
                opt_stoch_k_min = st.session_state.get('opt_stoch_k_min', 10)
                opt_stoch_k_max = st.session_state.get('opt_stoch_k_max', 18)
                results = optimizer.optimize_strategy(
                    df,
                    strategy_type="STOCH",
                    stoch_k_range=(int(opt_stoch_k_min), int(opt_stoch_k_max)),
                    step=int(opt_step),
                    timeframe=opt_timeframe,
                    commission_percent=commission,
                    take_profit=opt_take_profit,
                    trailing_stop=opt_trailing_indent
                )
            elif strategy_type == "PATTERN":
                opt_pattern_type = st.session_state.get('opt_pattern_type', 'candlestick')
                
                pattern_names = []
                if opt_pattern_type == "candlestick" or opt_pattern_type == "combined":
                    if st.session_state.get('opt_pattern_hammer', True):
                        pattern_names.append('hammer')
                    if st.session_state.get('opt_pattern_inv_hammer', True):
                        pattern_names.append('inverted_hammer')
                    if st.session_state.get('opt_pattern_bull_eng', True):
                        pattern_names.append('bullish_engulfing')
                    if st.session_state.get('opt_pattern_bear_eng', True):
                        pattern_names.append('bearish_engulfing')
                    if st.session_state.get('opt_pattern_shooting', True):
                        pattern_names.append('shooting_star')
                    if st.session_state.get('opt_pattern_hanging', True):
                        pattern_names.append('hanging_man')
                
                if opt_pattern_type == "chart" or opt_pattern_type == "combined":
                    if st.session_state.get('opt_pattern_double_bottom', True):
                        pattern_names.append('double_bottom')
                    if st.session_state.get('opt_pattern_double_top', True):
                        pattern_names.append('double_top')
                    if st.session_state.get('opt_pattern_triple_bottom', True):
                        pattern_names.append('triple_bottom')
                    if st.session_state.get('opt_pattern_triple_top', True):
                        pattern_names.append('triple_top')
                
                if not pattern_names:
                    raise ValueError("Не выбран ни один паттерн. Выберите хотя бы один паттерн для тестирования.")
                
                results = optimizer.optimize_strategy(
                    df,
                    strategy_type="PATTERN",
                    pattern_type=opt_pattern_type,
                    pattern_names=pattern_names,
                    step=int(opt_step),
                    timeframe=opt_timeframe,
                    commission_percent=commission,
                    take_profit=opt_take_profit,
                    trailing_stop=opt_trailing_indent
                )
            else:
                raise ValueError(f"Неизвестный тип стратегии: {strategy_type}")
            
            if not results:
                raise ValueError("Оптимизация не вернула результатов. Попробуйте изменить параметры.")
            
            status_text.markdown("🔍 **Поиск лучшей стратегии...**")
            
            best = optimizer.find_best_strategy(results, metric="profit_percent")
            
            status_text.markdown("📊 **Формирование результатов...**")
            
            results_copy = list(results)
            results_df = pd.DataFrame([
                {
                    'strategy_type': r.strategy_type,
                    'ema_short': r.ema_short,
                    'ema_long': r.ema_long,
                    'rsi_period': r.rsi_period,
                    'macd_fast': r.macd_fast,
                    'macd_slow': r.macd_slow,
                    'bb_period': r.bb_period,
                    'stoch_k': r.stoch_k,
                    'pattern_type': r.pattern_type,
                    'pattern_name': r.pattern_name,
                    'timeframe': r.timeframe.name if hasattr(r.timeframe, 'name') else str(r.timeframe),
                    'total_trades': r.total_trades,
                    'winning_trades': r.winning_trades,
                    'losing_trades': r.losing_trades,
                    'total_profit': r.total_profit,
                    'profit_percent': r.profit_percent,
                    'max_drawdown': r.max_drawdown,
                    'sharpe_ratio': r.sharpe_ratio,
                    'win_rate': r.win_rate
                }
                for r in results_copy
            ])
            
            st.session_state.optimization_results = results_df
            st.session_state.best_strategy = best
            st.session_state.optimization_completed = True
            st.session_state.optimization_in_progress = False
            
            status_text.markdown("✅ **Оптимизация завершена!**")
        
        except ValueError as e:
            st.session_state.optimization_in_progress = False
            error_display.markdown(f"❌ **Ошибка валидации:** {e}")
            error_display.warning("💡 Попробуйте изменить параметры оптимизации (диапазоны EMA или период истории)")
            logger.error(f"ValueError: {e}", exc_info=True)
        
        except Exception as e:
            st.session_state.optimization_in_progress = False
            error_display.markdown(f"❌ **Произошла ошибка при оптимизации стратегии**")
            error_display.code(str(e), language="text")
            error_display.info("💡 Проверьте логи в консоли для детальной информации. Убедитесь, что токен TINKOFF_INVEST_TOKEN установлен в .env")
            logger.error(f"Ошибка оптимизации: {e}", exc_info=True)

# Показываем статус оптимизации
if st.session_state.get('optimization_in_progress', False):
    st.info("⏳ Оптимизация выполняется... Пожалуйста, подождите.")

# Показываем результаты оптимизации
if 'best_strategy' in st.session_state and st.session_state.best_strategy and st.session_state.get('optimization_completed', False):
    st.success("✅ Оптимизация завершена успешно!")
    best = st.session_state.best_strategy
    
    st.markdown("### 🏆 Лучшая стратегия")
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    
    with col_b1:
        st.metric("Тип стратегии", best.strategy_type)
        timeframe_name = best.timeframe.name if hasattr(best.timeframe, 'name') else str(best.timeframe)
        timeframe_display = {
            'CANDLE_INTERVAL_1_MIN': '1 минута',
            'CANDLE_INTERVAL_5_MIN': '5 минут',
            'CANDLE_INTERVAL_15_MIN': '15 минут',
            'CANDLE_INTERVAL_30_MIN': '30 минут',
            'CANDLE_INTERVAL_HOUR': '1 час',
            'CANDLE_INTERVAL_DAY': '1 день',
        }.get(timeframe_name, timeframe_name)
        st.metric("Таймфрейм", timeframe_display)
        
        if best.strategy_type == "EMA":
            st.metric("EMA Short", best.ema_short)
            st.metric("EMA Long", best.ema_long)
        elif best.strategy_type == "RSI":
            st.metric("RSI Period", best.rsi_period)
        elif best.strategy_type == "MACD":
            st.metric("MACD Fast", best.macd_fast)
            st.metric("MACD Slow", best.macd_slow)
        elif best.strategy_type == "BB":
            st.metric("BB Period", best.bb_period)
        elif best.strategy_type == "STOCH":
            st.metric("Stochastic K", best.stoch_k)
        elif best.strategy_type == "PATTERN":
            st.metric("Тип паттернов", best.pattern_type)
            if best.pattern_name:
                st.metric("Паттерны", best.pattern_name.replace(',', ', '))
    
    with col_b2:
        st.metric("Всего сделок", best.total_trades)
        st.metric("Винрейт", f"{best.win_rate:.2f}%")
    
    with col_b3:
        st.metric("Прибыль", f"{best.total_profit:.2f} ₽", delta=f"{best.profit_percent:.2f}%")
        st.metric("Прибыльных", best.winning_trades)
    
    with col_b4:
        st.metric("Убыточных", best.losing_trades)
        st.metric("Sharpe Ratio", f"{best.sharpe_ratio:.2f}")
    
    # Кнопка применения лучшей стратегии
    if st.button("✅ Применить лучшую стратегию", use_container_width=True, key="apply_best_strategy"):
        Config.STRATEGY_TYPE = best.strategy_type
        
        if best.strategy_type == "EMA":
            Config.EMA_SHORT = best.ema_short
            Config.EMA_LONG = best.ema_long
            strategy_desc = f"EMA({best.ema_short}, {best.ema_long})"
        elif best.strategy_type == "RSI":
            Config.RSI_PERIOD = best.rsi_period
            strategy_desc = f"RSI(period={best.rsi_period})"
        elif best.strategy_type == "MACD":
            Config.MACD_FAST = best.macd_fast
            Config.MACD_SLOW = best.macd_slow
            strategy_desc = f"MACD(fast={best.macd_fast}, slow={best.macd_slow})"
        elif best.strategy_type == "BB":
            Config.BB_PERIOD = best.bb_period
            strategy_desc = f"BB(period={best.bb_period})"
        elif best.strategy_type == "STOCH":
            Config.STOCH_K = best.stoch_k
            strategy_desc = f"Stochastic(K={best.stoch_k})"
        elif best.strategy_type == "PATTERN":
            Config.PATTERN_TYPE = best.pattern_type
            Config.PATTERN_NAMES = best.pattern_name.split(',') if best.pattern_name else []
            pattern_names_display = best.pattern_name.replace(',', ', ') if best.pattern_name else "all"
            strategy_desc = f"Patterns({best.pattern_type}: {pattern_names_display})"
        else:
            strategy_desc = best.strategy_type
        
        Config.TIMEFRAME = best.timeframe
        st.success(f"Стратегия обновлена: {strategy_desc} на таймфрейме {timeframe_display}")
    
    st.divider()
    
    # Таблица топ-10 стратегий
    if 'optimization_results' in st.session_state:
        st.markdown("### 📈 Топ-10 стратегий по прибыльности")
        top_10 = st.session_state.optimization_results.nlargest(10, 'profit_percent')
        
        display_df = top_10.copy()
        if 'timeframe' in display_df.columns:
            timeframe_map = {
                'CANDLE_INTERVAL_1_MIN': '1 минута',
                'CANDLE_INTERVAL_5_MIN': '5 минут',
                'CANDLE_INTERVAL_15_MIN': '15 минут',
                'CANDLE_INTERVAL_30_MIN': '30 минут',
                'CANDLE_INTERVAL_HOUR': '1 час',
                'CANDLE_INTERVAL_DAY': '1 день',
            }
            display_df['timeframe'] = display_df['timeframe'].map(
                lambda x: timeframe_map.get(x, x)
            )
        
            base_columns = ['strategy_type', 'timeframe', 'total_trades', 
                          'winning_trades', 'profit_percent', 'win_rate', 'sharpe_ratio']
            
            if 'strategy_type' in display_df.columns:
                strategy_types = display_df['strategy_type'].unique()
                if 'EMA' in strategy_types:
                    base_columns.extend(['ema_short', 'ema_long'])
                if 'RSI' in strategy_types:
                    base_columns.append('rsi_period')
                if 'MACD' in strategy_types:
                    base_columns.extend(['macd_fast', 'macd_slow'])
                if 'BB' in strategy_types:
                    base_columns.append('bb_period')
                if 'STOCH' in strategy_types:
                    base_columns.append('stoch_k')
                if 'PATTERN' in strategy_types:
                    base_columns.extend(['pattern_type', 'pattern_name'])
            
            columns_to_show = [c for c in base_columns if c in display_df.columns]
        
        st.dataframe(
            display_df[columns_to_show].style.format({
                'profit_percent': '{:.2f}%',
                'win_rate': '{:.2f}%',
                'sharpe_ratio': '{:.2f}'
            }),
            width='stretch',
            height=400
        )
    
    # График распределения прибыльности
    if 'optimization_results' in st.session_state:
        st.markdown("### 📊 Распределение прибыльности стратегий")
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=st.session_state.optimization_results['profit_percent'],
            nbinsx=30,
            name='Распределение прибыльности'
        ))
        fig_dist.update_layout(
            title="Распределение прибыльности всех протестированных стратегий",
            xaxis_title="Прибыльность (%)",
            yaxis_title="Количество стратегий",
            template="plotly_dark"
        )
        st.plotly_chart(fig_dist, width='stretch')
