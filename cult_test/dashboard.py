import streamlit as st
import pandas as pd
import time
import logging
import sys
import os
import traceback
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Логика запуска и логирования ---

class StreamlitLogger(logging.Handler):
    def __init__(self):
        super().__init__()
        # ВАЖНО: больше не храним ссылку на виджет!
        # Храним логи только в памяти или в файле
        self.log_buffer = []

    def emit(self, record):
        try:
            msg = self.format(record)
            # Добавляем эмодзи для красоты логов
            if "BUY" in msg: msg = "🟢 " + msg
            elif "SELL" in msg: msg = "🔴 " + msg
            elif "HOLD" in msg: msg = "⚪ " + msg
            
            self.log_buffer.append(msg)
            if len(self.log_buffer) > 50: # Держим последние 50 строк
                self.log_buffer.pop(0)
                
            # Записываем в файл для надежности
            try:
                with open("bot_execution.log", "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except:
                pass
                
        except Exception:
            pass

# Настройка логгера (выполняется один раз при старте или перезагрузке)
root_logger = logging.getLogger()
logger = logging.getLogger(__name__)  # Инициализируем logger
st_handler = None

# Ищем существующий по имени, а не по типу (так как класс пересоздается при перезагрузке)
for h in root_logger.handlers:
    if getattr(h, 'name', '') == 'StreamlitLogger':
        st_handler = h
        break

if not st_handler:
    st_handler = StreamlitLogger()
    st_handler.name = 'StreamlitLogger' # Задаем имя для поиска
    st_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
    root_logger.addHandler(st_handler)
    root_logger.setLevel(logging.INFO)
else:
    # Если хендлер найден, обновляем его (на случай если код класса изменился, 
    # но тут мы просто используем старый экземпляр, что безопаснее)
    pass

# Очищаем лог файл при старте сессии
if 'log_cleared' not in st.session_state:
    try:
        with open("bot_execution.log", "w", encoding="utf-8") as f:
            f.write("--- Start Log ---\n")
        st.session_state.log_cleared = True
    except:
        pass

# Добавляем корень проекта в PYTHONPATH для корректных импортов
# Определяем корень проекта (директория, содержащая cult_test)
project_root = None

# Стратегия 1: Определяем по расположению файла dashboard.py
if __file__:
    try:
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)
        # Если файл в cult_test/, то поднимаемся на уровень выше
        if os.path.basename(current_dir) == 'cult_test':
            candidate_root = os.path.dirname(current_dir)
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

# Настройка страницы (должна быть первой командой Streamlit)
st.set_page_config(page_title="Trading Bot Dashboard", layout="wide", page_icon="📈")

# Импортируем вашего бота и конфиг
try:
    from cult_test.cult_main import TradingBot, Config
    from cult_test.strategy_optimizer import StrategyOptimizer
    # FIXED: Use t_tech instead of tinkoff
    from t_tech.invest import Client, CandleInterval, InstrumentIdType
    from t_tech.invest.utils import now, quotation_to_decimal
    from datetime import timedelta
except ImportError as e:
    st.error("❌ Ошибка импорта модулей")
    with st.expander("🔍 Детали ошибки", expanded=True):
        st.code(f"Ошибка: {e}", language="text")
        st.write("**Информация о путях:**")
        st.write(f"- Корень проекта: `{project_root}`")
        st.write(f"- Текущая директория: `{os.getcwd()}`")
        st.write(f"- Файл dashboard.py: `{__file__}`")
        st.write(f"- PYTHONPATH (первые 3): `{sys.path[:3]}`")
        
        # Проверка существования файлов
        st.write("**Проверка файлов:**")
        cult_test_path = os.path.join(project_root, 'cult_test') if project_root else None
        if cult_test_path and os.path.exists(cult_test_path):
            st.success(f"✅ Директория cult_test найдена: `{cult_test_path}`")
            files = os.listdir(cult_test_path)
            st.write(f"Файлы в cult_test: {', '.join(files)}")
        else:
            st.error(f"❌ Директория cult_test не найдена в `{project_root}`")
    
    st.error("**Решение:**")
    st.write("1. Убедитесь, что запускаете streamlit из корня проекта:")
    st.code("streamlit run cult_test/dashboard.py", language="bash")
    st.write("2. Или используйте Docker:")
    st.code("docker compose up dashboard", language="bash")
    st.write("3. Проверьте установку пакета:")
    st.code("pip install t-tech-investments", language="bash")
    st.stop()

st.title("📈 Панель управления и Аналитика")

# --- Боковая панель с настройками ---
# Навигация по страницам
# Удаляем из сайдбара и делаем горизонтальное меню сверху
# Используем columns для имитации вкладок
st.sidebar.header("⚙️ Настройки")

# Кнопки навигации в верхней части
nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 4])

with nav_col1:
    if st.button("🤖 Торговый Бот", use_container_width=True, type="primary" if st.session_state.get('active_page') != 'optimization' else "secondary"):
        st.session_state.active_page = 'bot'
        st.rerun()

with nav_col2:
    if st.button("🔬 Оптимизация", use_container_width=True, type="primary" if st.session_state.get('active_page') == 'optimization' else "secondary"):
        st.session_state.active_page = 'optimization'
        st.rerun()

# Инициализация активной страницы
if 'active_page' not in st.session_state:
    st.session_state.active_page = 'bot'

# Сохраняем ticker в session_state для использования на других страницах
if 'ticker' not in st.session_state:
    st.session_state.ticker = Config.TICKER

# Параметры инструмента (в сайдбаре)
ticker = st.sidebar.text_input("Тикер", value=st.session_state.ticker)
st.session_state.ticker = ticker
class_code = st.sidebar.text_input("Class Code", value=Config.CLASS_CODE)

# Определяем текущую страницу
page = "🔬 Оптимизация Стратегий" if st.session_state.active_page == 'optimization' else "🤖 Торговый Бот"

# Параметры Стратегии (в зависимости от выбранного типа)
st.sidebar.subheader("Параметры текущей стратегии")

# Список доступных стратегий
AVAILABLE_STRATEGIES = ["EMA", "RSI", "MACD", "BB", "STOCH", "PATTERN"]

# Текущая активная стратегия из конфига
active_strategy = getattr(Config, 'STRATEGY_TYPE', 'EMA')

# Индекс для selectbox
try:
    strategy_index = AVAILABLE_STRATEGIES.index(active_strategy)
except ValueError:
    strategy_index = 0

# Выбор стратегии (саджест)
selected_strategy = st.sidebar.selectbox(
    "Выберите стратегию",
    options=AVAILABLE_STRATEGIES,
    index=strategy_index,
    help="Выберите стратегию, которую будет использовать бот для торговли"
)

# Если пользователь изменил стратегию в selectbox - обновляем конфиг
if selected_strategy != active_strategy:
    Config.STRATEGY_TYPE = selected_strategy
    # Можно сбросить параметры на дефолтные для новой стратегии, если нужно
    # Но пока оставляем текущие значения из конфига (или дефолты при чтении)
    st.rerun() # Перезагружаем страницу, чтобы обновить отображаемые параметры ниже

# Отображаем параметры для ВЫБРАННОЙ стратегии
if selected_strategy == "EMA":
    ema_short_val = st.sidebar.number_input("EMA Fast", min_value=1, value=Config.EMA_SHORT)
    ema_long_val = st.sidebar.number_input("EMA Slow", min_value=1, value=Config.EMA_LONG)
    Config.EMA_SHORT = int(ema_short_val)
    Config.EMA_LONG = int(ema_long_val)
elif selected_strategy == "RSI":
    rsi_period_val = st.sidebar.number_input("RSI Period", min_value=2, value=getattr(Config, 'RSI_PERIOD', 14))
    Config.RSI_PERIOD = int(rsi_period_val)
elif selected_strategy == "MACD":
    macd_fast = st.sidebar.number_input("MACD Fast", min_value=2, value=getattr(Config, 'MACD_FAST', 12))
    macd_slow = st.sidebar.number_input("MACD Slow", min_value=2, value=getattr(Config, 'MACD_SLOW', 26))
    Config.MACD_FAST = int(macd_fast)
    Config.MACD_SLOW = int(macd_slow)
elif selected_strategy == "BB":
    bb_period = st.sidebar.number_input("BB Period", min_value=2, value=getattr(Config, 'BB_PERIOD', 20))
    Config.BB_PERIOD = int(bb_period)
elif selected_strategy == "STOCH":
    stoch_k = st.sidebar.number_input("Stochastic K", min_value=2, value=getattr(Config, 'STOCH_K', 14))
    Config.STOCH_K = int(stoch_k)
elif selected_strategy == "PATTERN":
    st.sidebar.info("Паттерны выбираются через оптимизацию или редактирование кода (пока)")
    # Можно добавить мультиселект для паттернов
    current_patterns = getattr(Config, 'PATTERN_NAMES', [])
    # Доступные паттерны
    all_patterns = ['hammer', 'inverted_hammer', 'bullish_engulfing', 'bearish_engulfing', 'shooting_star', 'hanging_man', 'double_bottom', 'double_top', 'triple_bottom', 'triple_top']
    selected_patterns = st.sidebar.multiselect("Активные паттерны", options=all_patterns, default=[p for p in current_patterns if p in all_patterns])
    Config.PATTERN_NAMES = selected_patterns
    Config.PATTERN_TYPE = "combined" # Принудительно ставим combined если выбираем руками

# Добавить остальные типы по аналогии, если нужно редактировать их вручную

# Параметры Риск-менеджмента
st.sidebar.subheader("Риск-менеджмент")
take_profit = st.sidebar.number_input("Take Profit (%)", min_value=0.1, value=Config.TAKE_PROFIT * 100, step=0.1)
trailing_indent = st.sidebar.number_input("Trailing Indent (%)", min_value=0.01, value=Config.TRAILING_INDENT * 100, step=0.01)

# Обновление глобальной конфигурации
Config.TICKER = ticker
Config.CLASS_CODE = class_code
# EMA параметры обновляются выше в блоке if
Config.TAKE_PROFIT = take_profit / 100
Config.TRAILING_INDENT = trailing_indent / 100

# --- Функция загрузки данных для графика ---
# ВАЖНО: функция должна быть чистой (pure function) - без использования элементов Streamlit внутри
@st.cache_data(ttl=60, show_spinner=True)  # Кешируем данные на 60 секунд, показываем спиннер при загрузке
def load_market_data(ticker, short_period, long_period, token, class_code, timeframe, days):
    """Загружает свечи и считает индикаторы для визуализации
    
    Все параметры передаются явно, чтобы избежать проблем с кешированием Streamlit
    """
    try:
        # Используем клиент только для чтения данных
        with Client(token) as client:
            # Настраиваем инструмент по тикеру
            item = client.instruments.get_instrument_by(
                id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER,
                id=ticker,
                class_code=class_code,
            ).instrument
            
            # Загружаем свечи (берем больше данных для корректного расчета EMA)
            candles = client.get_all_candles(
                instrument_id=item.uid,
                from_=now() - timedelta(days=days),
                to=now(),
                interval=timeframe,
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
                return None, "Нет данных по свечам"
            
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
                
            # Расчет индикаторов (дублируем логику бота для визуализации)
            df['ema_short'] = df['close'].ewm(span=short_period, adjust=False).mean()
            df['ema_long'] = df['close'].ewm(span=long_period, adjust=False).mean()
            
            return df, None
    except Exception as e:
        # Используем стандартный logging вместо глобального logger, чтобы избежать проблем с кешированием
        # logger может быть связан с элементами Streamlit через StreamlitLogger
        import logging as std_logging
        std_logging.error(f"Ошибка загрузки данных: {e}", exc_info=True)
        return None, str(e)

# --- Основной экран ---

# Выбор параметров отображения над графиком
col_tf, col_days = st.columns([1, 1])

with col_tf:
    # Выбор таймфрейма
    timeframe_map = {
        "1 минута": CandleInterval.CANDLE_INTERVAL_1_MIN,
        "5 минут": CandleInterval.CANDLE_INTERVAL_5_MIN,
        "15 минут": CandleInterval.CANDLE_INTERVAL_15_MIN,
        "30 минут": CandleInterval.CANDLE_INTERVAL_30_MIN,
        "1 час": CandleInterval.CANDLE_INTERVAL_HOUR,
        "1 день": CandleInterval.CANDLE_INTERVAL_DAY,
    }

    # Определяем индекс по умолчанию (30 мин)
    default_tf_index = 3
    if Config.TIMEFRAME in timeframe_map.values():
        default_tf_index = list(timeframe_map.values()).index(Config.TIMEFRAME)

    selected_tf_name = st.selectbox(
        "Таймфрейм графика", 
        options=list(timeframe_map.keys()), 
        index=default_tf_index,
        key="graph_timeframe"
    )
    Config.TIMEFRAME = timeframe_map[selected_tf_name]

with col_days:
    # Выбор периода истории
    days_back = st.number_input("Дней истории", min_value=1, max_value=365, value=60, step=1, key="graph_days")

# 1. Секция Графика
st.subheader(f"График {ticker} ({selected_tf_name})")

# Кнопка обновления графика
if st.button("🔄 Обновить данные рынка", help="Сбросить кеш и загрузить свежие данные", key="refresh_market_data"):
    load_market_data.clear()
    st.info("✅ Кеш очищен. Данные будут обновлены при следующей загрузке.")
    # НЕ используем rerun - это может вызывать белый экран

# Загрузка данных с обработкой ошибок
# НЕ используем st.spinner здесь, так как функция кешируется и это вызывает конфликт
df = None
error = None
try:
    df, error = load_market_data(
        ticker, 
        Config.EMA_SHORT, 
        Config.EMA_LONG,
        Config.TOKEN,
        Config.CLASS_CODE,
        Config.TIMEFRAME,
        days_back
    )
except Exception as e:
    error = f"Ошибка при загрузке данных: {e}"
    logger.error(f"Ошибка загрузки рыночных данных: {e}", exc_info=True)

if error:
    st.error(f"Ошибка загрузки данных: {error}")
elif df is not None:
    # Отрисовка с Plotly
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    # Свечи
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Цена'
    ))

    # EMA Short
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['ema_short'],
        line=dict(color='orange', width=1.5),
        name=f'EMA {Config.EMA_SHORT}'
    ))

    # EMA Long
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['ema_long'],
        line=dict(color='blue', width=1.5),
        name=f'EMA {Config.EMA_LONG}'
    ))

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Показываем последние значения
    last_candle = df.iloc[-1]
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Цена закрытия", f"{last_candle['close']:.2f}")
    col_m2.metric(f"EMA {Config.EMA_SHORT}", f"{last_candle['ema_short']:.2f}")
    col_m3.metric(f"EMA {Config.EMA_LONG}", f"{last_candle['ema_long']:.2f}", 
                  delta=f"{last_candle['ema_short'] - last_candle['ema_long']:.2f}")

# --- Секция AI-Аналитика ---
st.divider()
st.subheader("🧙‍♂️ AI-Аналитик")

with st.expander("🔍 Найти лучшую стратегию для этого графика", expanded=False):
    st.write("Робот автоматически проверит все типы стратегий (EMA, RSI, MACD, BB, Stochastic, Patterns) на текущих данных и найдет ту, которая дала бы максимальную прибыль.")
    
    if st.button("✨ Найти лучшую стратегию", type="primary", use_container_width=True):
        if df is None:
            st.error("Сначала загрузите данные графика")
        else:
            with st.spinner("Анализ всех стратегий... Это может занять около 30 секунд."):
                try:
                    optimizer = StrategyOptimizer()
                    
                    # Общие настройки для быстрого поиска
                    commission = 0.05
                    # Используем текущие настройки риска или дефолтные
                    tp = Config.TAKE_PROFIT
                    sl = Config.TRAILING_INDENT
                    
                    all_results = []
                    
                    # 1. EMA
                    res_ema = optimizer.optimize_strategy(
                        df, strategy_type="EMA",
                        ema_short_range=(5, 50), ema_long_range=(50, 200),
                        step=10, timeframe=Config.TIMEFRAME, commission_percent=commission, take_profit=tp, trailing_stop=sl
                    )
                    all_results.extend(res_ema)
                    
                    # 2. RSI
                    res_rsi = optimizer.optimize_strategy(
                        df, strategy_type="RSI",
                        rsi_period_range=(5, 25), step=5,
                        timeframe=Config.TIMEFRAME, commission_percent=commission, take_profit=tp, trailing_stop=sl
                    )
                    all_results.extend(res_rsi)
                    
                    # 3. MACD
                    res_macd = optimizer.optimize_strategy(
                        df, strategy_type="MACD",
                        macd_fast_range=(8, 20), macd_slow_range=(20, 40), step=4,
                        timeframe=Config.TIMEFRAME, commission_percent=commission, take_profit=tp, trailing_stop=sl
                    )
                    all_results.extend(res_macd)
                    
                    # 4. Bollinger Bands
                    res_bb = optimizer.optimize_strategy(
                        df, strategy_type="BB",
                        bb_period_range=(10, 30), step=5,
                        timeframe=Config.TIMEFRAME, commission_percent=commission, take_profit=tp, trailing_stop=sl
                    )
                    all_results.extend(res_bb)
                    
                    # 5. Stochastic
                    res_stoch = optimizer.optimize_strategy(
                        df, strategy_type="STOCH",
                        stoch_k_range=(10, 25), step=5,
                        timeframe=Config.TIMEFRAME, commission_percent=commission, take_profit=tp, trailing_stop=sl
                    )
                    all_results.extend(res_stoch)
                    
                    # 6. Patterns
                    # Проверяем базовые свечные паттерны
                    res_pattern = optimizer.optimize_strategy(
                        df, strategy_type="PATTERN",
                        pattern_type="candlestick",
                        pattern_names=['hammer', 'bullish_engulfing', 'shooting_star', 'bearish_engulfing'],
                        step=1,
                        timeframe=Config.TIMEFRAME, commission_percent=commission, take_profit=tp, trailing_stop=sl
                    )
                    all_results.extend(res_pattern)
                    
                    # Поиск победителя
                    best_ai = optimizer.find_best_strategy(all_results, metric="profit_percent")
                    
                    if best_ai:
                        st.success(f"🎉 Найдена лучшая стратегия: **{best_ai.strategy_type}** с прибылью **{best_ai.profit_percent:.2f}%**")
                        
                        col_ai1, col_ai2, col_ai3 = st.columns(3)
                        col_ai1.metric("Винрейт", f"{best_ai.win_rate:.2f}%")
                        col_ai1.metric("Сделок", best_ai.total_trades)
                        
                        col_ai2.metric("Прибыль", f"{best_ai.total_profit:.2f} ₽")
                        col_ai2.metric("Sharpe", f"{best_ai.sharpe_ratio:.2f}")
                        
                        col_ai3.write("**Параметры:**")
                        if best_ai.strategy_type == "EMA":
                            col_ai3.write(f"Short: {best_ai.ema_short}, Long: {best_ai.ema_long}")
                        elif best_ai.strategy_type == "RSI":
                            col_ai3.write(f"Period: {best_ai.rsi_period}")
                        elif best_ai.strategy_type == "MACD":
                            col_ai3.write(f"Fast: {best_ai.macd_fast}, Slow: {best_ai.macd_slow}")
                        elif best_ai.strategy_type == "BB":
                            col_ai3.write(f"Period: {best_ai.bb_period}")
                        elif best_ai.strategy_type == "STOCH":
                            col_ai3.write(f"K: {best_ai.stoch_k}")
                        elif best_ai.strategy_type == "PATTERN":
                            col_ai3.write(f"Patterns: {best_ai.pattern_name}")
                            
                        if st.button("✅ Применить эту стратегию"):
                            Config.STRATEGY_TYPE = best_ai.strategy_type
                            if best_ai.strategy_type == "EMA":
                                Config.EMA_SHORT = best_ai.ema_short
                                Config.EMA_LONG = best_ai.ema_long
                            elif best_ai.strategy_type == "RSI":
                                Config.RSI_PERIOD = best_ai.rsi_period
                            elif best_ai.strategy_type == "MACD":
                                Config.MACD_FAST = best_ai.macd_fast
                                Config.MACD_SLOW = best_ai.macd_slow
                            elif best_ai.strategy_type == "BB":
                                Config.BB_PERIOD = best_ai.bb_period
                            elif best_ai.strategy_type == "STOCH":
                                Config.STOCH_K = best_ai.stoch_k
                            elif best_ai.strategy_type == "PATTERN":
                                Config.PATTERN_TYPE = best_ai.pattern_type
                                Config.PATTERN_NAMES = best_ai.pattern_name.split(',') if best_ai.pattern_name else []
                            
                            st.rerun()
                    else:
                        st.warning("Не удалось найти прибыльную стратегию на этом участке.")
                        
                except Exception as e:
                    st.error(f"Ошибка анализа: {e}")
                    logger.error(f"AI Analysis error: {e}", exc_info=True)

st.divider()

# Навигация между страницами
if page == "🔬 Оптимизация Стратегий":
    # Загружаем страницу оптимизации из модуля
    import importlib.util
    import sys
    
    # Путь к файлу оптимизации (теперь в panels, чтобы скрыть из нативного меню)
    optimization_path = os.path.join(project_root, 'cult_test', 'panels', 'optimization.py')
    
    if os.path.exists(optimization_path):
        # Используем runpy для исполнения файла в текущем контексте
        import runpy
        # Передаем глобальные переменные, чтобы модуль видел config и другие
        runpy.run_path(optimization_path, init_globals=globals())
    else:
        st.error(f"Файл оптимизации не найден: {optimization_path}")
else:
    # Страница торгового бота
    # Устанавливаем флаг активной страницы
    st.session_state.active_tab = "bot"
    
    # Секция Управления Ботом
    st.subheader("🤖 Управление ботом")
    
    col_ctrl, col_log = st.columns([1, 2])
    
    with col_ctrl:
        if 'running' not in st.session_state:
            st.session_state.running = False

        start_btn = st.button("🚀 Запустить проверку", type="primary", use_container_width=True, key="bot_start_btn")
        loop_check = st.checkbox("Авто-режим (каждые 60 сек)", key="bot_loop_check")
        
        if start_btn:
            st.session_state.running = True
                
    with col_log:
        st.subheader("Лог операций")
        # Показываем логи ПОСЛЕ выполнения, читая из файла
        if os.path.exists("bot_execution.log"):
            with open("bot_execution.log", "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            # Разворачиваем порядок строк (новые сверху)
            lines = [line.strip() for line in lines if line.strip()]
            lines.reverse()
            logs = "\n".join(lines)
            
            # Создаем прокручиваемый контейнер с фиксированной высотой
            with st.container(height=400, border=True):
                st.code(logs, language="text")
        else:
            st.info("Логов пока нет")
    
    # Запуск логики бота
    if st.session_state.get('running', False):
        try:
            # Очищаем файл логов перед запуском
            with open("bot_execution.log", "w", encoding="utf-8") as f:
                f.write("--- Запуск бота ---\n")
            
            # Очищаем буфер логгера
            if st_handler:
                st_handler.log_buffer = []
                
            status_placeholder = st.empty()
            status_placeholder.info("🚀 Анализ рынка и проверка условий... (Логи появятся после завершения)")
            
            bot = TradingBot()
            bot.run()
            
            status_placeholder.success("✅ Выполнено")
            
            if not loop_check:
                st.session_state.running = False
                st.success("Проверка завершена")
            else:
                # Для авто-режима НЕ используем rerun, чтобы не ломать верстку
                # Вместо этого просто ждем и обновляем состояние
                st.info("⏳ Авто-режим активен. Следующая проверка через 60 секунд...")
                # Используем time.sleep только для демонстрации, но не rerun
                # Пользователь может вручную обновить страницу или запустить снова
                st.session_state.running = False
                st.warning("💡 Для непрерывной работы используйте внешний скрипт или планировщик задач")
                    
        except Exception as e:
            st.error(f"Ошибка исполнения: {e}")
            logger.error(f"Ошибка исполнения бота: {e}", exc_info=True)
            st.session_state.running = False
            # Не вызываем rerun при ошибке
