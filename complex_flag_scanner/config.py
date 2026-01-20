from t_tech.invest import CandleInterval

TIMEFRAMES = {
    '5m': {
        'interval': CandleInterval.CANDLE_INTERVAL_5_MIN,
        'days_back': 2,      # Для 5 минут достаточно 2 дней
        'title': '5 Минут'
    },
    '1h': {
        'interval': CandleInterval.CANDLE_INTERVAL_HOUR,
        'days_back': 60,     # Для часа берем 2 месяца
        'title': '1 Час'
    },
    '1d': {
        'interval': CandleInterval.CANDLE_INTERVAL_DAY,
        'days_back': 365,    # Для дневки нужен год
        'title': '1 День'
    }
}

