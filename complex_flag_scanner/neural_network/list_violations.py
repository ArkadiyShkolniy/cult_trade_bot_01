#!/usr/bin/env python3
"""
Выводит список записей с нарушениями геометрических ограничений для переразметки
"""

import pandas as pd
from pathlib import Path
from check_annotations_geometry import check_long_constraints, check_short_constraints

def main():
    annotations_file = Path("neural_network/data/annotations.csv")
    
    if not annotations_file.exists():
        print("❌ Файл аннотаций не найден!")
        return
    
    df = pd.read_csv(annotations_file)
    df_valid = df.dropna(subset=['t0_price', 't1_price', 't2_price', 't3_price', 't4_price'])
    
    violations_list = []
    
    for idx, row in df_valid.iterrows():
        T0 = row['t0_price']
        T1 = row['t1_price']
        T2 = row['t2_price']
        T3 = row['t3_price']
        T4 = row['t4_price']
        label = row['label']
        timeframe = row.get('timeframe', '1h')
        
        violations = []
        if label == 1:  # LONG
            violations = check_long_constraints(T0, T1, T2, T3, T4, timeframe)
        elif label == 2:  # SHORT
            violations = check_short_constraints(T0, T1, T2, T3, T4, timeframe)
        
        if violations:
            violations_list.append({
                'idx': idx,
                'file': row.get('file', 'unknown'),
                'ticker': row.get('ticker', 'unknown'),
                'timeframe': timeframe,
                'label': label,
                'T0_idx': row['t0_idx'],
                'T0_price': T0,
                'T1_idx': row['t1_idx'],
                'T1_price': T1,
                'T2_idx': row['t2_idx'],
                'T2_price': T2,
                'T3_idx': row['t3_idx'],
                'T3_price': T3,
                'T4_idx': row['t4_idx'],
                'T4_price': T4,
                'violations': violations
            })
    
    print("=" * 80)
    print("📋 СПИСОК ЗАПИСЕЙ С НАРУШЕНИЯМИ ДЛЯ ПЕРЕРАЗМЕТКИ")
    print("=" * 80)
    print()
    print(f"Всего найдено: {len(violations_list)} записей")
    print()
    
    for i, viol in enumerate(violations_list, 1):
        pattern_type = "LONG (бычий)" if viol['label'] == 1 else "SHORT (медвежий)"
        print(f"{i}. {pattern_type} - {viol['ticker']} ({viol['timeframe']})")
        print(f"   Файл: {viol['file']}")
        print(f"   Текущие координаты:")
        print(f"      T0: индекс {viol['T0_idx']:.0f}, цена {viol['T0_price']:.2f}")
        print(f"      T1: индекс {viol['T1_idx']:.0f}, цена {viol['T1_price']:.2f}")
        print(f"      T2: индекс {viol['T2_idx']:.0f}, цена {viol['T2_price']:.2f}")
        print(f"      T3: индекс {viol['T3_idx']:.0f}, цена {viol['T3_price']:.2f}")
        print(f"      T4: индекс {viol['T4_idx']:.0f}, цена {viol['T4_price']:.2f}")
        print(f"   Нарушения:")
        for v in viol['violations']:
            print(f"      • {v}")
        print()
    
    # Сохраняем список в JSON для дашборда
    import json
    violations_json = []
    for viol in violations_list:
        violations_json.append({
            'file': viol['file'],
            'ticker': viol['ticker'],
            'timeframe': viol['timeframe'],
            'label': viol['label'],
            'current_points': {
                'T0': {'idx': float(viol['T0_idx']), 'price': float(viol['T0_price'])},
                'T1': {'idx': float(viol['T1_idx']), 'price': float(viol['T1_price'])},
                'T2': {'idx': float(viol['T2_idx']), 'price': float(viol['T2_price'])},
                'T3': {'idx': float(viol['T3_idx']), 'price': float(viol['T3_price'])},
                'T4': {'idx': float(viol['T4_idx']), 'price': float(viol['T4_price'])}
            },
            'violations': viol['violations']
        })
    
    violations_file = Path("neural_network/data/violations_list.json")
    with open(violations_file, 'w', encoding='utf-8') as f:
        json.dump(violations_json, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Список сохранен в: {violations_file}")
    print()
    print("💡 Используйте дашборд разметки для исправления этих записей")
    print()

if __name__ == "__main__":
    main()

