"""
=========================================
NLP 數據驗證 + 統計儀表板
=========================================

用途：
1. 驗證收集到嘅 CSV 數據完整性
2. 顯示即時統計（每條內容、平台分佈、主題分佈）
3. 預測分析結果

使用：
  python nlp_validator.py
"""

import pandas as pd
import sys
from pathlib import Path


def validate_csv(csv_path):
    """驗證 CSV 完整性"""
    print("="*60)
    print("📋 CSV 數據驗證")
    print("="*60)

    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"❌ 找不到檔案: {csv_path}")
        return None
    except Exception as e:
        print(f"❌ 讀取失敗: {e}")
        return None

    print(f"\n✓ 成功讀取 {len(df)} 條內容")

    # 必要欄位
    required_cols = ['Post_ID', 'Date', 'Platform', 'Author_Handle', 'Title', 'Content']

    # 1. 欄位完整性
    print("\n【1. 欄位完整性】")
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"   ❌ 缺失欄位: {missing_cols}")
    else:
        print(f"   ✓ 所有必要欄位齊全")

    # 2. 缺失值檢查
    print("\n【2. 缺失值統計】")
    for col in required_cols:
        if col in df.columns:
            missing = df[col].isna().sum() + (df[col] == '').sum()
            pct = missing / len(df) * 100
            status = "✓" if pct < 5 else "⚠️" if pct < 20 else "❌"
            print(f"   {status} {col}: {missing}/{len(df)} 缺失 ({pct:.1f}%)")

    # 3. 重複檢查
    print("\n【3. 重複檢查】")
    duplicates = df.duplicated(subset=['Title', 'Content']).sum()
    if duplicates > 0:
        print(f"   ⚠️  發現 {duplicates} 條重複內容")
    else:
        print(f"   ✓ 無重複內容")

    # 4. 數據類型驗證
    print("\n【4. 數據類型】")
    if 'Likes' in df.columns:
        df['Likes_Num'] = pd.to_numeric(df['Likes'], errors='coerce')
        invalid_likes = df['Likes_Num'].isna().sum()
        print(f"   {'✓' if invalid_likes == 0 else '⚠️'} Likes: {invalid_likes} 條無效")

    if 'Date' in df.columns:
        df['Date_parsed'] = pd.to_datetime(df['Date'], errors='coerce')
        invalid_dates = df['Date_parsed'].isna().sum()
        print(f"   {'✓' if invalid_dates == 0 else '⚠️'} Date: {invalid_dates} 條無效")

    return df


def show_dashboard(df):
    """顯示即時統計儀表板"""
    print("\n" + "="*60)
    print("📊 即時統計儀表板")
    print("="*60)

    # 1. 總體統計
    print(f"\n【總體】")
    print(f"   總內容數: {len(df)}")
    print(f"   涵蓋平台數: {df['Platform'].nunique()}")
    print(f"   涵蓋作者數: {df['Author_Handle'].nunique()}")
    print(f"   最早日期: {df['Date'].min()}")
    print(f"   最晚日期: {df['Date'].max()}")

    # 2. 平台分佈
    print(f"\n【平台分佈】")
    platform_counts = df['Platform'].value_counts()
    for platform, count in platform_counts.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"   {platform:<15} {count:>3} ({pct:>5.1f}%) {bar}")

    # 3. 互動數據
    if 'Likes' in df.columns:
        print(f"\n【互動數據】")
        df['Likes_Num'] = pd.to_numeric(df['Likes'], errors='coerce')
        print(f"   平均 Likes: {df['Likes_Num'].mean():.0f}")
        print(f"   中位數 Likes: {df['Likes_Num'].median():.0f}")
        print(f"   最高 Likes: {df['Likes_Num'].max():.0f}")
        print(f"   總互動: {(df['Likes_Num'].fillna(0)).sum():,.0f}")

    # 4. 主題預測（用關鍵詞）
    print(f"\n【預測主題分佈（用關鍵詞估算）】")
    topic_keywords = {
        '投資理財': ['投資', '理財', '股票', '基金', 'ETF', '回報', '複利'],
        '帶貨推銷': ['帶貨', '推薦', '入手', '必買', '超值'],
        '理財教育': ['教學', '學習', '知識', '入門'],
        '詐騙警示': ['騙', '假', '小心', '風險'],
        '消費分享': ['買咗', '用咗', '分享', '開箱'],
    }

    all_text = ' '.join(df['Title'].astype(str) + ' ' + df['Content'].astype(str))
    for topic, keywords in topic_keywords.items():
        count = sum(1 for kw in keywords if kw in all_text)
        print(f"   {topic}: ~{count} 條可能")

    # 5. 收集進度評估
    print(f"\n【收集進度評估】")
    target = 100
    progress = len(df) / target * 100
    bar_length = 30
    filled = int(progress / 100 * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)

    status = "✓ 達標" if progress >= 100 else "⚠️ 不足" if progress < 50 else "🔄 進行中"
    print(f"   [{bar}] {progress:.0f}% ({len(df)}/{target}) {status}")


def check_topic_balance(df):
    """檢查主題分佈是否平衡"""
    print("\n" + "="*60)
    print("⚖️  主題平衡檢查")
    print("="*60)

    all_text = ' '.join(df['Title'].astype(str) + ' ' + df['Content'].astype(str))

    topic_keywords = {
        '理財教育': ['教學', '學習', '知識', '入門', '基礎', '概念', '教育'],
        '投資理財': ['投資', '理財', '股票', '基金', 'ETF', '回報', '複利', 'MPay'],
        '帶貨推銷': ['帶貨', '推薦', '入手', '必買', '超值', '好用', '回購'],
        '詐騙警示': ['騙', '假', '小心', '注意', '風險', '警示'],
        '消費分享': ['買咗', '用咗', '分享', '開箱', '體驗'],
        '情緒消費': ['心情', '開心', '壓力', '減壓', '獎勵自己'],
    }

    print("\n各主題佔比（粗略估算）:")
    total = len(df)
    for topic, keywords in topic_keywords.items():
        # 統計包含任何關鍵詞嘅內容
        count = df.apply(lambda row: any(kw in str(row['Title']) + str(row['Content'])
                                          for kw in keywords), axis=1).sum()
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        status = "✓" if pct >= 15 else "⚠️"
        print(f"   {status} {topic:<12} {count:>3} ({pct:>5.1f}%) {bar}")

    # 建議
    print("\n【建議】")
    print("   • 每個主題理想比例: 15-25%")
    print("   • 帶貨類可以略高: 20-30%")
    print("   • 詐騙警示可以略低: 5-10%")


def export_summary(df, output_path="nlp_collection_summary.txt"):
    """匯出摘要報告"""
    summary = f"""
========================================
NLP 數據收集摘要報告
生成時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
========================================

【總體統計】
總內容數: {len(df)}
涵蓋平台: {df['Platform'].nunique()}
涵蓋作者: {df['Author_Handle'].nunique()}
日期範圍: {df['Date'].min()} 至 {df['Date'].max()}

【平台分佈】
{df['Platform'].value_counts().to_string()}

【準備狀態】
{'✓ 數據充足，可以運行 NLP 分析' if len(df) >= 50 else '⚠️  數據偏少，建議再收集' if len(df) >= 20 else '❌ 數據嚴重不足'}
目標: 100 條
當前: {len(df)} 條 ({len(df)/100*100:.0f}%)
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"\n✓ 摘要已匯出: {output_path}")


def main():
    # 預設路徑
    csv_paths = [
        '/sessions/determined-quirky-franklin/mnt/Venus_competition/nlp_content_collected_2026.csv',
        '/sessions/determined-quirky-franklin/mnt/outputs/nlp_content_template.csv',
    ]

    for csv_path in csv_paths:
        if Path(csv_path).exists():
            print(f"\n🔍 檢查: {csv_path}")
            df = validate_csv(csv_path)
            if df is not None:
                show_dashboard(df)
                check_topic_balance(df)
                export_summary(df)
            break
    else:
        print("❌ 找不到任何 CSV 檔案")
        print("\n請先：")
        print("1. 將收集到嘅內容儲存為 nlp_content_collected_2026.csv")
        print("2. 放入 Venus_competition 資料夾")
        print("3. 再運行呢個腳本")


if __name__ == "__main__":
    main()
