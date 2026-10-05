"""簡化版分析 - 直接從 CSV 讀取並輸出統計"""
import pandas as pd
import numpy as np

# 讀取數據
df = pd.read_csv('/sessions/determined-quirky-franklin/mnt/outputs/questionnaire_responses.csv')
print(f"✓ 載入 {len(df)} 份問卷")
print()

# 計算各量表
df['FCS_Conversation'] = df[['B1','B2','B3','B4','B5']].mean(axis=1)
df['FCS_Conformity'] = df[['B6','B7','B8','B9','B10']].mean(axis=1)
df['FCS_FinEdu'] = df[['B11','B12','B13','B14','B15','B16']].mean(axis=1)
df['FCS_Conflict'] = df[['B17','B18','B19','B20']].mean(axis=1)
df['Impulse_Score'] = df[['C1','C2','C3','C4','C5','C6','C7','C8']].mean(axis=1)
df['Self_Control'] = df[['C9','C10','C11','C12']].mean(axis=1)

# 量化閒置物品
waste_map = {
    '$0': 0, '$1–$100': 50, '$101–$500': 300,
    '$501–$1,000': 750, '$1,001–$2,000': 1500,
    '$2,001–$5,000': 3500, '> $5,000': 7500
}
df['Unused_6mo'] = df['D11'].map(waste_map).fillna(0)
df['Monthly_Waste'] = df['Unused_6mo'] / 6

# 計算 NPV
def future_value(monthly, years, rate):
    months = years * 12
    mr = (1 + rate) ** (1/12) - 1
    return monthly * (((1 + mr) ** months - 1) / mr)

df['NPV_8pct'] = df['Monthly_Waste'].apply(lambda x: future_value(x, 18, 0.08))

# G3/G4 分類
df['G3_Correct'] = df['G3'].isin(['$120,000–$200,000', '$200,000 以上']).astype(int)
df['G4_Patient'] = (df['G4'] == '(B) 等 18 年後拎 $2,500').astype(int)

def kw_classify(row):
    if row['G3_Correct'] and row['G4_Patient']:
        return '理性耐性型'
    elif row['G3_Correct'] and not row['G4_Patient']:
        return '識唔等型'
    elif not row['G3_Correct'] and row['G4_Patient']:
        return '直覺耐性型'
    else:
        return '兩頭空型'

df['KW_Group'] = df.apply(kw_classify, axis=1)

# ============================================
# 輸出結果摘要
# ============================================
print("=" * 70)
print("📊 模擬數據分析結果摘要 (N = {})".format(len(df)))
print("=" * 70)

print("\n【1. 樣本基本資料】")
print(f"  • 男生: {(df['A1']=='男').sum()} ({(df['A1']=='男').mean()*100:.1f}%)")
print(f"  • 女生: {(df['A1']=='女').sum()} ({(df['A1']=='女').mean()*100:.1f}%)")
print(f"  • 平均每日社交媒體時間眾數: {df['A4'].mode()[0]}")
print(f"  • 平均每月網購金額眾數: {df['A8'].mode()[0]}")

print("\n【2. 家庭溝通模式 (Likert 1-7)】")
print(f"  • FCS_Conversation (開放對話): {df['FCS_Conversation'].mean():.2f} ± {df['FCS_Conversation'].std():.2f}")
print(f"  • FCS_Conformity (服從權威):  {df['FCS_Conformity'].mean():.2f} ± {df['FCS_Conformity'].std():.2f}")
print(f"  • FCS_FinEdu (理財教育):      {df['FCS_FinEdu'].mean():.2f} ± {df['FCS_FinEdu'].std():.2f}")
print(f"  • FCS_Conflict (家庭衝突):    {df['FCS_Conflict'].mean():.2f} ± {df['FCS_Conflict'].std():.2f}")

print("\n【3. 衝動消費 + 自我控制】")
print(f"  • Impulse_Score: {df['Impulse_Score'].mean():.2f} ± {df['Impulse_Score'].std():.2f}")
print(f"  • Self_Control:  {df['Self_Control'].mean():.2f} ± {df['Self_Control'].std():.2f}")
print(f"  • 相關性 r = {df['Impulse_Score'].corr(df['Self_Control']):.3f}")

print("\n【4. 過度購買 + NPV 機會成本】")
print(f"  • 過去6個月平均閒置物品價值: MOP {df['Unused_6mo'].mean():,.0f}")
print(f"  • 平均每月浪費: MOP {df['Monthly_Waste'].mean():.0f}")
print(f"  • 18年 NPV (8%回報): MOP {df['NPV_8pct'].mean():,.0f}")
print(f"  • 中位數 NPV: MOP {df['NPV_8pct'].median():,.0f}")
print(f"  • 範圍: MOP {df['NPV_8pct'].min():,.0f} - {df['NPV_8pct'].max():,.0f}")

print("\n【5. 投資認知測試】")
print(f"  • G3 NPV 認知答對率: {df['G3_Correct'].mean()*100:.1f}%")
print(f"  • G4 延遲滿足率:    {df['G4_Patient'].mean()*100:.1f}%")
print(f"  • 🎯 Knowledge-Willingness Gap 比例: {(df['KW_Group']=='識唔等型').mean()*100:.1f}%")

print("\n【6. Knowledge-Willingness 4 種類型分佈】")
kw_counts = df['KW_Group'].value_counts()
for name, n in kw_counts.items():
    pct = n / len(df) * 100
    emoji = {'理性耐性型': '🟢', '識唔等型': '🟠', '直覺耐性型': '🟡', '兩頭空型': '🔴'}[name]
    print(f"  {emoji} {name}: {n} 人 ({pct:.1f}%)")

print("\n【7. 關鍵相關性分析】")
print(f"  • FCS_Conversation vs Impulse:    r = {df['FCS_Conversation'].corr(df['Impulse_Score']):.3f}")
print(f"  • FCS_Conformity vs Impulse:     r = {df['FCS_Conformity'].corr(df['Impulse_Score']):.3f}")
print(f"  • FCS_FinEdu vs Impulse:         r = {df['FCS_FinEdu'].corr(df['Impulse_Score']):.3f}")
print(f"  • FCS_Conflict vs Monthly_Waste: r = {df['FCS_Conflict'].corr(df['Monthly_Waste']):.3f}")
print(f"  • Impulse vs Monthly_Waste:      r = {df['Impulse_Score'].corr(df['Monthly_Waste']):.3f}")

print("\n【8. 情境題分析 (D8/D9/D10)】")
print(f"  D8 (減價衝動): {df['D8'].value_counts().to_dict()}")
print(f"  D9 (沉沒成本): {df['D9'].value_counts().to_dict()}")
print(f"  D10 (FOMO):    {df['D10'].value_counts().to_dict()}")

print("\n【9. 結論性發現】")
print(f"  • 「識 NPV 但等唔到 18 年」嘅學生: {(df['KW_Group']=='識唔等型').sum()} 人")
print(f"  • 假設 50 人全部有每月 MOP ${df['Monthly_Waste'].mean():.0f} 嘅浪費，")
print(f"    18 年後集體失去嘅財富約 MOP {df['NPV_8pct'].sum():,.0f}")
print(f"    平均每人 MOP {df['NPV_8pct'].mean():,.0f}")

print("\n" + "=" * 70)
print("✅ 分析完成！呢個係模擬數據結果，真實數據可能有差異。")
print("=" * 70)
