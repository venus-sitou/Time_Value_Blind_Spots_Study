"""
analyze_from_csv_v2.py  (N=60 真實數據版)

基於 N=60 xlsx 數據嘅完整量化分析：
- 真實 Likert scale（B1-B20, C1-C12）
- 真實字串金額（D2, D10）經 mapping 處理
- KW Gap 分類（G3 x G4）
- 預測模型（Logistic Regression + Random Forest）
- 8 大進階模型結果輸出
"""
import pandas as pd
import numpy as np
import json
import os

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'questionnaire_responses.csv')

df = pd.read_csv(CSV_PATH)
N = len(df)

print("=" * 70)
print(f"📊 真實問卷數據完整分析 (N={N}) — 2026-10-04 xlsx 版")
print("=" * 70)

# === Mappings ===
d2_map = {
    '< $100': 50, '$100–$300': 200, '$300–$500': 400,
    '$500–$1,000': 750, '$1,000–$2,000': 1500, '> $2,000': 2500,
    '從不網購': 0,
}
d10_map = {
    '0.0': 0, '0': 0, '$1–$100': 50, '$101–$500': 300,
    '$501–$1,000': 750, '$1,001–$2,000': 1500,
    '$2,001–$5,000': 3500, '> $5,000': 7500,
    '$2,001–$5,000': 3500,
}
# G3: NPV 認知 — 正確答案係 $60,000–$120,000
# G4: 時間偏好 — 答案「等 18 年後領取」= patient

def parse_d10(v):
    if pd.isna(v):
        return 0
    s = str(v).strip()
    if s in d10_map:
        return d10_map[s]
    # try numeric
    try:
        return float(s)
    except:
        return 0

def parse_d2(v):
    if pd.isna(v):
        return 0
    return d2_map.get(str(v).strip(), 0)

def fv(monthly, years=18, rate=0.08):
    if monthly <= 0:
        return 0
    months = years * 12
    mr = (1 + rate) ** (1/12) - 1
    return monthly * (((1 + mr) ** months - 1) / mr)

# === 1. Demographics ===
print("\n【一、人口統計】")
print(f"性別分佈: 男={int((df['A1']=='男').sum())} 女={int((df['A1']=='女').sum())}")
print(f"學校: {df['A3'].value_counts().to_dict()}")
print(f"年級: {df['A2'].value_counts().to_dict()}")
print(f"社交媒體時間: {df['A4'].value_counts().to_dict()}")
print(f"零用錢: {df['A7'].value_counts().to_dict()}")

# === 2. FCS ===
print("\n【二、家庭溝通模式 (FCS, Likert 1-5)】")
df['FCS_Conversation'] = df[['B1', 'B2', 'B3', 'B4', 'B5']].mean(axis=1)
df['FCS_Conformity'] = df[['B6', 'B7', 'B8', 'B9', 'B10']].mean(axis=1)
df['FCS_FinEdu'] = df[['B11', 'B12', 'B13', 'B14', 'B15', 'B16']].mean(axis=1)
df['FCS_Conflict'] = df[['B17', 'B18', 'B19', 'B20']].mean(axis=1)
for k in ['FCS_Conversation', 'FCS_Conformity', 'FCS_FinEdu', 'FCS_Conflict']:
    print(f"  {k}: {df[k].mean():.3f} ± {df[k].std():.3f}")

# === 3. Impulse & Self-Control ===
print("\n【三、衝動消費同自我控制】")
df['Impulse_Score'] = df[['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']].mean(axis=1)
df['Self_Control'] = df[['C9', 'C10', 'C11', 'C12']].mean(axis=1)
print(f"  Impulse_Score: {df['Impulse_Score'].mean():.3f} ± {df['Impulse_Score'].std():.3f}")
print(f"  Self_Control:  {df['Self_Control'].mean():.3f} ± {df['Self_Control'].std():.3f}")
print(f"  相關 (r): {df['Impulse_Score'].corr(df['Self_Control']):.3f}")

# === 4. NPV quantification ===
print("\n【四、NPV 量化（18 年投資 S&P 500, 8% 年回報）】")
df['D2_numeric'] = df['D2'].apply(parse_d2)
df['D10_numeric'] = df['D10'].apply(parse_d10)
df['Monthly_Waste'] = df['D10_numeric'] / 6
df['NPV_18y'] = df['Monthly_Waste'].apply(lambda x: fv(x, 18, 0.08))

print(f"  D2 每月網購消費: MOP {df['D2_numeric'].mean():.0f}")
print(f"  D10 過去 6 個月閒置物品: MOP {df['D10_numeric'].mean():.0f}")
print(f"  → 平均每月浪費: MOP {df['Monthly_Waste'].mean():.0f}")
print(f"  → 平均 18 年 NPV: MOP {df['NPV_18y'].mean():,.0f}")
print(f"  → {N} 人總損失: MOP {df['NPV_18y'].sum():,.0f}")
print(f"  中位數 NPV: MOP {df['NPV_18y'].median():,.0f}")
print(f"  最大 NPV: MOP {df['NPV_18y'].max():,.0f}")

# === 5. KW Gap 分類 ===
print("\n【五、KW Gap 4 種類型】")
df['G3_Correct'] = df['G3'].str.contains(r'\$60,000–\$120,000', regex=True).astype(int)
df['G4_Patient'] = df['G4'].str.contains('等 18 年後', regex=False).astype(int)

def kw_classify(row):
    if row['G3_Correct'] and row['G4_Patient']:
        return '理性耐性型 (Rational-Patient)'
    elif row['G3_Correct'] and not row['G4_Patient']:
        return '識唔等型 (Knows-But-Impatient)'
    elif not row['G3_Correct'] and row['G4_Patient']:
        return '直覺耐性型 (Instinct-Patient)'
    else:
        return '兩頭空型 (Knows-Nothing-Impatient)'

df['KW_Class'] = df.apply(kw_classify, axis=1)
kw_counts = df['KW_Class'].value_counts()
for cls in ['理性耐性型 (Rational-Patient)', '識唔等型 (Knows-But-Impatient)',
            '直覺耐性型 (Instinct-Patient)', '兩頭空型 (Knows-Nothing-Impatient)']:
    n = kw_counts.get(cls, 0)
    print(f"  {cls}: {n} 人 ({n/N*100:.1f}%)")

print(f"\n  G3 答對率: {df['G3_Correct'].sum()}/{N} ({df['G3_Correct'].mean()*100:.1f}%)")
print(f"  G4 選擇耐心: {df['G4_Patient'].sum()}/{N} ({df['G4_Patient'].mean()*100:.1f}%)")

# KW Gap × NPV
print("\n  KW Gap × NPV 交叉分析:")
for cls in ['理性耐性型 (Rational-Patient)', '識唔等型 (Knows-But-Impatient)',
            '直覺耐性型 (Instinct-Patient)', '兩頭空型 (Knows-Nothing-Impatient)']:
    sub = df[df['KW_Class'] == cls]
    if len(sub):
        print(f"    {cls}: NPV 平均 MOP {sub['NPV_18y'].mean():,.0f}, N={len(sub)}")

# === 6. SEM-style correlations ===
print("\n【六、核心 SEM 路徑相關性】")
print(f"  FCS_Conversation → Impulse: r = {df['FCS_Conversation'].corr(df['Impulse_Score']):.3f}")
print(f"  FCS_Conformity   → Impulse: r = {df['FCS_Conformity'].corr(df['Impulse_Score']):.3f}")
print(f"  FCS_FinEdu       → Impulse: r = {df['FCS_FinEdu'].corr(df['Impulse_Score']):.3f}")
print(f"  FCS_Conflict     → Impulse: r = {df['FCS_Conflict'].corr(df['Impulse_Score']):.3f}")
print(f"  FCS_Conflict     → C7 (情緒消費): r = {df['FCS_Conflict'].corr(df['C7']):.3f}")
print(f"  Impulse          → NPV_18y: r = {df['Impulse_Score'].corr(df['NPV_18y']):.3f}")

# === 7. 預測模型 ===
print("\n【七、預測模型：高衝動消費識別】")
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    features = ['FCS_Conversation', 'FCS_Conformity', 'FCS_FinEdu', 'FCS_Conflict',
                'Self_Control', 'D2_numeric']
    X = df[features].fillna(df[features].mean())
    y = (df['Impulse_Score'] >= df['Impulse_Score'].median()).astype(int)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(Xs, y)
    train_acc = logreg.score(Xs, y)
    cv_scores = cross_val_score(logreg, Xs, y, cv=5)
    print(f"  Logistic Regression: Train {train_acc*100:.1f}%, 5-fold CV {cv_scores.mean()*100:.1f}%")

    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4)
    rf.fit(X, y)
    rf_train = rf.score(X, y)
    rf_cv = cross_val_score(rf, X, y, cv=5).mean()
    print(f"  Random Forest:       Train {rf_train*100:.1f}%, 5-fold CV {rf_cv*100:.1f}%")

    feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    print(f"\n  RF Feature Importance:")
    for k, v in feat_imp.items():
        print(f"    {k}: {v*100:.1f}%")
except Exception as e:
    print(f"  ⚠️  sklearn unavailable: {e}")

# === 8. Cluster analysis ===
print("\n【八、K-means 集群分析（4 種消費人格）】")
try:
    from sklearn.cluster import KMeans
    cluster_features = ['Impulse_Score', 'Self_Control', 'FCS_Conversation', 'FCS_Conflict']
    Xc = df[cluster_features].fillna(df[cluster_features].mean())
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = km.fit_predict(Xc)
    print(f"  Cluster sizes: {df['Cluster'].value_counts().to_dict()}")
    cluster_means = df.groupby('Cluster')[cluster_features].mean()
    print(cluster_means.round(2))
except Exception as e:
    print(f"  ⚠️  sklearn unavailable: {e}")

# === 9. Save JSON ===
result = {
    'analysis_date': '2026-10-04',
    'data_source': '[SURVEY TITLE] (回應)_2026-10-04_21-22.xlsx',
    'N': N,
    'demographics': {
        'gender': df['A1'].value_counts().to_dict(),
        'school_type': df['A3'].value_counts().to_dict(),
        'grade': df['A2'].value_counts().to_dict(),
        'social_media_time': df['A4'].value_counts().to_dict(),
        'allowance': df['A7'].value_counts().to_dict(),
    },
    'FCS': {
        'Conversation': float(df['FCS_Conversation'].mean()),
        'Conformity': float(df['FCS_Conformity'].mean()),
        'FinEdu': float(df['FCS_FinEdu'].mean()),
        'Conflict': float(df['FCS_Conflict'].mean()),
    },
    'Impulse': {
        'mean': float(df['Impulse_Score'].mean()),
        'std': float(df['Impulse_Score'].std()),
    },
    'Self_Control': {
        'mean': float(df['Self_Control'].mean()),
        'std': float(df['Self_Control'].std()),
    },
    'NPV': {
        'monthly_waste_mean': float(df['Monthly_Waste'].mean()),
        'npv_18y_mean': float(df['NPV_18y'].mean()),
        'npv_18y_median': float(df['NPV_18y'].median()),
        'npv_18y_max': float(df['NPV_18y'].max()),
        'npv_18y_sum': float(df['NPV_18y'].sum()),
    },
    'KW_Gap': {
        'G3_correct_pct': float(df['G3_Correct'].mean() * 100),
        'G4_patient_pct': float(df['G4_Patient'].mean() * 100),
        'distribution': df['KW_Class'].value_counts().to_dict(),
    },
    'correlations': {
        'Conv_to_Impulse': float(df['FCS_Conversation'].corr(df['Impulse_Score'])),
        'Conf_to_Impulse': float(df['FCS_Conformity'].corr(df['Impulse_Score'])),
        'FinEdu_to_Impulse': float(df['FCS_FinEdu'].corr(df['Impulse_Score'])),
        'Conflict_to_Impulse': float(df['FCS_Conflict'].corr(df['Impulse_Score'])),
        'Conflict_to_C7': float(df['FCS_Conflict'].corr(df['C7'])),
        'Impulse_to_NPV': float(df['Impulse_Score'].corr(df['NPV_18y'])),
    },
}

out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis_result_v2.json')
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
print(f"\n✓ Saved {out_json}")
print("\n" + "=" * 70)
print("✅ N=60 完整分析完成")
print("=" * 70)
