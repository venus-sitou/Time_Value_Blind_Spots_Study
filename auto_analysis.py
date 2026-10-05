"""
========================================
[RESEARCH PROJECT] - 自動化分析系統
========================================

功能：
1. 直接從 Google Sheets 讀取問卷數據
2. 自動執行 5 大模型分析（SEM、Bootstrap、Cluster、NPV、Monte Carlo）
3. 生成所有結果圖表 + 摘要報告

使用方法：
1. 設定 Google Form（見 GOOGLE_FORM_SETUP.md）
2. 喺 Google Sheets 入面點「共用」取得連結
3. 將連結貼入下方 GOOGLE_SHEET_URL
4. 運行：python auto_analysis.py

作者：[RESEARCH PROJECT]參賽隊伍
日期：2026 年 10 月
"""

# ============================================
# Part 1: 環境設置 + 數據讀取
# ============================================

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC',
                                     'Noto Sans CJK TC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 設定隨機種子（確保結果可重複）
np.random.seed(42)

# ============================================
# 設定 Google Sheets URL
# ============================================

# 將 Google Sheets 設定為「知道連結嘅人都可以檢視」
# 然後複製連結貼到下面
# 連結格式：https://docs.google.com/spreadsheets/d/SHEET_ID/edit#gid=0

GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/export?format=csv"

# 或者用本機 CSV 檔案
LOCAL_CSV_PATH = "questionnaire_responses.csv"


def load_data():
    """從 Google Sheets 或本機 CSV 讀取數據"""
    try:
        # 嘗試從 Google Sheets 讀取
        if GOOGLE_SHEET_URL != "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/export?format=csv":
            print("📊 正在從 Google Sheets 讀取數據...")
            df = pd.read_csv(GOOGLE_SHEET_URL)
            print(f"✓ 成功讀取 {len(df)} 份問卷")
        else:
            raise ValueError("請設定 GOOGLE_SHEET_URL")
    except Exception as e:
        # Fallback: 讀取本機 CSV
        print(f"⚠️  無法從 Google Sheets 讀取: {e}")
        print(f"📁 嘗試讀取本機檔案: {LOCAL_CSV_PATH}")
        try:
            df = pd.read_csv(LOCAL_CSV_PATH)
            print(f"✓ 成功讀取 {len(df)} 份問卷")
        except FileNotFoundError:
            print(f"❌ 找不到檔案 {LOCAL_CSV_PATH}")
            print("\n請先：")
            print("1. 設定 GOOGLE_SHEET_URL，或")
            print("2. 將 CSV 檔案命名為 'questionnaire_responses.csv' 並放到同目錄")
            return None

    return df


# ============================================
# Part 2: 數據清洗 + 量表計分
# ============================================

def clean_and_score(df):
    """數據清洗 + 計算各個量表分數"""
    print("\n" + "="*60)
    print("🔧 數據清洗 + 量表計分")
    print("="*60)

    # === FCS 子構念 ===
    df['FCS_Conversation'] = df[['B1', 'B2', 'B3', 'B4', 'B5']].mean(axis=1)
    df['FCS_Conformity'] = df[['B6', 'B7', 'B8', 'B9', 'B10']].mean(axis=1)
    df['FCS_FinEdu'] = df[['B11', 'B12', 'B13', 'B14', 'B15', 'B16']].mean(axis=1)
    df['FCS_Conflict'] = df[['B17', 'B18', 'B19', 'B20']].mean(axis=1)

    # === 衝動消費 + 自我控制 ===
    df['Impulse_Score'] = df[['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']].mean(axis=1)
    df['Self_Control'] = df[['C9', 'C10', 'C11', 'C12']].mean(axis=1)

    # === 過度購買 (D11 量化) ===
    waste_map = {
        '$0': 0,
        '$1–$100': 50,
        '$101–$500': 300,
        '$501–$1,000': 750,
        '$1,001–$2,000': 1500,
        '$2,001–$5,000': 3500,
        '> $5,000': 7500
    }
    df['Unused_Value_6mo'] = df['D11'].map(waste_map).fillna(0)
    df['Monthly_Waste'] = df['Unused_Value_6mo'] / 6

    # === G3 NPV 認知（答案：$120,000 以上 = 答啱）===
    df['G3_Correct'] = df['G3'].isin(['$120,000–$200,000', '$200,000 以上']).astype(int)

    # === G4 時間偏好（答案：等 18 年 = 理性耐性）===
    df['G4_Patient'] = (df['G4'] == '(B) 等 18 年後拎 $2,500').astype(int)

    # === Knowledge-Willingness 4 組分類 ===
    df['KW_Group'] = 'Unknown'
    df.loc[df['G3_Correct'].astype(bool) & df['G4_Patient'].astype(bool), 'KW_Group'] = '理性耐性型'
    df.loc[df['G3_Correct'].astype(bool) & ~df['G4_Patient'].astype(bool), 'KW_Group'] = '識唔等型'
    df.loc[~df['G3_Correct'].astype(bool) & df['G4_Patient'].astype(bool), 'KW_Group'] = '直覺耐性型'
    df.loc[~df['G3_Correct'].astype(bool) & ~df['G4_Patient'].astype(bool), 'KW_Group'] = '兩頭空型'

    print(f"✓ 樣本量: {len(df)}")
    print(f"✓ G3 答啱率: {df['G3_Correct'].mean()*100:.1f}%")
    print(f"✓ G4 延遲滿足率: {df['G4_Patient'].mean()*100:.1f}%")
    print(f"\n📊 Knowledge-Willingness 分組:")
    print(df['KW_Group'].value_counts())

    return df


# ============================================
# Part 3: 信度分析 (Cronbach's α)
# ============================================

def cronbach_alpha(items_df):
    """計算 Cronbach α 係數"""
    k = items_df.shape[1]
    item_vars = items_df.var(axis=0, ddof=1)
    total_var = items_df.sum(axis=1).var(ddof=1)
    return (k / (k-1)) * (1 - item_vars.sum() / total_var)


def reliability_analysis(df):
    """信度分析"""
    print("\n" + "="*60)
    print("📏 信度分析 (Cronbach's α)")
    print("="*60)

    alphas = {}
    alphas['FCS_Conversation'] = cronbach_alpha(df[['B1','B2','B3','B4','B5']])
    alphas['FCS_Conformity'] = cronbach_alpha(df[['B6','B7','B8','B9','B10']])
    alphas['FCS_FinEdu'] = cronbach_alpha(df[['B11','B12','B13','B14','B15','B16']])
    alphas['FCS_Conflict'] = cronbach_alpha(df[['B17','B18','B19','B20']])
    alphas['Impulse_Score'] = cronbach_alpha(df[['C1','C2','C3','C4','C5','C6','C7','C8']])
    alphas['Self_Control'] = cronbach_alpha(df[['C9','C10','C11','C12']])

    print("\n量表".ljust(20), "α 係數".rjust(10), "評價".rjust(15))
    print("-" * 50)
    for name, alpha in alphas.items():
        rating = "優秀" if alpha > 0.9 else "良好" if alpha > 0.8 else "可接受" if alpha > 0.7 else "需改善"
        print(f"{name:<20} {alpha:.3f}".rjust(10).ljust(30), f"{rating}")

    # 視覺化
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(alphas.keys())
    values = list(alphas.values())
    colors = ['green' if v > 0.8 else 'orange' if v > 0.7 else 'red' for v in values]
    ax.barh(names, values, color=colors, edgecolor='black')
    ax.axvline(x=0.7, color='red', linestyle='--', label='可接受閾值 (0.7)')
    ax.axvline(x=0.8, color='orange', linestyle='--', label='良好閾值 (0.8)')
    ax.set_xlabel("Cronbach's α")
    ax.set_title("各量表信度分析")
    ax.legend()
    plt.tight_layout()
    plt.savefig("01_reliability.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ 圖表已儲存：01_reliability.png")

    return alphas


# ============================================
# Part 4: 模型 1 - SEM 結構方程模型
# ============================================

def sem_analysis(df):
    """結構方程模型分析"""
    print("\n" + "="*60)
    print("🔬 模型 1：結構方程模型 (SEM)")
    print("="*60)

    try:
        import semopy

        # 定義 SEM 模型
        model_desc = """
        # 測量模型
        FCS_Conv =~ B1 + B2 + B3 + B4 + B5
        FCS_Conf =~ B6 + B7 + B8 + B9 + B10
        FCS_Edu =~ B11 + B12 + B13 + B14 + B15 + B16
        FCS_Conf2 =~ B17 + B18 + B19 + B20
        Impulse =~ C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8
        SelfCtrl =~ C9 + C10 + C11 + C12

        # 結構模型
        Impulse ~ FCS_Conv + FCS_Conf + FCS_Edu + FCS_Conf2
        SelfCtrl ~ FCS_Conv + FCS_Edu
        Monthly_Waste ~ Impulse + FCS_Conv + FCS_Conf2
        """

        model = semopy.Model(model_desc)
        result = model.fit(df)
        print("✓ 模型擬合完成")

        # 擬合指標
        stats_df = semopy.calc_stats(model)
        print("\n📊 模型擬合指標:")
        print(stats_df.T[['Value']])

        # 路徑係數
        try:
            inspect = model.inspect()
        except AttributeError:
            # 兼容較舊嘅 semopy API
            inspect = semopy.inspect(model)
        structural_paths = inspect[inspect['op'] == '~'].copy()
        structural_paths = structural_paths.sort_values('Estimate', ascending=False)

        print("\n📊 結構路徑係數（由大到小）:")
        print(structural_paths.to_string(index=False))

        # 視覺化
        try:
            semopy.semplot(model, "02_sem_diagram.png")
            print("\n✓ SEM 路徑圖已儲存：02_sem_diagram.png")
        except Exception as e:
            print(f"⚠️  路徑圖生成失敗: {e}")

        return inspect

    except ImportError:
        print("⚠️  semopy 未安裝")
        print("安裝指令：pip install semopy --break-system-packages")
        return None


# ============================================
# Part 5: 模型 2 - Bootstrap 中介效應
# ============================================

def bootstrap_mediation(df, x_var, m_var, y_var, n_boot=5000):
    """檢驗中介效應"""
    indirect_effects = []
    direct_effects = []

    for _ in range(n_boot):
        sample = resample(df, n_samples=len(df), replace=True)

        # Path a: X → M
        try:
            model_a = smf.ols(f"{m_var} ~ {x_var}", data=sample).fit()
            a = model_a.params[x_var]
        except:
            continue

        # Path b & c': M → Y controlling X
        try:
            model_b = smf.ols(f"{y_var} ~ {m_var} + {x_var}", data=sample).fit()
            b = model_b.params[m_var]
            cp = model_b.params[x_var]
        except:
            continue

        indirect_effects.append(a * b)
        direct_effects.append(cp)

    return {
        'mean_indirect': np.mean(indirect_effects),
        'ci_lower': np.percentile(indirect_effects, 2.5),
        'ci_upper': np.percentile(indirect_effects, 97.5),
        'significant': (np.percentile(indirect_effects, 2.5) > 0) |
                       (np.percentile(indirect_effects, 97.5) < 0),
        'mean_direct': np.mean(direct_effects)
    }


def mediation_analysis(df):
    """執行多組中介效應檢驗"""
    print("\n" + "="*60)
    print("🔄 模型 2：Bootstrap 中介效應檢驗 (5000 次重抽)")
    print("="*60)

    # 三組中介分析
    paths = [
        ('FCS_Conversation', 'Impulse_Score', 'Monthly_Waste', '對話型 → 衝動 → 過度購買'),
        ('FCS_Conflict', 'C7', 'Monthly_Waste', '家庭衝突 → 情緒消費 → 過度購買'),
        ('FCS_FinEdu', 'Impulse_Score', 'Monthly_Waste', '理財教育 → 衝動 → 過度購買'),
    ]

    results = []
    for x_var, m_var, y_var, label in paths:
        print(f"\n📊 {label}")
        result = bootstrap_mediation(df, x_var, m_var, y_var)
        print(f"   Indirect effect: {result['mean_indirect']:.3f}")
        print(f"   95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
        sig = "✓ 顯著" if result['significant'] else "✗ 不顯著"
        print(f"   {sig}")
        results.append({'path': label, **result})

    # 視覺化
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [r['path'] for r in results]
    means = [r['mean_indirect'] for r in results]
    errors = [(r['mean_indirect'] - r['ci_lower'], r['ci_upper'] - r['mean_indirect'])
              for r in results]
    errors = np.array(errors).T

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, means, xerr=errors, color='steelblue',
            edgecolor='black', alpha=0.7, capsize=5)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('中介效應係數 (Indirect Effect)')
    ax.set_title('Bootstrap 中介效應檢驗 (5000 次)')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("03_mediation.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ 圖表已儲存：03_mediation.png")

    return results


# ============================================
# Part 6: 模型 3 - Cluster Analysis
# ============================================

def cluster_analysis(df):
    """K-means 集群分析"""
    print("\n" + "="*60)
    print("🎯 模型 3：K-means 集群分析")
    print("="*60)

    features = ['FCS_Conversation', 'FCS_Conflict', 'Impulse_Score',
                'Self_Control', 'Monthly_Waste']
    X = df[features].values
    X_scaled = StandardScaler().fit_transform(X)

    # 決定最佳 k
    sil_scores = []
    K_range = range(2, 8)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        sil_scores.append(sil)

    optimal_k = list(K_range)[np.argmax(sil_scores)]
    print(f"\n✓ 最佳群數: k = {optimal_k} (Silhouette = {max(sil_scores):.3f})")

    # 執行 K-means
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # 各群特徵
    cluster_profile = df.groupby('Cluster')[features].mean()
    print("\n📊 各集群特徵平均:")
    print(cluster_profile.round(2))

    # 命名（根據特徵自動命名）
    cluster_names = {}
    name_candidates = {
        0: ['🟢 節制型', '🟡 平衡型', '🟠 情緒型', '🔴 隱藏型'],
        1: ['🟢 節制型', '🟡 平衡型', '🟠 情緒型', '🔴 隱藏型'],
        2: ['🟢 節制型', '🟡 平衡型', '🟠 情緒型', '🔴 隱藏型'],
        3: ['🟢 節制型', '🟡 平衡型', '🟠 情緒型', '🔴 隱藏型'],
    }

    # 按「衝動消費 + 浪費」自動分配名稱
    sorted_clusters = cluster_profile.sort_values(
        by=['Impulse_Score', 'Monthly_Waste']
    ).index.tolist()

    for i, cluster_id in enumerate(sorted_clusters):
        if i == 0:
            cluster_names[cluster_id] = '🟢 節制型'
        elif i == len(sorted_clusters) - 1:
            cluster_names[cluster_id] = '🔴 隱藏型'
        elif cluster_profile.loc[cluster_id, 'FCS_Conflict'] > cluster_profile['FCS_Conflict'].median():
            cluster_names[cluster_id] = '🟠 情緒型'
        else:
            cluster_names[cluster_id] = '🟡 平衡型'

    df['Cluster_Name'] = df['Cluster'].map(cluster_names)

    print("\n📊 各群人數:")
    counts = df['Cluster_Name'].value_counts()
    for name, count in counts.items():
        pct = count / len(df) * 100
        print(f"   {name}: {count} 人 ({pct:.1f}%)")

    # 視覺化：雷達圖
    from math import pi

    fig, axes = plt.subplots(1, optimal_k, figsize=(4*optimal_k, 4),
                              subplot_kw=dict(polar=True))
    if optimal_k == 1:
        axes = [axes]

    angles = [n / float(len(features)) * 2 * pi for n in range(len(features))]
    angles += angles[:1]

    for idx, cluster_id in enumerate(sorted_clusters):
        cluster_data = cluster_profile.loc[cluster_id].values.tolist()
        cluster_data += cluster_data[:1]

        axes[idx].plot(angles, cluster_data, linewidth=2)
        axes[idx].fill(angles, cluster_data, alpha=0.25)
        axes[idx].set_xticks(angles[:-1])
        axes[idx].set_xticklabels(features, size=8)
        axes[idx].set_title(f"{cluster_names[cluster_id]}\n(N={counts[cluster_names[cluster_id]]})",
                            size=10)

    plt.tight_layout()
    plt.savefig("04_clusters_radar.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ 雷達圖已儲存：04_clusters_radar.png")

    # 視覺化：PCA 散佈圖
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    for cluster_id in sorted_clusters:
        mask = df['Cluster'] == cluster_id
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=cluster_names[cluster_id], s=100, alpha=0.7)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('4 種消費人格 (PCA 視覺化)')
    ax.legend()
    plt.tight_layout()
    plt.savefig("05_clusters_pca.png", dpi=300, bbox_inches='tight')
    print(f"✓ PCA 圖已儲存：05_clusters_pca.png")

    return cluster_names


# ============================================
# Part 7: 模型 4 - NPV 機會成本計算
# ============================================

def future_value_monthly(monthly_amount, years, annual_return):
    """計算每月投資嘅未來價值"""
    months = years * 12
    monthly_return = (1 + annual_return) ** (1/12) - 1
    if monthly_return == 0:
        fv = monthly_amount * months
    else:
        fv = monthly_amount * (((1 + monthly_return) ** months - 1) / monthly_return)
    return {
        'fv': fv,
        'contributed': monthly_amount * months,
        'interest': fv - monthly_amount * months,
        'multiplier': fv / (monthly_amount * months) if monthly_amount > 0 else 0
    }


def npv_analysis(df):
    """NPV 機會成本分析"""
    print("\n" + "="*60)
    print("💰 模型 4：NPV 機會成本計算")
    print("="*60)

    # 計算每位學生嘅 NPV
    df['fv_5pct'] = df['Monthly_Waste'].apply(
        lambda x: future_value_monthly(x, 18, 0.05)['fv']
    )
    df['fv_8pct'] = df['Monthly_Waste'].apply(
        lambda x: future_value_monthly(x, 18, 0.08)['fv']
    )
    df['fv_10pct'] = df['Monthly_Waste'].apply(
        lambda x: future_value_monthly(x, 18, 0.10)['fv']
    )

    # 統計
    print(f"\n📊 NPV 統計 (MOP):")
    print(f"   平均每月浪費: ${df['Monthly_Waste'].mean():.0f}")
    print(f"   平均 18年 NPV (5%): ${df['fv_5pct'].mean():,.0f}")
    print(f"   平均 18年 NPV (8%): ${df['fv_8pct'].mean():,.0f}")
    print(f"   平均 18年 NPV (10%): ${df['fv_10pct'].mean():,.0f}")

    print(f"\n📊 18年 NPV 分佈 (8%):")
    print(df['fv_8pct'].describe().round(0))

    # 消費 vs 投資對比圖
    fig, ax = plt.subplots(figsize=(12, 7))
    years_range = np.arange(0, 19)
    avg_monthly = df['Monthly_Waste'].median()

    consumption_path = avg_monthly * 12 * years_range
    investment_path = [future_value_monthly(avg_monthly, y, 0.08)['fv']
                       for y in years_range]

    ax.plot(years_range, consumption_path / 1000, 'r-o',
            label=f'如果用嚟消費 (累計 MOP ${avg_monthly*12*18:,.0f})',
            linewidth=3, markersize=8)
    ax.plot(years_range, np.array(investment_path) / 1000, 'g-s',
            label=f'如果用嚟投資 S&P 500 (最終 MOP ${investment_path[-1]:,.0f})',
            linewidth=3, markersize=8)
    ax.fill_between(years_range, consumption_path / 1000,
                    np.array(investment_path) / 1000,
                    alpha=0.25, color='gold', label='機會成本區域')

    # 標註最終差距
    final_gap = investment_path[-1] - consumption_path[-1]
    ax.annotate(f'最終差距:\nMOP {final_gap:,.0f}\n([LOCAL ASSET]\n10-30% 首期!)',
                xy=(18, investment_path[-1] / 1000),
                xytext=(12, investment_path[-1] / 1000 * 0.6),
                fontsize=11, color='darkred',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2))

    ax.set_xlabel('年數', fontsize=12)
    ax.set_ylabel('累積金額 (千 MOP)', fontsize=12)
    ax.set_title(f'每月 MOP ${avg_monthly:.0f} 嘅兩種命運 (基於 median)',
                 fontsize=14)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("06_consumption_vs_investment.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ 消費 vs 投資對比圖已儲存：06_consumption_vs_investment.png")

    return df


# ============================================
# Part 8: 模型 5 - Monte Carlo Simulation
# ============================================

def monte_carlo_investment(monthly_amount=300, years=18, n_sim=10000):
    """Monte Carlo 模擬投資組合"""
    mu, sigma = 0.08, 0.18
    final_values = []

    for _ in range(n_sim):
        portfolio = 0
        annual_returns = np.random.normal(mu, sigma, years)
        for ar in annual_returns:
            for month in range(12):
                portfolio += monthly_amount
                portfolio *= (1 + ar / 12)
        final_values.append(portfolio)

    return np.array(final_values)


def monte_carlo_analysis(df):
    """Monte Carlo 模擬分析"""
    print("\n" + "="*60)
    print("🎲 模型 5：Monte Carlo 模擬 (10,000 次)")
    print("="*60)

    avg_monthly = df['Monthly_Waste'].median()
    mc_results = monte_carlo_investment(avg_monthly, years=18, n_sim=10000)

    print(f"\n📊 Monte Carlo 結果 (每月 MOP ${avg_monthly:.0f} × 18年):")
    print(f"   P10 (差情境): MOP {np.percentile(mc_results, 10):,.0f}")
    print(f"   P50 (中位數): MOP {np.median(mc_results):,.0f}")
    print(f"   P90 (好情境): MOP {np.percentile(mc_results, 90):,.0f}")
    print(f"   P5 (最差5%):  MOP {np.percentile(mc_results, 5):,.0f}")
    print(f"   P95 (最好5%): MOP {np.percentile(mc_results, 95):,.0f}")

    # 視覺化
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(mc_results, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.median(mc_results), color='red', linestyle='--', linewidth=2,
               label=f'P50: MOP {np.median(mc_results):,.0f}')
    ax.axvline(np.percentile(mc_results, 10), color='orange', linestyle='--',
               linewidth=2, label=f'P10: MOP {np.percentile(mc_results, 10):,.0f}')
    ax.axvline(np.percentile(mc_results, 90), color='green', linestyle='--',
               linewidth=2, label=f'P90: MOP {np.percentile(mc_results, 90):,.0f}')
    ax.set_xlabel('18年後投資組合價值 (MOP)', fontsize=12)
    ax.set_ylabel('模擬次數', fontsize=12)
    ax.set_title('Monte Carlo 模擬：每月 $X 投資 S&P 500 18年 (10,000次)',
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("07_monte_carlo.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Monte Carlo 圖已儲存：07_monte_carlo.png")

    return mc_results


# ============================================
# Part 9: Knowledge-Willingness Gap 視覺化
# ============================================

def kw_gap_analysis(df):
    """Knowledge-Willingness Gap 分析 + 視覺化"""
    print("\n" + "="*60)
    print("🎯 Knowledge-Willingness Gap 分析")
    print("="*60)

    # 統計
    counts = df['KW_Group'].value_counts()
    pct = counts / len(df) * 100

    print("\n📊 4 種類型分佈:")
    for name, n in counts.items():
        print(f"   {name}: {n} 人 ({pct[name]:.1f}%)")

    # 2x2 矩陣視覺化
    fig, ax = plt.subplots(figsize=(10, 8))

    g3_correct_pct = df['G3_Correct'].mean() * 100
    g4_patient_pct = df['G4_Patient'].mean() * 100

    # 四個象限
    groups = df.groupby(['G3_Correct', 'G4_Patient']).size().reset_index(name='count')

    colors_map = {
        (1, 1): '🟢',
        (1, 0): '🟠',
        (0, 1): '🟡',
        (0, 0): '🔴'
    }
    labels_map = {
        (1, 1): '理性耐性型',
        (1, 0): '識唔等型\n(KW Gap 核心)',
        (0, 1): '直覺耐性型',
        (0, 0): '兩頭空型'
    }

    # 計算每個象限嘅百分比同位置
    for g3_val in [0, 1]:
        for g4_val in [0, 1]:
            subset = df[(df['G3_Correct'] == g3_val) & (df['G4_Patient'] == g4_val)]
            n = len(subset)
            pct_val = n / len(df) * 100

            # 象限中心位置
            x = g3_val * 100 + 50
            y = g4_val * 100 + 50

            color = 'lightgreen' if (g3_val, g4_val) == (1, 1) else \
                    'lightsalmon' if (g3_val, g4_val) == (1, 0) else \
                    'lightyellow' if (g3_val, g4_val) == (0, 1) else \
                    'lightcoral'

            rect = plt.Rectangle((g3_val * 100, g4_val * 100), 100, 100,
                                  facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)

            ax.text(x, y, f"{colors_map[(g3_val, g4_val)]} {labels_map[(g3_val, g4_val)]}\n\n{n} 人 ({pct_val:.1f}%)",
                    ha='center', va='center', fontsize=13,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='black', alpha=0.8))

    # 標註關鍵 gap
    kw_gap_pct = pct.get('識唔等型', 0)
    ax.text(150, 220,
            f'⚠️ Knowledge-Willingness Gap\n「識 NPV 但唔等 18 年」\n{kw_gap_pct:.1f}% 嘅學生',
            fontsize=14, color='darkred', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))

    ax.set_xlim(-5, 205)
    ax.set_ylim(-5, 240)
    ax.set_xticks([0, 100, 200])
    ax.set_xticklabels(['答錯', '答啱', ''], fontsize=12)
    ax.set_yticks([0, 100, 200])
    ax.set_yticklabels(['即時拎', '等 18 年', ''], fontsize=12)
    ax.set_xlabel('G3 NPV 認知測試', fontsize=13)
    ax.set_ylabel('G4 時間偏好測試', fontsize=13)
    ax.set_title('Knowledge-Willingness Gap 矩陣', fontsize=15, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("08_kw_gap_matrix.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ KW Gap 矩陣圖已儲存：08_kw_gap_matrix.png")

    return counts


# ============================================
# Part 10: 自動生成摘要報告
# ============================================

def generate_summary_report(df, alphas, mc_results):
    """生成最終摘要報告"""
    print("\n" + "="*60)
    print("📋 生成摘要報告")
    print("="*60)

    report = f"""
{'='*60}
       [STUDY TITLE]
                      結果摘要報告
{'='*60}

【樣本資料】
  樣本量: N = {len(df)}
  男生比例: {(df['Gender']=='男').mean()*100:.1f}%
  女生比例: {(df['Gender']=='女').mean()*100:.1f}%
  平均每月零用錢: ${df['A7'].mode()[0]}
  平均每月網購: ${df['A8'].mode()[0]}

【信度分析】
"""
    for name, alpha in alphas.items():
        rating = "優秀" if alpha > 0.9 else "良好" if alpha > 0.8 else "可接受"
        report += f"  {name}: α = {alpha:.3f} ({rating})\n"

    report += f"""
【消費行為】
  平均每月因衝動消費而產生嘅閒置物品: MOP ${df['Monthly_Waste'].mean():.0f}
  平均 18 年 NPV (8% 回報): MOP ${df['fv_8pct'].mean():,.0f}
  範圍: MOP ${df['fv_8pct'].min():,.0f} - ${df['fv_8pct'].max():,.0f}

【Monte Carlo 模擬 (10,000 次)】
  P10 (差情境): MOP {np.percentile(mc_results, 10):,.0f}
  P50 (中位數): MOP {np.median(mc_results):,.0f}
  P90 (好情境): MOP {np.percentile(mc_results, 90):,.0f}

【投資認知】
  G3 NPV 認知答對率: {df['G3_Correct'].mean()*100:.1f}%
  G4 延遲滿足率: {df['G4_Patient'].mean()*100:.1f}%

【Knowledge-Willingness Gap】
"""
    kw_counts = df['KW_Group'].value_counts()
    for name, n in kw_counts.items():
        report += f"  {name}: {n} 人 ({n/len(df)*100:.1f}%)\n"

    report += f"""
  ⚠️ 識 NPV 但等唔到 18 年嘅「識唔等型」:
     {kw_counts.get('識唔等型', 0)} 人 ({kw_counts.get('識唔等型', 0)/len(df)*100:.1f}%)

【4 種消費人格】
"""
    cluster_counts = df['Cluster_Name'].value_counts()
    for name, n in cluster_counts.items():
        report += f"  {name}: {n} 人 ({n/len(df)*100:.1f}%)\n"

    report += f"""
{'='*60}
【政策建議】

1️⃣ 中學必修金融素養課
   包括複利計算、機會成本、消費決策模擬

2️⃣ 家庭金融溝通指南
   針對「權威型」家庭，避免產生反效果

3️⃣ 「消費 vs 投資」+「延遲滿足」模擬 App
   幫助學生直觀理解機會成本

4️⃣ 「情緒型消費」早期介入
   針對 ~25% 高風險群提供工作坊

【服務目標政策】
  ✓ [POLICY] [STUDENTS]金融教育計劃
  ✓ [POLICY] 經濟適度多元發展（現代金融業）
  ✓ 數字澳門元 CBDC 推廣

{'='*60}
報告生成時間: 2026-10-02
研究團隊: [RESEARCH PROJECT]參賽隊伍
{'='*60}
"""

    print(report)

    # 儲存到文件
    with open("00_results_summary.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✓ 摘要報告已儲存：00_results_summary.txt")

    # 同時儲存處理後嘅 CSV
    df.to_csv("00_processed_data.csv", index=False, encoding="utf-8-sig")
    print(f"✓ 處理後數據已儲存：00_processed_data.csv")

    return report


# ============================================
# Part 11: 主程式
# ============================================

def main():
    """主程式入口"""
    print("\n" + "="*60)
    print("🚀 [STUDY TITLE]")
    print("   自動分析系統 v1.0")
    print("="*60)

    # 1. 讀取數據
    df = load_data()
    if df is None:
        return

    # 2. 數據清洗 + 計分
    df = clean_and_score(df)

    # 3. 信度分析
    alphas = reliability_analysis(df)

    # 4. SEM 模型
    sem_results = sem_analysis(df)

    # 5. Bootstrap 中介效應
    mediation_results = mediation_analysis(df)

    # 6. Cluster 分析
    cluster_names = cluster_analysis(df)

    # 7. NPV 計算
    df = npv_analysis(df)

    # 8. Monte Carlo
    mc_results = monte_carlo_analysis(df)

    # 9. Knowledge-Willingness Gap
    kw_counts = kw_gap_analysis(df)

    # 10. 摘要報告
    report = generate_summary_report(df, alphas, mc_results)

    print("\n" + "="*60)
    print("✅ 所有分析完成！")
    print("="*60)
    print("\n📁 已生成嘅檔案:")
    print("  00_results_summary.txt    - 摘要報告")
    print("  00_processed_data.csv     - 處理後數據")
    print("  01_reliability.png        - 信度分析")
    print("  02_sem_diagram.png        - SEM 路徑圖")
    print("  03_mediation.png          - 中介效應")
    print("  04_clusters_radar.png     - 集群雷達圖")
    print("  05_clusters_pca.png       - 集群 PCA")
    print("  06_consumption_vs_investment.png - 消費 vs 投資")
    print("  07_monte_carlo.png        - Monte Carlo")
    print("  08_kw_gap_matrix.png      - KW Gap 矩陣")


if __name__ == "__main__":
    main()
