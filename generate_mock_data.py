"""
模擬數據生成器 - 用嚟測試分析腳本

當你未收集到真實數據時，可以用呢個腳本生成模擬數據嚟測試。
運行：python generate_mock_data.py
會產生 questionnaire_responses.csv，包含 50 份模擬問卷。
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# 模擬 50 份問卷
N = 50

# 生成模擬數據
data = {
    # A 部分
    'A1': np.random.choice(['男', '女'], N, p=[0.48, 0.52]),
    'A2': np.random.choice(['初二', '初三', '高一', '高二', '高三'], N),
    'A3': np.random.choice(['公立學校', '私立學校', '國際學校'], N, p=[0.6, 0.3, 0.1]),
    'A4': np.random.choice(['< 1 小時', '1–2 小時', '2–3 小時', '3–4 小時', '> 4 小時'], N,
                            p=[0.1, 0.2, 0.3, 0.25, 0.15]),
    'A5': np.random.choice(['從未見過', '1–3 次', '3–5 次', '5–10 次', '> 10 次'], N,
                            p=[0.05, 0.2, 0.3, 0.3, 0.15]),
    'A6': np.random.choice(['有買，而且成日會傾', '有買，但唔係成日傾',
                            '有買，但我唔清楚細節', '無買', '唔知道'], N,
                            p=[0.15, 0.25, 0.25, 0.2, 0.15]),
    'A7': np.random.choice(['< $100', '$100–$300', '$300–$500', '$500–$1,000',
                            '$1,000–$2,000', '> $2,000'], N,
                            p=[0.15, 0.3, 0.25, 0.15, 0.1, 0.05]),
    'A8': np.random.choice(['從不網購', '< $100', '$100–$300', '$300–$500',
                            '$500–$1,000', '$1,000–$2,000', '> $2,000'], N,
                            p=[0.05, 0.15, 0.3, 0.25, 0.15, 0.07, 0.03]),
    'A9': ['淘寶 / 天貓, 拼多多'] * N,
    'A10': np.random.choice(['MPay（澳門錢包）', '支付寶', '微信支付', '信用卡', '現金'], N,
                             p=[0.4, 0.3, 0.15, 0.05, 0.1]),
}

# B 部分：Family Communication Patterns (1-7 Likert)
# Conversation Orientation
for i in range(1, 6):
    data[f'B{i}'] = np.random.randint(2, 7, N)  # 中等偏高

# Conformity Orientation
for i in range(6, 11):
    data[f'B{i}'] = np.random.randint(2, 6, N)  # 中等

# Parental Financial Discussion
for i in range(11, 17):
    data[f'B{i}'] = np.random.randint(2, 7, N)

# Family Financial Conflict
for i in range(17, 21):
    data[f'B{i}'] = np.random.randint(1, 6, N)

# C 部分：Impulse Buying
for i in range(1, 9):
    data[f'C{i}'] = np.random.randint(3, 7, N)  # 中高衝動

# Self-Control
for i in range(9, 13):
    data[f'C{i}'] = np.random.randint(2, 6, N)

# D 部分
data['D1'] = np.random.choice(['0 次', '1–3 次', '3–5 次', '5–10 次', '10–20 次'], N,
                              p=[0.05, 0.25, 0.3, 0.25, 0.15])
data['D2'] = np.random.randint(100, 3000, N)
data['D3'] = np.random.choice(['有，而且好滿', '有，但唔係好多', '冇', '唔確定'], N,
                              p=[0.2, 0.4, 0.3, 0.1])
data['D4'] = np.random.choice(['從未', '1–3 次', '3–5 次', '5–10 次', '> 10 次'], N,
                              p=[0.1, 0.3, 0.3, 0.2, 0.1])
data['D5'] = np.random.choice(['從未', '1–3 次', '3–5 次', '5–10 次', '> 10 次'], N,
                              p=[0.1, 0.3, 0.3, 0.2, 0.1])
data['D6'] = np.random.choice(['從未', '有過 1–2 次', '有過 3–5 次', '成日都會'], N,
                              p=[0.3, 0.4, 0.2, 0.1])
data['D7'] = np.random.choice(['有，好多', '有，少少', '冇', '唔確定'], N,
                              p=[0.15, 0.45, 0.3, 0.1])
data['D8'] = np.random.choice(['A) 即刻買', 'B) 加入購物車', 'C) 唔買', 'D) 其他'], N,
                              p=[0.3, 0.4, 0.25, 0.05])
data['D9'] = np.random.choice(['A) 退貨', 'B) 保留', 'C) 放埋一邊', 'D) 其他'], N,
                              p=[0.5, 0.2, 0.25, 0.05])
data['D10'] = np.random.choice(['A) 即刻搵', 'B) 睇下朋友', 'C) 唔理', 'D) 問朋友'], N,
                               p=[0.2, 0.3, 0.35, 0.15])
# D11 重要：閒置物品價值
data['D11'] = np.random.choice(['$0', '$1–$100', '$101–$500', '$501–$1,000',
                                 '$1,001–$2,000', '$2,001–$5,000'], N,
                                 p=[0.1, 0.25, 0.3, 0.2, 0.1, 0.05])
data['D12'] = ['屋企堆埋一邊'] * N

# E 部分
for i in range(1, 8):
    data[f'E{i}'] = np.random.randint(1, 7, N)

# F 部分
for i in range(1, 6):
    data[f'F{i}'] = np.random.randint(2, 6, N)

# G 部分
data['G1'] = np.random.choice(['完全唔知', '聽過但唔識解釋', '識少少', '完全明白'], N,
                              p=[0.15, 0.4, 0.35, 0.1])
data['G2'] = np.random.choice(['差唔多 $1,000', '$2,000–$3,000', '$5,000–$10,000',
                                '$10,000 以上', '唔知道'], N,
                                p=[0.2, 0.3, 0.25, 0.1, 0.15])
# G3 NPV 認知測試 - 約 40% 答啱
data['G3'] = np.random.choice(['$30,000 以下', '$30,000–$60,000', '$60,000–$120,000',
                                '$120,000–$200,000', '$200,000 以上', '唔知道'], N,
                                p=[0.15, 0.2, 0.2, 0.2, 0.1, 0.15])
# G4 時間偏好測試 - 約 35% 揀等 18 年
data['G4'] = np.random.choice(['(A) 今日就拎 $1,000', '(B) 等 18 年後拎 $2,500'], N,
                              p=[0.65, 0.35])
data['G5'] = ['樓宇首期（澳門一層細單位）'] * N
data['G6'] = ['基金 / ETF'] * N
data['G7'] = np.random.choice(['1–3%', '3–6%', '6–10%', '10–20%', '唔知道'], N,
                              p=[0.1, 0.25, 0.3, 0.15, 0.2])
data['G8'] = np.random.choice(['從未', '諗過但唔識做', '試過'], N,
                              p=[0.4, 0.45, 0.15])
data['G9'] = np.random.randint(3, 7, N)
data['G10'] = np.random.randint(4, 7, N)

# 建立 DataFrame
df = pd.DataFrame(data)

# 儲存
df.to_csv('questionnaire_responses.csv', index=False, encoding='utf-8-sig')
print(f"✓ 已生成 {N} 份模擬問卷：questionnaire_responses.csv")
print(f"\n接住可以運行：python auto_analysis.py")
print(f"就會自動讀取呢個 CSV 並執行所有 5 大模型分析")
