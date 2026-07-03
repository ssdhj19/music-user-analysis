import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

conn = sqlite3.connect('music_analysis.db')

# ============================================================
# 3.1.1 从 SQL 计算每个用户的 R、F、M 原始值
# ============================================================

# 参考日：数据集的最后一天
ref_date = pd.read_sql("SELECT MAX(date) AS ref FROM actions", conn)['ref'][0]
print(f"参考日期（数据最后一天）: {ref_date}")

# 用 SQL 一步算出每个用户的 R、F、M
# R = 参考日 - 用户最后活跃日（天数）
# F = 用户总播放次数（只看播放行为）
# M = 用户预估收听总时长（播放次数 × 4分钟）
rfm_query = """
SELECT
    user_id,
    -- R: 距今天数 = 参考日 - 用户最后一次播放日期
    CAST(JULIANDAY(?) - JULIANDAY(MAX(date)) AS INTEGER) AS R,
    -- F: 播放总次数
    COUNT(*) AS F,
    -- M: 预估收听总时长（分钟）= 播放次数 × 4
    COUNT(*) * 4 AS M
FROM actions
WHERE action_type = 1    -- 只看播放
GROUP BY user_id
"""

rfm = pd.read_sql(rfm_query, conn, params=[str(ref_date)])
print(f"用户总数: {len(rfm):,}")
print(f"\n原始 RFM 描述统计:")
print(rfm[['R', 'F', 'M']].describe())

# ============================================================
# 3.1.2 检查分布，确认 qcut 参数
# ============================================================
print(f"\nR 的分位数:")
for q in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    print(f"  {q:.0%}: {rfm['R'].quantile(q):.0f} 天")

print(f"\nF 的分位数:")
for q in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    print(f"  {q:.0%}: {rfm['F'].quantile(q):.0f} 次")

print(f"\nM 的分位数:")
for q in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    print(f"  {q:.0%}: {rfm['M'].quantile(q):.0f} 分钟")

# ============================================================
# 3.1.3 自定义业务阈值分箱（基于数据分布 + 业务含义）
# ============================================================

# --- R: 最近播放距今天数（越小越好）---
# 阈值含义: 5分=本周内, 4分=本周内, 3分=两周内, 2分=一个月内, 1分=超过一个月
R_bins = [-1, 3, 7, 14, 30, 999]      # 0-3天/4-7天/8-14天/15-30天/31天+
R_labels = [5, 4, 3, 2, 1]             # 越近分越高
rfm['R_score'] = pd.cut(rfm['R'], bins=R_bins, labels=R_labels).astype(int)

# --- F: 播放总次数（越大越好）---
# 阈值含义: 1分=只来过1-2次(试一下就走了)
#           2分=偶尔用(3-10次, 6个月里约每月1-2次)
#           3分=定期用(11-30次, 约每周1次)
#           4分=常用(31-100次, 约每周多次)
#           5分=重度(100次+, 几乎每天)
F_bins = [0, 2, 10, 30, 100, 99999]
F_labels = [1, 2, 3, 4, 5]
rfm['F_score'] = pd.cut(rfm['F'], bins=F_bins, labels=F_labels).astype(int)

# --- M: 预估收听总时长（分钟，越大越好）---
# M = F × 4分钟，阈值与 F 同步放大4倍
M_bins = [0, 8, 40, 120, 400, 999999]
M_labels = [1, 2, 3, 4, 5]
rfm['M_score'] = pd.cut(rfm['M'], bins=M_bins, labels=M_labels).astype(int)

print("\n各维度评分分布（自定义业务阈值）:")
for col in ['R_score', 'F_score', 'M_score']:
    print(f"\n{col}:")
    dist = rfm[col].value_counts().sort_index()
    for score, cnt in dist.items():
        print(f"  {score}分: {cnt:,} 人 ({cnt/len(rfm)*100:.1f}%)")

# ============================================================
# 3.1.4 业务规则: 把 R/F/M 组合归并为 4 个群体
# ============================================================

def classify_user(row):
    r, f, m = row['R_score'], row['F_score'], row['M_score']

    # 高价值用户: 最近一周内活跃 + 播放30次以上（每周多次）+ 深度消费
    #   → 这是产品的核心用户，贡献了不成比例的价值
    if r >= 4 and f >= 4 and m >= 4:
        return '高价值用户'

    # 活跃用户: 最近两周内活跃 + 播放10次以上（至少每周来一次）
    #   → 有稳定使用习惯，是增长的"基本盘"
    elif r >= 3 and f >= 3:
        return '活跃用户'

    # 流失风险用户: 超过两周没来 + 但曾经常用（播放10次以上）
    #   → 这是"最可惜"的群体，召回优先级最高
    elif r <= 2 and f >= 3:
        return '流失风险用户'

    # 沉默用户: 低频 + 长期不来，或只来过一两次就走了
    #   → 占比最大的群体，需要用新功能或新内容重新激活
    else:
        return '沉默用户'

rfm['segment'] = rfm.apply(classify_user, axis=1)

# ============================================================
# 3.1.5 分群统计
# ============================================================
print("\n" + "=" * 60)
print("RFM 用户分群结果")
print("=" * 60)

segment_stats = rfm.groupby('segment').agg(
    用户数=('user_id', 'count'),
    用户占比=('user_id', lambda x: f"{len(x)/len(rfm)*100:.1f}%"),
    平均R_天数=('R', 'mean'),
    平均F_次数=('F', 'mean'),
    平均M_分钟=('M', 'mean'),
    F总和=('F', 'sum')
).reindex(['高价值用户', '活跃用户', '流失风险用户', '沉默用户'])

# 计算各群体播放总量占比
total_F = rfm['F'].sum()
segment_stats['播放量占比'] = (segment_stats['F总和'] / total_F * 100).round(1).astype(str) + '%'
segment_stats = segment_stats.drop(columns=['F总和'])

print(segment_stats)

# 核心发现
gao = segment_stats.loc['高价值用户']
liushi = segment_stats.loc['流失风险用户']
print(f"\n🔑 核心发现:")
print(f"  高价值用户占 {gao['用户占比']}，贡献 {gao['播放量占比']} 总播放量")
print(f"  流失风险用户占 {liushi['用户占比']}，曾是高频用户但最近已流失")

# ============================================================
# 3.1.6 可视化：分群饼图 + RFM 散点图
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1: 用户分群占比（饼图）
colors = {'高价值用户': '#27AE60', '活跃用户': '#2E86C1', '流失风险用户': '#E74C3C', '沉默用户': '#95A5A6'}
seg_counts = rfm['segment'].value_counts().reindex(['高价值用户', '活跃用户', '流失风险用户', '沉默用户'])
axes[0].pie(seg_counts.values, labels=seg_counts.index, autopct='%1.1f%%',
            colors=[colors[s] for s in seg_counts.index], startangle=90)
axes[0].set_title('用户价值分群占比', fontsize=14, fontweight='bold')

# 图2: R vs F 散点图（按分群着色）
for seg in ['高价值用户', '活跃用户', '流失风险用户', '沉默用户']:
    subset = rfm[rfm['segment'] == seg]
    # 采样防止点太多
    sample_n = min(5000, len(subset))
    subset_sample = subset.sample(sample_n, random_state=42)
    axes[1].scatter(subset_sample['R'], subset_sample['F'],
                    c=colors[seg], label=seg, alpha=0.5, s=10)

axes[1].set_xlabel('R — 最近播放距今天数（越小越好）→', fontsize=11)
axes[1].set_ylabel('F — 播放总次数（越大越好）→', fontsize=11)
axes[1].set_title('R × F 用户分布（散点图）', fontsize=14, fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].invert_xaxis()  # R 越小越好，反转 X 轴

plt.tight_layout()
plt.savefig('rfm_segments.png', dpi=150, bbox_inches='tight')
plt.show()
print("图表已保存: rfm_segments.png")