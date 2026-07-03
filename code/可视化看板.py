import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sqlite3
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

conn = sqlite3.connect('music_analysis.db')

# ============================================================
# 图 1-1: DAU 趋势（每日活跃用户数）
# ============================================================
dau = pd.read_sql("""
    SELECT date, COUNT(DISTINCT user_id) AS dau
    FROM actions
    GROUP BY date
    ORDER BY date
""", conn)
dau['date'] = pd.to_datetime(dau['date'])

# ============================================================
# 图 1-2: 行为类型占比
# ============================================================
action_dist = pd.read_sql("""
    SELECT action_name, COUNT(*) AS cnt
    FROM actions
    GROUP BY action_name
    ORDER BY cnt DESC
""", conn)

# ============================================================
# 图 1-3: 用户活跃度分布（每个用户的总行为次数分布）
# ============================================================
user_activity = pd.read_sql("""
    SELECT user_id, COUNT(*) AS total_actions
    FROM actions
    GROUP BY user_id
""", conn)

# 分桶统计
bins = [0, 2, 5, 10, 20, 50, 100, 99999]
labels = ['1-2次', '3-5次', '6-10次', '11-20次', '21-50次', '51-100次', '100次+']
user_activity['bucket'] = pd.cut(user_activity['total_actions'], bins=bins, labels=labels)
bucket_dist = user_activity['bucket'].value_counts().reindex(labels)

# ============================================================
# 拼图: 1 行 3 列
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('用户概览', fontsize=18, fontweight='bold', y=1.02)

# --- 左: DAU 趋势 ---
axes[0].plot(dau['date'], dau['dau'], color='#2E86C1', linewidth=0.8)
axes[0].fill_between(dau['date'], dau['dau'], alpha=0.15, color='#2E86C1')
axes[0].set_title('每日活跃用户数 (DAU)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('活跃用户数')
axes[0].set_xlabel('日期')
# 添加均值线
mean_dau = dau['dau'].mean()
axes[0].axhline(y=mean_dau, color='#E74C3C', linestyle='--', linewidth=1,
                label=f'日均: {mean_dau:,.0f}')
axes[0].legend(fontsize=9)

# --- 中: 行为类型占比 ---
colors_action = ['#2E86C1', '#27AE60', '#F39C12']
wedges, texts, autotexts = axes[1].pie(
    action_dist['cnt'], labels=action_dist['action_name'],
    colors=colors_action, autopct='%1.1f%%',
    explode=(0.02, 0.02, 0.02), startangle=90
)
for at in autotexts:
    at.set_fontsize(11)
axes[1].set_title('用户行为类型分布', fontsize=13, fontweight='bold')

# --- 右: 用户活跃度分布 ---
colors_bar = ['#E74C3C' if l == '1-2次' else '#2E86C1' for l in labels]
axes[2].bar(range(len(bucket_dist)), bucket_dist.values, color=colors_bar, edgecolor='white')
axes[2].set_xticks(range(len(bucket_dist)))
axes[2].set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
axes[2].set_title('用户活跃度分布', fontsize=13, fontweight='bold')
axes[2].set_ylabel('用户数')
# 标注百分比
total_users = bucket_dist.sum()
for i, v in enumerate(bucket_dist.values):
    pct = v / total_users * 100
    axes[2].text(i, v + total_users * 0.005, f'{pct:.1f}%',
                 ha='center', fontsize=8, color='#555')

plt.tight_layout()
plt.savefig('dashboard_page1_user_overview.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
print("✅ 第1页已保存: dashboard_page1_user_overview.png")

# ============================================================
# 图 2-1: 24小时 × 7天 收听热力图
# ============================================================
cross_df = pd.read_sql("""
    SELECT
        CASE CAST(STRFTIME('%w', date) AS INTEGER)
            WHEN 0 THEN '周日' WHEN 1 THEN '周一' WHEN 2 THEN '周二'
            WHEN 3 THEN '周三' WHEN 4 THEN '周四' WHEN 5 THEN '周五'
            WHEN 6 THEN '周六'
        END AS weekday,
        hour,
        COUNT(*) AS play_count
    FROM actions WHERE action_type = 1
    GROUP BY weekday, hour
""", conn)

# 转为透视表
heatmap_data = cross_df.pivot_table(
    values='play_count', index='weekday', columns='hour', aggfunc='sum'
)
weekday_order = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
heatmap_data = heatmap_data.reindex(weekday_order)

# ============================================================
# 图 2-2: 热门歌曲 TOP10
# ============================================================
top_songs = pd.read_sql("""
    SELECT song_id, COUNT(*) AS play_count
    FROM actions WHERE action_type = 1
    GROUP BY song_id
    ORDER BY play_count DESC
    LIMIT 10
""", conn)

# ============================================================
# 拼图: 2 行 1 列
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle('收听行为分析', fontsize=18, fontweight='bold', y=1.02)

# --- 上: 24h×7天 热力图 ---
import matplotlib.colors as mcolors
im = axes[0].imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto',
                     norm=mcolors.LogNorm())  # 对数归一化，突出差异
axes[0].set_xticks(range(24))
axes[0].set_xticklabels([f'{h}:00' for h in range(24)], rotation=45, fontsize=8)
axes[0].set_yticks(range(7))
axes[0].set_yticklabels(heatmap_data.index, fontsize=10)
axes[0].set_title('24小时 × 7天 收听热力图（对数色阶）', fontsize=13, fontweight='bold')
axes[0].set_xlabel('小时（北京时间）', fontsize=11)

# 标注峰值和谷值
peak_idx = np.unravel_index(heatmap_data.values.argmax(), heatmap_data.values.shape)
trough_idx = np.unravel_index(heatmap_data.values.argmin(), heatmap_data.values.shape)
peak_day, peak_hour = heatmap_data.index[peak_idx[0]], peak_idx[1]
axes[0].annotate('★ 峰值', (peak_idx[1], peak_idx[0]),
                 color='#E74C3C', fontsize=10, fontweight='bold',
                 ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# 添加 colorbar
cbar = plt.colorbar(im, ax=axes[0], shrink=0.85)
cbar.set_label('播放次数（对数）', fontsize=10)

# --- 下: 热门歌曲 TOP10 水平柱状图 ---
# 用 song_id 前 8 位作为标签
song_labels = [f'歌曲 {sid[:8]}' for sid in top_songs['song_id']]
colors_top = ['#E74C3C' if i == 0 else '#2E86C1' for i in range(10)]
bars = axes[1].barh(range(10), top_songs['play_count'].values[::-1],
                     color=colors_top[::-1], edgecolor='white', height=0.7)
axes[1].set_yticks(range(10))
axes[1].set_yticklabels(song_labels[::-1], fontsize=9)
axes[1].set_title('热门歌曲 TOP10（按播放次数）', fontsize=13, fontweight='bold')
axes[1].set_xlabel('播放次数')
# 标注数值
for i, (bar, val) in enumerate(zip(bars, top_songs['play_count'].values[::-1])):
    axes[1].text(bar.get_width() + top_songs['play_count'].max() * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f'{val:,}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('dashboard_page2_behavior.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
print("✅ 第2页已保存: dashboard_page2_behavior.png")

# ============================================================
# 先重新计算 RFM（沿用 Step 3 的逻辑）
# ============================================================
ref_date = pd.read_sql("SELECT MAX(date) AS ref FROM actions", conn)['ref'][0]

rfm = pd.read_sql("""
    SELECT
        user_id,
        CAST(JULIANDAY(?) - JULIANDAY(MAX(date)) AS INTEGER) AS R,
        COUNT(*) AS F,
        COUNT(*) * 4 AS M
    FROM actions WHERE action_type = 1
    GROUP BY user_id
""", conn, params=[str(ref_date)])

# 自定义阈值分箱
rfm['R_score'] = pd.cut(rfm['R'], bins=[-1, 3, 7, 14, 30, 999], labels=[5,4,3,2,1]).astype(int)
rfm['F_score'] = pd.cut(rfm['F'], bins=[0, 2, 10, 30, 100, 99999], labels=[1,2,3,4,5]).astype(int)
rfm['M_score'] = pd.cut(rfm['M'], bins=[0, 8, 40, 120, 400, 999999], labels=[1,2,3,4,5]).astype(int)

def classify_user(row):
    r, f, m = row['R_score'], row['F_score'], row['M_score']
    if r >= 4 and f >= 4 and m >= 4:
        return '高价值用户'
    elif r >= 3 and f >= 3:
        return '活跃用户'
    elif r <= 2 and f >= 3:
        return '流失风险用户'
    else:
        return '沉默用户'

rfm['segment'] = rfm.apply(classify_user, axis=1)

# ============================================================
# 图 3-1: RFM 分群饼图
# 图 3-2: R × F 散点图（按分群着色）
# 图 3-3: 各群体人均指标对比
# ============================================================
fig = plt.figure(figsize=(18, 14))
fig.suptitle('RFM 用户分群画像', fontsize=18, fontweight='bold', y=0.98)

colors_seg = {'高价值用户': '#27AE60', '活跃用户': '#2E86C1',
              '流失风险用户': '#E74C3C', '沉默用户': '#95A5A6'}
seg_order = ['高价值用户', '活跃用户', '流失风险用户', '沉默用户']

# --- 左上: 分群饼图 ---
ax1 = fig.add_subplot(2, 2, 1)
seg_counts = rfm['segment'].value_counts().reindex(seg_order)
wedges, texts, autotexts = ax1.pie(
    seg_counts.values, labels=seg_counts.index,
    colors=[colors_seg[s] for s in seg_order],
    autopct='%1.1f%%', explode=(0.05, 0.02, 0.05, 0.02),
    startangle=90
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight('bold')
ax1.set_title('用户价值分群占比', fontsize=13, fontweight='bold')

# --- 右上: R × F 散点图 ---
ax2 = fig.add_subplot(2, 2, 2)
for seg in seg_order:
    subset = rfm[rfm['segment'] == seg]
    n_sample = min(3000, len(subset))
    subset_sample = subset.sample(n_sample, random_state=42)
    ax2.scatter(subset_sample['R'], subset_sample['F'],
                c=colors_seg[seg], label=seg, alpha=0.4, s=8)
ax2.set_xlabel('R — 最近播放距今天数（越少越活跃）→', fontsize=10)
ax2.set_ylabel('F — 播放总次数', fontsize=10)
ax2.set_title('R × F 用户分布散点图（采样）', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.invert_xaxis()
ax2.set_xlim(-5, rfm['R'].quantile(0.95))  # 截断极端值

# --- 下半: 群体对比柱状图 ---
ax3 = fig.add_subplot(2, 1, 2)
seg_stats = rfm.groupby('segment').agg(
    用户数=('user_id', 'count'),
    人均播放=('F', 'mean'),
    人均时长_分钟=('M', 'mean'),
    平均R=('R', 'mean')
).reindex(seg_order)

x = np.arange(len(seg_order))
width = 0.25

bars1 = ax3.bar(x - width, seg_stats['人均播放'].values, width,
                label='人均播放次数', color='#2E86C1', edgecolor='white')
bars2 = ax3.bar(x, seg_stats['人均时长_分钟'].values, width,
                label='人均收听时长(分钟)', color='#27AE60', edgecolor='white')
bars3 = ax3.bar(x + width, seg_stats['平均R'].values, width,
                label='平均距今天数(R)', color='#E74C3C', edgecolor='white')

# 标注数值
for bars_obj in [bars1, bars2, bars3]:
    for bar in bars_obj:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., h + max(seg_stats['人均播放'].max() * 0.01, 1),
                 f'{h:.0f}', ha='center', va='bottom', fontsize=8)

ax3.set_xticks(x)
ax3.set_xticklabels(seg_order, fontsize=11)
ax3.set_title('各群体核心指标对比', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('dashboard_page3_rfm.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
print("✅ 第3页已保存: dashboard_page3_rfm.png")