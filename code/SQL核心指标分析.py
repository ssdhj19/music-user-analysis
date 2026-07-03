import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sqlite3
import pandas as pd

conn = sqlite3.connect('music_analysis.db')
print("数据库已创建: music_analysis.db")

songs_cols = ['song_id', 'artist_id', 'publish_time', 'song_init_plays', 'language', 'gender']
songs = pd.read_csv('E:\厦门大学数字经济专硕\就业\项目\音乐平台用户数据分析\数据/mars_tianchi_songs.csv', names=songs_cols, header=None)
songs['publish_date'] = pd.to_datetime(songs['publish_time'].astype(str), format='%Y%m%d', errors='coerce')

songs.to_sql('songs', conn, if_exists='replace', index=False)
print(f"歌曲表已导入: {len(songs):,} 行")

actions_cols = ['user_id', 'song_id', 'gmt_create', 'action_type', 'ds']

# 先删旧表（如果有）
conn.execute("DROP TABLE IF EXISTS actions")

chunk_size = 100_000
total_imported = 0

for i, chunk in enumerate(pd.read_csv('E:\厦门大学数字经济专硕\就业\项目\音乐平台用户数据分析\数据/mars_tianchi_user_actions.csv',
    names=actions_cols, header=None, chunksize=chunk_size)):

    # 数据清洗（与 Step 1 保持一致）
    chunk = chunk.drop_duplicates()
    chunk['gmt_datetime'] = pd.to_datetime(chunk['gmt_create'], unit='s', errors='coerce')
    # ⚠️ 时区修正: Unix 时间戳是 UTC，中国用户在北京时间(UTC+8)，+8小时后才是本地时间
    chunk['gmt_datetime_beijing'] = chunk['gmt_datetime'] + pd.Timedelta(hours=8)
    chunk['action_name'] = chunk['action_type'].map({1: '播放', 2: '下载', 3: '收藏'})
    chunk['date'] = chunk['gmt_datetime_beijing'].dt.date
    chunk['hour'] = chunk['gmt_datetime_beijing'].dt.hour
    chunk['day_of_week'] = chunk['gmt_datetime_beijing'].dt.dayofweek

    # 分批写入（if_exists='append' 第一批之后追加）
    chunk.to_sql('actions', conn, if_exists='append', index=False)
    total_imported += len(chunk)

    if (i + 1) % 50 == 0:
        print(f"  已导入 {total_imported:,} 行...")

print(f"行为表导入完成: {total_imported:,} 行")

conn.execute("CREATE INDEX IF NOT EXISTS idx_actions_user ON actions(user_id)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_actions_date ON actions(date)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_actions_type ON actions(action_type)")
print("索引创建完成")

# 分析1 每个用户收听top3歌曲
query_top3 = """
WITH user_song_plays AS (
    SELECT user_id, song_id, COUNT(*) AS play_count
    FROM actions WHERE action_type = 1
    GROUP BY user_id, song_id
),
user_song_rank AS (
    SELECT user_id, song_id, play_count,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY play_count DESC) AS rn
    FROM user_song_plays
)
SELECT user_id, song_id, play_count, rn AS rank
FROM user_song_rank
WHERE rn <= 3
ORDER BY user_id, rn
"""

# 只看前 20 个用户的结果做验证
top3_sample = pd.read_sql(query_top3 + " LIMIT 60", conn)
print(top3_sample.head(20))

# 统计：每个用户 Top1 歌曲平均占该用户总播放量的比例
# 先算每个用户的总播放量
user_total = pd.read_sql("""
    SELECT user_id, COUNT(*) AS total_plays
    FROM actions WHERE action_type = 1
    GROUP BY user_id
""", conn)

# 再取 Top3 数据中的 Top1
top1 = pd.read_sql("""
    WITH user_song_plays AS (
        SELECT user_id, song_id, COUNT(*) AS play_count
        FROM actions WHERE action_type = 1
        GROUP BY user_id, song_id
    ),
    ranked AS (
        SELECT user_id, song_id, play_count,
               ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY play_count DESC) AS rn
        FROM user_song_plays
    )
    SELECT user_id, play_count AS top1_plays
    FROM ranked WHERE rn = 1
""", conn)

# 合并计算占比
merged = top1.merge(user_total, on='user_id')
merged['top1_ratio'] = merged['top1_plays'] / merged['total_plays']
print(f"\nTop1 歌曲占比分布: mean={merged['top1_ratio'].mean():.2%}, median={merged['top1_ratio'].median():.2%}")


# 分析2 留存率
query_retention_precise = """
WITH user_first_day AS (
    SELECT user_id, MIN(date) AS first_date
    FROM actions GROUP BY user_id
),
user_active_days AS (
    SELECT DISTINCT user_id, date FROM actions
),
retention_detail AS (
    SELECT
        ufd.user_id, ufd.first_date, uad.date AS active_date,
        CAST(JULIANDAY(uad.date) - JULIANDAY(ufd.first_date) AS INTEGER) AS day_diff
    FROM user_first_day ufd
    LEFT JOIN user_active_days uad
        ON ufd.user_id = uad.user_id AND uad.date >= ufd.first_date
)
SELECT
    -- 次日留存: 只算首日 ≤ 2015-08-29 的用户
    COUNT(DISTINCT CASE WHEN first_date <= '2015-08-29' THEN user_id END) AS day1_total,
    COUNT(DISTINCT CASE WHEN day_diff = 1 AND first_date <= '2015-08-29'
        THEN user_id END) AS day1_retained,
    ROUND(COUNT(DISTINCT CASE WHEN day_diff = 1 AND first_date <= '2015-08-29'
        THEN user_id END) * 100.0 /
        NULLIF(COUNT(DISTINCT CASE WHEN first_date <= '2015-08-29'
        THEN user_id END), 0), 1) AS day1_pct,
    -- 7日留存: 只算首日 ≤ 2015-08-23 的用户
    COUNT(DISTINCT CASE WHEN first_date <= '2015-08-23' THEN user_id END) AS day7_total,
    COUNT(DISTINCT CASE WHEN day_diff = 7 AND first_date <= '2015-08-23'
        THEN user_id END) AS day7_retained,
    ROUND(COUNT(DISTINCT CASE WHEN day_diff = 7 AND first_date <= '2015-08-23'
        THEN user_id END) * 100.0 /
        NULLIF(COUNT(DISTINCT CASE WHEN first_date <= '2015-08-23'
        THEN user_id END), 0), 1) AS day7_pct,
    -- 额外: 全部用户数
    COUNT(DISTINCT user_id) AS total_users
FROM retention_detail
"""

retention2 = pd.read_sql(query_retention_precise, conn)
print("\n=== 留存率（过滤了数据窗口不完整的用户）===")
print(f"总用户数: {retention2['total_users'][0]:,}")
print(f"次日留存: {retention2['day1_pct'][0]}%  "
      f"({retention2['day1_retained'][0]:,} / {retention2['day1_total'][0]:,})")
print(f"7日留存:  {retention2['day7_pct'][0]}%  "
      f"({retention2['day7_retained'][0]:,} / {retention2['day7_total'][0]:,})")

# 分析3 按小时 × 星期维度的收听高峰识别
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 查询：按小时统计
hourly_df = pd.read_sql("""
    SELECT hour, COUNT(*) AS play_count, COUNT(DISTINCT user_id) AS unique_users
    FROM actions WHERE action_type = 1
    GROUP BY hour ORDER BY hour
""", conn)

# 查询：按星期统计
weekly_df = pd.read_sql("""
    SELECT
        CASE CAST(STRFTIME('%w', date) AS INTEGER)
            WHEN 0 THEN '周日' WHEN 1 THEN '周一' WHEN 2 THEN '周二'
            WHEN 3 THEN '周三' WHEN 4 THEN '周四' WHEN 5 THEN '周五'
            WHEN 6 THEN '周六'
        END AS weekday,
        COUNT(*) AS play_count
    FROM actions WHERE action_type = 1
    GROUP BY weekday
""", conn)

# 查询：小时 × 星期交叉表
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
    ORDER BY hour
""", conn)

# 打印结果
print("=== 按小时分布 ===")
print(hourly_df.to_string())
print(f"\n峰值时段: {hourly_df.loc[hourly_df['play_count'].idxmax(), 'hour']}:00")
print(f"谷值时段: {hourly_df.loc[hourly_df['play_count'].idxmin(), 'hour']}:00")

print("\n=== 按星期分布 ===")
print(weekly_df.to_string())

print("\n=== 核心结论 ===")
peak_hour = hourly_df.loc[hourly_df['play_count'].idxmax(), 'hour']
trough_hour = hourly_df.loc[hourly_df['play_count'].idxmin(), 'hour']
peak_to_trough = hourly_df['play_count'].max() / hourly_df['play_count'].min()
print(f"峰值时段: {peak_hour}:00 ({hourly_df['play_count'].max():,} 次)")
print(f"谷值时段: {trough_hour}:00 ({hourly_df['play_count'].min():,} 次)")
print(f"峰谷比: {peak_to_trough:.1f}:1")
print(f"日均播放: {hourly_df['play_count'].sum() / 183:,.0f} 次")