import pandas as pd
import numpy as np

print("=" * 60)
print("1. 加载数据")
print("=" * 60)
print('数据加载中...')
# 歌曲元数据
songs_cols = ['song_id', 'artist_id', 'publish_time', 'song_init_plays', 'language', 'gender']
songs = pd.read_csv('数据/mars_tianchi_songs.csv', names=songs_cols, header=None)

# 用户行为数据（1.4GB）
actions_cols = ['user_id', 'song_id', 'gmt_create', 'action_type', 'ds']
actions = pd.read_csv('数据/mars_tianchi_user_actions.csv', names=actions_cols, header=None)
print('加载完毕。')

# 2.歌曲表数据清洗
print("\n" + "=" * 60)
print("2. 数据清洗 — 歌曲表")
print("=" * 60)

print(f"\n缺失值统计:\n{songs.isnull().sum()}")
print(f"\n重复行数: {songs.duplicated().sum()}")

# publish_time 转换 为日期类数据
songs['publish_date'] = pd.to_datetime(songs['publish_time'].astype(str), format='%Y%m%d', errors='coerce')
print(f"\n发布日期范围: {songs['publish_date'].min()} ~ {songs['publish_date'].max()}")
print(f"日期解析失败数: {songs['publish_date'].isna().sum()}")

# language 映射【目前未知数字与语种的关系】

# gender 映射
gender_map = {3: '组合/乐队', 1: '男', 2: '女', 4:'未知'}
songs['gender_name'] = songs['gender'].map(gender_map)
print(f"\n性别分布:\n{songs['gender_name'].value_counts()}")

# 3.行为表数据清洗
print("\n" + "=" * 60)
print("3. 数据清洗 — 行为表")
print("=" * 60)

print(f"\n缺失值统计:\n{actions.isnull().sum()}")
print(f"\n重复行数: {actions.duplicated().sum()}")
# 去重
act = actions.drop_duplicates().copy()
print(f"去重后行数: {act.shape[0]:,}")

# action_type 映射
action_map = {1: '播放', 2: '下载', 3: '收藏'}
act['action_name'] = act['action_type'].map(action_map)
print(f"\n行为类型分布:\n{act['action_name'].value_counts()}")
print(f"占比:\n{(act['action_name'].value_counts(normalize=True) * 100).round(2)}")

# 时间戳转换
act['gmt_datetime'] = pd.to_datetime(act['gmt_create'], unit='s', errors='coerce')
print(f"\n时间范围: {act['gmt_datetime'].min()} ~ {act['gmt_datetime'].max()}")
print(f"时间解析失败数: {act['gmt_datetime'].isna().sum()}")

# ds 校验
act['ds_str'] = act['ds'].astype(str)
print(f"DS 分区日期范围: {act['ds_str'].min()} ~ {act['ds_str'].max()}")


# 4.探索性分析
print("\n" + "=" * 60)
print("4. 探索性分析（EDA）")
print("=" * 60)

# 4.1 用户行为概览
print("\n--- 4.1 用户行为概览 ---")
unique_users = act['user_id'].nunique()
unique_songs_acted = act['song_id'].nunique()
print(f"用户数={unique_users:,}, 歌曲数={unique_songs_acted:,}")
print(f"人均行为次数: {len(act)/unique_users:.1f}")
act_per_user = act.groupby('user_id').size()
print(f"用户行为次数分布: mean={act_per_user.mean():.1f}, median={act_per_user.median():.0f}, max={act_per_user.max():,}")

# 4.2 播放行为占比
print("\n--- 4.2 各行为类型占比 ---")
for atype, aname in action_map.items():
    cnt = (act['action_type'] == atype).sum()
    users = act[act['action_type'] == atype]['user_id'].nunique()
    songs_cnt = act[act['action_type'] == atype]['song_id'].nunique()
    print(f"  {aname}: {cnt:,} 次 | {users:,} 用户 | {songs_cnt:,} 歌曲")

# 4.3 时间维度分析
print("\n--- 4.3 时间维度分析 ---")
act['hour'] = act['gmt_datetime'].dt.hour
act['day_of_week'] = act['gmt_datetime'].dt.dayofweek  # 0=周一
act['date'] = act['gmt_datetime'].dt.date

# 按小时分布
hourly = act.groupby('hour').size()
print(f"\n按小时播放量分布:")
for h in range(24):
    bar = '█' * int(hourly.get(h, 0) / hourly.max() * 40)
    print(f"  {h:02d}:00 | {bar} {hourly.get(h, 0):,}")

# 按星期分布
weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
daily = act.groupby('day_of_week').size()
print(f"\n按星期分布:")
for d in range(7):
    print(f"  {weekday_names[d]}: {daily.get(d, 0):,}")

# 4.4 歌曲热度分析
print("\n--- 4.4 歌曲热度 TOP10 ---")
song_pop = act.groupby('song_id').size().sort_values(ascending=False)
print(f"总歌曲数: {len(song_pop)}")
print(f"Top 10 歌曲播放次数:")
for rank, (sid, cnt) in enumerate(song_pop.head(10).items(), 1):
    print(f"  {rank:2d}. {sid[:12]}... | {cnt:,} 次")

# 4.5 与歌曲元数据关联分析
print("\n--- 4.5 歌曲元数据关联 ---")
# 热门歌曲关联到歌曲表
top_songs_detail = songs[songs['song_id'].isin(song_pop.head(20).index)]
print(f"Top 20 热门歌曲中能关联到元数据的: {len(top_songs_detail)} 首")

# 各语言歌曲的平均热度
act_with_meta = act.merge(songs[['song_id', 'language', 'gender_name']], on='song_id', how='left')
lang_pop = act_with_meta.groupby('language').size().sort_values(ascending=False)
print(f"\n各语言歌曲播放次数:")
for lang, cnt in lang_pop.items():
    print(f"  {lang}: {cnt:,}")
