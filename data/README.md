# 数据集说明

## 来源

阿里云天池 [虾米音乐用户行为数据](https://tianchi.aliyun.com/dataset/51810)

## 文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `mars_tianchi_songs.csv` | 2.2 MB | 歌曲元数据（26,957 首） |
| `mars_tianchi_user_actions.csv` | 1.4 GB | 用户行为记录（原始约 1588 万条） |

## 字段

### 歌曲表
| 字段 | 说明 |
|------|------|
| song_id | 歌曲 ID（MD5 脱敏） |
| artist_id | 艺人 ID（MD5 脱敏） |
| publish_time | 发布时间（YYYYMMDD） |
| song_init_plays | 初始播放量 |
| language | 语言编码 |
| gender | 艺人性别（1=男, 2=女, 3=组合/乐队） |

### 行为表
| 字段 | 说明 |
|------|------|
| user_id | 用户 ID（MD5 脱敏） |
| song_id | 歌曲 ID（MD5 脱敏） |
| gmt_create | 行为时间（Unix 秒数，UTC） |
| action_type | 行为类型（1=播放, 2=下载, 3=收藏） |
| ds | 日期分区（YYYYMMDD） |

## 注意

原始 CSV 文件较大，未上传至 GitHub。请从天池下载后放入 `data/` 目录。
