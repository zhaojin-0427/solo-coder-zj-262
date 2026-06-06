import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime


def _standardize(X):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def _kmeans(X, n_clusters=3, n_init=10, max_iter=300, random_state=42):
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]
    if n_samples <= n_clusters:
        return np.zeros(n_samples, dtype=int)

    rng = np.random.RandomState(random_state)
    best_labels = None
    best_inertia = np.inf

    for _ in range(n_init):
        idx = rng.choice(n_samples, n_clusters, replace=False)
        centers = X[idx].copy()

        for _ in range(max_iter):
            distances = np.sqrt(((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)

            new_centers = np.zeros_like(centers)
            for k in range(n_clusters):
                if np.sum(labels == k) > 0:
                    new_centers[k] = X[labels == k].mean(axis=0)
                else:
                    new_centers[k] = centers[k]

            if np.allclose(centers, new_centers):
                break
            centers = new_centers

        distances = np.sqrt(((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))
        inertia = distances.min(axis=1).sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return best_labels if best_labels is not None else np.zeros(n_samples, dtype=int)


def load_data(file_path=None, uploaded_df=None):
    if uploaded_df is not None:
        df = uploaded_df
    elif file_path is not None:
        df = pd.read_csv(file_path)
    else:
        return None

    required_columns = [
        'game_id', 'player', 'opponent', 'player_level', 'opponent_level',
        'game_type', 'opening', 'opening_family', 'result', 'total_moves',
        'aggressive_moves', 'defensive_moves', 'territory_gain',
        'key_mistakes', 'key_winning_moves', 'match_date', 'era',
        'school', 'competition_type', 'game_class'
    ]

    for col in required_columns:
        if col not in df.columns:
            if col == 'match_date':
                df[col] = '2024-01-01'
            elif col in ['era', 'school', 'competition_type', 'game_class']:
                df[col] = '未知'
            elif col in ['aggressive_moves', 'defensive_moves', 'territory_gain',
                         'key_mistakes', 'key_winning_moves', 'total_moves']:
                df[col] = 0
            else:
                df[col] = '未知'

    df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
    df['win'] = df['result'].apply(lambda x: 1 if str(x).lower() in ['win', '胜', '赢', '1-0', '0-1'] else 0)
    df['year'] = df['match_date'].dt.year.fillna(2024).astype(int)

    return df


def compute_player_stats(df):
    if df is None or len(df) == 0:
        return pd.DataFrame()

    stats = df.groupby('player').agg(
        total_games=('game_id', 'count'),
        wins=('win', 'sum'),
        avg_moves=('total_moves', 'mean'),
        avg_aggressive=('aggressive_moves', 'mean'),
        avg_defensive=('defensive_moves', 'mean'),
        avg_territory=('territory_gain', 'mean'),
        avg_mistakes=('key_mistakes', 'mean'),
        avg_winning_moves=('key_winning_moves', 'mean')
    ).reset_index()

    stats['win_rate'] = (stats['wins'] / stats['total_games'] * 100).round(2)
    stats['aggression_ratio'] = (stats['avg_aggressive'] / (stats['avg_aggressive'] + stats['avg_defensive'] + 1)).round(3)
    stats['defense_ratio'] = (stats['avg_defensive'] / (stats['avg_aggressive'] + stats['avg_defensive'] + 1)).round(3)
    stats['efficiency'] = (stats['avg_winning_moves'] / (stats['avg_mistakes'] + 1)).round(3)

    return stats


def classify_playing_style(player_stats):
    if player_stats is None or len(player_stats) == 0:
        return player_stats

    features = player_stats[['aggression_ratio', 'defense_ratio', 'efficiency', 'win_rate', 'avg_territory']].fillna(0)

    if len(features) >= 3:
        scaled = _standardize(features.values)
        player_stats['style_cluster'] = _kmeans(scaled, n_clusters=min(3, len(features)), random_state=42)
    else:
        player_stats['style_cluster'] = 0

    def label_style(row):
        if row['aggression_ratio'] > 0.6 and row['efficiency'] > 0.8:
            return '激进攻击型'
        elif row['defense_ratio'] > 0.55 and row['avg_mistakes'] < 1.5:
            return '稳健防守型'
        elif row['avg_territory'] > 30 and row['win_rate'] > 50:
            return '布局控场型'
        elif row['efficiency'] > 1.0 and row['win_rate'] > 55:
            return '精准计算型'
        else:
            return '均衡型'

    player_stats['playing_style'] = player_stats.apply(label_style, axis=1)
    return player_stats


def get_opening_distribution(df, player=None):
    if df is None or len(df) == 0:
        return pd.DataFrame()

    filtered = df if player is None else df[df['player'] == player]

    opening_stats = filtered.groupby(['opening_family', 'opening']).agg(
        count=('game_id', 'count'),
        wins=('win', 'sum')
    ).reset_index()

    opening_stats['win_rate'] = (opening_stats['wins'] / opening_stats['count'] * 100).round(2)
    opening_stats = opening_stats.sort_values('count', ascending=False)

    return opening_stats


def get_win_rate_trend(df, player=None):
    if df is None or len(df) == 0:
        return pd.DataFrame()

    filtered = df if player is None else df[df['player'] == player]
    filtered = filtered.sort_values('match_date')

    filtered['game_number'] = range(1, len(filtered) + 1)
    filtered['cumulative_wins'] = filtered['win'].cumsum()
    filtered['cumulative_win_rate'] = (filtered['cumulative_wins'] / filtered['game_number'] * 100).round(2)

    monthly = filtered.groupby(filtered['match_date'].dt.to_period('M')).agg(
        games=('game_id', 'count'),
        wins=('win', 'sum')
    ).reset_index()
    monthly['match_date'] = monthly['match_date'].astype(str)
    monthly['win_rate'] = (monthly['wins'] / monthly['games'] * 100).round(2)

    return filtered[['match_date', 'game_number', 'cumulative_win_rate']], monthly


def get_heatmap_data(df):
    if df is None or len(df) == 0:
        return pd.DataFrame()

    pivot = df.pivot_table(
        index='player',
        columns='opening_family',
        values='win',
        aggfunc='mean',
        fill_value=0
    ) * 100

    return pivot.round(2)


def get_level_analysis(df):
    if df is None or len(df) == 0:
        return pd.DataFrame()

    level_stats = df.groupby(['player_level', 'opponent_level']).agg(
        games=('game_id', 'count'),
        wins=('win', 'sum'),
        avg_aggressive=('aggressive_moves', 'mean'),
        avg_defensive=('defensive_moves', 'mean')
    ).reset_index()

    level_stats['win_rate'] = (level_stats['wins'] / level_stats['games'] * 100).round(2)
    return level_stats


def get_key_move_analysis(df, player=None):
    if df is None or len(df) == 0:
        return pd.DataFrame()

    filtered = df if player is None else df[df['player'] == player]

    move_stats = filtered.groupby('opening_family').agg(
        avg_winning_moves=('key_winning_moves', 'mean'),
        avg_mistakes=('key_mistakes', 'mean'),
        games=('game_id', 'count'),
        wins=('win', 'sum')
    ).reset_index()

    move_stats['win_rate'] = (move_stats['wins'] / move_stats['games'] * 100).round(2)
    move_stats['net_gain'] = (move_stats['avg_winning_moves'] - move_stats['avg_mistakes']).round(3)

    return move_stats


def generate_training_advice(player_stats, opening_stats, key_move_stats):
    advice = []

    if player_stats is not None and len(player_stats) > 0:
        best_row = player_stats.loc[player_stats['win_rate'].idxmax()]
        style = best_row['playing_style']

        style_advice = {
            '激进攻击型': '建议加强攻杀后收官阶段的稳定性，注意保留实空平衡，避免过度进攻导致后方空虚。',
            '稳健防守型': '建议适当增加主动出击比例，在防守反击中寻找战机，提升棋局掌控力。',
            '布局控场型': '建议研究中盘复杂战斗的处理方式，在保持布局优势的同时增强中后盘战斗力。',
            '精准计算型': '建议加强直觉判断和大局观训练，减少时间压力下的计算误差。',
            '均衡型': '建议形成个人特色开局体系，在某一两种布局上深入研究，建立技术特长。'
        }
        advice.append({
            'category': '棋风分析',
            'level': 'info',
            'content': f"您的棋风评估为「{style}」。{style_advice.get(style, '')}"
        })

    if opening_stats is not None and len(opening_stats) > 0:
        high_win = opening_stats[opening_stats['count'] >= 3].nlargest(3, 'win_rate')
        low_win = opening_stats[opening_stats['count'] >= 3].nsmallest(2, 'win_rate')

        if len(high_win) > 0:
            top = high_win.iloc[0]
            advice.append({
                'category': '擅长布局',
                'level': 'success',
                'content': f"您在「{top['opening']}」布局中胜率达 {top['win_rate']}%（{int(top['count'])}局），建议作为核心开局体系深入钻研。"
            })

        if len(low_win) > 0:
            weak = low_win.iloc[0]
            advice.append({
                'category': '待改进布局',
                'level': 'warning',
                'content': f"「{weak['opening']}」布局胜率仅 {weak['win_rate']}%（{int(weak['count'])}局），建议加强该布局的变例研究或更换应对策略。"
            })

    if key_move_stats is not None and len(key_move_stats) > 0:
        high_mistakes = key_move_stats.nlargest(2, 'avg_mistakes')
        if len(high_mistakes) > 0:
            hm = high_mistakes.iloc[0]
            advice.append({
                'category': '关键失误',
                'level': 'danger',
                'content': f"在「{hm['opening_family']}」布局中平均失误 {hm['avg_mistakes'].round(2)} 次，建议重点复盘该布局中盘转折点。"
            })

    advice.append({
        'category': '训练建议',
        'level': 'primary',
        'content': '建议每日进行 30 分钟死活题训练，每周至少完成 2 局慢棋并深度复盘，重点关注第 50-100 手的关键决策。'
    })

    common_traps = [
        '大雪崩定式内拐变例：注意白方在第 18 手的断，黑方需正确应对否则损失约 15 目。',
        '中国流布局大飞挂角：警惕白方二路托后的复杂转换，黑方应坚持取势方针。',
        '星位小飞挂一间夹：白方跳后黑方需注意征子关系，避免被分断攻击。',
        '屏风马进七兵：红方过河车后黑方可考虑平炮兑车简化局势，或左马盘河反击。',
        '顺炮直车对横车：红方需注意黑方车 4 进 7 塞象眼的凶悍手段，提前补厚中路。'
    ]

    for trap in common_traps:
        advice.append({
            'category': '常见陷阱',
            'level': 'secondary',
            'content': trap
        })

    return advice


def get_classic_games(df, top_n=5):
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df_scored = df.copy()
    df_scored['game_score'] = (
        df_scored['win'] * 50 +
        df_scored['key_winning_moves'] * 10 -
        df_scored['key_mistakes'] * 5 +
        df_scored['total_moves'] * 0.1
    )

    return df_scored.nlargest(top_n, 'game_score')[[
        'game_id', 'player', 'opponent', 'opening', 'result',
        'total_moves', 'key_winning_moves', 'match_date', 'competition_type'
    ]]
