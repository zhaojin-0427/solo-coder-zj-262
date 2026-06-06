import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_sample_data(n_games=500, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    go_players = [
        '柯浩然', '李星河', '王天元', '张云飞', '赵清风',
        '孙弈秋', '周落霞', '吴观澜', '郑思远', '冯静远',
        '陈观海', '褚星驰', '卫临风', '蒋南屏', '沈墨白'
    ]

    xiangqi_players = [
        '吕文远', '苏慕白', '韩铁衣', '杨梦尘', '朱弘毅',
        '秦望舒', '许惊鸿', '何雨辰', '施砚秋', '张仲谋'
    ]

    levels = ['业余1段', '业余3段', '业余5段', '职业初段', '职业三段', '职业五段', '职业七段', '职业九段']

    go_openings = {
        '星布局': ['星·小飞挂', '星·一间夹', '星·二间高夹', '星·大飞挂', '星·三间夹'],
        '小目布局': ['小目·小飞挂', '小目·一间高挂', '小目·大飞挂', '小目·二间高挂'],
        '中国流': ['中国流·高中国流', '中国流·低中国流', '迷你中国流'],
        '三连星': ['三连星·大模样', '三连星·快速布局'],
        '目外布局': ['目外·飞压', '目外·大斜'],
        '高目布局': ['高目·内托', '高目·外靠'],
        '大雪崩': ['大雪崩·内拐', '大雪崩·外拐'],
        '妖刀定式': ['妖刀·基本型', '妖刀·变例']
    }

    xiangqi_openings = {
        '中炮类': ['中炮对屏风马', '中炮对反宫马', '中炮对单提马', '中炮对顺炮', '中炮对列炮'],
        '仙人指路': ['仙人指路对卒底炮', '仙人指路对进马', '仙人指路对飞象'],
        '飞象局': ['飞象对进马', '飞象对仙人指路', '飞象对过宫炮'],
        '起马局': ['起马对进卒', '起马对中炮'],
        '过宫炮': ['过宫炮对中炮', '过宫炮对起马'],
        '士角炮': ['士角炮对中炮', '士角炮对进马']
    }

    eras = ['古代(清代以前)', '近代(1912-1949)', '现代(1950-1990)', '当代(1990-2010)', '新世纪(2010-至今)']
    schools = ['日本棋院', '中国棋院', '韩国棋院', '吴清源流', '聂卫平道场', '马晓春道场', '传统古谱']
    competitions = ['世界大赛', '全国锦标赛', '甲级联赛', '杯赛', '友谊赛', '网络对弈']
    game_classes = ['慢棋', '快棋', '超快棋']

    records = []

    for game_idx in range(n_games):
        is_go = np.random.random() > 0.4
        game_type = '围棋' if is_go else '象棋'

        if is_go:
            players = go_players
            openings = go_openings
        else:
            players = xiangqi_players
            openings = xiangqi_openings

        player = random.choice(players)
        possible_opponents = [p for p in players if p != player]
        opponent = random.choice(possible_opponents)

        player_level_idx = np.random.randint(2, len(levels))
        player_level = levels[player_level_idx]
        opponent_level = levels[max(0, min(len(levels) - 1, player_level_idx + np.random.randint(-2, 3)))]

        opening_family = random.choice(list(openings.keys()))
        opening = random.choice(openings[opening_family])

        days_ago = np.random.randint(0, 365 * 5)
        match_date = datetime.now() - timedelta(days=days_ago)

        year = match_date.year
        if year < 1912:
            era = '古代(清代以前)'
        elif year < 1950:
            era = '近代(1912-1949)'
        elif year < 1990:
            era = '现代(1950-1990)'
        elif year < 2010:
            era = '当代(1990-2010)'
        else:
            era = '新世纪(2010-至今)'

        player_skill = player_level_idx / len(levels) + np.random.normal(0, 0.1)
        opponent_skill = levels.index(opponent_level) / len(levels) + np.random.normal(0, 0.1)

        aggressive = np.random.randint(3, 30) if is_go else np.random.randint(2, 18)
        defensive = np.random.randint(2, 25) if is_go else np.random.randint(1, 15)
        territory = np.random.randint(0, 60) if is_go else np.random.randint(0, 30)
        mistakes = max(0, int(np.random.exponential(1.5)))
        winning_moves = max(0, int(np.random.poisson(2)))
        total_moves = np.random.randint(80, 280) if is_go else np.random.randint(30, 120)

        win_prob = 1 / (1 + np.exp(-(player_skill - opponent_skill + np.random.normal(0, 0.3))))
        win = 1 if np.random.random() < win_prob else 0
        result = '胜' if win else '负'

        records.append({
            'game_id': f'G{10000 + game_idx}',
            'player': player,
            'opponent': opponent,
            'player_level': player_level,
            'opponent_level': opponent_level,
            'game_type': game_type,
            'opening': opening,
            'opening_family': opening_family,
            'result': result,
            'total_moves': total_moves,
            'aggressive_moves': aggressive,
            'defensive_moves': defensive,
            'territory_gain': territory,
            'key_mistakes': mistakes,
            'key_winning_moves': winning_moves,
            'match_date': match_date.strftime('%Y-%m-%d'),
            'era': era,
            'school': random.choice(schools),
            'competition_type': random.choice(competitions),
            'game_class': random.choice(game_classes)
        })

    return pd.DataFrame(records)


def save_sample_data(file_path='sample_chess_data.csv', n_games=500):
    df = generate_sample_data(n_games)
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    return df


if __name__ == '__main__':
    df = save_sample_data()
    print(f'生成示例数据 {len(df)} 条')
    print(df.head())
    print('\n数据列名:', df.columns.tolist())
