import base64
import io
import os
import json

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, dash_table, callback_context, ALL, MATCH
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from data_processing import (
    load_data, compute_player_stats, classify_playing_style,
    get_opening_distribution, get_win_rate_trend, get_heatmap_data,
    get_level_analysis, get_key_move_analysis, generate_training_advice,
    get_classic_games, compare_players_stats, compare_players_win_trend,
    compare_players_top_openings, compare_players_styles, compare_players_mistakes,
    get_game_detail, generate_game_training_advice, get_training_list_summary,
    get_opening_weaknesses
)
from sample_data import generate_sample_data

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title='传统棋类对弈棋谱胜率与布局偏好分析台'
)

server = app.server

sample_df = generate_sample_data(500)

STYLE_COLORS = {
    '激进攻击型': '#e74c3c',
    '稳健防守型': '#3498db',
    '布局控场型': '#27ae60',
    '精准计算型': '#9b59b6',
    '均衡型': '#95a5a6'
}

COMPARE_COLORS = ['#e74c3c', '#3498db', '#27ae60']

NAVBAR = dbc.Navbar(
    [
        html.A(
            dbc.Row(
                [
                    dbc.Col(html.Div('♚', style={'fontSize': '2rem', 'color': 'white'})),
                    dbc.Col(
                        dbc.NavbarBrand(
                            '传统棋类对弈棋谱胜率与布局偏好分析台',
                            className='ms-2',
                            style={'fontSize': '1.3rem', 'fontWeight': 'bold'}
                        )
                    ),
                ],
                align='center',
                className='g-0',
            ),
            href='#',
            style={'textDecoration': 'none'},
        ),
        dbc.NavbarToggler(id='navbar-toggler'),
        dbc.Collapse(
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink('📊 总览分析', href='#overview')),
                    dbc.NavItem(dbc.NavLink('🎯 布局偏好', href='#opening')),
                    dbc.NavItem(dbc.NavLink('📈 胜率分析', href='#winrate')),
                    dbc.NavItem(dbc.NavLink('🎭 棋风聚类', href='#style')),
                    dbc.NavItem(dbc.NavLink('🏆 经典对局', href='#games')),
                    dbc.NavItem(dbc.NavLink('📋 训练清单', href='#training')),
                    dbc.NavItem(dbc.NavLink('⚔️ 多棋手对比', href='#compare')),
                    dbc.NavItem(dbc.NavLink('💡 训练建议', href='#advice')),
                ],
                className='ms-auto',
                navbar=True,
            ),
            id='navbar-collapse',
            navbar=True,
        ),
    ],
    color='primary',
    dark=True,
    sticky='top',
    className='mb-4 shadow'
)

UPLOAD_SECTION = dbc.Card(
    [
        dbc.CardHeader(
            html.H5('📁 棋谱数据上传', className='mb-0'),
            className='bg-light'
        ),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Upload(
                                    id='upload-data',
                                    children=html.Div(
                                        [
                                            '拖拽 CSV 文件到此处 或 ',
                                            html.A('点击选择文件', style={'color': '#2c3e50', 'fontWeight': 'bold', 'textDecoration': 'underline'})
                                        ],
                                        style={'textAlign': 'center', 'padding': '2rem'}
                                    ),
                                    style={
                                        'width': '100%',
                                        'height': '80px',
                                        'lineHeight': '80px',
                                        'borderWidth': '2px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '10px',
                                        'backgroundColor': '#f8f9fa'
                                    },
                                    multiple=False,
                                    accept='.csv'
                                ),
                                html.Div(id='upload-status', className='mt-2 text-center small text-muted')
                            ],
                            md=8
                        ),
                        dbc.Col(
                            [
                                dbc.Button(
                                    '🎲 使用示例数据',
                                    id='use-sample-btn',
                                    color='secondary',
                                    className='w-100 h-100',
                                    style={'fontSize': '1.1rem', 'padding': '1rem'}
                                )
                            ],
                            md=4
                        )
                    ]
                ),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label('🎮 棋类:', className='fw-bold'),
                                dcc.Dropdown(
                                    id='filter-game-type',
                                    options=[
                                        {'label': '全部', 'value': '全部'},
                                        {'label': '围棋', 'value': '围棋'},
                                        {'label': '象棋', 'value': '象棋'}
                                    ],
                                    value='全部',
                                    clearable=False
                                )
                            ],
                            md=3
                        ),
                        dbc.Col(
                            [
                                html.Label('🏛️ 历史时期:', className='fw-bold'),
                                dcc.Dropdown(
                                    id='filter-era',
                                    options=[{'label': '全部', 'value': '全部'}],
                                    value='全部',
                                    clearable=False
                                )
                            ],
                            md=3
                        ),
                        dbc.Col(
                            [
                                html.Label('📚 流派传承:', className='fw-bold'),
                                dcc.Dropdown(
                                    id='filter-school',
                                    options=[{'label': '全部', 'value': '全部'}],
                                    value='全部',
                                    clearable=False
                                )
                            ],
                            md=3
                        ),
                        dbc.Col(
                            [
                                html.Label('🏆 比赛类型:', className='fw-bold'),
                                dcc.Dropdown(
                                    id='filter-competition',
                                    options=[{'label': '全部', 'value': '全部'}],
                                    value='全部',
                                    clearable=False
                                )
                            ],
                            md=3
                        )
                    ]
                ),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label('👤 选择棋手:', className='fw-bold'),
                                dcc.Dropdown(
                                    id='player-select',
                                    options=[{'label': '全体棋手', 'value': 'ALL'}],
                                    value='ALL',
                                    clearable=False
                                )
                            ],
                            md=6
                        ),
                        dbc.Col(
                            [
                                html.Label('⚔️ 对比棋手 (最多 3 人):', className='fw-bold'),
                                dcc.Dropdown(
                                    id='compare-player-select',
                                    options=[],
                                    value=[],
                                    multi=True,
                                    placeholder='选择要对比的棋手...',
                                    clearable=True
                                ),
                                html.Div(id='compare-select-hint', className='small text-muted mt-1')
                            ],
                            md=6
                        )
                    ]
                )
            ]
        )
    ],
    className='mb-4 shadow-sm'
)

OVERVIEW_CARDS = dbc.Row(
    [
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.Div('🎯 总对局数', className='text-muted small'),
                            html.H3(id='stat-total-games', className='text-primary fw-bold mt-1'),
                            html.Div(id='stat-game-breakdown', className='small text-muted mt-1')
                        ]
                    )
                ],
                className='shadow-sm h-100'
            ),
            md=3
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.Div('🏆 平均胜率', className='text-muted small'),
                            html.H3(id='stat-avg-winrate', className='text-success fw-bold mt-1'),
                            html.Div('近 30 日', className='small text-muted mt-1')
                        ]
                    )
                ],
                className='shadow-sm h-100'
            ),
            md=3
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.Div('📊 棋手总数', className='text-muted small'),
                            html.H3(id='stat-player-count', className='text-info fw-bold mt-1'),
                            html.Div('参与分析', className='small text-muted mt-1')
                        ]
                    )
                ],
                className='shadow-sm h-100'
            ),
            md=3
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.Div('📖 开局类型', className='text-muted small'),
                            html.H3(id='stat-opening-count', className='text-warning fw-bold mt-1'),
                            html.Div('不同布局体系', className='small text-muted mt-1')
                        ]
                    )
                ],
                className='shadow-sm h-100'
            ),
            md=3
        )
    ],
    className='mb-4'
)

HEATMAP_SECTION = html.Div(
    [
        html.Div(id='overview', style={'position': 'relative', 'top': '-70px'}),
        dbc.Card(
            [
                dbc.CardHeader([html.I(className='fas fa-fire me-2'), '🔥 布局偏好热力图 (棋手 × 开局胜率)'], className='bg-light fw-bold'),
                dbc.CardBody(dcc.Graph(id='heatmap-chart', style={'height': '500px'}))
            ],
            className='mb-4 shadow-sm'
        )
    ]
)

WINRATE_TREND_SECTION = html.Div(
    [
        html.Div(id='winrate', style={'position': 'relative', 'top': '-70px'}),
        dbc.Card(
            [
                dbc.CardHeader('📈 胜率趋势分析', className='bg-light fw-bold'),
                dbc.CardBody(
                    [
                        dbc.Tabs(
                            [
                                dbc.Tab(
                                    dcc.Graph(id='winrate-trend-chart', style={'height': '400px'}),
                                    label='📅 月度胜率趋势'
                                ),
                                dbc.Tab(
                                    dcc.Graph(id='cumulative-winrate-chart', style={'height': '400px'}),
                                    label='📊 累计胜率走势'
                                ),
                                dbc.Tab(
                                    dcc.Graph(id='level-analysis-chart', style={'height': '400px'}),
                                    label='⚖️ 等级差胜率分析'
                                )
                            ]
                        )
                    ]
                )
            ],
            className='mb-4 shadow-sm'
        )
    ]
)

OPENING_SECTION = html.Div(
    [
        html.Div(id='opening', style={'position': 'relative', 'top': '-70px'}),
        dbc.Card(
            [
                dbc.CardHeader('🎯 开局偏好与胜率分布', className='bg-light fw-bold'),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(id='opening-bar-chart', style={'height': '450px'}),
                                    md=7
                                ),
                                dbc.Col(
                                    dcc.Graph(id='opening-pie-chart', style={'height': '450px'}),
                                    md=5
                                )
                            ]
                        ),
                        html.Hr(),
                        html.H6('📋 布局详细统计', className='mb-3'),
                        dash_table.DataTable(
                            id='opening-detail-table',
                            page_size=8,
                            style_table={'overflowX': 'auto'},
                            style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'},
                            style_cell={'textAlign': 'center', 'padding': '10px'},
                            style_data_conditional=[
                                {
                                    'if': {'filter_query': '{win_rate} >= 60'},
                                    'backgroundColor': '#d4edda',
                                    'color': '#155724'
                                },
                                {
                                    'if': {'filter_query': '{win_rate} < 40'},
                                    'backgroundColor': '#f8d7da',
                                    'color': '#721c24'
                                }
                            ]
                        )
                    ]
                )
            ],
            className='mb-4 shadow-sm'
        )
    ]
)

STYLE_SECTION = html.Div(
    [
        html.Div(id='style', style={'position': 'relative', 'top': '-70px'}),
        dbc.Card(
            [
                dbc.CardHeader('🎭 棋风类型聚类分析', className='bg-light fw-bold'),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(id='style-scatter-chart', style={'height': '500px'}),
                                    md=8
                                ),
                                dbc.Col(
                                    [
                                        html.H6('🏷️ 棋风分布统计', className='mb-3'),
                                        dcc.Graph(id='style-pie-chart', style={'height': '280px'}),
                                        html.Hr(),
                                        html.H6('📊 各棋风平均指标', className='mb-3 mt-3'),
                                        dash_table.DataTable(
                                            id='style-stats-table',
                                            page_size=5,
                                            style_table={'overflowX': 'auto'},
                                            style_header={'backgroundColor': '#34495e', 'color': 'white', 'fontWeight': 'bold', 'fontSize': '12px'},
                                            style_cell={'textAlign': 'center', 'padding': '6px', 'fontSize': '12px'}
                                        )
                                    ],
                                    md=4
                                )
                            ]
                        )
                    ]
                )
            ],
            className='mb-4 shadow-sm'
        )
    ]
)

KEY_MOVE_SECTION = dbc.Card(
    [
        dbc.CardHeader('⚡ 关键手得失分析', className='bg-light fw-bold'),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(id='keymove-radar-chart', style={'height': '400px'}),
                            md=6
                        ),
                        dbc.Col(
                            dcc.Graph(id='keymove-bar-chart', style={'height': '400px'}),
                            md=6
                        )
                    ]
                )
            ]
        )
    ],
    className='mb-4 shadow-sm'
)

GAMES_SECTION = html.Div(
    [
        html.Div(id='games', style={'position': 'relative', 'top': '-70px'}),
        dbc.Card(
            [
                dbc.CardHeader('🏆 经典对局复盘（点击任意对局查看详情）', className='bg-light fw-bold'),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dash_table.DataTable(
                                            id='classic-games-table',
                                            page_size=6,
                                            style_table={'overflowX': 'auto', 'cursor': 'pointer'},
                                            style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'},
                                            style_cell={'textAlign': 'center', 'padding': '8px'},
                                            style_data={'whiteSpace': 'normal', 'height': 'auto'},
                                            style_data_conditional=[
                                                {
                                                    'if': {'state': 'active'},
                                                    'backgroundColor': '#d6eaf8',
                                                    'border': '1px solid #2980b9'
                                                },
                                                {
                                                    'if': {'state': 'selected'},
                                                    'backgroundColor': '#d6eaf8',
                                                    'border': '1px solid #2980b9'
                                                }
                                            ],
                                            row_selectable='single',
                                            selected_rows=[]
                                        ),
                                        html.Div(id='game-click-hint', className='text-center text-muted small mt-2',
                                                 children='👆 点击表格中任意行展开对局详情')
                                    ],
                                    md=12,
                                    lg=7
                                ),
                                dbc.Col(
                                    [
                                        html.Div(
                                            id='game-detail-panel',
                                            children=[
                                                dbc.Card(
                                                    [
                                                        dbc.CardBody(
                                                            html.Div(
                                                                '请在左侧选择一局经典对局，此处将展示详细复盘信息与训练建议',
                                                                className='text-center text-muted py-5'
                                                            )
                                                        )
                                                    ],
                                                    className='border-secondary'
                                                )
                                            ]
                                        )
                                    ],
                                    md=12,
                                    lg=5,
                                    className='mt-3 mt-lg-0'
                                )
                            ]
                        )
                    ]
                )
            ],
            className='mb-4 shadow-sm'
        )
    ]
)


TRAINING_LIST_SECTION = html.Div(
    [
        html.Div(id='training', style={'position': 'relative', 'top': '-70px'}),
        dbc.Card(
            [
                dbc.CardHeader('📋 训练清单（按当前筛选条件）', className='bg-light fw-bold'),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H6('📊 训练概览', className='fw-bold mb-3'),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.Div('今日待复盘对局', className='text-muted small'),
                                                                html.H4(id='training-total-count', className='text-primary fw-bold mt-1', children='0'),
                                                                html.Div('局', className='small text-muted')
                                                            ]
                                                        ),
                                                        className='shadow-sm h-100'
                                                    ),
                                                    md=6
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.Div('开局弱点数量', className='text-muted small'),
                                                                html.H4(id='training-weakness-count', className='text-warning fw-bold mt-1', children='0'),
                                                                html.Div('种', className='small text-muted')
                                                            ]
                                                        ),
                                                        className='shadow-sm h-100'
                                                    ),
                                                    md=6
                                                )
                                            ],
                                            className='mb-3'
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.Div('弱点对局总数', className='text-muted small'),
                                                                html.H4(id='training-weak-games', className='text-danger fw-bold mt-1', children='0'),
                                                                html.Div('局', className='small text-muted')
                                                            ]
                                                        ),
                                                        className='shadow-sm h-100'
                                                    ),
                                                    md=6
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.Div('弱点平均胜率', className='text-muted small'),
                                                                html.H4(id='training-avg-wr', className='text-info fw-bold mt-1', children='-'),
                                                                html.Div('%', className='small text-muted')
                                                            ]
                                                        ),
                                                        className='shadow-sm h-100'
                                                    ),
                                                    md=6
                                                )
                                            ]
                                        ),
                                        html.Hr(),
                                        html.H6('🎯 待练习对局', className='fw-bold mb-2'),
                                        html.Div(id='training-games-list',
                                                 children=html.Div('暂无待练习对局，可在「经典对局复盘」区将感兴趣的对局加入训练清单',
                                                                   className='text-muted small py-3 text-center'))
                                    ],
                                    md=12,
                                    lg=6
                                ),
                                dbc.Col(
                                    [
                                        html.H6('⚠️ 开局弱点分析（按胜率从低到高）', className='fw-bold mb-3'),
                                        html.Div(id='training-weakness-table',
                                                 children=html.Div('暂无足够数据生成开局弱点分析',
                                                                   className='text-muted small py-5 text-center')),
                                        html.Hr(),
                                        html.H6('💡 训练重点建议', className='fw-bold mb-3'),
                                        html.Div(id='training-focus-advice',
                                                 children=html.Div('根据当前筛选条件自动生成训练重点',
                                                                   className='text-muted small py-3 text-center'))
                                    ],
                                    md=12,
                                    lg=6,
                                    className='mt-4 mt-lg-0'
                                )
                            ]
                        )
                    ]
                )
            ],
            className='mb-4 shadow-sm'
        )
    ]
)


COMPARE_SECTION = html.Div(
    [
        html.Div(id='compare', style={'position': 'relative', 'top': '-70px'}),
        dbc.Card(
            [
                dbc.CardHeader('⚔️ 多棋手对比分析', className='bg-light fw-bold'),
                dbc.CardBody(
                    [
                        html.Div(id='compare-empty-hint', className='text-center text-muted py-4',
                                 children='请在上方筛选区选择 1-3 名棋手进行对比分析'),
                        html.Div(
                            id='compare-content',
                            children=[
                                html.H6('📊 棋手核心指标对比', className='mb-3 mt-2'),
                                dash_table.DataTable(
                                    id='compare-stats-table',
                                    page_size=5,
                                    style_table={'overflowX': 'auto'},
                                    style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'},
                                    style_cell={'textAlign': 'center', 'padding': '8px'},
                                    style_data_conditional=[
                                        {
                                            'if': {'column_id': '胜率(%)', 'filter_query': '{胜率(%)} >= 60'},
                                            'backgroundColor': '#d4edda',
                                            'color': '#155724'
                                        },
                                        {
                                            'if': {'column_id': '胜率(%)', 'filter_query': '{胜率(%)} < 40'},
                                            'backgroundColor': '#f8d7da',
                                            'color': '#721c24'
                                        }
                                    ]
                                ),
                                html.Hr(),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.H6('📈 胜率趋势对比', className='mb-3'),
                                                dcc.Graph(id='compare-winrate-chart', style={'height': '380px'})
                                            ],
                                            md=6
                                        ),
                                        dbc.Col(
                                            [
                                                html.H6('🎭 棋风类型对比', className='mb-3'),
                                                dcc.Graph(id='compare-style-chart', style={'height': '380px'})
                                            ],
                                            md=6
                                        )
                                    ]
                                ),
                                html.Hr(),
                                html.H6('🎯 常用开局 Top5 对比', className='mb-3'),
                                dcc.Graph(id='compare-opening-chart', style={'height': '420px'}),
                                html.Hr(),
                                html.H6('⚠️ 关键失误分布对比', className='mb-3'),
                                dcc.Graph(id='compare-mistake-chart', style={'height': '420px'})
                            ],
                            style={'display': 'none'}
                        )
                    ]
                )
            ],
            className='mb-4 shadow-sm'
        )
    ]
)


def _make_advice_card(advice):
    color_map = {
        'success': 'success',
        'warning': 'warning',
        'danger': 'danger',
        'info': 'info',
        'primary': 'primary',
        'secondary': 'secondary'
    }
    icon_map = {
        'success': '✅',
        'warning': '⚠️',
        'danger': '❌',
        'info': 'ℹ️',
        'primary': '🎯',
        'secondary': '📌'
    }
    color = color_map.get(advice.get('level', 'info'), 'info')
    icon = icon_map.get(advice.get('level', 'info'), '📌')
    return dbc.Alert(
        [
            html.Strong(f"{icon} {advice.get('category', '建议')}: "),
            html.Span(advice.get('content', ''))
        ],
        color=color,
        className='mb-2'
    )


ADVICE_SECTION = html.Div(
    [
        html.Div(id='advice', style={'position': 'relative', 'top': '-70px'}),
        dbc.Card(
            [
                dbc.CardHeader('💡 智能训练建议区', className='bg-light fw-bold'),
                dbc.CardBody(
                    [
                        html.Div(id='advice-cards'),
                        html.Hr(),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H6('📚 推荐开局策略', className='fw-bold text-primary'),
                                                html.Div(id='recommended-openings', className='mt-2')
                                            ]
                                        ),
                                        className='border-primary'
                                    ),
                                    md=6
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H6('🎯 需要避开的陷阱', className='fw-bold text-danger'),
                                                html.Div(id='common-traps', className='mt-2')
                                            ]
                                        ),
                                        className='border-danger'
                                    ),
                                    md=6
                                )
                            ]
                        )
                    ]
                )
            ],
            className='mb-4 shadow-sm'
        )
    ]
)

app.layout = html.Div(
    [
        NAVBAR,
        html.Div(
            [
                UPLOAD_SECTION,
                OVERVIEW_CARDS,
                HEATMAP_SECTION,
                WINRATE_TREND_SECTION,
                OPENING_SECTION,
                KEY_MOVE_SECTION,
                STYLE_SECTION,
                GAMES_SECTION,
                TRAINING_LIST_SECTION,
                COMPARE_SECTION,
                ADVICE_SECTION,
                html.Footer(
                    html.Div(
                        '© 2024 传统棋类对弈棋谱胜率与布局偏好分析台 | 基于 Dash 构建',
                        className='text-center text-muted py-3'
                    ),
                    className='border-top mt-5'
                )
            ],
            className='container-fluid px-4'
        ),
        dcc.Store(id='stored-data'),
        dcc.Store(id='filtered-data'),
        dcc.Store(id='training-game-ids', data=[]),
        dcc.Store(id='selected-game-id', data=None)
    ]
)


def parse_contents(contents, filename):
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if 'csv' in filename.lower():
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8-sig')))
        else:
            return None, '请上传 CSV 格式的文件'
        return df, f'成功加载 {len(df)} 条棋谱记录: {filename}'
    except Exception as e:
        return None, f'文件解析出错: {str(e)}'


@app.callback(
    Output('stored-data', 'data'),
    Output('upload-status', 'children'),
    Input('upload-data', 'contents'),
    Input('use-sample-btn', 'n_clicks'),
    State('upload-data', 'filename'),
    prevent_initial_call=False
)
def load_dataset(upload_contents, btn_clicks, upload_filename):
    ctx = callback_context
    if not ctx.triggered:
        return sample_df.to_dict('records'), '当前使用内置示例数据，点击上方按钮可上传自定义棋谱'

    trigger = ctx.triggered[0]['prop_id']
    if 'use-sample-btn' in trigger:
        df = generate_sample_data(500)
        return df.to_dict('records'), '✅ 已加载 500 条示例棋谱数据'

    if upload_contents is not None:
        df, msg = parse_contents(upload_contents, upload_filename or '棋谱文件.csv')
        if df is not None:
            return df.to_dict('records'), f'✅ {msg}'
        return dash.no_update, f'❌ {msg}'

    return sample_df.to_dict('records'), '当前使用内置示例数据'


@app.callback(
    Output('filter-era', 'options'),
    Output('filter-school', 'options'),
    Output('filter-competition', 'options'),
    Input('stored-data', 'data')
)
def update_filter_options(data):
    if not data:
        return [{'label': '全部', 'value': '全部'}], [{'label': '全部', 'value': '全部'}], [{'label': '全部', 'value': '全部'}]
    df = pd.DataFrame(data)

    def make_options(series):
        opts = [{'label': '全部', 'value': '全部'}]
        for val in sorted(series.dropna().unique().tolist()):
            opts.append({'label': str(val), 'value': str(val)})
        return opts

    return make_options(df['era']), make_options(df['school']), make_options(df['competition_type'])


@app.callback(
    Output('player-select', 'options'),
    Input('filtered-data', 'data')
)
def update_player_options(data):
    if not data:
        return [{'label': '全体棋手', 'value': 'ALL'}]
    df = pd.DataFrame(data)
    player_opts = [{'label': '全体棋手', 'value': 'ALL'}]
    for p in sorted(df['player'].unique().tolist()):
        player_opts.append({'label': f'{p}', 'value': p})
    return player_opts


@app.callback(
    Output('filtered-data', 'data'),
    Input('stored-data', 'data'),
    Input('filter-game-type', 'value'),
    Input('filter-era', 'value'),
    Input('filter-school', 'value'),
    Input('filter-competition', 'value')
)
def apply_filters(data, game_type, era, school, competition):
    if not data:
        return {}
    df = pd.DataFrame(data)
    if game_type != '全部':
        df = df[df['game_type'] == game_type]
    if era != '全部':
        df = df[df['era'] == era]
    if school != '全部':
        df = df[df['school'] == school]
    if competition != '全部':
        df = df[df['competition_type'] == competition]
    return df.to_dict('records')


@app.callback(
    Output('stat-total-games', 'children'),
    Output('stat-game-breakdown', 'children'),
    Output('stat-avg-winrate', 'children'),
    Output('stat-player-count', 'children'),
    Output('stat-opening-count', 'children'),
    Input('filtered-data', 'data'),
    Input('player-select', 'value')
)
def update_overview_stats(data, player):
    if not data:
        return '0', '-', '0%', '0', '0'
    df = load_data(uploaded_df=pd.DataFrame(data))
    if df is None:
        return '0', '-', '0%', '0', '0'

    if player and player != 'ALL':
        df = df[df['player'] == player]

    total = len(df)
    go_count = len(df[df['game_type'] == '围棋'])
    xq_count = len(df[df['game_type'] == '象棋'])
    if player and player != 'ALL':
        breakdown = f'棋手: {player}'
    else:
        breakdown = f'围棋 {go_count} 局 · 象棋 {xq_count} 局'
    avg_wr = f"{df['win'].mean() * 100:.1f}%" if len(df) > 0 else '0%'
    player_cnt = df['player'].nunique()
    opening_cnt = df['opening_family'].nunique()

    return f'{total} 局', breakdown, avg_wr, f'{player_cnt} 位', f'{opening_cnt} 种'


@app.callback(
    Output('heatmap-chart', 'figure'),
    Input('filtered-data', 'data'),
    Input('player-select', 'value')
)
def update_heatmap(data, player):
    if not data:
        return go.Figure()
    df = load_data(uploaded_df=pd.DataFrame(data))
    if df is None or len(df) == 0:
        return go.Figure()

    if player and player != 'ALL':
        df = df[df['player'] == player]

    heatmap_data = get_heatmap_data(df)
    if heatmap_data.empty:
        return go.Figure()

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns.tolist(),
            y=heatmap_data.index.tolist(),
            colorscale='RdYlGn',
            zmid=50,
            text=np.round(heatmap_data.values, 1),
            texttemplate='%{text}%',
            textfont={'size': 10},
            hoverongaps=False,
            colorbar=dict(title='胜率(%)')
        )
    )

    title_suffix = f' - {player}' if player and player != 'ALL' else ''
    fig.update_layout(
        title={'text': f'各棋手在不同布局体系下的胜率分布{title_suffix}', 'x': 0.5, 'font': {'size': 16}},
        xaxis={'title': '开局体系', 'tickangle': -30},
        yaxis={'title': '棋手'},
        height=500,
        template='plotly_white'
    )
    return fig


@app.callback(
    Output('winrate-trend-chart', 'figure'),
    Output('cumulative-winrate-chart', 'figure'),
    Output('level-analysis-chart', 'figure'),
    Input('filtered-data', 'data'),
    Input('player-select', 'value')
)
def update_winrate_charts(data, player):
    empty_fig = go.Figure()
    if not data:
        return empty_fig, empty_fig, empty_fig

    df = load_data(uploaded_df=pd.DataFrame(data))
    if df is None or len(df) == 0:
        return empty_fig, empty_fig, empty_fig

    p = None if player == 'ALL' else player
    cumulative, monthly = get_win_rate_trend(df, p)

    fig1 = go.Figure()
    if not monthly.empty:
        fig1.add_trace(
            go.Scatter(
                x=monthly['match_date'],
                y=monthly['win_rate'],
                mode='lines+markers',
                name='月度胜率',
                line=dict(color='#27ae60', width=2),
                marker=dict(size=8)
            )
        )
        fig1.add_trace(
            go.Bar(
                x=monthly['match_date'],
                y=monthly['games'],
                name='对局数',
                yaxis='y2',
                marker=dict(color='#3498db', opacity=0.5)
            )
        )
        fig1.update_layout(
            yaxis2=dict(title='对局数', overlaying='y', side='right', showgrid=False),
            yaxis=dict(title='胜率(%)', range=[0, 100]),
            legend=dict(orientation='h', y=1.1),
            template='plotly_white'
        )
    fig1.update_layout(title={'text': f"月度胜率趋势 ({player if player != 'ALL' else '全体棋手'})", 'x': 0.5})

    fig2 = go.Figure()
    if not cumulative.empty:
        fig2.add_trace(
            go.Scatter(
                x=cumulative['game_number'],
                y=cumulative['cumulative_win_rate'],
                mode='lines',
                name='累计胜率',
                line=dict(color='#e67e22', width=3),
                fill='tonexty',
                fillcolor='rgba(230, 126, 34, 0.15)'
            )
        )
        fig2.add_hline(y=50, line_dash='dash', line_color='gray', annotation_text='50% 基准线')
        fig2.update_layout(
            xaxis={'title': '第 N 场对局'},
            yaxis={'title': '累计胜率(%)', 'range': [0, 100]},
            template='plotly_white'
        )
    fig2.update_layout(title={'text': f"累计胜率走势 ({player if player != 'ALL' else '全体棋手'})", 'x': 0.5})

    level_df = get_level_analysis(df)
    fig3 = go.Figure()
    if not level_df.empty:
        for lvl in sorted(level_df['player_level'].unique()):
            sub = level_df[level_df['player_level'] == lvl]
            fig3.add_trace(
                go.Bar(
                    x=sub['opponent_level'],
                    y=sub['win_rate'],
                    name=f'己方: {lvl}',
                    text=sub['games'].apply(lambda x: f'{x}局'),
                    textposition='outside'
                )
            )
        fig3.update_layout(
            barmode='group',
            xaxis={'title': '对手等级'},
            yaxis={'title': '胜率(%)', 'range': [0, 100]},
            legend=dict(orientation='h', y=1.1),
            template='plotly_white'
        )
    fig3.update_layout(title={'text': '不同等级差下的胜率对比', 'x': 0.5})

    return fig1, fig2, fig3


@app.callback(
    Output('opening-bar-chart', 'figure'),
    Output('opening-pie-chart', 'figure'),
    Output('opening-detail-table', 'data'),
    Output('opening-detail-table', 'columns'),
    Input('filtered-data', 'data'),
    Input('player-select', 'value')
)
def update_opening_charts(data, player):
    empty_fig = go.Figure()
    if not data:
        return empty_fig, empty_fig, [], []

    df = load_data(uploaded_df=pd.DataFrame(data))
    if df is None or len(df) == 0:
        return empty_fig, empty_fig, [], []

    p = None if player == 'ALL' else player
    opening_stats = get_opening_distribution(df, p)
    if opening_stats.empty:
        return empty_fig, empty_fig, [], []

    top_openings = opening_stats.head(15)

    fig1 = go.Figure()
    fig1.add_trace(
        go.Bar(
            x=top_openings['opening'],
            y=top_openings['count'],
            name='使用次数',
            marker_color='#3498db',
            opacity=0.8
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=top_openings['opening'],
            y=top_openings['win_rate'],
            name='胜率(%)',
            mode='lines+markers',
            yaxis='y2',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=10)
        )
    )
    fig1.update_layout(
        title={'text': f"常用开局及胜率 (Top 15)", 'x': 0.5},
        xaxis={'title': '开局名称', 'tickangle': -45},
        yaxis={'title': '使用次数'},
        yaxis2=dict(title='胜率(%)', overlaying='y', side='right', range=[0, 100], showgrid=False),
        legend=dict(orientation='h', y=1.1),
        template='plotly_white'
    )

    family_stats = opening_stats.groupby('opening_family').agg(
        count=('count', 'sum'),
        wins=('wins', 'sum')
    ).reset_index()
    family_stats['win_rate'] = (family_stats['wins'] / family_stats['count'] * 100).round(1)

    fig2 = px.pie(
        family_stats,
        values='count',
        names='opening_family',
        title=f"开局体系使用占比",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig2.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>使用: %{value} 次<br>占比: %{percent}<extra></extra>'
    )
    fig2.update_layout(template='plotly_white', title={'x': 0.5})

    table_df = opening_stats[['opening_family', 'opening', 'count', 'wins', 'win_rate']].copy()
    table_df['win_rate'] = table_df['win_rate'].apply(lambda x: f'{x:.1f}%')
    columns = [
        {'name': '开局体系', 'id': 'opening_family'},
        {'name': '具体开局', 'id': 'opening'},
        {'name': '使用次数', 'id': 'count'},
        {'name': '获胜局数', 'id': 'wins'},
        {'name': '胜率', 'id': 'win_rate'}
    ]

    return fig1, fig2, table_df.to_dict('records'), columns


@app.callback(
    Output('keymove-radar-chart', 'figure'),
    Output('keymove-bar-chart', 'figure'),
    Input('filtered-data', 'data'),
    Input('player-select', 'value')
)
def update_keymove_charts(data, player):
    empty_fig = go.Figure()
    if not data:
        return empty_fig, empty_fig

    df = load_data(uploaded_df=pd.DataFrame(data))
    if df is None or len(df) == 0:
        return empty_fig, empty_fig

    p = None if player == 'ALL' else player
    key_stats = get_key_move_analysis(df, p)

    fig1 = go.Figure()
    if not key_stats.empty and len(key_stats) >= 3:
        categories = ['平均胜势手', '平均失误', '胜率(%)/10', '对局数/10', '净收益×10']
        for _, row in key_stats.head(6).iterrows():
            values = [
                row['avg_winning_moves'],
                row['avg_mistakes'],
                row['win_rate'] / 10,
                row['games'] / 10,
                max(0, row['net_gain']) * 10
            ]
            fig1.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=row['opening_family'],
                    opacity=0.6
                )
            )
        fig1.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 12])),
            template='plotly_white'
        )
    fig1.update_layout(title={'text': '各开局关键手雷达图', 'x': 0.5})

    fig2 = go.Figure()
    if not key_stats.empty:
        key_stats_sorted = key_stats.sort_values('net_gain', ascending=True)
        fig2.add_trace(
            go.Bar(
                x=key_stats_sorted['avg_winning_moves'],
                y=key_stats_sorted['opening_family'],
                name='胜势手(次)',
                orientation='h',
                marker_color='#27ae60'
            )
        )
        fig2.add_trace(
            go.Bar(
                x=-key_stats_sorted['avg_mistakes'],
                y=key_stats_sorted['opening_family'],
                name='失误(次)',
                orientation='h',
                marker_color='#e74c3c'
            )
        )
        fig2.update_layout(
            barmode='relative',
            xaxis={'title': '次数'},
            yaxis={'title': '开局体系'},
            legend=dict(orientation='h', y=1.1),
            template='plotly_white'
        )
    fig2.update_layout(title={'text': '各开局胜势手 vs 失误对比', 'x': 0.5})

    return fig1, fig2


@app.callback(
    Output('style-scatter-chart', 'figure'),
    Output('style-pie-chart', 'figure'),
    Output('style-stats-table', 'data'),
    Output('style-stats-table', 'columns'),
    Input('filtered-data', 'data'),
    Input('player-select', 'value')
)
def update_style_charts(data, player):
    empty_fig = go.Figure()
    if not data:
        return empty_fig, empty_fig, [], []

    df = load_data(uploaded_df=pd.DataFrame(data))
    if df is None or len(df) == 0:
        return empty_fig, empty_fig, [], []

    player_stats = compute_player_stats(df)
    if player_stats.empty:
        return empty_fig, empty_fig, [], []

    player_stats = classify_playing_style(player_stats)

    player_stats['highlight'] = player_stats['player'].apply(
        lambda x: '★ ' + x if (player and player != 'ALL' and x == player) else x
    )
    player_stats['is_selected'] = player_stats['player'].apply(
        lambda x: 2 if (player and player != 'ALL' and x == player) else 1
    )

    fig1 = px.scatter(
        player_stats,
        x='aggression_ratio',
        y='defense_ratio',
        color='playing_style',
        size='is_selected',
        size_max=25,
        hover_data=['player', 'win_rate', 'efficiency', 'total_games'],
        color_discrete_map=STYLE_COLORS,
        title='棋风类型聚类散点图 (进攻倾向 vs 防守倾向)'
    )

    if player and player != 'ALL' and player in player_stats['player'].values:
        sel = player_stats[player_stats['player'] == player].iloc[0]
        fig1.add_annotation(
            x=sel['aggression_ratio'],
            y=sel['defense_ratio'],
            text=f"⭐ {player}",
            showarrow=True,
            arrowhead=2,
            ax=50,
            ay=-40,
            font=dict(size=14, color='black', family='Arial Bold'),
            bgcolor='rgba(255,255,0,0.7)',
            bordercolor='#f1c40f',
            borderwidth=2
        )

    fig1.update_layout(
        xaxis={'title': '进攻倾向比率 (越高越激进)', 'range': [0, 1]},
        yaxis={'title': '防守倾向比率 (越高越稳健)', 'range': [0, 1]},
        template='plotly_white',
        title={'x': 0.5, 'font': {'size': 15}}
    )

    style_counts = player_stats['playing_style'].value_counts().reset_index()
    style_counts.columns = ['playing_style', 'count']

    fig2 = px.pie(
        style_counts,
        values='count',
        names='playing_style',
        color='playing_style',
        color_discrete_map=STYLE_COLORS,
        hole=0.4
    )
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    fig2.update_layout(template='plotly_white', showlegend=False)

    if player and player != 'ALL' and player in player_stats['player'].values:
        sel_player = player_stats[player_stats['player'] == player]
        style_grouped = sel_player.rename(columns={
            'player': '棋手',
            'playing_style': '棋风类型',
            'win_rate': '胜率',
            'aggression_ratio': '进攻率',
            'defense_ratio': '防守率',
            'efficiency': '效率',
            'total_games': '总局数'
        })[['棋手', '棋风类型', '总局数', '胜率', '进攻率', '防守率', '效率']].copy()
        style_grouped['胜率'] = style_grouped['胜率'].apply(lambda x: f'{x:.1f}%')
        style_grouped['进攻率'] = style_grouped['进攻率'].apply(lambda x: f'{x*100:.1f}%')
        style_grouped['防守率'] = style_grouped['防守率'].apply(lambda x: f'{x*100:.1f}%')
        style_grouped['效率'] = style_grouped['效率'].apply(lambda x: f'{x:.2f}')
    else:
        style_grouped = player_stats.groupby('playing_style').agg(
            棋手数=('player', 'count'),
            平均胜率=('win_rate', 'mean'),
            平均进攻率=('aggression_ratio', lambda x: f'{x.mean()*100:.1f}%'),
            平均防守率=('defense_ratio', lambda x: f'{x.mean()*100:.1f}%'),
            平均效率=('efficiency', 'mean')
        ).reset_index()
        style_grouped['平均胜率'] = style_grouped['平均胜率'].apply(lambda x: f'{x:.1f}%')
        style_grouped['平均效率'] = style_grouped['平均效率'].apply(lambda x: f'{x:.2f}')
        style_grouped = style_grouped.rename(columns={'playing_style': '棋风类型'})

    columns = [{'name': col, 'id': col} for col in style_grouped.columns]

    return fig1, fig2, style_grouped.to_dict('records'), columns


@app.callback(
    Output('classic-games-table', 'data'),
    Output('classic-games-table', 'columns'),
    Input('filtered-data', 'data'),
    Input('player-select', 'value')
)
def update_classic_games(data, player):
    if not data:
        return [], []

    df = load_data(uploaded_df=pd.DataFrame(data))
    if df is None or len(df) == 0:
        return [], []

    filtered = df if player == 'ALL' else df[df['player'] == player]
    classics = get_classic_games(filtered, top_n=10)
    if classics.empty:
        return [], []

    classics['match_date'] = pd.to_datetime(classics['match_date']).dt.strftime('%Y-%m-%d')
    columns = [
        {'name': '对局ID', 'id': 'game_id'},
        {'name': '己方', 'id': 'player'},
        {'name': '对手', 'id': 'opponent'},
        {'name': '开局', 'id': 'opening'},
        {'name': '结果', 'id': 'result'},
        {'name': '手数', 'id': 'total_moves'},
        {'name': '关键胜手', 'id': 'key_winning_moves'},
        {'name': '日期', 'id': 'match_date'},
        {'name': '赛事', 'id': 'competition_type'}
    ]

    return classics.to_dict('records'), columns


@app.callback(
    Output('advice-cards', 'children'),
    Output('recommended-openings', 'children'),
    Output('common-traps', 'children'),
    Input('filtered-data', 'data'),
    Input('player-select', 'value')
)
def update_advice_section(data, player):
    if not data:
        return html.Div('暂无数据'), html.Div('暂无'), html.Div('暂无')

    df = load_data(uploaded_df=pd.DataFrame(data))
    if df is None or len(df) == 0:
        return html.Div('数据不足，无法生成建议'), html.Div('-'), html.Div('-')

    p = None if player == 'ALL' else player

    player_stats = compute_player_stats(df)
    if not player_stats.empty and p is not None:
        player_stats = player_stats[player_stats['player'] == p]
    player_stats = classify_playing_style(player_stats) if not player_stats.empty else player_stats

    opening_stats = get_opening_distribution(df, p)
    key_stats = get_key_move_analysis(df, p)

    advice_list = generate_training_advice(player_stats, opening_stats, key_stats)

    advice_cards = [_make_advice_card(a) for a in advice_list if a['category'] not in ['常见陷阱', '训练建议']]
    training_advice = [a for a in advice_list if a['category'] == '训练建议']
    for ta in training_advice:
        advice_cards.append(_make_advice_card(ta))

    if not opening_stats.empty:
        top = opening_stats[opening_stats['count'] >= 2].nlargest(3, 'win_rate')
        rec_list = []
        for _, r in top.iterrows():
            rec_list.append(
                html.Div(
                    [
                        html.Strong(f"✓ {r['opening']}"),
                        html.Span(f" — 胜率 {r['win_rate']}% ({int(r['count'])}局)", className='text-muted')
                    ],
                    className='mb-2'
                )
            )
        recommended = html.Div(rec_list) if rec_list else html.Div('暂无足够数据')
    else:
        recommended = html.Div('暂无数据')

    traps = [a for a in advice_list if a['category'] == '常见陷阱']
    trap_list = [
        html.Div(f"⚠️ {t['content']}", className='mb-2 text-dark') for t in traps
    ]
    common_traps_div = html.Div(trap_list) if trap_list else html.Div('暂无')

    return html.Div(advice_cards), recommended, common_traps_div


@app.callback(
    Output('compare-player-select', 'options'),
    Input('filtered-data', 'data')
)
def update_compare_player_options(data):
    if not data:
        return []
    df = pd.DataFrame(data)
    player_opts = []
    for p in sorted(df['player'].unique().tolist()):
        player_opts.append({'label': f'{p}', 'value': p})
    return player_opts


@app.callback(
    Output('compare-player-select', 'value'),
    Output('compare-select-hint', 'children'),
    Input('compare-player-select', 'value'),
    prevent_initial_call=True
)
def enforce_max_compare_players(selected):
    if selected is None:
        return [], ''
    if len(selected) > 3:
        return selected[:3], f'⚠️ 最多只能选择 3 名棋手进行对比，已自动截取前 3 人'
    if len(selected) == 0:
        return [], '请选择 1-3 名棋手进行对比'
    return selected, f'已选择 {len(selected)} 名棋手'


@app.callback(
    Output('compare-empty-hint', 'style'),
    Output('compare-content', 'style'),
    Output('compare-stats-table', 'data'),
    Output('compare-stats-table', 'columns'),
    Output('compare-winrate-chart', 'figure'),
    Output('compare-style-chart', 'figure'),
    Output('compare-opening-chart', 'figure'),
    Output('compare-mistake-chart', 'figure'),
    Input('filtered-data', 'data'),
    Input('compare-player-select', 'value')
)
def update_comparison_view(data, selected_players):
    empty_fig = go.Figure()
    empty_style_hide = {'display': 'none'}
    empty_style_show = {'textAlign': 'center', 'color': 'gray', 'padding': '2rem'}

    if not data or not selected_players or len(selected_players) == 0:
        return empty_style_show, empty_style_hide, [], [], empty_fig, empty_fig, empty_fig, empty_fig

    df = load_data(uploaded_df=pd.DataFrame(data))
    if df is None or len(df) == 0:
        return empty_style_show, empty_style_hide, [], [], empty_fig, empty_fig, empty_fig, empty_fig

    players = selected_players[:3]

    stats_df = compare_players_stats(df, players)
    if stats_df.empty:
        return empty_style_show, empty_style_hide, [], [], empty_fig, empty_fig, empty_fig, empty_fig

    show_style = {}
    hide_style = {'display': 'none'}

    stats_display = stats_df.rename(columns={
        'player': '棋手',
        'total_games': '总对局数',
        'wins': '胜局',
        'losses': '负局',
        'win_rate': '胜率(%)',
        'avg_moves': '平均手数',
        'avg_mistakes': '平均失误',
        'avg_winning_moves': '平均胜势手'
    })
    stats_display['胜率(%)'] = stats_display['胜率(%)'].apply(lambda x: f'{x:.1f}')
    stats_cols = [{'name': col, 'id': col} for col in stats_display.columns]

    trend_df = compare_players_win_trend(df, players)
    fig1 = go.Figure()
    if not trend_df.empty:
        for i, player in enumerate(players):
            sub = trend_df[trend_df['player'] == player]
            if not sub.empty:
                fig1.add_trace(
                    go.Scatter(
                        x=sub['match_date'],
                        y=sub['win_rate'],
                        mode='lines+markers',
                        name=player,
                        line=dict(color=COMPARE_COLORS[i % len(COMPARE_COLORS)], width=2.5),
                        marker=dict(size=7)
                    )
                )
        fig1.update_layout(
            yaxis={'title': '胜率(%)', 'range': [0, 100]},
            xaxis={'title': '月份', 'tickangle': -30},
            template='plotly_white',
            legend=dict(orientation='h', y=1.1),
            title={'text': '月度胜率趋势对比', 'x': 0.5}
        )
    else:
        fig1.update_layout(title={'text': '暂无趋势数据', 'x': 0.5}, template='plotly_white')

    style_df = compare_players_styles(df, players)
    fig2 = go.Figure()
    if not style_df.empty:
        fig2 = px.scatter(
            style_df,
            x='aggression_ratio',
            y='defense_ratio',
            color='player',
            color_discrete_sequence=COMPARE_COLORS[:len(players)],
            size='win_rate',
            size_max=25,
            hover_data=['player', 'playing_style', 'win_rate', 'efficiency', 'total_games'],
            title='棋风类型分布 (进攻倾向 vs 防守倾向)'
        )
        for _, row in style_df.iterrows():
            fig2.add_annotation(
                x=row['aggression_ratio'],
                y=row['defense_ratio'],
                text=f"{row['player']}<br>({row['playing_style']})",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-35,
                font=dict(size=11),
                bgcolor='rgba(255,255,255,0.85)'
            )
        fig2.update_layout(
            xaxis={'title': '进攻倾向比率 (越高越激进)', 'range': [0, 1]},
            yaxis={'title': '防守倾向比率 (越高越稳健)', 'range': [0, 1]},
            template='plotly_white',
            legend=dict(orientation='h', y=1.1),
            title={'x': 0.5}
        )
    else:
        fig2.update_layout(title={'text': '暂无棋风数据', 'x': 0.5}, template='plotly_white')

    opening_df = compare_players_top_openings(df, players, top_n=5)
    fig3 = go.Figure()
    if not opening_df.empty:
        for i, player in enumerate(players):
            sub = opening_df[opening_df['player'] == player].copy()
            if not sub.empty:
                sub = sub.sort_values('rank', ascending=True)
                fig3.add_trace(
                    go.Bar(
                        x=sub['opening'],
                        y=sub['count'],
                        name=f'{player} (使用次数)',
                        marker_color=COMPARE_COLORS[i % len(COMPARE_COLORS)],
                        opacity=0.8,
                        offsetgroup=i
                    )
                )
                fig3.add_trace(
                    go.Scatter(
                        x=sub['opening'],
                        y=sub['win_rate'],
                        mode='lines+markers',
                        name=f'{player} (胜率%)',
                        line=dict(color=COMPARE_COLORS[i % len(COMPARE_COLORS)], width=2, dash='dot'),
                        marker=dict(size=9, symbol='diamond'),
                        yaxis='y2'
                    )
                )
        fig3.update_layout(
            barmode='group',
            xaxis={'title': '开局名称', 'tickangle': -35},
            yaxis={'title': '使用次数'},
            yaxis2=dict(title='胜率(%)', overlaying='y', side='right', range=[0, 100], showgrid=False),
            template='plotly_white',
            legend=dict(orientation='h', y=1.15),
            title={'text': '常用开局 Top5 对比 (使用次数 & 胜率)', 'x': 0.5}
        )
    else:
        fig3.update_layout(title={'text': '暂无开局数据', 'x': 0.5}, template='plotly_white')

    mistake_df = compare_players_mistakes(df, players)
    fig4 = go.Figure()
    if not mistake_df.empty:
        for i, player in enumerate(players):
            sub = mistake_df[mistake_df['player'] == player].head(6)
            if not sub.empty:
                fig4.add_trace(
                    go.Bar(
                        x=sub['opening_family'],
                        y=sub['avg_mistakes'],
                        name=f'{player} (平均失误)',
                        marker_color=COMPARE_COLORS[i % len(COMPARE_COLORS)],
                        opacity=0.85
                    )
                )
        fig4.update_layout(
            barmode='group',
            xaxis={'title': '开局体系', 'tickangle': -30},
            yaxis={'title': '平均失误次数'},
            template='plotly_white',
            legend=dict(orientation='h', y=1.1),
            title={'text': '各开局体系关键失误分布对比', 'x': 0.5}
        )
    else:
        fig4.update_layout(title={'text': '暂无失误数据', 'x': 0.5}, template='plotly_white')

    return (
        hide_style, show_style,
        stats_display.to_dict('records'), stats_cols,
        fig1, fig2, fig3, fig4
    )


@app.callback(
    Output('selected-game-id', 'data'),
    Output('game-detail-panel', 'children'),
    Input('classic-games-table', 'derived_virtual_selected_rows'),
    Input('classic-games-table', 'derived_virtual_data'),
    State('filtered-data', 'data'),
    State('player-select', 'value'),
    State('training-game-ids', 'data'),
    prevent_initial_call=False
)
def render_game_detail(selected_rows, table_data, filtered_data, player, training_ids):
    if not selected_rows or not table_data or len(selected_rows) == 0:
        game_id = None
        panel = dbc.Card(
            [
                dbc.CardBody(
                    html.Div(
                        '请在左侧选择一局经典对局，此处将展示详细复盘信息与训练建议',
                        className='text-center text-muted py-5'
                    )
                )
            ],
            className='border-secondary'
        )
        return game_id, panel

    row_idx = selected_rows[0]
    if row_idx >= len(table_data):
        game_id = None
        panel = dbc.Card(
            [
                dbc.CardBody(
                    html.Div(
                        '请在左侧选择一局经典对局，此处将展示详细复盘信息与训练建议',
                        className='text-center text-muted py-5'
                    )
                )
            ],
            className='border-secondary'
        )
        return game_id, panel

    selected = table_data[row_idx]
    game_id = selected.get('game_id')

    df = load_data(uploaded_df=pd.DataFrame(filtered_data)) if filtered_data else None
    if df is None or len(df) == 0:
        panel = dbc.Card(
            [
                dbc.CardBody(html.Div('数据加载中...', className='text-center text-muted py-4'))
            ],
            className='border-secondary'
        )
        return game_id, panel

    detail = get_game_detail(df, game_id)
    if not detail:
        panel = dbc.Card(
            [
                dbc.CardBody(html.Div('未找到该对局的详细信息', className='text-center text-muted py-4'))
            ],
            className='border-secondary'
        )
        return game_id, panel

    advice_list = generate_game_training_advice(detail, df)
    already_in_training = game_id in (training_ids or [])

    result_color = 'success' if detail.get('win') else 'danger'
    result_text = '胜利' if detail.get('win') else '失利'

    panel_children = [
        dbc.CardHeader(
            [
                html.H5(f"🎯 {detail.get('game_id', '')} 对局详情", className='mb-0'),
                dbc.Badge(
                    f"{'🏆 ' + result_text}",
                    color=result_color,
                    className='ms-2',
                    style={'fontSize': '0.85rem'}
                )
            ],
            className='bg-light d-flex align-items-center'
        ),
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.H6('👥 棋手信息', className='fw-bold text-primary mb-2'),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div([html.Strong('己方：'), detail.get('player', '')]),
                                        html.Div([html.Strong('等级：'), detail.get('player_level', '')]),
                                        html.Div([html.Strong('棋风：'), detail.get('player_style', '')], className='text-muted small')
                                    ],
                                    md=6
                                ),
                                dbc.Col(
                                    [
                                        html.Div([html.Strong('对手：'), detail.get('opponent', '')]),
                                        html.Div([html.Strong('等级：'), detail.get('opponent_level', '')]),
                                        html.Div([html.Strong('赛事：'), detail.get('competition_type', '')], className='text-muted small')
                                    ],
                                    md=6
                                )
                            ],
                            className='mb-2'
                        ),
                        html.Div([html.Strong('日期：'), detail.get('match_date', '')], className='small text-muted')
                    ],
                    className='mb-3 p-3 bg-light rounded'
                ),

                html.Div(
                    [
                        html.H6('📚 开局体系', className='fw-bold text-info mb-2'),
                        html.Div([
                            html.Span('体系：', className='fw-bold'),
                            dbc.Badge(detail.get('opening_family', ''), color='info', className='me-2'),
                            html.Span('变例：', className='fw-bold'),
                            dbc.Badge(detail.get('opening', ''), color='secondary')
                        ])
                    ],
                    className='mb-3'
                ),

                html.Div(
                    [
                        html.H6('⚡ 关键数据', className='fw-bold text-warning mb-2'),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.Div('关键胜势手', className='small text-muted'),
                                            html.H4(str(detail.get('key_winning_moves', 0)), className='text-success fw-bold mb-0')
                                        ]),
                                        className='text-center border-success'
                                    ),
                                    md=4
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.Div('关键失误', className='small text-muted'),
                                            html.H4(str(detail.get('key_mistakes', 0)), className='text-danger fw-bold mb-0')
                                        ]),
                                        className='text-center border-danger'
                                    ),
                                    md=4
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.Div('总手数', className='small text-muted'),
                                            html.H4(str(detail.get('total_moves', 0)), className='text-primary fw-bold mb-0')
                                        ]),
                                        className='text-center border-primary'
                                    ),
                                    md=4
                                )
                            ],
                            className='mb-2'
                        ),
                        html.Div(
                            [
                                html.Span(f"⚔️ 进攻招法: {detail.get('aggressive_moves', 0)}", className='me-3'),
                                html.Span(f"🛡️ 防守招法: {detail.get('defensive_moves', 0)}", className='me-3'),
                                html.Span(f"📊 实空收益: {detail.get('territory_gain', 0)}")
                            ],
                            className='small text-muted'
                        )
                    ],
                    className='mb-3'
                ),

                html.H6('💡 训练建议', className='fw-bold mb-2'),
                html.Div([_make_advice_card(a) for a in advice_list], className='mb-3'),

                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button(
                                '➕ 加入今日训练清单' if not already_in_training else '✅ 已在训练清单中',
                                id='add-to-training-btn',
                                color='primary' if not already_in_training else 'success',
                                className='w-100',
                                disabled=already_in_training
                            ),
                            md=8
                        ),
                        dbc.Col(
                            dbc.Button(
                                '⬆️ 跳转到训练清单',
                                id='goto-training-btn',
                                color='outline-secondary',
                                className='w-100',
                                outline=True
                            ),
                            md=4
                        )
                    ]
                ),
                html.Div(id='add-training-status', className='mt-2 small text-center')
            ]
        )
    ]

    panel = dbc.Card(panel_children, className='border-primary')
    return game_id, panel


@app.callback(
    Output('training-game-ids', 'data'),
    Output('add-training-status', 'children'),
    Input('add-to-training-btn', 'n_clicks'),
    State('selected-game-id', 'data'),
    State('training-game-ids', 'data'),
    prevent_initial_call=True
)
def add_game_to_training(n_clicks, game_id, training_ids):
    if not game_id:
        return dash.no_update, '❌ 请先选择一局对局'

    training_ids = training_ids or []
    if game_id in training_ids:
        return dash.no_update, '⚠️ 该局已在训练清单中'

    new_ids = training_ids + [game_id]
    return new_ids, f'✅ 已将 {game_id} 加入今日训练清单（共 {len(new_ids)} 局）'


@app.callback(
    Output('training-total-count', 'children'),
    Output('training-weakness-count', 'children'),
    Output('training-weak-games', 'children'),
    Output('training-avg-wr', 'children'),
    Output('training-games-list', 'children'),
    Output('training-weakness-table', 'children'),
    Output('training-focus-advice', 'children'),
    Input('filtered-data', 'data'),
    Input('player-select', 'value'),
    Input('training-game-ids', 'data')
)
def update_training_list_view(filtered_data, player, training_ids):
    empty_result = (
        '0', '0', '0', '-',
        html.Div('暂无待练习对局，可在「经典对局复盘」区将感兴趣的对局加入训练清单',
                 className='text-muted small py-3 text-center'),
        html.Div('暂无足够数据生成开局弱点分析',
                 className='text-muted small py-5 text-center'),
        html.Div('根据当前筛选条件自动生成训练重点',
                 className='text-muted small py-3 text-center')
    )

    if not filtered_data:
        return empty_result

    df = load_data(uploaded_df=pd.DataFrame(filtered_data))
    if df is None or len(df) == 0:
        return empty_result

    summary = get_training_list_summary(df, training_ids or [], player)
    stats = summary.get('stats', {})
    games = summary.get('games', [])
    weaknesses = summary.get('weaknesses', pd.DataFrame())

    total_count = str(stats.get('total_training', 0))
    weak_count = str(stats.get('weak_opening_count', 0))
    weak_games = str(stats.get('total_weak_games', 0))
    avg_wr = f"{stats.get('avg_weak_winrate', 0):.1f}" if stats.get('avg_weak_winrate') else '-'

    if games:
        game_items = []
        for i, g in enumerate(games):
            res_color = 'success' if g.get('win') else 'danger'
            res_icon = '🏆' if g.get('win') else '❌'
            game_items.append(
                dbc.ListGroupItem(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            [
                                                html.Strong(f"{g.get('game_id', '')}"),
                                                dbc.Badge(f"{res_icon} {'胜' if g.get('win') else '负'}",
                                                          color=res_color, className='ms-2', pill=True)
                                            ],
                                            className='mb-1'
                                        ),
                                        html.Div(
                                            [
                                                html.Span(f"{g.get('player', '')} vs {g.get('opponent', '')}"),
                                                html.Span(f" · {g.get('match_date', '')}", className='text-muted')
                                            ],
                                            className='small'
                                        ),
                                        html.Div(
                                            [
                                                dbc.Badge(g.get('opening_family', ''), color='info', className='me-1'),
                                                html.Span(f"胜势手: {g.get('key_winning_moves', 0)} / 失误: {g.get('key_mistakes', 0)}",
                                                          className='text-muted small')
                                            ],
                                            className='mt-1'
                                        )
                                    ],
                                    md=9
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        '移除',
                                        id={'type': 'remove-training-btn', 'index': g.get('game_id', '')},
                                        color='outline-danger',
                                        size='sm',
                                        className='w-100'
                                    ),
                                    md=3,
                                    className='d-flex align-items-center'
                                )
                            ],
                            align='center'
                        )
                    ]
                )
            )
        games_list = dbc.ListGroup(game_items, flush=True)
    else:
        games_list = html.Div(
            '暂无待练习对局，可在「经典对局复盘」区将感兴趣的对局加入训练清单',
            className='text-muted small py-3 text-center'
        )

    if not weaknesses.empty:
        weak_table_data = weaknesses.copy()
        table_cols = [
            {'name': '开局体系', 'id': 'opening_family'},
            {'name': '具体开局', 'id': 'opening'},
            {'name': '使用数', 'id': 'count'},
            {'name': '胜率(%)', 'id': 'win_rate'}
        ]
        if 'avg_mistakes' in weak_table_data.columns:
            table_cols.append({'name': '平均失误', 'id': 'avg_mistakes'})
        if 'net_gain' in weak_table_data.columns:
            table_cols.append({'name': '净收益', 'id': 'net_gain'})

        show_cols = [c['id'] for c in table_cols]
        for c in show_cols:
            if c not in weak_table_data.columns:
                weak_table_data[c] = 0

        display_df = weak_table_data[show_cols].copy()
        if 'win_rate' in display_df.columns:
            display_df['win_rate'] = display_df['win_rate'].round(1)
        if 'avg_mistakes' in display_df.columns:
            display_df['avg_mistakes'] = display_df['avg_mistakes'].round(2)
        if 'net_gain' in display_df.columns:
            display_df['net_gain'] = display_df['net_gain'].round(2)

        weakness_table = dash_table.DataTable(
            data=display_df.to_dict('records'),
            columns=table_cols,
            page_size=5,
            style_table={'overflowX': 'auto'},
            style_header={'backgroundColor': '#8b4513', 'color': 'white', 'fontWeight': 'bold', 'fontSize': '12px'},
            style_cell={'textAlign': 'center', 'padding': '6px', 'fontSize': '12px'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{胜率(%)} < 40', 'column_id': 'win_rate'},
                    'backgroundColor': '#f8d7da',
                    'color': '#721c24',
                    'fontWeight': 'bold'
                }
            ]
        )
    else:
        weakness_table = html.Div(
            '暂无足够数据生成开局弱点分析（至少需要 2 局以上同类开局）',
            className='text-muted small py-5 text-center'
        )

    focus_advice_children = []
    if not weaknesses.empty:
        worst = weaknesses.iloc[0]
        focus_advice_children.append(
            html.Div(
                [
                    html.I('🎯 ', className='text-danger'),
                    html.Strong('首要突破点：'),
                    f"「{worst['opening']}」胜率仅 {worst['win_rate']:.1f}%（{int(worst['count'])}局），建议本周重点攻克"
                ],
                className='mb-2'
            )
        )
        if len(weaknesses) >= 2:
            second = weaknesses.iloc[1]
            focus_advice_children.append(
                html.Div(
                    [
                        html.I('📌 '),
                        html.Strong('次要训练点：'),
                        f"「{second['opening']}」胜率 {second['win_rate']:.1f}%（{int(second['count'])}局）"
                    ],
                    className='mb-2 text-muted'
                )
            )

        training_ids_list = training_ids or []
        if len(training_ids_list) > 0:
            focus_advice_children.append(
                html.Div(
                    [
                        html.I('📝 '),
                        html.Strong('训练进度：'),
                        f"已添加 {len(training_ids_list)} 局待复盘，建议每日完成 2-3 局深度复盘"
                    ],
                    className='mt-2 small text-muted'
                )
            )
    else:
        focus_advice_children.append(
            html.Div(
                '当前筛选条件下暂无明显开局弱点，可继续保持现有训练节奏',
                className='text-muted small py-2'
            )
        )

    focus_advice = html.Div(focus_advice_children)

    return total_count, weak_count, weak_games, avg_wr, games_list, weakness_table, focus_advice


@app.callback(
    Output('training-game-ids', 'data', allow_duplicate=True),
    Input({'type': 'remove-training-btn', 'index': ALL}, 'n_clicks'),
    State('training-game-ids', 'data'),
    prevent_initial_call=True
)
def remove_game_from_training(n_clicks_list, training_ids):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update

    trigger = ctx.triggered[0]['prop_id']
    try:
        trigger_dict = json.loads(trigger.split('.')[0])
        game_id = trigger_dict.get('index')
    except (ValueError, KeyError, AttributeError):
        return dash.no_update

    if not game_id:
        return dash.no_update

    training_ids = training_ids or []
    if game_id in training_ids:
        new_ids = [gid for gid in training_ids if gid != game_id]
        return new_ids

    return dash.no_update


@app.callback(
    Output('navbar-collapse', 'is_open'),
    Input('navbar-toggler', 'n_clicks'),
    State('navbar-collapse', 'is_open')
)
def toggle_navbar(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9501))
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception:
        app.run_server(host='0.0.0.0', port=port, debug=False)
