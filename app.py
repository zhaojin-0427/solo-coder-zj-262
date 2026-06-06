import base64
import io
import os

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from data_processing import (
    load_data, compute_player_stats, classify_playing_style,
    get_opening_distribution, get_win_rate_trend, get_heatmap_data,
    get_level_analysis, get_key_move_analysis, generate_training_advice,
    get_classic_games
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
                    dbc.NavItem(dbc.NavLink('📊 总览分析', href='#overview', external_link=True)),
                    dbc.NavItem(dbc.NavLink('🎯 布局偏好', href='#opening', external_link=True)),
                    dbc.NavItem(dbc.NavLink('📈 胜率分析', href='#winrate', external_link=True)),
                    dbc.NavItem(dbc.NavLink('🎭 棋风聚类', href='#style', external_link=True)),
                    dbc.NavItem(dbc.NavLink('🏆 经典对局', href='#games', external_link=True)),
                    dbc.NavItem(dbc.NavLink('💡 训练建议', href='#advice', external_link=True)),
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
                            md=12
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

HEATMAP_SECTION = dbc.Card(
    [
        dbc.CardHeader([html.I(className='fas fa-fire me-2'), html.Span(id='overview', style={'display': 'none'}), '🔥 布局偏好热力图 (棋手 × 开局胜率)'], className='bg-light fw-bold'),
        dbc.CardBody(dcc.Graph(id='heatmap-chart', style={'height': '500px'}))
    ],
    className='mb-4 shadow-sm'
)

WINRATE_TREND_SECTION = dbc.Card(
    [
        dbc.CardHeader([html.Span(id='winrate', style={'display': 'none'}), '📈 胜率趋势分析'], className='bg-light fw-bold'),
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

OPENING_SECTION = dbc.Card(
    [
        dbc.CardHeader([html.Span(id='opening', style={'display': 'none'}), '🎯 开局偏好与胜率分布'], className='bg-light fw-bold'),
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

STYLE_SECTION = dbc.Card(
    [
        dbc.CardHeader([html.Span(id='style', style={'display': 'none'}), '🎭 棋风类型聚类分析'], className='bg-light fw-bold'),
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

GAMES_SECTION = dbc.Card(
    [
        dbc.CardHeader([html.Span(id='games', style={'display': 'none'}), '🏆 经典对局复盘'], className='bg-light fw-bold'),
        dbc.CardBody(
            [
                dash_table.DataTable(
                    id='classic-games-table',
                    page_size=5,
                    style_table={'overflowX': 'auto'},
                    style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'},
                    style_cell={'textAlign': 'center', 'padding': '10px'},
                    style_data={'whiteSpace': 'normal', 'height': 'auto'}
                )
            ]
        )
    ],
    className='mb-4 shadow-sm'
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


ADVICE_SECTION = dbc.Card(
    [
        dbc.CardHeader([html.Span(id='advice', style={'display': 'none'}), '💡 智能训练建议区'], className='bg-light fw-bold'),
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
        dcc.Store(id='filtered-data')
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
    Output('player-select', 'options'),
    Input('stored-data', 'data')
)
def update_filter_options(data):
    if not data:
        return [{'label': '全部', 'value': '全部'}], [{'label': '全部', 'value': '全部'}], [{'label': '全部', 'value': '全部'}], [{'label': '全体棋手', 'value': 'ALL'}]
    df = pd.DataFrame(data)

    def make_options(series):
        opts = [{'label': '全部', 'value': '全部'}]
        for val in sorted(series.dropna().unique().tolist()):
            opts.append({'label': str(val), 'value': str(val)})
        return opts

    player_opts = [{'label': '全体棋手', 'value': 'ALL'}]
    for p in sorted(df['player'].unique().tolist()):
        player_opts.append({'label': f'{p}', 'value': p})

    return make_options(df['era']), make_options(df['school']), make_options(df['competition_type']), player_opts


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
    Input('filtered-data', 'data')
)
def update_overview_stats(data):
    if not data:
        return '0', '-', '0%', '0', '0'
    df = load_data(uploaded_df=pd.DataFrame(data))
    if df is None:
        return '0', '-', '0%', '0', '0'

    total = len(df)
    go_count = len(df[df['game_type'] == '围棋'])
    xq_count = len(df[df['game_type'] == '象棋'])
    breakdown = f'围棋 {go_count} 局 · 象棋 {xq_count} 局'
    avg_wr = f"{df['win'].mean() * 100:.1f}%"
    player_cnt = df['player'].nunique()
    opening_cnt = df['opening_family'].nunique()

    return f'{total} 局', breakdown, avg_wr, f'{player_cnt} 位', f'{opening_cnt} 种'


@app.callback(
    Output('heatmap-chart', 'figure'),
    Input('filtered-data', 'data')
)
def update_heatmap(data):
    if not data:
        return go.Figure()
    df = load_data(uploaded_df=pd.DataFrame(data))
    if df is None or len(df) == 0:
        return go.Figure()

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

    fig.update_layout(
        title={'text': '各棋手在不同布局体系下的胜率分布', 'x': 0.5, 'font': {'size': 16}},
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
    Input('filtered-data', 'data')
)
def update_style_charts(data):
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

    fig1 = px.scatter(
        player_stats,
        x='aggression_ratio',
        y='defense_ratio',
        color='playing_style',
        size='total_games',
        hover_data=['player', 'win_rate', 'efficiency', 'total_games'],
        color_discrete_map=STYLE_COLORS,
        title='棋风类型聚类散点图 (进攻倾向 vs 防守倾向)'
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
