import json
import re
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba, LinearSegmentedColormap
import seaborn as sns
import requests
import matplotlib.patches as patches
from mplsoccer import Pitch, VerticalPitch, add_image
from matplotlib import rcParams
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patheffects as path_effects
from highlight_text import ax_text, fig_text
from PIL import Image
from urllib.request import urlopen
from unidecode import unidecode
from scipy.spatial import ConvexHull
import streamlit as st
import os
import arabic_reshaper
from bidi.algorithm import get_display

# إعدادات matplotlib لدعم العربية
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Noto Sans Arabic', 'Amiri', 'DejaVu Sans', 'Arial', 'Tahoma']
mpl.rcParams['axes.unicode_minus'] = False

# دالة لتحويل النص العربي
def reshape_arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(str(text))
    return get_display(reshaped_text)

# تعريف الدالة reset_confirmed قبل استخدامها
def reset_confirmed():
    st.session_state['confirmed'] = False

# إضافة CSS محسّن لدعم RTL في Streamlit مع استثناءات للرسومات
st.markdown("""
    <style>
    @font-face {
        font-family: 'Noto Sans Arabic';
        src: url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;700&display=swap');
    }
    /* تطبيق RTL فقط على العناصر النصية */
    body, .stApp {
        font-family: 'Noto Sans Arabic', 'Amiri', 'DejaVu Sans', 'Arial', sans-serif !important;
    }
    h1, h2, h3, h4, h5, h6, p, div, span, label, button, input, select, option, table, th, td {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Noto Sans Arabic', 'Amiri', 'DejaVu Sans', 'Arial', sans-serif !important;
    }
    .stSelectbox, .stSelectbox div, .stSelectbox label, .stSelectbox select, .stSelectbox option {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Noto Sans Arabic', 'Amiri', 'DejaVu Sans', 'Arial', sans-serif !important;
    }
    .stRadio, .stRadio div, .stRadio label, .stRadio input {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Noto Sans Arabic', 'Amiri', 'DejaVu Sans', 'Arial', sans-serif !important;
    }
    .stTabs, .stTabs div, .stTabs button {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Noto Sans Arabic', 'Amiri', 'DejaVu Sans', 'Arial', sans-serif !important;
    }
    .stDataFrame, .dataframe, table, th, td {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Noto Sans Arabic', 'Amiri', 'DejaVu Sans', 'Arial', sans-serif !important;
    }
    .stSidebar, .stSidebar div, .stSidebar label, .stSidebar select, .stSidebar option {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Noto Sans Arabic', 'Amiri', 'DejaVu Sans', 'Arial', sans-serif !important;
    }
    [data-testid="stMarkdownContainer"], [data-testid="stText"] {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Noto Sans Arabic', 'Amiri', 'DejaVu Sans', 'Arial', sans-serif !important;
    }
    .stButton, .stButton button {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Noto Sans Arabic', 'Amiri', 'DejaVu Sans', 'Arial', sans-serif !important;
    }
    /* استثناء الرسومات والصور من RTL */
    canvas, img, .stImage, [data-testid="stImage"], .stPlotlyChart, .stPyplot {
        direction: ltr !important;
    }
    </style>
    """, unsafe_allow_html=True)

# تعريف القيم الافتراضية للألوان
default_hcol = '#d00000'
default_acol = '#003087'
default_bg_color = '#1e1e2f'
default_gradient_colors = ['#003087', '#d00000']

# إضافة أدوات اختيار الألوان في الشريط الجانبي
st.sidebar.title(reshape_arabic_text('اختيار الألوان'))
hcol = st.sidebar.color_picker(reshape_arabic_text('لون الفريق المضيف'), default_hcol, key='hcol_picker')
acol = st.sidebar.color_picker(reshape_arabic_text('لون الفريق الضيف'), default_acol, key='acol_picker')
bg_color = st.sidebar.color_picker(reshape_arabic_text('لون الخلفية'), default_bg_color, key='bg_color_picker')
gradient_start = st.sidebar.color_picker(reshape_arabic_text('بداية التدرج'), default_gradient_colors[0], key='gradient_start_picker')
gradient_end = st.sidebar.color_picker(reshape_arabic_text('نهاية التدرج'), default_gradient_colors[1], key='gradient_end_picker')
gradient_colors = [gradient_start, gradient_end]
line_color = st.sidebar.color_picker(reshape_arabic_text('لون الخطوط'), '#ffffff', key='line_color_picker')

st.sidebar.title(reshape_arabic_text('اختيار المباراة'))

season = None
league = None
stage = None
htn = None
atn = None

if 'confirmed' not in st.session_state:
    st.session_state.confirmed = False

season = st.sidebar.selectbox(reshape_arabic_text('اختر الموسم:'), [reshape_arabic_text('2024_25')], key='season', index=0, on_change=reset_confirmed)
if season:
    league_options = [
        reshape_arabic_text('الدوري الإسباني'),
        reshape_arabic_text('الدوري الإنجليزي الممتاز'),
        reshape_arabic_text('الدوري الإيطالي'),
        reshape_arabic_text('دوري أبطال أوروبا')
    ]
    league = st.sidebar.selectbox(reshape_arabic_text('اختر الدوري:'), league_options, key='league', index=None, on_change=reset_confirmed)

    if league == reshape_arabic_text('الدوري الإسباني'):
        team_list = [
            reshape_arabic_text('Athletic Club'), reshape_arabic_text('Atletico Madrid'), reshape_arabic_text('Barcelona'),
            reshape_arabic_text('Celta Vigo'), reshape_arabic_text('Deportivo Alaves'), reshape_arabic_text('Espanyol'),
            reshape_arabic_text('Getafe'), reshape_arabic_text('Girona'), reshape_arabic_text('Las Palmas'),
            reshape_arabic_text('Leganes'), reshape_arabic_text('Mallorca'), reshape_arabic_text('Osasuna'),
            reshape_arabic_text('Rayo Vallecano'), reshape_arabic_text('Real Betis'), reshape_arabic_text('Real Madrid'),
            reshape_arabic_text('Real Sociedad'), reshape_arabic_text('Real Valladolid'), reshape_arabic_text('Sevilla'),
            reshape_arabic_text('Valencia'), reshape_arabic_text('Villarreal')
        ]
    elif league == reshape_arabic_text('الدوري الإنجليزي الممتاز'):
        team_list = [
            reshape_arabic_text('Arsenal'), reshape_arabic_text('Aston Villa'), reshape_arabic_text('Bournemouth'),
            reshape_arabic_text('Brentford'), reshape_arabic_text('Brighton'), reshape_arabic_text('Chelsea'),
            reshape_arabic_text('Crystal Palace'), reshape_arabic_text('Everton'), reshape_arabic_text('Fulham'),
            reshape_arabic_text('Ipswich'), reshape_arabic_text('Leicester'), reshape_arabic_text('Liverpool'),
            reshape_arabic_text('Manchester City'), reshape_arabic_text('Manchester United'), reshape_arabic_text('Newcastle'),
            reshape_arabic_text('Nottingham Forest'), reshape_arabic_text('Southampton'), reshape_arabic_text('Tottenham'),
            reshape_arabic_text('West Ham'), reshape_arabic_text('Wolves')
        ]
    elif league == reshape_arabic_text('الدوري الإيطالي'):
        team_list = [
            reshape_arabic_text('AC Milan'), reshape_arabic_text('Atalanta'), reshape_arabic_text('Bologna'),
            reshape_arabic_text('Cagliari'), reshape_arabic_text('Como'), reshape_arabic_text('Empoli'),
            reshape_arabic_text('Fiorentina'), reshape_arabic_text('Genoa'), reshape_arabic_text('Inter'),
            reshape_arabic_text('Juventus'), reshape_arabic_text('Lazio'), reshape_arabic_text('Lecce'),
            reshape_arabic_text('Monza'), reshape_arabic_text('Napoli'), reshape_arabic_text('Parma Calcio'),
            reshape_arabic_text('Roma'), reshape_arabic_text('Torino'), reshape_arabic_text('Udinese'),
            reshape_arabic_text('Venezia'), reshape_arabic_text('Verona')
        ]
    elif league == reshape_arabic_text('دوري أبطال أوروبا'):
        team_list = [
            reshape_arabic_text('AC Milan'), reshape_arabic_text('Arsenal'), reshape_arabic_text('Aston Villa'),
            reshape_arabic_text('Atalanta'), reshape_arabic_text('Atletico Madrid'), reshape_arabic_text('BSC Young Boys'),
            reshape_arabic_text('Barcelona'), reshape_arabic_text('Bayer Leverkusen'), reshape_arabic_text('Bayern Munich'),
            reshape_arabic_text('Benfica'), reshape_arabic_text('Bologna'), reshape_arabic_text('Borussia Dortmund'),
            reshape_arabic_text('Brest'), reshape_arabic_text('Celtic'), reshape_arabic_text('Club Brugge'),
            reshape_arabic_text('Dinamo Zagreb'), reshape_arabic_text('FK Crvena Zvezda'), reshape_arabic_text('Feyenoord'),
            reshape_arabic_text('Girona'), reshape_arabic_text('Inter'), reshape_arabic_text('Juventus'),
            reshape_arabic_text('Lille'), reshape_arabic_text('Liverpool'), reshape_arabic_text('Manchester City'),
            reshape_arabic_text('Monaco'), reshape_arabic_text('PSV Eindhoven'), reshape_arabic_text('Paris Saint-Germain'),
            reshape_arabic_text('RB Leipzig'), reshape_arabic_text('Real Madrid'), reshape_arabic_text('Salzburg'),
            reshape_arabic_text('Shakhtar Donetsk'), reshape_arabic_text('Slovan Bratislava'), reshape_arabic_text('Sparta Prague'),
            reshape_arabic_text('Sporting CP'), reshape_arabic_text('Sturm Graz'), reshape_arabic_text('VfB Stuttgart')
        ]

    if league and league != reshape_arabic_text('دوري أبطال أوروبا'):
        htn = st.sidebar.selectbox(reshape_arabic_text('اختر الفريق المضيف'), team_list, key='home_team', index=None, on_change=reset_confirmed)
        
        if htn:
            atn_options = [team for team in team_list if team != htn]
            atn = st.sidebar.selectbox(reshape_arabic_text('اختر الفريق الضيف'), atn_options, key='away_team', index=None, on_change=reset_confirmed)
            
    elif league == reshape_arabic_text('دوري أبطال أوروبا'):
        stage_options = [
            reshape_arabic_text('مرحلة الدوري'),
            reshape_arabic_text('الملحق التأهيلي'),
            reshape_arabic_text('دور الـ 16'),
            reshape_arabic_text('ربع النهائي'),
            reshape_arabic_text('نصف النهائي'),
            reshape_arabic_text('النهائي')
        ]
        stage = st.sidebar.selectbox(reshape_arabic_text('اختر المرحلة'), stage_options, key='stage_selection', index=None, on_change=reset_confirmed)
        if stage:
            htn = st.sidebar.selectbox(reshape_arabic_text('اختر الفريق المضيف'), team_list, key='home_team', index=None, on_change=reset_confirmed)
            
            if htn:
                atn_options = [team for team in team_list if team != htn]
                atn = st.sidebar.selectbox(reshape_arabic_text('اختر الفريق الضيف'), atn_options, key='away_team', index=None, on_change=reset_confirmed)

    if league and league != reshape_arabic_text('دوري أبطال أوروبا') and league != reshape_arabic_text('الدوري الإيطالي') and htn and atn:
        league_en = {
            reshape_arabic_text('الدوري الإسباني'): 'La Liga',
            reshape_arabic_text('الدوري الإنجليزي الممتاز'): 'Premier League',
            reshape_arabic_text('الدوري الإيطالي'): 'Serie A',
            reshape_arabic_text('دوري أبطال أوروبا'): 'UEFA Champions League'
        }.get(league, league)
        league = league_en.replace(' ', '_')
        match_html_path = f"https://raw.githubusercontent.com/leo997a/{season}_{league}/refs/heads/main/{htn}_vs_{atn}.html"
        match_html_path = match_html_path.replace(' ', '%20')
        try:
            response = requests.get(match_html_path)
            response.raise_for_status()
            match_input = st.sidebar.button(reshape_arabic_text('تأكيد الاختيارات'), on_click=lambda: st.session_state.update({'confirmed': True}))
        except:
            st.session_state['confirmed'] = False
            st.sidebar.write(reshape_arabic_text('لم يتم العثور على المباراة'))
            
    elif league and league == reshape_arabic_text('الدوري الإيطالي') and htn and atn:
        league_en = 'Serie A'
        league = league_en.replace(' ', '_')
        match_html_path = f"https://raw.githubusercontent.com/leo997a/{season}_{league}/refs/heads/main/{htn}_vs_{atn}.html"
        match_html_path = match_html_path.replace(' ', '%20')
        try:
            response = requests.get(match_html_path)
            response.raise_for_status()
            match_input = st.sidebar.button(reshape_arabic_text('تأكيد الاختيارات'), on_click=lambda: st.session_state.update({'confirmed': True}))
        except:
            st.session_state['confirmed'] = False
            st.sidebar.write(reshape_arabic_text('مباريات الدوري الإيطالي متاحة حتى الأسبوع 12\nسيتم رفع باقي البيانات قريبًا\nشكرًا لصبرك'))
            
    elif league and league == reshape_arabic_text('دوري أبطال أوروبا') and stage and htn and atn:
        league_en = 'UEFA Champions League'
        stage_en = stage
        stage_mapping = {
            reshape_arabic_text('مرحلة الدوري'): 'League Phase',
            reshape_arabic_text('الملحق التأهيلي'): 'Knockout Playoff',
            reshape_arabic_text('دور الـ 16'): 'Round of 16',
            reshape_arabic_text('ربع النهائي'): 'Quarter Final',
            reshape_arabic_text('نصف النهائي'): 'Semi Final',
            reshape_arabic_text('النهائي'): 'Final'
        }
        stage_en = stage_mapping[stage]
        league = league_en.replace(' ', '_')
        match_html_path = f"https://raw.githubusercontent.com/leo997a/{season}_{league}/refs/heads/main/{stage_en}/{htn}_vs_{atn}.html"
        match_html_path = match_html_path.replace(' ', '%20')
        try:
            response = requests.get(match_html_path)
            response.raise_for_status()
            match_input = st.sidebar.button(reshape_arabic_text('تأكيد الاختيارات'), on_click=lambda: st.session_state.update({'confirmed': True}))
        except:
            st.session_state['confirmed'] = False
            st.sidebar.write(reshape_arabic_text('لم يتم العثور على المباراة'))

if league and htn and atn and st.session_state.confirmed:
    @st.cache_data
    def get_event_data(season, league, stage, htn, atn):
        if league == 'UEFA_Champions_League':
            match_html_path = f"https://raw.githubusercontent.com/leo997a/{season}_{league}/refs/heads/main/{stage}/{htn}_vs_{atn}.html"
        else:
            match_html_path = f"https://raw.githubusercontent.com/leo997a/{season}_{league}/refs/heads/main/{htn}_vs_{atn}.html"
        match_html_path = match_html_path.replace(' ', '%20')
        response = requests.get(match_html_path)
        response.raise_for_status()
        match_html = response.text
        match_json = re.search(r'var matchCentreData = ({.*});', match_html).group(1)
        match_data = json.loads(match_json)
        events = match_data['events']
        df = pd.DataFrame(events)
        teams_dict = match_data['header']['teams']
        players_dict = match_data['playerStats']
        players_list = []
        for player_id, player_info in players_dict.items():
            player_info['playerId'] = player_id
            players_list.append(player_info)
        players_df = pd.DataFrame(players_list)
        return df, teams_dict, players_df

    df, teams_dict, players_df = get_event_data(season, league, stage, htn, atn)

    hteamID = list(teams_dict.keys())[0]
    ateamID = list(teams_dict.keys())[1]
    hteamName = teams_dict[hteamID]
    ateamName = teams_dict[ateamID]

    homedf = df[(df['teamName'] == hteamName)]
    awaydf = df[(df['teamName'] == ateamName)]
    hxT = homedf['xT'].sum().round(2)
    axT = awaydf['xT'].sum().round(2)

    hgoal_count = len(homedf[(homedf['teamName'] == hteamName) & (homedf['type'] == 'Goal') & (~homedf['qualifiers'].str.contains('OwnGoal'))])
    agoal_count = len(awaydf[(awaydf['teamName'] == ateamName) & (awaydf['type'] == 'Goal') & (~awaydf['qualifiers'].str.contains('OwnGoal'))])
    hgoal_count = hgoal_count + len(awaydf[(awaydf['teamName'] == ateamName) & (awaydf['type'] == 'Goal') & (awaydf['qualifiers'].str.contains('OwnGoal'))])
    agoal_count = agoal_count + len(homedf[(homedf['teamName'] == hteamName) & (homedf['type'] == 'Goal') & (homedf['qualifiers'].str.contains('OwnGoal'))])

    df_teamNameId = pd.read_csv('https://raw.githubusercontent.com/adnaaan433/pmr_app/refs/heads/main/teams_name_and_id.csv')
    hftmb_tid = df_teamNameId[df_teamNameId['teamName'] == hteamName].teamId.to_list()[0]
    aftmb_tid = df_teamNameId[df_teamNameId['teamName'] == ateamName].teamId.to_list()[0]

    league_display_mapping = {
        'La Liga': reshape_arabic_text('الدوري الإسباني'),
        'Premier League': reshape_arabic_text('الدوري الإنجليزي الممتاز'),
        'Serie A': reshape_arabic_text('الدوري الإيطالي'),
        'UEFA Champions League': reshape_arabic_text('دوري أبطال أوروبا')
    }
    league_display = league_display_mapping.get(league, league)

    st.header(reshape_arabic_text(f'{hteamName} {hgoal_count} - {agoal_count} {ateamName}'))
    st.text(league_display)

    tab1, tab2 = st.tabs([
        reshape_arabic_text("تحليل المباراة"),
        reshape_arabic_text("تبويب آخر")
    ])

    options = [
        'شبكة التمريرات',
        'Defensive Actions Heatmap',
        'Progressive Passes',
        'Progressive Carries',
        'Shotmap',
        'GK Saves',
        'Match Momentum',
        'Zone14 & Half-Space Passes',
        'Final Third Entries',
        'Box Entries',
        'High-Turnovers',
        'Chances Creating Zones',
        'Crosses',
        'Team Domination Zones',
        'Pass Target Zones'
    ]
    options_display = [reshape_arabic_text(opt) for opt in options]
    st.session_state['analysis_type'] = st.selectbox(
        reshape_arabic_text('نوع التحليل:'),
        options_display,
        index=0,
        key='analysis_type_selectbox'
    )

    def pass_network(ax, team_name, col, phase_tag):
        if phase_tag == 'Full Time':
            df_pass = df.copy()
            df_pass = df_pass.reset_index(drop=True)
        elif phase_tag == 'First Half':
            df_pass = df[df['period'] == 'FirstHalf']
            df_pass = df_pass.reset_index(drop=True)
        elif phase_tag == 'Second Half':
            df_pass = df[df['period'] == 'SecondHalf']
            df_pass = df_pass.reset_index(drop=True)

        total_pass = df_pass[(df_pass['teamName'] == team_name) & (df_pass['type'] == 'Pass')]
        accrt_pass = df_pass[(df_pass['teamName'] == team_name) & (df_pass['type'] == 'Pass') & (df_pass['outcomeType'] == 'Successful')]
        
        if len(total_pass) == 0:
            ax.text(34, 50, reshape_arabic_text('لا توجد بيانات تمريرات متاحة'), color='white', fontsize=14, ha='center', va='center')
            return None

        accuracy = round((len(accrt_pass) / len(total_pass)) * 100, 2)

        df_pass['pass_receiver'] = df_pass.loc[(df_pass['type'] == 'Pass') & (df_pass['outcomeType'] == 'Successful') & (df_pass['teamName'].shift(-1) == team_name), 'name'].shift(-1)
        df_pass['pass_receiver'] = df_pass['pass_receiver'].fillna('No')

        off_acts_df = df_pass[(df_pass['teamName'] == team_name) & (df_pass['type'].isin(['Pass', 'Goal', 'MissedShots', 'SavedShot', 'ShotOnPost', 'TakeOn', 'BallTouch', 'KeeperPickup']))]
        off_acts_df = off_acts_df[['name', 'x', 'y']].reset_index(drop=True)
        avg_locs_df = off_acts_df.groupby('name').agg(avg_x=('x', 'median'), avg_y=('y', 'median')).reset_index()
        team_pdf = players_df[['name', 'shirtNo', 'position', 'isFirstEleven']]
        avg_locs_df = avg_locs_df.merge(team_pdf, on='name', how='left')

        df_pass = df_pass[(df_pass['type'] == 'Pass') & (df_pass['outcomeType'] == 'Successful') & (df_pass['teamName'] == team_name) & (~df_pass['qualifiers'].str.contains('Corner|Freekick'))]
        df_pass = df_pass[['type', 'name', 'pass_receiver']].reset_index(drop=True)

        pass_count_df = df_pass.groupby(['name', 'pass_receiver']).size().reset_index(name='pass_count').sort_values(by='pass_count', ascending=False)
        pass_count_df = pass_count_df.reset_index(drop=True)

        pass_counts_df = pd.merge(pass_count_df, avg_locs_df, on='name', how='left')
        pass_counts_df.rename(columns={'avg_x': 'pass_avg_x', 'avg_y': 'pass_avg_y'}, inplace=True)
        pass_counts_df = pd.merge(pass_counts_df, avg_locs_df, left_on='pass_receiver', right_on='name', how='left', suffixes=('', '_receiver'))
        pass_counts_df.drop(columns=['name_receiver'], inplace=True)
        pass_counts_df.rename(columns={'avg_x': 'receiver_avg_x', 'avg_y': 'receiver_avg_y'}, inplace=True)
        pass_counts_df = pass_counts_df.sort_values(by='pass_count', ascending=False).reset_index(drop=True)
        pass_counts_df = pass_counts_df.dropna(subset=['shirtNo_receiver'])
        pass_btn = pass_counts_df[['name', 'shirtNo', 'pass_receiver', 'shirtNo_receiver', 'pass_count']]
        pass_btn['shirtNo_receiver'] = pass_btn['shirtNo_receiver'].astype(float).astype(int)

        MAX_LINE_WIDTH = 8
        MIN_LINE_WIDTH = 0.5
        MIN_TRANSPARENCY = 0.2
        MAX_TRANSPARENCY = 0.9

        pass_counts_df['line_width'] = (pass_counts_df['pass_count'] / pass_counts_df['pass_count'].max()) * (MAX_LINE_WIDTH - MIN_LINE_WIDTH) + MIN_LINE_WIDTH
        c_transparency = pass_counts_df['pass_count'] / pass_counts_df['pass_count'].max()
        c_transparency = (c_transparency * (MAX_TRANSPARENCY - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
        color = np.array(to_rgba(col))
        color = np.tile(color, (len(pass_counts_df), 1))
        color[:, 3] = c_transparency

        pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, linewidth=1.5, line_color=line_color)
        pitch.draw(ax=ax)

        gradient = LinearSegmentedColormap.from_list("pitch_gradient", gradient_colors, N=100)
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = Y
        ax.imshow(Z, extent=[0, 68, 0, 105], cmap=gradient, alpha=0.8, aspect='auto', zorder=0)
        pitch.draw(ax=ax)

        for idx in range(len(pass_counts_df)):
            pitch.lines(
                pass_counts_df['pass_avg_x'].iloc[idx],
                pass_counts_df['pass_avg_y'].iloc[idx],
                pass_counts_df['receiver_avg_x'].iloc[idx],
                pass_counts_df['receiver_avg_y'].iloc[idx],
                lw=pass_counts_df['line_width'].iloc[idx],
                color=color[idx],
                zorder=1,
                ax=ax
            )

        for index, row in avg_locs_df.iterrows():
            if row['isFirstEleven'] == True:
                pitch.scatter(row['avg_x'], row['avg_y'], s=800, marker='o', color=col, edgecolor=line_color, linewidth=1.5, alpha=0.9, ax=ax)
            else:
                pitch.scatter(row['avg_x'], row['avg_y'], s=800, marker='s', color=col, edgecolor=line_color, linewidth=1.5, alpha=0.7, ax=ax)

        for index, row in avg_locs_df.iterrows():
            player_initials = row["shirtNo"]
            pitch.annotate(player_initials, xy=(row.avg_x, row.avg_y), c='white', ha='center', va='center', size=14, weight='bold', ax=ax)

        avgph = round(avg_locs_df['avg_x'].median(), 2)
        ax.axhline(y=avgph, color='white', linestyle='--', alpha=0.5, linewidth=1.5)

        center_backs_height = avg_locs_df[avg_locs_df['position'] == 'DC']
        def_line_h = round(center_backs_height['avg_x'].median(), 2)
        Forwards_height = avg_locs_df[avg_locs_df['isFirstEleven'] == 1].sort_values(by='avg_x', ascending=False).head(2)
        fwd_line_h = round(Forwards_height['avg_x'].mean(), 2)
        ymid = [0, 0, 68, 68]
        xmid = [def_line_h, fwd_line_h, fwd_line_h, def_line_h]
        ax.fill(ymid, xmid, col, alpha=0.2)

        v_comp = round((1 - ((fwd_line_h - def_line_h) / 105)) * 100, 2)

        if phase_tag == 'Full Time':
            ax.text(34, 115, reshape_arabic_text('الوقت بالكامل: 0-90 دقيقة'), color='white', fontsize=14, ha='center', va='center', weight='bold')
            ax.text(34, 110, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), color='white', fontsize=12, ha='center', va='center')
        elif phase_tag == 'First Half':
            ax.text(34, 115, reshape_arabic_text('الشوط الأول: 0-45 دقيقة'), color='white', fontsize=14, ha='center', va='center', weight='bold')
            ax.text(34, 110, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), color='white', fontsize=12, ha='center', va='center')
        elif phase_tag == 'Second Half':
            ax.text(34, 115, reshape_arabic_text('الشوط الثاني: 45-90 دقيقة'), color='white', fontsize=14, ha='center', va='center', weight='bold')
            ax.text(34, 110, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), color='white', fontsize=12, ha='center', va='center')

        ax.text(34, -5, reshape_arabic_text(f"على الكرة\nالتماسك العمودي (المنطقة المظللة): {v_comp}%"), color='white', fontsize=12, ha='center', va='center', weight='bold')

        return pass_btn

    with tab1:
        if st.session_state['analysis_type'] == reshape_arabic_text('شبكة التمريرات'):
            st.header(reshape_arabic_text('شبكة التمريرات'))
            
            pn_time_phase = st.radio(
                reshape_arabic_text("اختر فترة المباراة:"),
                [reshape_arabic_text("الوقت الكامل"), reshape_arabic_text("الشوط الأول"), reshape_arabic_text("الشوط الثاني")],
                index=0,
                key='pn_time_pill'
            )

            fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
            home_pass_btn = None
            away_pass_btn = None

            if pn_time_phase == reshape_arabic_text('الوقت الكامل'):
                home_pass_btn = pass_network(axs[0], hteamName, hcol, 'Full Time')
                away_pass_btn = pass_network(axs[1], ateamName, acol, 'Full Time')
            elif pn_time_phase == reshape_arabic_text('الشوط الأول'):
                home_pass_btn = pass_network(axs[0], hteamName, hcol, 'First Half')
                away_pass_btn = pass_network(axs[1], ateamName, acol, 'First Half')
            elif pn_time_phase == reshape_arabic_text('الشوط الثاني'):
                home_pass_btn = pass_network(axs[0], hteamName, hcol, 'Second Half')
                away_pass_btn = pass_network(axs[1], ateamName, acol, 'Second Half')

            if home_pass_btn is not None and away_pass_btn is not None:
                # معالجة العنوان
                home_part = reshape_arabic_text(f"{hteamName} {hgoal_count}")
                away_part = reshape_arabic_text(f"{agoal_count} {ateamName}")
                title = f"<{home_part}> - <{away_part}>"
                fig_text(0.5, 1.05, title, 
                         highlight_textprops=[{'color': hcol}, {'color': acol}],
                         fontsize=28, fontweight='bold', ha='center', va='center', ax=fig)
                fig.text(0.5, 1.01, reshape_arabic_text('شبكة التمريرات'), fontsize=18, ha='center', va='center', color='white', weight='bold')
                fig.text(0.5, 0.97, '@REO_SHOW', fontsize=10, ha='center', va='center', color='white')

                fig.text(0.5, 0.02, reshape_arabic_text('*الدوائر = اللاعبون الأساسيون، المربعات = اللاعبون البدلاء، الأرقام داخلها = أرقام القمصان'),
                         fontsize=10, fontstyle='italic', ha='center', va='center', color='white')
                fig.text(0.5, 0.00, reshape_arabic_text('*عرض وإضاءة الخطوط تمثل عدد التمريرات الناجحة في اللعب المفتوح بين اللاعبين'),
                         fontsize=10, fontstyle='italic', ha='center', va='center', color='white')

                himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png")
                himage = Image.open(himage)
                ax_himage = add_image(himage, fig, left=0.085, bottom=0.97, width=0.125, height=0.125)

                aimage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png")
                aimage = Image.open(aimage)
                ax_aimage = add_image(aimage, fig, left=0.815, bottom=0.97, width=0.125, height=0.125)

                plt.subplots_adjust(top=0.85, bottom=0.15)
                st.pyplot(fig)

                col1, col2 = st.columns(2)
                with col1:
                    st.write(reshape_arabic_text(f'أزواج التمرير لفريق {hteamName}:'))
                    home_pass_btn_display = home_pass_btn.copy()
                    for col in home_pass_btn_display.columns:
                        if home_pass_btn_display[col].dtype == "object":
                            home_pass_btn_display[col] = home_pass_btn_display[col].apply(lambda x: reshape_arabic_text(str(x)))
                    st.dataframe(home_pass_btn_display, hide_index=True)
                with col2:
                    st.write(reshape_arabic_text(f'أزواج التمرير لفريق {ateamName}:'))
                    away_pass_btn_display = away_pass_btn.copy()
                    for col in away_pass_btn_display.columns:
                        if away_pass_btn_display[col].dtype == "object":
                            away_pass_btn_display[col] = away_pass_btn_display[col].apply(lambda x: reshape_arabic_text(str(x)))
                    st.dataframe(away_pass_btn_display, hide_index=True)
            else:
                st.write(reshape_arabic_text("لا توجد بيانات متاحة لعرض شبكة التمريرات."))
if an_tp == 'Defensive Actions Heatmap':
    st.header(f'{an_tp}')
            
def def_acts_hm(ax, team_name, col, phase_tag):
    def_acts_id = df.index[((df['type'] == 'Aerial') & (df['qualifiers'].str.contains('Defensive'))) |
                           (df['type'] == 'BallRecovery') |
                           (df['type'] == 'BlockedPass') |
                           (df['type'] == 'Challenge') |
                           (df['type'] == 'Clearance') |
                           ((df['type'] == 'Save') & (df['position'] != 'GK')) |
                           ((df['type'] == 'Foul') & (df['outcomeType'] == 'Unsuccessful')) |
                           (df['type'] == 'Interception') |
                           (df['type'] == 'Tackle')]
    df_def = df.loc[def_acts_id, ["x", "y", "teamName", "name", "type", "outcomeType", "period"]]
    if phase_tag == 'Full Time':
        df_def = df_def.reset_index(drop=True)
    elif phase_tag == 'First Half':
        df_def = df_def[df_def['period'] == 'FirstHalf']
        df_def = df_def.reset_index(drop=True)
    elif phase_tag == 'Second Half':
        df_def = df_def[df_def['period'] == 'SecondHalf']
        df_def = df_def.reset_index(drop=True)

    total_def_acts = df_def[(df_def['teamName'] == team_name)]

    avg_locs_df = total_def_acts.groupby('name').agg({'x': ['median'], 'y': ['median', 'count']}).reset_index('name')
    avg_locs_df.columns = ['name', 'x', 'y', 'def_acts_count']
    avg_locs_df = avg_locs_df.sort_values(by='def_acts_count', ascending=False)
    team_pdf = players_df[['name', 'shirtNo', 'position', 'isFirstEleven']]
    avg_locs_df = avg_locs_df.merge(team_pdf, on='name', how='left')
    avg_locs_df = avg_locs_df[avg_locs_df['position'] != 'GK']
    avg_locs_df = avg_locs_df.dropna(subset=['shirtNo'])
    df_def_show = avg_locs_df[['name', 'def_acts_count', 'shirtNo', 'position']]

    MAX_MARKER_SIZE = 3000
    avg_locs_df['marker_size'] = (avg_locs_df['def_acts_count'] / avg_locs_df['def_acts_count'].max() * MAX_MARKER_SIZE)
    MIN_TRANSPARENCY = 0.05
    MAX_TRANSPARENCY = 0.85
    color = np.array(to_rgba(col))
    color = np.tile(color, (len(avg_locs_df), 1))
    c_transparency = avg_locs_df.def_acts_count / avg_locs_df.def_acts_count.max()
    c_transparency = (c_transparency * (MAX_TRANSPARENCY - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
    color[:, 3] = c_transparency

    pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, line_zorder=2, linewidth=2)
    pitch.draw(ax=ax)

    color = np.array(to_rgba(col))
    flamingo_cmap = LinearSegmentedColormap.from_list("Flamingo - 100 colors", ['#000000', col], N=250)
    pitch.kdeplot(total_def_acts.x, total_def_acts.y, ax=ax, fill=True, levels=2500, thresh=0.02, cut=4, cmap=flamingo_cmap)

    for index, row in avg_locs_df.iterrows():
        if row['isFirstEleven'] == True:
            pitch.scatter(row['x'], row['y'], s=row['marker_size'], marker='o', color=bg_color, edgecolor=line_color, linewidth=2, zorder=3, alpha=1, ax=ax)
        else:
            pitch.scatter(row['x'], row['y'], s=row['marker_size'], marker='s', color=bg_color, edgecolor=line_color, linewidth=2, zorder=3, alpha=0.75, ax=ax)

    for index, row in avg_locs_df.iterrows():
        player_initials = int(row["shirtNo"])
        pitch.annotate(player_initials, xy=(row.x, row.y), c=col, ha='center', va='center', size=12, zorder=4, ax=ax)

    avgph = round(avg_locs_df['x'].median(), 2)
    ax.axhline(y=avgph, color='gray', linestyle='--', alpha=0.75, linewidth=2)

    center_backs_height = avg_locs_df[avg_locs_df['position'] == 'DC']
    def_line_h = round(center_backs_height['x'].median(), 2)
    ax.axhline(y=def_line_h, color=violet, linestyle='dotted', alpha=1, linewidth=2)
    Forwards_height = avg_locs_df[avg_locs_df['isFirstEleven'] == 1]
    Forwards_height = Forwards_height.sort_values(by='x', ascending=False)
    Forwards_height = Forwards_height.head(2)
    fwd_line_h = round(Forwards_height['x'].mean(), 2)
    ax.axhline(y=fwd_line_h, color=violet, linestyle='dotted', alpha=1, linewidth=2)

    v_comp = round((1 - ((fwd_line_h - def_line_h) / 105)) * 100, 2)

    if phase_tag == 'Full Time':
        ax.text(34, 115, reshape_arabic_text('الوقت بالكامل: 0-90 دقيقة'), color='white', fontsize=14, ha='center', va='center', weight='bold')
        ax.text(34, 110, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), color='white', fontsize=12, ha='center', va='center')
    elif phase_tag == 'First Half':
        ax.text(34, 115, reshape_arabic_text('الشوط الأول: 0-45 دقيقة'), color='white', fontsize=14, ha='center', va='center', weight='bold')
        ax.text(34, 110, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), color='white', fontsize=12, ha='center', va='center')
    elif phase_tag == 'Second Half':
        ax.text(34, 115, reshape_arabic_text('الشوط الثاني: 45-90 دقيقة'), color='white', fontsize=14, ha='center', va='center', weight='bold')
        ax.text(34, 110, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), color='white', fontsize=12, ha='center', va='center')

    ax.text(34, -5, f"الأفعال الدفاعية\nالتماسك العمودي: {v_comp}%", color=violet, fontsize=12, ha='center', va='center')
    if team_name == hteamName:
        ax.text(-5, avgph, f'متوسط ارتفاع الأفعال الدفاعية: {avgph:.2f}م', color='gray', rotation=90, ha='left', va='center')
    if team_name == ateamName:
        ax.text(73, avgph, f'متوسط ارتفاع الأفعال الدفاعية: {avgph:.2f}م', color='gray', rotation=-90, ha='right', va='center')
    return df_def_show
                    
    if an_tp == 'Defensive Actions Heatmap':
        st.header(f'{an_tp}')
        
        dah_time_phase = st.pills(" ", ['Full Time', 'First Half', 'Second Half'], default='Full Time', key='dah_time_pill')
        
        if dah_time_phase == 'Full Time':
            fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
            home_df_def = def_acts_hm(axs[0], hteamName, hcol, 'Full Time')
            away_df_def = def_acts_hm(axs[1], ateamName, acol, 'Full Time')
        elif dah_time_phase == 'First Half':
            fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
            home_df_def = def_acts_hm(axs[0], hteamName, hcol, 'First Half')
            away_df_def = def_acts_hm(axs[1], ateamName, acol, 'First Half')
        elif dah_time_phase == 'Second Half':
            fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
            home_df_def = def_acts_hm(axs[0], hteamName, hcol, 'Second Half')
            away_df_def = def_acts_hm(axs[1], ateamName, acol, 'Second Half')

        fig_text(0.5, 1.05, f'<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>', highlight_textprops=[{'color': hcol}, {'color': acol}], fontsize=30, fontweight='bold', ha='center', va='center', ax=fig)
        fig.text(0.5, 1.01, 'الخريطة الحرارية للأفعال الدفاعية', fontsize=20, ha='center', va='center')
        fig.text(0.5, 0.97, '@REO_SHOW', fontsize=10, ha='center', va='center')

        fig.text(0.5, 0.05, '*الدوائر = اللاعبون الأساسيون، المربعات = اللاعبون البدلاء، الأرقام داخلها = أرقام القمصان', fontsize=10, fontstyle='italic', ha='center', va='center')
        fig.text(0.5, 0.03, '*حجم الدوائر/المربعات يمثل عدد الأفعال الدفاعية للاعبي الميدان', fontsize=10, fontstyle='italic', ha='center', va='center')

        himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png")
        himage = Image.open(himage)
        ax_himage = add_image(himage, fig, left=0.085, bottom=0.97, width=0.125, height=0.125)

        aimage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png")
        aimage = Image.open(aimage)
        ax_aimage = add_image(aimage, fig, left=0.815, bottom=0.97, width=0.125, height=0.125)

        st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            st.write(f'الأفعال الدفاعية للاعبي فريق {hteamName}:')
            st.dataframe(home_df_def, hide_index=True)
        with col2:
            st.write(f'الأفعال الدفاعية للاعبي فريق {ateamName}:')
            st.dataframe(away_df_def, hide_index=True)
            
        if an_tp == 'Progressive Passes':
            # st.header(f'{st.session_state.analysis_type}')
            st.header(f'{an_tp}')
            
            def progressive_pass(ax, team_name, col, phase_tag):
                if phase_tag == 'Full Time':
                    df_prop = df[(df['teamName']==team_name) & (df['outcomeType']=='Successful') & (df['prog_pass']>9.144) & (~df['qualifiers'].str.contains('Corner|Freekick')) & (df['x']>=35)]
                elif phase_tag == 'First Half':
                    df_fh = df[df['period'] == 'FirstHalf']
                    df_prop = df_fh[(df_fh['teamName']==team_name) & (df_fh['outcomeType']=='Successful') & (df_fh['prog_pass']>9.11) & (~df_fh['qualifiers'].str.contains('Corner|Freekick')) & (df_fh['x']>=35)]
                elif phase_tag == 'Second Half':
                    df_sh = df[df['period'] == 'SecondHalf']
                    df_prop = df_sh[(df_sh['teamName']==team_name) & (df_sh['outcomeType']=='Successful') & (df_sh['prog_pass']>9.11) & (~df_sh['qualifiers'].str.contains('Corner|Freekick')) & (df_sh['x']>=35)]
                
    pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, line_zorder=3, linewidth=2)
    pitch.draw(ax=ax)

            
    left_prop = df_prop[df_prop['y']>136/3]
    midd_prop = df_prop[(df_prop['y']<=136/3) & (df_prop['y']>=68/3)]
    rigt_prop = df_prop[df_prop['y']<68/3]
            
    if len(df_prop) != 0:
        name_counts = df_prop['shortName'].value_counts()
        name_counts_df = name_counts.reset_index()
        name_counts_df.columns = ['name', 'count']
        name_counts_df = name_counts_df.sort_values(by='count', ascending=False)  
        name_counts_df_show = name_counts_df.reset_index(drop=True)
        most_name = name_counts_df_show['name'][0]
        most_count = name_counts_df_show['count'][0]
    else:
                most_name = 'None'
                most_count = 0  
                
    if len(left_prop) != 0:
        name_counts = left_prop['shortName'].value_counts()
        name_counts_df = name_counts.reset_index()
        name_counts_df.columns = ['name', 'count']
        name_counts_df = name_counts_df.sort_values(by='count', ascending=False)  
        name_counts_df = name_counts_df.reset_index()
        l_name = name_counts_df['name'][0]
        l_count = name_counts_df['count'][0]
    else:
        l_name = 'None'
        l_count = 0   
            
    if len(midd_prop) != 0:
        name_counts = midd_prop['shortName'].value_counts()
        name_counts_df = name_counts.reset_index()
        name_counts_df.columns = ['name', 'count']
        name_counts_df = name_counts_df.sort_values(by='count', ascending=False)  
        name_counts_df = name_counts_df.reset_index()
        m_name = name_counts_df['name'][0]
        m_count = name_counts_df['count'][0]
    else:
            m_name = 'None'
            m_count = 0   
            
    if len(rigt_prop) != 0:
        name_counts = rigt_prop['shortName'].value_counts()
        name_counts_df = name_counts.reset_index()
        name_counts_df.columns = ['name', 'count']
        name_counts_df = name_counts_df.sort_values(by='count', ascending=False)  
        name_counts_df = name_counts_df.reset_index()
        r_name = name_counts_df['name'][0]
        r_count = name_counts_df['count'][0]
    else:
        r_name = 'None'
        r_count = 0   
            
    pitch.lines(df_prop.x, df_prop.y, df_prop.endX, df_prop.endY, comet=True, lw=4, color=col, ax=ax)
    pitch.scatter(df_prop.endX, df_prop.endY, s=75, zorder=3, color=bg_color, ec=col, lw=1.5, ax=ax)
            
    if phase_tag == 'Full Time':
        ax.text(34, 116, 'Full Time: 0-90 minutes', color=col, fontsize=13, ha='center', va='center')
    elif phase_tag == 'First Half':
        ax.text(34, 116, 'First Half: 0-45 minutes', color=col, fontsize=13, ha='center', va='center')
    elif phase_tag == 'Second Half':
        ax.text(34, 116, 'Second Half: 45-90 minutes', color=col, fontsize=13, ha='center', va='center')
        ax.text(34, 112, f'Open-Play Progressive Passes: {len(df_prop)}', color=col, fontsize=13, ha='center', va='center')
        ax.text(34, 108, f'Most by: {most_name}({most_count})', color=col, fontsize=13, ha='center', va='center')
            
        ax.text(10, 10, f'التقدم بالكرة من الجهة اليسرى\n{prop_left} تقدم ({prop_left_per}%)', color=col, fontsize=12, ha='center', va='center')
        ax.text(58, 10, f'التقدم بالكرة من الجهة اليمنى\n{prop_right} تقدم ({prop_right_per}%)', color=col, fontsize=12, ha='center', va='center')
            
        ax.text(340/6, -5, f'From Left: {len(left_prop)}', color=col, ha='center', va='center')
        ax.text(34, -5, f'From Mid: {len(midd_prop)}', color=col, ha='center', va='center')
        ax.text(68/6, -5, f'From Right: {len(rigt_prop)}', color=col, ha='center', va='center')
            
        ax.text(340/6, -7, f'Most by:\n{l_name}({l_count})', color=col, ha='center', va='top')
        ax.text(34, -7, f'Most by:\n{m_name}({m_count})', color=col, ha='center', va='top')
        ax.text(68/6, -7, f'Most by:\n{r_name}({r_count})', color=col, ha='center', va='top')
                 
    return name_counts_df_show
            
    pp_time_phase = st.pills(" ", ['Full Time', 'First Half', 'Second Half'], default='Full Time', key='pp_time_pill')
    if pp_time_phase == 'Full Time':
            fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
            shotmap(axs[0], hteamName, hcol, 'Full Time')
            shotmap(axs[1], ateamName, acol, 'Full Time')
        
    if pp_time_phase == 'First Half':
            fig, axs = plt.subplots(1,2, figsize=(15, 10), facecolor=bg_color)
            home_prop = progressive_pass(axs[0], hteamName, hcol, 'First Half')
            away_prop = progressive_pass(axs[1], ateamName, acol, 'First Half')
                
    if pp_time_phase == 'Second Half':
            fig, axs = plt.subplots(1,2, figsize=(15, 10), facecolor=bg_color)
            home_prop = progressive_pass(axs[0], hteamName, hcol, 'Second Half')
            away_prop = progressive_pass(axs[1], ateamName, acol, 'Second Half')
                
            fig_text(0.5, 1.05, f'<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>', highlight_textprops=[{'color':hcol}, {'color':acol}], fontsize=30, fontweight='bold', ha='center', va='center', ax=fig)
            fig.text(0.5, 1.01, 'Progressive Passes', fontsize=20, ha='center', va='center')
            fig.text(0.5, 0.97, '@REO_SHOW', fontsize=10, ha='center', va='center')
            
            fig.text(0.5, 0.02, '*Progressive Passes : Open-Play Successful Passes that move the ball at least 10 yards towards the Opponent Goal Center', fontsize=10, fontstyle='italic', ha='center', va='center')
            fig.text(0.5, 0.00, '*Excluding the passes started from Own Defensive Third of the Pitch', fontsize=10, fontstyle='italic', ha='center', va='center')
            
            himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png")
            himage = Image.open(himage)
            ax_himage = add_image(himage, fig, left=0.085, bottom=0.97, width=0.125, height=0.125)
            
            aimage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png")
            aimage = Image.open(aimage)
            ax_aimage = add_image(aimage, fig, left=0.815, bottom=0.97, width=0.125, height=0.125)
            
            st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f'{hteamName} Progressive Passers:')
                st.dataframe(home_prop, hide_index=True)
            with col2:
                st.write(f'{ateamName} Progressive Passers:')
                st.dataframe(away_prop, hide_index=True)
            
    if an_tp == 'Progressive Carries':
            # st.header(f'{st.session_state.analysis_type}')
            st.header(f'{an_tp}')
def progressive_carry(ax, team_name, col, phase_tag):
    if phase_tag == 'Full Time':
            df_proc = df[(df['teamName']==team_name) & (df['prog_carry']>9.144) & (df['endX']>=35)]
    elif phase_tag == 'First Half':
            df_fh = df[df['period'] == 'FirstHalf']
            df_proc = df_fh[(df_fh['teamName']==team_name) & (df_fh['prog_carry']>9.11) & (df_fh['endX']>=35)]
    elif phase_tag == 'Second Half':
            df_sh = df[df['period'] == 'SecondHalf']
            df_proc = df_sh[(df_sh['teamName']==team_name) & (df_sh['prog_carry']>9.11) & (df_sh['endX']>=35)]
                
    pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, line_zorder=3, linewidth=2)
    pitch.draw(ax=ax)
            
    left_proc = df_proc[df_proc['y']>136/3]
    midd_proc = df_proc[(df_proc['y']<=136/3) & (df_proc['y']>=68/3)]
    rigt_proc = df_proc[df_proc['y']<68/3]
            
    if len(df_proc) != 0:
            name_counts = df_proc['shortName'].value_counts()
            name_counts_df = name_counts.reset_index()
            name_counts_df.columns = ['name', 'count']
            name_counts_df = name_counts_df.sort_values(by='count', ascending=False)  
            name_counts_df_show = name_counts_df.reset_index(drop=True)
            most_name = name_counts_df_show['name'][0]
            most_count = name_counts_df_show['count'][0]
    else:
            most_name = 'None'
            most_count = 0  
                
    if len(left_proc) != 0:
                    name_counts = left_proc['shortName'].value_counts()
                    name_counts_df = name_counts.reset_index()
                    name_counts_df.columns = ['name', 'count']
                    name_counts_df = name_counts_df.sort_values(by='count', ascending=False)  
                    name_counts_df = name_counts_df.reset_index()
                    l_name = name_counts_df['name'][0]
                    l_count = name_counts_df['count'][0]
    else:
                    l_name = 'None'
                    l_count = 0   
            
    if len(midd_proc) != 0:
                    name_counts = midd_proc['shortName'].value_counts()
                    name_counts_df = name_counts.reset_index()
                    name_counts_df.columns = ['name', 'count']
                    name_counts_df = name_counts_df.sort_values(by='count', ascending=False)  
                    name_counts_df = name_counts_df.reset_index()
                    m_name = name_counts_df['name'][0]
                    m_count = name_counts_df['count'][0]
    else:
                    m_name = 'None'
                    m_count = 0   
            
    if len(rigt_proc) != 0:
                    name_counts = rigt_proc['shortName'].value_counts()
                    name_counts_df = name_counts.reset_index()
                    name_counts_df.columns = ['name', 'count']
                    name_counts_df = name_counts_df.sort_values(by='count', ascending=False)  
                    name_counts_df = name_counts_df.reset_index()
                    r_name = name_counts_df['name'][0]
                    r_count = name_counts_df['count'][0]
    else:
                    r_name = 'None'
                    r_count = 0   
            
    for index, row in df_proc.iterrows():
                    arrow = patches.FancyArrowPatch((row['y'], row['x']), (row['endY'], row['endX']), arrowstyle='->', color=col, zorder=4, mutation_scale=20, 
                                                    alpha=0.9, linewidth=2, linestyle='--')
                    ax.add_patch(arrow)
            
    if phase_tag == 'Full Time':
                    ax.text(34, 116, 'Full Time: 0-90 minutes', color=col, fontsize=13, ha='center', va='center')
    elif phase_tag == 'First Half':
                    ax.text(34, 116, 'First Half: 0-45 minutes', color=col, fontsize=13, ha='center', va='center')
    elif phase_tag == 'Second Half':
                    ax.text(34, 116, 'Second Half: 45-90 minutes', color=col, fontsize=13, ha='center', va='center')
                    ax.text(34, 112, f'Progressive Carries: {len(df_proc)}', color=col, fontsize=13, ha='center', va='center')
                    ax.text(34, 108, f'Most by: {most_name}({most_count})', color=col, fontsize=13, ha='center', va='center')
            
                    ax.vlines(136/3, ymin=0, ymax=105, color='gray', ls='dashed', lw=2)
                    ax.vlines(68/3, ymin=0, ymax=105, color='gray', ls='dashed', lw=2)
            
                    ax.text(340/6, -5, f'From Left: {len(left_proc)}', color=col, ha='center', va='center')
                    ax.text(34, -5, f'From Mid: {len(midd_proc)}', color=col, ha='center', va='center')
                    ax.text(68/6, -5, f'From Right: {len(rigt_proc)}', color=col, ha='center', va='center')
            
                    ax.text(340/6, -7, f'Most by:\n{l_name}({l_count})', color=col, ha='center', va='top')
                    ax.text(34, -7, f'Most by:\n{m_name}({m_count})', color=col, ha='center', va='top')
                    ax.text(68/6, -7, f'Most by:\n{r_name}({r_count})', color=col, ha='center', va='top')
                 
    return name_counts_df_show
            
    pc_time_phase = st.pills(" ", ['Full Time', 'First Half', 'Second Half'], default='Full Time', key='pc_time_pill')
    if pc_time_phase == 'Full Time':
        fig, axs = plt.subplots(1,2, figsize=(15, 10), facecolor=bg_color)
        home_proc = progressive_carry(axs[0], hteamName, hcol, 'Full Time')
        away_proc = progressive_carry(axs[1], ateamName, acol, 'Full Time')
                
    if pc_time_phase == 'First Half':
        fig, axs = plt.subplots(1,2, figsize=(15, 10), facecolor=bg_color)
        home_proc = progressive_carry(axs[0], hteamName, hcol, 'First Half')
        away_proc = progressive_carry(axs[1], ateamName, acol, 'First Half')
                
    if pc_time_phase == 'Second Half':
        fig, axs = plt.subplots(1,2, figsize=(15, 10), facecolor=bg_color)
        home_proc = progressive_carry(axs[0], hteamName, hcol, 'Second Half')
        away_proc = progressive_carry(axs[1], ateamName, acol, 'Second Half')
            
        fig_text(0.5, 1.05, f'<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>', highlight_textprops=[{'color':hcol}, {'color':acol}], fontsize=30, fontweight='bold', ha='center', va='center', ax=fig)
        fig.text(0.5, 1.01, 'Progressive Carries', fontsize=20, ha='center', va='center')
        fig.text(0.5, 0.97, '@REO_SHOW', fontsize=10, ha='center', va='center')
            
        fig.text(0.5, 0.02, '*Progressive Carry : Carries that move the ball at least 10 yards towards the Opponent Goal Center', fontsize=10, fontstyle='italic', ha='center', va='center')
        fig.text(0.5, 0.00, '*Excluding the carries ended at the Own Defensive Third of the Pitch', fontsize=10, fontstyle='italic', ha='center', va='center')
            
        himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png")
        himage = Image.open(himage)
        ax_himage = add_image(himage, fig, left=0.085, bottom=0.97, width=0.125, height=0.125)
            
        aimage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png")
        aimage = Image.open(aimage)
        ax_aimage = add_image(aimage, fig, left=0.815, bottom=0.97, width=0.125, height=0.125)
            
        st.pyplot(fig)
            
        col1, col2 = st.columns(2)
        with col1:
            st.write(f'{hteamName} Progressive Carriers:')
            st.dataframe(home_proc, hide_index=True)
        with col2:
            st.write(f'{ateamName} Progressive Carriers:')
            st.dataframe(away_proc, hide_index=True)
            
    if an_tp == 'Shotmap':
            # st.header(f'{st.session_state.analysis_type}')
            st.header(f'{an_tp}')
