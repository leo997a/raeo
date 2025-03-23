import json
import re
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, LinearSegmentedColormap
import seaborn as sns
import requests
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

# تهيئة matplotlib لدعم العربية
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Amiri', 'Noto Sans Arabic', 'Arial', 'Tahoma']
mpl.rcParams['axes.unicode_minus'] = False

# دالة لتحويل النص العربي
def reshape_arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# إضافة CSS محسّن لدعم RTL في Streamlit
st.markdown("""
    <style>
    @font-face {
        font-family: 'Noto Sans Arabic';
        src: url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;700&display=swap');
    }
    body, .stApp {
        font-family: 'Noto Sans Arabic', 'Amiri', 'DejaVu Sans', 'Arial', sans-serif !important;
    }
    h1, h2, h3, h4, h5, h6, p, div, span, label, button, input, select, option, table, th, td {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Noto Sans Arabic', 'Amiri', 'DejaVu Sans', 'Arial', sans-serif !important;
    }
    .stSelectbox, .stRadio, .stTabs, .stDataFrame, .stSidebar, .stButton {
        direction: rtl !important;
        text-align: right !important;
    }
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
st.sidebar.title('اختيار الألوان')
hcol = st.sidebar.color_picker('لون الفريق المضيف', default_hcol, key='hcol_picker')
acol = st.sidebar.color_picker('لون الفريق الضيف', default_acol, key='acol_picker')
bg_color = st.sidebar.color_picker('لون الخلفية', default_bg_color, key='bg_color_picker')
gradient_start = st.sidebar.color_picker('بداية التدرج', default_gradient_colors[0], key='gradient_start_picker')
gradient_end = st.sidebar.color_picker('نهاية التدرج', default_gradient_colors[1], key='gradient_end_picker')
gradient_colors = [gradient_start, gradient_end]
line_color = st.sidebar.color_picker('لون الخطوط', '#ffffff', key='line_color_picker')

# دالة لإعادة تعيين حالة التأكيد
def reset_confirmed():
    st.session_state['confirmed'] = False

# تعريف دالة get_event_data الكاملة
@st.cache_data
def get_event_data(season, league, stage, htn, atn):
    league_en = {
        'الدوري الإسباني': 'La_Liga',
        'الدوري الإنجليزي الممتاز': 'Premier_League',
        'الدوري الإيطالي': 'Serie_A',
        'دوري أبطال أوروبا': 'UEFA_Champions_League'
    }.get(league, league)
    stage_mapping = {
        'مرحلة الدوري': 'League Phase',
        'الملحق التأهيلي': 'Knockout Playoff',
        'دور الـ 16': 'Round of 16',
        'ربع النهائي': 'Quarter Final',
        'نصف النهائي': 'Semi Final',
        'النهائي': 'Final'
    }
    stage_en = stage_mapping.get(stage, stage) if stage else ''
    
    if league == 'دوري أبطال أوروبا':
        match_html_path = f"https://raw.githubusercontent.com/leo997a/{season}_{league_en}/refs/heads/main/{stage_en}/{htn}_vs_{atn}.html"
    else:
        match_html_path = f"https://raw.githubusercontent.com/leo997a/{season}_{league_en}/refs/heads/main/{htn}_vs_{atn}.html"
    match_html_path = match_html_path.replace(' ', '%20')

    def extract_json_from_html(html_path):
        response = requests.get(html_path)
        response.raise_for_status()
        html = response.text
        regex_pattern = r'(?<=require\.config\.params\["args"\].=.)[\s\S]*?;'
        data_txt = re.findall(regex_pattern, html)[0]
        data_txt = data_txt.replace('matchId', '"matchId"').replace('matchCentreData', '"matchCentreData"')
        data_txt = data_txt.replace('matchCentreEventTypeJson', '"matchCentreEventTypeJson"')
        data_txt = data_txt.replace('formationIdNameMappings', '"formationIdNameMappings"').replace('};', '}')
        return data_txt

    def extract_data_from_dict(data):
        events_dict = data["matchCentreData"]["events"]
        teams_dict = {data["matchCentreData"]['home']['teamId']: data["matchCentreData"]['home']['name'],
                      data["matchCentreData"]['away']['teamId']: data["matchCentreData"]['away']['name']}
        players_home_df = pd.DataFrame(data["matchCentreData"]['home']['players'])
        players_home_df["teamId"] = data["matchCentreData"]['home']['teamId']
        players_away_df = pd.DataFrame(data["matchCentreData"]['away']['players'])
        players_away_df["teamId"] = data["matchCentreData"]['away']['teamId']
        players_df = pd.concat([players_home_df, players_away_df])
        players_df['name'] = players_df['name'].apply(unidecode)
        return events_dict, players_df, teams_dict

    json_data_txt = extract_json_from_html(match_html_path)
    data = json.loads(json_data_txt)
    events_dict, players_df, teams_dict = extract_data_from_dict(data)
    df = pd.DataFrame(events_dict)

    df['type'] = df['type'].astype(str).str.extract(r"'displayName': '([^']+)")
    df['outcomeType'] = df['outcomeType'].astype(str).str.extract(r"'displayName': '([^']+)")
    df['period'] = df['period'].astype(str).str.extract(r"'displayName': '([^']+)")
    df['period'] = df['period'].replace({'FirstHalf': 1, 'SecondHalf': 2, 'FirstPeriodOfExtraTime': 3, 'SecondPeriodOfExtraTime': 4, 'PenaltyShootout': 5, 'PostGame': 14, 'PreMatch': 16})

    def cumulative_match_mins(events_df):
        events_out = pd.DataFrame()
        match_events = events_df.copy()
        match_events['cumulative_mins'] = match_events['minute'] + (1/60) * match_events['second']
        for period in np.arange(1, match_events['period'].max() + 1, 1):
            if period > 1:
                t_delta = match_events[match_events['period'] == period - 1]['cumulative_mins'].max() - \
                          match_events[match_events['period'] == period]['cumulative_mins'].min()
            else:
                t_delta = 0
            match_events.loc[match_events['period'] == period, 'cumulative_mins'] += t_delta
        events_out = pd.concat([events_out, match_events])
        return events_out

    df = cumulative_match_mins(df)

    def insert_ball_carries(events_df):
        events_out = pd.DataFrame()
        match_events = events_df.reset_index()
        match_events.loc[match_events['type'] == 'BallRecovery', 'endX'] = match_events.loc[match_events['type'] == 'BallRecovery', 'endX'].fillna(match_events['x'])
        match_events.loc[match_events['type'] == 'BallRecovery', 'endY'] = match_events.loc[match_events['type'] == 'BallRecovery', 'endY'].fillna(match_events['y'])
        match_carries = pd.DataFrame()
        
        for idx, match_event in match_events.iterrows():
            if idx < len(match_events) - 1:
                prev_evt_team = match_event['teamId']
                next_evt_idx = idx + 1
                init_next_evt = match_events.loc[next_evt_idx]
                take_ons = 0
                incorrect_next_evt = True
                while incorrect_next_evt and next_evt_idx < len(match_events):
                    next_evt = match_events.loc[next_evt_idx]
                    if next_evt['type'] == 'TakeOn' and next_evt['outcomeType'] == 'Successful':
                        take_ons += 1
                    elif ((next_evt['type'] == 'TakeOn' and next_evt['outcomeType'] == 'Unsuccessful') or
                          (next_evt['teamId'] != prev_evt_team and next_evt['type'] == 'Challenge' and next_evt['outcomeType'] == 'Unsuccessful') or
                          (next_evt['type'] in ['Foul', 'Card'])):
                        next_evt_idx += 1
                        continue
                    incorrect_next_evt = False
                
                same_team = prev_evt_team == next_evt['teamId']
                not_ball_touch = match_event['type'] != 'BallTouch'
                dx = 105 * (match_event['endX'] - next_evt['x']) / 100
                dy = 68 * (match_event['endY'] - next_evt['y']) / 100
                far_enough = dx ** 2 + dy ** 2 >= 3.0 ** 2
                not_too_far = dx ** 2 + dy ** 2 <= 100.0 ** 2
                dt = 60 * (next_evt['cumulative_mins'] - match_event['cumulative_mins'])
                min_time = dt >= 1.0
                same_phase = dt < 50.0
                same_period = match_event['period'] == next_evt['period']

                if same_team & not_ball_touch & far_enough & not_too_far & min_time & same_phase & same_period:
                    carry = pd.DataFrame([{
                        'eventId': match_event['eventId'] + 0.5,
                        'minute': np.floor(((init_next_evt['minute'] * 60 + init_next_evt['second']) + (match_event['minute'] * 60 + match_event['second'])) / (2 * 60)),
                        'second': (((init_next_evt['minute'] * 60 + init_next_evt['second']) + (match_event['minute'] * 60 + match_event['second'])) / 2) % 60,
                        'teamId': next_evt['teamId'],
                        'x': match_event['endX'],
                        'y': match_event['endY'],
                        'period': next_evt['period'],
                        'type': 'Carry',
                        'outcomeType': 'Successful',
                        'playerId': next_evt['playerId'],
                        'endX': next_evt['x'],
                        'endY': next_evt['y'],
                        'cumulative_mins': (match_event['cumulative_mins'] + init_next_evt['cumulative_mins']) / 2
                    }])
                    match_carries = pd.concat([match_carries, carry], ignore_index=True)

        match_events_and_carries = pd.concat([match_carries, match_events], ignore_index=True).sort_values(['period', 'cumulative_mins']).reset_index(drop=True)
        events_out = pd.concat([events_out, match_events_and_carries])
        return events_out

    df = insert_ball_carries(df)
    df['index'] = range(1, len(df) + 1)
    
    # Assign xT values
    dfxT = df.copy()
    dfxT['qualifiers'] = dfxT['qualifiers'].astype(str)
    dfxT = dfxT[(~dfxT['qualifiers'].str.contains('Corner')) & (dfxT['type'].isin(['Pass', 'Carry'])) & (dfxT['outcomeType'] == 'Successful')]
    xT = pd.read_csv("https://raw.githubusercontent.com/adnaaan433/Post-Match-Report-2.0/refs/heads/main/xT_Grid.csv", header=None)
    xT = np.array(xT)
    xT_rows, xT_cols = xT.shape

    dfxT['x1_bin_xT'] = pd.cut(dfxT['x'], bins=xT_cols, labels=False)
    dfxT['y1_bin_xT'] = pd.cut(dfxT['y'], bins=xT_rows, labels=False)
    dfxT['x2_bin_xT'] = pd.cut(dfxT['endX'], bins=xT_cols, labels=False)
    dfxT['y2_bin_xT'] = pd.cut(dfxT['endY'], bins=xT_rows, labels=False)
    dfxT['start_zone_value_xT'] = dfxT[['x1_bin_xT', 'y1_bin_xT']].apply(lambda x: xT[x[1]][x[0]] if pd.notna(x[0]) and pd.notna(x[1]) else 0, axis=1)
    dfxT['end_zone_value_xT'] = dfxT[['x2_bin_xT', 'y2_bin_xT']].apply(lambda x: xT[x[1]][x[0]] if pd.notna(x[0]) and pd.notna(x[1]) else 0, axis=1)
    dfxT['xT'] = dfxT['end_zone_value_xT'] - dfxT['start_zone_value_xT']
    df = df.merge(dfxT[['index', 'xT']], on='index', how='left')
    df['teamName'] = df['teamId'].map(teams_dict)

    # Reshape coordinates
    df['x'] = df['x'] * 1.05
    df['y'] = df['y'] * 0.68
    df['endX'] = df['endX'] * 1.05
    df['endY'] = df['endY'] * 0.68
    df['goalMouthY'] = df['goalMouthY'] * 0.68

    df = df.merge(players_df[['playerId', 'name', 'shirtNo', 'position', 'isFirstEleven']], on='playerId', how='left')
    df['prog_pass'] = np.where(df['type'] == 'Pass', 
                               np.sqrt((105 - df['x'])**2 + (34 - df['y'])**2) - np.sqrt((105 - df['endX'])**2 + (34 - df['endY'])**2), 0)
    df['prog_carry'] = np.where(df['type'] == 'Carry', 
                                np.sqrt((105 - df['x'])**2 + (34 - df['y'])**2) - np.sqrt((105 - df['endX'])**2 + (34 - df['endY'])**2), 0)
    print(df['name'].isnull().sum())  # عدد القيم الفارغة
    print(df['name'].apply(type).value_counts())  # أنواع البيانات في العمود
    df['shortName'] = df['name'].apply(lambda x: x.split()[0][0] + ". " + x.split()[-1] if pd.notna(x) and isinstance(x, str) and len(x.split()) > 1 else x)
    df['period'] = df['period'].replace({1: 'FirstHalf', 2: 'SecondHalf', 3: 'FirstPeriodOfExtraTime', 4: 'SecondPeriodOfExtraTime', 5: 'PenaltyShootout', 14: 'PostGame', 16: 'PreMatch'})
    df = df[df['period'] != 'PenaltyShootout'].reset_index(drop=True)
    return df, teams_dict, players_df

# واجهة اختيار المباراة
st.sidebar.title('اختيار المباراة')
if 'confirmed' not in st.session_state:
    st.session_state.confirmed = False

season = st.sidebar.selectbox('اختر الموسم:', ['2024_25'], key='season', index=0, on_change=reset_confirmed)
if season:
    league_options = ['الدوري الإسباني', 'الدوري الإنجليزي الممتاز', 'الدوري الإيطالي', 'دوري أبطال أوروبا']
    league = st.sidebar.selectbox('اختر الدوري:', league_options, key='league', index=None, on_change=reset_confirmed)

    if league == 'الدوري الإسباني':
        team_list = ['Athletic Club', 'Atletico Madrid', 'Barcelona', 'Celta Vigo', 'Deportivo Alaves', 'Espanyol', 'Getafe', 'Girona', 'Las Palmas', 'Leganes', 'Mallorca', 'Osasuna', 'Rayo Vallecano', 'Real Betis', 'Real Madrid', 'Real Sociedad', 'Real Valladolid', 'Sevilla', 'Valencia', 'Villarreal']
    elif league == 'الدوري الإنجليزي الممتاز':
        team_list = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich', 'Leicester', 'Liverpool', 'Manchester City', 'Manchester United', 'Newcastle', 'Nottingham Forest', 'Southampton', 'Tottenham', 'West Ham', 'Wolves']
    elif league == 'الدوري الإيطالي':
        team_list = ['AC Milan', 'Atalanta', 'Bologna', 'Cagliari', 'Como', 'Empoli', 'Fiorentina', 'Genoa', 'Inter', 'Juventus', 'Lazio', 'Lecce', 'Monza', 'Napoli', 'Parma Calcio', 'Roma', 'Torino', 'Udinese', 'Venezia', 'Verona']
    elif league == 'دوري أبطال أوروبا':
        team_list = ['AC Milan', 'Arsenal', 'Aston Villa', 'Atalanta', 'Atletico Madrid', 'BSC Young Boys', 'Barcelona', 'Bayer Leverkusen', 'Bayern Munich', 'Benfica', 'Bologna', 'Borussia Dortmund', 'Brest', 'Celtic', 'Club Brugge', 'Dinamo Zagreb', 'FK Crvena Zvezda', 'Feyenoord', 'Girona', 'Inter', 'Juventus', 'Lille', 'Liverpool', 'Manchester City', 'Monaco', 'PSV Eindhoven', 'Paris Saint-Germain', 'RB Leipzig', 'Real Madrid', 'Salzburg', 'Shakhtar Donetsk', 'Slovan Bratislava', 'Sparta Prague', 'Sporting CP', 'Sturm Graz', 'VfB Stuttgart']

    if league and league != 'دوري أبطال أوروبا':
        htn = st.sidebar.selectbox('اختر الفريق المضيف', team_list, key='home_team', index=None, on_change=reset_confirmed)
        if htn:
            atn_options = [team for team in team_list if team != htn]
            atn = st.sidebar.selectbox('اختر الفريق الضيف', atn_options, key='away_team', index=None, on_change=reset_confirmed)
    elif league == 'دوري أبطال أوروبا':
        stage_options = ['مرحلة الدوري', 'الملحق التأهيلي', 'دور الـ 16', 'ربع النهائي', 'نصف النهائي', 'النهائي']
        stage = st.sidebar.selectbox('اختر المرحلة', stage_options, key='stage_selection', index=None, on_change=reset_confirmed)
        if stage:
            htn = st.sidebar.selectbox('اختر الفريق المضيف', team_list, key='home_team', index=None, on_change=reset_confirmed)
            if htn:
                atn_options = [team for team in team_list if team != htn]
                atn = st.sidebar.selectbox('اختر الفريق الضيف', atn_options, key='away_team', index=None, on_change=reset_confirmed)

    if league and htn and atn:
        if league != 'دوري أبطال أوروبا':
            try:
                match_html_path = f"https://raw.githubusercontent.com/leo997a/{season}_{league.replace(' ', '_')}/refs/heads/main/{htn}_vs_{atn}.html".replace(' ', '%20')
                response = requests.get(match_html_path)
                response.raise_for_status()
                st.sidebar.button('تأكيد الاختيارات', on_click=lambda: st.session_state.update({'confirmed': True}))
            except:
                st.session_state['confirmed'] = False
                st.sidebar.write('لم يتم العثور على المباراة')
        else:
            stage_en = {'مرحلة الدوري': 'League Phase', 'الملحق التأهيلي': 'Knockout Playoff', 'دور الـ 16': 'Round of 16', 'ربع النهائي': 'Quarter Final', 'نصف النهائي': 'Semi Final', 'النهائي': 'Final'}[stage]
            try:
                match_html_path = f"https://raw.githubusercontent.com/leo997a/{season}_UEFA_Champions_League/refs/heads/main/{stage_en}/{htn}_vs_{atn}.html".replace(' ', '%20')
                response = requests.get(match_html_path)
                response.raise_for_status()
                st.sidebar.button('تأكيد الاختيارات', on_click=lambda: st.session_state.update({'confirmed': True}))
            except:
                st.session_state['confirmed'] = False
                st.sidebar.write('لم يتم العثور على المباراة')
if league and htn and atn and st.session_state.confirmed:
    df, teams_dict, players_df = get_event_data(season, league, stage, htn, atn)
    if df is not None:  # تحقق من أن البيانات تم جلبها بنجاح
        hteamID = list(teams_dict.keys())[0]
        ateamID = list(teams_dict.keys())[1]
        hteamName = teams_dict[hteamID]
        ateamName = teams_dict[ateamID]

        homedf = df[df['teamName'] == hteamName]
        awaydf = df[df['teamName'] == ateamName]
        hxT = homedf['xT'].sum().round(2)
        axT = awaydf['xT'].sum().round(2)

        hgoal_count = len(homedf[(homedf['type'] == 'Goal') & (~homedf['qualifiers'].str.contains('OwnGoal', na=False))]) + len(awaydf[(awaydf['type'] == 'Goal') & (awaydf['qualifiers'].str.contains('OwnGoal', na=False))])
        agoal_count = len(awaydf[(awaydf['type'] == 'Goal') & (~awaydf['qualifiers'].str.contains('OwnGoal', na=False))]) + len(homedf[(homedf['type'] == 'Goal') & (homedf['qualifiers'].str.contains('OwnGoal', na=False))])

        df_teamNameId = pd.read_csv('https://raw.githubusercontent.com/adnaaan433/pmr_app/refs/heads/main/teams_name_and_id.csv')
        hftmb_tid = df_teamNameId[df_teamNameId['teamName'] == hteamName].teamId.to_list()[0]
        aftmb_tid = df_teamNameId[df_teamNameId['teamName'] == ateamName].teamId.to_list()[0]

        st.header(f'{hteamName} {hgoal_count} - {agoal_count} {ateamName}')
        st.text(league)

        tab1, tab2 = st.tabs([reshape_arabic_text("تحليل المباراة"), reshape_arabic_text("تبويب آخر")])

        with tab1:
            an_tp = st.selectbox(reshape_arabic_text('نوع التحليل:'), [
                reshape_arabic_text('شبكة التمريرات'), 'Defensive Actions Heatmap', 'Progressive Passes', 
                'Progressive Carries', 'Shotmap', 'GK Saves', 'Match Momentum',
                reshape_arabic_text('Zone14 & Half-Space Passes'), reshape_arabic_text('Final Third Entries'),
                reshape_arabic_text('Box Entries'), reshape_arabic_text('High-Turnovers'),
                reshape_arabic_text('Chances Creating Zones'), reshape_arabic_text('Crosses'),
                reshape_arabic_text('Team Domination Zones'), reshape_arabic_text('Pass Target Zones')
            ], index=0, key='analysis_type_tab1')

            if an_tp == reshape_arabic_text('شبكة التمريرات'):
                st.header(reshape_arabic_text('شبكة التمريرات'))
                pn_time_phase = st.radio(" ", ['Full Time', 'First Half', 'Second Half'], index=0, key='pn_time_pill')
                # ... (باقي كود شبكة التمريرات كما هو)

            elif an_tp == 'Defensive Actions Heatmap':
                st.header(f'{an_tp}')
                dah_time_phase = st.radio(" ", ['Full Time', 'First Half', 'Second Half'], index=0, key='dah_time_pill')
                fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
                if dah_time_phase == 'Full Time':
                    home_df_def = def_acts_hm(axs[0], hteamName, hcol, 'Full Time')
                    away_df_def = def_acts_hm(axs[1], ateamName, acol, 'Full Time')
                elif dah_time_phase == 'First Half':
                    home_df_def = def_acts_hm(axs[0], hteamName, hcol, 'First Half')
                    away_df_def = def_acts_hm(axs[1], ateamName, acol, 'First Half')
                elif dah_time_phase == 'Second Half':
                    home_df_def = def_acts_hm(axs[0], hteamName, hcol, 'Second Half')
                    away_df_def = def_acts_hm(axs[1], ateamName, acol, 'Second Half')

                    # اختيار الأفعال الدفاعية للفريق المحدد
                    total_def_acts = df_def[df_def['teamName'] == team_name]
                    avg_locs_df = total_def_acts.groupby('name').agg({'x': 'median', 'y': ['median', 'count']}).reset_index()
                    avg_locs_df.columns = ['name', 'x', 'y', 'def_acts_count']
                    avg_locs_df = avg_locs_df.merge(players_df[['name', 'shirtNo', 'position', 'isFirstEleven']], on='name', how='left')
                    avg_locs_df = avg_locs_df[avg_locs_df['position'] != 'GK'].dropna(subset=['shirtNo'])
                    df_def_show = avg_locs_df[['name', 'def_acts_count', 'shirtNo', 'position']]

                    # تحديد حجم العلامات على الرسم
                    MAX_MARKER_SIZE = 3000
                    avg_locs_df['marker_size'] = (avg_locs_df['def_acts_count'] / avg_locs_df['def_acts_count'].max() * MAX_MARKER_SIZE)
                    color = np.array(to_rgba(col))
                    color = np.tile(color, (len(avg_locs_df), 1))
                    c_transparency = avg_locs_df.def_acts_count / avg_locs_df.def_acts_count.max()
                    c_transparency = (c_transparency * (0.85 - 0.05)) + 0.05
                    color[:, 3] = c_transparency

                    # رسم الملعب والخريطة الحرارية
                    pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, line_zorder=2, linewidth=2)
                    pitch.draw(ax=ax)
                    flamingo_cmap = LinearSegmentedColormap.from_list("Flamingo", ['#000000', col], N=250)
                    pitch.kdeplot(total_def_acts.x, total_def_acts.y, ax=ax, fill=True, levels=2500, thresh=0.02, cut=4, cmap=flamingo_cmap)

                    # إضافة العلامات للاعبين
                    for index, row in avg_locs_df.iterrows():
                        pitch.scatter(row['x'], row['y'], s=row['marker_size'], marker='o' if row['isFirstEleven'] else 's', color=bg_color, edgecolor=line_color, linewidth=2, zorder=3, alpha=1 if row['isFirstEleven'] else 0.75, ax=ax)
                        pitch.annotate(int(row["shirtNo"]), xy=(row.x, row.y), c=col, ha='center', va='center', size=12, zorder=4, ax=ax)

                    # حساب التماسك العمودي
                    avgph = round(avg_locs_df['x'].median(), 2)
                    ax.axhline(y=avgph, color='gray', linestyle='--', alpha=0.75, linewidth=2)
                    center_backs_height = avg_locs_df[avg_locs_df['position'] == 'DC']
                    def_line_h = round(center_backs_height['x'].median(), 2) if not center_backs_height.empty else avgph
                    Forwards_height = avg_locs_df[avg_locs_df['isFirstEleven'] == True].sort_values(by='x', ascending=False).head(2)
                    fwd_line_h = round(Forwards_height['x'].mean(), 2) if not Forwards_height.empty else avgph
                    v_comp = round((1 - ((fwd_line_h - def_line_h) / 105)) * 100, 2)

                    # إضافة النصوص حسب فترة المباراة
                    if phase_tag == 'Full Time':
                        ax.text(34, 112, reshape_arabic_text('الوقت الكامل: 0-90 دقيقة'), color=col, fontsize=15, ha='center', va='center')
                        ax.text(34, 108, reshape_arabic_text(f'إجمالي الأفعال الدفاعية: {len(total_def_acts)}'), color=col, fontsize=12, ha='center', va='center')
                    elif phase_tag == 'First Half':
                        ax.text(34, 112, reshape_arabic_text('الشوط الأول: 0-45 دقيقة'), color=col, fontsize=15, ha='center', va='center')
                        ax.text(34, 108, reshape_arabic_text(f'إجمالي الأفعال الدفاعية: {len(total_def_acts)}'), color=col, fontsize=12, ha='center', va='center')
                    elif phase_tag == 'Second Half':
                        ax.text(34, 112, reshape_arabic_text('الشوط الثاني: 45-90 دقيقة'), color=col, fontsize=15, ha='center', va='center')
                        ax.text(34, 108, reshape_arabic_text(f'إجمالي الأفعال الدفاعية: {len(total_def_acts)}'), color=col, fontsize=12, ha='center', va='center')
                    ax.text(34, -5, reshape_arabic_text(f"الأفعال الدفاعية\nالتماسك العمودي: {v_comp}%"), color=col, fontsize=12, ha='center', va='center')
                    
        return df_def_show  # إرجاع بيانات الأفعال الدفاعية

                # اختيار فترة المباراة باستخدام أداة الراديو
        dah_time_phase = st.radio(reshape_arabic_text("اختر فترة المباراة:"), 
                                          [reshape_arabic_text('الوقت الكامل'), reshape_arabic_text('الشوط الأول'), reshape_arabic_text('الشوط الثاني')], 
                                          index=0, key='dah_time_pill')
        fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
                
        # رسم الخرائط الحرارية للفريقين حسب الفترة المختارة
        if dah_time_phase == reshape_arabic_text('الوقت الكامل'):
            home_df_def = def_acts_hm(axs[0], hteamName, hcol, 'Full Time')
            away_df_def = def_acts_hm(axs[1], ateamName, acol, 'Full Time')
        elif dah_time_phase == reshape_arabic_text('الشوط الأول'):
            home_df_def = def_acts_hm(axs[0], hteamName, hcol, 'First Half')
            away_df_def = def_acts_hm(axs[1], ateamName, acol, 'First Half')
        elif dah_time_phase == reshape_arabic_text('الشوط الثاني'):
            home_df_def = def_acts_hm(axs[0], hteamName, hcol, 'Second Half')
            away_df_def = def_acts_hm(axs[1], ateamName, acol, 'Second Half')

                # إضافة العنوان والنصوص التوضيحية
        fig_text(0.5, 1.05, f'<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>', 
                    highlight_textprops=[{'color': hcol}, {'color': acol}], fontsize=30, fontweight='bold', ha='center', va='center', ax=fig)
        fig.text(0.5, 1.01, reshape_arabic_text('الخريطة الحرارية للأفعال الدفاعية'), fontsize=20, ha='center', va='center', color='white')
        fig.text(0.5, 0.97, '@REO_SHOW', fontsize=10, ha='center', va='center', color='white')
        fig.text(0.5, 0.05, reshape_arabic_text('*الدوائر = اللاعبون الأساسيون، المربعات = اللاعبون البدلاء، الأرقام داخلها = أرقام القمصان'), 
                    fontsize=10, fontstyle='italic', ha='center', va='center', color='white')
        fig.text(0.5, 0.03, reshape_arabic_text('*حجم الدوائر/المربعات يمثل عدد الأفعال الدفاعية'), 
                    fontsize=10, fontstyle='italic', ha='center', va='center', color='white')

                # إضافة شعارات الفريقين
        himage = Image.open(urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png"))
        add_image(himage, fig, left=0.085, bottom=0.97, width=0.125, height=0.125)
        aimage = Image.open(urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png"))
        add_image(aimage, fig, left=0.815, bottom=0.97, width=0.125, height=0.125)

                # عرض الرسم في Streamlit
                st.pyplot(fig)
                
                # عرض البيانات في عمودين
                col1, col2 = st.columns(2)
                with col1:
                    st.write(reshape_arabic_text(f'الأفعال الدفاعية لفريق {hteamName}:'))
                    if home_df_def is not None:
                        st.dataframe(home_df_def, hide_index=True)
                with col2:
                    st.write(reshape_arabic_text(f'الأفعال الدفاعية لفريق {ateamName}:'))
                    if away_df_def is not None:
                        st.dataframe(away_df_def, hide_index=True)

            elif an_tp == 'Progressive Passes':
                st.header(reshape_arabic_text('التمريرات التقدمية'))
                # أضف كود التمريرات التقدمية هنا إذا لزم الأمر

            elif an_tp == 'Progressive Carries':
                st.header(reshape_arabic_text('الحملات التقدمية'))
                # أضف كود الحملات التقدمية هنا (موجود بالفعل في الكود الخاص بك)

            elif an_tp == 'Shotmap':
                st.header(reshape_arabic_text('خريطة التسديدات'))
                # أضف كود خريطة التسديدات هنا إذا لزم الأمر
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
