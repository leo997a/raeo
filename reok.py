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
from highlight_text import fig_text
from matplotlib import rcParams
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import patheffects
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
import time

# تهيئة matplotlib لدعم العربية
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Amiri', 'Noto Sans Arabic', 'Arial', 'Tahoma']
mpl.rcParams['axes.unicode_minus'] = False

# دالة لتحويل النص العربي
def reshape_arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# إضافة CSS لدعم RTL في Streamlit
st.markdown("""
    <style>
    body {
        direction: rtl;
        text-align: right;
    }
    .stSelectbox > div > div > div {
        text-align: right;
    }
    .stRadio > div {
        flex-direction: row-reverse;
    }
    </style>
    """, unsafe_allow_html=True)

# تعريف القيم الافتراضية للألوان
default_hcol = '#d00000'
default_acol = '#003087'
default_bg_color = '#1e1e2f'
default_gradient_colors = ['#003087', '#d00000']
violet = '#800080'

# تهيئة الحالة
if 'confirmed' not in st.session_state:
    st.session_state.confirmed = False

def reset_confirmed():
    st.session_state['confirmed'] = False

# الشريط الجانبي
st.sidebar.title('إدخال رابط المباراة')
match_url = st.sidebar.text_input('أدخل رابط المباراة (من WhoScored):', placeholder='https://www.whoscored.com/Matches/...', key='match_url_input')

# إضافة أدوات اختيار الألوان في الشريط الجانبي
st.sidebar.title('اختيار الألوان')
hcol = st.sidebar.color_picker('لون الفريق المضيف', default_hcol, key='hcol_picker_unique')
acol = st.sidebar.color_picker('لون الفريق الضيف', default_acol, key='acol_picker_unique')
bg_color = st.sidebar.color_picker('لون الخلفية', default_bg_color, key='bg_color_picker_unique')
gradient_start = st.sidebar.color_picker('بداية التدرج', default_gradient_colors[0], key='gradient_start_picker_unique')
gradient_end = st.sidebar.color_picker('نهاية التدرج', default_gradient_colors[1], key='gradient_end_picker_unique')
gradient_colors = [gradient_start, gradient_end]
line_color = st.sidebar.color_picker('لون الخطوط', '#ffffff', key='line_color_picker_unique')

# التحقق من إدخال الرابط والضغط على زر التأكيد
if match_url:
    try:
        if not (match_url.startswith('https://www.whoscored.com/Matches/') or match_url.startswith('https://1xbet.whoscored.com/matches/')):
            raise ValueError("الرابط يجب أن يكون من موقع WhoScored")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(match_url, headers=headers, timeout=10)
        response.raise_for_status()
        match_input = st.sidebar.button('تأكيد الرابط', key='confirm_button_unique', on_click=lambda: st.session_state.update({'confirmed': True}))
    except requests.exceptions.HTTPError as e:
        st.session_state['confirmed'] = False
        st.sidebar.error(f'خطأ في الرابط: {str(e)} (رمز الحالة: {response.status_code})')
    except requests.exceptions.Timeout:
        st.session_state['confirmed'] = False
        st.sidebar.error('انتهت مهلة الطلب: تأكد من اتصالك بالإنترنت وحاول مرة أخرى')
    except requests.exceptions.RequestException as e:
        st.session_state['confirmed'] = False
        st.sidebar.error(f'الرابط غير صالح أو المباراة غير موجودة: {str(e)}')
else:
    st.session_state['confirmed'] = False

# تعريف الدوال المساعدة
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

def insert_ball_carries(events_df, min_carry_length=3, max_carry_length=100, min_carry_duration=1, max_carry_duration=50):
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
                    incorrect_next_evt = True
                elif ((next_evt['type'] == 'TakeOn' and next_evt['outcomeType'] == 'Unsuccessful')
                      or (next_evt['teamId'] != prev_evt_team and next_evt['type'] == 'Challenge' and next_evt['outcomeType'] == 'Unsuccessful')
                      or (next_evt['type'] == 'Foul')
                      or (next_evt['type'] == 'Card')):
                    incorrect_next_evt = True
                else:
                    incorrect_next_evt = False
                next_evt_idx += 1
    
            if next_evt_idx <= len(match_events):
                next_evt = match_events.loc[next_evt_idx - 1]
                same_team = prev_evt_team == next_evt['teamId']
                not_ball_touch = match_event['type'] != 'BallTouch'
                dx = 105*(match_event['endX'] - next_evt['x'])/100
                dy = 68*(match_event['endY'] - next_evt['y'])/100
                far_enough = dx ** 2 + dy ** 2 >= min_carry_length ** 2
                not_too_far = dx ** 2 + dy ** 2 <= max_carry_length ** 2
                dt = 60 * (next_evt['cumulative_mins'] - match_event['cumulative_mins'])
                min_time = dt >= min_carry_duration
                same_phase = dt < max_carry_duration
                same_period = match_event['period'] == next_evt['period']
    
                valid_carry = same_team & not_ball_touch & far_enough & not_too_far & min_time & same_phase & same_period
    
                if valid_carry:
                    carry = pd.DataFrame()
                    prev = match_event
                    nex = next_evt
    
                    carry.loc[0, 'eventId'] = prev['eventId'] + 0.5
                    carry['minute'] = np.floor(((init_next_evt['minute'] * 60 + init_next_evt['second']) + (
                            prev['minute'] * 60 + prev['second'])) / (2 * 60))
                    carry['second'] = (((init_next_evt['minute'] * 60 + init_next_evt['second']) +
                                        (prev['minute'] * 60 + prev['second'])) / 2) - (carry['minute'] * 60)
                    carry['teamId'] = nex['teamId']
                    carry['x'] = prev['endX']
                    carry['y'] = prev['endY']
                    carry['expandedMinute'] = np.floor(((init_next_evt['expandedMinute'] * 60 + init_next_evt['second']) +
                                                        (prev['expandedMinute'] * 60 + prev['second'])) / (2 * 60))
                    carry['period'] = nex['period']
                    carry['type'] = carry.apply(lambda x: {'value': 99, 'displayName': 'Carry'}, axis=1)
                    carry['outcomeType'] = 'Successful'
                    carry['qualifiers'] = carry.apply(lambda x: {'type': {'value': 999, 'displayName': 'takeOns'}, 'value': str(take_ons)}, axis=1)
                    carry['satisfiedEventsTypes'] = carry.apply(lambda x: [], axis=1)
                    carry['isTouch'] = True
                    carry['playerId'] = nex['playerId']
                    carry['endX'] = nex['x']
                    carry['endY'] = nex['y']
                    carry['blockedX'] = np.nan
                    carry['blockedY'] = np.nan
                    carry['goalMouthZ'] = np.nan
                    carry['goalMouthY'] = np.nan
                    carry['isShot'] = np.nan
                    carry['relatedEventId'] = nex['eventId']
                    carry['relatedPlayerId'] = np.nan
                    carry['isGoal'] = np.nan
                    carry['cardType'] = np.nan
                    carry['isOwnGoal'] = np.nan
                    carry['type'] = 'Carry'
                    carry['cumulative_mins'] = (prev['cumulative_mins'] + init_next_evt['cumulative_mins']) / 2
    
                    match_carries = pd.concat([match_carries, carry], ignore_index=True, sort=False)
    
    match_events_and_carries = pd.concat([match_carries, match_events], ignore_index=True, sort=False)
    match_events_and_carries = match_events_and_carries.sort_values(['period', 'cumulative_mins']).reset_index(drop=True)
    events_out = pd.concat([events_out, match_events_and_carries])
    return events_out

def get_possession_chains(events_df, chain_check, suc_evts_in_chain):
    events_out = pd.DataFrame()
    match_events_df = events_df.reset_index()
    match_pos_events_df = match_events_df[~match_events_df['type'].isin(['OffsideGiven', 'CornerAwarded','Start', 'Card', 'SubstitutionOff',
                                                                          'SubstitutionOn', 'FormationChange','FormationSet', 'End'])].copy()
    match_pos_events_df['outcomeBinary'] = (match_pos_events_df['outcomeType']
                                                .apply(lambda x: 1 if x == 'Successful' else 0))
    match_pos_events_df['teamBinary'] = (match_pos_events_df['teamName']
                         .apply(lambda x: 1 if x == min(match_pos_events_df['teamName']) else 0))
    match_pos_events_df['goalBinary'] = ((match_pos_events_df['type'] == 'Goal')
                         .astype(int).diff(periods=1).apply(lambda x: 1 if x < 0 else 0))
    pos_chain_df = pd.DataFrame()
    for n in np.arange(1, chain_check):
        pos_chain_df[f'evt_{n}_same_team'] = abs(match_pos_events_df['teamBinary'].diff(periods=-n))
        pos_chain_df[f'evt_{n}_same_team'] = pos_chain_df[f'evt_{n}_same_team'].apply(lambda x: 1 if x > 1 else x)
    pos_chain_df['enough_evt_same_team'] = pos_chain_df.sum(axis=1).apply(lambda x: 1 if x < chain_check - suc_evts_in_chain else 0)
    pos_chain_df['enough_evt_same_team'] = pos_chain_df['enough_evt_same_team'].diff(periods=1)
    pos_chain_df[pos_chain_df['enough_evt_same_team'] < 0] = 0
    match_pos_events_df['period'] = pd.to_numeric(match_pos_events_df['period'], errors='coerce')
    pos_chain_df['upcoming_ko'] = 0
    for ko in match_pos_events_df[(match_pos_events_df['goalBinary'] == 1) | (match_pos_events_df['period'].diff(periods=1))].index.values:
        ko_pos = match_pos_events_df.index.to_list().index(ko)
        pos_chain_df.iloc[ko_pos - suc_evts_in_chain:ko_pos, 5] = 1
    pos_chain_df['valid_pos_start'] = (pos_chain_df.fillna(0)['enough_evt_same_team'] - pos_chain_df.fillna(0)['upcoming_ko'])
    pos_chain_df['kick_off_period_change'] = match_pos_events_df['period'].diff(periods=1)
    pos_chain_df['kick_off_goal'] = ((match_pos_events_df['type'] == 'Goal')
                     .astype(int).diff(periods=1).apply(lambda x: 1 if x < 0 else 0))
    pos_chain_df.loc[pos_chain_df['kick_off_period_change'] == 1, 'valid_pos_start'] = 1
    pos_chain_df.loc[pos_chain_df['kick_off_goal'] == 1, 'valid_pos_start'] = 1
    pos_chain_df['teamName'] = match_pos_events_df['teamName']
    pos_chain_df.loc[pos_chain_df.head(1).index, 'valid_pos_start'] = 1
    pos_chain_df.loc[pos_chain_df.head(1).index, 'possession_id'] = 1
    pos_chain_df.loc[pos_chain_df.head(1).index, 'possession_team'] = pos_chain_df.loc[pos_chain_df.head(1).index, 'teamName']
    valid_pos_start_id = pos_chain_df[pos_chain_df['valid_pos_start'] > 0].index
    possession_id = 2
    for idx in np.arange(1, len(valid_pos_start_id)):
        current_team = pos_chain_df.loc[valid_pos_start_id[idx], 'teamName']
        previous_team = pos_chain_df.loc[valid_pos_start_id[idx - 1], 'teamName']
        if ((previous_team == current_team) & (pos_chain_df.loc[valid_pos_start_id[idx], 'kick_off_goal'] != 1) &
                (pos_chain_df.loc[valid_pos_start_id[idx], 'kick_off_period_change'] != 1)):
            pos_chain_df.loc[valid_pos_start_id[idx], 'possession_id'] = np.nan
        else:
            pos_chain_df.loc[valid_pos_start_id[idx], 'possession_id'] = possession_id
            pos_chain_df.loc[valid_pos_start_id[idx], 'possession_team'] = pos_chain_df.loc[valid_pos_start_id[idx], 'teamName']
            possession_id += 1
    match_events_df = pd.merge(match_events_df, pos_chain_df[['possession_id', 'possession_team']], how='left', left_index=True, right_index=True)
    match_events_df[['possession_id', 'possession_team']] = (match_events_df[['possession_id', 'possession_team']].fillna(method='ffill'))
    match_events_df[['possession_id', 'possession_team']] = (match_events_df[['possession_id', 'possession_team']].fillna(method='bfill'))
    events_out = pd.concat([events_out, match_events_df])
    return events_out

# دالة get_event_data
@st.cache_data
def get_event_data(match_url):
    def extract_json_from_html(html_path):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(html_path, headers=headers, timeout=10)
            response.raise_for_status()
            html = response.text
            regex_pattern = r'(?<=require\.config\.params\["args"\].=.)[\s\S]*?;'
            data_txt = re.findall(regex_pattern, html)
            if not data_txt:
                raise ValueError("لم يتم العثور على بيانات JSON في الصفحة")
            data_txt = data_txt[0]
            data_txt = data_txt.replace('matchId', '"matchId"')
            data_txt = data_txt.replace('matchCentreData', '"matchCentreData"')
            data_txt = data_txt.replace('matchCentreEventTypeJson', '"matchCentreEventTypeJson"')
            data_txt = data_txt.replace('formationIdNameMappings', '"formationIdNameMappings"')
            data_txt = data_txt.replace('};', '}')
            return data_txt
        except Exception as e:
            st.error(f"خطأ أثناء استخراج البيانات: {str(e)}")
            raise

    def extract_data_from_dict(data):
        event_types_json = data["matchCentreEventTypeJson"]
        formation_mappings = data["formationIdNameMappings"]
        events_dict = data["matchCentreData"]["events"]
        teams_dict = {
            data["matchCentreData"]['home']['teamId']: data["matchCentreData"]['home']['name'],
            data["matchCentreData"]['away']['teamId']: data["matchCentreData"]['away']['name']
        }
        players_dict = data["matchCentreData"]["playerIdNameDictionary"]
        players_home_df = pd.DataFrame(data["matchCentreData"]['home']['players'])
        players_home_df["teamId"] = data["matchCentreData"]['home']['teamId']
        players_away_df = pd.DataFrame(data["matchCentreData"]['away']['players'])
        players_away_df["teamId"] = data["matchCentreData"]['away']['teamId']
        players_df = pd.concat([players_home_df, players_away_df])
        players_df['name'] = players_df['name'].astype(str)
        players_df['name'] = players_df['name'].apply(unidecode)
        return events_dict, players_df, teams_dict

    json_data_txt = extract_json_from_html(match_url)
    data = json.loads(json_data_txt)
    events_dict, players_df, teams_dict = extract_data_from_dict(data)
    df = pd.DataFrame(events_dict)
    dfp = pd.DataFrame(players_df)

    # معالجة البيانات
    df['type'] = df['type'].astype(str)
    df['outcomeType'] = df['outcomeType'].astype(str)
    df['period'] = df['period'].astype(str)
    df['type'] = df['type'].str.extract(r"'displayName': '([^']+)")
    df['outcomeType'] = df['outcomeType'].str.extract(r"'displayName': '([^']+)")
    df['period'] = df['period'].str.extract(r"'displayName': '([^']+)")
    df['period'] = df['period'].replace({
        'FirstHalf': 1, 'SecondHalf': 2, 'FirstPeriodOfExtraTime': 3,
        'SecondPeriodOfExtraTime': 4, 'PenaltyShootout': 5, 'PostGame': 14, 'PreMatch': 16
    })

    df = cumulative_match_mins(df)
    df = insert_ball_carries(df, min_carry_length=3, max_carry_length=100, min_carry_duration=1, max_carry_duration=50)
    df = df.reset_index(drop=True)
    df['index'] = range(1, len(df) + 1)
    df = df[['index'] + [col for col in df.columns if col != 'index']]

    # معالجة xT
    dfxT = df.copy()
    dfxT['qualifiers'] = dfxT['qualifiers'].astype(str)
    dfxT = dfxT[(~dfxT['qualifiers'].str.contains('Corner'))]
    dfxT = dfxT[(dfxT['type'].isin(['Pass', 'Carry'])) & (dfxT['outcomeType'] == 'Successful')]
    xT = pd.read_csv("https://raw.githubusercontent.com/adnaaan433/Post-Match-Report-2.0/refs/heads/main/xT_Grid.csv", header=None)
    xT = np.array(xT)
    xT_rows, xT_cols = xT.shape
    dfxT['x1_bin_xT'] = pd.cut(dfxT['x'], bins=xT_cols, labels=False)
    dfxT['y1_bin_xT'] = pd.cut(dfxT['y'], bins=xT_rows, labels=False)
    dfxT['x2_bin_xT'] = pd.cut(dfxT['endX'], bins=xT_cols, labels=False)
    dfxT['y2_bin_xT'] = pd.cut(dfxT['endY'], bins=xT_rows, labels=False)
    dfxT['start_zone_value_xT'] = dfxT[['x1_bin_xT', 'y1_bin_xT']].apply(lambda x: xT[x[1]][x[0]] if pd.notnull(x[0]) and pd.notnull(x[1]) else 0, axis=1)
    dfxT['end_zone_value_xT'] = dfxT[['x2_bin_xT', 'y2_bin_xT']].apply(lambda x: xT[x[1]][x[0]] if pd.notnull(x[0]) and pd.notnull(x[1]) else 0, axis=1)
    dfxT['xT'] = dfxT['end_zone_value_xT'] - dfxT['start_zone_value_xT']
    columns_to_drop = ['id', 'eventId', 'minute', 'second', 'teamId', 'x', 'y', 'expandedMinute', 'period',
                       'outcomeType', 'qualifiers', 'type', 'satisfiedEventsTypes', 'isTouch', 'playerId', 'endX',
                       'endY', 'relatedEventId', 'relatedPlayerId', 'blockedX', 'blockedY', 'goalMouthZ', 'goalMouthY', 'isShot', 'cumulative_mins']
    dfxT.drop(columns=[col for col in columns_to_drop if col in dfxT.columns], inplace=True)
    df = df.merge(dfxT, on='index', how='left')
    df['teamName'] = df['teamId'].map(teams_dict)
    team_names = list(teams_dict.values())
    opposition_dict = {team_names[i]: team_names[1-i] for i in range(len(team_names))}
    df['oppositionTeamName'] = df['teamName'].map(opposition_dict)

    # تحويل الإحداثيات
    df['x'] = df['x'] * 1.05
    df['y'] = df['y'] * 0.68
    df['endX'] = df['endX'] * 1.05
    df['endY'] = df['endY'] * 0.68
    df['goalMouthY'] = df['goalMouthY'] * 0.68

    columns_to_drop = ['height', 'weight', 'age', 'isManOfTheMatch', 'field', 'stats', 'subbedInPlayerId',
                       'subbedOutPeriod', 'subbedOutExpandedMinute', 'subbedInPeriod', 'subbedInExpandedMinute',
                       'subbedOutPlayerId', 'teamId']
    dfp.drop(columns=[col for col in columns_to_drop if col in dfp.columns], inplace=True)
    df = df.merge(dfp, on='playerId', how='left')

    df['qualifiers'] = df['qualifiers'].astype(str)
    df['prog_pass'] = np.where((df['type'] == 'Pass'),
                               np.sqrt((105 - df['x'])**2 + (34 - df['y'])**2) - np.sqrt((105 - df['endX'])**2 + (34 - df['endY'])**2), 0)
    df['prog_carry'] = np.where((df['type'] == 'Carry'),
                                np.sqrt((105 - df['x'])**2 + (34 - df['y'])**2) - np.sqrt((105 - df['endX'])**2 + (34 - df['endY'])**2), 0)
    df['pass_or_carry_angle'] = np.degrees(np.arctan2(df['endY'] - df['y'], df['endX'] - df['x']))
    df['name'] = df['name'].astype(str)
    df['name'] = df['name'].apply(unidecode)

    def get_short_name(full_name):
        if pd.isna(full_name):
            return full_name
        parts = full_name.split()
        if len(parts) == 1:
            return full_name
        elif len(parts) == 2:
            return parts[0][0] + ". " + parts[1]
        else:
            return parts[0][0] + ". " + parts[1][0] + ". " + " ".join(parts[2:])
    df['shortName'] = df['name'].apply(get_short_name)
    df['qualifiers'] = df['qualifiers'].astype(str)
    columns_to_drop2 = ['id']
    df.drop(columns=[col for col in columns_to_drop2 if col in df.columns], inplace=True)

    df = get_possession_chains(df, 5, 3)
    df['period'] = df['period'].replace({
        1: 'FirstHalf', 2: 'SecondHalf', 3: 'FirstPeriodOfExtraTime',
        4: 'SecondPeriodOfExtraTime', 5: 'PenaltyShootout', 14: 'PostGame', 16: 'PreMatch'
    })
    df = df[df['period'] != 'PenaltyShootout']
    df = df.reset_index(drop=True)
    return df, teams_dict, players_df

# دالة pass_network
def pass_network(ax, team_name, col, phase_tag):
    global df, players_df
    if phase_tag == 'Full Time':
        df_pass = df.copy()
        df_pass = df_pass.reset_index(drop=True)
    elif phase_tag == 'First Half':
        df_pass = df[df['period'] == 'FirstHalf']
        df_pass = df_pass.reset_index(drop=True)
    elif phase_tag == 'Second Half':
        df_pass = df[df['period'] == 'SecondHalf']
        df_pass = df_pass.reset_index(drop=True)

    if df_pass.empty:
        ax.text(0.5, 0.5, reshape_arabic_text('لا توجد بيانات متاحة'), ha='center', va='center', color='white')
        return None

    total_pass = df_pass[(df_pass['teamName'] == team_name) & (df_pass['type'] == 'Pass')]
    accrt_pass = df_pass[(df_pass['teamName'] == team_name) & (df_pass['type'] == 'Pass') & (df_pass['outcomeType'] == 'Successful')]
    accuracy = round((len(accrt_pass) / len(total_pass)) * 100, 2) if len(total_pass) > 0 else 0

    df_pass['pass_receiver'] = df_pass.loc[(df_pass['type'] == 'Pass') & (df_pass['outcomeType'] == 'Successful') & (df_pass['teamName'].shift(-1) == team_name), 'name'].shift(-1)
    df_pass['pass_receiver'] = df_pass['pass_receiver'].fillna('No')

    off_acts_df = df_pass[(df_pass['teamName'] == team_name) & (df_pass['type'].isin(['Pass', 'Goal', 'MissedShots', 'SavedShot', 'ShotOnPost', 'TakeOn', 'BallTouch', 'KeeperPickup']))]
    off_acts_df = off_acts_df[['name', 'x', 'y']].reset_index(drop=True)
    if off_acts_df.empty:
        ax.text(0.5, 0.5, reshape_arabic_text('لا توجد أحداث هجومية متاحة'), ha='center', va='center', color='white')
        return None

    avg_locs_df = off_acts_df.groupby('name').agg(avg_x=('x', 'median'), avg_y=('y', 'median')).reset_index()
    if avg_locs_df.empty:
        ax.text(0.5, 0.5, reshape_arabic_text('لا يمكن حساب المواقع المتوسطة'), ha='center', va='center', color='white')
        return None

    team_pdf = players_df[['name', 'shirtNo', 'position', 'isFirstEleven']]
    avg_locs_df = avg_locs_df.merge(team_pdf, on='name', how='left')

    df_pass = df_pass[(df_pass['type'] == 'Pass') & (df_pass['outcomeType'] == 'Successful') & (df_pass['teamName'] == team_name) & (~df_pass['qualifiers'].str.contains('Corner|Freekick'))]
    df_pass = df_pass[['type', 'name', 'pass_receiver']].reset_index(drop=True)

    pass_count_df = df_pass.groupby(['name', 'pass_receiver']).size().reset_index(name='pass_count').sort_values(by='pass_count', ascending=False)
    pass_count_df = pass_count_df.reset_index(drop=True)
    if pass_count_df.empty:
        ax.text(0.5, 0.5, reshape_arabic_text('لا توجد تمريرات ناجحة متاحة'), ha='center', va='center', color='white')
        return None

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
        if pd.notnull(row['isFirstEleven']) and row['isFirstEleven']:
            pitch.scatter(row['avg_x'], row['avg_y'], s=800, marker='o', color=col, edgecolor=line_color, linewidth=1.5, alpha=0.9, ax=ax)
        else:
            pitch.scatter(row['avg_x'], row['avg_y'], s=800, marker='s', color=col, edgecolor=line_color, linewidth=1.5, alpha=0.7, ax=ax)

    for index, row in avg_locs_df.iterrows():
        if pd.notnull(row['shirtNo']):
            player_initials = int(row["shirtNo"])
            pitch.annotate(player_initials, xy=(row['avg_x'], row['avg_y']), c='white', ha='center', va='center', size=14, weight='bold', ax=ax)

    avgph = round(avg_locs_df['avg_x'].median(), 2) if not avg_locs_df['avg_x'].empty else 0
    ax.axhline(y=avgph, color='white', linestyle='--', alpha=0.5, linewidth=1.5)

    center_backs_height = avg_locs_df[avg_locs_df['position'] == 'DC']
    def_line_h = round(center_backs_height['avg_x'].median(), 2) if not center_backs_height.empty else 0
    Forwards_height = avg_locs_df[avg_locs_df['isFirstEleven'] == True].sort_values(by='avg_x', ascending=False).head(2)
    fwd_line_h = round(Forwards_height['avg_x'].mean(), 2) if not Forwards_height.empty else 0
    ymid = [0, 0, 68, 68]
    xmid = [def_line_h, fwd_line_h, fwd_line_h, def_line_h]
    ax.fill(ymid, xmid, col, alpha=0.2)

    v_comp = round((1 - ((fwd_line_h - def_line_h) / 105)) * 100, 2) if def_line_h and fwd_line_h else 0

    if phase_tag == 'Full Time':
        ax.text(34, 115, reshape_arabic_text('الوقت بالكامل: 0-90 دقيقة'), color='white', fontsize=14, ha='center', va='center', weight='bold')
        ax.text(34, 112, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), color='white', fontsize=12, ha='center', va='center')
    elif phase_tag == 'First Half':
        ax.text(34, 115, reshape_arabic_text('الشوط الأول: 0-45 دقيقة'), color='white', fontsize=14, ha='center', va='center', weight='bold')
        ax.text(34, 112, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), color='white', fontsize=12, ha='center', va='center')
    elif phase_tag == 'Second Half':
        ax.text(34, 115, reshape_arabic_text('الشوط الثاني: 45-90 دقيقة'), color='white', fontsize=14, ha='center', va='center', weight='bold')
        ax.text(34, 112, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), color='white', fontsize=12, ha='center', va='center')

    ax.text(34, -6, reshape_arabic_text(f"على الكرة\nالتماسك العمودي (المنطقة المظللة): {v_comp}%"), color='white', fontsize=12, ha='center', va='center', weight='bold')

    return pass_btn
# دالة def_acts_hm
def def_acts_hm(ax, team_name, col, phase_tag):
    global df, players_df
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

    avg_locs_df = total_def_acts.groupby('name').agg({'x': 'median', 'y': ['median', 'count']}).reset_index()
    avg_locs_df.columns = ['name', 'x', 'y', 'def_acts_count']
    avg_locs_df = avg_locs_df.sort_values(by='def_acts_count', ascending=False)
    team_pdf = players_df[['name', 'shirtNo', 'position', 'isFirstEleven']]
    avg_locs_df = avg_locs_df.merge(team_pdf, on='name', how='left')
    avg_locs_df = avg_locs_df[avg_locs_df['position'] != 'GK']
    avg_locs_df = avg_locs_df.dropna(subset=['shirtNo'])
    df_def_show = avg_locs_df[['name', 'def_acts_count', 'shirtNo', 'position']]

    MAX_MARKER_SIZE = 3000
    avg_locs_df['marker_size'] = (avg_locs_df['def_acts_count'] / avg_locs_df['def_acts_count'].max() * MAX_MARKER_SIZE) if avg_locs_df['def_acts_count'].max() > 0 else MAX_MARKER_SIZE
    MIN_TRANSPARENCY = 0.05
    MAX_TRANSPARENCY = 0.85
    color = np.array(to_rgba(col))
    color = np.tile(color, (len(avg_locs_df), 1))
    c_transparency = avg_locs_df.def_acts_count / avg_locs_df.def_acts_count.max() if avg_locs_df.def_acts_count.max() > 0 else 1
    c_transparency = (c_transparency * (MAX_TRANSPARENCY - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
    color[:, 3] = c_transparency

    pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, line_zorder=2, linewidth=2)
    pitch.draw(ax=ax)

    flamingo_cmap = LinearSegmentedColormap.from_list("Flamingo - 100 colors", ['#000000', col], N=250)
    if not total_def_acts.empty:
        pitch.kdeplot(total_def_acts.x, total_def_acts.y, ax=ax, fill=True, levels=2500, thresh=0.02, cut=4, cmap=flamingo_cmap)

    for index, row in avg_locs_df.iterrows():
        if row['isFirstEleven'] == True:
            pitch.scatter(row['x'], row['y'], s=row['marker_size'], marker='o', color=bg_color, edgecolor=line_color, linewidth=2, zorder=3, alpha=1, ax=ax)
        else:
            pitch.scatter(row['x'], row['y'], s=row['marker_size'], marker='s', color=bg_color, edgecolor=line_color, linewidth=2, zorder=3, alpha=0.75, ax=ax)

    for index, row in avg_locs_df.iterrows():
        if pd.notnull(row['shirtNo']):
            player_initials = int(row["shirtNo"])
            pitch.annotate(player_initials, xy=(row['x'], row['y']), c=col, ha='center', va='center', size=12, zorder=4, ax=ax)

    avgph = round(avg_locs_df['x'].median(), 2) if not avg_locs_df['x'].empty else 0
    ax.axhline(y=avgph, color='gray', linestyle='--', alpha=0.75, linewidth=2)

    center_backs_height = avg_locs_df[avg_locs_df['position'] == 'DC']
    def_line_h = round(center_backs_height['x'].median(), 2) if not center_backs_height.empty else 0
    Forwards_height = avg_locs_df[avg_locs_df['isFirstEleven'] == True].sort_values(by='x', ascending=False).head(2)
    fwd_line_h = round(Forwards_height['x'].mean(), 2) if not Forwards_height.empty else 0

    ax.axhline(y=def_line_h, color=violet, linestyle='dotted', alpha=1, linewidth=2)
    ax.axhline(y=fwd_line_h, color=violet, linestyle='dotted', alpha=1, linewidth=2)

    v_comp = round((1 - ((fwd_line_h - def_line_h) / 105)) * 100, 2) if def_line_h and fwd_line_h else 0

    if phase_tag == 'Full Time':
        ax.text(34, 112, reshape_arabic_text('الوقت الكامل: 0-90 دقيقة'), color=col, fontsize=15, ha='center', va='center')
        ax.text(34, 108, reshape_arabic_text(f'إجمالي الأفعال الدفاعية: {len(total_def_acts)}'), color=col, fontsize=15, ha='center', va='center')
    elif phase_tag == 'First Half':
        ax.text(34, 112, reshape_arabic_text('الشوط الأول: 0-45 دقيقة'), color=col, fontsize=15, ha='center', va='center')
        ax.text(34, 108, reshape_arabic_text(f'إجمالي الأفعال الدفاعية: {len(total_def_acts)}'), color=col, fontsize=15, ha='center', va='center')
    elif phase_tag == 'Second Half':
        ax.text(34, 112, reshape_arabic_text('الشوط الثاني: 45-90 دقيقة'), color=col, fontsize=15, ha='center', va='center')
        ax.text(34, 108, reshape_arabic_text(f'إجمالي الأفعال الدفاعية: {len(total_def_acts)}'), color=col, fontsize=15, ha='center', va='center')

    ax.text(34, -5, reshape_arabic_text(f"الأفعال الدفاعية\nالتماسك العمودي: {v_comp}%"), color=violet, fontsize=12, ha='center', va='center')
    if team_name == hteamName:
        ax.text(-5, avgph, reshape_arabic_text(f'متوسط ارتفاع الأفعال الدفاعية: {avgph:.2f}م'), color='gray', rotation=90, ha='left', va='center')
    if team_name == ateamName:
        ax.text(73, avgph, reshape_arabic_text(f'متوسط ارتفاع الأفعال الدفاعية: {avgph:.2f}م'), color='gray', rotation=-90, ha='right', va='center')
    return df_def_show

# دالة progressive_pass
def progressive_pass(ax, team_name, col, phase_tag):
    global df
    if phase_tag == 'Full Time':
        df_prop = df[(df['teamName'] == team_name) & (df['outcomeType'] == 'Successful') & (df['prog_pass'] > 9.144) & (~df['qualifiers'].str.contains('Corner|Freekick')) & (df['x'] >= 35)]
    elif phase_tag == 'First Half':
        df_prop = df[(df['period'] == 'FirstHalf') & (df['teamName'] == team_name) & (df['outcomeType'] == 'Successful') & (df['prog_pass'] > 9.144) & (~df['qualifiers'].str.contains('Corner|Freekick')) & (df['x'] >= 35)]
    elif phase_tag == 'Second Half':
        df_prop = df[(df['period'] == 'SecondHalf') & (df['teamName'] == team_name) & (df['outcomeType'] == 'Successful') & (df['prog_pass'] > 9.144) & (~df['qualifiers'].str.contains('Corner|Freekick')) & (df['x'] >= 35)]

    pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, line_zorder=3, linewidth=2)
    pitch.draw(ax=ax)

    left_prop = df_prop[df_prop['y'] > 136/3]
    midd_prop = df_prop[(df_prop['y'] <= 136/3) & (df_prop['y'] >= 68/3)]
    right_prop = df_prop[df_prop['y'] < 68/3]

    total_prop = len(df_prop)
    prop_left = len(left_prop)
    prop_mid = len(midd_prop)
    prop_right = len(right_prop)
    prop_left_per = round((prop_left / total_prop) * 100, 2) if total_prop > 0 else 0
    prop_mid_per = round((prop_mid / total_prop) * 100, 2) if total_prop > 0 else 0
    prop_right_per = round((prop_right / total_prop) * 100, 2) if total_prop > 0 else 0

    if total_prop > 0:
        name_counts = df_prop['shortName'].value_counts()
        name_counts_df = name_counts.reset_index()
        name_counts_df.columns = ['name', 'count']
        name_counts_df = name_counts_df.sort_values(by='count', ascending=False)
        name_counts_df_show = name_counts_df.reset_index(drop=True)
        most_name = name_counts_df_show['name'].iloc[0]
        most_count = name_counts_df_show['count'].iloc[0]
    else:
        most_name = 'لا يوجد'
        most_count = 0

    if prop_left > 0:
        name_counts = left_prop['shortName'].value_counts()
        name_counts_df = name_counts.reset_index()
        name_counts_df.columns = ['name', 'count']
        name_counts_df = name_counts_df.sort_values(by='count', ascending=False)
        l_name = name_counts_df['name'].iloc[0]
        l_count = name_counts_df['count'].iloc[0]
    else:
        l_name = 'لا يوجد'
        l_count = 0

    if prop_mid > 0:
        name_counts = midd_prop['shortName'].value_counts()
        name_counts_df = name_counts.reset_index()
        name_counts_df.columns = ['name', 'count']
        name_counts_df = name_counts_df.sort_values(by='count', ascending=False)
        m_name = name_counts_df['name'].iloc[0]
        m_count = name_counts_df['count'].iloc[0]
    else:
        m_name = 'لا يوجد'
        m_count = 0

    if prop_right > 0:
        name_counts = right_prop['shortName'].value_counts()
        name_counts_df = name_counts.reset_index()
        name_counts_df.columns = ['name', 'count']
        name_counts_df = name_counts_df.sort_values(by='count', ascending=False)
        r_name = name_counts_df['name'].iloc[0]
        r_count = name_counts_df['count'].iloc[0]
    else:
        r_name = 'لا يوجد'
        r_count = 0

    pitch.lines(df_prop.x, df_prop.y, df_prop.endX, df_prop.endY, comet=True, lw=4, color=col, ax=ax)
    pitch.scatter(df_prop.endX, df_prop.endY, s=75, zorder=3, color=bg_color, ec=col, lw=1.5, ax=ax)

    if phase_tag == 'Full Time':
        ax.text(34, 116, reshape_arabic_text('الوقت الكامل: 0-90 دقيقة'), color=col, fontsize=13, ha='center', va='center')
    elif phase_tag == 'First Half':
        ax.text(34, 116, reshape_arabic_text('الشوط الأول: 0-45 دقيقة'), color=col, fontsize=13, ha='center', va='center')
    elif phase_tag == 'Second Half':
        ax.text(34, 116, reshape_arabic_text('الشوط الثاني: 45-90 دقيقة'), color=col, fontsize=13, ha='center', va='center')

    ax.text(34, 112, reshape_arabic_text(f'التمريرات التقدمية في اللعب المفتوح: {total_prop}'), color=col, fontsize=13, ha='center', va='center')
    ax.text(34, 108, reshape_arabic_text(f'الأكثر تقدماً: {most_name} ({most_count})'), color=col, fontsize=13, ha='center', va='center')

    ax.text(10, 10, reshape_arabic_text(f'التقدم من الجهة اليسرى\n{prop_left} تقدم ({prop_left_per}%)'), color=col, fontsize=12, ha='center', va='center')
    ax.text(58, 10, reshape_arabic_text(f'التقدم من الجهة اليمنى\n{prop_right} تقدم ({prop_right_per}%)'), color=col, fontsize=12, ha='center', va='center')

    ax.text(340/6, -5, reshape_arabic_text(f'من اليسار: {prop_left}'), color=col, ha='center', va='center')
    ax.text(34, -5, reshape_arabic_text(f'من الوسط: {prop_mid}'), color=col, ha='center', va='center')
    ax.text(68/6, -5, reshape_arabic_text(f'من اليمين: {prop_right}'), color=col, ha='center', va='center')

    ax.text(340/6, -7, reshape_arabic_text(f'الأكثر: {l_name} ({l_count})'), color=col, ha='center', va='top')
    ax.text(34, -7, reshape_arabic_text(f'الأكثر: {m_name} ({m_count})'), color=col, ha='center', va='top')
    ax.text(68/6, -7, reshape_arabic_text(f'الأكثر: {r_name} ({r_count})'), color=col, ha='center', va='top')

    return name_counts_df_show

# دالة progressive_carry (جديدة)
def progressive_carry(ax, team_name, col, phase_tag):
    global df
    if phase_tag == 'Full Time':
        df_prop = df[(df['teamName'] == team_name) & (df['type'] == 'Carry') & (df['outcomeType'] == 'Successful') & (df['prog_carry'] > 9.144) & (df['x'] >= 35)]
    elif phase_tag == 'First Half':
        df_prop = df[(df['period'] == 'FirstHalf') & (df['teamName'] == team_name) & (df['type'] == 'Carry') & (df['outcomeType'] == 'Successful') & (df['prog_carry'] > 9.144) & (df['x'] >= 35)]
    elif phase_tag == 'Second Half':
        df_prop = df[(df['period'] == 'SecondHalf') & (df['teamName'] == team_name) & (df['type'] == 'Carry') & (df['outcomeType'] == 'Successful') & (df['prog_carry'] > 9.144) & (df['x'] >= 35)]

    pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, line_zorder=3, linewidth=2)
    pitch.draw(ax=ax)

    left_prop = df_prop[df_prop['y'] > 136/3]
    midd_prop = df_prop[(df_prop['y'] <= 136/3) & (df_prop['y'] >= 68/3)]
    right_prop = df_prop[df_prop['y'] < 68/3]

    total_prop = len(df_prop)
    prop_left = len(left_prop)
    prop_mid = len(midd_prop)
    prop_right = len(right_prop)
    prop_left_per = round((prop_left / total_prop) * 100, 2) if total_prop > 0 else 0
    prop_mid_per = round((prop_mid / total_prop) * 100, 2) if total_prop > 0 else 0
    prop_right_per = round((prop_right / total_prop) * 100, 2) if total_prop > 0 else 0

    if total_prop > 0:
        name_counts = df_prop['shortName'].value_counts()
        name_counts_df = name_counts.reset_index()
        name_counts_df.columns = ['name', 'count']
        name_counts_df = name_counts_df.sort_values(by='count', ascending=False)
        name_counts_df_show = name_counts_df.reset_index(drop=True)
        most_name = name_counts_df_show['name'].iloc[0]
        most_count = name_counts_df_show['count'].iloc[0]
    else:
        most_name = 'لا يوجد'
        most_count = 0

    if prop_left > 0:
        name_counts = left_prop['shortName'].value_counts()
        name_counts_df = name_counts.reset_index()
        name_counts_df.columns = ['name', 'count']
        name_counts_df = name_counts_df.sort_values(by='count', ascending=False)
        l_name = name_counts_df['name'].iloc[0]
        l_count = name_counts_df['count'].iloc[0]
    else:
        l_name = 'لا يوجد'
        l_count = 0

    if prop_mid > 0:
        name_counts = midd_prop['shortName'].value_counts()
        name_counts_df = name_counts.reset_index()
        name_counts_df.columns = ['name', 'count']
        name_counts_df = name_counts_df.sort_values(by='count', ascending=False)
        m_name = name_counts_df['name'].iloc[0]
        m_count = name_counts_df['count'].iloc[0]
    else:
        m_name = 'لا يوجد'
        m_count = 0

    if prop_right > 0:
        name_counts = right_prop['shortName'].value_counts()
        name_counts_df = name_counts.reset_index()
        name_counts_df.columns = ['name', 'count']
        name_counts_df = name_counts_df.sort_values(by='count', ascending=False)
        r_name = name_counts_df['name'].iloc[0]
        r_count = name_counts_df['count'].iloc[0]
    else:
        r_name = 'لا يوجد'
        r_count = 0

    pitch.lines(df_prop.x, df_prop.y, df_prop.endX, df_prop.endY, comet=True, lw=4, color=col, ax=ax)
    pitch.scatter(df_prop.endX, df_prop.endY, s=75, zorder=3, color=bg_color, ec=col, lw=1.5, ax=ax)

    if phase_tag == 'Full Time':
        ax.text(34, 116, reshape_arabic_text('الوقت الكامل: 0-90 دقيقة'), color=col, fontsize=13, ha='center', va='center')
    elif phase_tag == 'First Half':
        ax.text(34, 116, reshape_arabic_text('الشوط الأول: 0-45 دقيقة'), color=col, fontsize=13, ha='center', va='center')
    elif phase_tag == 'Second Half':
        ax.text(34, 116, reshape_arabic_text('الشوط الثاني: 45-90 دقيقة'), color=col, fontsize=13, ha='center', va='center')

    ax.text(34, 112, reshape_arabic_text(f'حمل الكرة التقدمي في اللعب المفتوح: {total_prop}'), color=col, fontsize=13, ha='center', va='center')
    ax.text(34, 108, reshape_arabic_text(f'الأكثر تقدماً: {most_name} ({most_count})'), color=col, fontsize=13, ha='center', va='center')

    ax.text(10, 10, reshape_arabic_text(f'التقدم من الجهة اليسرى\n{prop_left} تقدم ({prop_left_per}%)'), color=col, fontsize=12, ha='center', va='center')
    ax.text(58, 10, reshape_arabic_text(f'التقدم من الجهة اليمنى\n{prop_right} تقدم ({prop_right_per}%)'), color=col, fontsize=12, ha='center', va='center')

    ax.text(340/6, -5, reshape_arabic_text(f'من اليسار: {prop_left}'), color=col, ha='center', va='center')
    ax.text(34, -5, reshape_arabic_text(f'من الوسط: {prop_mid}'), color=col, ha='center', va='center')
    ax.text(68/6, -5, reshape_arabic_text(f'من اليمين: {prop_right}'), color=col, ha='center', va='center')

    ax.text(340/6, -7, reshape_arabic_text(f'الأكثر: {l_name} ({l_count})'), color=col, ha='center', va='top')
    ax.text(34, -7, reshape_arabic_text(f'الأكثر: {m_name} ({m_count})'), color=col, ha='center', va='top')
    ax.text(68/6, -7, reshape_arabic_text(f'الأكثر: {r_name} ({r_count})'), color=col, ha='center', va='top')

    return name_counts_df_show

# الجزء الرئيسي
if match_url and st.session_state.confirmed:
    try:
        df, teams_dict, players_df = get_event_data(match_url)
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
        hgoal_count += len(awaydf[(awaydf['teamName'] == ateamName) & (awaydf['type'] == 'Goal') & (awaydf['qualifiers'].str.contains('OwnGoal'))])
        agoal_count += len(homedf[(homedf['teamName'] == hteamName) & (homedf['type'] == 'Goal') & (homedf['qualifiers'].str.contains('OwnGoal'))])

        # معرفات الفرق لشعارات FotMob (قيم افتراضية، استبدلها إذا كنت تستخدم API)
        hftmb_tid = '9879'  # مثال: معرف لبرشلونة
        aftmb_tid = '10267'  # مثال: معرف لريال مدريد

        st.header(f'{hteamName} {hgoal_count} - {agoal_count} {ateamName}')
        st.text(reshape_arabic_text('تحليل المباراة بناءً على الرابط'))

# تبويبات التحليل
# تعريف التبويبات
        tab1, tab2, tab3, tab4 = st.tabs([reshape_arabic_text('تحليل الفريق'), reshape_arabic_text('تحليل اللاعبين'), reshape_arabic_text('إحصائيات المباراة'), reshape_arabic_text('أفضل اللاعبين')])

        with tab1:
            an_tp = st.selectbox(reshape_arabic_text('نوع التحليل:'), [
                reshape_arabic_text('شبكة التمريرات'),
                reshape_arabic_text('الخريطة الحرارية للأفعال الدفاعية'),
                reshape_arabic_text('التمريرات التقدمية'),
                reshape_arabic_text('حمل الكرة التقدمي'),
                reshape_arabic_text('خريطة التسديدات'),
                reshape_arabic_text('إحصائيات الحراس'),
                reshape_arabic_text('زخم المباراة'),
                reshape_arabic_text('تمريرات المنطقة 14 ونصف المساحات'),
                reshape_arabic_text('الدخول للثلث الأخير'),
                reshape_arabic_text('الدخول لمنطقة الجزاء'),
                reshape_arabic_text('الاستعادات العالية'),
                reshape_arabic_text('مناطق خلق الفرص'),
                reshape_arabic_text('العرضيات'),
                reshape_arabic_text('مناطق سيطرة الفريق'),
                reshape_arabic_text('مناطق استهداف التمريرات'),
                reshape_arabic_text('الثلث الهجومي'),
                'Shotmap'  # Added to match the condition
            ], index=0, key='analysis_type')

            if an_tp == reshape_arabic_text('شبكة التمريرات'):
                st.header(reshape_arabic_text('شبكة التمريرات'))
                pn_time_phase = st.radio(reshape_arabic_text("اختر الفترة:"), [reshape_arabic_text('الوقت الكامل'), reshape_arabic_text('الشوط الأول'), reshape_arabic_text('الشوط الثاني')], index=0, key='pn_time_pill')

                fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
                home_pass_btn = None
                away_pass_btn = None

                phase_map = {
                    reshape_arabic_text('الوقت الكامل'): 'Full Time',
                    reshape_arabic_text('الشوط الأول'): 'First Half',
                    reshape_arabic_text('الشوط الثاني'): 'Second Half'
                }

                home_pass_btn = pass_network(axs[0], hteamName, hcol, phase_map[pn_time_phase])
                away_pass_btn = pass_network(axs[1], ateamName, acol, phase_map[pn_time_phase])

                home_part = reshape_arabic_text(f"{hteamName} {hgoal_count}")
                away_part = reshape_arabic_text(f"{agoal_count} {ateamName}")
                title = f"<{home_part}> - <{away_part}>"
                fig_text(0.5, 1.05, title, highlight_textprops=[{'color': hcol}, {'color': acol}], fontsize=28, fontweight='bold', ha='center', va='center')
                fig.text(0.5, 1.01, reshape_arabic_text('شبكة التمريرات'), fontsize=18, ha='center', va='center', color='white', weight='bold')
                fig.text(0.5, 0.97, '✦ @REO_SHOW ✦', fontsize=14, fontfamily='Roboto', fontweight='bold', color='#FFD700', ha='center', va='center',
                         bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=2),
                         path_effects=[patheffects.withStroke(linewidth=2, foreground='white')])

                fig.text(0.5, 0.02, reshape_arabic_text('*الدوائر = اللاعبون الأساسيون، المربعات = اللاعبون البدلاء، الأرقام داخلها = أرقام القمصان'), fontsize=10, fontstyle='italic', ha='center', va='center', color='white')
                fig.text(0.5, 0.00, reshape_arabic_text('*عرض وإضاءة الخطوط تمثل عدد التمريرات الناجحة في اللعب المفتوح بين اللاعبين'), fontsize=10, fontstyle='italic', ha='center', va='center', color='white')

                try:
                    himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png")
                    himage = Image.open(himage)
                    ax_himage = add_image(himage, fig, left=0.085, bottom=0.97, width=0.125, height=0.125)

                    aimage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png")
                    aimage = Image.open(aimage)
                    ax_aimage = add_image(aimage, fig, left=0.815, bottom=0.97, width=0.125, height=0.125)
                except:
                    st.warning(reshape_arabic_text("تعذر تحميل شعارات الفرق"))

                plt.subplots_adjust(top=0.85, bottom=0.15)
                st.pyplot(fig)

                col1, col2 = st.columns(2)
                with col1:
                    st.write(reshape_arabic_text(f'أزواج التمرير لفريق {hteamName}:'))
                    if home_pass_btn is not None:
                        st.dataframe(home_pass_btn, hide_index=True)
                    else:
                        st.write(reshape_arabic_text("لا توجد بيانات متاحة."))
                with col2:
                    st.write(reshape_arabic_text(f'أزواج التمرير لفريق {ateamName}:'))
                    if away_pass_btn is not None:
                        st.dataframe(away_pass_btn, hide_index=True)
                    else:
                        st.write(reshape_arabic_text("لا توجد بيانات متاحة."))

            elif an_tp == reshape_arabic_text('الخريطة الحرارية للأفعال الدفاعية'):
                st.header(reshape_arabic_text('الخريطة الحرارية للأفعال الدفاعية'))
                dah_time_phase = st.radio(reshape_arabic_text("اختر الفترة:"), [reshape_arabic_text('الوقت الكامل'), reshape_arabic_text('الشوط الأول'), reshape_arabic_text('الشوط الثاني')], index=0, key='dah_time_pill')

                fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
                phase_map = {
                    reshape_arabic_text('الوقت الكامل'): 'Full Time',
                    reshape_arabic_text('الشوط الأول'): 'First Half',
                    reshape_arabic_text('الشوط الثاني'): 'Second Half'
                }

                home_df_def = def_acts_hm(axs[0], hteamName, hcol, phase_map[dah_time_phase])
                away_df_def = def_acts_hm(axs[1], ateamName, acol, phase_map[dah_time_phase])

                home_part = reshape_arabic_text(f"{hteamName} {hgoal_count}")
                away_part = reshape_arabic_text(f"{agoal_count} {ateamName}")
                title = f"<{home_part}> - <{away_part}>"
                fig_text(0.5, 1.05, title, highlight_textprops=[{'color': hcol}, {'color': acol}], fontsize=30, fontweight='bold', ha='center', va='center')
                fig.text(0.5, 1.01, reshape_arabic_text('الخريطة الحرارية للأفعال الدفاعية'), fontsize=20, ha='center', va='center', color='white')
                fig.text(0.5, 0.97, '@REO_SHOW', fontsize=15, ha='center', va='center', color='#FFD700')

                fig.text(0.5, 0.05, reshape_arabic_text('*الدوائر = اللاعبون الأساسيون، المربعات = اللاعبون البدلاء، الأرقام داخلها = أرقام القمصان'), fontsize=10, fontstyle='italic', ha='center', va='center', color='white')
                fig.text(0.5, 0.03, reshape_arabic_text('*حجم الدوائر/المربعات يمثل عدد الأفعال الدفاعية للاعبي الميدان'), fontsize=10, fontstyle='italic', ha='center', va='center', color='white')

                try:
                    himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png")
                    himage = Image.open(himage)
                    ax_himage = add_image(himage, fig, left=0.085, bottom=0.97, width=0.125, height=0.125)

                    aimage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png")
                    aimage = Image.open(aimage)
                    ax_aimage = add_image(aimage, fig, left=0.815, bottom=0.97, width=0.125, height=0.125)
                except:
                    st.warning(reshape_arabic_text("تعذر تحميل شعارات الفرق"))

                plt.subplots_adjust(top=0.85, bottom=0.15)
                st.pyplot(fig)

                col1, col2 = st.columns(2)
                with col1:
                    st.write(reshape_arabic_text(f'الأفعال الدفاعية للاعبي فريق {hteamName}:'))
                    if home_df_def is not None:
                        st.dataframe(home_df_def, hide_index=True)
                    else:
                        st.write(reshape_arabic_text("لا توجد بيانات متاحة."))
                with col2:
                    st.write(reshape_arabic_text(f'الأفعال الدفاعية للاعبي فريق {ateamName}:'))
                    if away_df_def is not None:
                        st.dataframe(away_df_def, hide_index=True)
                    else:
                        st.write(reshape_arabic_text("لا توجد بيانات متاحة."))
        def progressive_pass(ax, team_name, col, phase_tag):
            if phase_tag == 'Full Time':
                df_prop = df[(df['teamName'] == team_name) & (df['outcomeType'] == 'Successful') & (df['prog_pass'] > 9.144) & (~df['qualifiers'].str.contains('Corner|Freekick')) & (df['x'] >= 35)]
            elif phase_tag == 'First Half':
                df_fh = df[df['period'] == 'FirstHalf']
                df_prop = df_fh[(df_fh['teamName'] == team_name) & (df_fh['outcomeType'] == 'Successful') & (df_fh['prog_pass'] > 9.144) & (~df_fh['qualifiers'].str.contains('Corner|Freekick')) & (df_fh['x'] >= 35)]
            elif phase_tag == 'Second Half':
                df_sh = df[df['period'] == 'SecondHalf']
                df_prop = df_sh[(df_sh['teamName'] == team_name) & (df_sh['outcomeType'] == 'Successful') & (df_sh['prog_pass'] > 9.144) & (~df_sh['qualifiers'].str.contains('Corner|Freekick')) & (df_sh['x'] >= 35)]

            pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, line_zorder=3, linewidth=2)
            pitch.draw(ax=ax)

            left_prop = df_prop[df_prop['y'] > 136/3]
            midd_prop = df_prop[(df_prop['y'] <= 136/3) & (df_prop['y'] >= 68/3)]
            rigt_prop = df_prop[df_prop['y'] < 68/3]

            total_prop = len(df_prop)
            prop_left = len(left_prop)
            prop_mid = len(midd_prop)
            prop_right = len(rigt_prop)

            prop_left_per = round((prop_left / total_prop * 100), 2) if total_prop > 0 else 0
            prop_mid_per = round((prop_mid / total_prop * 100), 2) if total_prop > 0 else 0
            prop_right_per = round((prop_right / total_prop * 100), 2) if total_prop > 0 else 0

            if total_prop != 0:
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

            if prop_left != 0:
                name_counts = left_prop['shortName'].value_counts()
                name_counts_df = name_counts.reset_index()
                name_counts_df.columns = ['name', 'count']
                name_counts_df = name_counts_df.sort_values(by='count', ascending=False)
                l_name = name_counts_df['name'][0]
                l_count = name_counts_df['count'][0]
            else:
                l_name = 'None'
                l_count = 0

            if prop_mid != 0:
                name_counts = midd_prop['shortName'].value_counts()
                name_counts_df = name_counts.reset_index()
                name_counts_df.columns = ['name', 'count']
                name_counts_df = name_counts_df.sort_values(by='count', ascending=False)
                m_name = name_counts_df['name'][0]
                m_count = name_counts_df['count'][0]
            else:
                m_name = 'None'
                m_count = 0

            if prop_right != 0:
                name_counts = rigt_prop['shortName'].value_counts()
                name_counts_df = name_counts.reset_index()
                name_counts_df.columns = ['name', 'count']
                name_counts_df = name_counts_df.sort_values(by='count', ascending=False)
                r_name = name_counts_df['name'][0]
                r_count = name_counts_df['count'][0]
            else:
                r_name = 'None'
                r_count = 0

            pitch.lines(df_prop.x, df_prop.y, df_prop.endX, df_prop.endY, comet=True, lw=4, color=col, ax=ax)
            pitch.scatter(df_prop.endX, df_prop.endY, s=75, zorder=3, color=bg_color, ec=col, lw=1.5, ax=ax)

            if phase_tag == 'Full Time':
                ax.text(34, 116, reshape_arabic_text('الوقت الكامل: 0-90 دقيقة'), color=col, fontsize=13, ha='center', va='center')
            elif phase_tag == 'First Half':
                ax.text(34, 116, reshape_arabic_text('الشوط الأول: 0-45 دقيقة'), color=col, fontsize=13, ha='center', va='center')
            elif phase_tag == 'Second Half':
                ax.text(34, 116, reshape_arabic_text('الشوط الثاني: 45-90 دقيقة'), color=col, fontsize=13, ha='center', va='center')

            ax.text(34, 112, reshape_arabic_text(f'التمريرات التقدمية في اللعب المفتوح: {total_prop}'), color=col, fontsize=13, ha='center', va='center')
            ax.text(34, 108, reshape_arabic_text(f'الأكثر: {most_name} ({most_count})'), color=col, fontsize=13, ha='center', va='center')

            ax.text(10, 10, reshape_arabic_text(f'التقدم من الجهة اليسرى\n{prop_left} تقدم ({prop_left_per}%)'), color=col, fontsize=12, ha='center', va='center')
            ax.text(34, 10, reshape_arabic_text(f'التقدم من الوسط\n{prop_mid} تقدم ({prop_mid_per}%)'), color=col, fontsize=12, ha='center', va='center')
            ax.text(58, 10, reshape_arabic_text(f'التقدم من الجهة اليمنى\n{prop_right} تقدم ({prop_right_per}%)'), color=col, fontsize=12, ha='center', va='center')

            ax.text(340/6, -5, reshape_arabic_text(f'من اليسار: {prop_left}'), color=col, ha='center', va='center')
            ax.text(34, -5, reshape_arabic_text(f'من الوسط: {prop_mid}'), color=col, ha='center', va='center')
            ax.text(68/6, -5, reshape_arabic_text(f'من اليمين: {prop_right}'), color=col, ha='center', va='center')

            ax.text(340/6, -7, reshape_arabic_text(f'الأكثر:\n{l_name} ({l_count})'), color=col, ha='center', va='top')
            ax.text(34, -7, reshape_arabic_text(f'الأكثر:\n{m_name} ({m_count})'), color=col, ha='center', va='top')
            ax.text(68/6, -7, reshape_arabic_text(f'الأكثر:\n{r_name} ({r_count})'), color=col, ha='center', va='top')

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
                st.header(reshape_arabic_text('خريطة التسديدات'))
                st.header(f'{an_tp}')
