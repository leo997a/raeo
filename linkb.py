import streamlit as st
import pandas as pd
import json
import re
from unidecode import unidecode
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import to_rgba, LinearSegmentedColormap
from mplsoccer import VerticalPitch, add_image
from highlight_text import fig_text
from PIL import Image
from urllib.request import urlopen
import arabic_reshaper
from bidi.algorithm import get_display
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
from requests.exceptions import RequestException

# تهيئة matplotlib لدعم العربية
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Amiri', 'Noto Sans Arabic', 'Arial', 'Tahoma']
mpl.rcParams['axes.unicode_minus'] = False

# دالة لتحويل النص العربي
def reshape_arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# إضافة CSS لدعم RTL
st.markdown("""
    <style>
    body {
        direction: rtl;
        text-align: right;
    }
    .stSelectbox > div > div > div {
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

# تعريف الألوان الافتراضية
default_hcol = '#d00000'
default_acol = '#003087'
default_bg_color = '#1e1e2f'
default_gradient_colors = ['#003087', '#d00000']

# واجهة Streamlit
st.sidebar.title('إدخال رابط المباراة')
match_url = st.sidebar.text_input('أدخل رابط المباراة:', key='match_url', placeholder='https://1xbet.whoscored.com/Matches/1234567/Live')
match_input = st.sidebar.button('تحليل المباراة', key='confirm_url', on_click=lambda: st.session_state.update({'confirmed': True}))

st.sidebar.title('اختيار الألوان')
hcol = st.sidebar.color_picker('لون الفريق المضيف', default_hcol, key='hcol_picker')
acol = st.sidebar.color_picker('لون الفريق الضيف', default_acol, key='acol_picker')
bg_color = st.sidebar.color_picker('لون الخلفية', default_bg_color, key='bg_color_picker')
gradient_start = st.sidebar.color_picker('بداية التدرج', default_gradient_colors[0], key='gradient_start_picker')
gradient_end = st.sidebar.color_picker('نهاية التدرج', default_gradient_colors[1], key='gradient_end_picker')
gradient_colors = [gradient_start, gradient_end]
line_color = st.sidebar.color_picker('لون الخطوط', '#ffffff', key='line_color_picker')

if 'confirmed' not in st.session_state:
    st.session_state.confirmed = False

# دالة لتحويل الأسماء إلى صيغة مختصرة
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

# دالة لحساب الدقائق التراكمية
def cumulative_match_mins(events_df):
    events_out = pd.DataFrame()
    match_events = events_df.copy()
    match_events['cumulative_mins'] = match_events['minute'] + (1/60) * match_events['second']
    for period in np.arange(1, match_events['period'].max() + 1, 1):
        if period > 1:
            t_delta = match_events[match_events['period'] == period - 1]['cumulative_mins'].max() - \
                      match_events[match_events['period'] == period]['cumulative_mins'].min()
        elif period == 1 or period == 5:
            t_delta = 0
        else:
            t_delta = 0
        match_events.loc[match_events['period'] == period, 'cumulative_mins'] += t_delta
    events_out = pd.concat([events_out, match_events])
    return events_out

# دالة لإضافة حمل الكرة
def insert_ball_carries(events_df, min_carry_length=3, max_carry_length=100, min_carry_duration=1, max_carry_duration=50):
    events_out = pd.DataFrame()
    min_carry_length = 3.0
    max_carry_length = 100.0
    min_carry_duration = 1.0
    max_carry_duration = 50.0
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
            while incorrect_next_evt:
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

# دالة لسلاسل الاستحواذ
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

# دالة استخراج البيانات باستخدام Selenium
@st.cache_data
def get_event_data(match_url):
    # تهيئة Selenium
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # تشغيل بدون واجهة
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(match_url)
        time.sleep(5)  # انتظار تحميل JavaScript
        html = driver.page_source
    except Exception as e:
        st.error(f"فشل في تحميل الصفحة: {e}")
        return None, None, None
    finally:
        driver.quit()

    # استخراج JSON
    regex_pattern = r'(?<=require\.config\.params\["args"\].=.)[\s\S]*?;'
    try:
        data_txt = re.findall(regex_pattern, html)[0]
        data_txt = data_txt.replace('matchId', '"matchId"')
        data_txt = data_txt.replace('matchCentreData', '"matchCentreData"')
        data_txt = data_txt.replace('matchCentreEventTypeJson', '"matchCentreEventTypeJson"')
        data_txt = data_txt.replace('formationIdNameMappings', '"formationIdNameMappings"')
        data_txt = data_txt.replace('};', '}')
        data = json.loads(data_txt)
    except (IndexError, json.JSONDecodeError):
        st.error("فشل في استخراج أو تحليل بيانات JSON من الصفحة.")
        return None, None, None

    # معالجة البيانات
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

    events_dict, players_df, teams_dict = extract_data_from_dict(data)
    df = pd.DataFrame(events_dict)
    dfp = pd.DataFrame(players_df)

    # معالجة الأحداث
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

    # حساب xT
    df_base = df
    dfxT = df_base.copy()
    dfxT['qualifiers'] = dfxT['qualifiers'].astype(str)
    dfxT = dfxT[(~dfxT['qualifiers'].str.contains('Corner'))]
    dfxT = dfxT[(dfxT['type'].isin(['Pass', 'Carry'])) & (dfxT['outcomeType'] == 'Successful')]
    try:
        xT = pd.read_csv("https://raw.githubusercontent.com/adnaaan433/Post-Match-Report-2.0/refs/heads/main/xT_Grid.csv", header=None)
        xT = np.array(xT)
    except RequestException:
        st.error("فشل في تحميل شبكة xT.")
        return None, None, None
    xT_rows, xT_cols = xT.shape
    dfxT['x1_bin_xT'] = pd.cut(dfxT['x'], bins=xT_cols, labels=False)
    dfxT['y1_bin_xT'] = pd.cut(dfxT['y'], bins=xT_rows, labels=False)
    dfxT['x2_bin_xT'] = pd.cut(dfxT['endX'], bins=xT_cols, labels=False)
    dfxT['y2_bin_xT'] = pd.cut(dfxT['endY'], bins=xT_rows, labels=False)
    dfxT['start_zone_value_xT'] = dfxT[['x1_bin_xT', 'y1_bin_xT']].apply(lambda x: xT[x[1]][x[0]] if pd.notnull(x[0]) and pd.notnull(x[1]) else 0, axis=1)
    dfxT['end_zone_value_xT'] = dfxT[['x2_bin_xT', 'y2_bin_xT']].apply(lambda x: xT[x[1]][x[0]] if pd.notnull(x[0]) and pd.notnull(x[1]) else 0, axis=1)
    dfxT['xT'] = dfxT['end_zone_value_xT'] - dfxT['start_zone_value_xT']
    columns_to_drop = ['id', 'eventId', 'minute', 'second', 'teamId', 'x', 'y', 'expandedMinute', 'period',
                       'outcomeType', 'qualifiers', 'type', 'satisfiedEventsTypes', 'isTouch', 'playerId', 'endX', 'endY',
                       'relatedEventId', 'relatedPlayerId', 'blockedX', 'blockedY', 'goalMouthZ', 'goalMouthY', 'isShot', 'cumulative_mins']
    dfxT.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    df = df.merge(dfxT, on='index', how='left')
    df['teamName'] = df['teamId'].map(teams_dict)
    team_names = list(teams_dict.values())
    opposition_dict = {team_names[i]: team_names[1 - i] for i in range(len(team_names))}
    df['oppositionTeamName'] = df['teamName'].map(opposition_dict)

    # تحجيم الملعب
    df['x'] = df['x'] * 1.05
    df['y'] = df['y'] * 0.68
    df['endX'] = df['endX'] * 1.05
    df['endY'] = df['endY'] * 0.68
    df['goalMouthY'] = df['goalMouthY'] * 0.68
    columns_to_drop = ['height', 'weight', 'age', 'isManOfTheMatch', 'field', 'stats', 'subbedInPlayerId',
                       'subbedOutPeriod', 'subbedOutExpandedMinute', 'subbedInPeriod', 'subbedInExpandedMinute', 'subbedOutPlayerId', 'teamId']
    dfp.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    df = df.merge(dfp, on='playerId', how='left')
    df['qualifiers'] = df['qualifiers'].astype(str)
    df['prog_pass'] = np.where((df['type'] == 'Pass'),
                               np.sqrt((105 - df['x']) ** 2 + (34 - df['y']) ** 2) - np.sqrt(
                                   (105 - df['endX']) ** 2 + (34 - df['endY']) ** 2), 0)
    df['prog_carry'] = np.where((df['type'] == 'Carry'),
                                np.sqrt((105 - df['x']) ** 2 + (34 - df['y']) ** 2) - np.sqrt(
                                    (105 - df['endX']) ** 2 + (34 - df['endY']) ** 2), 0)
    df['pass_or_carry_angle'] = np.degrees(np.arctan2(df['endY'] - df['y'], df['endX'] - df['x']))
    df['name'] = df['name'].astype(str)
    df['name'] = df['name'].apply(unidecode)
    df['shortName'] = df['name'].apply(get_short_name)
    df['qualifiers'] = df['qualifiers'].astype(str)
    columns_to_drop2 = ['id']
    df.drop(columns=columns_to_drop2, inplace=True, errors='ignore')

    df = get_possession_chains(df, 5, 3)
    df['period'] = df['period'].replace({
        1: 'FirstHalf', 2: 'SecondHalf', 3: 'FirstPeriodOfExtraTime', 4: 'SecondPeriodOfExtraTime',
        5: 'PenaltyShootout', 14: 'PostGame', 16: 'PreMatch'
    })
    df = df[df['period'] != 'PenaltyShootout']
    df = df.reset_index(drop=True)
    return df, teams_dict, players_df

# دالة شبكة التمريرات
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
    accuracy = round((len(accrt_pass) / len(total_pass)) * 100, 2) if len(total_pass) != 0 else 0

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
        ax.text(34, 112, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), color='white', fontsize=12, ha='center', va='center')
    elif phase_tag == 'First Half':
        ax.text(34, 115, reshape_arabic_text('الشوط الأول: 0-45 دقيقة'), color='white', fontsize=14, ha='center', va='center', weight='bold')
        ax.text(34, 112, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), color='white', fontsize=12, ha='center', va='center')
    elif phase_tag == 'Second Half':
        ax.text(34, 115, reshape_arabic_text('الشوط الثاني: 45-90 دقيقة'), color='white', fontsize=14, ha='center', va='center', weight='bold')
        ax.text(34, 112, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), color='white', fontsize=12, ha='center', va='center')

    ax.text(34, -6, reshape_arabic_text(f"على الكرة\nالتماسك العمودي (المنطقة المظللة): {v_comp}%"), color='white', fontsize=12, ha='center', va='center', weight='bold')
    return pass_btn

# معالجة البيانات وتحليل المباراة
if match_url and st.session_state.confirmed:
    df, teams_dict, players_df = get_event_data(match_url)
    if df is None or teams_dict is None or players_df is None:
        st.stop()

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

    # محاولة جلب معرفات الفرق لشعارات FotMob
    try:
        df_teamNameId = pd.read_csv('https://raw.githubusercontent.com/adnaaan433/Post-Match-Report-2.0/63f5b51d8bd8b3f40e3d02fece1defb2f18ddf54/teams_name_and_id.csv')
        hftmb_tid = df_teamNameId[df_teamNameId['teamName'] == hteamName].teamId.to_list()[0]
        aftmb_tid = df_teamNameId[df_teamNameId['teamName'] == ateamName].teamId.to_list()[0]
    except:
        st.warning("فشل في جلب معرفات الفرق. الشعارات قد لا تظهر.")
        hftmb_tid = None
        aftmb_tid = None

    st.header(f'{hteamName} {hgoal_count} - {agoal_count} {ateamName}')

    # تبويبات التحليل
    tab1, tab2, tab3, tab4 = st.tabs(['تحليل الفريق', 'تحليل اللاعب', 'إحصائيات المباراة', 'أفضل اللاعبين'])

    with tab1:
        an_tp = st.selectbox('نوع التحليل:', [
            'شبكة التمريرات', 'Defensive Actions Heatmap', 'Progressive Passes', 'Progressive Carries', 'Shotmap',
            'احصائيات الحراس', 'Match Momentum', reshape_arabic_text('Zone14 & Half-Space Passes'),
            reshape_arabic_text('Final Third Entries'), reshape_arabic_text('Box Entries'),
            reshape_arabic_text('High-Turnovers'), reshape_arabic_text('Chances Creating Zones'),
            reshape_arabic_text('Crosses'), reshape_arabic_text('Team Domination Zones'),
            reshape_arabic_text('Pass Target Zones'), 'Attacking Thirds'
        ], index=0, key='analysis_type')

        if an_tp == 'شبكة التمريرات':
            st.header('شبكة التمريرات')
            pn_time_phase = st.radio(" ", ['Full Time', 'First Half', 'Second Half'], index=0, key='pn_time_pill')
            fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
            home_pass_btn = pass_network(axs[0], hteamName, hcol, pn_time_phase)
            away_pass_btn = pass_network(axs[1], ateamName, acol, pn_time_phase)

            home_part = reshape_arabic_text(f"{hteamName} {hgoal_count}")
            away_part = reshape_arabic_text(f"{agoal_count} {ateamName}")
            title = f"<{home_part}> - <{away_part}>"
            fig_text(0.5, 1.05, title, 
                     highlight_textprops=[{'color': hcol}, {'color': acol}],
                     fontsize=28, fontweight='bold', ha='center', va='center', ax=fig)
            fig.text(0.5, 1.01, reshape_arabic_text('شبكة التمريرات'), fontsize=18, ha='center', va='center', color='white', weight='bold')
            fig.text(0.5, 0.97, '✦ @REO_SHOW ✦', 
                     fontsize=14, fontfamily='Roboto', fontweight='bold', 
                     color='#FFD700', ha='center', va='center',
                     bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=2))

            fig.text(0.5, 0.02, reshape_arabic_text('*الدوائر = اللاعبون الأساسيون، المربعات = اللاعبون البدلاء، الأرقام داخلها = أرقام القمصان'),
                     fontsize=10, fontstyle='italic', ha='center', va='center', color='white')
            fig.text(0.5, 0.00, reshape_arabic_text('*عرض وإضاءة الخطوط تمثل عدد التمريرات الناجحة في اللعب المفتوح بين اللاعبين'),
                     fontsize=10, fontstyle='italic', ha='center', va='center', color='white')

            if hftmb_tid and aftmb_tid:
                try:
                    himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png")
                    himage = Image.open(himage)
                    ax_himage = add_image(himage, fig, left=0.085, bottom=0.97, width=0.125, height=0.125)
                    aimage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png")
                    aimage = Image.open(aimage)
                    ax_aimage = add_image(aimage, fig, left=0.815, bottom=0.97, width=0.125, height=0.125)
                except:
                    st.warning("فشل في تحميل شعارات الفرق.")

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
