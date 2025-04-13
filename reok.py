# الاستيرادات
import json
import re
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba, LinearSegmentedColormap
import seaborn as sns
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
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# تهيئة matplotlib لدعم العربية
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Amiri', 'Noto Sans Arabic', 'Arial', 'Tahoma']
mpl.rcParams['axes.unicode_minus'] = False

# دالة لتحويل النص العربي
def reshape_arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# إضافة CSS لدعم RTL في streamlit
st.markdown("""
    <style>
    body {
        direction: rtl;
        text-align: right;
    }
    .stTextInput > div > div > input {
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

# تعريف القيم الافتراضية للألوان
default_hcol = '#d00000'
default_acol = '#003087'
default_bg_color = '#1e1e2f'
default_gradient_colors = ['#003087', '#d00000']
violet = '#800080'

# الشريط الجانبي لاختيار الألوان
st.sidebar.title(reshape_arabic_text('اختيار الألوان'))
hcol = st.sidebar.color_picker(reshape_arabic_text('لون الفريق المضيف'), default_hcol, key='hcol_picker')
acol = st.sidebar.color_picker(reshape_arabic_text('لون الفريق الضيف'), default_acol, key='acol_picker')
bg_color = st.sidebar.color_picker(reshape_arabic_text('لون الخلفية'), default_bg_color, key='bg_color_picker')
gradient_start = st.sidebar.color_picker(reshape_arabic_text('بداية التدرج'), default_gradient_colors[0], key='gradient_start_picker')
gradient_end = st.sidebar.color_picker(reshape_arabic_text('نهاية التدرج'), default_gradient_colors[1], key='gradient_end_picker')
gradient_colors = [gradient_start, gradient_end]
line_color = st.sidebar.color_picker(reshape_arabic_text('لون الخطوط'), '#ffffff', key='line_color_picker')

# حقل إدخال رابط المباراة
st.title(reshape_arabic_text('تحليل المباراة'))
match_url = st.text_input(reshape_arabic_text('أدخل رابط المباراة من Whoscored (مثال: https://1xbet.whoscored.com/Matches/1809770/Live/Europe-Europa-League-2023-2024-West-Ham-Bayer-Leverkusen)'), '')

# تهيئة الحالة
if 'confirmed' not in st.session_state:
    st.session_state.confirmed = False

def reset_confirmed():
    st.session_state['confirmed'] = False

# دالة لاستخراج بيانات المباراة من Whoscored باستخدام Firefox
@st.cache_data
def extract_match_dict(match_url, max_retries=2):
    """Extract match event from Whoscored match center using Firefox"""
    firefox_options = Options()
    firefox_options.add_argument("--headless")
    firefox_options.add_argument("--no-sandbox")
    firefox_options.add_argument("--disable-dev-shm-usage")
    firefox_options.add_argument("--disable-gpu")
    
    for attempt in range(max_retries):
        driver = None
        try:
            driver = webdriver.Firefox(
                service=Service(GeckoDriverManager().install()),
                options=firefox_options
            )
            driver.get(match_url)
            
            # الانتظار حتى يتم تحميل العنصر المطلوب
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "script"))
            )
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            element = soup.select_one('script:-soup-contains("matchCentreData")')
            
            if element:
                try:
                    matchdict = json.loads(element.text.split("matchCentreData: ")[1].split(',\n')[0])
                    return matchdict
                except json.JSONDecodeError:
                    st.warning(reshape_arabic_text(f"فشل تحليل JSON في المحاولة {attempt + 1}. إعادة المحاولة..."))
                    continue
            else:
                st.warning(reshape_arabic_text(f"تعذر العثور على matchCentreData في المحاولة {attempt + 1}. إعادة المحاولة..."))
                continue
        
        except Exception as e:
            st.error(reshape_arabic_text(f"خطأ في جلب البيانات (المحاولة {attempt + 1}): {str(e)}"))
            continue
        
        finally:
            if driver:
                driver.quit()
    
    st.error(reshape_arabic_text("فشل جلب البيانات بعد عدة محاولات. تأكد من الرابط وحاول مرة أخرى."))
    return None

# دالة لاستخراج الأحداث، اللاعبين، والفرق من القاموس
def extract_data_from_dict(data):
    """Extract events, players, and teams from match dictionary"""
    if not data:
        return None, None, None
    
    try:
        events_dict = data.get("events", [])
        teams_dict = {
            data['home']['teamId']: data['home']['name'],
            data['away']['teamId']: data['away']['name']
        } if data.get('home') and data.get('away') else {}
        
        players_home_df = pd.DataFrame(data.get('home', {}).get('players', []))
        players_away_df = pd.DataFrame(data.get('away', {}).get('players', []))
        
        if not players_home_df.empty:
            players_home_df["teamId"] = data['home']['teamId']
        if not players_away_df.empty:
            players_away_df["teamId"] = data['away']['teamId']
        
        players_df = pd.concat([players_home_df, players_away_df], ignore_index=True) if not (players_home_df.empty and players_away_df.empty) else pd.DataFrame()
        
        return events_dict, players_df, teams_dict
    
    except KeyError as e:
        st.error(reshape_arabic_text(f"خطأ في هيكلية البيانات: مفتاح مفقود {str(e)}"))
        return None, None, None

# دالة معدلة لجلب بيانات المباراة
@st.cache_data
def get_event_data(match_url):
    json_data = extract_match_dict(match_url)
    if not json_data:
        return None, None, None
    
    events_dict, players_df, teams_dict = extract_data_from_dict(json_data)
    if not events_dict:
        return None, None, None
    
    df = pd.DataFrame(events_dict)
    if df.empty:
        return None, None, None
    
    # معالجة البيانات كما في الكود الأصلي
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
    
    df = insert_ball_carries(df, min_carry_length=3, max_carry_length=100, min_carry_duration=1, max_carry_duration=50)
    
    df = df.reset_index(drop=True)
    df['index'] = range(1, len(df) + 1)
    df = df[['index'] + [col for col in df.columns if col != 'index']]
    
    # Assign xT values
    df_base = df
    dfxT = df_base.copy()
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
    
    dfxT['start_zone_value_xT'] = dfxT[['x1_bin_xT', 'y1_bin_xT']].apply(lambda x: xT[x[1]][x[0]], axis=1)
    dfxT['end_zone_value_xT'] = dfxT[['x2_bin_xT', 'y2_bin_xT']].apply(lambda x: xT[x[1]][x[0]], axis=1)
    
    dfxT['xT'] = dfxT['end_zone_value_xT'] - dfxT['start_zone_value_xT']
    columns_to_drop = ['id', 'eventId', 'minute', 'second', 'teamId', 'x', 'y', 'expandedMinute', 'period', 'outcomeType', 'qualifiers', 'type', 'satisfiedEventsTypes', 'isTouch', 'playerId', 'endX', 'endY', 
                       'relatedEventId', 'relatedPlayerId', 'blockedX', 'blockedY', 'goalMouthZ', 'goalMouthY', 'isShot', 'cumulative_mins']
    dfxT.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    df = df.merge(dfxT, on='index', how='left')
    df['teamName'] = df['teamId'].map(teams_dict)
    team_names = list(teams_dict.values())
    opposition_dict = {team_names[i]: team_names[1-i] for i in range(len(team_names))}
    df['oppositionTeamName'] = df['teamName'].map(opposition_dict)
    
    # Reshaping the data from 100x100 to 105x68
    df['x'] = df['x']*1.05
    df['y'] = df['y']*0.68
    df['endX'] = df['endX']*1.05
    df['endY'] = df['endY']*0.68
    df['goalMouthY'] = df['goalMouthY']*0.68
    
    columns_to_drop = ['height', 'weight', 'age', 'isManOfTheMatch', 'field', 'stats', 'subbedInPlayerId', 'subbedOutPeriod', 'subbedOutExpandedMinute', 'subbedInPeriod', 'subbedInExpandedMinute', 'subbedOutPlayerId', 'teamId']
    players_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    df = df.merge(players_df, on='playerId', how='left')
    
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
    df.drop(columns=columns_to_drop2, inplace=True, errors='ignore')
    
    def get_possession_chains(events_df, chain_check, suc_evts_in_chain):
        events_out = pd.DataFrame()
        match_events_df = events_df.reset_index()
        match_pos_events_df = match_events_df[~match_events_df['type'].isin(['OffsideGiven', 'CornerAwarded', 'Start', 'Card', 'SubstitutionOff',
                                                                            'SubstitutionOn', 'FormationChange', 'FormationSet', 'End'])].copy()
        match_pos_events_df['outcomeBinary'] = (match_pos_events_df['outcomeType'].apply(lambda x: 1 if x == 'Successful' else 0))
        match_pos_events_df['teamBinary'] = (match_pos_events_df['teamName'].apply(lambda x: 1 if x == min(match_pos_events_df['teamName']) else 0))
        match_pos_events_df['goalBinary'] = ((match_pos_events_df['type'] == 'Goal').astype(int).diff(periods=1).apply(lambda x: 1 if x < 0 else 0))
        
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
        pos_chain_df['kick_off_goal'] = ((match_pos_events_df['type'] == 'Goal').astype(int).diff(periods=1).apply(lambda x: 1 if x < 0 else 0))
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
        match_events_df[['possession_id', 'possession_team']] = match_events_df[['possession_id', 'possession_team']].fillna(method='ffill')
        match_events_df[['possession_id', 'possession_team']] = match_events_df[['possession_id', 'possession_team']].fillna(method='bfill')
        
        events_out = pd.concat([events_out, match_events_df])
        return events_out
    
    df = get_possession_chains(df, 5, 3)
    
    df['period'] = df['period'].replace({1: 'FirstHalf', 2: 'SecondHalf', 3: 'FirstPeriodOfExtraTime', 4: 'SecondPeriodOfExtraTime', 5: 'PenaltyShootout', 14: 'PostGame', 16: 'PreMatch'})
    df = df[df['period'] != 'PenaltyShootout']
    df = df.reset_index(drop=True)
    
    return df, teams_dict, players_df

# دالة لحساب إحصائيات الحراس
def calculate_gk_stats(df, players_df, hteamName, ateamName):
    gk_stats = {}
    
    # تصفية الحراس
    gk_home = players_df[(players_df['teamId'] == df[df['teamName'] == hteamName]['teamId'].iloc[0]) & (players_df['position'] == 'GK')]
    gk_away = players_df[(players_df['teamId'] == df[df['teamName'] == ateamName]['teamId'].iloc[0]) & (players_df['position'] == 'GK')]
    
    for gk in gk_home['playerId']:
        gk_name = players_df[players_df['playerId'] == gk]['name'].iloc[0]
        saves = len(df[(df['playerId'] == gk) & (df['type'] == 'Save')])
        passes = len(df[(df['playerId'] == gk) & (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful')])
        goals_conceded = len(df[(df['teamName'] != hteamName) & (df['type'] == 'Goal') & (~df['qualifiers'].str.contains('OwnGoal'))])
        gk_stats[f"{gk_name} ({hteamName})"] = {
            'Saves': saves,
            'Successful Passes': passes,
            'Goals Conceded': goals_conceded
        }
    
    for gk in gk_away['playerId']:
        gk_name = players_df[players_df['playerId'] == gk]['name'].iloc[0]
        saves = len(df[(df['playerId'] == gk) & (df['type'] == 'Save')])
        passes = len(df[(df['playerId'] == gk) & (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful')])
        goals_conceded = len(df[(df['teamName'] != ateamName) & (df['type'] == 'Goal') & (~df['qualifiers'].str.contains('OwnGoal'))])
        gk_stats[f"{gk_name} ({ateamName})"] = {
            'Saves': saves,
            'Successful Passes': passes,
            'Goals Conceded': goals_conceded
        }
    
    return pd.DataFrame(gk_stats).T

# معالجة الرابط المدخل
if match_url:
    match_input = st.button(reshape_arabic_text('تأكيد الرابط'), on_click=lambda: st.session_state.update({'confirmed': True}))
else:
    st.info(reshape_arabic_text('يرجى إدخال رابط المباراة للمتابعة.'))

# معالجة البيانات إذا تم التأكيد
if match_url and st.session_state.confirmed:
    df, teams_dict, players_df = get_event_data(match_url)
    
    if df is None or teams_dict is None or players_df is None:
        st.error(reshape_arabic_text('فشل جلب بيانات المباراة. تأكد من الرابط وحاول مرة أخرى.'))
    else:
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
        
        df_teamNameId = pd.read_csv('https://raw.githubusercontent.com/adnaaan433/Post-Match-Report-2.0/63f5b51d8bd8b3f40e3d02fece1defb2f18ddf54/teams_name_and_id.csv')
        try:
            hftmb_tid = df_teamNameId[df_teamNameId['teamName'] == hteamName].teamId.to_list()[0]
            aftmb_tid = df_teamNameId[df_teamNameId['teamName'] == ateamName].teamId.to_list()[0]
        except IndexError:
            st.warning(reshape_arabic_text('تعذر العثور على معرفات الفرق. قد يكون اسم الفريق غير مطابق.'))
            hftmb_tid = hteamID
            aftmb_tid = ateamID
        
        st.header(f'{hteamName} {hgoal_count} - {agoal_count} {ateamName}')
        
        # إضافة تبويبات التحليلات
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            reshape_arabic_text('شبكة التمريرات'),
            reshape_arabic_text('خريطة التسديدات'),
            reshape_arabic_text('الضغط'),
            reshape_arabic_text('إحصائيات الفريق'),
            reshape_arabic_text('إحصائيات الحراس')
        ])
        
        with tab1:
            st.subheader(reshape_arabic_text('شبكة تمريرات الفريق المضيف'))
            try:
                fig1 = pass_network(df, hteamName, hcol, bg_color, line_color)  # من الكود الأصلي
                st.pyplot(fig1)
            except NameError:
                st.error(reshape_arabic_text('دالة pass_network غير معرفة. يرجى إضافتها.'))
            
            st.subheader(reshape_arabic_text('شبكة تمريرات الفريق الضيف'))
            try:
                fig2 = pass_network(df, ateamName, acol, bg_color, line_color)
                st.pyplot(fig2)
            except NameError:
                st.error(reshape_arabic_text('دالة pass_network غير معرفة. يرجى إضافتها.'))
        
        with tab2:
            st.subheader(reshape_arabic_text('خريطة تسديدات الفريق المضيف'))
            try:
                fig3 = plot_ShotsMap(df, hteamName, hcol, bg_color, line_color, gradient_colors, violet)
                st.pyplot(fig3)
            except NameError:
                st.error(reshape_arabic_text('دالة plot_ShotsMap غير معرفة. يرجى إضافتها.'))
            
            st.subheader(reshape_arabic_text('خريطة تسديدات الفريق الضيف'))
            try:
                fig4 = plot_ShotsMap(df, ateamName, acol, bg_color, line_color, gradient_colors, violet)
                st.pyplot(fig4)
            except NameError:
                st.error(reshape_arabic_text('دالة plot_ShotsMap غير معرفة. يرجى إضافتها.'))
        
        with tab3:
            st.subheader(reshape_arabic_text('خريطة الضغط'))
            try:
                fig5 = plot_pressure(df, hteamName, ateamName, hcol, acol, bg_color, line_color)
                st.pyplot(fig5)
            except NameError:
                st.error(reshape_arabic_text('دالة plot_pressure غير معرفة. يرجى إضافتها.'))
        
        with tab4:
            st.subheader(reshape_arabic_text('إحصائيات الفريق'))
            try:
                team_stats = calculate_team_stats(df, hteamName, ateamName)
                st.table(team_stats)
            except NameError:
                st.error(reshape_arabic_text('دالة calculate_team_stats غير معرفة. يرجى إضافتها.'))
        
        with tab5:
            st.subheader(reshape_arabic_text('إحصائيات الحراس'))
            gk_stats = calculate_gk_stats(df, players_df, hteamName, ateamName)
            st.table(gk_stats)
