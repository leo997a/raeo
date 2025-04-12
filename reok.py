import json
import re
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba, LinearSegmentedColormap
import requests
import matplotlib.patches as patches
from mplsoccer import Pitch, VerticalPitch, add_image
from highlight_text import fig_text, ax_text
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import patheffects, rcParams
from PIL import Image
from urllib.request import urlopen
from unidecode import unidecode
from scipy.spatial import ConvexHull
import streamlit as st
import arabic_reshaper
from bidi.algorithm import get_display
import time

# Selenium & parsing
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup

# --------------------------------------------------------------------
# إعداد matplotlib لدعم العربية
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Amiri', 'Noto Sans Arabic', 'Arial', 'Tahoma']
mpl.rcParams['axes.unicode_minus'] = False

def reshape_arabic_text(text):
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

st.markdown("""
    <style>
    body { direction: rtl; text-align: right; }
    .stSelectbox > div > div > div { text-align: right; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar color pickers
default_hcol, default_acol = '#d00000', '#003087'
default_bg_color = '#1e1e2f'
default_gradient = ['#003087', '#d00000']
st.sidebar.title('اختيار الألوان')
hcol = st.sidebar.color_picker('لون الفريق المضيف', default_hcol)
acol = st.sidebar.color_picker('لون الفريق الضيف', default_acol)
bg_color = st.sidebar.color_picker('لون الخلفية', default_bg_color)
grad_start = st.sidebar.color_picker('بداية التدرج', default_gradient[0])
grad_end   = st.sidebar.color_picker('نهاية التدرج', default_gradient[1])
line_color = st.sidebar.color_picker('لون الخطوط', '#ffffff')
gradient_colors = [grad_start, grad_end]

# Sidebar match URL input
st.sidebar.title('رابط المباراة')
match_url = st.sidebar.text_input('أدخل رابط WhoScored:', 
    placeholder='https://1xbet.whoscored.com/matches/.../live/...')
confirm = st.sidebar.button('تأكيد الرابط')

if 'confirmed' not in st.session_state:
    st.session_state.confirmed = False
if confirm:
    st.session_state.confirmed = True

# --------------------------------------------------------------------
# 1) دالة استخراج JSON من WhoScored
def extract_json_from_url(match_url):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    service = Service()  # عدّل إذا لزم مسار chromedriver
    driver = webdriver.Chrome(service=service, options=options)
    try:
        driver.get(match_url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        script = soup.find('script', string=lambda s: s and 'matchCentreData' in s)
        if not script:
            return None
        txt = script.string
        prefix = 'matchCentreData: '
        start = txt.find(prefix) + len(prefix)
        end   = txt.find(',\n', start)
        return json.loads(txt[start:end])
    finally:
        driver.quit()

# 2) دوال تفكيك البيانات
def extract_data_from_dict(data):
    mc = data['matchCentreData']
    events = mc['events']
    teams_dict = {
        mc['home']['teamId']: mc['home']['name'],
        mc['away']['teamId']: mc['away']['name']
    }
    ph = pd.DataFrame(mc['home']['players'])
    ph['teamId'] = mc['home']['teamId']
    pa = pd.DataFrame(mc['away']['players'])
    pa['teamId'] = mc['away']['teamId']
    players_df = pd.concat([ph, pa])
    players_df['name'] = players_df['name'].astype(str).apply(unidecode)
    return events, players_df, teams_dict

        def cumulative_match_mins(events_df):
            events_out = pd.DataFrame()
            # Add cumulative time to events data, resetting for each unique match
            match_events = events_df.copy()
            match_events['cumulative_mins'] = match_events['minute'] + (1/60) * match_events['second']
            # Add time increment to cumulative minutes based on period of game.
            for period in np.arange(1, match_events['period'].max() + 1, 1):
                if period > 1:
                    t_delta = match_events[match_events['period'] == period - 1]['cumulative_mins'].max() - \
                                           match_events[match_events['period'] == period]['cumulative_mins'].min()
                elif period == 1 or period == 5:
                    t_delta = 0
                else:
                    t_delta = 0
                match_events.loc[match_events['period'] == period, 'cumulative_mins'] += t_delta
            # Rebuild events dataframe
            events_out = pd.concat([events_out, match_events])
            return events_out
        
        df = cumulative_match_mins(df)
        
        def insert_ball_carries(events_df, min_carry_length=3, max_carry_length=100, min_carry_duration=1, max_carry_duration=50):
            events_out = pd.DataFrame()
            # Carry conditions (convert from metres to opta)
            min_carry_length = 3.0
            max_carry_length = 100.0
            min_carry_duration = 1.0
            max_carry_duration = 50.0
            # match_events = events_df[events_df['match_id'] == match_id].reset_index()
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
                              or (next_evt['type'] == 'Card')
                             ):
                            incorrect_next_evt = True
        
                        else:
                            incorrect_next_evt = False
        
                        next_evt_idx += 1
        
                    # Apply some conditioning to determine whether carry criteria is satisfied
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
        
                    valid_carry = same_team & not_ball_touch & far_enough & not_too_far & min_time & same_phase &same_period
        
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
        
            # Rebuild events dataframe
            events_out = pd.concat([events_out, match_events_and_carries])
        
            return events_out
        
        df = insert_ball_carries(df, min_carry_length=3, max_carry_length=100, min_carry_duration=1, max_carry_duration=50)
        
        df = df.reset_index(drop=True)
        df['index'] = range(1, len(df) + 1)
        df = df[['index'] + [col for col in df.columns if col != 'index']]
        
        # Assign xT values
        df_base  = df
        dfxT = df_base.copy()
        dfxT['qualifiers'] = dfxT['qualifiers'].astype(str)
        dfxT = dfxT[(~dfxT['qualifiers'].str.contains('Corner'))]
        dfxT = dfxT[(dfxT['type'].isin(['Pass', 'Carry'])) & (dfxT['outcomeType']=='Successful')]
        
        
        # xT = pd.read_csv('https://raw.githubusercontent.com/mckayjohns/youtube-videos/main/data/xT_Grid.csv', header=None)
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
        columns_to_drop = ['id', 'eventId', 'minute', 'second', 'teamId', 'x', 'y', 'expandedMinute', 'period', 'outcomeType', 'qualifiers',  'type', 'satisfiedEventsTypes', 'isTouch', 'playerId', 'endX', 'endY', 
                           'relatedEventId', 'relatedPlayerId', 'blockedX', 'blockedY', 'goalMouthZ', 'goalMouthY', 'isShot', 'cumulative_mins']
        dfxT.drop(columns=columns_to_drop, inplace=True)
        
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
        dfp.drop(columns=columns_to_drop, inplace=True)
        df = df.merge(dfp, on='playerId', how='left')
        
        df['qualifiers'] = df['qualifiers'].astype(str)
        # Calculating passing distance, to find out progressive pass
        df['prog_pass'] = np.where((df['type'] == 'Pass'), 
                                   np.sqrt((105 - df['x'])**2 + (34 - df['y'])**2) - np.sqrt((105 - df['endX'])**2 + (34 - df['endY'])**2), 0)
        # Calculating carrying distance, to find out progressive carry
        df['prog_carry'] = np.where((df['type'] == 'Carry'), 
                                    np.sqrt((105 - df['x'])**2 + (34 - df['y'])**2) - np.sqrt((105 - df['endX'])**2 + (34 - df['endY'])**2), 0)
        df['pass_or_carry_angle'] = np.degrees(np.arctan2(df['endY'] - df['y'], df['endX'] - df['x']))
        
        df['name'] = df['name'].astype(str)
        df['name'] = df['name'].apply(unidecode)
        # Function to extract short names
        def get_short_name(full_name):
            if pd.isna(full_name):
                return full_name
            parts = full_name.split()
            if len(parts) == 1:
                return full_name  # No need for short name if there's only one word
            elif len(parts) == 2:
                return parts[0][0] + ". " + parts[1]
            else:
                return parts[0][0] + ". " + parts[1][0] + ". " + " ".join(parts[2:])
        
        # Applying the function to create 'shortName' column
        df['shortName'] = df['name'].apply(get_short_name)
        
        df['qualifiers'] = df['qualifiers'].astype(str)
        columns_to_drop2 = ['id']
        df.drop(columns=columns_to_drop2, inplace=True)
        
        def get_possession_chains(events_df, chain_check, suc_evts_in_chain):
            # Initialise output
            events_out = pd.DataFrame()
            match_events_df = df.reset_index()
        
            # Isolate valid event types that contribute to possession
            match_pos_events_df = match_events_df[~match_events_df['type'].isin(['OffsideGiven', 'CornerAwarded','Start', 'Card', 'SubstitutionOff',
                                                                                          'SubstitutionOn', 'FormationChange','FormationSet', 'End'])].copy()
        
            # Add temporary binary outcome and team identifiers
            match_pos_events_df['outcomeBinary'] = (match_pos_events_df['outcomeType']
                                                        .apply(lambda x: 1 if x == 'Successful' else 0))
            match_pos_events_df['teamBinary'] = (match_pos_events_df['teamName']
                                 .apply(lambda x: 1 if x == min(match_pos_events_df['teamName']) else 0))
            match_pos_events_df['goalBinary'] = ((match_pos_events_df['type'] == 'Goal')
                                 .astype(int).diff(periods=1).apply(lambda x: 1 if x < 0 else 0))
        
            # Create a dataframe to investigate possessions chains
            pos_chain_df = pd.DataFrame()
        
            # Check whether each event is completed by same team as the next (check_evts-1) events
            for n in np.arange(1, chain_check):
                pos_chain_df[f'evt_{n}_same_team'] = abs(match_pos_events_df['teamBinary'].diff(periods=-n))
                pos_chain_df[f'evt_{n}_same_team'] = pos_chain_df[f'evt_{n}_same_team'].apply(lambda x: 1 if x > 1 else x)
            pos_chain_df['enough_evt_same_team'] = pos_chain_df.sum(axis=1).apply(lambda x: 1 if x < chain_check - suc_evts_in_chain else 0)
            pos_chain_df['enough_evt_same_team'] = pos_chain_df['enough_evt_same_team'].diff(periods=1)
            pos_chain_df[pos_chain_df['enough_evt_same_team'] < 0] = 0
        
            match_pos_events_df['period'] = pd.to_numeric(match_pos_events_df['period'], errors='coerce')
            # Check there are no kick-offs in the upcoming (check_evts-1) events
            pos_chain_df['upcoming_ko'] = 0
            for ko in match_pos_events_df[(match_pos_events_df['goalBinary'] == 1) | (match_pos_events_df['period'].diff(periods=1))].index.values:
                ko_pos = match_pos_events_df.index.to_list().index(ko)
                pos_chain_df.iloc[ko_pos - suc_evts_in_chain:ko_pos, 5] = 1
        
            # Determine valid possession starts based on event team and upcoming kick-offs
            pos_chain_df['valid_pos_start'] = (pos_chain_df.fillna(0)['enough_evt_same_team'] - pos_chain_df.fillna(0)['upcoming_ko'])
        
            # Add in possession starts due to kick-offs (period changes and goals).
            pos_chain_df['kick_off_period_change'] = match_pos_events_df['period'].diff(periods=1)
            pos_chain_df['kick_off_goal'] = ((match_pos_events_df['type'] == 'Goal')
                             .astype(int).diff(periods=1).apply(lambda x: 1 if x < 0 else 0))
            pos_chain_df.loc[pos_chain_df['kick_off_period_change'] == 1, 'valid_pos_start'] = 1
            pos_chain_df.loc[pos_chain_df['kick_off_goal'] == 1, 'valid_pos_start'] = 1
        
            # Add first possession manually
            pos_chain_df['teamName'] = match_pos_events_df['teamName']
            pos_chain_df.loc[pos_chain_df.head(1).index, 'valid_pos_start'] = 1
            pos_chain_df.loc[pos_chain_df.head(1).index, 'possession_id'] = 1
            pos_chain_df.loc[pos_chain_df.head(1).index, 'possession_team'] = pos_chain_df.loc[pos_chain_df.head(1).index, 'teamName']
        
            # Iterate through valid possession starts and assign them possession ids
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
        
            # Assign possession id and team back to events dataframe
            match_events_df = pd.merge(match_events_df, pos_chain_df[['possession_id', 'possession_team']], how='left', left_index=True, right_index=True)
        
            # Fill in possession ids and possession team
            match_events_df[['possession_id', 'possession_team']] = (match_events_df[['possession_id', 'possession_team']].fillna(method='ffill'))
            match_events_df[['possession_id', 'possession_team']] = (match_events_df[['possession_id', 'possession_team']].fillna(method='bfill'))
        
            # Rebuild events dataframe
            events_out = pd.concat([events_out, match_events_df])
        
            return events_out
        
        df = get_possession_chains(df, 5, 3)
        
        df['period'] = df['period'].replace({1: 'FirstHalf', 2: 'SecondHalf', 3: 'FirstPeriodOfExtraTime', 4: 'SecondPeriodOfExtraTime', 5: 'PenaltyShootout', 14: 'PostGame', 16: 'PreMatch'})
        
        df = df[df['period']!='PenaltyShootout']
        df = df.reset_index(drop=True)
        return df, teams_dict, players_df
    
    df, teams_dict, players_df = get_event_data(season, league, stage, htn, atn)
    
    def get_short_name(full_name):
        if pd.isna(full_name):
            return full_name
        parts = full_name.split()
        if len(parts) == 1:
            return full_name  # No need for short name if there's only one word
        elif len(parts) == 2:
            return parts[0][0] + ". " + parts[1]
        else:
            return parts[0][0] + ". " + parts[1][0] + ". " + " ".join(parts[2:])
    
    hteamID = list(teams_dict.keys())[0]  # selected home team
    ateamID = list(teams_dict.keys())[1]  # selected away team
    hteamName= teams_dict[hteamID]
    ateamName= teams_dict[ateamID]
    
    homedf = df[(df['teamName']==hteamName)]
    awaydf = df[(df['teamName']==ateamName)]
    hxT = homedf['xT'].sum().round(2)
    axT = awaydf['xT'].sum().round(2)
    # الألوان مأخوذة مباشرة من اختيارات المستخدم في الشريط الجانبي
    # لا حاجة لتعيين hcol و acol هنا لأنهما مُعرفتان في الشريط الجانبي
    
    hgoal_count = len(homedf[(homedf['teamName']==hteamName) & (homedf['type']=='Goal') & (~homedf['qualifiers'].str.contains('OwnGoal'))])
    agoal_count = len(awaydf[(awaydf['teamName']==ateamName) & (awaydf['type']=='Goal') & (~awaydf['qualifiers'].str.contains('OwnGoal'))])
    hgoal_count = hgoal_count + len(awaydf[(awaydf['teamName']==ateamName) & (awaydf['type']=='Goal') & (awaydf['qualifiers'].str.contains('OwnGoal'))])
    agoal_count = agoal_count + len(homedf[(homedf['teamName']==hteamName) & (homedf['type']=='Goal') & (homedf['qualifiers'].str.contains('OwnGoal'))])
    
    df_teamNameId = pd.read_csv('https://raw.githubusercontent.com/adnaaan433/Post-Match-Report-2.0/63f5b51d8bd8b3f40e3d02fece1defb2f18ddf54/teams_name_and_id.csv')
    hftmb_tid = df_teamNameId[df_teamNameId['teamName']==hteamName].teamId.to_list()[0]
    aftmb_tid = df_teamNameId[df_teamNameId['teamName']==ateamName].teamId.to_list()[0]
    
    st.header(f'{hteamName} {hgoal_count} - {agoal_count} {ateamName}')
    st.text(f'{league}')
    
# دالة pass_network المعدلة
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
    if len(total_pass) != 0:
        accuracy = round((len(accrt_pass) / len(total_pass)) * 100, 2)
    else:
        accuracy = 0

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

    # إعدادات التصميم العصري
    MAX_LINE_WIDTH = 8  # الحد الأقصى لعرض الخطوط
    MIN_LINE_WIDTH = 0.5  # الحد الأدنى لعرض الخطوط
    MIN_TRANSPARENCY = 0.2  # الحد الأدنى للشفافية
    MAX_TRANSPARENCY = 0.9  # الحد الأقصى للشفافية

    # حساب عرض الخطوط بناءً على عدد التمريرات
    pass_counts_df['line_width'] = (pass_counts_df['pass_count'] / pass_counts_df['pass_count'].max()) * (MAX_LINE_WIDTH - MIN_LINE_WIDTH) + MIN_LINE_WIDTH

    # حساب الشفافية بناءً على عدد التمريرات
    c_transparency = pass_counts_df['pass_count'] / pass_counts_df['pass_count'].max()
    c_transparency = (c_transparency * (MAX_TRANSPARENCY - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY

    # إنشاء اللون مع الشفافية
    color = np.array(to_rgba(col))
    color = np.tile(color, (len(pass_counts_df), 1))
    color[:, 3] = c_transparency
