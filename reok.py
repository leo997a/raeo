# Import Packages
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
    body {
        direction: rtl;
        text-align: right;
    }
    .stApp {
        direction: rtl;
        text-align: right;
        font-family: 'Amiri', 'Noto Sans Arabic', 'Arial', sans-serif;
    }
    h1, h2, h3, h4, h5, h6, p, div, span, label, button, input, select, option, table, th, td {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Amiri', 'Noto Sans Arabic', 'Arial', sans-serif !important;
    }
    /* تحسين الجداول */
    .stDataFrame, .dataframe {
        direction: rtl !important;
        text-align: right !important;
    }
    th, td {
        direction: rtl !important;
        text-align: right !important;
    }
    /* تحسين القوائم المنسدلة */
    select, .stSelectbox, .stSelectbox div, .stSelectbox label {
        direction: rtl !important;
        text-align: right !important;
    }
    /* تحسين st.radio */
    .stRadio, .stRadio div, .stRadio label, .stRadio input {
        direction: rtl !important;
        text-align: right !important;
    }
    /* تحسين علامات التبويب */
    .stTabs, .stTabs div, .stTabs button {
        direction: rtl !important;
        text-align: right !important;
    }
    .custom-tabs {
        display: flex;
        flex-direction: row-reverse;
        border-bottom: 2px solid #ddd;
        margin-bottom: 20px;
    }
    .custom-tab {
        padding: 10px 20px;
        cursor: pointer;
        font-size: 16px;
        font-weight: bold;
        color: #555;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
    }
    .custom-tab:hover {
        color: #000;
        border-bottom: 3px solid #007bff;
    }
    .custom-tab.active {
        color: #007bff;
        border-bottom: 3px solid #007bff;
    }
    </style>
    """, unsafe_allow_html=True)

# تعريف القيم الافتراضية للألوان أولاً
default_hcol = '#d00000'  # لون الفريق المضيف الافتراضي
default_acol = '#003087'  # لون الفريق الضيف الافتراضي
default_bg_color = '#1e1e2f'  # لون الخلفية الافتراضي
default_gradient_colors = ['#003087', '#d00000']  # ألوان التدرج الافتراضية

# إضافة أدوات اختيار الألوان في الشريط الجانبي
st.sidebar.title('اختيار الألوان')
hcol = st.sidebar.color_picker('لون الفريق المضيف', default_hcol, key='hcol_picker')
acol = st.sidebar.color_picker('لون الفريق الضيف', default_acol, key='acol_picker')
bg_color = st.sidebar.color_picker('لون الخلفية', default_bg_color, key='bg_color_picker')
gradient_start = st.sidebar.color_picker('بداية التدرج', default_gradient_colors[0], key='gradient_start_picker')
gradient_end = st.sidebar.color_picker('نهاية التدرج', default_gradient_colors[1], key='gradient_end_picker')
gradient_colors = [gradient_start, gradient_end]  # تحديث قائمة ألوان التدرج
line_color = st.sidebar.color_picker('لون الخطوط', '#ffffff', key='line_color_picker')  # اختياري

st.sidebar.title('Match Selection')

season = None
league = None
stage = None
htn = None
atn = None

# Set up session state for selected values
if 'confirmed' not in st.session_state:
    st.session_state.confirmed = False

def reset_confirmed():
    st.session_state['confirmed'] = False

season = st.sidebar.selectbox('Select a season:', ['2024_25'], key='season', index=0, on_change=reset_confirmed)
if season:
    league = st.sidebar.selectbox('Select a League', ['La Liga', 'Premier League', 'Serie A', 'UEFA Champions League'], key='league', index=None, on_change=reset_confirmed)

    if league == 'La Liga':
        team_list = ['Athletic Club', 'Atletico Madrid', 'Barcelona', 'Celta Vigo', 'Deportivo Alaves', 'Espanyol', 'Getafe', 'Girona', 'Las Palmas', 'Leganes', 'Mallorca', 'Osasuna', 'Rayo Vallecano', 'Real Betis', 
                     'Real Madrid', 'Real Sociedad', 'Real Valladolid', 'Sevilla', 'Valencia', 'Villarreal']
    elif league == 'Premier League':
        team_list = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich', 'Leicester', 'Liverpool', 'Manchester City', 'Manchester United', 'Newcastle',
                     'Nottingham Forest', 'Southampton', 'Tottenham', 'West Ham', 'Wolves']
    elif league == 'Serie A':
        team_list = ['AC Milan', 'Atalanta', 'Bologna', 'Cagliari', 'Como', 'Empoli', 'Fiorentina', 'Genoa', 'Inter', 'Juventus', 'Lazio', 'Lecce', 'Monza', 'Napoli', 'Parma Calcio',
                     'Roma', 'Torino', 'Udinese', 'Venezia', 'Verona']
    elif league == 'UEFA Champions League':
        team_list = ['AC Milan', 'Arsenal', 'Aston Villa', 'Atalanta', 'Atletico Madrid', 'BSC Young Boys', 'Barcelona', 'Bayer Leverkusen', 'Bayern Munich', 'Benfica', 'Bologna', 'Borussia Dortmund', 'Brest', 'Celtic',
                     'Club Brugge', 'Dinamo Zagreb', 'FK Crvena Zvezda', 'Feyenoord', 'Girona', 'Inter', 'Juventus', 'Lille', 'Liverpool', 'Manchester City', 'Monaco', 'PSV Eindhoven', 'Paris Saint-Germain', 'RB Leipzig',
                     'Real Madrid', 'Salzburg', 'Shakhtar Donetsk', 'Slovan Bratislava', 'Sparta Prague', 'Sporting CP', 'Sturm Graz', 'VfB Stuttgart']

    if league and league != 'UEFA Champions League':
        htn = st.sidebar.selectbox('Select Home Team', team_list, key='home_team', index=None, on_change=reset_confirmed)
        
        if htn:
            atn_options = [team for team in team_list if team != htn]
            atn = st.sidebar.selectbox('Select Away Team Name', atn_options, key='away_team', index=None, on_change=reset_confirmed)
            
    elif league == 'UEFA Champions League':
        stage = st.sidebar.selectbox('Select Stage', ['League Phase', 'Knockout Playoff', 'Round of 16', 'Quarter Final', 'Quarter Final', 'Final'], key='stage_selection', index=None, on_change=reset_confirmed)
        if stage:
            htn = st.sidebar.selectbox('Select Home Team', team_list, key='home_team', index=None, on_change=reset_confirmed)
            
            if htn:
                atn_options = [team for team in team_list if team != htn]
                atn = st.sidebar.selectbox('Select Away Team Name', atn_options, key='away_team', index=None, on_change=reset_confirmed)

    if league and league != 'UEFA Champions League' and league != 'Serie A' and htn and atn:
        league = league.replace(' ', '_')
        match_html_path = f"https://raw.githubusercontent.com/leo997a/{season}_{league}/refs/heads/main/{htn}_vs_{atn}.html"
        match_html_path = match_html_path.replace(' ', '%20')
        try:
            response = requests.get(match_html_path)
            response.raise_for_status()  # Raise an error for invalid responses (e.g., 404, 500)
            # Only show the button if the response is successful
            match_input = st.sidebar.button('Confirm Selections', on_click=lambda: st.session_state.update({'confirmed': True}))
        except:
            st.session_state['confirmed'] = False
            st.sidebar.write('Match not found')
            
    elif league and league == 'Serie A' and htn and atn:
        league = league.replace(' ', '_')
        match_html_path = f"https://raw.githubusercontent.com/leo997a/{season}_{league}/refs/heads/main/{htn}_vs_{atn}.html"
        match_html_path = match_html_path.replace(' ', '%20')
        try:
            response = requests.get(match_html_path)
            response.raise_for_status()  # Raise an error for invalid responses (e.g., 404, 500)
            # Only show the button if the response is successful
            match_input = st.sidebar.button('Confirm Selections', on_click=lambda: st.session_state.update({'confirmed': True}))
        except:
            st.session_state['confirmed'] = False
            st.sidebar.write('Serie A Matches available till GameWeek 12\nRemaining data will be uploaded soon\nThanks for your patience')
            
    elif league and league == 'UEFA Champions League' and stage and htn and atn:
        league = league.replace(' ', '_')
        match_html_path = f"https://raw.githubusercontent.com/leo997a/{season}_{league}/refs/heads/main/{stage}/{htn}_vs_{atn}.html"
        match_html_path = match_html_path.replace(' ', '%20')
        try:
            response = requests.get(match_html_path)
            response.raise_for_status()  # Raise an error for invalid responses (e.g., 404, 500)
            # Only show the button if the response is successful
            match_input = st.sidebar.button('Confirm Selections', on_click=lambda: st.session_state.update({'confirmed': True}))
        except:
            st.session_state['confirmed'] = False
            st.sidebar.write('Match not found')

if league and htn and atn and st.session_state.confirmed:
    @st.cache_data
    def get_event_data(season, league, stage, htn, atn):
        
        def extract_json_from_html(html_path, save_output=False):
            response = requests.get(html_path)
            response.raise_for_status()  # Ensure the request was successful
            html = response.text
        
            regex_pattern = r'(?<=require\.config\.params\["args"\].=.)[\s\S]*?;'
            data_txt = re.findall(regex_pattern, html)[0]
        
            # add quotations for JSON parser
            data_txt = data_txt.replace('matchId', '"matchId"')
            data_txt = data_txt.replace('matchCentreData', '"matchCentreData"')
            data_txt = data_txt.replace('matchCentreEventTypeJson', '"matchCentreEventTypeJson"')
            data_txt = data_txt.replace('formationIdNameMappings', '"formationIdNameMappings"')
            data_txt = data_txt.replace('};', '}')
        
            if save_output:
                # save JSON data to txt
                output_file = open(f"{html_path}.txt", "wt", encoding='utf-8')
                n = output_file.write(data_txt)
                output_file.close()
        
            return data_txt
        
        def extract_data_from_dict(data):
            # load data from json
            event_types_json = data["matchCentreEventTypeJson"]
            formation_mappings = data["formationIdNameMappings"]
            events_dict = data["matchCentreData"]["events"]
            teams_dict = {data["matchCentreData"]['home']['teamId']: data["matchCentreData"]['home']['name'],
                          data["matchCentreData"]['away']['teamId']: data["matchCentreData"]['away']['name']}
            players_dict = data["matchCentreData"]["playerIdNameDictionary"]
            # create players dataframe
            players_home_df = pd.DataFrame(data["matchCentreData"]['home']['players'])
            players_home_df["teamId"] = data["matchCentreData"]['home']['teamId']
            players_away_df = pd.DataFrame(data["matchCentreData"]['away']['players'])
            players_away_df["teamId"] = data["matchCentreData"]['away']['teamId']
            players_df = pd.concat([players_home_df, players_away_df])
            players_df['name'] = players_df['name'].astype(str)
            players_df['name'] = players_df['name'].apply(unidecode)
            players_ids = data["matchCentreData"]["playerIdNameDictionary"]
            
            return events_dict, players_df, teams_dict
        
        json_data_txt = extract_json_from_html(match_html_path)
        data = json.loads(json_data_txt)
        events_dict, players_df, teams_dict = extract_data_from_dict(data)
        
        df = pd.DataFrame(events_dict)
        dfp = pd.DataFrame(players_df)
        
        # Extract the 'displayName' value
        df['type'] = df['type'].astype(str)
        df['outcomeType'] = df['outcomeType'].astype(str)
        df['period'] = df['period'].astype(str)
        df['type'] = df['type'].str.extract(r"'displayName': '([^']+)")
        df['outcomeType'] = df['outcomeType'].str.extract(r"'displayName': '([^']+)")
        df['period'] = df['period'].str.extract(r"'displayName': '([^']+)")
        
        # temprary use of typeId of period column
        df['period'] = df['period'].replace({'FirstHalf': 1, 'SecondHalf': 2, 'FirstPeriodOfExtraTime': 3, 'SecondPeriodOfExtraTime': 4, 'PenaltyShootout': 5, 'PostGame': 14, 'PreMatch': 16})
        
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
    
    hgoal_count = len(homedf[(homedf['teamName']==hteamName) & (homedf['type']=='Goal') & (~homedf['qualifiers'].str.contains('OwnGoal'))])
    agoal_count = len(awaydf[(awaydf['teamName']==ateamName) & (awaydf['type']=='Goal') & (~awaydf['qualifiers'].str.contains('OwnGoal'))])
    hgoal_count = hgoal_count + len(awaydf[(awaydf['teamName']==ateamName) & (awaydf['type']=='Goal') & (awaydf['qualifiers'].str.contains('OwnGoal'))])
    agoal_count = agoal_count + len(homedf[(homedf['teamName']==hteamName) & (homedf['type']=='Goal') & (homedf['qualifiers'].str.contains('OwnGoal'))])
    
    df_teamNameId = pd.read_csv('https://raw.githubusercontent.com/adnaaan433/pmr_app/refs/heads/main/teams_name_and_id.csv')
    hftmb_tid = df_teamNameId[df_teamNameId['teamName']==hteamName].teamId.to_list()[0]
    aftmb_tid = df_teamNameId[df_teamNameId['teamName']==ateamName].teamId.to_list()[0]
    
    st.header(reshape_arabic_text(f'{hteamName} {hgoal_count} - {agoal_count} {ateamName}'))
    st.text(reshape_arabic_text(f'{league}'))
    
    # إنشاء علامات تبويب مخصصة باستخدام HTML
    tabs_html = f"""
    <div class="custom-tabs" id="custom-tabs">
        <div class="custom-tab active" onclick="showTab('tab1')">{reshape_arabic_text('تحليل المباراة')}</div>
        <div class="custom-tab" onclick="showTab('tab2')">{reshape_arabic_text('تبويب آخر')}</div>
    </div>

    <div id="tab1" class="tab-content">
        <!-- سيتم ملء هذا المحتوى ديناميكيًا -->
    </div>
    <div id="tab2" class="tab-content" style="display: none;">
        <h3>{reshape_arabic_text('هذا التبويب قيد التطوير')}</h3>
    </div>

    <script>
    function showTab(tabId) {{
        // إخفاء جميع المحتويات
        document.querySelectorAll('.tab-content').forEach(tab => {{
            tab.style.display = 'none';
        }});
        // إزالة الفئة active من جميع علامات التبويب
        document.querySelectorAll('.custom-tab').forEach(tab => {{
            tab.classList.remove('active');
        }});
        // إظهار المحتوى المحدد وإضافة الفئة active إلى علامة التبويب المحددة
        document.getElementById(tabId).style.display = 'block';
        document.querySelector(`[onclick="showTab('${{tabId}}')"]`).classList.add('active');
    }}
    </script>
    """
    st.markdown(tabs_html, unsafe_allow_html=True)

    # إنشاء قائمة منسدلة مخصصة باستخدام HTML
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

    # تحويل الخيارات إلى نصوص عربية معالجة
    options_display = [reshape_arabic_text(opt) for opt in options]

    # استخدام st.selectbox بدلاً من HTML مخصص
    an_tp = st.selectbox(reshape_arabic_text('نوع التحليل:'), options_display, index=0)

    # تحديث st.session_state
    st.session_state['analysis_type'] = an_tp


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
    
        # التحقق من وجود بيانات
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

    # ... (باقي الكود كما هو)
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

        # إنشاء ملعب بتصميم متدرج
        pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, linewidth=1.5, line_color=line_color)
        pitch.draw(ax=ax)

        # تطبيق خلفية متدرجة
        gradient = LinearSegmentedColormap.from_list("pitch_gradient", gradient_colors, N=100)
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = Y
        ax.imshow(Z, extent=[0, 68, 0, 105], cmap=gradient, alpha=0.8, aspect='auto', zorder=0)
        pitch.draw(ax=ax)

        # رسم الخطوط بين اللاعبين مع عرض متغير وشفافية متغيرة
        for idx in range(len(pass_counts_df)):
            pitch.lines(
                pass_counts_df['pass_avg_x'].iloc[idx],
                pass_counts_df['pass_avg_y'].iloc[idx],
                pass_counts_df['receiver_avg_x'].iloc[idx],
                pass_counts_df['receiver_avg_y'].iloc[idx],
                lw=pass_counts_df['line_width'].iloc[idx],  # عرض الخط بناءً على عدد التمريرات
                color=color[idx],  # اللون مع الشفافية
                zorder=1,
                ax=ax
            )

        # رسم دوائر اللاعبين
        for index, row in avg_locs_df.iterrows():
            if row['isFirstEleven'] == True:
                pitch.scatter(row['avg_x'], row['avg_y'], s=800, marker='o', color=col, edgecolor=line_color, linewidth=1.5, alpha=0.9, ax=ax)
            else:
                pitch.scatter(row['avg_x'], row['avg_y'], s=800, marker='s', color=col, edgecolor=line_color, linewidth=1.5, alpha=0.7, ax=ax)

        # كتابة أرقام القمصان
        for index, row in avg_locs_df.iterrows():
            player_initials = row["shirtNo"]
            pitch.annotate(player_initials, xy=(row.avg_x, row.avg_y), c='white', ha='center', va='center', size=14, weight='bold', ax=ax)

        # خط التماسك العمودي
        avgph = round(avg_locs_df['avg_x'].median(), 2)
        ax.axhline(y=avgph, color='white', linestyle='--', alpha=0.5, linewidth=1.5)

        # ارتفاع خط الدفاع والهجوم
        center_backs_height = avg_locs_df[avg_locs_df['position'] == 'DC']
        def_line_h = round(center_backs_height['avg_x'].median(), 2)
        Forwards_height = avg_locs_df[avg_locs_df['isFirstEleven'] == 1].sort_values(by='avg_x', ascending=False).head(2)
        fwd_line_h = round(Forwards_height['avg_x'].mean(), 2)
        ymid = [0, 0, 68, 68]
        xmid = [def_line_h, fwd_line_h, fwd_line_h, def_line_h]
        ax.fill(ymid, xmid, col, alpha=0.2)

        v_comp = round((1 - ((fwd_line_h - def_line_h) / 105)) * 100, 2)

        # إضافة النصوص مع معالجة العربية وضبط الإحداثيات
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


# الجزء الخارجي من الكود مع معالجة النصوص العربية وضبط الإحداثيات
tab1, tab2 = st.tabs([reshape_arabic_text("تحليل المباراة"), reshape_arabic_text("تبويب آخر")])

with tab1:
    if an_tp == reshape_arabic_text('شبكة التمريرات'):
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
