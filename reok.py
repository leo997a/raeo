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

# تهيئة matplotlib لدعم العربية
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Amiri', 'Noto Sans Arabic', 'Arial', 'Tahoma']
mpl.rcParams['axes.unicode_minus'] = False

# دالة لتحويل النص العربي
def reshape_arabic_text(text):
    print(f"النص قبل التحويل: {text}")
    reshaped_text = arabic_reshaper.reshape(text)
    print(f"النص بعد التحويل: {reshaped_text}")
    return get_display(reshaped_text)

# إضافة CSS لدعم RTL في streamlit
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
# ... (باقي الكود)
    
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
    
    df_teamNameId = pd.read_csv('https://raw.githubusercontent.com/adnaaan433/pmr_app/refs/heads/main/teams_name_and_id.csv')
    hftmb_tid = df_teamNameId[df_teamNameId['teamName']==hteamName].teamId.to_list()[0]
    aftmb_tid = df_teamNameId[df_teamNameId['teamName']==ateamName].teamId.to_list()[0]
    
    st.header(f'{hteamName} {hgoal_count} - {agoal_count} {ateamName}')
    st.text(f'{league}')
    
    tab1, tab2, tab3, tab4 = st.tabs(['تحليل الفريق', 'Player Analysis', 'Match Statistics', 'Top Players'])
    
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
        ax.text(34, 112, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), color='white', fontsize=12, ha='center', va='center')
    elif phase_tag == 'First Half':
        ax.text(34, 115, reshape_arabic_text('الشوط الأول: 0-45 دقيقة'), color='white', fontsize=14, ha='center', va='center', weight='bold')
        ax.text(34, 112, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), color='white', fontsize=12, ha='center', va='center')
    elif phase_tag == 'Second Half':
        ax.text(34, 115, reshape_arabic_text('الشوط الثاني: 45-90 دقيقة'), color='white', fontsize=14, ha='center', va='center', weight='bold')
        ax.text(34, 112, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), color='white', fontsize=12, ha='center', va='center')

    ax.text(34, -6, reshape_arabic_text(f"على الكرة\nالتماسك العمودي (المنطقة المظللة): {v_comp}%"), color='white', fontsize=12, ha='center', va='center', weight='bold')

    return pass_btn

# الجزء الخارجي من الكود مع معالجة النصوص العربية وضبط الإحداثيات
tab1, tab2, tab3, tab4 = st.tabs(['Team Analysis', 'Player Analysis', 'Match Statistics', 'Top Players'])

with tab1:
    an_tp = st.selectbox('نوع التحليل:', [
        'شبكة التمريرات', 
        'Defensive Actions Heatmap', 
        'Progressive Passes', 
        'Progressive Carries', 
        'Shotmap', 
        'احصائيات الحراس', 
        'Match Momentum',
        reshape_arabic_text('Zone14 & Half-Space Passes'), 
        reshape_arabic_text('Final Third Entries'), 
        reshape_arabic_text('Box Entries'), 
        reshape_arabic_text('High-Turnovers'), 
        reshape_arabic_text('Chances Creating Zones'), 
        reshape_arabic_text('Crosses'), 
        reshape_arabic_text('Team Domination Zones'), 
        reshape_arabic_text('Pass Target Zones'),
        'Attacking Thirds'  # خيار جديد
    ], index=0, key='analysis_type_tab1')
    
    if an_tp == 'شبكة التمريرات':
        st.header('شبكة التمريرات')
        
        # استبدال st.pills بـ st.radio لأن st.pills غير مدعوم افتراضيًا في Streamlit
        pn_time_phase = st.radio(" ", ['Full Time', 'First Half', 'Second Half'], index=0, key='pn_time_pill')

        fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
        home_pass_btn = None
        away_pass_btn = None

        if pn_time_phase == 'Full Time':
            home_pass_btn = pass_network(axs[0], hteamName, hcol, 'Full Time')
            away_pass_btn = pass_network(axs[1], ateamName, acol, 'Full Time')
        elif pn_time_phase == 'First Half':
            home_pass_btn = pass_network(axs[0], hteamName, hcol, 'First Half')
            away_pass_btn = pass_network(axs[1], ateamName, acol, 'First Half')
        elif pn_time_phase == 'Second Half':
            home_pass_btn = pass_network(axs[0], hteamName, hcol, 'Second Half')
            away_pass_btn = pass_network(axs[1], ateamName, acol, 'Second Half')

        # معالجة العنوان
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
         bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=2),
         path_effects=[patheffects.withStroke(linewidth=2, foreground='white')])

        # ضبط النصوص في الأسفل مع تطبيق reshape_arabic_text
        fig.text(0.5, 0.02, reshape_arabic_text('*الدوائر = اللاعبون الأساسيون، المربعات = اللاعبون البدلاء، الأرقام داخلها = أرقام القمصان'),
                 fontsize=10, fontstyle='italic', ha='center', va='center', color='white')
        fig.text(0.5, 0.00, reshape_arabic_text('*عرض وإضاءة الخطوط تمثل عدد التمريرات الناجحة في اللعب المفتوح بين اللاعبين'),
                 fontsize=10, fontstyle='italic', ha='center', va='center', color='white')

        # إضافة الصور
        himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png")
        himage = Image.open(himage)
        ax_himage = add_image(himage, fig, left=0.085, bottom=0.97, width=0.125, height=0.125)

        aimage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png")
        aimage = Image.open(aimage)
        ax_aimage = add_image(aimage, fig, left=0.815, bottom=0.97, width=0.125, height=0.125)

        # ضبط المساحات العلوية والسفلية للرسم
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
        ax.text(34, 112, 'الوقت الكامل: 0-90 دقيقة', color=col, fontsize=15, ha='center', va='center')
        ax.text(34, 108, f'إجمالي الأفعال الدفاعية: {len(total_def_acts)}', color=col, fontsize=15, ha='center', va='center')
    elif phase_tag == 'First Half':
        ax.text(34, 112, 'الشوط الأول: 0-45 دقيقة', color=col, fontsize=15, ha='center', va='center')
        ax.text(34, 108, f'إجمالي الأفعال الدفاعية: {len(total_def_acts)}', color=col, fontsize=16, ha='center', va='center')
    elif phase_tag == 'Second Half':
        ax.text(34, 112, 'الشوط الثاني: 45-90 دقيقة', color=col, fontsize=15, ha='center', va='center')
        ax.text(34, 108, f'إجمالي الأفعال الدفاعية: {len(total_def_acts)}', color=col, fontsize=16, ha='center', va='center')

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
        fig.text(0.5, 0.97, '@REO_SHOW', fontsize=15, ha='center', va='center')

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
        st.header(reshape_arabic_text('خريطة التسديدات'))
        st.header(f'{an_tp}')

    def plot_ShotsMap(ax, team_name, col, phase_tag):
        if phase_tag == 'Full Time':
            shots_df = df[(df['teamName'] == team_name) & (df['type'].isin(['Goal', 'MissedShots', 'SavedShot', 'ShotOnPost'])) & (~df['qualifiers'].str.contains('OwnGoal'))]
        elif phase_tag == 'First Half':
            shots_df = df[(df['teamName'] == team_name) & (df['type'].isin(['Goal', 'MissedShots', 'SavedShot', 'ShotOnPost'])) & (~df['qualifiers'].str.contains('OwnGoal')) &
                          (df['period'] == 'FirstHalf')]
        elif phase_tag == 'Second Half':
            shots_df = df[(df['teamName'] == team_name) & (df['type'].isin(['Goal', 'MissedShots', 'SavedShot', 'ShotOnPost'])) & (~df['qualifiers'].str.contains('OwnGoal')) &
                          (df['period'] == 'SecondHalf')]

        goal = shots_df[(shots_df['type'] == 'Goal') & (~shots_df['qualifiers'].str.contains('BigChance')) & (~shots_df['qualifiers'].str.contains('OwnGoal'))]
        goal_bc = shots_df[(shots_df['type'] == 'Goal') & (shots_df['qualifiers'].str.contains('BigChance')) & (~shots_df['qualifiers'].str.contains('OwnGoal'))]
        miss = shots_df[(shots_df['type'] == 'MissedShots') & (~shots_df['qualifiers'].str.contains('BigChance'))]
        miss_bc = shots_df[(shots_df['type'] == 'MissedShots') & (shots_df['qualifiers'].str.contains('BigChance'))]
        ontr = shots_df[(shots_df['type'] == 'SavedShot') & (~shots_df['qualifiers'].str.contains(': 82,')) & 
                        (~shots_df['qualifiers'].str.contains('BigChance'))]
        ontr_bc = shots_df[(shots_df['type'] == 'SavedShot') & (~shots_df['qualifiers'].str.contains(': 82,')) & 
                        (shots_df['qualifiers'].str.contains('BigChance'))]
        blkd = shots_df[(shots_df['type'] == 'SavedShot') & (shots_df['qualifiers'].str.contains(': 82,')) & 
                        (~shots_df['qualifiers'].str.contains('BigChance'))]
        blkd_bc = shots_df[(shots_df['type'] == 'SavedShot') & (shots_df['qualifiers'].str.contains(': 82,')) & 
                        (shots_df['qualifiers'].str.contains('BigChance'))]
        post = shots_df[(shots_df['type'] == 'ShotOnPost') & (~shots_df['qualifiers'].str.contains('BigChance'))]
        post_bc = shots_df[(shots_df['type'] == 'ShotOnPost') & (shots_df['qualifiers'].str.contains('BigChance'))]

        sotb = shots_df[(shots_df['qualifiers'].str.contains('OutOfBox'))]
        opsh = shots_df[(shots_df['qualifiers'].str.contains('RegularPlay'))]
        ogdf = df[(df['type'] == 'Goal') & (df['qualifiers'].str.contains('OwnGoal')) & (df['teamName'] != team_name)]

        pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, line_zorder=3, linewidth=2)
        pitch.draw(ax=ax)
        xps = [0, 0, 68, 68]
        yps = [0, 35, 35, 0]
        ax.fill(xps, yps, color=bg_color, edgecolor=line_color, lw=3, ls='--', alpha=1, zorder=5)
        ax.vlines(34, ymin=0, ymax=35, color=line_color, ls='--', lw=3, zorder=5)

        pitch.scatter(post.x, post.y, s=200, edgecolors=col, c=col, marker='o', ax=ax)
        pitch.scatter(ontr.x, ontr.y, s=200, edgecolors=col, c='None', hatch='///////', marker='o', ax=ax)
        pitch.scatter(blkd.x, blkd.y, s=200, edgecolors=col, c='None', hatch='///////', marker='s', ax=ax)
        pitch.scatter(miss.x, miss.y, s=200, edgecolors=col, c='None', marker='o', ax=ax)
        pitch.scatter(goal.x, goal.y, s=350, edgecolors='green', linewidths=0.6, c='None', marker='football', zorder=3, ax=ax)
        pitch.scatter((105 - ogdf.x), (68 - ogdf.y), s=350, edgecolors='orange', linewidths=0.6, c='None', marker='football', zorder=3, ax=ax)

        pitch.scatter(post_bc.x, post_bc.y, s=700, edgecolors=col, c=col, marker='o', ax=ax)
        pitch.scatter(ontr_bc.x, ontr_bc.y, s=700, edgecolors=col, c='None', hatch='///////', marker='o', ax=ax)
        pitch.scatter(blkd_bc.x, blkd_bc.y, s=700, edgecolors=col, c='None', hatch='///////', marker='s', ax=ax)
        pitch.scatter(miss_bc.x, miss_bc.y, s=700, edgecolors=col, c='None', marker='o', ax=ax)
        pitch.scatter(goal_bc.x, goal_bc.y, s=850, edgecolors='green', linewidths=0.6, c='None', marker='football', ax=ax)

        if phase_tag == 'Full Time':
            ax.text(34, 112, reshape_arabic_text('الوقت الكامل: 0-90 دقيقة'), color=col, fontsize=13, ha='center', va='center')
        elif phase_tag == 'First Half':
            ax.text(34, 112, reshape_arabic_text('الشوط الأول: 0-45 دقيقة'), color=col, fontsize=13, ha='center', va='center')
        elif phase_tag == 'Second Half':
            ax.text(34, 112, reshape_arabic_text('الشوط الثاني: 45-90 دقيقة'), color=col, fontsize=13, ha='center', va='center')
        ax.text(34, 108, reshape_arabic_text(f'إجمالي التسديدات: {len(shots_df)} | على المرمى: {len(goal) + len(goal_bc) + len(ontr) + len(ontr_bc)}'), color=col, fontsize=13, ha='center', va='center')

        pitch.scatter(12 + (4 * 0), 64, s=200, zorder=6, edgecolors=col, c=col, marker='o', ax=ax)
        pitch.scatter(12 + (4 * 1), 64, s=200, zorder=6, edgecolors=col, c='None', hatch='///////', marker='s', ax=ax)
        pitch.scatter(12 + (4 * 2), 64, s=200, zorder=6, edgecolors=col, c='None', marker='o', ax=ax)
        pitch.scatter(12 + (4 * 3), 64, s=200, zorder=6, edgecolors=col, c='None', hatch='///////', marker='o', ax=ax)
        pitch.scatter(12 + (4 * 4), 64, s=350, zorder=6, edgecolors='green', linewidths=0.6, c='None', marker='football', ax=ax)

        ax.text(34, 39, reshape_arabic_text('إحصائيات التسديد'), fontsize=15, fontweight='bold', zorder=7, ha='center', va='center')

        ax.text(60, 12 + (4 * 4), reshape_arabic_text(f'الأهداف: {len(goal) + len(goal_bc)}'), zorder=6, ha='left', va='center')
        ax.text(60, 12 + (4 * 3), reshape_arabic_text(f'التسديدات المصدودة: {len(ontr) + len(ontr_bc)}'), zorder=6, ha='left', va='center')
        ax.text(60, 12 + (4 * 2), reshape_arabic_text(f'التسديدات خارج المرمى: {len(miss) + len(miss_bc)}'), zorder=6, ha='left', va='center')
        ax.text(60, 12 + (4 * 1), reshape_arabic_text(f'التسديدات المحجوبة: {len(blkd) + len(blkd_bc)}'), zorder=6, ha='left', va='center')
        ax.text(60, 12 + (4 * 0), reshape_arabic_text(f'التسديدات على العارضة: {len(post) + len(post_bc)}'), zorder=6, ha='left', va='center')
        ax.text(30, 12 + (4 * 4), reshape_arabic_text(f'التسديدات خارج الصندوق: {len(sotb)}'), zorder=6, ha='left', va='center')
        ax.text(30, 12 + (4 * 3), reshape_arabic_text(f'التسديدات داخل الصندوق: {len(shots_df) - len(sotb)}'), zorder=6, ha='left', va='center')
        ax.text(30, 12 + (4 * 2), reshape_arabic_text(f'إجمالي الفرص الكبيرة: {len(goal_bc) + len(ontr_bc) + len(miss_bc) + len(blkd_bc) + len(post_bc)}'), zorder=6, ha='left', va='center')
        ax.text(30, 12 + (4 * 1), reshape_arabic_text(f'الفرص الكبيرة المهدرة: {len(ontr_bc) + len(miss_bc) + len(blkd_bc) + len(post_bc)}'), zorder=6, ha='left', va='center')
        ax.text(30, 12 + (4 * 0), reshape_arabic_text(f'التسديدات من اللعب المفتوح: {len(opsh)}'), zorder=6, ha='left', va='center')

        p_list = shots_df.name.unique()
        player_stats = {'Name': p_list, 'Total Shots': [], 'Goals': [], 'Shots Saved': [], 'Shots Off Target': [], 'Shots Blocked': [], 'Shots On Post': [],
                        'Shots outside the box': [], 'Shots inside the box': [], 'Total Big Chances': [], 'Big Chances Missed': [], 'Open-Play Shots': []}
        for name in p_list:
            p_df = shots_df[shots_df['name'] == name]
            player_stats['Total Shots'].append(len(p_df[p_df['type'].isin(['Goal', 'SavedShot', 'MissedShots', 'ShotOnPost'])]))
            player_stats['Goals'].append(len(p_df[p_df['type'] == 'Goal']))
            player_stats['Shots Saved'].append(len(p_df[(p_df['type'] == 'SavedShot') & (~p_df['qualifiers'].str.contains(': 82'))]))
            player_stats['Shots Off Target'].append(len(p_df[p_df['type'] == 'MissedShots']))
            player_stats['Shots Blocked'].append(len(p_df[(p_df['type'] == 'SavedShot') & (p_df['qualifiers'].str.contains(': 82'))]))
            player_stats['Shots On Post'].append(len(p_df[p_df['type'] == 'ShotOnPost']))
            player_stats['Shots outside the box'].append(len(p_df[p_df['qualifiers'].str.contains('OutOfBox')]))
            player_stats['Shots inside the box'].append(len(p_df[~p_df['qualifiers'].str.contains('OutOfBox')]))
            player_stats['Total Big Chances'].append(len(p_df[p_df['qualifiers'].str.contains('BigChance')]))
            player_stats['Big Chances Missed'].append(len(p_df[(p_df['type'] != 'Goal') & (p_df['qualifiers'].str.contains('BigChance'))]))
            player_stats['Open-Play Shots'].append(len(p_df[p_df['qualifiers'].str.contains('RegularPlay')]))

        player_stats_df = pd.DataFrame(player_stats)
        player_stats_df = player_stats_df.sort_values(by='Total Shots', ascending=False)
        return player_stats_df

    # الكود التالي يجب أن يكون داخل كتلة if an_tp == 'Shotmap':
    sm_time_phase = st.radio(" ", ['Full Time', 'First Half', 'Second Half'], index=0, key='sm_time_pill')  # استبدلت st.pills بـ st.radio لأن st.pills غير مدعوم افتراضيًا
    if sm_time_phase == 'Full Time':
        fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
        home_shots_stats = plot_ShotsMap(axs[0], hteamName, hcol, 'Full Time')
        away_shots_stats = plot_ShotsMap(axs[1], ateamName, acol, 'Full Time')
    elif sm_time_phase == 'First Half':
        fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
        home_shots_stats = plot_ShotsMap(axs[0], hteamName, hcol, 'First Half')
        away_shots_stats = plot_ShotsMap(axs[1], ateamName, acol, 'First Half')
    elif sm_time_phase == 'Second Half':
        fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
        home_shots_stats = plot_ShotsMap(axs[0], hteamName, hcol, 'Second Half')
        away_shots_stats = plot_ShotsMap(axs[1], ateamName, acol, 'Second Half')

    fig_text(0.5, 1.05, f'<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>', highlight_textprops=[{'color': hcol}, {'color': acol}], fontsize=30, fontweight='bold', ha='center', va='center', ax=fig)
    fig.text(0.5, 1.01, reshape_arabic_text('خريطة التسديدات'), fontsize=20, ha='center', va='center')
    fig.text(0.5, 0.97, '@adnaaan433', fontsize=10, ha='center', va='center')
    fig.text(0.5, 0.08, reshape_arabic_text('*الشكل الأكبر يعني تسديدات من فرص كبيرة'), fontsize=10, fontstyle='italic', ha='center', va='center')

    himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png")
    himage = Image.open(himage)
    ax_himage = add_image(himage, fig, left=0.085, bottom=0.97, width=0.125, height=0.125)

    aimage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png")
    aimage = Image.open(aimage)
    ax_aimage = add_image(aimage, fig, left=0.815, bottom=0.97, width=0.125, height=0.125)

    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.write(reshape_arabic_text(f'أفضل المسددين في فريق {hteamName}:'))
        st.dataframe(home_shots_stats, hide_index=True)
    with col2:
        st.write(reshape_arabic_text(f'أفضل المسددين في فريق {ateamName}:'))
        st.dataframe(away_shots_stats, hide_index=True)

if an_tp == 'احصائيات الحراس':
    # st.header(f'{st.session_state.analysis_type}')
    st.header(f'{an_tp}')
    
    def plot_goal_post(ax, team_name, col, phase_tag):
        if phase_tag == 'Full Time':
            shots_df = df[(df['teamName']!=team_name) & (df['type'].isin(['Goal', 'MissedShots', 'SavedShot', 'ShotOnPost']))]
        elif phase_tag == 'First Half':
            shots_df = df[(df['teamName']!=team_name) & (df['type'].isin(['Goal', 'MissedShots', 'SavedShot', 'ShotOnPost'])) & (df['period']=='FirstHalf')]
        elif phase_tag == 'Second Half':
            shots_df = df[(df['teamName']!=team_name) & (df['type'].isin(['Goal', 'MissedShots', 'SavedShot', 'ShotOnPost'])) & (df['period']=='SecondHalf')]
    
        shots_df['goalMouthZ'] = (shots_df['goalMouthZ']*0.75) + 38
        shots_df['goalMouthY'] = ((37.66 - shots_df['goalMouthY'])*12.295) + 7.5
    
        # plotting an invisible pitch using the pitch color and line color same color, because the goalposts are being plotted inside the pitch using pitch's dimension
        pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=bg_color, linewidth=2)
        pitch.draw(ax=ax)
        ax.set_ylim(-0.5, 68.5)
        ax.set_xlim(-0.5, 105.5)
        
        # goalpost bars
        ax.plot([7.5, 7.5], [38, 68], color=line_color, linewidth=5)
        ax.plot([7.5, 97.5], [68, 68], color=line_color, linewidth=5)
        ax.plot([97.5, 97.5], [68, 38], color=line_color, linewidth=5)
        ax.plot([0, 105], [38, 38], color=line_color, linewidth=3)
        # plotting the home net
        y_values = (np.arange(0, 6) * 6) + 38
        for y in y_values:
            ax.plot([7.5, 97.5], [y, y], color=line_color, linewidth=2, alpha=0.2)
        x_values = (np.arange(0, 11) * 9) + 7.5
        for x in x_values:
            ax.plot([x, x], [38, 68], color=line_color, linewidth=2, alpha=0.2)
    
        # filtering different types of shots without BigChance
        hSavedf = shots_df[(shots_df['type']=='SavedShot') & (~shots_df['qualifiers'].str.contains(': 82,')) & (~shots_df['qualifiers'].str.contains('BigChance'))]
        hGoaldf = shots_df[(shots_df['type']=='Goal') & (~shots_df['qualifiers'].str.contains('OwnGoal')) & (~shots_df['qualifiers'].str.contains('BigChance'))]
        hPostdf = shots_df[(shots_df['type']=='ShotOnPost') & (~shots_df['qualifiers'].str.contains('BigChance'))]
        
        # filtering different types of shots with BigChance
        hSavedf_bc = shots_df[(shots_df['type']=='SavedShot') & (~shots_df['qualifiers'].str.contains(': 82,')) & (shots_df['qualifiers'].str.contains('BigChance'))]
        hGoaldf_bc = shots_df[(shots_df['type']=='Goal') & (~shots_df['qualifiers'].str.contains('OwnGoal')) & (shots_df['qualifiers'].str.contains('BigChance'))]
        hPostdf_bc = shots_df[(shots_df['type']=='ShotOnPost') & (shots_df['qualifiers'].str.contains('BigChance'))]
    
        # scattering those shots without BigChance
        pitch.scatter(hSavedf.goalMouthY, hSavedf.goalMouthZ, marker='o', c=bg_color, zorder=3, edgecolor=col, hatch='/////', s=350, ax=ax)
        pitch.scatter(hGoaldf.goalMouthY, hGoaldf.goalMouthZ, marker='football', c=bg_color, zorder=3, edgecolors='green', s=350, ax=ax)
        pitch.scatter(hPostdf.goalMouthY, hPostdf.goalMouthZ, marker='o', c=bg_color, zorder=3, edgecolors='orange', hatch='/////', s=350, ax=ax)
        # scattering those shots with BigChance
        pitch.scatter(hSavedf_bc.goalMouthY, hSavedf_bc.goalMouthZ, marker='o', c=bg_color, zorder=3, edgecolor=col, hatch='/////', s=1000, ax=ax)
        pitch.scatter(hGoaldf_bc.goalMouthY, hGoaldf_bc.goalMouthZ, marker='football', c=bg_color, zorder=3, edgecolors='green', s=1000, ax=ax)
        pitch.scatter(hPostdf_bc.goalMouthY, hPostdf_bc.goalMouthZ, marker='o', c=bg_color, zorder=3, edgecolors='orange', hatch='/////', s=1000, ax=ax)
    
        if phase_tag == 'Full Time':
            ax.text(52.5, 80, 'Full Time: 0-90 minutes', color=col, fontsize=13, ha='center', va='center')
        elif phase_tag == 'First Half':
            ax.text(52.5, 80, 'First Half: 0-45 minutes', color=col, fontsize=13, ha='center', va='center')
        elif phase_tag == 'Second Half':
            ax.text(52.5, 80, 'Second Half: 45-90 minutes', color=col, fontsize=13, ha='center', va='center')
            
        ax.text(52.5, 73, f'{team_name} GK Saves', color=col, fontsize=15, fontweight='bold', ha='center', va='center')
    
        ax.text(52.5, 28-(5*0), f'Total Shots faced: {len(shots_df)}', fontsize=13, ha='center', va='center')
        ax.text(52.5, 28-(5*1), f'Shots On Target faced: {len(hSavedf)+len(hSavedf_bc)+len(hGoaldf)+len(hGoaldf_bc)}', fontsize=13, ha='center', va='center')
        ax.text(52.5, 28-(5*2), f'Shots Saved: {len(hSavedf)+len(hSavedf_bc)}', fontsize=13, ha='center', va='center')
        ax.text(52.5, 28-(5*3), f'Goals Conceded: {len(hGoaldf)+len(hGoaldf_bc)}', fontsize=13, ha='center', va='center')
        ax.text(52.5, 28-(5*4), f'Goals Conceded from Big Chances: {len(hGoaldf_bc)}', fontsize=13, ha='center', va='center')
        ax.text(52.5, 28-(5*5), f'Big Chances Saved: {len(hSavedf_bc)}', fontsize=13, ha='center', va='center')
    
        return
    
    gp_time_phase = st.radio(" ", ['Full Time', 'First Half', 'Second Half'], index=0, key='gp_time_pill')  # استبدلت st.pills بـ st.radio
    if gp_time_phase == 'Full Time':
        fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
        home_shots_stats = plot_goal_post(axs[0], hteamName, hcol, 'Full Time')
        away_shots_stats = plot_goal_post(axs[1], ateamName, acol, 'Full Time')
    elif gp_time_phase == 'First Half':
        fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
        plot_goal_post(axs[0], hteamName, hcol, 'First Half')
        plot_goal_post(axs[1], ateamName, acol, 'First Half')
    elif gp_time_phase == 'Second Half':
        fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
        plot_goal_post(axs[0], hteamName, hcol, 'Second Half')
        plot_goal_post(axs[1], ateamName, acol, 'Second Half')
    
    fig_text(0.5, 0.94, f'<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>', highlight_textprops=[{'color': hcol}, {'color': acol}], fontsize=30, fontweight='bold', ha='center', va='center', ax=fig)
    fig.text(0.5, 0.89, 'GoalKeeper Saves', fontsize=20, ha='center', va='center')
    fig.text(0.5, 0.84, '@adnaaan433', fontsize=10, ha='center', va='center')
    
    fig.text(0.5, 0.2, '*Bigger circle means shots from Big Chances', fontsize=10, fontstyle='italic', ha='center', va='center')
    
    himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png")
    himage = Image.open(himage)
    ax_himage = add_image(himage, fig, left=0.085, bottom=0.86, width=0.125, height=0.125)
    
    aimage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png")
    aimage = Image.open(aimage)
    ax_aimage = add_image(aimage, fig, left=0.815, bottom=0.86, width=0.125, height=0.125)
    
    st.pyplot(fig)

if an_tp == 'Match Momentum':
    # st.header(f'{st.session_state.analysis_type}')
    st.header(f'{an_tp}')
elif an_tp == 'Attacking Thirds':
    st.header(reshape_arabic_text('الثلث الهجومي'))

    # خيار اختيار الوقت
    time_option = st.selectbox(
        reshape_arabic_text('اختر الوقت:'),
        [reshape_arabic_text('90 دقيقة'), reshape_arabic_text('الشوط الأول'), reshape_arabic_text('الشوط الثاني')]
    )

    # تقسيم البيانات حسب الفريقين
    homedf = df[df['teamName'] == hteamName]
    awaydf = df[df['teamName'] == ateamName]

    # التحقق من قيم period
    st.write("قيم عمود period:", df['period'].unique())

    # تصفية البيانات حسب الوقت المختار
    if time_option == reshape_arabic_text('الشوط الأول'):
        homedf = homedf[homedf['period'] == 1]
        awaydf = awaydf[awaydf['period'] == 1]
    elif time_option == reshape_arabic_text('الشوط الثاني'):
        homedf = homedf[homedf['period'] == 2]
        awaydf = awaydf[awaydf['period'] == 2]

    # تصفية التمريرات فقط (ناجحة وغير ناجحة)
    home_passes = homedf[homedf['type'] == 'Pass']
    away_passes = awaydf[awaydf['type'] == 'Pass']

    # تصفية التمريرات الناجحة فقط
    home_successful_passes = home_passes[home_passes['outcomeType'] == 'Successful']
    away_successful_passes = away_passes[away_passes['outcomeType'] == 'Successful']

    # التحقق من عدد التمريرات
    st.write("إجمالي التمريرات للفريق المضيف:", len(home_passes))
    st.write("إجمالي التمريرات للفريق الضيف:", len(away_passes))

    # دالة لحساب المسافة بين نقطتي التمريرة
    def calculate_distance(row):
        x1, y1 = row['x'], row['y']
        x2, y2 = row['endX'], row['endY']
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    # تصنيف التمريرات إلى طويلة وقصيرة (الناجحة فقط)
    home_successful_passes['distance'] = home_successful_passes.apply(calculate_distance, axis=1)
    away_successful_passes['distance'] = away_successful_passes.apply(calculate_distance, axis=1)

    # التمريرات القصيرة (أقل من 15 مترًا) والطويلة (15 مترًا أو أكثر)
    home_short_passes = home_successful_passes[home_successful_passes['distance'] < 15]
    home_long_passes = home_successful_passes[home_successful_passes['distance'] >= 15]

    away_short_passes = away_successful_passes[away_successful_passes['distance'] < 15]
    away_long_passes = away_successful_passes[away_successful_passes['distance'] >= 15]

    # تقسيم الملعب إلى ثلاثة أقسام (باستخدام UEFA dimensions: 105x68)
    def split_thirds(df_team):
        left_third = df_team[df_team['x'] <= 35]  # الثلث الدفاعي (0-35)
        middle_third = df_team[(df_team['x'] > 35) & (df_team['x'] <= 70)]  # الثلث الأوسط (35-70)
        right_third = df_team[df_team['x'] > 70]  # الثلث الهجومي (70-105)
        return left_third, middle_third, right_third

    # تقسيم التمريرات (ناجحة وإجمالي) لكل فريق
    h_left_success, h_middle_success, h_right_success = split_thirds(home_successful_passes)
    a_left_success, a_middle_success, a_right_success = split_thirds(away_successful_passes)

    h_left_total, h_middle_total, h_right_total = split_thirds(home_passes)
    a_left_total, a_middle_total, a_right_total = split_thirds(away_passes)

    # تقسيم التمريرات الطويلة والقصيرة الناجحة
    h_left_short, h_middle_short, h_right_short = split_thirds(home_short_passes)
    h_left_long, h_middle_long, h_right_long = split_thirds(home_long_passes)

    a_left_short, a_middle_short, a_right_short = split_thirds(away_short_passes)
    a_left_long, a_middle_long, a_right_long = split_thirds(away_long_passes)

    # حساب عدد التمريرات الناجحة والإجمالية في كل منطقة
    h_left_success_count = len(h_left_success)
    h_middle_success_count = len(h_middle_success)
    h_right_success_count = len(h_right_success)

    a_left_success_count = len(a_left_success)
    a_middle_success_count = len(a_middle_success)
    a_right_success_count = len(a_right_success)

    h_left_total_count = len(h_left_total)
    h_middle_total_count = len(h_middle_total)
    h_right_total_count = len(h_right_total)

    a_left_total_count = len(a_left_total)
    a_middle_total_count = len(a_middle_total)
    a_right_total_count = len(a_right_total)

    # حساب عدد التمريرات الطويلة والقصيرة الناجحة في كل منطقة
    h_left_short_count = len(h_left_short)
    h_middle_short_count = len(h_middle_short)
    h_right_short_count = len(h_right_short)

    h_left_long_count = len(h_left_long)
    h_middle_long_count = len(h_middle_long)
    h_right_long_count = len(h_right_long)

    a_left_short_count = len(a_left_short)
    a_middle_short_count = len(a_middle_short)
    a_right_short_count = len(a_right_short)

    a_left_long_count = len(a_left_long)
    a_middle_long_count = len(a_middle_long)
    a_right_long_count = len(a_right_long)

    # حساب نسب النجاح
    h_left_success_rate = (h_left_success_count / h_left_total_count * 100) if h_left_total_count > 0 else 0
    h_middle_success_rate = (h_middle_success_count / h_middle_total_count * 100) if h_middle_total_count > 0 else 0
    h_right_success_rate = (h_right_success_count / h_right_total_count * 100) if h_right_total_count > 0 else 0

    a_left_success_rate = (a_left_success_count / a_left_total_count * 100) if a_left_total_count > 0 else 0
    a_middle_success_rate = (a_middle_success_count / a_middle_total_count * 100) if a_middle_total_count > 0 else 0
    a_right_success_rate = (a_right_success_count / a_right_total_count * 100) if a_right_total_count > 0 else 0

    # التحقق من وجود بيانات
    if len(home_passes) == 0 and len(away_passes) == 0:
        st.warning(reshape_arabic_text("لا توجد بيانات متاحة للتمريرات في هذه الفترة."))
        st.stop()

    # إعداد الرسم
    fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
    pitch = VerticalPitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, corner_arcs=True, linewidth=1.5)

    # رسم ملعب الفريق المضيف
    pitch.draw(ax=axs[0])
    # إضافة الأسهم للثلثات
    arrowprops = dict(facecolor=hcol, edgecolor='white', linewidth=2, alpha=0.8)
    axs[0].annotate('', xy=(34, 35), xytext=(34, 0), arrowprops=dict(arrowstyle='->', **arrowprops))
    axs[0].annotate('', xy=(34, 70), xytext=(34, 35), arrowprops=dict(arrowstyle='->', **arrowprops))
    axs[0].annotate('', xy=(34, 105), xytext=(34, 70), arrowprops=dict(arrowstyle='->', **arrowprops))

    # إضافة النصوص للفريق المضيف (نسبة النجاح وعدد التمريرات الناجحة)
    axs[0].text(34, 17.5, reshape_arabic_text(f'{h_left_success_rate:.1f}%'), fontsize=18, ha='center', va='center', color='white', fontweight='bold')
    axs[0].text(34, 52.5, reshape_arabic_text(f'{h_middle_success_rate:.1f}%'), fontsize=18, ha='center', va='center', color='white', fontweight='bold')
    axs[0].text(34, 87.5, reshape_arabic_text(f'{h_right_success_rate:.1f}%'), fontsize=18, ha='center', va='center', color='white', fontweight='bold')
    axs[0].text(34, 115, reshape_arabic_text(f'{hteamName}'), fontsize=16, ha='center', va='center', color=hcol, fontweight='bold')

    # إضافة عدد التمريرات الناجحة
    axs[0].text(34, 10, reshape_arabic_text(f'{h_left_success_count} تمريرات'), fontsize=12, ha='center', va='center', color='white')
    axs[0].text(34, 45, reshape_arabic_text(f'{h_middle_success_count} تمريرات'), fontsize=12, ha='center', va='center', color='white')
    axs[0].text(34, 80, reshape_arabic_text(f'{h_right_success_count} تمريرات'), fontsize=12, ha='center', va='center', color='white')

    # تلوين الثلثات
    axs[0].fill_between(x=[0, 68], y1=0, y2=35, color=hcol, alpha=0.3)
    axs[0].fill_between(x=[0, 68], y1=35, y2=70, color=hcol, alpha=0.5)
    axs[0].fill_between(x=[0, 68], y1=70, y2=105, color=hcol, alpha=0.7)

    # رسم ملعب الفريق الضيف
    pitch.draw(ax=axs[1])
    # إضافة الأسهم للثلثات
    arrowprops = dict(facecolor=acol, edgecolor='white', linewidth=2, alpha=0.8)
    axs[1].annotate('', xy=(34, 35), xytext=(34, 0), arrowprops=dict(arrowstyle='->', **arrowprops))
    axs[1].annotate('', xy=(34, 70), xytext=(34, 35), arrowprops=dict(arrowstyle='->', **arrowprops))
    axs[1].annotate('', xy=(34, 105), xytext=(34, 70), arrowprops=dict(arrowstyle='->', **arrowprops))

    # إضافة النصوص للفريق الضيف (نسبة النجاح وعدد التمريرات الناجحة)
    axs[1].text(34, 17.5, reshape_arabic_text(f'{a_left_success_rate:.1f}%'), fontsize=18, ha='center', va='center', color='white', fontweight='bold')
    axs[1].text(34, 52.5, reshape_arabic_text(f'{a_middle_success_rate:.1f}%'), fontsize=18, ha='center', va='center', color='white', fontweight='bold')
    axs[1].text(34, 87.5, reshape_arabic_text(f'{a_right_success_rate:.1f}%'), fontsize=18, ha='center', va='center', color='white', fontweight='bold')
    axs[1].text(34, 115, reshape_arabic_text(f'{ateamName}'), fontsize=16, ha='center', va='center', color=acol, fontweight='bold')

    # إضافة عدد التمريرات الناجحة
    axs[1].text(34, 10, reshape_arabic_text(f'{a_left_success_count} تمريرات'), fontsize=12, ha='center', va='center', color='white')
    axs[1].text(34, 45, reshape_arabic_text(f'{a_middle_success_count} تمريرات'), fontsize=12, ha='center', va='center', color='white')
    axs[1].text(34, 80, reshape_arabic_text(f'{a_right_success_count} تمريرات'), fontsize=12, ha='center', va='center', color='white')

    # تلوين الثلثات
    axs[1].fill_between(x=[0, 68], y1=0, y2=35, color=acol, alpha=0.3)
    axs[1].fill_between(x=[0, 68], y1=35, y2=70, color=acol, alpha=0.5)
    axs[1].fill_between(x=[0, 68], y1=70, y2=105, color=acol, alpha=0.7)

    # إضافة العنوان العام
    home_part = reshape_arabic_text(f"{hteamName} {hgoal_count}")
    away_part = reshape_arabic_text(f"{agoal_count} {ateamName}")
    title = f"<{home_part}> - <{away_part}>"
    fig_text(0.5, 1.05, title, highlight_textprops=[{'color': hcol}, {'color': acol}], fontsize=28, fontweight='bold', ha='center', va='center', ax=fig)

    fig.text(0.5, 1.01, reshape_arabic_text('الثلثات الهجومية'), fontsize=18, ha='center', va='center', color='white', weight='bold')
    fig.text(0.5, 0.97, '✦ @REO_SHOW ✦', fontsize=14, fontfamily='Roboto', fontweight='bold', color='#FFD700', ha='center', va='center',
             bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', pad=2),
             path_effects=[patheffects.withStroke(linewidth=2, foreground='white')])

    # إضافة وصف
    fig.text(0.5, 0.05, reshape_arabic_text(f'نسبة نجاح التمريرات وعدد التمريرات الناجحة في كل ثلث - {time_option}'), fontsize=12, ha='center', va='center', color='white')

    # إضافة شعارات الفريقين
    try:
        himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png")
        himage = Image.open(himage)
        add_image(himage, fig, left=0.085, bottom=0.97, width=0.125, height=0.125)

        aimage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png")
        aimage = Image.open(aimage)
        add_image(aimage, fig, left=0.815, bottom=0.97, width=0.125, height=0.125)
    except Exception as e:
        st.warning(f"خطأ في تحميل شعارات الفريقين: {e}")

    st.pyplot(fig)

    # عرض البيانات في جدول
    col1, col2 = st.columns(2)
    with col1:
        st.write(reshape_arabic_text(f'تفاصيل تمريرات {hteamName} ({time_option}):'))
        h_data = pd.DataFrame({
            'الثلث': ['دفاعي', 'أوسط', 'هجومي'],
            'التمريرات الناجحة': [h_left_success_count, h_middle_success_count, h_right_success_count],
            'إجمالي التمريرات': [h_left_total_count, h_middle_total_count, h_right_total_count],
            'نسبة النجاح (%)': [f'{h_left_success_rate:.1f}', f'{h_middle_success_rate:.1f}', f'{h_right_success_rate:.1f}'],
            'التمريرات القصيرة (ناجحة)': [h_left_short_count, h_middle_short_count, h_right_short_count],
            'التمريرات الطويلة (ناجحة)': [h_left_long_count, h_middle_long_count, h_right_long_count]
        })
        st.dataframe(h_data, hide_index=True)
    with col2:
        st.write(reshape_arabic_text(f'تفاصيل تمريرات {ateamName} ({time_option}):'))
        a_data = pd.DataFrame({
            'الثلث': ['دفاعي', 'أوسط', 'هجومي'],
            'التمريرات الناجحة': [a_left_success_count, a_middle_success_count, a_right_success_count],
            'إجمالي التمريرات': [a_left_total_count, a_middle_total_count, a_right_total_count],
            'نسبة النجاح (%)': [f'{a_left_success_rate:.1f}', f'{a_middle_success_rate:.1f}', f'{a_right_success_rate:.1f}'],
            'التمريرات القصيرة (ناجحة)': [a_left_short_count, a_middle_short_count, a_right_short_count],
            'التمريرات الطويلة (ناجحة)': [a_left_long_count, a_middle_long_count, a_right_long_count]
        })
        st.dataframe(a_data, hide_index=True)
def plot_match_momentm(ax, phase_tag):
                u_df = df[df['period']==phase_tag]
                u_df = u_df[~u_df['qualifiers'].str.contains('CornerTaken')]
                u_df = u_df[['x', 'minute', 'type', 'teamName', 'qualifiers']]
                u_df = u_df[~u_df['type'].isin(['Start', 'OffsidePass', 'OffsideProvoked', 'CornerAwarded', 'End', 
                                'OffsideGiven', 'SubstitutionOff', 'SubstitutionOn', 'FormationChange', 'FormationSet'])].reset_index(drop=True)
                u_df.loc[u_df['teamName'] == ateamName, 'x'] = 105 - u_df.loc[u_df['teamName'] == ateamName, 'x']
            
                homedf = u_df[u_df['teamName']==hteamName]
                awaydf = u_df[u_df['teamName']==ateamName]
                
                Momentumdf = u_df.groupby('minute')['x'].mean()
                Momentumdf = Momentumdf.reset_index()
                Momentumdf.columns = ['minute', 'average_x']
                Momentumdf['average_x'] = Momentumdf['average_x'] - 52.5
            
                # Creating the bar plot
                ax.bar(Momentumdf['minute'], Momentumdf['average_x'], width=1, color=[hcol if x > 0 else acol for x in Momentumdf['average_x']])
                
                ax.axhline(0, color=line_color, linewidth=2)
                # ax.set_xticks(False)
                ax.set_facecolor('#ededed')
                # Hide spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                # # Hide ticks
                ax.tick_params(axis='both', which='both', length=0)
                ax.tick_params(axis='x', colors=line_color)
                ax.tick_params(axis='y', colors=bg_color)
                # Add labels and title
                ax.set_xlabel('Minute', color=line_color, fontsize=20)
                ax.grid(True, ls='dotted')
            
                
                # making a list of munutes when goals are scored
                hgoal_list = homedf[(homedf['type'] == 'Goal') & (~homedf['qualifiers'].str.contains('OwnGoal'))]['minute'].tolist()
                agoal_list = awaydf[(awaydf['type'] == 'Goal') & (~awaydf['qualifiers'].str.contains('OwnGoal'))]['minute'].tolist()
                hog_list = homedf[(homedf['type'] == 'Goal') & (homedf['qualifiers'].str.contains('OwnGoal'))]['minute'].tolist()
                aog_list = awaydf[(awaydf['type'] == 'Goal') & (awaydf['qualifiers'].str.contains('OwnGoal'))]['minute'].tolist()
                hred_list = homedf[homedf['qualifiers'].str.contains('Red|SecondYellow')]['minute'].tolist()
                ared_list = awaydf[awaydf['qualifiers'].str.contains('Red|SecondYellow')]['minute'].tolist()
            
                ax.scatter(hgoal_list, [60]*len(hgoal_list), s=250, c=bg_color, edgecolor='green', hatch='////', marker='o', zorder=4)
                ax.vlines(hgoal_list, ymin=0, ymax=60, ls='--',  color='green')
                ax.scatter(agoal_list, [-60]*len(agoal_list), s=250, c=bg_color, edgecolor='green', hatch='////', marker='o', zorder=4)
                ax.vlines(agoal_list, ymin=0, ymax=-60, ls='--',  color='green')
                ax.scatter(hog_list, [-60]*len(hog_list), s=250, c=bg_color, edgecolor='orange', hatch='////', marker='o', zorder=4)
                ax.vlines(hog_list, ymin=0, ymax=60, ls='--',  color='orange')
                ax.scatter(aog_list, [60]*len(aog_list), s=250, c=bg_color, edgecolor='orange', hatch='////', marker='o', zorder=4)
                ax.vlines(aog_list, ymin=0, ymax=60, ls='--',  color='orange')
                ax.scatter(hred_list, [60]*len(hred_list), s=250, c=bg_color, edgecolor='red', hatch='////', marker='s', zorder=4)
                ax.scatter(ared_list, [-60]*len(ared_list), s=250, c=bg_color, edgecolor='red', hatch='////', marker='s', zorder=4)
            
                ax.set_ylim(-65, 65)
            
                if phase_tag=='FirstHalf':
                    ax.set_xticks(range(0, int(Momentumdf['minute'].max()), 5))
                    ax.set_title('First Half', fontsize=20)
                    ax.set_xlim(-1, Momentumdf['minute'].max()+1)
                    ax.axvline(45, color='gray', linewidth=2, linestyle='dotted')
                    ax.set_ylabel('Momentum', color=line_color, fontsize=20)
                else:
                    ax.set_xticks(range(45, int(Momentumdf['minute'].max()), 5))
                    ax.set_title('Second Half', fontsize=20)
                    ax.set_xlim(44, Momentumdf['minute'].max()+1)
                    ax.axvline(90, color='gray', linewidth=2, linestyle='dotted')
                return
            
                fig,axs=plt.subplots(1,2, figsize=(20,10), facecolor=bg_color)
                plot_match_momentm(axs[0], 'FirstHalf')
                plot_match_momentm(axs[1], 'SecondHalf')
                fig.subplots_adjust(wspace=0.025)
            
                fig_text(0.5, 1.1, f'<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>', highlight_textprops=[{'color':hcol}, {'color':acol}], fontsize=40, fontweight='bold', ha='center', va='center', ax=fig)
                fig.text(0.5, 1.04, 'Match Momentum', fontsize=30, ha='center', va='center')
                fig.text(0.5, 0.98, '@adnaaan433', fontsize=15, ha='center', va='center')
            
                fig.text(0.5, -0.01, '*Momentum is the measure of the Avg. Open-Play Attacking Threat of a team per minute', fontsize=15, fontstyle='italic', ha='center', va='center')
                fig.text(0.5, -0.05, '*green circle: Goals, orange circle: own goal', fontsize=15, fontstyle='italic', ha='center', va='center')
            
                himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png")
                himage = Image.open(himage)
                ax_himage = add_image(himage, fig, left=0.085, bottom=1.02, width=0.125, height=0.125)
            
                aimage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png")
                aimage = Image.open(aimage)
                ax_aimage = add_image(aimage, fig, left=0.815, bottom=1.02, width=0.125, height=0.125)
            
                st.pyplot(fig)
            
                st.header('Cumulative xT')
            
if an_tp == 'xT Momentum':
    st.header('xT Momentum')

    def plot_xT_momentum(ax, phase_tag):
        hxt_df = df[(df['teamName'] == hteamName) & (df['xT'] > 0)]
        axt_df = df[(df['teamName'] == ateamName) & (df['xT'] > 0)]
        
        hcm_xt = hxt_df.groupby(['period', 'minute'])['xT'].sum().reset_index()
        hcm_xt['cumulative_xT'] = hcm_xt['xT'].cumsum()
        htop_xt = hcm_xt['cumulative_xT'].max()
        hcm_xt = hcm_xt[hcm_xt['period'] == phase_tag]
        htop_mint = hcm_xt['minute'].max()
        h_max_cum = hcm_xt.cumulative_xT.iloc[-1]
        
        acm_xt = axt_df.groupby(['period', 'minute'])['xT'].sum().reset_index()
        acm_xt['cumulative_xT'] = acm_xt['xT'].cumsum()
        atop_xt = acm_xt['cumulative_xT'].max()
        acm_xt = acm_xt[acm_xt['period'] == phase_tag]
        atop_mint = acm_xt['minute'].max()
        a_max_cum = acm_xt.cumulative_xT.iloc[-1]
        
        if htop_mint > atop_mint:
            add_last = {'period': phase_tag, 'minute': htop_mint, 'xT': 0, 'cumulative_xT': a_max_cum}
            acm_xt = pd.concat([acm_xt, pd.DataFrame([add_last])], ignore_index=True)
        if atop_mint > htop_mint:
            add_last = {'period': phase_tag, 'minute': atop_mint, 'xT': 0, 'cumulative_xT': h_max_cum}
            hcm_xt = pd.concat([hcm_xt, pd.DataFrame([add_last])], ignore_index=True)
        
        ax.step(hcm_xt['minute'], hcm_xt['cumulative_xT'], where='pre', color=hcol)
        ax.fill_between(hcm_xt['minute'], hcm_xt['cumulative_xT'], step='pre', color=hcol, alpha=0.25)
        
        ax.step(acm_xt['minute'], acm_xt['cumulative_xT'], where='pre', color=acol)
        ax.fill_between(acm_xt['minute'], acm_xt['cumulative_xT'], step='pre', color=acol, alpha=0.25)
        
        top_xT_list = [htop_xt, atop_xt]
        top_mn_list = [htop_mint, atop_mint]
        ax.set_ylim(0, max(top_xT_list))
        
        if phase_tag == 'FirstHalf':
            ax.set_xlim(-1, max(top_mn_list) + 1)
            ax.set_title('First Half', fontsize=20)
            ax.set_ylabel('Cumulative Expected Threat (CxT)', color=line_color, fontsize=20)
        else:
            ax.set_xlim(44, max(top_mn_list) + 1)
            ax.set_title('Second Half', fontsize=20)
            ax.text(htop_mint + 0.5, h_max_cum, f"{hteamName}\nCxT: {h_max_cum:.2f}", fontsize=15, color=hcol)
            ax.text(atop_mint + 0.5, a_max_cum, f"{ateamName}\nCxT: {a_max_cum:.2f}", fontsize=15, color=acol)
        
        ax.set_facecolor('#ededed')
        # Hide spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        # Hide ticks
        ax.tick_params(axis='both', which='both', length=0)
        ax.tick_params(axis='x', colors=line_color)
        ax.tick_params(axis='y', colors='None')
        # Add labels and title
        ax.set_xlabel('Minute', color=line_color, fontsize=20)
        ax.grid(True, ls='dotted')
        
        return hcm_xt, acm_xt

    # إعداد الرسم
    fig, axs = plt.subplots(1, 2, figsize=(20, 10), facecolor=bg_color)
    h_fh, a_fh = plot_xT_momentum(axs[0], 'FirstHalf')
    h_sh, a_sh = plot_xT_momentum(axs[1], 'SecondHalf')
    fig.subplots_adjust(wspace=0.025)
    
    fig_text(0.5, 1.1, f'<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>', highlight_textprops=[{'color': hcol}, {'color': acol}], fontsize=40, fontweight='bold', ha='center', va='center', ax=fig)
    fig.text(0.5, 1.04, 'Cumulative Expected Threat (CxT)', fontsize=30, ha='center', va='center')
    fig.text(0.5, 0.98, '@adnaaan433', fontsize=15, ha='center', va='center')
    
    fig.text(0.5, -0.01, '*Cumulative xT is the sum of the consecutive xT from Open-Play Pass and Carries', fontsize=15, fontstyle='italic', ha='center', va='center')
    
    himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png")
    himage = Image.open(himage)
    ax_himage = add_image(himage, fig, left=0.085, bottom=1.02, width=0.125, height=0.125)
    
    aimage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png")
    aimage = Image.open(aimage)
    ax_aimage = add_image(aimage, fig, left=0.815, bottom=1.02, width=0.125, height=0.125)
    
    st.pyplot(fig)

elif an_tp == 'Zone14 & Half-Space Passes':
    st.header('Passes Into Zone14 & Half-Spaces')

    def plot_zone14i(ax, team_name, col, phase_tag):
        if phase_tag == 'Full Time':
            pass_df = df[(df['teamName'] == team_name) & (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful') & (~df['qualifiers'].str.contains('Freekick|Corner'))]
        elif phase_tag == 'First Half':
            pass_df = df[(df['teamName'] == team_name) & (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful') & (~df['qualifiers'].str.contains('Freekick|Corner')) & (df['period'] == 'FirstHalf')]
        elif phase_tag == 'Second Half':
            pass_df = df[(df['teamName'] == team_name) & (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful') & (~df['qualifiers'].str.contains('Freekick|Corner')) & (df['period'] == 'SecondHalf')]
        
        pitch = VerticalPitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
        pitch.draw(ax=ax)
        
        z14_x = [68/3, 68/3, 136/3, 136/3]
        z14_y = [70, 88, 88, 70]
        ax.fill(z14_x, z14_y, color='orange', edgecolor='None', alpha=0.3)
        lhs_x = [136/3, 136/3, 272/5, 272/5]
        lhs_y = [70, 105, 105, 70]
        ax.fill(lhs_x, lhs_y, color=col, edgecolor='None', alpha=0.3)
        rhs__x = [68/5, 68/5, 68/3, 68/3]
        rhs__y = [70, 105, 105, 70]
        ax.fill(rhs__x, rhs__y, color=col, edgecolor='None', alpha=0.3)
        
        z14_pass = pass_df[(pass_df['endX'] >= 70) & (pass_df['endX'] <= 88) & (pass_df['endY'] >= 68/3) & (pass_df['endY'] <= 136/3)]
        pitch.lines(z14_pass.x, z14_pass.y, z14_pass.endX, z14_pass.endY, comet=True, lw=4, color='orange', zorder=4, ax=ax)
        pitch.scatter(z14_pass.endX, z14_pass.endY, s=75, color=bg_color, ec='orange', lw=2, zorder=5, ax=ax)
        z14_kp = z14_pass[z14_pass['qualifiers'].str.contains('KeyPass')]
        z14_as = z14_pass[z14_pass['qualifiers'].str.contains('GoalAssist')]
        
        lhs_pass = pass_df[(pass_df['endX'] >= 70) & (pass_df['endY'] >= 136/3) & (pass_df['endY'] <= 272/5)]
        pitch.lines(lhs_pass.x, lhs_pass.y, lhs_pass.endX, lhs_pass.endY, comet=True, lw=4, color=col, zorder=4, ax=ax)
        pitch.scatter(lhs_pass.endX, lhs_pass.endY, s=75, color=bg_color, ec=col, lw=2, zorder=5, ax=ax)
        lhs_kp = lhs_pass[lhs_pass['qualifiers'].str.contains('KeyPass')]
        lhs_as = lhs_pass[lhs_pass['qualifiers'].str.contains('GoalAssist')]
        
        rhs_pass = pass_df[(pass_df['endX'] >= 70) & (pass_df['endY'] >= 68/5) & (pass_df['endY'] <= 68/3)]
        pitch.lines(rhs_pass.x, rhs_pass.y, rhs_pass.endX, rhs_pass.endY, comet=True, lw=4, color=col, zorder=4, ax=ax)
        pitch.scatter(rhs_pass.endX, rhs_pass.endY, s=75, color=bg_color, ec=col, lw=2, zorder=5, ax=ax)
        rhs_kp = rhs_pass[rhs_pass['qualifiers'].str.contains('KeyPass')]
        rhs_as = rhs_pass[rhs_pass['qualifiers'].str.contains('GoalAssist')]
        
        pitch.scatter(17, 34, s=12000, color='orange', ec=line_color, lw=2, marker='h', zorder=7, ax=ax)
        pitch.scatter(35, 68/3, s=12000, color=col, ec=line_color, lw=2, marker='h', zorder=7, alpha=0.8, ax=ax)
        pitch.scatter(35, 136/3, s=12000, color=col, ec=line_color, lw=2, marker='h', zorder=7, alpha=0.8, ax=ax)
        
        ax.text(34, 21, 'Zone14', size=15, color='k', fontweight='bold', ha='center', va='center', zorder=10)
        ax.text(34, 16, f' \nTotal:{len(z14_pass)}\nKeyPass:{len(z14_kp)}\nAssist:{len(z14_as)}', size=13, color='k', ha='center', va='center', zorder=10)
        
        ax.text(136/3, 39, 'Left H.S.', size=15, color='k', fontweight='bold', ha='center', va='center', zorder=10)
        ax.text(136/3, 34, f' \nTotal:{len(lhs_pass)}\nKeyPass:{len(lhs_kp)}\nAssist:{len(lhs_as)}', size=13, color='k', ha='center', va='center', zorder=10)
        
        ax.text(68/3, 39, 'Right H.S.', size=15, color='k', fontweight='bold', ha='center', va='center', zorder=10)
        ax.text(68/3, 34, f' \nTotal:{len(rhs_pass)}\nKeyPass:{len(rhs_kp)}\nAssist:{len(rhs_as)}', size=13, color='k', ha='center', va='center', zorder=10)
        
        if phase_tag == 'Full Time':
            ax.text(34, 110, 'Full Time: 0-90 minutes', color=col, fontsize=13, ha='center', va='center')
        elif phase_tag == 'First Half':
            ax.text(34, 110, 'First Half: 0-45 minutes', color=col, fontsize=13, ha='center', va='center')
        elif phase_tag == 'Second Half':
            ax.text(34, 110, 'Second Half: 45-90 minutes', color=col, fontsize=13, ha='center', va='center')
        
        z14_pass['zone'] = 'Zone 14'
        lhs_pass['zone'] = 'Left Half Space'
        rhs_pass['zone'] = 'Right Half Space'
        total_df = pd.concat([z14_pass, lhs_pass, rhs_pass], ignore_index=True)
        total_df = total_df[['name', 'zone']]
        stats = total_df.groupby(['name', 'zone']).size().unstack(fill_value=0)
        stats['Total'] = stats.sum(axis=1)
        stats = stats.sort_values(by='Total', ascending=False)
        return stats

    fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
    zhsi_time_phase = st.pills(" ", ['Full Time', 'First Half', 'Second Half'], default='Full Time', key='Into')
    if zhsi_time_phase == 'Full Time':
        home_z14hsi_stats = plot_zone14i(axs[0], hteamName, hcol, 'Full Time')
        away_z14hsi_stats = plot_zone14i(axs[1], ateamName, acol, 'Full Time')
    if zhsi_time_phase == 'First Half':
        home_z14hsi_stats = plot_zone14i(axs[0], hteamName, hcol, 'First Half')
        away_z14hsi_stats = plot_zone14i(axs[1], ateamName, acol, 'First Half')
    if zhsi_time_phase == 'Second Half':
        home_z14hsi_stats = plot_zone14i(axs[0], hteamName, hcol, 'Second Half')
        away_z14hsi_stats = plot_zone14i(axs[1], ateamName, acol, 'Second Half')
    
    fig_text(0.5, 1.05, f'<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>', highlight_textprops=[{'color': hcol}, {'color': acol}], fontsize=30, fontweight='bold', ha='center', va='center', ax=fig)
    fig.text(0.5, 1.01, 'Passes into Zone14 and Half-Spaces', fontsize=20, ha='center', va='center')
    fig.text(0.5, 0.97, '@adnaaan433', fontsize=10, ha='center', va='center')
    
    fig.text(0.5, 0.1, '*Open-Play Successful Passes which ended inside Zone14 and Half-Spaces area', fontsize=10, fontstyle='italic', ha='center', va='center')
    
    himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png")
    himage = Image.open(himage)
    ax_himage = add_image(himage, fig, left=0.085, bottom=0.97, width=0.125, height=0.125)
    
    aimage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png")
    aimage = Image.open(aimage)
    ax_aimage = add_image(aimage, fig, left=0.815, bottom=0.97, width=0.125, height=0.125)
    
    st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f'{hteamName} Passers into Zone14 & Half-Spaces:')
        st.dataframe(home_z14hsi_stats)
    with col2:
        st.write(f'{ateamName} Passers into Zone14 & Half-Spaces:')
        st.dataframe(away_z14hsi_stats)
    
    st.header('Passes From Zone14 & Half-Spaces')
    
    def plot_zone14f(ax, team_name, col, phase_tag):
        if phase_tag == 'Full Time':
            pass_df = df[(df['teamName'] == team_name) & (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful') & (~df['qualifiers'].str.contains('Freekick|Corner'))]
        elif phase_tag == 'First Half':
            pass_df = df[(df['teamName'] == team_name) & (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful') & (~df['qualifiers'].str.contains('Freekick|Corner')) & (df['period'] == 'FirstHalf')]
        elif phase_tag == 'Second Half':
            pass_df = df[(df['teamName'] == team_name) & (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful') & (~df['qualifiers'].str.contains('Freekick|Corner')) & (df['period'] == 'SecondHalf')]
        
        pitch = VerticalPitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
        pitch.draw(ax=ax)
        
        z14_x = [68/3, 68/3, 136/3, 136/3]
        z14_y = [70, 88, 88, 70]
        ax.fill(z14_x, z14_y, color='orange', edgecolor='None', alpha=0.3)
        lhs_x = [136/3, 136/3, 272/5, 272/5]
        lhs_y = [70, 105, 105, 70]
        ax.fill(lhs_x, lhs_y, color=col, edgecolor='None', alpha=0.3)
        rhs__x = [68/5, 68/5, 68/3, 68/3]
        rhs__y = [70, 105, 105, 70]
        ax.fill(rhs__x, rhs__y, color=col, edgecolor='None', alpha=0.3)
        
        z14_pass = pass_df[(pass_df['x'] >= 70) & (pass_df['x'] <= 88) & (pass_df['y'] >= 68/3) & (pass_df['y'] <= 136/3)]
        pitch.lines(z14_pass.x, z14_pass.y, z14_pass.endX, z14_pass.endY, comet=True, lw=4, color='orange', zorder=4, ax=ax)
        pitch.scatter(z14_pass.endX, z14_pass.endY, s=75, color=bg_color, ec='orange', lw=2, zorder=5, ax=ax)
        z14_kp = z14_pass[z14_pass['qualifiers'].str.contains('KeyPass')]
        z14_as = z14_pass[z14_pass['qualifiers'].str.contains('GoalAssist')]
        
        lhs_pass = pass_df[(pass_df['x'] >= 70) & (pass_df['y'] >= 136/3) & (pass_df['y'] <= 272/5)]
        pitch.lines(lhs_pass.x, lhs_pass.y, lhs_pass.endX, lhs_pass.endY, comet=True, lw=4, color=col, zorder=4, ax=ax)
        pitch.scatter(lhs_pass.endX, lhs_pass.endY, s=75, color=bg_color, ec=col, lw=2, zorder=5, ax=ax)
        lhs_kp = lhs_pass[lhs_pass['qualifiers'].str.contains('KeyPass')]
        lhs_as = lhs_pass[lhs_pass['qualifiers'].str.contains('GoalAssist')]
        
        rhs_pass = pass_df[(pass_df['x'] >= 70) & (pass_df['y'] >= 68/5) & (pass_df['y'] <= 68/3)]
        pitch.lines(rhs_pass.x, rhs_pass.y, rhs_pass.endX, rhs_pass.endY, comet=True, lw=4, color=col, zorder=4, ax=ax)
        pitch.scatter(rhs_pass.endX, rhs_pass.endY, s=75, color=bg_color, ec=col, lw=2, zorder=5, ax=ax)
        rhs_kp = rhs_pass[rhs_pass['qualifiers'].str.contains('KeyPass')]
        rhs_as = rhs_pass[rhs_pass['qualifiers'].str.contains('GoalAssist')]
        
        pitch.scatter(17, 34, s=12000, color='orange', ec=line_color, lw=2, marker='h', zorder=7, ax=ax)
        pitch.scatter(35, 68/3, s=12000, color=col, ec=line_color, lw=2, marker='h', zorder=7, alpha=0.8, ax=ax)
        pitch.scatter(35, 136/3, s=12000, color=col, ec=line_color, lw=2, marker='h', zorder=7, alpha=0.8, ax=ax)
        
        ax.text(34, 21, 'Zone14', size=15, color='k', fontweight='bold', ha='center', va='center', zorder=10)
        ax.text(34, 16, f' \nTotal:{len(z14_pass)}\nKeyPass:{len(z14_kp)}\nAssist:{len(z14_as)}', size=13, color='k', ha='center', va='center', zorder=10)
        
        ax.text(136/3, 39, 'Left H.S.', size=15, color='k', fontweight='bold', ha='center', va='center', zorder=10)
        ax.text(136/3, 34, f' \nTotal:{len(lhs_pass)}\nKeyPass:{len(lhs_kp)}\nAssist:{len(lhs_as)}', size=13, color='k', ha='center', va='center', zorder=10)
        
        ax.text(68/3, 39, 'Right H.S.', size=15, color='k', fontweight='bold', ha='center', va='center', zorder=10)
        ax.text(68/3, 34, f' \nTotal:{len(rhs_pass)}\nKeyPass:{len(rhs_kp)}\nAssist:{len(rhs_as)}', size=13, color='k', ha='center', va='center', zorder=10)
        
        if phase_tag == 'Full Time':
            ax.text(34, 110, 'Full Time: 0-90 minutes', color=col, fontsize=13, ha='center', va='center')
        elif phase_tag == 'First Half':
            ax.text(34, 110, 'First Half: 0-45 minutes', color=col, fontsize=13, ha='center', va='center')
        elif phase_tag == 'Second Half':
            ax.text(34, 110, 'Second Half: 45-90 minutes', color=col, fontsize=13, ha='center', va='center')
        
        z14_pass['zone'] = 'Zone 14'
        lhs_pass['zone'] = 'Left Half Space'
        rhs_pass['zone'] = 'Right Half Space'
        total_df = pd.concat([z14_pass, lhs_pass, rhs_pass], ignore_index=True)
        total_df = total_df[['name', 'zone']]
        stats = total_df.groupby(['name', 'zone']).size().unstack(fill_value=0)
        stats['Total'] = stats.sum(axis=1)
        stats = stats.sort_values(by='Total', ascending=False)
        return stats

    fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
    zhsf_time_phase = st.pills(" ", ['Full Time', 'First Half', 'Second Half'], default='Full Time', key='From')
    if zhsf_time_phase == 'Full Time':
        home_z14hsf_stats = plot_zone14f(axs[0], hteamName, hcol, 'Full Time')
        away_z14hsf_stats = plot_zone14f(axs[1], ateamName, acol, 'Full Time')
    if zhsf_time_phase == 'First Half':
        home_z14hsf_stats = plot_zone14f(axs[0], hteamName, hcol, 'First Half')
        away_z14hsf_stats = plot_zone14f(axs[1], ateamName, acol, 'First Half')
    if zhsf_time_phase == 'Second Half':
        home_z14hsf_stats = plot_zone14f(axs[0], hteamName, hcol, 'Second Half')
        away_z14hsf_stats = plot_zone14f(axs[1], ateamName, acol, 'Second Half')
    
    fig_text(0.5, 1.03, f'<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>', highlight_textprops=[{'color': hcol}, {'color': acol}], fontsize=30, fontweight='bold', ha='center', va='center', ax=fig)
    fig.text(0.5, 0.99, 'Passes from Zone14 and Half-Spaces', fontsize=20, ha='center', va='center')
    fig.text(0.5, 0.95, '@adnaaan433', fontsize=10, ha='center', va='center')
    
    fig.text(0.5, 0.1, '*Open-Play Successful Passes which initiated inside Zone14 and Half-Spaces area', fontsize=10, fontstyle='italic', ha='center', va='center')
    
    st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f'{hteamName} Passers from Zone14 & Half-Spaces:')
        st.dataframe(home_z14hsf_stats)
    with col2:
        st.write(f'{ateamName} Passers from Zone14 & Half-Spaces:')
        st.dataframe(away_z14hsf_stats)

elif an_tp == 'Final Third Entries':
    st.header(f'{an_tp}')

    def final_third_entry(ax, team_name, col, phase_tag):
        if phase_tag == 'Full Time':
            fentry = df[(df['teamName'] == team_name) & (df['type'].isin(['Pass', 'Carry'])) & (df['outcomeType'] == 'Successful') & (~df['qualifiers'].str.contains('Freekick|Corner'))]
        elif phase_tag == 'First Half':
            fentry = df[(df['teamName'] == team_name) & (df['type'].isin(['Pass', 'Carry'])) & (df['outcomeType'] == 'Successful') & (~df['qualifiers'].str.contains('Freekick|Corner')) & (df['period'] == 'FirstHalf')]
        elif phase_tag == 'Second Half':
            fentry = df[(df['teamName'] == team_name) & (df['type'].isin(['Pass', 'Carry'])) & (df['outcomeType'] == 'Successful') & (~df['qualifiers'].str.contains('Freekick|Corner')) & (df['period'] == 'SecondHalf')]

        pitch = VerticalPitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
        pitch.draw(ax=ax)

        ax.hlines(70, xmin=0, xmax=68, color='gray', ls='--', lw=2)
        ax.vlines(68/3, ymin=0, ymax=70, color='gray', ls='--', lw=2)
        ax.vlines(136/3, ymin=0, ymax=70, color='gray', ls='--', lw=2)

        fep = fentry[(fentry['type'] == 'Pass') & (fentry['x'] < 70) & (fentry['endX'] > 70)]
        fec = fentry[(fentry['type'] == 'Carry') & (fentry['x'] < 70) & (fentry['endX'] > 70)]
        tfent = pd.concat([fep, fec], ignore_index=True)
        lent = tfent[tfent['y'] > 136/3]
        ment = tfent[(tfent['y'] <= 136/3) & (tfent['y'] >= 68/3)]
        rent = tfent[tfent['y'] < 68/3]

        pitch.lines(fep.x, fep.y, fep.endX, fep.endY, comet=True, lw=3, color=col, zorder=4, ax=ax)
        pitch.scatter(fep.endX, fep.endY, s=60, color=bg_color, ec=col, lw=1.5, zorder=5, ax=ax)
        for index, row in fec.iterrows():
            arrow = patches.FancyArrowPatch((row['y'], row['x']), (row['endY'], row['endX']), arrowstyle='->', color=violet, zorder=6, mutation_scale=20, 
                                            alpha=0.9, linewidth=3, linestyle='--')
            ax.add_patch(arrow)

        ax.text(340/6, -5, f"form left\n{len(lent)}", fontsize=13, color=line_color, ha='center', va='center')
        ax.text(34, -5, f"form mid\n{len(ment)}", fontsize=13, color=line_color, ha='center', va='center')
        ax.text(68/6, -5, f"form right\n{len(rent)}", fontsize=13, color=line_color, ha='center', va='center')

        if phase_tag == 'Full Time':
            ax.text(34, 112, 'Full Time: 0-90 minutes', color=col, fontsize=13, ha='center', va='center')
            ax_text(34, 108, f'Total: {len(fep)+len(fec)} | <By Pass: {len(fep)}> | <By Carry: {len(fec)}>', ax=ax, highlight_textprops=[{'color': col}, {'color': violet}],
                    color=line_color, fontsize=13, ha='center', va='center')
        elif phase_tag == 'First Half':
            ax.text(34, 112, 'First Half: 0-45 minutes', color=col, fontsize=13, ha='center', va='center')
            ax_text(34, 108, f'Total: {len(fep)+len(fec)} | <By Pass: {len(fep)}> | <By Carry: {len(fec)}>', ax=ax, highlight_textprops=[{'color': col}, {'color': violet}],
                    color=line_color, fontsize=13, ha='center', va='center')
        elif phase_tag == 'Second Half':
            ax.text(34, 112, 'Second Half: 45-90 minutes', color=col, fontsize=13, ha='center', va='center')
            ax_text(34, 108, f'Total: {len(fep)+len(fec)} | <By Pass: {len(fep)}> | <By Carry: {len(fec)}>', ax=ax, highlight_textprops=[{'color': col}, {'color': violet}],
                    color=line_color, fontsize=13, ha='center', va='center')

        tfent = tfent[['name', 'type']]
        stats = tfent.groupby(['name', 'type']).size().unstack(fill_value=0)
        stats['Total'] = stats.sum(axis=1)
        stats = stats.sort_values(by='Total', ascending=False)
        return stats

    fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
    fthE_time_phase = st.pills(" ", ['Full Time', 'First Half', 'Second Half'], default='Full Time', key='fthE_time_pill')
    if fthE_time_phase == 'Full Time':
        home_fthirdE_stats = final_third_entry(axs[0], hteamName, hcol, 'Full Time')
        away_fthirdE_stats = final_third_entry(axs[1], ateamName, acol, 'Full Time')
    if fthE_time_phase == 'First Half':
        home_fthirdE_stats = final_third_entry(axs[0], hteamName, hcol, 'First Half')
        away_fthirdE_stats = final_third_entry(axs[1], ateamName, acol, 'First Half')
    if fthE_time_phase == 'Second Half':
        home_fthirdE_stats = final_third_entry(axs[0], hteamName, hcol, 'Second Half')
        away_fthirdE_stats = final_third_entry(axs[1], ateamName, acol, 'Second Half')

    fig_text(0.5, 1.05, f'<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>', highlight_textprops=[{'color': hcol}, {'color': acol}], fontsize=30, fontweight='bold', ha='center', va='center', ax=fig)
    fig.text(0.5, 1.01, 'Final Third Entries', fontsize=20, ha='center', va='center')
    fig.text(0.5, 0.97, '@adnaaan433', fontsize=10, ha='center', va='center')

    fig.text(0.5, 0.05, '*Open-Play Successful Passes & Carries which ended inside the Final third, starting from outside the Final third', fontsize=10, 
             fontstyle='italic', ha='center', va='center')

    himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png")
    himage = Image.open(himage)
    ax_himage = add_image(himage, fig, left=0.085, bottom=0.97, width=0.125, height=0.125)

    aimage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png")
    aimage = Image.open(aimage)
    ax_aimage = add_image(aimage, fig, left=0.815, bottom=0.97, width=0.125, height=0.125)

    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.write(f'{hteamName} Players Final Third Entries:')
        st.dataframe(home_fthirdE_stats)
    with col2:
        st.write(f'{ateamName} Players Final Third Entries:')
        st.dataframe(away_fthirdE_stats)

elif an_tp == 'Box Entries':
    st.header(f'{an_tp}')

    def penalty_box_entry(ax, team_name, col, phase_tag):
        if phase_tag == 'Full Time':
            bentry = df[(df['type'].isin(['Pass', 'Carry'])) & (df['outcomeType'] == 'Successful') & (df['endX'] >= 88.5) &
                       ~((df['x'] >= 88.5) & (df['y'] >= 13.6) & (df['y'] <= 54.6)) & (df['endY'] >= 13.6) & (df['endY'] <= 54.4) &
                        (~df['qualifiers'].str.contains('CornerTaken|Freekick|ThrowIn'))]
        elif phase_tag == 'First Half':
            bentry = df[(df['type'].isin(['Pass', 'Carry'])) & (df['outcomeType'] == 'Successful') & (df['endX'] >= 88.5) &
                       ~((df['x'] >= 88.5) & (df['y'] >= 13.6) & (df['y'] <= 54.6)) & (df['endY'] >= 13.6) & (df['endY'] <= 54.4) &
                        (~df['qualifiers'].str.contains('CornerTaken|Freekick|ThrowIn')) & (df['period'] == 'FirstHalf')]
        elif phase_tag == 'Second Half':
            bentry = df[(df['type'].isin(['Pass', 'Carry'])) & (df['outcomeType'] == 'Successful') & (df['endX'] >= 88.5) &
                       ~((df['x'] >= 88.5) & (df['y'] >= 13.6) & (df['y'] <= 54.6)) & (df['endY'] >= 13.6) & (df['endY'] <= 54.4) &
                        (~df['qualifiers'].str.contains('CornerTaken|Freekick|ThrowIn')) & (df['period'] == 'SecondHalf')]

        pitch = VerticalPitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True, half=True)
        pitch.draw(ax=ax)

        bep = bentry[(bentry['type'] == 'Pass') & (bentry['teamName'] == team_name)]
        bec = bentry[(bentry['type'] == 'Carry') & (bentry['teamName'] == team_name)]
        tbent = pd.concat([bep, bec], ignore_index=True)
        lent = tbent[tbent['y'] > 136/3]
        ment = tbent[(tbent['y'] <= 136/3) & (tbent['y'] >= 68/3)]
        rent = tbent[tbent['y'] < 68/3]

        pitch.lines(bep.x, bep.y, bep.endX, bep.endY, comet=True, lw=3, color=col, zorder=4, ax=ax)
        pitch.scatter(bep.endX, bep.endY, s=60, color=bg_color, ec=col, lw=1.5, zorder=5, ax=ax)
        for index, row in bec.iterrows():
            arrow = patches.FancyArrowPatch((row['y'], row['x']), (row['endY'], row['endX']), arrowstyle='->', color=violet, zorder=6, mutation_scale=20, 
                                            alpha=0.9, linewidth=3, linestyle='--')
            ax.add_patch(arrow)

        ax.text(340/6, 46, f"form left\n{len(lent)}", fontsize=13, color=line_color, ha='center', va='center')
        ax.text(34, 46, f"form mid\n{len(ment)}", fontsize=13, color=line_color, ha='center', va='center')
        ax.text(68/6, 46, f"form right\n{len(rent)}", fontsize=13, color=line_color, ha='center', va='center')
        ax.vlines(68/3, ymin=0, ymax=88.5, color='gray', ls='--', lw=2)
        ax.vlines(136/3, ymin=0, ymax=88.5, color='gray', ls='--', lw=2)

        if phase_tag == 'Full Time':
            ax.text(34, 112, 'Full Time: 0-90 minutes', color=col, fontsize=13, ha='center', va='center')
            ax_text(34, 108, f'Total: {len(bep)+len(bec)} | <By Pass: {len(bep)}> | <By Carry: {len(bec)}>', ax=ax, highlight_textprops=[{'color': col}, {'color': violet}],
                    color=line_color, fontsize=13, ha='center', va='center')
        elif phase_tag == 'First Half':
            ax.text(34, 112, 'First Half: 0-45 minutes', color=col, fontsize=13, ha='center', va='center')
            ax_text(34, 108, f'Total: {len(bep)+len(bec)} | <By Pass: {len(bep)}> | <By Carry: {len(bec)}>', ax=ax, highlight_textprops=[{'color': col}, {'color': violet}],
                    color=line_color, fontsize=13, ha='center', va='center')
        elif phase_tag == 'Second Half':
            ax.text(34, 112, 'Second Half: 45-90 minutes', color=col, fontsize=13, ha='center', va='center')
            ax_text(34, 108, f'Total: {len(bep)+len(bec)} | <By Pass: {len(bep)}> | <By Carry: {len(bec)}>', ax=ax, highlight_textprops=[{'color': col}, {'color': violet}],
                    color=line_color, fontsize=13, ha='center', va='center')
            
        tbent = tbent[['name', 'type']]
        stats = tbent.groupby(['name', 'type']).size().unstack(fill_value=0)
        stats['Total'] = stats.sum(axis=1)
        stats = stats.sort_values(by='Total', ascending=False)
        return stats

    fig, axs = plt.subplots(1, 2, figsize=(15, 6), facecolor=bg_color)
    bent_time_phase = st.pills(" ", ['Full Time', 'First Half', 'Second Half'], default='Full Time', key='bent_time_pill')
    if bent_time_phase == 'Full Time':
        home_boxE_stats = penalty_box_entry(axs[0], hteamName, hcol, 'Full Time')
        away_boxE_stats = penalty_box_entry(axs[1], ateamName, acol, 'Full Time')
    if bent_time_phase == 'First Half':
        home_boxE_stats = penalty_box_entry(axs[0], hteamName, hcol, 'First Half')
        away_boxE_stats = penalty_box_entry(axs[1], ateamName, acol, 'First Half')
    if bent_time_phase == 'Second Half':
        home_boxE_stats = penalty_box_entry(axs[0], hteamName, hcol, 'Second Half')
        away_boxE_stats = penalty_box_entry(axs[1], ateamName, acol, 'Second Half')

    fig_text(0.5, 1.08, f'<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>', highlight_textprops=[{'color': hcol}, {'color': acol}], fontsize=30, fontweight='bold', ha='center', va='center', ax=fig)
    fig.text(0.5, 1.01, "Opponent's Penalty Box Entries", fontsize=20, ha='center', va='center')
    fig.text(0.5, 0.96, '@adnaaan433', fontsize=10, ha='center', va='center')

    fig.text(0.5, 0.00, '*Open-Play Successful Passes & Carries which ended inside the Opponent Penalty Box, starting from outside the Penalty Box', fontsize=10, 
             fontstyle='italic', ha='center', va='center')

    himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png")
    himage = Image.open(himage)
    ax_himage = add_image(himage, fig, left=0.065, bottom=0.99, width=0.16, height=0.16)

    aimage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png")
    aimage = Image.open(aimage)
    ax_aimage = add_image(aimage, fig, left=0.8, bottom=0.99, width=0.16, height=0.16)

    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.write(f'{hteamName} Players Penalty Box Entries:')
        st.dataframe(home_boxE_stats)
    with col2:
        st.write(f'{ateamName} Players Penalty Box Entries:')
        st.dataframe(away_boxE_stats)

elif an_tp == 'High-Turnovers':
    st.header(f'{an_tp}')

    def plot_high_turnover(ax, team_name, col, phase_tag):
        if phase_tag == 'Full Time':
            dfhto = df.copy()
        elif phase_tag == 'First Half':
            dfhto = df[df['period'] == 'FirstHalf']
        elif phase_tag == 'Second Half':
            dfhto = df[df['period'] == 'SecondHalf'].reset_index(drop=True)

        pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
        pitch.draw(ax=ax)
        ax.set_ylim(-0.5, 105.5)
        ax.set_xlim(68.5, -0.5)

        dfhto['Distance'] = ((dfhto['x'] - 105)**2 + (dfhto['y'] - 34)**2)**0.5

        goal_count = 0
        p_goal_list = []
        p_blost_goal = []
        # Iterate through the DataFrame
        for i in range(len(dfhto)):
            if ((dfhto.loc[i, 'type'] in ['BallRecovery', 'Interception']) and 
                (dfhto.loc[i, 'teamName'] == team_name) and 
                (dfhto.loc[i, 'Distance'] <= 40)):
                
                possession_id = dfhto.loc[i, 'possession_id']
                
                # Check the following rows within the same possession
                j = i + 1
                while j < len(dfhto) and dfhto.loc[j, 'possession_id'] == possession_id and dfhto.loc[j, 'teamName'] == team_name:
                    if dfhto.loc[j, 'type'] == 'Goal' and dfhto.loc[j, 'teamName'] == team_name:
                        pitch.scatter(dfhto.loc[i, 'x'], dfhto.loc[i, 'y'], s=1000, marker='*', color='green', edgecolor=bg_color, zorder=3, ax=ax)
                        goal_count += 1
                        p_goal_list.append(dfhto.loc[i, 'name'])
                        # Check the ball looser
                        k = i - 1
                        while k > i - 10:
                            if dfhto.loc[k, 'teamName'] != team_name:
                                p_blost_goal.append(dfhto.loc[k, 'name'])
                                break
                            k = k - 1
                        break
                    j += 1

        shot_count = 0
        p_shot_list = []
        p_blost_shot = []
        # Iterate through the DataFrame
        for i in range(len(dfhto)):
            if ((dfhto.loc[i, 'type'] in ['BallRecovery', 'Interception']) and 
                (dfhto.loc[i, 'teamName'] == team_name) and 
                (dfhto.loc[i, 'Distance'] <= 40)):
                
                possession_id = dfhto.loc[i, 'possession_id']
                
                # Check the following rows within the same possession
                j = i + 1
                while j < len(dfhto) and dfhto.loc[j, 'possession_id'] == possession_id and dfhto.loc[j, 'teamName'] == team_name:
                    if ('Shot' in dfhto.loc[j, 'type']) and (dfhto.loc[j, 'teamName'] == team_name):
                        pitch.scatter(dfhto.loc[i, 'x'], dfhto.loc[i, 'y'], s=200, color=col, edgecolor=bg_color, zorder=2, ax=ax)
                        shot_count += 1
                        p_shot_list.append(dfhto.loc[i, 'name'])
                        # Check the ball looser
                        k = i - 1
                        while k > i - 10:
                            if dfhto.loc[k, 'teamName'] != team_name:
                                p_blost_shot.append(dfhto.loc[k, 'name'])
                                break
                            k = k - 1
                        break
                    j += 1

        ht_count = 0
        p_hto_list = []
        p_blost = []
        # Iterate through the DataFrame
        for i in range(len(dfhto)):
            if ((dfhto.loc[i, 'type'] in ['BallRecovery', 'Interception']) and 
                (dfhto.loc[i, 'teamName'] == team_name) and 
                (dfhto.loc[i, 'Distance'] <= 40)):
                
                # Check the following rows
                j = i + 1
                if ((dfhto.loc[j, 'teamName'] == team_name) and
                    (dfhto.loc[j, 'type'] != 'Dispossessed') and (dfhto.loc[j, 'type'] != 'OffsidePass')):
                    pitch.scatter(dfhto.loc[i, 'x'], dfhto.loc[i, 'y'], s=200, color='None', edgecolor=col, ax=ax)
                    ht_count += 1
                    p_hto_list.append(dfhto.loc[i, 'name'])

                # Check the ball looser
                k = i - 1
                while k > i - 10:
                    if dfhto.loc[k, 'teamName'] != team_name:
                        p_blost.append(dfhto.loc[k, 'name'])
                        break
                    k = k - 1

        # Plotting the half circle
        left_circle = plt.Circle((34, 105), 40, color=col, fill=True, alpha=0.25, lw=2, linestyle='dashed')
        ax.add_artist(left_circle)

        ax.scatter(34, 35, s=12000, marker='h', color=col, edgecolor=line_color, lw=2)
        ax.scatter(136/3, 18, s=12000, marker='h', color=col, edgecolor=line_color, lw=2)
        ax.scatter(68/3, 18, s=12000, marker='h', color=col, edgecolor=line_color, lw=2)
        ax.text(34, 35, f'Total:\n{ht_count}', color=bg_color, fontsize=18, fontweight='bold', ha='center', va='center')
        ax.text(136/3, 18, f'Led\nto Shot:\n{shot_count}', color=bg_color, fontsize=18, fontweight='bold', ha='center', va='center')
        ax.text(68/3, 18, f'Led\nto Goal:\n{goal_count}', color=bg_color, fontsize=18, fontweight='bold', ha='center', va='center')

        if phase_tag == 'Full Time':
            ax.text(34, 109, 'Full Time: 0-90 minutes', color=col, fontsize=13, ha='center', va='center')
        elif phase_tag == 'First Half':
            ax.text(34, 109, 'First Half: 0-45 minutes', color=col, fontsize=13, ha='center', va='center')
        elif phase_tag == 'Second Half':
            ax.text(34, 109, 'Second Half: 45-90 minutes', color=col, fontsize=13, ha='center', va='center')

        unique_players = set(p_hto_list + p_shot_list + p_goal_list)
        player_hto_data = {
            'Name': list(unique_players),
            'Total_High_Turnovers': [p_hto_list.count(player) for player in unique_players],
            'Led_to_Shot': [p_shot_list.count(player) for player in unique_players],
            'Led_to_Goal': [p_goal_list.count(player) for player in unique_players],
        }
        
        player_hto_stats = pd.DataFrame(player_hto_data)
        player_hto_stats = player_hto_stats.sort_values(by=['Total_High_Turnovers', 'Led_to_Goal', 'Led_to_Shot'], ascending=[False, False, False])

        unique_players = set(p_blost + p_blost_shot + p_blost_goal)
        player_blost_data = {
            'Name': list(unique_players),
            'Ball_Loosing_Led_to_High_Turnovers': [p_blost.count(player) for player in unique_players],
            'Led_to_Shot': [p_blost_shot.count(player) for player in unique_players],
            'Led_to_Goal': [p_blost_goal.count(player) for player in unique_players],
        }
        
        player_blost_stats = pd.DataFrame(player_blost_data)
        player_blost_stats = player_blost_stats.sort_values(by=['Ball_Loosing_Led_to_High_Turnovers', 'Led_to_Goal', 'Led_to_Shot'], ascending=[False, False, False])

        return player_hto_stats, player_blost_stats

    fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
    hto_time_phase = st.pills(" ", ['Full Time', 'First Half', 'Second Half'], default='Full Time', key='hto_time_pill')
    if hto_time_phase == 'Full Time':
        home_hto_stats, home_blost_stats = plot_high_turnover(axs[0], hteamName, hcol, 'Full Time')
        away_hto_stats, away_blost_stats = plot_high_turnover(axs[1], ateamName, acol, 'Full Time')
    if hto_time_phase == 'First Half':
        home_hto_stats, home_blost_stats = plot_high_turnover(axs[0], hteamName, hcol, 'First Half')
        away_hto_stats, away_blost_stats = plot_high_turnover(axs[1], ateamName, acol, 'First Half')
    if hto_time_phase == 'Second Half':
        home_hto_stats, home_blost_stats = plot_high_turnover(axs[0], hteamName, hcol, 'Second Half')
        away_hto_stats, away_blost_stats = plot_high_turnover(axs[1], ateamName, acol, 'Second Half')

    fig_text(0.5, 1.05, f'<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>', highlight_textprops=[{'color': hcol}, {'color': acol}], fontsize=30, fontweight='bold', ha='center', va='center', ax=fig)
    fig.text(0.5, 1.01, 'High Turnovers', fontsize=20, ha='center', va='center')
    fig.text(0.5, 0.97, '@adnaaan433', fontsize=10, ha='center', va='center')

    fig.text(0.5, 0.05, '*High Turnovers means winning possession within the 40m radius from the Opponent Goal Center', fontsize=10, 
             fontstyle='italic', ha='center', va='center')

    himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png")
    himage = Image.open(himage)
    ax_himage = add_image(himage, fig, left=0.085, bottom=0.97, width=0.125, height=0.125)

    aimage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png")
    aimage = Image.open(aimage)
    ax_aimage = add_image(aimage, fig, left=0.815, bottom=0.97, width=0.125, height=0.125)

    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.write(f'{hteamName} Ball Winners for High-Turnovers:')
        st.dataframe(home_hto_stats, hide_index=True)
        st.write(f'{hteamName} Ball Losers for High-Turnovers:')
        st.dataframe(away_blost_stats, hide_index=True)
    with col2:
        st.write(f'{ateamName} Ball Winners for High-Turnovers:')
        st.dataframe(away_hto_stats, hide_index=True)
        st.write(f'{ateamName} Ball Losers for High-Turnovers:')
        st.dataframe(home_blost_stats, hide_index=True)

elif an_tp == 'Chances Creating Zones':
    st.header(f'{an_tp}')
if an_tp == 'Chances Creating Zones':
    st.header(f'{an_tp}')

    def plot_cc_zone(ax, team_name, col, phase_tag):
        if phase_tag == 'Full Time':
            dfcc = df[(df['teamName'] == team_name) & (df['qualifiers'].str.contains('KeyPass'))]
        elif phase_tag == 'First Half':
            dfcc = df[(df['teamName'] == team_name) & (df['qualifiers'].str.contains('KeyPass')) & (df['period'] == 'FirstHalf')]
        elif phase_tag == 'Second Half':
            dfcc = df[(df['teamName'] == team_name) & (df['qualifiers'].str.contains('KeyPass')) & (df['period'] == 'SecondHalf')]

        pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, line_zorder=2, linewidth=2)
        pitch.draw(ax=ax)

        dfass = dfcc[dfcc['qualifiers'].str.contains('GoalAssist')]
        opcc = dfcc[~dfcc['qualifiers'].str.contains('Corner|Freekick')]

        pc_map = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors", [bg_color, col], N=20)
        path_eff = [path_effects.Stroke(linewidth=3, foreground=bg_color), path_effects.Normal()]
        bin_statistic = pitch.bin_statistic(dfcc.x, dfcc.y, bins=(6, 5), statistic='count', normalize=False)
        pitch.heatmap(bin_statistic, ax=ax, cmap=pc_map, edgecolors='#ededed')
        pitch.label_heatmap(bin_statistic, color=line_color, fontsize=25, ax=ax, ha='center', va='center', zorder=5, str_format='{:.0f}', path_effects=path_eff)

        pitch.lines(dfcc.x, dfcc.y, dfcc.endX, dfcc.endY, color=violet, comet=True, lw=4, zorder=3, ax=ax)
        pitch.scatter(dfcc.endX, dfcc.endY, s=50, linewidth=1, color=bg_color, edgecolor=violet, zorder=4, ax=ax)
        pitch.lines(dfass.x, dfass.y, dfass.endX, dfass.endY, color=green, comet=True, lw=5, zorder=3, ax=ax)
        pitch.scatter(dfass.endX, dfass.endY, s=75, linewidth=1, color=bg_color, edgecolor=green, zorder=4, ax=ax)

        all_cc = dfcc.name.to_list()
        op_cc = opcc.name.to_list()
        ass_c = dfass.name.to_list()
        unique_players = set(all_cc + op_cc + ass_c)
        player_cc_data = {
            'Name': list(unique_players),
            'Total_Chances_Created': [all_cc.count(player) for player in unique_players],
            'OpenPlay_Chances_Created': [op_cc.count(player) for player in unique_players],
            'Assists': [ass_c.count(player) for player in unique_players]
        }

        player_cc_stats = pd.DataFrame(player_cc_data)
        player_cc_stats = player_cc_stats.sort_values(by=['Total_Chances_Created', 'OpenPlay_Chances_Created', 'Assists'], ascending=[False, False, False]).reset_index(drop=True)

        most_by = player_cc_stats.Name.to_list()[0]
        most_count = player_cc_stats.Total_Chances_Created.to_list()[0]

        if phase_tag == 'Full Time':
            ax.text(34, 116, 'Full Time: 0-90 minutes', color=col, fontsize=13, ha='center', va='center')
            ax.text(34, 112, f'Total Chances: {len(dfcc)} | Open-Play Chances: {len(opcc)}', color=col, fontsize=13, ha='center', va='center')
            ax.text(34, 108, f'Most by: {most_by} ({most_count})', color=col, fontsize=13, ha='center', va='center')
        elif phase_tag == 'First Half':
            ax.text(34, 116, 'First Half: 0-45 minutes', color=col, fontsize=13, ha='center', va='center')
            ax.text(34, 112, f'Total Chances: {len(dfcc)} | Open-Play Chances: {len(opcc)}', color=col, fontsize=13, ha='center', va='center')
            ax.text(34, 108, f'Most by: {most_by} ({most_count})', color=col, fontsize=13, ha='center', va='center')
        elif phase_tag == 'Second Half':
            ax.text(34, 116, 'Second Half: 45-90 minutes', color=col, fontsize=13, ha='center', va='center')
            ax.text(34, 112, f'Total Chances: {len(dfcc)} | Open-Play Chances: {len(opcc)}', color=col, fontsize=13, ha='center', va='center')
            ax.text(34, 108, f'Most by: {most_by} ({most_count})', color=col, fontsize=13, ha='center', va='center')

        return player_cc_stats

    fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
    cc_time_phase = st.pills(" ", ['Full Time', 'First Half', 'Second Half'], default='Full Time', key='cc_time_pill')
    if cc_time_phase == 'Full Time':
        home_cc_stats = plot_cc_zone(axs[0], hteamName, hcol, 'Full Time')
        away_cc_stats = plot_cc_zone(axs[1], ateamName, acol, 'Full Time')
    if cc_time_phase == 'First Half':
        home_cc_stats = plot_cc_zone(axs[0], hteamName, hcol, 'First Half')
        away_cc_stats = plot_cc_zone(axs[1], ateamName, acol, 'First Half')
    if cc_time_phase == 'Second Half':
        home_cc_stats = plot_cc_zone(axs[0], hteamName, hcol, 'Second Half')
        away_cc_stats = plot_cc_zone(axs[1], ateamName, acol, 'Second Half')

    fig_text(0.5, 1.05, f'<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>', highlight_textprops=[{'color': hcol}, {'color': acol}], fontsize=30, 
             fontweight='bold', ha='center', va='center', ax=fig)
    fig.text(0.5, 1.01, 'Chances Creating Zones', fontsize=20, ha='center', va='center')
    fig.text(0.5, 0.97, '@adnaaan433', fontsize=10, ha='center', va='center')

    himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png")
    himage = Image.open(himage)
    ax_himage = add_image(himage, fig, left=0.085, bottom=0.97, width=0.125, height=0.125)

    aimage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png")
    aimage = Image.open(aimage)
    ax_aimage = add_image(aimage, fig, left=0.815, bottom=0.97, width=0.125, height=0.125)

    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.write(f'{hteamName} Players Crossing Stats:')
        st.dataframe(home_cc_stats, hide_index=True)
    with col2:
        st.write(f'{ateamName} Players Crossing Stats:')
        st.dataframe(away_cc_stats, hide_index=True)

if an_tp == 'Crosses':
    st.header(f'{an_tp}')

    def plot_crossed(ax, team_name, col, phase_tag):
        if phase_tag == 'Full Time':
            dfcrs = df[(df['teamName'] == team_name) & (df['qualifiers'].str.contains('Cross')) & (~df['qualifiers'].str.contains('Corner|Freekick'))]
        elif phase_tag == 'First Half':
            dfcrs = df[(df['teamName'] == team_name) & (df['qualifiers'].str.contains('Cross')) & (~df['qualifiers'].str.contains('Corner|Freekick')) & (df['period'] == 'FirstHalf')]
        elif phase_tag == 'Second Half':
            dfcrs = df[(df['teamName'] == team_name) & (df['qualifiers'].str.contains('Cross')) & (~df['qualifiers'].str.contains('Corner|Freekick')) & (df['period'] == 'SecondHalf')]

        pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2, half=True)
        pitch.draw(ax=ax)

        right_s = dfcrs[(dfcrs['y'] < 34) & (dfcrs['outcomeType'] == 'Successful')]
        right_u = dfcrs[(dfcrs['y'] < 34) & (dfcrs['outcomeType'] == 'Unsuccessful')]
        left_s = dfcrs[(dfcrs['y'] > 34) & (dfcrs['outcomeType'] == 'Successful')]
        left_u = dfcrs[(dfcrs['y'] > 34) & (dfcrs['outcomeType'] == 'Unsuccessful')]
        right_df = pd.concat([right_s, right_u], ignore_index=True)
        left_df = pd.concat([left_s, left_u], ignore_index=True)
        success_ful = pd.concat([right_s, left_s], ignore_index=True)
        unsuccess_ful = pd.concat([right_u, left_u], ignore_index=True)
        keypass = dfcrs[dfcrs['qualifiers'].str.contains('KeyPass')]
        assist = dfcrs[dfcrs['qualifiers'].str.contains('GoalAssist')]

        for index, row in success_ful.iterrows():
            arrow = patches.FancyArrowPatch((row['y'], row['x']), (row['endY'], row['endX']), arrowstyle='->', color=col, zorder=4, mutation_scale=20, 
                                            alpha=0.9, linewidth=3)
            ax.add_patch(arrow)
        for index, row in unsuccess_ful.iterrows():
            arrow = patches.FancyArrowPatch((row['y'], row['x']), (row['endY'], row['endX']), arrowstyle='->', color='gray', zorder=3, mutation_scale=15, 
                                            alpha=0.7, linewidth=2)
            ax.add_patch(arrow)
        for index, row in keypass.iterrows():
            arrow = patches.FancyArrowPatch((row['y'], row['x']), (row['endY'], row['endX']), arrowstyle='->', color=violet, zorder=5, mutation_scale=20, 
                                            alpha=0.9, linewidth=3.5)
            ax.add_patch(arrow)
        for index, row in assist.iterrows():
            arrow = patches.FancyArrowPatch((row['y'], row['x']), (row['endY'], row['endX']), arrowstyle='->', color='green', zorder=6, mutation_scale=20, 
                                            alpha=0.9, linewidth=3.5)
            ax.add_patch(arrow)

        most_by = dfcrs['name'].value_counts().idxmax() if not dfcrs.empty else None
        most_count = dfcrs['name'].value_counts().max() if not dfcrs.empty else 0
        most_left = left_df['shortName'].value_counts().idxmax() if not left_df.empty else None
        left_count = left_df['shortName'].value_counts().max() if not left_df.empty else 0
        most_right = right_df['shortName'].value_counts().idxmax() if not right_df.empty else None
        right_count = right_df['shortName'].value_counts().max() if not right_df.empty else 0

        if phase_tag == 'Full Time':
            ax.text(34, 116, 'Full Time: 0-90 minutes', color=col, fontsize=13, ha='center', va='center')
            ax_text(34, 112, f'Total Attempts: {len(dfcrs)} | <Successful: {len(right_s)+len(left_s)}> | <Unsuccessful: {len(right_u)+len(left_u)}>', color=line_color, fontsize=12, ha='center', va='center',
                    highlight_textprops=[{'color': col}, {'color': 'gray'}], ax=ax)
            ax.text(34, 108, f'Most by: {most_by} ({most_count})', color=col, fontsize=13, ha='center', va='center')
        elif phase_tag == 'First Half':
            ax.text(34, 116, 'First Half: 0-45 minutes', color=col, fontsize=13, ha='center', va='center')
            ax_text(34, 112, f'Total Attempts: {len(dfcrs)} | <Successful: {len(right_s)+len(left_s)}> | <Unsuccessful: {len(right_u)+len(left_u)}>', color=line_color, fontsize=12, ha='center', va='center',
                    highlight_textprops=[{'color': col}, {'color': 'gray'}], ax=ax)
            ax.text(34, 108, f'Most by: {most_by} ({most_count})', color=col, fontsize=13, ha='center', va='center')
        elif phase_tag == 'Second Half':
            ax.text(34, 116, 'Second Half: 45-90 minutes', color=col, fontsize=13, ha='center', va='center')
            ax_text(34, 112, f'Total Attempts: {len(dfcrs)} | <Successful: {len(right_s)+len(left_s)}> | <Unsuccessful: {len(right_u)+len(left_u)}>', color=line_color, fontsize=12, ha='center', va='center',
                    highlight_textprops=[{'color': col}, {'color': 'gray'}], ax=ax)
            ax.text(34, 108, f'Most by: {most_by} ({most_count})', color=col, fontsize=13, ha='center', va='center')

        ax.text(68, 46, f'From Left: {len(left_s)+len(left_u)}\nAccurate: {len(left_s)}\n\nMost by:\n{most_left} ({left_count})', color=col, fontsize=13, ha='left', va='top')
        ax.text(0, 46, f'From Right: {len(right_s)+len(right_u)}\nAccurate: {len(right_s)}\n\nMost by:\n{most_right} ({right_count})', color=col, fontsize=13, ha='right', va='top')

        all_crs = dfcrs.name.to_list()
        suc_crs = success_ful.name.to_list()
        uns_crs = unsuccess_ful.name.to_list()
        kp_crs = keypass.name.to_list()
        as_crs = assist.name.to_list()

        unique_players = set(all_crs + suc_crs + uns_crs)
        player_crs_data = {
            'Name': list(unique_players),
            'Total_Crosses': [all_crs.count(player) for player in unique_players],
            'Successful': [suc_crs.count(player) for player in unique_players],
            'Unsuccessful': [uns_crs.count(player) for player in unique_players],
            'Key_Pass_from_Cross': [kp_crs.count(player) for player in unique_players],
            'Assist_from_Cross': [as_crs.count(player) for player in unique_players],
        }

        player_crs_stats = pd.DataFrame(player_crs_data)
        player_crs_stats = player_crs_stats.sort_values(by=['Total_Crosses', 'Successful', 'Key_Pass_from_Cross', 'Assist_from_Cross'], ascending=[False, False, False, False])

        return player_crs_stats

    fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
    crs_time_phase = st.pills(" ", ['Full Time', 'First Half', 'Second Half'], default='Full Time', key='crs_time_pill')
    if crs_time_phase == 'Full Time':
        home_crs_stats = plot_crossed(axs[0], hteamName, hcol, 'Full Time')
        away_crs_stats = plot_crossed(axs[1], ateamName, acol, 'Full Time')
    if crs_time_phase == 'First Half':
        home_crs_stats = plot_crossed(axs[0], hteamName, hcol, 'First Half')
        away_crs_stats = plot_crossed(axs[1], ateamName, acol, 'First Half')
    if crs_time_phase == 'Second Half':
        home_crs_stats = plot_crossed(axs[0], hteamName, hcol, 'Second Half')
        away_crs_stats = plot_crossed(axs[1], ateamName, acol, 'Second Half')

    fig_text(0.5, 0.89, f'<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>', highlight_textprops=[{'color': hcol}, {'color': acol}], fontsize=30, fontweight='bold', ha='center', va='center', ax=fig)
    fig.text(0.5, 0.85, 'Open-Play Crosses', fontsize=20, ha='center', va='center')
    fig.text(0.5, 0.81, '@adnaaan433', fontsize=10, ha='center', va='center')

    fig.text(0.5, 0.1, '*Violet Arrow = KeyPass from Cross | Green Arrow = Assist from Cross', fontsize=10, fontstyle='italic', ha='center', va='center')

    himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png")
    himage = Image.open(himage)
    ax_himage = add_image(himage, fig, left=0.085, bottom=0.8, width=0.125, height=0.125)

    aimage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png")
    aimage = Image.open(aimage)
    ax_aimage = add_image(aimage, fig, left=0.815, bottom=0.8, width=0.125, height=0.125)

    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.write(f'{hteamName} Players Crossing Stats:')
        st.dataframe(home_crs_stats, hide_index=True)
    with col2:
        st.write(f'{ateamName} Players Crossing Stats:')
        st.dataframe(away_crs_stats, hide_index=True)

if an_tp == 'Team Domination Zones':
    st.header(f'{an_tp}')

    def plot_congestion(ax, phase_tag):
        if phase_tag == 'Full Time':
            dfdz = df.copy()
            ax.text(52.5, 76, 'Full Time: 0-90 minutes', fontsize=15, ha='center', va='center')
        elif phase_tag == 'First Half':
            dfdz = df[df['period'] == 'FirstHalf']
            ax.text(52.5, 76, 'First Half: 0-45 minutes', fontsize=15, ha='center', va='center')
        elif phase_tag == 'Second Half':
            dfdz = df[df['period'] == 'SecondHalf']
            ax.text(52.5, 76, 'Second Half: 0-45 minutes', fontsize=15, ha='center', va='center')
        pcmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors", [acol, 'gray', hcol], N=20)
        df1 = dfdz[(dfdz['teamName'] == hteamName) & (dfdz['isTouch'] == 1) & (~dfdz['qualifiers'].str.contains('CornerTaken|Freekick|ThrowIn'))]
        df2 = dfdz[(dfdz['teamName'] == ateamName) & (dfdz['isTouch'] == 1) & (~dfdz['qualifiers'].str.contains('CornerTaken|Freekick|ThrowIn'))]
        df2['x'] = 105 - df2['x']
        df2['y'] = 68 - df2['y']
        pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2, line_zorder=6)
        pitch.draw(ax=ax)
        ax.set_ylim(-0.5, 68.5)
        ax.set_xlim(-0.5, 105.5)

        bin_statistic1 = pitch.bin_statistic(df1.x, df1.y, bins=(6, 5), statistic='count', normalize=False)
        bin_statistic2 = pitch.bin_statistic(df2.x, df2.y, bins=(6, 5), statistic='count', normalize=False)

        # Assuming 'cx' and 'cy' are as follows:
        cx = np.array([[ 8.75, 26.25, 43.75, 61.25, 78.75, 96.25],
                       [ 8.75, 26.25, 43.75, 61.25, 78.75, 96.25],
                       [ 8.75, 26.25, 43.75, 61.25, 78.75, 96.25],
                       [ 8.75, 26.25, 43.75, 61.25, 78.75, 96.25],
                       [ 8.75, 26.25, 43.75, 61.25, 78.75, 96.25]])

        cy = np.array([[61.2, 61.2, 61.2, 61.2, 61.2, 61.2],
                       [47.6, 47.6, 47.6, 47.6, 47.6, 47.6],
                       [34.0, 34.0, 34.0, 34.0, 34.0, 34.0],
                       [20.4, 20.4, 20.4, 20.4, 20.4, 20.4],
                       [ 6.8,  6.8,  6.8,  6.8,  6.8,  6.8]])

        # Flatten the arrays
        cx_flat = cx.flatten()
        cy_flat = cy.flatten()

        # Create a DataFrame
        df_cong = pd.DataFrame({'cx': cx_flat, 'cy': cy_flat})

        hd_values = []

        # Loop through the 2D arrays
        for i in range(bin_statistic1['statistic'].shape[0]):
            for j in range(bin_statistic1['statistic'].shape[1]):
                stat1 = bin_statistic1['statistic'][i, j]
                stat2 = bin_statistic2['statistic'][i, j]

                if (stat1 / (stat1 + stat2)) > 0.55:
                    hd_values.append(1)
                elif (stat1 / (stat1 + stat2)) < 0.45:
                    hd_values.append(0)
                else:
                    hd_values.append(0.5)

        df_cong['hd'] = hd_values
        bin_stat = pitch.bin_statistic(df_cong.cx, df_cong.cy, bins=(6, 5), values=df_cong['hd'], statistic='sum', normalize=False)
        pitch.heatmap(bin_stat, ax=ax, cmap=pcmap, edgecolors='#000000', lw=0, zorder=3, alpha=0.85)

        ax_text(52.5, 71, s=f"<{hteamName}>  |  Contested  |  <{ateamName}>", highlight_textprops=[{'color': hcol}, {'color': acol}],
                color='gray', fontsize=18, ha='center', va='center', ax=ax)
        # ax.set_title("Team's Dominating Zone", color=line_color, fontsize=30, fontweight='bold', y=1.075)
        ax.text(0, -3, f'{hteamName}\nAttacking Direction--->', color=hcol, fontsize=13, ha='left', va='top')
        ax.text(105, -3, f'{ateamName}\n<---Attacking Direction', color=acol, fontsize=13, ha='right', va='top')

        ax.vlines(1*(105/6), ymin=0, ymax=68, color=bg_color, lw=2, ls='--', zorder=5)
        ax.vlines(2*(105/6), ymin=0, ymax=68, color=bg_color, lw=2, ls='--', zorder=5)
        ax.vlines(3*(105/6), ymin=0, ymax=68, color=bg_color, lw=2, ls='--', zorder=5)
        ax.vlines(4*(105/6), ymin=0, ymax=68, color=bg_color, lw=2, ls='--', zorder=5)
        ax.vlines(5*(105/6), ymin=0, ymax=68, color=bg_color, lw=2, ls='--', zorder=5)

        ax.hlines(1*(68/5), xmin=0, xmax=105, color=bg_color, lw=2, ls='--', zorder=5)
        ax.hlines(2*(68/5), xmin=0, xmax=105, color=bg_color, lw=2, ls='--', zorder=5)
        ax.hlines(3*(68/5), xmin=0, xmax=105, color=bg_color, lw=2, ls='--', zorder=5)
        ax.hlines(4*(68/5), xmin=0, xmax=105, color=bg_color, lw=2, ls='--', zorder=5)

        return

    fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)
    tdz_time_phase = st.pills(" ", ['Full Time', 'First Half', 'Second Half'], default='Full Time', key='tdz_time_pill')
    if tdz_time_phase == 'Full Time':
        plot_congestion(ax, 'Full Time')
    if tdz_time_phase == 'First Half':
        plot_congestion(ax, 'First Half')
    if tdz_time_phase == 'Second Half':
        plot_congestion(ax, 'Second Half')

    fig_text(0.5, 0.92, f'<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>', highlight_textprops=[{'color': hcol}, {'color': acol}], fontsize=22, fontweight='bold', ha='center', va='center', ax=fig)
    fig.text(0.5, 0.88, "Team's Dominating Zone", fontsize=16, ha='center', va='center')
    fig.text(0.5, 0.18, '@adnaaan433', fontsize=10, ha='center', va='center')

    fig.text(0.5, 0.13, '*Dominating Zone means where the team had more than 55% Open-Play touches than the Opponent', fontstyle='italic', fontsize=7, ha='center', va='center')
    fig.text(0.5, 0.11, '*Contested means where the team had 45-55% Open-Play touches than the Opponent, where less than 45% there Opponent was dominant', fontstyle='italic', fontsize=7, ha='center', va='center')

    himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png")
    himage = Image.open(himage)
    ax_himage = add_image(himage, fig, left=0.075, bottom=0.84, width=0.11, height=0.11)

    aimage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png")
    aimage = Image.open(aimage)
    ax_aimage = add_image(aimage, fig, left=0.84, bottom=0.84, width=0.11, height=0.11)

    st.pyplot(fig)

if an_tp == 'Pass Target Zones':
    st.header(f'Overall {an_tp}')
if an_tp == 'Pass Target Zones':
    st.header(f'Overall {an_tp}')

    def pass_target_zone(ax, team_name, col, phase_tag):
        if phase_tag == 'Full Time':
            dfptz = df[(df['teamName'] == team_name) & (df['type'] == 'Pass')]
            ax.text(34, 109, 'Full Time: 0-90 minutes', color=col, fontsize=13, ha='center', va='center')
        elif phase_tag == 'First Half':
            dfptz = df[(df['teamName'] == team_name) & (df['type'] == 'Pass') & (df['period'] == 'FirstHalf')]
            ax.text(34, 109, 'First Half: 0-45 minutes', color=col, fontsize=13, ha='center', va='center')
        elif phase_tag == 'Second Half':
            dfptz = df[(df['teamName'] == team_name) & (df['type'] == 'Pass') & (df['period'] == 'SecondHalf')]
            ax.text(34, 109, 'Second Half: 45-90 minutes', color=col, fontsize=13, ha='center', va='center')

        pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, line_zorder=2, linewidth=2)
        pitch.draw(ax=ax)

        pc_map = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors", [bg_color, col], N=20)
        path_eff = [path_effects.Stroke(linewidth=3, foreground=bg_color), path_effects.Normal()]
        bin_statistic = pitch.bin_statistic(dfptz.endX, dfptz.endY, bins=(6, 5), statistic='count', normalize=True)
        pitch.heatmap(bin_statistic, ax=ax, cmap=pc_map, edgecolors='#ededed')
        pitch.label_heatmap(bin_statistic, color=line_color, fontsize=20, ax=ax, ha='center', va='center', zorder=5, str_format='{:.0%}', path_effects=path_eff)
        pitch.scatter(dfptz.endX, dfptz.endY, s=5, color='gray', ax=ax)

        return

    fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
    ptz_time_phase = st.pills(" ", ['Full Time', 'First Half', 'Second Half'], default='Full Time', key='overall')
    if ptz_time_phase == 'Full Time':
        home_cc_stats = pass_target_zone(axs[0], hteamName, hcol, 'Full Time')
        away_cc_stats = pass_target_zone(axs[1], ateamName, acol, 'Full Time')
    if ptz_time_phase == 'First Half':
        home_cc_stats = pass_target_zone(axs[0], hteamName, hcol, 'First Half')
        away_cc_stats = pass_target_zone(axs[1], ateamName, acol, 'First Half')
    if ptz_time_phase == 'Second Half':
        home_cc_stats = pass_target_zone(axs[0], hteamName, hcol, 'Second Half')
        away_cc_stats = pass_target_zone(axs[1], ateamName, acol, 'Second Half')

    fig_text(0.5, 1.05, f'<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>', highlight_textprops=[{'color': hcol}, {'color': acol}], fontsize=30, fontweight='bold', ha='center', va='center', ax=fig)
    fig.text(0.5, 1.01, 'Pass Target Zones', fontsize=20, ha='center', va='center')
    fig.text(0.5, 0.97, '@adnaaan433', fontsize=10, ha='center', va='center')

    himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png")
    himage = Image.open(himage)
    ax_himage = add_image(himage, fig, left=0.085, bottom=0.97, width=0.125, height=0.125)

    aimage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png")
    aimage = Image.open(aimage)
    ax_aimage = add_image(aimage, fig, left=0.815, bottom=0.97, width=0.125, height=0.125)

    st.pyplot(fig)

    st.header('Successful Pass Ending Zones')

    def succ_pass_target_zone(ax, team_name, col, phase_tag):
        if phase_tag == 'Full Time':
            dfptz = df[(df['teamName'] == team_name) & (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful')]
            ax.text(34, 109, 'Full Time: 0-90 minutes', color=col, fontsize=13, ha='center', va='center')
        elif phase_tag == 'First Half':
            dfptz = df[(df['teamName'] == team_name) & (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful') & (df['period'] == 'FirstHalf')]
            ax.text(34, 109, 'First Half: 0-45 minutes', color=col, fontsize=13, ha='center', va='center')
        elif phase_tag == 'Second Half':
            dfptz = df[(df['teamName'] == team_name) & (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful') & (df['period'] == 'SecondHalf')]
            ax.text(34, 109, 'Second Half: 45-90 minutes', color=col, fontsize=13, ha='center', va='center')

        pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, line_zorder=2, linewidth=2)
        pitch.draw(ax=ax)

        pc_map = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors", [bg_color, col], N=20)
        path_eff = [path_effects.Stroke(linewidth=3, foreground=bg_color), path_effects.Normal()]
        bin_statistic = pitch.bin_statistic(dfptz.endX, dfptz.endY, bins=(6, 5), statistic='count', normalize=True)
        pitch.heatmap(bin_statistic, ax=ax, cmap=pc_map, edgecolors='#ededed')
        pitch.label_heatmap(bin_statistic, color=line_color, fontsize=20, ax=ax, ha='center', va='center', zorder=5, str_format='{:.0%}', path_effects=path_eff)
        pitch.scatter(dfptz.endX, dfptz.endY, s=5, color='gray', ax=ax)

        return

    fig, axs = plt.subplots(1, 2, figsize=(15, 10), facecolor=bg_color)
    sptz_time_phase = st.pills(" ", ['Full Time', 'First Half', 'Second Half'], default='Full Time', key='successful_only')
    if sptz_time_phase == 'Full Time':
        home_cc_stats = succ_pass_target_zone(axs[0], hteamName, hcol, 'Full Time')
        away_cc_stats = succ_pass_target_zone(axs[1], ateamName, acol, 'Full Time')
    if sptz_time_phase == 'First Half':
        home_cc_stats = succ_pass_target_zone(axs[0], hteamName, hcol, 'First Half')
        away_cc_stats = succ_pass_target_zone(axs[1], ateamName, acol, 'First Half')
    if sptz_time_phase == 'Second Half':
        home_cc_stats = succ_pass_target_zone(axs[0], hteamName, hcol, 'Second Half')
        away_cc_stats = succ_pass_target_zone(axs[1], ateamName, acol, 'Second Half')

    fig_text(0.5, 1.05, f'<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>', highlight_textprops=[{'color': hcol}, {'color': acol}], fontsize=30, fontweight='bold', ha='center', va='center', ax=fig)
    fig.text(0.5, 1.01, 'Successful Pass Ending Zones', fontsize=20, ha='center', va='center')
    fig.text(0.5, 0.97, '@adnaaan433', fontsize=10, ha='center', va='center')

    himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{hftmb_tid}.png")
    himage = Image.open(himage)
    ax_himage = add_image(himage, fig, left=0.085, bottom=0.97, width=0.125, height=0.125)

    aimage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{aftmb_tid}.png")
    aimage = Image.open(aimage)
    ax_aimage = add_image(aimage, fig, left=0.815, bottom=0.97, width=0.125, height=0.125)

    st.pyplot(fig)

with tab2:
    team_player = st.pills(" ", [f"{hteamName} Players", f"{ateamName} Players", f'{hteamName} GK', f'{ateamName} GK'], selection_mode='single', default=f"{hteamName} Players", key='selecting_team_for_player_analysis')

    def offensive_actions(ax, pname):
        # Viz Dfs:
        playerdf = df[df['name'] == pname]
        passdf = playerdf[playerdf['type'] == 'Pass']
        succ_passdf = passdf[passdf['outcomeType'] == 'Successful']
        prg_pass = playerdf[(playerdf['prog_pass'] > 9.144) & (playerdf['outcomeType'] == 'Successful') & (playerdf['x'] > 35) &
                            (~playerdf['qualifiers'].str.contains('Freekick|Corner'))]
        prg_carry = playerdf[(playerdf['prog_carry'] > 9.144) & (playerdf['endX'] > 35)]
        cc = playerdf[(playerdf['qualifiers'].str.contains('KeyPass'))]
        ga = playerdf[(playerdf['qualifiers'].str.contains('GoalAssist'))]
        goal = playerdf[(playerdf['type'] == 'Goal') & (~playerdf['qualifiers'].str.contains('OwnGoal'))]
        owngoal = playerdf[(playerdf['type'] == 'Goal') & (playerdf['qualifiers'].str.contains('OwnGoal'))]
        ontr = playerdf[(playerdf['type'] == 'SavedShot') & (~playerdf['qualifiers'].str.contains(': 82'))]
        oftr = playerdf[playerdf['type'].isin(['MissedShots', 'ShotOnPost'])]
        takeOns = playerdf[(playerdf['type'] == 'TakeOn') & (playerdf['outcomeType'] == 'Successful')]
        takeOnu = playerdf[(playerdf['type'] == 'TakeOn') & (playerdf['outcomeType'] == 'Unsuccessful')]

        # Pitch Plot
        pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, line_zorder=2, linewidth=2, pad_bottom=27)
        pitch.draw(ax=ax)

        # line, arrow, scatter Plots
        pitch.lines(succ_passdf.x, succ_passdf.y, succ_passdf.endX, succ_passdf.endY, color='gray', comet=True, lw=2, alpha=0.65, zorder=1, ax=ax)
        pitch.scatter(succ_passdf.endX, succ_passdf.endY, color=bg_color, ec='gray', s=20, zorder=2, ax=ax)
        pitch.lines(prg_pass.x, prg_pass.y, prg_pass.endX, prg_pass.endY, color=acol, comet=True, lw=3, zorder=2, ax=ax)
        pitch.scatter(prg_pass.endX, prg_pass.endY, color=bg_color, ec=acol, s=40, zorder=3, ax=ax)
        pitch.lines(cc.x, cc.y, cc.endX, cc.endY, color=violet, comet=True, lw=3.5, zorder=3, ax=ax)
        pitch.scatter(cc.endX, cc.endY, color=bg_color, ec=violet, s=50, lw=1.5, zorder=4, ax=ax)
        pitch.lines(ga.x, ga.y, ga.endX, ga.endY, color='green', comet=True, lw=4, zorder=4, ax=ax)
        pitch.scatter(ga.endX, ga.endY, color=bg_color, ec='green', s=60, lw=2, zorder=5, ax=ax)

        for index, row in prg_carry.iterrows():
            arrow = patches.FancyArrowPatch((row['y'], row['x']), (row['endY'], row['endX']), arrowstyle='->', color=acol, zorder=2, mutation_scale=20, 
                                            linewidth=2, linestyle='--')
            ax.add_patch(arrow)

        pitch.scatter(goal.x, goal.y, c=bg_color, edgecolors='green', linewidths=1.2, s=300, marker='football', zorder=10, ax=ax)
        pitch.scatter(owngoal.x, owngoal.y, c=bg_color, edgecolors='orange', linewidths=1.2, s=300, marker='football', zorder=10, ax=ax)
        pitch.scatter(ontr.x, ontr.y, c=hcol, edgecolors=line_color, linewidths=1.2, s=200, alpha=0.75, zorder=9, ax=ax)
        pitch.scatter(oftr.x, oftr.y, c=bg_color, edgecolors=hcol, linewidths=1.2, s=200, alpha=0.75, zorder=8, ax=ax)

        pitch.scatter(takeOns.x, takeOns.y, c='orange', edgecolors=line_color, marker='h', s=200, alpha=0.75, zorder=7, ax=ax)
        pitch.scatter(takeOnu.x, takeOnu.y, c=bg_color, edgecolors='orange', marker='h', lw=1.2, hatch='//////', s=200, alpha=0.85, zorder=7, ax=ax)

        # Stats:
        pitch.scatter(-5, 68, c=bg_color, edgecolors='green', linewidths=1.2, s=300, marker='football', zorder=10, ax=ax)
        pitch.scatter(-10, 68, c=hcol, edgecolors=line_color, linewidths=1.2, s=300, alpha=0.75, zorder=9, ax=ax)
        pitch.scatter(-15, 68, c=bg_color, edgecolors=hcol, linewidths=1.2, s=300, alpha=0.75, zorder=8, ax=ax)
        pitch.scatter(-20, 68, c='orange', edgecolors=line_color, marker='h', s=300, alpha=0.75, zorder=7, ax=ax)
        pitch.scatter(-25, 68, c=bg_color, edgecolors='orange', marker='h', lw=1.2, hatch='//////', s=300, alpha=0.85, zorder=7, ax=ax)
        if len(owngoal) > 0:
            ax_text(64, -4.5, f'Goals: {len(goal)} | <OwnGoal: {len(owngoal)}>', fontsize=12, highlight_textprops=[{'color': 'orange'}], ax=ax)
        else:
            ax.text(64, -5.5, f'Goals: {len(goal)}', fontsize=12)
        ax.text(64, -10.5, f'Shots on Target: {len(ontr)}', fontsize=12)
        ax.text(64, -15.5, f'Shots off Target: {len(oftr)}', fontsize=12)
        ax.text(64, -20.5, f'TakeOn (Succ.): {len(takeOns)}', fontsize=12)
        ax.text(64, -25.5, f'TakeOn (Unsucc.): {len(takeOnu)}', fontsize=12)

        pitch.lines(-5, 34, -5, 24, color='gray', comet=True, lw=2, alpha=0.65, zorder=1, ax=ax)
        pitch.scatter(-5, 24, color=bg_color, ec='gray', s=20, zorder=2, ax=ax)
        pitch.lines(-10, 34, -10, 24, color=acol, comet=True, lw=3, zorder=2, ax=ax)
        pitch.scatter(-10, 24, color=bg_color, ec=acol, s=40, zorder=3, ax=ax)
        arrow = patches.FancyArrowPatch((34, -15), (23, -15), arrowstyle='->', color=acol, zorder=2, mutation_scale=20, 
                                        linewidth=2, linestyle='--')
        ax.add_patch(arrow)
        pitch.lines(-20, 34, -20, 24, color=violet, comet=True, lw=3.5, zorder=3, ax=ax)
        pitch.scatter(-20, 24, color=bg_color, ec=violet, s=50, lw=1.5, zorder=4, ax=ax)
        pitch.lines(-25, 34, -25, 24, color='green', comet=True, lw=4, zorder=4, ax=ax)
        pitch.scatter(-25, 24, color=bg_color, ec='green', s=60, lw=2, zorder=5, ax=ax)

        ax.text(21, -5.5, f'Successful Pass: {len(succ_passdf)}', fontsize=12)
        ax.text(21, -10.5, f'Progressive Pass: {len(prg_pass)}', fontsize=12)
        ax.text(21, -15.5, f'Progressive Carry: {len(prg_carry)}', fontsize=12)
        ax.text(21, -20.5, f'Key Passes: {len(cc)}', fontsize=12)
        ax.text(21, -25.5, f'Assists: {len(ga)}', fontsize=12)

        ax.text(34, 110, 'Offensive Actions', fontsize=20, fontweight='bold', ha='center', va='center')
        return

    def defensive_actions(ax, pname):
        # Viz Dfs:
        playerdf = df[df['name'] == pname]
        tackles = playerdf[(playerdf['type'] == 'Tackle') & (playerdf['outcomeType'] == 'Successful')]
        tackleu = playerdf[(playerdf['type'] == 'Tackle') & (playerdf['outcomeType'] == 'Unsuccessful')]
        ballrec = playerdf[playerdf['type'] == 'BallRecovery']
        intercp = playerdf[playerdf['type'] == 'Interception']
        clearnc = playerdf[playerdf['type'] == 'Clearance']
        passbkl = playerdf[playerdf['type'] == 'BlockedPass']
        shotbkl = playerdf[playerdf['type'] == 'Save']
        chalnge = playerdf[playerdf['type'] == 'Challenge']
        aerialw = playerdf[(playerdf['type'] == 'Aerial') & (playerdf['qualifiers'].str.contains('Defensive')) & (playerdf['outcomeType'] == 'Successful')]
        aerialu = playerdf[(playerdf['type'] == 'Aerial') & (playerdf['qualifiers'].str.contains('Defensive')) & (playerdf['outcomeType'] == 'Unsuccessful')]

        # Pitch Plot
        pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2, pad_bottom=27)
        pitch.draw(ax=ax)

        # Scatter Plots
        sns.scatterplot(x=tackles.y, y=tackles.x, marker='X', s=300, color=acol, edgecolor=line_color, linewidth=1.5, alpha=0.8, ax=ax)
        sns.scatterplot(x=tackleu.y, y=tackleu.x, marker='X', s=300, color=hcol, edgecolor=line_color, linewidth=1.5, alpha=0.8, ax=ax)
        pitch.scatter(ballrec.x, ballrec.y, marker='o', lw=1.5, s=300, c=acol, edgecolors=line_color, ax=ax, alpha=0.8)
        pitch.scatter(intercp.x, intercp.y, marker='*', lw=1.25, s=600, c=acol, edgecolors=line_color, ax=ax, alpha=0.8)
        pitch.scatter(clearnc.x, clearnc.y, marker='h', lw=1.5, s=400, c=acol, edgecolors=line_color, ax=ax, alpha=0.8)
        pitch.scatter(passbkl.x, passbkl.y, marker='s', lw=1.5, s=300, c=acol, edgecolors=line_color, ax=ax, alpha=0.8)
        pitch.scatter(shotbkl.x, shotbkl.y, marker='s', lw=1.5, s=300, c=hcol, edgecolors=line_color, ax=ax, alpha=0.8)
        pitch.scatter(chalnge.x, chalnge.y, marker='+', lw=5, s=300, c=hcol, edgecolors=line_color, ax=ax, alpha=0.8)
        pitch.scatter(aerialw.x, aerialw.y, marker='^', lw=1.5, s=300, c=acol, edgecolors=line_color, ax=ax, alpha=0.8)
        pitch.scatter(aerialu.x, aerialu.y, marker='^', lw=1.5, s=300, c=hcol, edgecolors=line_color, ax=ax, alpha=0.8)

        # Stats
        sns.scatterplot(x=[65], y=[-5], marker='X', s=300, color=acol, edgecolor=line_color, linewidth=1.5, alpha=0.8, ax=ax)
        sns.scatterplot(x=[65], y=[-10], marker='X', s=300, color=hcol, edgecolor=line_color, linewidth=1.5, alpha=0.8, ax=ax)
        pitch.scatter(-15, 65, marker='o', lw=1.5, s=300, c=acol, edgecolors=line_color, ax=ax, alpha=0.8)
        pitch.scatter(-20, 65, marker='*', lw=1.25, s=600, c=acol, edgecolors=line_color, ax=ax, alpha=0.8)
        pitch.scatter(-25, 65, marker='h', lw=1.5, s=400, c=acol, edgecolors=line_color, ax=ax, alpha=0.8)

        pitch.scatter(-5, 26, marker='s', lw=1.5, s=300, c=acol, edgecolors=line_color, ax=ax, alpha=0.8)
        pitch.scatter(-10, 26, marker='s', lw=1.5, s=300, c=hcol, edgecolors=line_color, ax=ax, alpha=0.8)
        pitch.scatter(-15, 26, marker='+', lw=5, s=300, c=hcol, edgecolors=line_color, ax=ax, alpha=0.8)
        pitch.scatter(-20, 26, marker='^', lw=1.5, s=300, c=acol, edgecolors=line_color, ax=ax, alpha=0.8)
        pitch.scatter(-25, 26, marker='^', lw=1.5, s=300, c=hcol, edgecolors=line_color, ax=ax, alpha=0.8)

        ax.text(60, -5.5, f'Tackle (Succ.): {len(tackles)}', fontsize=12)
        ax.text(60, -10.5, f'Tackle (Unsucc.): {len(tackleu)}', fontsize=12)
        ax.text(60, -15.5, f'Ball Recoveries: {len(ballrec)}', fontsize=12)
        ax.text(60, -20.5, f'Interceptions: {len(intercp)}', fontsize=12)
        ax.text(60, -25.5, f'Clearance: {len(clearnc)}', fontsize=12)

        ax.text(21, -5.5, f'Passes Blocked: {len(passbkl)}', fontsize=12)
        ax.text(21, -10.5, f'Shots Blocked: {len(shotbkl)}', fontsize=12)
        ax.text(21, -15.5, f'Dribble Past: {len(chalnge)}', fontsize=12)
        ax.text(21, -20.5, f'Aerials Won: {len(aerialw)}', fontsize=12)
        ax.text(21, -25.5, f'Aerials Lost: {len(aerialu)}', fontsize=12)

        ax.text(34, 110, 'Defensive Actions', fontsize=20, fontweight='bold', ha='center', va='center')
        return

    def pass_receiving_and_touchmap(ax, pname):
        # Viz Dfs:
        playerdf = df[df['name'] == pname]
        touch_df = playerdf[(playerdf['x'] > 0) & (playerdf['y'] > 0)]
        pass_rec = df[(df['type'] == 'Pass') & (df['outcomeType'] == 'Successful') & (df['name'].shift(-1) == pname)]
        # touch_df = pd.concat([acts_df, pass_rec], ignore_index=True)
        actual_touch = playerdf[playerdf['isTouch'] == 1]

        fthd_tch = actual_touch[actual_touch['x'] >= 70]
        penbox_tch = actual_touch[(actual_touch['x'] >= 88.5) & (actual_touch['y'] >= 13.6) & (actual_touch['y'] <= 54.4)]

        fthd_rec = pass_rec[pass_rec['endX'] >= 70]
        penbox_rec = pass_rec[(pass_rec['endX'] >= 88.5) & (pass_rec['endY'] >= 13.6) & (pass_rec['endY'] <= 54.4)]

        pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2, pad_bottom=27)
        pitch.draw(ax=ax)

        ax.scatter(touch_df.y, touch_df.x, marker='o', s=30, c='None', edgecolor=acol, lw=2)
        if len(touch_df) > 3:
            # Calculate mean point
            mean_point = np.mean(touch_df[['y', 'x']].values, axis=0)

            # Calculate distances from the mean point
            distances = np.linalg.norm(touch_df[['y', 'x']].values - mean_point[None, :], axis=1)

            # Compute the interquartile range (IQR)
            q1, q3 = np.percentile(distances, [20, 80])  # Middle 75%: 12.5th to 87.5th percentile
            iqr_mask = (distances >= q1) & (distances <= q3)

            # Filter points within the IQR
            points_within_iqr = touch_df[['y', 'x']].values[iqr_mask]

            # Check if we have enough points for a convex hull
            if len(points_within_iqr) >= 3:
                hull = ConvexHull(points_within_iqr)
                for simplex in hull.simplices:
                    ax.plot(points_within_iqr[simplex, 0], points_within_iqr[simplex, 1], color=acol, linestyle='--')
                ax.fill(points_within_iqr[hull.vertices, 0], points_within_iqr[hull.vertices, 1], 
                        facecolor='none', edgecolor=acol, alpha=0.3, hatch='/////', zorder=1)
            else:
                pass
        else:
            pass

        ax.scatter(pass_rec.endY, pass_rec.endX, marker='o', s=30, c='None', edgecolor=hcol, lw=2)
        if len(touch_df) > 4:
            # Calculate mean point
            mean_point = np.mean(pass_rec[['endY', 'endX']].values, axis=0)

            # Calculate distances from the mean point
            distances = np.linalg.norm(pass_rec[['endY', 'endX']].values - mean_point[None, :], axis=1)

            # Compute the interquartile range (IQR)
            q1, q3 = np.percentile(distances, [25, 75])  # Middle 75%: 12.5th to 87.5th percentile
            iqr_mask = (distances >= q1) & (distances <= q3)

            # Filter points within the IQR
            points_within_iqr = pass_rec[['endY', 'endX']].values[iqr_mask]

            # Check if we have enough points for a convex hull
            if len(points_within_iqr) >= 3:
                hull = ConvexHull(points_within_iqr)
                for simplex in hull.simplices:
                    ax.plot(points_within_iqr[simplex, 0], points_within_iqr[simplex, 1], color=hcol, linestyle='--')
                ax.fill(points_within_iqr[hull.vertices, 0], points_within_iqr[hull.vertices, 1], 
                        facecolor='none', edgecolor=hcol, alpha=0.3, hatch='/////', zorder=1)
            else:
                pass
        else:
            pass

        ax_text(34, 110, '<Touches> & <Pass Receiving> Points', fontsize=20, fontweight='bold', ha='center', va='center', 
                highlight_textprops=[{'color': acol}, {'color': hcol}])
        ax.text(34, -5, f'Total Touches: {len(actual_touch)} | at Final Third: {len(fthd_tch)} | at Penalty Box: {len(penbox_tch)}', color=acol, fontsize=13, ha='center', va='center')
        ax.text(34, -9, f'Total Pass Received: {len(pass_rec)} | at Final Third: {len(fthd_rec)} | at Penalty Box: {len(penbox_rec)}', color=hcol, fontsize=13, ha='center', va='center')
        ax.text(34, -17, '*blue area = middle 75% touches area', color=acol, fontsize=13, fontstyle='italic', ha='center', va='center')
        ax.text(34, -21, '*red area = middle 75% pass receiving area', color=hcol, fontsize=13, fontstyle='italic', ha='center', va='center')
        return

    def gk_passmap(ax, pname):
        df_gk = df[(df['name'] == pname)]
        gk_pass = df_gk[df_gk['type'] == 'Pass']
        op_pass = df_gk[(df_gk['type'] == 'Pass') & (~df_gk['qualifiers'].str.contains('GoalKick|FreekickTaken'))]
        sp_pass = df_gk[(df_gk['type'] == 'Pass') & (df_gk['qualifiers'].str.contains('GoalKick|FreekickTaken'))]

        pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2, pad_bottom=15)
        pitch.draw(ax=ax)

        # gk_name = df_gk['shortName'].unique()[0]
        op_succ = sp_succ = 0
        for index, row in op_pass.iterrows():
            if row['outcomeType'] == 'Successful':
                pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=hcol, lw=4, comet=True, alpha=0.5, zorder=2, ax=ax)
                pitch.scatter(row['endX'], row['endY'], s=40, color=hcol, edgecolor=line_color, zorder=3, ax=ax)
                op_succ += 1
            if row['outcomeType'] == 'Unsuccessful':
                pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=hcol, lw=4, comet=True, alpha=0.5, zorder=2, ax=ax)
                pitch.scatter(row['endX'], row['endY'], s=40, color=bg_color, edgecolor=hcol, zorder=3, ax=ax)
        for index, row in sp_pass.iterrows():
            if row['outcomeType'] == 'Successful':
                pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=violet, lw=4, comet=True, alpha=0.5, zorder=2, ax=ax)
                pitch.scatter(row['endX'], row['endY'], s=40, color=violet, edgecolor=line_color, zorder=3, ax=ax)
                sp_succ += 1
            if row['outcomeType'] == 'Unsuccessful':
                pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=violet, lw=4, comet=True, alpha=0.35, zorder=2, ax=ax)
                pitch.scatter(row['endX'], row['endY'], s=40, color=bg_color, edgecolor=violet, zorder=3, ax=ax)

        gk_pass['length'] = np.sqrt((gk_pass['x'] - gk_pass['endX'])**2 + (gk_pass['y'] - gk_pass['endY'])**2)
        sp_pass['length'] = np.sqrt((sp_pass['x'] - sp_pass['endX'])**2 + (sp_pass['y'] - sp_pass['endY'])**2)
        avg_len = gk_pass['length'].mean().round(2)
        avg_hgh = sp_pass['endX'].mean().round(2)

        ax.set_title('Pass Map', color=line_color, fontsize=25, fontweight='bold')
        ax.text(34, -10, f'Avg. Passing Length: {avg_len}m  |  Avg. Goalkick Length: {avg_hgh}m', color=line_color, fontsize=13, ha='center', va='center')
        ax_text(34, -5, s=f'<Open-play Pass (Acc.): {len(op_pass)} ({op_succ})>  |  <GoalKick/Freekick (Acc.): {len(sp_pass)} ({sp_succ})>', 
                fontsize=13, highlight_textprops=[{'color': hcol}, {'color': violet}], ha='center', va='center', ax=ax)

        return

    def gk_def_acts(ax, pname):
        df_gk = df[df['name'] == pname]
        claimed = df_gk[df_gk['type'] == 'Claim']
        notclmd = df_gk[df_gk['type'] == 'CrossNotClaimed']
        punched = df_gk[df_gk['type'] == 'Punch']
        smother = df_gk[df_gk['type'] == 'Smother']
        kprswpr = df_gk[df_gk['type'].isin(['KeeperSweeper', 'Clearance'])]
        ballwin = df_gk[df_gk['type'].isin(['BallRecovery', 'KeeperPickup', 'Interception'])]

        pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2, pad_bottom=15)
        pitch.draw(ax=ax)

        pitch.scatter(claimed.x, claimed.y, s=300, marker='x', ec=bg_color, lw=2, c='g', ax=ax)
        pitch.scatter(notclmd.x, notclmd.y, s=300, marker='x', ec=bg_color, lw=2, c='r', ax=ax)
        pitch.scatter(punched.x, punched.y, s=300, marker='+', ec=bg_color, lw=2, c='g', ax=ax)
        pitch.scatter(smother.x, smother.y, s=300, marker='o', ec=bg_color, lw=2, c='orange', ax=ax)
        pitch.scatter(kprswpr.x, kprswpr.y, s=300, marker='+', ec=bg_color, lw=2, c=violet, ax=ax)
        pitch.scatter(ballwin.x, ballwin.y, s=300, marker='+', ec=bg_color, lw=2, c='b', ax=ax)

        pitch.scatter(-5, 68-2, s=100, marker='x', ec=bg_color, lw=2, c='g', ax=ax)
        pitch.scatter(-5, 136/3-2, s=100, marker='x', ec=bg_color, lw=2, c='r', ax=ax)
        pitch.scatter(-5, 68/3-2, s=100, marker='+', ec=bg_color, lw=2, c='g', ax=ax)
        pitch.scatter(-10, 68-2, s=100, marker='o', ec=bg_color, lw=2, c='orange', ax=ax)
        pitch.scatter(-10, 136/3-2, s=100, marker='+', ec=bg_color, lw=2, c=violet, ax=ax)
        pitch.scatter(-10, 68/3-2, s=100, marker='+', ec=bg_color, lw=2, c='b', ax=ax)

        ax.set_title('GK Actions', color=line_color, fontsize=25, fontweight='bold')

        ax.text(68-4, -5, f'Cross Claim: {len(claimed)}', fontsize=13, color='g', ha='left', va='center')
        ax.text(136/3-4, -5, f'Missed Claim: {len(notclmd)}', fontsize=13, color='r', ha='left', va='center')
        ax.text(68/3-4, -5, f'Punches: {len(punched)}', fontsize=13, color='g', ha='left', va='center')
        ax.text(68-4, -10, f'Comes Out: {len(smother)}', fontsize=13, color='orange', ha='left', va='center')
        ax.text(136/3-4, -10, f'Sweeping Out: {len(kprswpr)}', fontsize=13, color=violet, ha='left', va='center')
        ax.text(68/3-4, -10, f'Ball Recovery: {len(ballwin)}', fontsize=13, color='b', ha='left', va='center')
        return

    def gk_touches(ax, pname):
        playerdf = df[df['name'] == pname]
        acts_df = playerdf[(playerdf['x'] > 0) & (playerdf['y'] > 0)]
        actual_touch = playerdf[playerdf['isTouch'] == 1]

        pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2, pad_bottom=15)
        pitch.draw(ax=ax)

        ax.scatter(acts_df.y, acts_df.x, marker='o', s=30, c='None', edgecolor=acol, lw=2)

        ax.set_title('Touches', color=line_color, fontsize=25, fontweight='bold')
        ax.text(34, -5, f'Total Touches: {len(actual_touch)}', fontsize=15, ha='center', va='center')
        return

    def playing_time(pname):
        # Filter player data
        df_player = df[df['name'] == pname]
        df_player['isFirstEleven'] = df_player['isFirstEleven'].fillna(0)

        # Identify substitution events
        df_sub_off = df_player[df_player['type'] == 'SubstitutionOff']
        df_sub_on = df_player[df_player['type'] == 'SubstitutionOn']

        # Get maximum match minute and extra time
        max_min = df['minute'].max()
        extra_time = max(0, max_min - 90)  # Ensure extra_time is non-negative

        # Initialize minutes played
        mins_played = 0

        # Case 1: Started the game and was not substituted off
        if df_player['isFirstEleven'].unique()[0] == 1 and len(df_sub_off) == 0:
            subs_text = None
            mins_played = 90

        # Case 2: Started the game and was substituted off
        elif df_player['isFirstEleven'].unique()[0] == 1 and len(df_sub_off) == 1:
            subs_text = 'Substituted Out'
            sub_off_min = df_sub_off['minute'].unique()[0]
            mins_played = min(90, sub_off_min)
            if sub_off_min > 90:
                mins_played = int((sub_off_min * 90) / max_min)  # Proportional adjustment for extra time

        # Case 3: Substituted on before or at the 80th minute
        elif df_player['isFirstEleven'].unique()[0] == 0 and len(df_sub_on) > 0:
            subs_text = 'Substituted In'
            sub_on_min = df_sub_on['minute'].unique()[0]
            if sub_on_min <= 80:
                mins_played = max_min - sub_on_min - extra_time
            else:
                mins_played = max_min - sub_on_min

        # Adjust for red cards
        df_red = df_player[(df_player['type'] == 'Card') & (df_player['qualifiers'].str.contains('SecondYellow|Red', na=False))]
        if len(df_red) == 1:
            subs_text = 'Got Red Card'
            red_min = df_red['minute'].max()
            mins_played = mins_played - (90 - red_min)

        return int(mins_played), subs_text

def generate_gk_dahsboard(pname, ftmb_tid):
    fig, axs = plt.subplots(1, 2, figsize=(16, 15), facecolor='#f5f5f5')

    # Calculate minutes played
    mins_played, subs_text = playing_time(pname)

    # Generate individual plots
    gk_passmap(axs[0], pname)
    gk_def_acts(axs[1], pname)
    fig.subplots_adjust(wspace=0.025)

    # Add text and images to the figure
    fig.text(0.22, 0.98, f'{pname}', fontsize=40, fontweight='bold', ha='left', va='center')
    if subs_text is None:
        fig.text(0.22, 0.94, f'in {hteamName} {hgoal_count} - {agoal_count} {ateamName}  |  Minutes played: {mins_played}', 
                 fontsize=25, ha='left', va='center')
    else:
        fig.text(0.22, 0.94, f'in {hteamName} {hgoal_count} - {agoal_count} {ateamName}  |  Minutes played: {mins_played} ({subs_text})', 
                 fontsize=25, ha='left', va='center')
    fig.text(0.87, 0.995, '@adnaaan433', fontsize=15, ha='right', va='center')

    himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{ftmb_tid}.png")
    himage = Image.open(himage)
    add_image(himage, fig, left=0.1, bottom=0.91, width=0.1, height=0.1)

    st.pyplot(fig)
    def generate_gk_dahsboard(pname, ftmb_tid):
    fig, axs = plt.subplots(1, 2, figsize=(16, 15), facecolor='#f5f5f5')

    # Calculate minutes played
    mins_played, subs_text = playing_time(pname)

    # Generate individual plots
    gk_passmap(axs[0], pname)
    gk_def_acts(axs[1], pname)
    fig.subplots_adjust(wspace=0.025)

    # Add text and images to the figure
    fig.text(0.22, 0.98, f'{pname}', fontsize=40, fontweight='bold', ha='left', va='center')
    if subs_text is None:
        fig.text(0.22, 0.94, f'in {hteamName} {hgoal_count} - {agoal_count} {ateamName}  |  Minutes played: {mins_played}', 
                 fontsize=25, ha='left', va='center')
    else:
        fig.text(0.22, 0.94, f'in {hteamName} {hgoal_count} - {agoal_count} {ateamName}  |  Minutes played: {mins_played} ({subs_text})', 
                 fontsize=25, ha='left', va='center')
    fig.text(0.87, 0.995, '@adnaaan433', fontsize=15, ha='right', va='center')

    himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{ftmb_tid}.png")
    himage = Image.open(himage)
    add_image(himage, fig, left=0.1, bottom=0.91, width=0.1, height=0.1)

    st.pyplot(fig)

    def player_detailed_data(pname):
        df_filt = df[~df['type'].str.contains('Carry|TakeOn|Challenge')].reset_index(drop=True)
        df_flt = df[~df['type'].str.contains('TakeOn|Challenge')].reset_index(drop=True)
        dfp = df[df['name'] == pname]

        # Shooting
        pshots = dfp[(dfp['type'].isin(['Goal', 'SavedShot', 'MissedShots', 'ShotOnPost'])) & (~dfp['qualifiers'].str.contains('OwnGoal'))]
        goals = pshots[pshots['type'] == 'Goal']
        saved = pshots[(pshots['type'] == 'SavedShot') & (~pshots['qualifiers'].str.contains(': 82'))]
        block = pshots[(pshots['type'] == 'SavedShot') & (pshots['qualifiers'].str.contains(': 82'))]
        missd = pshots[pshots['type'] == 'MissedShots']
        postd = pshots[pshots['type'] == 'ShotOnPost']
        big_c = pshots[pshots['qualifiers'].str.contains('BigChance')]
        big_cmis = big_c[big_c['type'] != 'Goal']
        op_shots = pshots[pshots['qualifiers'].str.contains('RegularPlay')]
        out_b = pshots[pshots['qualifiers'].str.contains('OutOfBox')]
        og = dfp[dfp['qualifiers'].str.contains('OwnGoal')]
        pen_t = pshots[pshots['qualifiers'].str.contains('Penalty')]
        pen_m = pen_t[pen_t['type'] != 'Goal']
        frk_shots = pshots[pshots['qualifiers'].str.contains('DirectFreekick')]
        frk_goals = frk_shots[frk_shots['type'] == 'Goal']
        pshots['shots_distance'] = np.sqrt((pshots['x'] - 105)**2 + (pshots['y'] - 34)**2)
        avg_shots_dist = round(pshots['shots_distance'].median(), 2) if len(pshots) != 0 else 'N/A'

        # Pass
        passdf = dfp[dfp['type'] == 'Pass']
        accpass = passdf[passdf['outcomeType'] == 'Successful']
        pass_accuracy = round((len(accpass) / len(passdf)) * 100, 2) if len(passdf) != 0 else 0
        pro_pass = accpass[(accpass['prog_pass'] > 9.144) & (~accpass['qualifiers'].str.contains('Freekick|Corner')) & (accpass['x'] > 35)]
        cc = dfp[dfp['qualifiers'].str.contains('KeyPass')]
        bcc = dfp[dfp['qualifiers'].str.contains('BigChanceCreated')]
        ass = dfp[dfp['qualifiers'].str.contains('GoalAssist')]
        preas = df_filt[(df_filt['name'] == pname) & (df_filt['type'] == 'Pass') & (df_filt['outcomeType'] == 'Successful') & (df_filt['qualifiers'].shift(-1).str.contains('GoalAssist'))]
        buildup_s = df_filt[(df_filt['name'] == pname) & (df_filt['type'] == 'Pass') & (df_filt['outcomeType'] == 'Successful') & (df_filt['qualifiers'].shift(-1).str.contains('KeyPass'))]
        fthird_pass = passdf[passdf['endX'] > 70]
        fthird_succ = fthird_pass[fthird_pass['outcomeType'] == 'Successful']
        penbox_pass = passdf[(passdf['endX'] >= 88.5) & (passdf['endY'] >= 13.6) & (passdf['endY'] <= 54.4)]
        penbox_succ = penbox_pass[penbox_pass['outcomeType'] == 'Successful']
        crs = passdf[(passdf['qualifiers'].str.contains('Cross')) & (~passdf['qualifiers'].str.contains('Freekick|Corner'))]
        crs_s = crs[crs['outcomeType'] == 'Successful']
        long = passdf[(passdf['qualifiers'].str.contains('Longball')) & (~passdf['qualifiers'].str.contains('Freekick|Corner'))]
        long_s = long[long['outcomeType'] == 'Successful']
        corner = passdf[passdf['qualifiers'].str.contains('Corner')]
        corners = corner[corner['outcomeType'] == 'Successful']
        throw_in = passdf[passdf['qualifiers'].str.contains('ThrowIn')]
        throw_ins = throw_in[throw_in['outcomeType'] == 'Successful']
        xT_df = accpass[accpass['xT'] > 0]
        xT_ip = round(xT_df['xT'].sum(), 2)
        pass_to = df[(df['type'].shift(1) == 'Pass') & (df['outcomeType'].shift(1) == 'Successful') & (df['name'].shift(1) == pname)]
        most_to = pass_to['name'].value_counts().idxmax() if not pass_to.empty else None
        most_count = pass_to['name'].value_counts().max() if not pass_to.empty else None
        forward_pass = passdf[(passdf['endX'] - passdf['x']) > 2]
        forward_pass_s = forward_pass[forward_pass['outcomeType'] == 'Successful']
        back_pass = passdf[(passdf['x'] - passdf['endX']) > 2]
        back_pass_s = back_pass[back_pass['outcomeType'] == 'Successful']
        side_pass = len(passdf) - len(forward_pass) - len(back_pass)
        side_pass_s = len(accpass) - len(forward_pass_s) - len(back_pass_s)

        # Carry
        carrydf = dfp[dfp['type'] == 'Carry']
        pro_carry = carrydf[(carrydf['prog_carry'] > 9.144) & (carrydf['endX'] > 35)]
        led_shot1 = df_flt[(df_flt['type'] == 'Carry') & (df_flt['name'] == pname) & (df_flt['qualifiers'].shift(-1).str.contains('KeyPass'))]
        led_shot2 = df_flt[(df_flt['type'] == 'Carry') & (df_flt['name'] == pname) & (df_flt['type'].shift(-1).str.contains('Shot'))]
        led_shot = pd.concat([led_shot1, led_shot2])
        led_goal1 = df_flt[(df_flt['type'] == 'Carry') & (df_flt['name'] == pname) & (df_flt['qualifiers'].shift(-1).str.contains('GoalAssist'))]
        led_goal2 = df_flt[(df_flt['type'] == 'Carry') & (df_flt['name'] == pname) & (df_flt['type'].shift(-1) == 'Goal')]
        led_goal = pd.concat([led_goal1, led_goal2])
        fth_carry = carrydf[(carrydf['x'] < 70) & (carrydf['endX'] >= 70)]
        box_carry = carrydf[(carrydf['endX'] >= 88.5) & (carrydf['endY'] >= 13.6) & (carrydf['endY'] <= 54.4) &
                            ~((carrydf['x'] >= 88.5) & (carrydf['y'] >= 13.6) & (carrydf['y'] <= 54.6))]
        carrydf['carry_len'] = np.sqrt((carrydf['x'] - carrydf['endX'])**2 + (carrydf['y'] - carrydf['endY'])**2)
        avg_carry_len = round(carrydf['carry_len'].mean(), 2)
        carry_xT = carrydf[carrydf['xT'] > 0]
        xT_ic = round(carry_xT['xT'].sum(), 2)
        forward_carry = carrydf[(carrydf['endX'] - carrydf['x']) > 2]
        back_carry = carrydf[(carrydf['x'] - carrydf['endX']) > 2]
        side_carry = len(carrydf) - len(forward_carry) - len(back_carry)
        t_on = dfp[dfp['type'] == 'TakeOn']
        t_ons = t_on[t_on['outcomeType'] == 'Successful']
        t_on_rate = round((len(t_ons) / len(t_on)) * 100, 2) if len(t_on) != 0 else 0

        # Pass Receiving
        df_rec = df[(df['type'] == 'Pass') & (df['outcomeType'] == 'Successful') & (df['name'].shift(-1) == pname)]
        kp_rec = df_rec[df_rec['qualifiers'].str.contains('KeyPass')]
        as_rec = df_rec[df_rec['qualifiers'].str.contains('GoalAssist')]
        fthd_rec = df_rec[df_rec['endX'] >= 70]
        pen_rec = df_rec[(df_rec['endX'] >= 87.5) & (df_rec['endY'] >= 13.6) & (df_rec['endY'] <= 54.6)]
        pro_rec = df_rec[(df_rec['x'] >= 35) & (df_rec['prog_pass'] >= 9.11) & (~df_rec['qualifiers'].str.contains('CornerTaken|Frerkick'))]
        crs_rec = df_rec[(df_rec['qualifiers'].str.contains('Cross')) & (~df_rec['qualifiers'].str.contains('CornerTaken|Frerkick'))]
        xT_rec = round(df_rec['xT'].sum(), 2)
        long_rec = df_rec[(df_rec['qualifiers'].str.contains('Longball'))]
        next_act = df[(df['name'] == pname) & (df['type'].shift(1) == 'Pass') & (df['outcomeType'].shift(1) == 'Successful')]
        ball_retain = next_act[(next_act['outcomeType'] == 'Successful') & ((next_act['type'] != 'Foul') & (next_act['type'] != 'Dispossessed'))]
        ball_retention = round((len(ball_retain) / len(next_act)) * 100, 2) if len(next_act) != 0 else 0
        most_from = df_rec['name'].value_counts().idxmax() if not df_rec.empty else None
        most_from_count = df_rec['name'].value_counts().max() if not df_rec.empty else 0

        # Defensive 
        ball_wins = dfp[dfp['type'].isin(['Interception', 'BallRecovery'])]
        f_third = ball_wins[ball_wins['x'] >= 70]
        p_tk = dfp[(dfp['type'] == 'Tackle')]
        p_tk_s = dfp[(dfp['type'] == 'Tackle') & (dfp['outcomeType'] == 'Successful')]
        p_intc = dfp[(dfp['type'] == 'Interception')]
        p_br = dfp[dfp['type'] == 'BallRecovery']
        p_cl = dfp[dfp['type'] == 'Clearance']
        p_fl = dfp[(dfp['type'] == 'Foul') & (dfp['outcomeType'] == 'Unsuccessful')]
        p_fls = dfp[(dfp['type'] == 'Foul') & (dfp['outcomeType'] == 'Successful')]
        fls_fthd = p_fls[p_fls['x'] >= 70]
        pen_won = p_fls[(p_fls['qualifiers'].str.contains('Penalty'))]
        pen_con = p_fl[(p_fl['qualifiers'].str.contains('Penalty'))]
        p_ard = dfp[(dfp['type'] == 'Aerial') & (dfp['qualifiers'].str.contains('Defensive'))]
        p_ard_s = p_ard[p_ard['outcomeType'] == 'Successful']
        p_ard_rate = round((len(p_ard_s) / len(p_ard)) * 100, 2) if len(p_ard) != 0 else 0
        p_aro = dfp[(dfp['type'] == 'Aerial') & (dfp['qualifiers'].str.contains('Offensive'))]
        p_aro_s = p_aro[p_aro['outcomeType'] == 'Successful']
        p_aro_rate = round((len(p_aro_s) / len(p_aro)) * 100, 2) if len(p_aro) != 0 else 0
        pass_bl = dfp[dfp['type'] == 'BlockedPass']
        shot_bl = dfp[dfp['type'] == 'Save']
        drb_pst = dfp[dfp['type'] == 'Challenge']
        err_lat = dfp[dfp['qualifiers'].str.contains('LeadingToAttempt')]
        err_lgl = dfp[dfp['qualifiers'].str.contains('LeadingToGoal')]
        prbr = df[(df['name'] == pname) & (df['type'].isin(['BallRecovery', 'Interception'])) & (df['name'].shift(-1) == pname) & 
                  (df['outcomeType'].shift(-1) == 'Successful') & (df['type'].shift(-1) != 'Dispossessed')]
        post_rec_ball_retention = round((len(prbr) / (len(p_br) + len(p_intc))) * 100, 2) if (len(p_br) + len(p_intc)) != 0 else 0

        # Miscellaneous
        player_id = str(int(dfp['playerId'].max()))
        off_df = df[df['type'] == 'OffsidePass']
        off_caught = off_df[off_df['qualifiers'].str.contains(player_id)]
        disps = dfp[dfp['type'] == 'Dispossessed']
        pos_lost_ptb = dfp[(dfp['type'].isin(['Pass', 'TakeOn', 'BallTouch'])) & (dfp['outcomeType'] == 'Unsuccessful')]
        pos_lost_edo = dfp[dfp['type'].isin(['Error', 'Dispossessed', 'OffsidePass'])]
        pos_lost = pd.concat([pos_lost_ptb, pos_lost_edo])
        touches = dfp[dfp['isTouch'] == 1]
        fth_touches = touches[touches['x'] >= 70]
        pen_touches = touches[(touches['x'] >= 88.5) & (touches['y'] >= 13.6) & (touches['y'] <= 54.4)]
        ycard = dfp[(dfp['type'] == 'Card') & (dfp['qualifiers'].str.contains('Yellow'))]
        rcard = dfp[(dfp['type'] == 'Card') & (dfp['qualifiers'].str.contains('Red'))]

        shooting_stats_dict = {
            'Total Shots': len(pshots),
            'Goal': len(goals),
            'Shots On Target': len(saved) + len(goals),
            'Shots Off Target': len(missd) + len(postd),
            'Blocked Shots': len(block),
            'Big Chances': len(big_c),
            'Big Chances Missed': len(big_cmis),
            'Hit Woodwork': len(postd),
            'Open Play Shots': len(op_shots),
            'Shots from Inside the Box': len(pshots) - len(out_b),
            'Shots from Outside the Box': len(out_b),
            'Avg. Shot Distance': f'{avg_shots_dist}m',
            'Penalties Taken': len(pen_t),
            'Penalties Missed': len(pen_m),
            'Shots from Freekick': len(frk_shots),
            'Goals from Freekick': len(frk_goals),
            'Own Goal': len(og)
        }

        passing_stats_dict = {
            'Accurate Passes': f'{len(accpass)}/{len(passdf)} ({pass_accuracy}%)',
            'Progressive Passes': len(pro_pass),
            'Chances Created': len(cc),
            'Big Chances Created': len(bcc),
            'Assists': len(ass),
            'Pre-Assists': len(preas),
            'BuildUp to Shot': len(buildup_s),
            'Final 1/3 Passes (Acc.)': f'{len(fthird_pass)} ({len(fthird_succ)})',
            'Passes in Penalty Box (Acc.)': f'{len(penbox_pass)} ({len(penbox_succ)})',
            'Crosses (Acc.)': f'{len(crs)} ({len(crs_s)})',
            'Longballs (Acc.)': f'{len(long)} ({len(long_s)})',
            'Corners Taken (Acc.)': f'{len(corner)} ({len(corners)})',
            'Throw-Ins Taken (Acc.)': f'{len(throw_in)} ({len(throw_ins)})',
            'Forward Pass (Acc.)': f'{len(forward_pass)} ({len(forward_pass_s)})',
            'Side Pass (Acc.)': f'{side_pass} ({side_pass_s})',
            'Back Pass (Acc.)': f'{len(back_pass)} ({len(back_pass_s)})',
            'xT from Pass': xT_ip,
            'Most Passes to': f'{most_to} ({most_count})'
        }

        carry_stats_dict = {
            'Total Carries': len(carrydf),
            'Progressive Carries': len(pro_carry),
            'Carries Led to Shot': len(led_shot),
            'Carries Led to Goal': len(led_goal),
            'Carries into Final Third': len(fth_carry),
            'Carries into Penalty Box': len(box_carry),
            'Avg. Carry Length': f'{avg_carry_len}m',
            'xT from Carry': xT_ic,
            'Ball Carry Forward': len(forward_carry),
            'Ball Carry Sidewise': side_carry,
            'Ball Carry Back': len(back_carry),
            'Take-On Attempts': len(t_on),
            'Successful Take-Ons': len(t_ons),
            'Take-On Success Rate': f'{t_on_rate}%'
        }

        pass_receiving_stats_dict = {
            'Total Passes Received': len(df_rec),
            'Key Passes Received': len(kp_rec),
            'Assists Received': len(as_rec),
            'Passes Received in Final 1/3': len(fthd_rec),
            'Passes Received in Penalty Box': len(pen_rec),
            'Progressive Passes Received': len(pro_rec),
            'Crosses Received': len(crs_rec),
            'Longball Received': len(long_rec),
            'xT Received': xT_rec,
            'Ball Retention': f'{ball_retention}%',
            'Most Passes Received From': f'{most_from} ({most_from_count})'
        }

        defensive_stats_dict = {
            'Tackles (Won)': f'{len(p_tk)} ({len(p_tk_s)})',
            'Interceptions': len(p_intc),
            'Passes Blocked': len(pass_bl),
            'Shots Blocked': len(shot_bl),
            'Ball Recoveries': len(p_br),
            'Post Recovery Ball Retention': f'{post_rec_ball_retention}%',
            'Possession Won at Final 1/3': len(f_third),
            'Clearances': len(p_cl),
            'Fouls Committed': len(p_fl),
            'Fouls Won': len(p_fls),
            'Fouls Won at Final Third': len(fls_fthd),
            'Penalties Won': len(pen_won),
            'Penalties Conceded': len(pen_con),
            'Defensive Aerial Duels Won': f'{len(p_ard_s)}/{len(p_ard)} ({p_ard_rate}%)',
            'Offensive Aerial Duels Won': f'{len(p_aro_s)}/{len(p_aro)} ({p_aro_rate}%)',
            'Dribble Past': len(drb_pst),
            'Errors Leading To Shot': len(err_lat),
            'Errors Leading To Goal': len(err_lgl)
        }

        other_stats_dict = {
            'Caught Offside': len(off_caught),
            'Dispossessed': len(disps),
            'Possession Lost': len(pos_lost),
            'Total Touches': len(touches),
            'Touches at Final Third': len(fth_touches),
            'Touches at Penalty Box': len(pen_touches),
            'Yellow Cards': len(ycard),
            'Red Cards': len(rcard)
        }

        return shooting_stats_dict, passing_stats_dict, carry_stats_dict, pass_receiving_stats_dict, defensive_stats_dict, other_stats_dict

    if team_player == f"{hteamName} Players":
        home_pname_df = homedf[(homedf['name'] != 'nan') & (homedf['position'] != 'GK')]
        hpname = st.selectbox('Select a Player:', home_pname_df.name.unique(), index=None, key='home_player_analysis')
        if st.session_state.home_player_analysis:
            st.header(f'{hpname} Performance Dashboard')
            generate_player_dahsboard(f'{hpname}', hftmb_tid)

            shooting_stats_dict, passing_stats_dict, carry_stats_dict, pass_receiving_stats_dict, defensive_stats_dict, other_stats_dict = player_detailed_data(hpname)
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader('Shooting Stats')
                for key, value in shooting_stats_dict.items():
                    st.text(f"{key}: {value}")
            with col2:
                st.subheader('Passing Stats')
                for key, value in passing_stats_dict.items():
                    st.write(f"{key}: {value}")
            with col3:
                st.subheader('Carry Stats')
                for key, value in carry_stats_dict.items():
                    st.write(f"{key}: {value}")
            st.divider()
            col4, col5, col6 = st.columns(3)
            with col4:
                st.subheader('Pass Receiving Stats')
                for key, value in pass_receiving_stats_dict.items():
                    st.write(f"{key}: {value}")
            with col5:
                st.subheader('Defensive Stats')
                for key, value in defensive_stats_dict.items():
                    st.write(f"{key}: {value}")
            with col6:
                st.subheader('Other Stats')
                for key, value in other_stats_dict.items():
                    st.write(f"{key}: {value}")

    if team_player == f"{ateamName} Players":
        away_pname_df = awaydf[(awaydf['name'] != 'nan') & (awaydf['position'] != 'GK')]
        apname = st.selectbox('Select a Player:', away_pname_df.name.unique(), index=None, key='away_player_analysis')
        if st.session_state.away_player_analysis:
            st.header(f'{apname} Performance Dashboard')
            generate_player_dahsboard(f'{apname}', aftmb_tid)

            shooting_stats_dict, passing_stats_dict, carry_stats_dict, pass_receiving_stats_dict, defensive_stats_dict, other_stats_dict = player_detailed_data(apname)
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader('Shooting Stats')
                for key, value in shooting_stats_dict.items():
                    st.text(f"{key}: {value}")
            with col2:
                st.subheader('Passing Stats')
                for key, value in passing_stats_dict.items():
                    st.write(f"{key}: {value}")
            with col3:
                st.subheader('Carry Stats')
                for key, value in carry_stats_dict.items():
                    st.write(f"{key}: {value}")
            st.divider()
            col4, col5, col6 = st.columns(3)
            with col4:
                st.subheader('Pass Receiving Stats')
                for key, value in pass_receiving_stats_dict.items():
                    st.write(f"{key}: {value}")
            with col5:
                st.subheader('Defensive Stats')
                for key, value in defensive_stats_dict.items():
                    st.write(f"{key}: {value}")
            with col6:
                st.subheader('Other Stats')
                for key, value in other_stats_dict.items():
                    st.write(f"{key}: {value}")

    if team_player == f'{hteamName} GK':
        home_gk_df = homedf[(homedf['name'] != 'nan') & (homedf['position'] == 'GK')]
        pname = st.selectbox('Select a Goal-Keeper:', home_gk_df.name.unique(), index=None, key='home_gk_analysis')
        if st.session_state.home_gk_analysis:
            st.header(f'{pname} Performance Dashboard')
            generate_gk_dahsboard(f'{pname}', hftmb_tid)

    if team_player == f'{ateamName} GK':
        away_gk_df = awaydf[(awaydf['name'] != 'nan') & (awaydf['position'] == 'GK')]
        pname = st.selectbox('Select a Goal-Keeper:', away_gk_df.name.unique(), index=None, key='away_gk_analysis')
        if st.session_state.away_gk_analysis:
            st.header(f'{pname} Performance Dashboard')
            generate_gk_dahsboard(f'{pname}', aftmb_tid)
