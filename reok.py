import json, time, re
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, LinearSegmentedColormap
import streamlit as st
import arabic_reshaper
from bidi.algorithm import get_display
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from unidecode import unidecode
from scipy.spatial import ConvexHull
from mplsoccer import Pitch, VerticalPitch
# --------------------------------------------------------------------
# إعداد matplotlib لدعم العربية
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Amiri','Noto Sans Arabic','Arial','Tahoma']
mpl.rcParams['axes.unicode_minus'] = False

def reshape_arabic_text(txt):
    return get_display(arabic_reshaper.reshape(txt))

st.markdown("""
<style>
body {direction: rtl; text-align: right;}
.stSelectbox > div > div > div {text-align: right;}
</style>""", unsafe_allow_html=True)

# Sidebar: ألوان + رابط المباراة
st.sidebar.title('اختيارات')
hcol = st.sidebar.color_picker('لون الفريق المضيف', '#d00000')
acol = st.sidebar.color_picker('لون الفريق الضيف', '#003087')
bg_color = st.sidebar.color_picker('لون الخلفية', '#1e1e2f')
grad_start = st.sidebar.color_picker('بداية التدرج', '#003087')
grad_end   = st.sidebar.color_picker('نهاية التدرج', '#d00000')
line_color = st.sidebar.color_picker('لون الخطوط', '#ffffff')
gradient_colors = [grad_start, grad_end]

st.sidebar.title('رابط المباراة')
match_url = st.sidebar.text_input('WhoScored URL', 
    placeholder='https://1xbet.whoscored.com/matches/.../live/...')
confirm = st.sidebar.button('تأكيد')
if 'confirmed' not in st.session_state: st.session_state.confirmed=False
if confirm: st.session_state.confirmed=True

# 1) استخراج JSON من WhoScored
def extract_json_from_url(url):
    opts = webdriver.ChromeOptions()
    opts.add_argument('--headless')
    opts.add_argument('--no-sandbox')
    opts.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(service=Service(), options=opts)
    try:
        driver.get(url); time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        script = soup.find('script', string=lambda s: s and 'matchCentreData' in s)
        if not script: return None
        txt = script.string
        prefix = 'matchCentreData: '
        start = txt.find(prefix)
        if start<0: return None
        start += len(prefix)
        end = txt.find(',\n', start)
        return json.loads(txt[start:end])
    finally:
        driver.quit()

# 2) تفكيك dict إلى أحداث ولاعبين وفرق
def extract_data_from_dict(data):
    mc = data['matchCentreData']
    teams = {
        mc['home']['teamId']: mc['home']['name'],
        mc['away']['teamId']: mc['away']['name']
    }
    ph = pd.DataFrame(mc['home']['players']); ph['teamId']=mc['home']['teamId']
    pa = pd.DataFrame(mc['away']['players']); pa['teamId']=mc['away']['teamId']
    players = pd.concat([ph,pa])
    players['name']=players['name'].astype(str).apply(unidecode)
    return mc['events'], players, teams


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
# دالة لاختصار الاسم
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

# دالة لحساب سلاسل الاستحواذ
def get_possession_chains(events_df, chain_check, suc_evts_in_chain):
    # هنا يجب أن تبدأ بـ events_out = pd.DataFrame()
    events_out = pd.DataFrame()
    # نستخدم dataframe المُمرّر وليس df خارجي
    match_events_df = events_df.reset_index(drop=True).copy()

    # استبعاد بعض الأحداث
    invalid = ['OffsideGiven','CornerAwarded','Start','Card','SubstitutionOff',
               'SubstitutionOn','FormationChange','FormationSet','End']
    match_pos = match_events_df[~match_events_df['type'].isin(invalid)].copy()

    # الأعمدة المؤقتة
    match_pos['outcomeBinary'] = (match_pos['outcomeType']=='Successful').astype(int)
    # لترتيب الفرق حرفياً
    first_team = match_pos['teamName'].min()
    match_pos['teamBinary'] = (match_pos['teamName']==first_team).astype(int)
    match_pos['goalBinary'] = ((match_pos['type']=='Goal').astype(int)
                               .diff().fillna(0).apply(lambda x: 1 if x<0 else 0))

    # بناء pos_chain_df
    pos_chain = pd.DataFrame(index=match_pos.index)
    # تحقق من نفس الفريق عبر n أحداث
    for n in range(1, chain_check):
        col = f'evt_{n}_same_team'
        pos_chain[col] = abs(match_pos['teamBinary'].diff(-n))
        pos_chain[col] = pos_chain[col].apply(lambda x: 1 if x>1 else x)
    pos_chain['enough_evt_same_team'] = pos_chain.sum(axis=1).apply(
        lambda x: 1 if x < chain_check - suc_evts_in_chain else 0)
    pos_chain['enough_evt_same_team'] = pos_chain['enough_evt_same_team'].diff().fillna(pos_chain['enough_evt_same_team'])

        
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
