
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
from highlight_text import fig_text         # استيراد fig_text من highlight-text
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
violet = '#800080'  # تعريف اللون البنفسجي كقيمة ثابتة
# إضافة أدوات اختيار الألوان في الشريط الجانبي
st.sidebar.title('اختيار الألوان')
hcol = st.sidebar.color_picker('لون الفريق المضيف', default_hcol, key='hcol_picker')
acol = st.sidebar.color_picker('لون الفريق الضيف', default_acol, key='acol_picker')
bg_color = st.sidebar.color_picker('لون الخلفية', default_bg_color, key='bg_color_picker')
gradient_start = st.sidebar.color_picker('بداية التدرج', default_gradient_colors[0], key='gradient_start_picker')
gradient_end = st.sidebar.color_picker('نهاية التدرج', default_gradient_colors[1], key='gradient_end_picker')
gradient_colors = [gradient_start, gradient_end]  # تحديث قائمة ألوان التدرج
line_color = st.sidebar.color_picker('لون الخطوط', '#ffffff', key='line_color_picker')  # اختياري

st.sidebar.title('إدخال رابط المباراة')
match_url = st.sidebar.text_input('أدخل رابط المباراة (من WhoScored):', placeholder='https://1xbet.whoscored.com/matches/...')

# إضافة أدوات اختيار الألوان في الشريط الجانبي (كما هي)
st.sidebar.title('اختيار الألوان')
hcol = st.sidebar.color_picker('لون الفريق المضيف', default_hcol, key='hcol_picker')
acol = st.sidebar.color_picker('لون الفريق الضيف', default_acol, key='acol_picker')
bg_color = st.sidebar.color_picker('لون الخلفية', default_bg_color, key='bg_color_picker')
gradient_start = st.sidebar.color_picker('بداية التدرج', default_gradient_colors[0], key='gradient_start_picker')
gradient_end = st.sidebar.color_picker('نهاية التدرج', default_gradient_colors[1], key='gradient_end_picker')
gradient_colors = [gradient_start, gradient_end]
line_color = st.sidebar.color_picker('لون الخطوط', '#ffffff', key='line_color_picker')

# تهيئة الحالة
if 'confirmed' not in st.session_state:
    st.session_state.confirmed = False

def reset_confirmed():
    st.session_state['confirmed'] = False

# التحقق من إدخال الرابط والضغط على زر التأكيد
if match_url:
    try:
        # محاولة جلب الرابط للتحقق من صحته
        response = requests.get(match_url)
        response.raise_for_status()  # رمي خطأ إذا كان الرابط غير صالح
        match_input = st.sidebar.button('تأكيد الرابط', on_click=lambda: st.session_state.update({'confirmed': True}))
    except requests.exceptions.RequestException:
        st.session_state['confirmed'] = False
        st.sidebar.error('الرابط غير صالح أو المباراة غير موجودة')
else:
    st.session_state['confirmed'] = False

# إذا تم تأكيد الرابط، جلب البيانات
if match_url and st.session_state.confirmed:
    @st.cache_data
    def get_event_data(match_url):
        def extract_json_from_html(html_path):
            try:
                response = requests.get(html_path)
                response.raise_for_status()
                html = response.text

                # البحث عن البيانات باستخدام regex (كما في الكود الأصلي)
                regex_pattern = r'(?<=require\.config\.params\["args"\].=.)[\s\S]*?;'
                data_txt = re.findall(regex_pattern, html)
                if not data_txt:
                    raise ValueError("لم يتم العثور على بيانات JSON في الصفحة")
                data_txt = data_txt[0]

                # تنظيف النص ليصبح JSON صالح
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

        # استخراج البيانات من الرابط
        json_data_txt = extract_json_from_html(match_url)
        data = json.loads(json_data_txt)
        events_dict, players_df, teams_dict = extract_data_from_dict(data)

        df = pd.DataFrame(events_dict)
        dfp = pd.DataFrame(players_df)

        # باقي معالجة البيانات كما في الكود الأصلي
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

        # ... (باقي دوال معالجة البيانات كما هي: cumulative_match_mins, insert_ball_carries, إلخ)
        df = cumulative_match_mins(df)
        df = insert_ball_carries(df, min_carry_length=3, max_carry_length=100, min_carry_duration=1, max_carry_duration=50)
        df = df.reset_index(drop=True)
        df['index'] = range(1, len(df) + 1)
        df = df[['index'] + [col for col in df.columns if col != 'index']]

        # معالجة xT والبيانات الأخرى (كما في الكود الأصلي)
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
        dfxT['start_zone_value_xT'] = dfxT[['x1_bin_xT', 'y1_bin_xT']].apply(lambda x: xT[x[1]][x[0]], axis=1)
        dfxT['end_zone_value_xT'] = dfxT[['x2_bin_xT', 'y2_bin_xT']].apply(lambda x: xT[x[1]][x[0]], axis=1)
        dfxT['xT'] = dfxT['end_zone_value_xT'] - dfxT['start_zone_value_xT']
        columns_to_drop = ['id', 'eventId', 'minute', 'second', 'teamId', 'x', 'y', 'expandedMinute', 'period',
                           'outcomeType', 'qualifiers', 'type', 'satisfiedEventsTypes', 'isTouch', 'playerId', 'endX',
                           'endY', 'relatedEventId', 'relatedPlayerId', 'blockedX', 'blockedY', 'goalMouthZ', 'goalMouthY', 'isShot', 'cumulative_mins']
        dfxT.drop(columns=columns_to_drop, inplace=True)
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
        dfp.drop(columns=columns_to_drop, inplace=True)
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
        df.drop(columns=columns_to_drop2, inplace=True)

        df = get_possession_chains(df, 5, 3)
        df['period'] = df['period'].replace({
            1: 'FirstHalf', 2: 'SecondHalf', 3: 'FirstPeriodOfExtraTime',
            4: 'SecondPeriodOfExtraTime', 5: 'PenaltyShootout', 14: 'PostGame', 16: 'PreMatch'
        })
        df = df[df['period'] != 'PenaltyShootout']
        df = df.reset_index(drop=True)
        return df, teams_dict, players_df

    # جلب البيانات باستخدام الرابط
    df, teams_dict, players_df = get_event_data(match_url)

    # باقي الكود كما هو (معالجة البيانات، عرض الشبكة، إلخ)
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

    st.header(f'{hteamName} {hgoal_count} - {agoal_count} {ateamName}')
    st.text('تحليل المباراة بناءً على الرابط')
    
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
