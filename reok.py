import json
import re
import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go
from urllib.request import urlopen
from PIL import Image

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

# تعريف الدالة reset_confirmed قبل استخدامها
def reset_confirmed():
    st.session_state['confirmed'] = False

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

st.sidebar.title('اختيار المباراة')

season = None
league = None
stage = None
htn = None
atn = None

if 'confirmed' not in st.session_state:
    st.session_state.confirmed = False

season = st.sidebar.selectbox('اختر الموسم:', ['2024_25'], key='season', index=0, on_change=reset_confirmed)
if season:
    league_options = [
        'الدوري الإسباني',
        'الدوري الإنجليزي الممتاز',
        'الدوري الإيطالي',
        'دوري أبطال أوروبا'
    ]
    league = st.sidebar.selectbox('اختر الدوري:', league_options, key='league', index=None, on_change=reset_confirmed)

    if league == 'الدوري الإسباني':
        team_list = [
            'Athletic Club', 'Atletico Madrid', 'Barcelona',
            'Celta Vigo', 'Deportivo Alaves', 'Espanyol',
            'Getafe', 'Girona', 'Las Palmas',
            'Leganes', 'Mallorca', 'Osasuna',
            'Rayo Vallecano', 'Real Betis', 'Real Madrid',
            'Real Sociedad', 'Real Valladolid', 'Sevilla',
            'Valencia', 'Villarreal'
        ]
    elif league == 'الدوري الإنجليزي الممتاز':
        team_list = [
            'Arsenal', 'Aston Villa', 'Bournemouth',
            'Brentford', 'Brighton', 'Chelsea',
            'Crystal Palace', 'Everton', 'Fulham',
            'Ipswich', 'Leicester', 'Liverpool',
            'Manchester City', 'Manchester United', 'Newcastle',
            'Nottingham Forest', 'Southampton', 'Tottenham',
            'West Ham', 'Wolves'
        ]
    elif league == 'الدوري الإيطالي':
        team_list = [
            'AC Milan', 'Atalanta', 'Bologna',
            'Cagliari', 'Como', 'Empoli',
            'Fiorentina', 'Genoa', 'Inter',
            'Juventus', 'Lazio', 'Lecce',
            'Monza', 'Napoli', 'Parma Calcio',
            'Roma', 'Torino', 'Udinese',
            'Venezia', 'Verona'
        ]
    elif league == 'دوري أبطال أوروبا':
        team_list = [
            'AC Milan', 'Arsenal', 'Aston Villa',
            'Atalanta', 'Atletico Madrid', 'BSC Young Boys',
            'Barcelona', 'Bayer Leverkusen', 'Bayern Munich',
            'Benfica', 'Bologna', 'Borussia Dortmund',
            'Brest', 'Celtic', 'Club Brugge',
            'Dinamo Zagreb', 'FK Crvena Zvezda', 'Feyenoord',
            'Girona', 'Inter', 'Juventus',
            'Lille', 'Liverpool', 'Manchester City',
            'Monaco', 'PSV Eindhoven', 'Paris Saint-Germain',
            'RB Leipzig', 'Real Madrid', 'Salzburg',
            'Shakhtar Donetsk', 'Slovan Bratislava', 'Sparta Prague',
            'Sporting CP', 'Sturm Graz', 'VfB Stuttgart'
        ]

    if league and league != 'دوري أبطال أوروبا':
        htn = st.sidebar.selectbox('اختر الفريق المضيف', team_list, key='home_team', index=None, on_change=reset_confirmed)
        
        if htn:
            atn_options = [team for team in team_list if team != htn]
            atn = st.sidebar.selectbox('اختر الفريق الضيف', atn_options, key='away_team', index=None, on_change=reset_confirmed)
            
    elif league == 'دوري أبطال أوروبا':
        stage_options = [
            'مرحلة الدوري',
            'الملحق التأهيلي',
            'دور الـ 16',
            'ربع النهائي',
            'نصف النهائي',
            'النهائي'
        ]
        stage = st.sidebar.selectbox('اختر المرحلة', stage_options, key='stage_selection', index=None, on_change=reset_confirmed)
        if stage:
            htn = st.sidebar.selectbox('اختر الفريق المضيف', team_list, key='home_team', index=None, on_change=reset_confirmed)
            
            if htn:
                atn_options = [team for team in team_list if team != htn]
                atn = st.sidebar.selectbox('اختر الفريق الضيف', atn_options, key='away_team', index=None, on_change=reset_confirmed)

    if league and league != 'دوري أبطال أوروبا' and league != 'الدوري الإيطالي' and htn and atn:
        league_en = {
            'الدوري الإسباني': 'La Liga',
            'الدوري الإنجليزي الممتاز': 'Premier League',
            'الدوري الإيطالي': 'Serie A',
            'دوري أبطال أوروبا': 'UEFA Champions League'
        }.get(league, league)
        league = league_en.replace(' ', '_')
        match_html_path = f"https://raw.githubusercontent.com/leo997a/{season}_{league}/refs/heads/main/{htn}_vs_{atn}.html"
        match_html_path = match_html_path.replace(' ', '%20')
        try:
            response = requests.get(match_html_path)
            response.raise_for_status()
            match_input = st.sidebar.button('تأكيد الاختيارات', on_click=lambda: st.session_state.update({'confirmed': True}))
        except:
            st.session_state['confirmed'] = False
            st.sidebar.write('لم يتم العثور على المباراة')
            
    elif league and league == 'الدوري الإيطالي' and htn and atn:
        league_en = 'Serie A'
        league = league_en.replace(' ', '_')
        match_html_path = f"https://raw.githubusercontent.com/leo997a/{season}_{league}/refs/heads/main/{htn}_vs_{atn}.html"
        match_html_path = match_html_path.replace(' ', '%20')
        try:
            response = requests.get(match_html_path)
            response.raise_for_status()
            match_input = st.sidebar.button('تأكيد الاختيارات', on_click=lambda: st.session_state.update({'confirmed': True}))
        except:
            st.session_state['confirmed'] = False
            st.sidebar.write('مباريات الدوري الإيطالي متاحة حتى الأسبوع 12\nسيتم رفع باقي البيانات قريبًا\nشكرًا لصبرك')
            
    elif league and league == 'دوري أبطال أوروبا' and stage and htn and atn:
        league_en = 'UEFA Champions League'
        stage_en = stage
        stage_mapping = {
            'مرحلة الدوري': 'League Phase',
            'الملحق التأهيلي': 'Knockout Playoff',
            'دور الـ 16': 'Round of 16',
            'ربع النهائي': 'Quarter Final',
            'نصف النهائي': 'Semi Final',
            'النهائي': 'Final'
        }
        stage_en = stage_mapping[stage]
        league = league_en.replace(' ', '_')
        match_html_path = f"https://raw.githubusercontent.com/leo997a/{season}_{league}/refs/heads/main/{stage_en}/{htn}_vs_{atn}.html"
        match_html_path = match_html_path.replace(' ', '%20')
        try:
            response = requests.get(match_html_path)
            response.raise_for_status()
            match_input = st.sidebar.button('تأكيد الاختيارات', on_click=lambda: st.session_state.update({'confirmed': True}))
        except:
            st.session_state['confirmed'] = False
            st.sidebar.write('لم يتم العثور على المباراة')

if league and htn and atn and st.session_state.confirmed:
    @st.cache_data
def get_event_data(season, league, stage, hteam, ateam):
        try:
        # تكوين رابط الملف
        match_html_path = f"https://raw.githubusercontent.com/leo997a/{season}_{league}/refs/heads/main/{stage}/{hteam}_vs_{ateam}.html"
        match_html_path = match_html_path.replace(" ", "%20")
        
        # جلب الملف
        response = requests.get(match_html_path)
        response.raise_for_status()  # التحقق من نجاح الطلب
        
        match_html = response.text
        
        # البحث عن البيانات
        match_json = re.search(r'var matchCentreData = ({.*});', match_html)
        if match_json is None:
            st.error(f"لم يتم العثور على 'matchCentreData' في الملف: {match_html_path}")
            return None, None, None
        
        events = match_data["events"]
        teams = match_data["teamsData"]
        players_list = []
        for p in match_data["playerIdNameDictionary"].items():
            players_list.append({"playerId": p[0], "name": p[1]})
        
        df = pd.DataFrame(events)
        teams_dict = teams
        players_df = pd.DataFrame(players_list)
        
        return df, teams_dict, players_df
        except requests.exceptions.HTTPError as e:
        st.error(f"خطأ في جلب البيانات من الرابط: {e}")
        return None, None, None
    except json.JSONDecodeError as e:
        st.error(f"خطأ في تحليل JSON: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"حدث خطأ غير متوقع: {e}")
        return None, None, None

# استخدام الدالة في الكود الرئيسي
season = "2024_25"
league = "UEFA_Champions_League"
stage = "دور الـ 16"  # تأكد من مطابقة اسم المجلد في GitHub
htn = "Barcelona"
atn = "Benfica"

    df, teams_dict, players_df = get_event_data(season, league, stage, htn, atn)
    
    if df is not None and teams_dict is not None and players_df is not None:

    hteamID = list(teams_dict.keys())[0]
    ateamID = list(teams_dict.keys())[1]
        # استمر في معالجة البيانات
    else:
    st.write("تعذر تحميل البيانات. تحقق من الرابط أو الملف.")
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
        'La Liga': 'الدوري الإسباني',
        'Premier League': 'الدوري الإنجليزي الممتاز',
        'Serie A': 'الدوري الإيطالي',
        'UEFA Champions League': 'دوري أبطال أوروبا'
    }
    league_display = league_display_mapping.get(league, league)

    st.header(f'{hteamName} {hgoal_count} - {agoal_count} {ateamName}')
    st.text(league_display)

    tab1, tab2 = st.tabs([
        "تحليل المباراة",
        "تبويب آخر"
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
    st.session_state['analysis_type'] = st.selectbox(
        'نوع التحليل:',
        options,
        index=0,
        key='analysis_type_selectbox'
    )

    def pass_network(team_name, col, phase_tag):
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
            return None, "لا توجد بيانات تمريرات متاحة"

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

        # إنشاء الرسم باستخدام Plotly
        fig = go.Figure()

        # رسم الملعب (UEFA dimensions: 105m x 68m)
        fig.add_shape(type="rect", x0=0, y0=0, x1=68, y1=105, line=dict(color=line_color, width=2), fillcolor="rgba(0, 255, 0, 0.1)")
        fig.add_shape(type="rect", x0=0, y0=0, x1=68, y1=52.5, line=dict(color=line_color, width=1))  # خط المنتصف
        fig.add_shape(type="rect", x0=0, y0=0, x1=16.5, y1=40.3, line=dict(color=line_color, width=1))  # منطقة الجزاء (الفريق السفلي)
        fig.add_shape(type="rect", x0=0, y0=64.7, x1=16.5, y1=105, line=dict(color=line_color, width=1))  # منطقة الجزاء (الفريق العلوي)
        fig.add_shape(type="rect", x0=0, y0=13.8, x1=5.5, y1=91.2, line=dict(color=line_color, width=1))  # منطقة المرمى (الفريق السفلي)
        fig.add_shape(type="rect", x0=0, y0=75.2, x1=5.5, y1=91.2, line=dict(color=line_color, width=1))  # منطقة المرمى (الفريق العلوي)
        fig.add_shape(type="circle", x0=34-9.15, y0=52.5-9.15, x1=34+9.15, y1=52.5+9.15, line=dict(color=line_color, width=1))  # دائرة المنتصف

        # إضافة الخطوط بين اللاعبين
        for idx in range(len(pass_counts_df)):
            fig.add_trace(go.Scatter(
                x=[pass_counts_df['pass_avg_y'].iloc[idx], pass_counts_df['receiver_avg_y'].iloc[idx]],
                y=[pass_counts_df['pass_avg_x'].iloc[idx], pass_counts_df['receiver_avg_x'].iloc[idx]],
                mode='lines',
                line=dict(color=col, width=pass_counts_df['line_width'].iloc[idx], opacity=c_transparency[idx]),
                showlegend=False
            ))

        # إضافة مواقع اللاعبين
        for index, row in avg_locs_df.iterrows():
            marker_symbol = 'circle' if row['isFirstEleven'] else 'square'
            fig.add_trace(go.Scatter(
                x=[row['avg_y']],
                y=[row['avg_x']],
                mode='markers+text',
                marker=dict(size=20, color=col, line=dict(color=line_color, width=2), symbol=marker_symbol, opacity=0.9 if row['isFirstEleven'] else 0.7),
                text=[str(int(row['shirtNo']))],
                textposition='middle center',
                textfont=dict(size=12, color='white'),
                showlegend=False
            ))

        # حساب التماسك العمودي
        avgph = round(avg_locs_df['avg_x'].median(), 2)
        fig.add_shape(type="line", x0=0, y0=avgph, x1=68, y1=avgph, line=dict(color="white", width=1.5, dash="dash"))

        center_backs_height = avg_locs_df[avg_locs_df['position'] == 'DC']
        def_line_h = round(center_backs_height['avg_x'].median(), 2)
        Forwards_height = avg_locs_df[avg_locs_df['isFirstEleven'] == 1].sort_values(by='avg_x', ascending=False).head(2)
        fwd_line_h = round(Forwards_height['avg_x'].mean(), 2)
        fig.add_shape(type="rect", x0=0, y0=def_line_h, x1=68, y1=fwd_line_h, fillcolor=col, opacity=0.2, line=dict(width=0))

        v_comp = round((1 - ((fwd_line_h - def_line_h) / 105)) * 100, 2)

        # إضافة النصوص
        if phase_tag == 'Full Time':
            title_text = f'الوقت بالكامل: 0-90 دقيقة<br>إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'
        elif phase_tag == 'First Half':
            title_text = f'الشوط الأول: 0-45 دقيقة<br>إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'
        elif phase_tag == 'Second Half':
            title_text = f'الشوط الثاني: 45-90 دقيقة<br>إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'

        fig.update_layout(
            title=dict(text=title_text, x=0.5, y=0.95, xanchor='center', yanchor='top', font=dict(size=14, color='white')),
            xaxis=dict(range=[-5, 73], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[-5, 110], showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            showlegend=False,
            margin=dict(l=0, r=0, t=50, b=50),
            annotations=[
                dict(
                    x=34, y=-5,
                    text=f"على الكرة<br>التماسك العمودي (المنطقة المظللة): {v_comp}%",
                    showarrow=False,
                    font=dict(size=12, color='white'),
                    align='center'
                )
            ]
        )

        return pass_btn, fig

    with tab1:
        if st.session_state['analysis_type'] == 'شبكة التمريرات':
            st.header('شبكة التمريرات')
            
            pn_time_phase = st.radio(
                "اختر فترة المباراة:",
                ["الوقت الكامل", "الشوط الأول", "الشوط الثاني"],
                index=0,
                key='pn_time_pill'
            )

            home_pass_btn, home_fig = None, None
            away_pass_btn, away_fig = None, None

            if pn_time_phase == 'الوقت الكامل':
                home_pass_btn, home_fig = pass_network(hteamName, hcol, 'Full Time')
                away_pass_btn, away_fig = pass_network(ateamName, acol, 'Full Time')
            elif pn_time_phase == 'الشوط الأول':
                home_pass_btn, home_fig = pass_network(hteamName, hcol, 'First Half')
                away_pass_btn, away_fig = pass_network(ateamName, acol, 'First Half')
            elif pn_time_phase == 'الشوط الثاني':
                home_pass_btn, home_fig = pass_network(hteamName, hcol, 'Second Half')
                away_pass_btn, away_fig = pass_network(ateamName, acol, 'Second Half')

            if home_pass_btn is not None and away_pass_btn is not None and home_fig is not None and away_fig is not None:
                # عرض الرسومات باستخدام Plotly
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(home_fig, use_container_width=True)
                with col2:
                    st.plotly_chart(away_fig, use_container_width=True)

                # عرض الجداول
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f'أزواج التمرير لفريق {hteamName}:')
                    st.dataframe(home_pass_btn, hide_index=True)
                with col2:
                    st.write(f'أزواج التمرير لفريق {ateamName}:')
                    st.dataframe(away_pass_btn, hide_index=True)
            else:
                st.write("لا توجد بيانات متاحة لعرض شبكة التمريرات.")
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
