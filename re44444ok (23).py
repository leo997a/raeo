import json
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
import warnings
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# تجاهل تحذيرات الإهمال
warnings.filterwarnings("ignore", category=DeprecationWarning)

# دالة لجلب بيانات المباراة من رابط
def extract_match_dict(match_url):
    """Extract match event from WhoScored match center"""
    try:
        service = webdriver.ChromeService()
        driver = webdriver.Chrome(service=service)
        driver.get(match_url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        element = soup.select_one('script:-soup-contains("matchCentreData")')
        if element is None:
            st.error("تعذر العثور على بيانات المباراة في الرابط المقدم.")
            return None
        matchdict = json.loads(element.text.split("matchCentreData: ")[1].split(',\n')[0])
        driver.quit()
        return matchdict
    except Exception as e:
        st.error(f"حدث خطأ أثناء جلب البيانات: {str(e)}")
        return None

# دالة لاستخراج البيانات من القاموس
def extract_data_from_dict(data):
    event_types_json = data["matchCentreEventTypeJson"]
    formation_mappings = data["formationIdNameMappings"]
    events_dict = data["events"]
    teams_dict = {data['home']['teamId']: data['home']['name'],
                  data['away']['teamId']: data['away']['name']}
    players_dict = data["playerIdNameDictionary"]
    players_home_df = pd.DataFrame(data['home']['players'])
    players_home_df["teamId"] = data['home']['teamId']
    players_away_df = pd.DataFrame(data['away']['players'])
    players_away_df["teamId"] = data['away']['teamId']
    players_df = pd.concat([players_home_df, players_away_df])
    return event_types_json, formation_mappings, events_dict, teams_dict, players_df, players_dict

# دالة لمعالجة الأحداث
def process_event(event, event_types_json, players_dict, teams_dict):
    event_type_id = event.get("type")
    event_type = event_types_json.get(str(event_type_id), {}).get("displayName", "Unknown")
    player_id = event.get("playerId")
    player_name = players_dict.get(str(player_id), "Unknown") if player_id else "Unknown"
    team_id = event.get("teamId")
    team_name = teams_dict.get(team_id, "Unknown") if team_id else "Unknown"
    minute = event.get("minute", 0)
    second = event.get("second", 0)
    time = f"{minute}:{second:02d}"
    outcome = event.get("outcomeType", {}).get("displayName", "Unknown")
    x, y = event.get("x", 0), event.get("y", 0)
    qualifiers = event.get("qualifiers", [])
    qualifier_list = [q.get("type", {}).get("displayName", "Unknown") for q in qualifiers]
    return {
        "event_type": event_type,
        "player_name": player_name,
        "team_name": team_name,
        "time": time,
        "outcome": outcome,
        "x": x,
        "y": y,
        "qualifiers": ", ".join(qualifier_list)
    }

# دالة لإنشاء جدول الأحداث
def make_event_df(events_dict, event_types_json, players_dict, teams_dict):
    events = [process_event(event, event_types_json, players_dict, teams_dict) for event in events_dict]
    return pd.DataFrame(events)

# تحليلات إضافية
def analyze_team_events(event_df, teams_dict):
    team_stats = {}
    for team_id, team_name in teams_dict.items():
        team_events = event_df[event_df['team_name'] == team_name]
        stats = {
            'Total Events': len(team_events),
            'Passes': len(team_events[team_events['event_type'] == 'Pass']),
            'Shots': len(team_events[team_events['event_type'] == 'Shot']),
            'Goals': len(team_events[team_events['event_type'] == 'Goal']),
            'Successful Passes': len(team_events[(team_events['event_type'] == 'Pass') & (team_events['outcome'] == 'Successful')])
        }
        team_stats[team_name] = stats
    return pd.DataFrame(team_stats).T

def plot_event_locations(event_df, event_type='Shot'):
    fig, ax = plt.subplots(figsize=(10, 7))
    shots = event_df[event_df['event_type'] == event_type]
    sns.scatterplot(data=shots, x='x', y='y', hue='team_name', style='outcome', ax=ax)
    ax.set_title(f'مواقع {event_type} على الملعب')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    return fig

# واجهة Streamlit
st.title("تحليل بيانات مباراة كرة القدم")
st.write("أدخل رابط مباراة من موقع WhoScored لتحليل بياناتها مباشرة.")

# حقل إدخال الرابط
match_url = st.text_input("رابط المباراة:", placeholder="https://1xbet.whoscored.com/Matches/...")

if match_url:
    with st.spinner("جارٍ جلب البيانات..."):
        json_data = extract_match_dict(match_url)
        
        if json_data:
            # استخراج البيانات
            event_types_json, formation_mappings, events_dict, teams_dict, players_df, players_dict = extract_data_from_dict(json_data)
            
            # إنشاء جدول الأحداث
            event_df = make_event_df(events_dict, event_types_json, players_dict, teams_dict)
            
            # عرض جدول الأحداث
            st.subheader("جدول الأحداث")
            st.dataframe(event_df.head(), hide_index=True)
            
            # عرض جدول اللاعبين
            st.subheader("جدول اللاعبين")
            st.dataframe(players_df.head(), hide_index=True)
            
            # عرض معلومات الفرق
            st.subheader("الفرق")
            st.write(teams_dict)
            
            # تحليلات إضافية
            st.subheader("إحصائيات الفرق")
            team_stats_df = analyze_team_events(event_df, teams_dict)
            st.dataframe(team_stats_df)
            
            st.subheader("توزيع التسديدات")
            fig = plot_event_locations(event_df, 'Shot')
            st.pyplot(fig)
else:
    st.info("الرجاء إدخال رابط مباراة لتحليل البيانات.")