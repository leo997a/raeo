import json
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import warnings
import streamlit as st
import plotly.express as px
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)

def extract_match_dict(match_url, chromedriver_path, save_output=False):
    """Extract match event from whoscored match center"""
    try:
        if not os.path.exists(chromedriver_path):
            st.error(f"ملف ChromeDriver غير موجود في: {chromedriver_path}. تأكد من المسار أو نزّل الملف.")
            return None
        
        service = Service(executable_path=chromedriver_path)
        driver = webdriver.Chrome(service=service)
        driver.get(match_url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        element = soup.select_one('script:-soup-contains("matchCentreData")')
        
        if not element:
            st.error("لم يتم العثور على بيانات المباراة. تأكد من صحة الرابط.")
            return None
        
        matchdict = json.loads(element.text.split("matchCentreData: ")[1].split(',\n')[0])
        driver.quit()
        return matchdict
    except Exception as e:
        st.error(f"خطأ في استخراج البيانات: {e}")
        return None

def extract_data_from_dict(data):
    """Process match data into events, players, and teams"""
    if not data:
        return None, None, None
    events_dict = data["events"]
    teams_dict = {data['home']['teamId']: data['home']['name'],
                  data['away']['teamId']: data['away']['name']}
    players_home_df = pd.DataFrame(data['home']['players'])
    players_home_df["teamId"] = data['home']['teamId']
    players_away_df = pd.DataFrame(data['away']['players'])
    players_away_df["teamId"] = data['away']['teamId']
    players_df = pd.concat([players_home_df, players_away_df])
    return events_dict, players_df, teams_dict

def analyze_events(events_dict, teams_dict):
    """Analyze events to generate statistics"""
    if not events_dict:
        return None, None
    df = pd.DataFrame(events_dict)
    
    # تحليل الأهداف
    goals = df[df['type'] == 'Goal'][['teamId', 'playerId', 'minute']]
    goals_count = goals.groupby('teamId').size().to_dict()
    stats = {
        'Goals': {teams_dict.get(k, 'Unknown'): v for k, v in goals_count.items()}
    }
    
    # تحليل التسديدات
    shots = df[df['type'].isin(['ShotOnTarget', 'ShotOffTarget'])][['teamId']]
    shots_count = shots.groupby('teamId').size().to_dict()
    stats['Shots'] = {teams_dict.get(k, 'Unknown'): v for k, v in shots_count.items()}
    
    # تحليل التمريرات الناجحة
    passes = df[df['type'] == 'Pass'][['teamId', 'outcomeType']]
    successful_passes = passes[passes['outcomeType'] == 'Successful'].groupby('teamId').size().to_dict()
    stats['Successful Passes'] = {teams_dict.get(k, 'Unknown'): v for k, v in successful_passes.items()}
    
    return stats, df

def main():
    st.title("محلل مباريات كرة القدم 🏆")
    st.write("أدخل رابط مباراة من موقع WhoScored لتحليل البيانات وعرض الإحصائيات.")
    
    # إدخال رابط المباراة
    default_url = "https://1xbet.whoscored.com/Matches/1809770/Live/Europe-Europa-League-2023-2024-West-Ham-Bayer-Leverkusen"
    match_url = st.text_input("رابط المباراة", value=default_url,
                             placeholder="مثال: https://1xbet.whoscored.com/Matches/...")
    
    # إدخال مسار ChromeDriver
    chromedriver_path = st.text_input("مسار ChromeDriver", 
                                    value=r"C:\Users\Reo k\chromedriver.exe",
                                    help=r"حدد مسار ملف chromedriver.exe على جهازك (مثال: C:\Users\Reo k\chromedriver.exe)")
    
    if st.button("تحليل المباراة"):
        if not match_url:
            st.warning("الرجاء إدخال رابط مباراة صحيح.")
            return
        if not chromedriver_path:
            st.warning("الرجاء إدخال مسار ChromeDriver صحيح.")
            return
        
        with st.spinner("جارٍ استخراج بيانات المباراة..."):
            json_data = extract_match_dict(match_url, chromedriver_path)
            if json_data:
                events_dict, players_df, teams_dict = extract_data_from_dict(json_data)
                
                if events_dict:
                    stats, events_df = analyze_events(events_dict, teams_dict)
                    
                    # عرض الأحداث
                    st.subheader("أحداث المباراة")
                    st.dataframe(events_df[['minute', 'type', 'playerId', 'teamId']].head(10), 
                                hide_index=True)
                    
                    # عرض الإحصائيات
                    st.subheader("إحصائيات المباراة")
                    if stats:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write("**الأهداف**")
                            st.json(stats['Goals'])
                        with col2:
                            st.write("**التسديدات**")
                            st.json(stats['Shots'])
                        with col3:
                            st.write("**التمريرات الناجحة**")
                            st.json(stats['Successful Passes'])
                        
                        # رسم بياني للأهداف
                        goals_df = pd.DataFrame.from_dict(stats['Goals'], orient='index', columns=['Goals'])
                        fig = px.bar(goals_df, x=goals_df.index, y='Goals', 
                                    title="الأهداف لكل فريق", 
                                    labels={'index': 'الفريق', 'Goals': 'عدد الأهداف'})
                        st.plotly_chart(fig)
                        
                        # رسم بياني للتسديدات
                        shots_df = pd.DataFrame.from_dict(stats['Shots'], orient='index', columns=['Shots'])
                        fig_shots = px.bar(shots_df, x=shots_df.index, y='Shots', 
                                          title="التسديدات لكل فريق", 
                                          labels={'index': 'الفريق', 'Shots': 'عدد التسديدات'})
                        st.plotly_chart(fig_shots)
                    
                    # عرض اللاعبين
                    st.subheader("قائمة اللاعبين")
                    st.dataframe(players_df[['name', 'position', 'teamId']], 
                                hide_index=True)
                else:
                    st.error("فشل في معالجة بيانات المباراة.")
            else:
                st.error("فشل في استخراج بيانات المباراة.")

if __name__ == "__main__":
    main()
