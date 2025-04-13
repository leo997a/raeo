import json
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
import warnings
import streamlit as st
import plotly.express as px  # للرسوم البيانية

warnings.filterwarnings("ignore", category=DeprecationWarning)

def extract_match_dict(match_url, save_output=False):
    """Extract match event from whoscored match center"""
    try:
        service = webdriver.ChromeService()
        driver = webdriver.Chrome(service=service)
        driver.get(match_url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        element = soup.select_one('script:-soup-contains("matchCentreData")')
        
        if not element:
            st.error("Could not find match data. Ensure the URL is correct.")
            return None
        
        matchdict = json.loads(element.text.split("matchCentreData: ")[1].split(',\n')[0])
        driver.quit()
        return matchdict
    except Exception as e:
        st.error(f"Error extracting data: {e}")
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
        return None
    df = pd.DataFrame(events_dict)
    
    # تحليل الأهداف
    goals = df[df['type'] == 'Goal'][['teamId', 'playerId', 'minute']]
    goals_count = goals.groupby('teamId').size().to_dict()
    stats = {
        'Goals': {teams_dict.get(k, 'Unknown'): v for k, v in goals_count.items()}
    }
    
    # تحليل التسديدات (مثال)
    shots = df[df['type'].isin(['ShotOnTarget', 'ShotOffTarget'])][['teamId']]
    shots_count = shots.groupby('teamId').size().to_dict()
    stats['Shots'] = {teams_dict.get(k, 'Unknown'): v for k, v in shots_count.items()}
    
    return stats

def main():
    st.title("Football Match Analyzer")
    st.write("Enter a WhoScored match URL to analyze the game.")
    
    # إدخال رابط المباراة
    match_url = st.text_input("Match URL", placeholder="e.g., https://1xbet.whoscored.com/Matches/...")
    
    if st.button("Analyze Match"):
        if not match_url:
            st.warning("Please enter a valid URL.")
            return
        
        with st.spinner("Extracting match data..."):
            json_data = extract_match_dict(match_url)
            if json_data:
                events_dict, players_df, teams_dict = extract_data_from_dict(json_data)
                
                if events_dict:
                    # عرض الأحداث
                    st.subheader("Match Events")
                    df = pd.DataFrame(events_dict)
                    st.dataframe(df[['minute', 'type', 'playerId', 'teamId']].head(), hide_index=True)
                    
                    # تحليل الإحصائيات
                    st.subheader("Match Statistics")
                    stats = analyze_events(events_dict, teams_dict)
                    if stats:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Goals**")
                            st.json(stats['Goals'])
                        with col2:
                            st.write("**Shots**")
                            st.json(stats['Shots'])
                        
                        # رسم بياني للأهداف
                        goals_df = pd.DataFrame.from_dict(stats['Goals'], orient='index', columns=['Goals'])
                        fig = px.bar(goals_df, x=goals_df.index, y='Goals', title="Goals per Team")
                        st.plotly_chart(fig)
                    
                    # عرض اللاعبين
                    st.subheader("Players")
                    st.dataframe(players_df[['name', 'position', 'teamId']], hide_index=True)
                else:
                    st.error("Failed to process match data.")

if __name__ == "__main__":
    main()
