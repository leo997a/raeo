import json
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import warnings
import streamlit as st
import time

warnings.filterwarnings("ignore", category=DeprecationWarning)

whoscored_url = "https://1xbet.whoscored.com/Matches/1809770/Live/Europe-Europa-League-2023-2024-West-Ham-Bayer-Leverkusen"

def extract_match_dict(match_url, save_output=False):
    """Extract match event from whoscored match center"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )
    
    try:
        driver.get(match_url)
        time.sleep(5)  # انتظار تحميل الصفحة
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        element = soup.select_one('script:-soup-contains("matchCentreData")')
        
        if element:
            matchdict = json.loads(element.text.split("matchCentreData: ")[1].split(',\n')[0])
            return matchdict
        else:
            st.error("Could not find matchCentreData in the page source")
            return None
    
    except Exception as e:
        st.error(f"Error extracting data: {str(e)}")
        return None
    
    finally:
        driver.quit()

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
        st.error(f"Data structure error: Missing key {str(e)}")
        return None, None, None

# استخراج البيانات
match_url = whoscored_url
json_data = extract_match_dict(match_url)
events_dict, players_df, teams_dict = extract_data_from_dict(json_data)

# عرض البيانات
if events_dict is not None:
    df = pd.DataFrame(events_dict)
    st.dataframe(df.head(), hide_index=True)
else:
    st.error("Failed to load event data")
