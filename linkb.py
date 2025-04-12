import json
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import warnings
import streamlit as st

warnings.filterwarnings("ignore", category=DeprecationWarning)

whoscored_url = "https://1xbet.whoscored.com/Matches/1809770/Live/Europe-Europa-League-2023-2024-West-Ham-Bayer-Leverkusen"

def extract_match_dict(match_url, save_output=False):
    """Extract match event from whoscored match center"""
    
    # إعداد خيارات Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # تشغيل بدون واجهة رسومية
    chrome_options.add_argument("--no-sandbox")  # ضروري في بيئات Docker
    chrome_options.add_argument("--disable-dev-shm-usage")  # تجنب مشاكل الذاكرة
    chrome_options.add_argument("--disable-gpu")  # تعطيل GPU في الوضع headless

    # إعداد Chrome WebDriver
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )
    
    try:
        driver.get(match_url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        element = soup.select_one('script:-soup-contains("matchCentreData")')
        
        if element:
            matchdict = json.loads(element.text.split("matchCentreData: ")[1].split(',\n')[0])
        else:
            raise ValueError("Could not find matchCentreData in the page source")

        return matchdict
    
    finally:
        driver.quit()

def extract_data_from_dict(data):
    """Extract events, players, and teams from match dictionary"""
    events_dict = data["events"]
    teams_dict = {
        data['home']['teamId']: data['home']['name'],
        data['away']['teamId']: data['away']['name']
    }
    
    players_home_df = pd.DataFrame(data['home']['players'])
    players_home_df["teamId"] = data['home']['teamId']
    players_away_df = pd.DataFrame(data['away']['players'])
    players_away_df["teamId"] = data['away']['teamId']
    players_df = pd.concat([players_home_df, players_away_df])
    
    return events_dict, players_df, teams_dict

# استخراج البيانات
match_url = whoscored_url
json_data = extract_match_dict(match_url)
events_dict, players_df, teams_dict = extract_data_from_dict(json_data)

# إنشاء إطارات البيانات
df = pd.DataFrame(events_dict)
dfp = pd.DataFrame(players_df)

# عرض البيانات في Streamlit
st.dataframe(df.head(), hide_index=True)
