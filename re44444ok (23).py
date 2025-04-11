import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import re
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
import streamlit as st
from urllib.request import urlopen
from PIL import Image
from unidecode import unidecode
import arabic_reshaper
from bidi.algorithm import get_display

# ------------------------ CONFIGURATION ------------------------
XT_GRID_URL = (
    "https://raw.githubusercontent.com/adnaaan433/Post-Match-Report-2.0/refs/heads/main/xT_Grid.csv"
)
DEFAULT_URL = (
    "https://1xbet.whoscored.com/Matches/1809770/Live/Europe-Europa-League-2023-2024-West-Ham-Bayer-Leverkusen"
)

# ------------------------ UTILITIES ------------------------
def reshape_arabic_text(text: str) -> str:
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

@st.cache_data
def fetch_html(url: str) -> str:
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.text

@st.cache_data
def extract_match_dict(match_url: str) -> dict:
    """Extract raw JSON dict from WhoScored match page"""
    # Selenium to load dynamic content
    service = ChromeService()
    driver = webdriver.Chrome(service=service)
    driver.get(match_url)
    html = driver.page_source
    driver.quit()
    # parse embedded JS
    soup = BeautifulSoup(html, 'html.parser')
    script = soup.select_one('script:-soup-contains("matchCentreData")')
    data_txt = script.text.split("matchCentreData: ")[1].split(',\n')[0]
    return json.loads(data_txt)

@st.cache_data
def extract_data_from_dict(data: dict):
    mc = data.get('matchCentreData', data)
    events = pd.DataFrame(mc['events'])
    # clean display names
    for col in ['type','outcomeType','period']:
        events[col] = (
            events[col].astype(str)
            .str.extract(r"'displayName': '([^']+)'" )
            .fillna(events[col])
        )
    # map period names to ints
    period_map = {
        'FirstHalf':1,'SecondHalf':2,'FirstPeriodOfExtraTime':3,
        'SecondPeriodOfExtraTime':4,'PenaltyShootout':5,
        'PostGame':14,'PreMatch':16
    }
    events['period'] = events['period'].map(period_map).fillna(0).astype(int)

    # players DataFrame
    home = pd.DataFrame(mc['home']['players']).assign(teamId=mc['home']['teamId'])
    away = pd.DataFrame(mc['away']['players']).assign(teamId=mc['away']['teamId'])
    players = pd.concat([home, away], ignore_index=True)
    players['name'] = players['name'].apply(lambda x: unidecode(str(x)))

    teams = {
        mc['home']['teamId']: mc['home']['name'],
        mc['away']['teamId']: mc['away']['name']
    }
    return events, players, teams

# ------------------------ STREAMLIT UI ------------------------
def main():
    st.set_page_config(layout="wide")
    st.sidebar.title(reshape_arabic_text("اختيارات المباراة"))
    match_url = st.sidebar.text_input(
        reshape_arabic_text("رابط المباراة في WhoScored:"),
        value=DEFAULT_URL
    )
    if st.sidebar.button(reshape_arabic_text("تحميل البيانات")):
        try:
            raw = extract_match_dict(match_url)
            events, players, teams = extract_data_from_dict(raw)
            st.success(reshape_arabic_text("تم استخراج البيانات بنجاح"))
            st.dataframe(events.head(), hide_index=True)
        except Exception as e:
            st.error(reshape_arabic_text(f"فشل في تحميل البيانات: {e}"))

if __name__ == '__main__':
    main()
