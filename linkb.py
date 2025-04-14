from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import json
import pandas as pd
import re
import streamlit as st

st.title("مستخرج بيانات مباريات WhoScored")

def fetch_whoscored_data(match_url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(match_url)
            page.wait_for_timeout(5000)
            soup = BeautifulSoup(page.content(), "html.parser")
            browser.close()
            
            scripts = soup.find_all("script")
            for script in scripts:
                if script.string and "matchCentreData" in script.string:
                    match = re.search(r"matchCentreData\s*:\s*({.*?})\s*,", script.string, re.DOTALL)
                    if match:
                        data_str = match.group(1)
                        return json.loads(data_str)
            st.error("لم يتم العثور على بيانات المباراة في السكربت!")
            return None
    except Exception as e:
        st.error(f"حدث خطأ: {e}")
        return None

match_url = st.text_input("أدخل رابط المباراة من WhoScored:", 
                          "https://1xbet.whoscored.com/matches/1821689/live/spain-laliga-2024-2025-deportivo-alaves-real-madrid")

if match_url:
    with st.spinner("جاري جلب البيانات..."):
        data = fetch_whoscored_data(match_url)
    
    if data:
        events = pd.DataFrame(data.get("events", []))
        st.success("تم جلب البيانات بنجاح!")
        if not events.empty:
            st.dataframe(events.head())
        else:
            st.warning("لا توجد أحداث في البيانات!")
