import json
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import streamlit as st

st.title("WhoScored Match Data Extractor")

# إدخال رابط المباراة من المستخدم
match_url = st.text_input("Enter WhoScored Match URL:", "https://www.whoscored.com/Matches/1809770/Live")

def extract_match_dict(match_url):
    """استخراج بيانات المباراة باستخدام Selenium"""
    try:
        # إعداد متصفح Chrome (مع webdriver-manager)
        service = Service(ChromeDriverManager().install())
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")  # تشغيل بدون واجهة (مهم للسيرفر)
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(service=service, options=options)
        
        driver.get(match_url)
        time.sleep(3)  # انتظر حتى يتم تحميل الصفحة
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        element = soup.select_one('script:-soup-contains("matchCentreData")')
        
        if not element:
            st.error("لم يتم العثور على بيانات المباراة!")
            return None
            
        match_data = json.loads(element.text.split("matchCentreData: ")[1].split(',\n')[0])
        driver.quit()  # أغلق المتصفح
        return match_data
        
    except Exception as e:
        st.error(f"حدث خطأ: {e}")
        return None

if match_url:
    with st.spinner("جاري جلب بيانات المباراة..."):
        data = extract_match_dict(match_url)
    
    if data:
        # استخراج الأحداث واللاعبين
        events = pd.DataFrame(data["events"])
        players_home = pd.DataFrame(data["home"]["players"])
        players_away = pd.DataFrame(data["away"]["players"])
        players = pd.concat([players_home, players_away])
        
        st.success("تم جلب البيانات بنجاح!")
        st.write("### أحداث المباراة", events.head())
        st.write("### لاعبي الفريقين", players.head())
