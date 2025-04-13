import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import chromedriver_autoinstaller
import re
import time

st.title("مستخرج بيانات مباريات WhoScored")

def fetch_whoscored_data(match_url):
    try:
        # تثبيت ChromeDriver تلقائيًا
        chromedriver_autoinstaller.install()

        # إعداد Selenium
        options = Options()
        options.add_argument("--headless")  # تشغيل بدون واجهة
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        # تحديد مسار Chromium في Streamlit Cloud
        options.binary_location = "/usr/lib/chromium-browser/chrome"  # مسار Chromium في Streamlit Cloud

        # إعداد Service لـ ChromeDriver
        service = Service("/usr/lib/chromium-browser/chromedriver")  # مسار ChromeDriver في Streamlit Cloud
        driver = webdriver.Chrome(service=service, options=options)
        
        st.write("جاري الوصول إلى الرابط...")
        driver.get(match_url)
        time.sleep(5)  # التأخير لضمان التحميل
        
        # استخراج المصدر
        soup = BeautifulSoup(driver.page_source, "html.parser")
        driver.quit()
        
        # البحث عن السكربت
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

# إدخال الرابط
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
