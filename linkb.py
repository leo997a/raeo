import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import re
import os
import time

st.title("مستخرج بيانات مباريات WhoScored باستخدام ScraperAPI")

# قراءة مفتاح API من متغير بيئي
SCRAPERAPI_KEY = os.getenv("SCRAPERAPI_KEY")

def fetch_whoscored_data(match_url, max_retries=3):
    if not SCRAPERAPI_KEY:
        st.error("مفتاح ScraperAPI غير محدد! أضفه في إعدادات Streamlit Cloud.")
        return None
    
    for attempt in range(1, max_retries + 1):
        try:
            # إعداد الطلب إلى ScraperAPI
            payload = {
                'api_key': SCRAPERAPI_KEY,
                'url': match_url,
                'render': 'true',  # تفعيل جلب محتوى JavaScript
                'country_code': 'us',  # وكيل في الولايات المتحدة
                'premium': 'true'  # استخدام وكلاء مميزين
            }
            
            # إرسال الطلب مع مهلة أطول
            st.write(f"جاري جلب البيانات من ScraperAPI (محاولة {attempt}/{max_retries})...")
            response = requests.get('https://api.scraperapi.com/', params=payload, timeout=60)
            response.raise_for_status()  # التحقق من نجاح الطلب
            
            # تحليل محتوى الصفحة
            soup = BeautifulSoup(response.text, "html.parser")
            
            # البحث عن السكربت الذي يحتوي على matchCentreData
            scripts = soup.find_all("script")
            for script in scripts:
                if script.string and "matchCentreData" in script.string:
                    match = re.search(r"matchCentreData\s*:\s*({.*?})\s*,", script.string, re.DOTALL)
                    if match:
                        data_str = match.group(1)
                        return json.loads(data_str)
            st.error("لم يتم العثور على بيانات المباراة في السكربت! تأكد من أن الرابط يحتوي على بيانات المباراة.")
            return None
        except requests.exceptions.Timeout:
            st.warning(f"انتهت مهلة الطلب في المحاولة {attempt}. جاري إعادة المحاولة...")
            if attempt == max_retries:
                st.error("فشل جلب البيانات بعد عدة محاولات بسبب انتهاء المهلة.")
                return None
            time.sleep(5)  # الانتظار قبل إعادة المحاولة
        except requests.exceptions.RequestException as e:
            st.error(f"خطأ في جلب البيانات: {e}")
            return None
        except json.JSONDecodeError as e:
            st.error(f"خطأ في تحليل JSON: {e}")
            return None
        except Exception as e:
            st.error(f"حدث خطأ غير متوقع: {e}")
            return None

# إدخال رابط المباراة
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
            # إضافة خيار تحميل البيانات كـ CSV
            csv = events.to_csv(index=False)
            st.download_button("تحميل البيانات كـ CSV", csv, "match_data.csv", "text/csv")
        else:
            st.warning("لا توجد أحداث في البيانات!")
