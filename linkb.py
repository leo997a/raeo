import streamlit as st
import requests
import cloudscraper
import pandas as pd
import numpy as np
import time
import random
from fake_useragent import UserAgent
import arabic_reshaper
from bidi.algorithm import get_display
import os

# إعدادات Streamlit
st.set_page_config(page_title=reshape_arabic_text("تحليل مباريات WhoScored"), layout="wide")

# دالة لتنسيق النصوص العربية
def reshape_arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# دالة لاستخراج بيانات المباراة من Hidden API
@st.cache_data(show_spinner=False)
def extract_match_dict(match_id, retries=3, delay=5):
    """
    استخراج بيانات مباراة من Hidden API باستخدام matchId.
    
    Args:
        match_id (str): معرف المباراة
        retries (int): عدد المحاولات
        delay (int): التأخير بين المحاولات (ثوان)
    
    Returns:
        dict: بيانات المباراة بصيغة JSON، أو None إذا فشل
    """
    # نقطة النهاية الافتراضية - استبدلها بالرابط الحقيقي من Network requests
    url = f"https://1xbet.whoscored.com/matches/1821690/live/spain-laliga-2024-2025-leganes-barcelona"
    headers = {
        "Accept": "*/*",
        "Referer": f"https://1xbet.whoscored.com/Matches/{match_id}/Live",
        "Sec-Ch-Ua": '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "User-Agent": UserAgent().random,
        "X-Kl-Saas-Ajax-Request": "Ajax_Request",
        "X-Requested-With": "XMLHttpRequest",
        # "model-last-mode": "tsDxcBt6yu5JL+yeWfokrty/eQU18bcLdmOZlweljBE="  # قد تكون ديناميكية، أزلها إذا لم تكن ضرورية
    }
    
    scraper = cloudscraper.create_scraper()
    
    for attempt in range(retries):
        try:
            response = scraper.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not data or "events" not in data:
                st.warning(reshape_arabic_text("البيانات المستلمة غير مكتملة، جارٍ المحاولة مرة أخرى..."))
                continue
            return data
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                st.warning(reshape_arabic_text(f"تم حظر الطلب (429)، المحاولة {attempt+1}/{retries} بعد {delay} ثوان..."))
                time.sleep(delay + random.uniform(0, 2))
            elif response.status_code == 403:
                st.error(reshape_arabic_text("الوصول مرفوض (403)، قد تكون هناك حماية Cloudflare."))
                return None
            else:
                st.error(reshape_arabic_text(f"خطأ HTTP: {str(e)}"))
        except requests.exceptions.RequestException as e:
            st.error(reshape_arabic_text(f"خطأ في الطلب: {str(e)}"))
        time.sleep(delay + random.uniform(0, 2))
    st.error(reshape_arabic_text(f"فشل استخراج البيانات لمباراة {match_id} بعد {retries} محاولات."))
    return None

# دالة لإنشاء إطار بيانات الأحداث
def create_events_df(match_data):
    """
    تحويل بيانات المباراة إلى DataFrame للأحداث.
    
    Args:
        match_data (dict): بيانات المباراة
    
    Returns:
        pd.DataFrame: إطار بيانات الأحداث
    """
    if not match_data or "events" not in match_data:
        return pd.DataFrame()
    
    events = match_data["events"]
    for event in events:
        event.update({
            "matchId": match_data.get("matchId", ""),
            "startDate": match_data.get("startDate", ""),
            "startTime": match_data.get("startTime", ""),
            "score": match_data.get("score", ""),
            "ftScore": match_data.get("ftScore", ""),
            "htScore": match_data.get("htScore", ""),
            "venueName": match_data.get("venueName", ""),
            "maxMinute": match_data.get("maxMinute", 0)
        })
    
    df = pd.DataFrame(events)
    
    # تنظيف الأعمدة
    for col in ["period", "type", "outcomeType"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.get("displayName", "") if isinstance(x, dict) else x)
    
    # إضافة أسماء اللاعبين
    if "playerId" in df.columns and "playerIdNameDictionary" in match_data:
        df["playerName"] = df["playerId"].map(match_data["playerIdNameDictionary"])
    
    # إضافة home/away
    if "teamId" in df.columns and "home" in match_data and "away" in match_data:
        team_map = {
            match_data["home"]["teamId"]: "h",
            match_data["away"]["teamId"]: "a"
        }
        df["h_a"] = df["teamId"].map(team_map)
    
    return df

# واجهة Streamlit
st.title(reshape_arabic_text("استخراج بيانات مباراة من WhoScored"))
st.markdown(reshape_arabic_text("أدخل معرف المباراة لاستخراج البيانات تلقائيًا باستخدام Hidden API."))

match_id = st.text_input(reshape_arabic_text("معرف المباراة (مثل 1821690)"), value="1821690")

if st.button(reshape_arabic_text("استخراج البيانات")):
    with st.spinner(reshape_arabic_text("جارٍ استخراج بيانات المباراة...")):
        match_data = extract_match_dict(match_id)
        if match_data:
            events_df = create_events_df(match_data)
            if not events_df.empty:
                st.success(reshape_arabic_text("تم استخراج البيانات بنجاح!"))
                
                # عرض معلومات أساسية
                home_team = match_data["home"]["name"]
                away_team = match_data["away"]["name"]
                score = match_data.get("score", "غير متوفر")
                st.markdown(reshape_arabic_text(f"**المباراة**: {home_team} ضد {away_team}"))
                st.markdown(reshape_arabic_text(f"**النتيجة**: {score}"))
                
                # عرض إطار البيانات
                st.markdown(reshape_arabic_text("### إطار بيانات الأحداث"))
                st.dataframe(events_df)
                
                # إحصائيات إضافية
                passes = events_df[(events_df["type"] == "Pass") & (events_df["outcomeType"] == "Successful")]
                st.markdown(reshape_arabic_text(f"**إجمالي التمريرات الناجحة**: {len(passes)}"))
            else:
                st.error(reshape_arabic_text("لم يتم العثور على أحداث للمباراة."))
        else:
            st.error(reshape_arabic_text("فشل استخراج البيانات. تحقق من معرف المباراة أو حاول لاحقًا."))
