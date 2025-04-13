import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import streamlit as st

st.title("WhoScored Match Data Extractor")

def fetch_whoscored_data(match_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(match_url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        script = soup.find("script", string=lambda t: "matchCentreData" in str(t))
        if script:
            data_str = script.text.split("matchCentreData: ")[1].split(",\n")[0]
            return json.loads(data_str)
        else:
            st.error("لم يتم العثور على بيانات المباراة!")
            return None
    except Exception as e:
        st.error(f"حدث خطأ: {e}")
        return None

match_url = st.text_input("أدخل رابط المباراة من WhoScored:", "https://www.whoscored.com/Matches/1809770/Live")

if match_url:
    with st.spinner("جاري جلب البيانات..."):
        data = fetch_whoscored_data(match_url)
    
    if data:
        events = pd.DataFrame(data["events"])
        st.success("تم جلب البيانات بنجاح!")
        st.dataframe(events.head())
