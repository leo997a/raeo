import streamlit as st
import pandas as pd
import requests

@st.cache_data(ttl=60)  # يخزّن البيانات لـ60 ثانية لتقليل الطلبات
def fetch_summary(match_id):
    url = f"https://prod-public-api.whoscored.com/api/v2/match/MatchSummary?matchId={match_id}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
    }
    return requests.get(url, headers=headers).json()

st.title("WhoScored Data Explorer")

match_id = st.text_input("Enter match ID:", "")

if match_id:
    try:
        data = fetch_summary(match_id)
        # تحويل إحصائيات اللاعبين إلى DataFrame
        home_df = pd.DataFrame(data["homeTeam"]["playerStatistics"])
        away_df = pd.DataFrame(data["awayTeam"]["playerStatistics"])

        st.subheader(f"Home Team: {data['homeTeam']['teamName']}")
        st.dataframe(home_df[["playerName", "rating", "minutesPlayed"]])

        st.subheader(f"Away Team: {data['awayTeam']['teamName']}")
        st.dataframe(away_df[["playerName", "rating", "minutesPlayed"]])
    except Exception as e:
        st.error(f"Error fetching data: {e}")
