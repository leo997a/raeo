import streamlit as st
import pandas as pd
import requests
import re

@st.cache_data(ttl=60)
def fetch_summary(match_id):
    url = f"https://prod-public-api.whoscored.com/api/v2/match/MatchSummary?matchId={match_id}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
    }
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()

def extract_match_id(url):
    patterns = [r"/Matches/(\d+)/", r"/matches/(\d+)/"]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    raise ValueError("Could not extract match ID from URL")

st.title("WhoScored Data Explorer")

input_url = st.text_input("Enter full match URL or ID:", "")

if input_url:
    try:
        # إذا كان الإدخال رابطًا، نستخرج الرقم؛ وإلا نفترض أنه رقم بحت
        if input_url.startswith("http"):
            match_id = extract_match_id(input_url)
        else:
            match_id = input_url.strip()

        data = fetch_summary(match_id)

        # عرض بيانات الفريقين
        home = data["homeTeam"]
        away = data["awayTeam"]

        home_df = pd.DataFrame(home["playerStatistics"])
        away_df = pd.DataFrame(away["playerStatistics"])

        st.subheader(f"Home Team: {home['teamName']}")
        st.dataframe(home_df[["playerName", "rating", "minutesPlayed"]])

        st.subheader(f"Away Team: {away['teamName']}")
        st.dataframe(away_df[["playerName", "rating", "minutesPlayed"]])

    except ValueError as ve:
        st.error(str(ve))
    except requests.exceptions.RequestException as re_err:
        st.error(f"Network error: {re_err}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
