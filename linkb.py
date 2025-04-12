import json
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
import warnings
import streamlit as st

warnings.filterwarnings("ignore", category=DeprecationWarning)  # Ignore deprecations in this file

whoscored_url = "https://1xbet.whoscored.com/Matches/1809770/Live/Europe-Europa-League-2023-2024-West-Ham-Bayer-Leverkusen"

def extract_match_dict(match_url, save_output=False):
    """Extract match event from whoscored match center"""
    
    # Initialize the Chrome driver with the service
    service = webdriver.ChromeService()
    driver = webdriver.Chrome(service=service)
    # Open Google Chrome with chromedriver
    driver.get(match_url)
    # driver.maximize_window()
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    element = soup.select_one('script:-soup-contains("matchCentreData")')
    
    matchdict = json.loads(element.text.split("matchCentreData: ")[1].split(',\n')[0])

    # if save_output:
    #     # save json data to txt
    #     output_file = open(f"{match_url}.txt", "wt")
    #     n = output_file.write(data)
    #     output_file.close()

    return matchdict


def extract_data_from_dict(data):
    # load data from json
    # event_types_json = data["matchCentreEventTypeJson"]
    # formation_mappings = data["formationIdNameMappings"]
    events_dict = data["events"]
    teams_dict = {data['home']['teamId']: data['home']['name'],
                  data['away']['teamId']: data['away']['name']}
    players_dict = data["playerIdNameDictionary"]
    # create players dataframe
    players_home_df = pd.DataFrame(data['home']['players'])
    players_home_df["teamId"] = data['home']['teamId']
    players_away_df = pd.DataFrame(data['away']['players'])
    players_away_df["teamId"] = data['away']['teamId']
    players_df = pd.concat([players_home_df, players_away_df])
    players_ids = data["playerIdNameDictionary"]
    return events_dict, players_df, teams_dict


match_url = whoscored_url
json_data = extract_match_dict(match_url)
data = json_data
events_dict, players_df, teams_dict = extract_data_from_dict(data)
    
df = pd.DataFrame(events_dict)
dfp = pd.DataFrame(players_df)
    
st.dataframe(df.head(), hide_index=True)

