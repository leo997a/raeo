#!/bin/bash
pip install --no-cache-dir -r requirements.txt
streamlit run linkb.py --server.port 8501
# Install prerequisites
apt-get update
apt-get install -y wget gnupg ca-certificates

# Add Google Chrome repository key
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add -

# Add Google Chrome repository
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list

# Update package lists
apt-get update

# Install Google Chrome and chromedriver
apt-get install -y google-chrome-stable
apt-get install -y chromedriver
chmod +x setup.sh
