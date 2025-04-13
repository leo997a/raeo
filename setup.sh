#!/bin/bash
# تحديث قائمة الحزم
apt-get update
# تثبيت مكتبات النظام المطلوبة لـ Chrome وChromeDriver
apt-get install -y \
    wget \
    gnupg \
    unzip \
    libglib2.0-0 \
    libnss3 \
    libgconf-2-4 \
    libfontconfig1 \
    libx11-6 \
    libx11-xcb1 \
    libxi6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libxtst6 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcairo2 \
    libcups2 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libasound2 \
    fonts-liberation \
    xdg-utils
# إضافة مفتاح Google Chrome ومستودعه
wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list
# تحديث المستودعات وتثبيت Google Chrome
apt-get update
apt-get install -y google-chrome-stable
# تنزيل وتثبيت ChromeDriver (الإصدار 135.0.7049.84)
wget -q https://storage.googleapis.com/chrome-for-testing-public/135.0.7049.84/linux64/chromedriver-linux64.zip
unzip chromedriver-linux64.zip
mv chromedriver-linux64/chromedriver /usr/local/bin/
chmod +x /usr/local/bin/chromedriver
# تنظيف الملفات المؤقتة
rm -rf chromedriver-linux64.zip chromedriver-linux64
