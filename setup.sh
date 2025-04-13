#!/bin/bash
# تحديث قائمة الحزم
apt-get update
# تثبيت الأدوات المساعدة
apt-get install -y wget gnupg unzip
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
