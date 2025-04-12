# استخدام صورة Python رسمية
FROM python:3.12-slim

# إعداد دليل العمل
WORKDIR /app

# نسخ ملفات المشروع
COPY . .

# تثبيت تبعيات النظام اللازمة لـ Chrome و chromedriver
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && wget https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/129.0.6668.100/linux64/chromedriver-linux64.zip \
    && unzip chromedriver-linux64.zip \
    && mv chromedriver-linux64/chromedriver /usr/local/bin/chromedriver \
    && chmod +x /usr/local/bin/chromedriver \
    && rm -rf chromedriver-linux64.zip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# تثبيت تبعيات Python من requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# تعيين أمر التشغيل
CMD ["bash", "setup.sh"]
