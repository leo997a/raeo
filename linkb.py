import streamlit as st
import requests
import cloudscraper
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
import seaborn as sns
import time
import random
from fake_useragent import UserAgent
import arabic_reshaper
from bidi.algorithm import get_display
import os

# إعدادات Streamlit
st.set_page_config(page_title=reshape_arabic_text("تحليل مباريات WhoScored"), layout="wide")
plt.style.use("dark_background")

# دالة لتنسيق النصوص العربية
def reshape_arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# دالة لاستخراج بيانات المباراة من Hidden API
@st.cache_data(show_spinner=False)
def get_match_data(match_id, retries=3, delay=5):
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
    url = f"https://1xbet.whoscored.com/Matches/{match_id}/LiveStatistics"
    headers = {
        "Accept": "*/*",
        "Referer": f"https://1xbet.whoscored.com/Matches/{match_id}/Live",
        "Sec-Ch-Ua": '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "User-Agent": UserAgent().random,  # تغيير ديناميكي لتجنب الحظر
        "X-Kl-Saas-Ajax-Request": "Ajax_Request",
        "X-Requested-With": "XMLHttpRequest"
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
    
    # إضافة endPlayerName (افتراضيًا للتمريرات)
    if "qualifiers" in df.columns:
        df["endPlayerName"] = None
        for i, row in df.iterrows():
            if row["type"] == "Pass" and isinstance(row["qualifiers"], list):
                for q in row["qualifiers"]:
                    if q.get("type", {}).get("displayName") == "PassRecipient":
                        df.at[i, "endPlayerName"] = match_data["playerIdNameDictionary"].get(str(q.get("value")))
    
    return df

# دالة لإنشاء شبكة التمريرات
def create_pass_network(df, team_id, team_name):
    """
    إنشاء شبكة التمريرات لفريق معين.
    
    Args:
        df (pd.DataFrame): إطار بيانات الأحداث
        team_id (int): معرف الفريق
        team_name (str): اسم الفريق
    
    Returns:
        dict: بيانات وشكل شبكة التمريرات
    """
    passes = df[(df["type"] == "Pass") & (df["outcomeType"] == "Successful") & (df["teamId"] == team_id)]
    if passes.empty:
        return None
    
    pass_counts = passes.groupby(["playerName", "endPlayerName"]).size().reset_index(name="passes")
    total_passes = passes.groupby("playerName").size().reset_index(name="total_passes")
    
    # إنشاء التصور
    pitch = VerticalPitch(pitch_color="#313332", line_color="white", line_zorder=2)
    fig, ax = pitch.draw(figsize=(6, 8))
    fig.set_facecolor("#313332")
    
    for _, row in pass_counts.iterrows():
        player1 = row["playerName"]
        player2 = row["endPlayerName"]
        count = row["passes"]
        if player1 in passes["playerName"].values and player2 in passes["endPlayerName"].values:
            x1, y1 = passes[passes["playerName"] == player1][["x", "y"]].mean()
            x2, y2 = passes[passes["endPlayerName"] == player2][["endX", "endY"]].mean()
            if pd.notna([x1, y1, x2, y2]).all():
                pitch.lines(x1, y1, x2, y2, ax=ax, color="cyan", lw=count/5, alpha=0.5, zorder=1)
                pitch.scatter(x1, y1, ax=ax, color="red", s=100, zorder=3)
                ax.text(x1, y1+2, player1.split()[-1][:3], color="white", fontsize=8, ha="center", va="bottom")
    
    ax.set_title(reshape_arabic_text(f"شبكة التمريرات - {team_name}"), color="white", fontsize=14, pad=10)
    return {"fig": fig, "data": pass_counts, "total_passes": total_passes}

# دالة لإنشاء Convex Hulls
def create_convex_hulls(df, team_id, team_name):
    """
    إنشاء Convex Hulls للاعبي فريق معين.
    
    Args:
        df (pd.DataFrame): إطار بيانات الأحداث
        team_id (int): معرف الفريق
        team_name (str): اسم الفريق
    
    Returns:
        dict: بيانات وشكل Convex Hulls
    """
    actions = df[(df["teamId"] == team_id) & (df["type"].isin(["Pass", "Tackle", "Interception", "Shot"]))]
    if actions.empty:
        return None
    
    pitch = VerticalPitch(pitch_color="#313332", line_color="white", line_zorder=2)
    fig, ax = pitch.draw(figsize=(6, 8))
    fig.set_facecolor("#313332")
    
    hulls = []
    for player in actions["playerName"].unique():
        player_actions = actions[actions["playerName"] == player][["x", "y"]].dropna()
        if len(player_actions) >= 5:  # الحد الأدنى للأحداث
            try:
                points = player_actions.values
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                hulls.append({"player": player, "hull_vertices": hull_points})
                # رسم الـ Hull
                for simplex in hull.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1], "cyan", lw=1, alpha=0.5)
                centroid = np.mean(hull_points, axis=0)
                ax.text(centroid[0], centroid[1], player.split()[-1][:3], color="white", fontsize=8, ha="center")
            except:
                continue
    
    ax.set_title(reshape_arabic_text(f"الشكل التكتيكي - {team_name}"), color="white", fontsize=14, pad=10)
    return {"fig": fig, "hulls": hulls}

# دالة لتحميل EPV Grid
def load_epv_grid(fname="EPV_grid.csv"):
    """
    تحميل ملف EPV Grid.
    
    Args:
        fname (str): مسار الملف
    
    Returns:
        np.ndarray: شبكة EPV
    """
    if not os.path.exists(fname):
        st.error(reshape_arabic_text(f"ملف {fname} غير موجود. سيتم تخطي تحليل EPV."))
        return None
    return np.loadtxt(fname, delimiter=",")

# دالة لحساب EPV في موقع معين
def get_epv_at_location(position, epv, attack_direction=1, field_dimen=(106., 68.)):
    """
    حساب قيمة EPV في موقع معين.
    
    Args:
        position (tuple): الإحداثيات (x,y)
        epv (np.ndarray): شبكة EPV
        attack_direction (int): اتجاه الهجوم
        field_dimen (tuple): أبعاد الملعب
    
    Returns:
        float: قيمة EPV
    """
    x, y = position
    if abs(x) > field_dimen[0]/2 or abs(y) > field_dimen[1]/2:
        return 0.0
    if attack_direction == -1:
        epv = np.fliplr(epv)
    ny, nx = epv.shape
    dx = field_dimen[0] / nx
    dy = field_dimen[1] / ny
    ix = int((x + field_dimen[0]/2) / dx)
    iy = int((y + field_dimen[1]/2) / dy)
    return epv[iy, ix]

# دالة لتحويل الإحداثيات إلى مترية
def to_metric_coordinates(df):
    """
    تحويل الإحداثيات من وحدات WhoScored إلى مترية.
    
    Args:
        df (pd.DataFrame): إطار بيانات الأحداث
    
    Returns:
        pd.DataFrame: إطار البيانات مع إحداثيات مترية
    """
    df["x_metrica"] = (df["x"] / 100 * 106) - 53
    df["y_metrica"] = (df["y"] / 100 * 68) - 34
    df["endX_metrica"] = (df["endX"] / 100 * 106) - 53 if "endX" in df.columns else df["x_metrica"]
    df["endY_metrica"] = (df["endY"] / 100 * 68) - 34 if "endY" in df.columns else df["y_metrica"]
    return df

# دالة لإضافة EPV إلى إطار البيانات
def add_epv_to_dataframe(df, epv_grid_path="EPV_grid.csv"):
    """
    إضافة قيم EPV إلى إطار بيانات الأحداث.
    
    Args:
        df (pd.DataFrame): إطار بيانات الأحداث
        epv_grid_path (str): مسار ملف EPV Grid
    
    Returns:
        pd.DataFrame: إطار البيانات مع عمود EPV
    """
    epv = load_epv_grid(epv_grid_path)
    if epv is None:
        return df
    
    df = to_metric_coordinates(df)
    df["EPV"] = 0.0
    
    for i, row in df.iterrows():
        if row["type"] == "Pass" and row["outcomeType"] == "Successful":
            start_pos = (row["x_metrica"], row["y_metrica"])
            end_pos = (row["endX_metrica"], row["endY_metrica"])
            start_epv = get_epv_at_location(start_pos, epv)
            end_epv = get_epv_at_location(end_pos, epv)
            df.at[i, "EPV"] = end_epv - start_epv
    
    df.drop(["x_metrica", "y_metrica", "endX_metrica", "endY_metrica"], axis=1, inplace=True)
    return df

# واجهة Streamlit
st.title(reshape_arabic_text("تحليل بيانات مباراة من WhoScored"))
st.markdown(reshape_arabic_text("أدخل معرف المباراة لاستخراج البيانات تلقائيًا باستخدام Hidden API."))

match_id = st.text_input(reshape_arabic_text("معرف المباراة (مثل 1821690)"), value="1821690")

if st.button(reshape_arabic_text("استخراج البيانات")):
    with st.spinner(reshape_arabic_text("جارٍ استخراج بيانات المباراة...")):
        match_data = get_match_data(match_id)
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
                
                # إضافة EPV
                events_df = add_epv_to_dataframe(events_df)
                
                # علامات تبويب للتحليلات
                tabs = st.tabs([
                    reshape_arabic_text("إطار البيانات"),
                    reshape_arabic_text("شبكة التمريرات"),
                    reshape_arabic_text("الشكل التكتيكي"),
                    reshape_arabic_text("تحليل EPV")
                ])
                
                with tabs[0]:
                    st.markdown(reshape_arabic_text("### إطار بيانات الأحداث"))
                    st.dataframe(events_df)
                
                with tabs[1]:
                    st.markdown(reshape_arabic_text("### شبكة التمريرات"))
                    home_pass_network = create_pass_network(events_df, match_data["home"]["teamId"], home_team)
                    away_pass_network = create_pass_network(events_df, match_data["away"]["teamId"], away_team)
                    
                    if home_pass_network:
                        st.pyplot(home_pass_network["fig"])
                        st.markdown(reshape_arabic_text(f"**إجمالي التمريرات الناجحة لـ {home_team}**: {home_pass_network['total_passes']['total_passes'].sum()}"))
                    else:
                        st.warning(reshape_arabic_text(f"لا توجد بيانات تمريرات كافية لـ {home_team}."))
                    
                    if away_pass_network:
                        st.pyplot(away_pass_network["fig"])
                        st.markdown(reshape_arabic_text(f"**إجمالي التمريرات الناجحة لـ {away_team}**: {away_pass_network['total_passes']['total_passes'].sum()}"))
                    else:
                        st.warning(reshape_arabic_text(f"لا توجد بيانات تمريرات كافية لـ {away_team}."))
                
                with tabs[2]:
                    st.markdown(reshape_arabic_text("### الشكل الهجومي/الدفاعي"))
                    home_hulls = create_convex_hulls(events_df, match_data["home"]["teamId"], home_team)
                    away_hulls = create_convex_hulls(events_df, match_data["away"]["teamId"], away_team)
                    
                    if home_hulls:
                        st.pyplot(home_hulls["fig"])
                        st.markdown(reshape_arabic_text(f"**عدد اللاعبين المحللين لـ {home_team}**: {len(home_hulls['hulls'])}"))
                    else:
                        st.warning(reshape_arabic_text(f"لا توجد بيانات كافية لتحليل الشكل التكتيكي لـ {home_team}."))
                    
                    if away_hulls:
                        st.pyplot(away_hulls["fig"])
                        st.markdown(reshape_arabic_text(f"**عدد اللاعبين المحللين لـ {away_team}**: {len(away_hulls['hulls'])}"))
                    else:
                        st.warning(reshape_arabic_text(f"لا توجد بيانات كافية لتحليل الشكل التكتيكي لـ {away_team}."))
                
                with tabs[3]:
                    st.markdown(reshape_arabic_text("### تحليل EPV"))
                    if "EPV" in events_df.columns and events_df["EPV"].sum() != 0:
                        home_epv = events_df[events_df["h_a"] == "h"]["EPV"].sum()
                        away_epv = events_df[events_df["h_a"] == "a"]["EPV"].sum()
                        st.markdown(reshape_arabic_text(f"**إجمالي EPV لـ {home_team}**: {home_epv:.2f}"))
                        st.markdown(reshape_arabic_text(f"**إجمالي EPV لـ {away_team}**: {away_epv:.2f}"))
                        
                        # رسم توزيع EPV
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.histplot(data=events_df[events_df["EPV"] != 0], x="EPV", hue="h_a", ax=ax)
                        ax.set_title(reshape_arabic_text("توزيع قيم EPV"), color="white")
                        ax.set_xlabel(reshape_arabic_text("قيمة EPV"), color="white")
                        ax.set_ylabel(reshape_arabic_text("العدد"), color="white")
                        st.pyplot(fig)
                    else:
                        st.warning(reshape_arabic_text("لا توجد بيانات EPV متاحة. تأكد من توفر ملف EPV_grid.csv."))
            else:
                st.error(reshape_arabic_text("لم يتم العثور على أحداث للمباراة."))
        else:
            st.error(reshape_arabic_text("فشل استخراج البيانات. تحقق من معرف المباراة أو حاول لاحقًا."))
