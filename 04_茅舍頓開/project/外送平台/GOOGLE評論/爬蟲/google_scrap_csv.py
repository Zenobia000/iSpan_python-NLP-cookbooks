import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import csv
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

# 創建抓貼文資訊的資料夾
comment_save_dir = './comment'
if not os.path.exists(comment_save_dir):
    print(f"保存數據的目錄 {comment_save_dir} 不存在，已創建 {comment_save_dir}")
    os.makedirs(comment_save_dir)
else:
    print("post_info 目錄已存在，將開始爬蟲")

# 讀取 CSV 文件中的店家名稱
df = pd.read_csv('./a_1.csv')
store_names = df['名稱'].iloc[0:4000]  # 假設店家名稱在「名稱」列

# 啟動瀏覽器
my_options = webdriver.ChromeOptions()
my_options.add_argument("--start-maximized")
my_options.add_argument("--incognito")
my_options.add_argument("--disable-popup-blocking")
my_options.add_argument("--disable-notifications")
my_options.add_argument("--lang=zh-TW")
driver = webdriver.Chrome(options=my_options)

# 打開 Google 地圖
driver.get("https://www.google.com/maps")

# 等待頁面加載
time.sleep(5)
google_reviews = []
google_stars = []

total_stores = len(store_names)
s_stores = 0
total_reviews = 0
csv_file_path = os.path.join(comment_save_dir, "店家評論資訊.csv")

#此處紀錄以爬取過的店家，放在txt檔
completed_keywords_file = '已爬暫存區.txt'

completed_keywords = set()
if os.path.exists(completed_keywords_file):
    with open(completed_keywords_file, 'r') as file:
        for line in file:
            completed_keywords.add(line.strip())


# 定義每次搜索的店家數
search_batch_size = 20

for batch_start in range(0, total_stores, search_batch_size):
    batch_end = min(batch_start + search_batch_size, total_stores)
    batch_store_names = store_names[batch_start:batch_end]

    for store_name in batch_store_names:
        if store_name in completed_keywords:
            print(f"關鍵字 {store_name} 已經爬取過，跳過")
            continue
        else:
            try:
                # 找到搜尋欄並輸入店家名稱
                search_box = driver.find_element(By.ID, "searchboxinput")
                search_box.clear()
                search_box.send_keys(store_name)
                time.sleep(2)  # 等待建議出現

                # 檢查第一個建議是否是 "在 Google 地圖中加入遺漏的地點"
                first_suggestion_text_element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div#cell0x0 span.cGyruf.fontBodyMedium.RYaupb > span"))
                )
                first_suggestion_text = first_suggestion_text_element.text
                if "在 Google 地圖中加入遺漏的地點" in first_suggestion_text:
                    s_stores += 1
                    print(f"跳過 {store_name}，因為沒有找到相關建議，已爬取 {s_stores}/{total_stores} 家店!")
                    continue

                # 點擊第一個建議
                first_suggestion = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "div#cell0x0 span.cGyruf.fontBodyMedium.RYaupb > span"))
                )
                first_suggestion.click()
                time.sleep(2)  # 等待頁面加載

                # 點擊第二個按鈕（索引為 1）以打開評論
                review_buttons = driver.find_elements(By.CSS_SELECTOR, "button.hh2c6")
                if len(review_buttons) > 1:
                    review_buttons[1].click()
                    time.sleep(1)

                # 滾動加載留言
                no_change_counter = 0
                prev_reviews = 0
                scrollable_element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "/html/body/div[3]/div[8]/div[9]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]")))
                actions = ActionChains(driver)
                while True:
                    # 找到當前的所有留言和星數
                    reviews = driver.find_elements(By.XPATH, "//span[@class='wiI7pd']")
                    stars = driver.find_elements(By.XPATH, "//span[@class='kvMYJc']")

                    if len(reviews) >= 400 or no_change_counter >= 7:
                        break  # 如果找到了200個留言或留言數不再增加，則停止滾動

                    if len(reviews) == prev_reviews:
                        no_change_counter += 1
                    else:
                        no_change_counter = 0  # 如果留言數增加，則重置計數器
                    prev_reviews = len(reviews)

                    # 滾爆
                    for _ in range(10):
                        scrollable_element.send_keys(Keys.PAGE_DOWN)
                        time.sleep(0.07)

                while True:
                    buttons = driver.find_elements(By.CSS_SELECTOR, "button.w8nwRe.kyuRq")
                    if not buttons:
                        break  # 如果沒有展開更多按鈕了，則退出

                    for button in buttons:
                        try:
                            button.click()
                        except Exception as e:
                            print(f"Could not click on button: {str(e)}")

                    time.sleep(1)  # 等待展開更多按鈕的內容加載

                # 等待一段時間，以便頁面加載完畢
                time.sleep(2)
                # 提取和打印符合條件的留言和星數
                for i, (review_element, star_element) in enumerate(zip(reviews, stars)):
                    review_text = review_element.text.strip()
                    star_text = star_element.get_attribute("aria-label").strip()

                    if len(review_text) > 5:
                        # 獲取CSV文件中的Code和店名
                        code = df.loc[df['名稱'] == store_name, 'Code'].values[0]
                        store_name = df.loc[df['名稱'] == store_name, '名稱'].values[0]

                        # 寫入CSV文件
                        with open(csv_file_path, 'a', newline='', encoding='utf-8-sig') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([code, store_name, review_text, star_text])

                s_stores += 1
                total_reviews += len(reviews)
                print(f"成功爬取店家 {store_name}，已爬取 {s_stores}/{total_stores} 家店，總共爬取 {total_reviews} 則留言數")

                completed_keywords.add(store_name)
                with open(completed_keywords_file, 'a') as file:
                    file.write(store_name + '\n')

            except Exception as e:
                s_stores += 1
                print(f"跳過 {store_name}，他不是正確的店家位置，已爬取 {s_stores}/{total_stores} 家店!")

    # 關閉瀏覽器
    driver.quit()

    # 重新啟動 Chrome
    driver = webdriver.Chrome(options=my_options)
    # 打開 Google 地圖
    driver.get("https://www.google.com/maps")
    time.sleep(5)

# 關閉瀏覽器
driver.quit()
