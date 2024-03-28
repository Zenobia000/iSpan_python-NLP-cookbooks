from fake_useragent import UserAgent
import requests
from bs4 import BeautifulSoup as bs
import random
import time
import os
import csv

#檢查要放資料的目錄否存在
save_dir = './res_data'
if not os.path.exists(save_dir):
    print(f"放檔案的 {save_dir} 目錄不在，已建立{save_dir}")
    os.makedirs(save_dir)
else:
    print("res_data目錄已存在，將開始爬蟲")

# 初始化一個空列表來存儲所有餐廳資料
all_restaurants = []
# 初始頁數
page = 1

#主要爬蟲迴圈，直到not res_info
while True:
    try:
        # 目標網址
        url = f"https://www.foodpanda.com.tw/city/changhua?page={page}"
        #  取得餐廳名，寫入csv時使用

        # 隨機取得user-Agent
        ua = UserAgent()
        my_headers = {'user-agent': ua.random}

        # 發送HTTP請求
        res = requests.get(url, headers=my_headers)
        res.encoding = 'utf-8'

        # 使用lxml解析HTML
        soup = bs(res.text, "lxml")

        # 使用CSS選擇器找到特定的元素
        res_info = soup.find_all('figcaption', {'class': 'vendor-info'})
        if not res_info:
                break
        
            #遍歷每一個figcaption
        for res in res_info:
            # 在每個figcaption4中找到'店名'
            res_name = res.find('span', {'class': 'name fn'})
            
            # 在figcaption的父元素（a標籤）中找到'餐廳網址'
            res_url = res.find_parent('a')['href']

            # 在每個figcaption元素中取得'店家評分'
            res_rating = res.find('span', {'class': 'ratings-component'})
            
            # 在每個figcaption元素中取得'餐點類型'
            res_type = res.find('li', {'class': 'vendor-characteristic'})  

            # 檢查是否找到了各個元素 
            res_name_text = res_name.text if res_name else "未知"
            res_url_text = res_url if res_url else "未知"
            res_rating_text = res_rating.get('aria-label', '未知') if res_rating else "未知"
            res_type_text = res_type.text if res_type else "未知"

            #個別店家資料加入至LIST
            restaurant = {
            '名稱': res_name_text,
            '網址': res_url_text,
            '評分': res_rating_text,
            '餐點類型': res_type_text
            }
            all_restaurants.append(restaurant)
            
        #提取完畢換下一頁，休息1-20秒
        print(f"已完成第{page}頁的爬蟲。")
        page += 1
        time.sleep(random.uniform(30, 120))
    except Ellipsis as e:
        print(f"爬蟲出現錯誤: {e}")
        break

#城市名稱擷取

print("爬蟲已完成")
print("正在寫入檔案...")

# 寫入CSV文件
try:
    city_name = url.split("/")[-1].split("?")[0]
    csv_file_path = os.path.join(save_dir, f"{city_name}.csv")
    with open(csv_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['名稱', '網址', '評分', '餐點類型']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for restaurant in all_restaurants:
            writer.writerow(restaurant)
    print(f"{city_name}地區CSV儲存成功，腳本已結束")

except Exception as e:
    print(f"儲存{city_name}的CSV時發生錯誤: {e}")  
#備用