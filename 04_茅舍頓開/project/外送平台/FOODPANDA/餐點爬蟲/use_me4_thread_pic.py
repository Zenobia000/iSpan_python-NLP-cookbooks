import os
import time
import csv
import requests
import pandas as pd
import random
import pickle
from fake_useragent import UserAgent
# 加入 concurrent.futures 內建函式庫
from concurrent.futures import ThreadPoolExecutor   

# 讀取網址的列表
df = pd.read_csv('./a_4.csv')

# 取第一列的數據
vendor_ids = df.iloc[:, 0]  

#所有店家數量
total_vendors = len(vendor_ids)

#總共爬的了幾個餐點
total_dishes = 0

#每次爬重時間點
product_crawl_times = []

#開始爬蟲時間
start_time = time.time()

#店家餐點數量
past_vendor_product_counts = []

# 創建存儲圖片的文件夾
if not os.path.exists('./food_img'):
    os.makedirs('./food_img')
    print("建立food_imp目錄，儲存餐點圖片")

# 創建存儲錯誤的文件夾
if not os.path.exists('./error'):
    os.makedirs('./error')
    print("建立error目錄，儲存錯誤店家")

# 創建存備份數據
if not os.path.exists('./backup'):
    os.makedirs('./backup')
    print("建立backup目錄，儲存備份數據")

# 創建存儲已爬取 vendor_id 的文件夾
if not os.path.exists('./crawled_vendors'):
    os.makedirs('./crawled_vendors')
    print("建立vendors目錄，儲存結果")
    
#計算時間用
past_crawl_times = []

# 讀取已爬取的 vendor_id
crawled_vendors = set()
if os.path.exists('./crawled_vendors/crawled_ids.txt'):
    with open('./crawled_vendors/crawled_ids.txt', 'r') as f:
        crawled_vendors = set(f.read().splitlines())

# 用於存儲所有餐點信息的列表
all_dishes_info = []

# 編輯下載函式
def download(url):    
    #print(url)
    # 下載圖片
    img_response = requests.get(url[0], headers=my_headers)
    #檔名為店家code+餐點id
    img_filename = url[1]
    #儲存位置
    img_filepath = f"./food_img/{img_filename}"    
    #二進位寫入
    with open(img_filepath, 'wb') as img_file:
        img_file.write(img_response.content)
        img_file.close()


#比對已爬過店家
for vendor_index, vendor_id in enumerate(vendor_ids):
    if vendor_id in crawled_vendors:
        print(f"跳過已爬店家: {vendor_id}")
        continue



    try:
        #隨機headers
        ua = UserAgent()
        global my_headers
        my_headers = {'user-agent': ua.random}

        #目標網址
        url = f"https://tw.fd-api.com/api/v5/vendors/{vendor_id}?include=menus"

        # 向伺服器發送 GET 請求
        response = requests.get(url, headers=my_headers)

        #將Result用json放入data
        data = response.json()

        #取得JSON中DATA的值
        vendor_data = data.get('data', {})

        #店家地址
        address = vendor_data.get('address', None)

        #經度
        latitude = vendor_data.get('latitude', None)

        #緯度
        longitude = vendor_data.get('longitude', None)

        #餐點類別
        cuisines = ",".join([cuisine['name'] for cuisine in vendor_data.get('cuisines', [])])

        #店家餐點數量
        total_products = sum(len(category.get('products', [])) for menu in vendor_data.get('menus', []) for category in menu.get('menu_categories', []))

        #初始化目前店家已爬餐點變數
        crawled_products = 0

        #圖片網址串列
        global img_urls
        img_urls = []    
        img_num = 0      
        #遍歷data dict中的menus_key(所有菜單)
        for menu in vendor_data.get('menus', []):

            #遍歷menus_key中的menu_categories key(菜單類型)
            for category in menu.get('menu_categories', []):
                
               
                #遍歷menu_categories中的所有餐點
                for product in category.get('products', []):
                    #紀錄時間
                    start_time_product = time.time()
                    
                    # 檢查是否有價格，沒有就跳下一個餐點
                    product_variations = product.get('product_variations', None)
                    price = product_variations[0].get('price', None) if product_variations else None
                    if price is None:
                        continue
                    
                    # 檢查是否有圖片網址，沒有就跳下一個餐點
                    file_path = product.get('file_path', None)
                    if not file_path:
                        continue  # 如果沒有圖片，跳過此餐點

                    # 檢查餐點名稱
                    name = product.get('name', None)

                    # 檢查餐點id
                    product_id = product.get('id', None)

                    # 檢查餐點描述
                    description = product.get('description', None)

                    #--------圖片下載--------#
                    
                    #檔名為店家code+餐點id
                    img_filename = f"{vendor_id}_{product_id}.jpg"
                    
                    # 圖片網址清單
                    img_urls.append([file_path,img_filename])
                    

                    # # 下載圖片
                    # img_response = requests.get(file_path, headers=my_headers)


                    # #儲存位置
                    # img_filepath = f"./food_img/{img_filename}"

                    # #二進位寫入
                    # with open(img_filepath, 'wb') as img_file:
                    #     img_file.write(img_response.content)
                    
                    # 每爬一張圖休息2-5秒
                    #time.sleep(random.uniform(2, 5))
                    #--------圖片下載--------#

                    # 將信息添加到列表中（包括圖片文件名）
                    all_dishes_info.append([img_filename, name, product_id, description, price, address, latitude, longitude, cuisines, vendor_id])
                    
                    #總爬過的餐點+1
                    total_dishes += 1

                    #目前店家爬過的餐點+1
                    crawled_products += 1



        executor = ThreadPoolExecutor()          # 建立非同步的多執行緒的啟動器
        with ThreadPoolExecutor() as executor:
            executor.map(download, img_urls)     # 同時下載圖片  

        # 每爬一家餐聽需要的時間
        elapsed_time_product = time.time() - start_time_product

        #每家餐聽耗費時間加入list
        product_crawl_times.append(elapsed_time_product)

        #平均每爬一個餐點所需時間
        average_crawl_time = sum(product_crawl_times) / total_dishes

        #每間餐廳有幾個餐點
        past_vendor_product_counts.append(crawled_products)

        #店家平均餐點數量
        average_products_per_vendor = sum(past_vendor_product_counts) / len(past_vendor_product_counts)
        
        #計算剩餘要爬取的店家數量
        estimated_remaining_vendors = total_vendors - (vendor_index + 1)

        #估計剩餘要爬取的餐點數量
        estimated_remaining_dishes = estimated_remaining_vendors * average_products_per_vendor

        #估計剩餘爬取時間
        estimated_time = average_crawl_time * estimated_remaining_dishes

        #將剩餘秒數加至目前時間
        endtime = time.time() + estimated_time

        #將估計的時間改成 時 分 秒
        hours, remainder = divmod(estimated_time, 3600)
        minutes, seconds = divmod(remainder, 60)
                    
        #CMD Output
        print(f"已爬: {vendor_index+1}/所有店家:{total_vendors}, "
            f"此店家已爬 :{crawled_products}/店家總餐點:{total_products}, "
            f"已爬了: {total_dishes}個餐點, "
            f"預估剩餘餐點數量: {int(estimated_remaining_dishes)}"
            f"預計剩餘時間: {int(hours)}時{int(minutes)}分{int(seconds)}秒,"
            f"預計完成時間: {time.strftime('%Y/%m/%d %H:%M:%S',time.localtime(endtime))}")

        # 數據持久化：每爬取一家就保存一次數據
        with open(f'./backup/dishes_info_backup_{vendor_id}.pkl', 'wb') as backup_file:
            pickle.dump(all_dishes_info, backup_file)

        # 將餐點信息寫入 CSV 文件
        with open('dishes_info.csv', 'a', newline='', encoding='utf-8-sig') as csvfile:
            csvwriter = csv.writer(csvfile)
            for dish_info in all_dishes_info:
                csvwriter.writerow(dish_info)

        # 清空列表
        all_dishes_info.clear()
        # 每次爬完一家店休息20-50秒隨機
        #time.sleep(random.uniform(20, 50))

        # 將已爬取的 vendor_id 寫入文件
        with open('./crawled_vendors/crawled_ids.txt', 'a') as f:
            f.write(f"{vendor_id}\n")

    except Exception as e:
        # 碰到錯誤時將錯誤的 vendor_id 寫入 ./error 目錄
        with open('./error/error_ids.txt', 'a') as error_file:
            error_file.write(f"{vendor_id}\n")
        print(f"Error with vendor_id {vendor_id}: {e}")

# 確保所有剩餘的數據也被寫入 CSV 文件
if all_dishes_info:
    with open('dishes_info.csv', 'a', newline='', encoding='utf-8-sig') as csvfile:
        csvwriter = csv.writer(csvfile)
        for dish_info in all_dishes_info:
            csvwriter.writerow(dish_info)

alltime= time.time() - start_time
hours, remainder = divmod(alltime, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"總時間: {int(hours)}時{int(minutes)}分{int(seconds)}秒,"
      f"完成時間{time.strftime('%Y/%m/%d %H:%M:%S',time.localtime(time.time()))}")