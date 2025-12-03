import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from .track_crop import crop
from .IR import IR
import sqlite3
import sqlite3
import unicodedata  # 한글 자소 분리 현상 해결용
from PIL import Image
import requests
from io import BytesIO

def listup(img_path):
    urldict = {}
    filenamedict ={}

    crop(img_path)
    output_result = IR()

    conn = sqlite3.connect('modules/IR/DB/ikea_image_data_multi_category2_deleted.db')
    cursor = conn.cursor()


    for i in range(len(output_result)):

        raw_name = str(output_result[i]['predicted_name'])


        # 1단계: 앞뒤 공백, 줄바꿈 제거
        target_name = raw_name.strip()

        # 2단계: 유니코드 정규화 (NFC: 한글 자음+모음을 하나로 합침)
        target_name = unicodedata.normalize('NFC', target_name)
        # ---------------------------------------------------------
        # 3. 쿼리 실행
        cursor.execute("SELECT image_url FROM products_images WHERE filename = ?", (target_name,))
        result = cursor.fetchone()

        if result:
            print("Found:", result[0])
            url = result[0]
            response = requests.get(url)
            # img = Image.open(BytesIO(response.content))
            # img.show()
            urldict[output_result[i]['folder_id']]=[url, raw_name]

        else:
            print("No result found in DB.")
       
    return urldict
