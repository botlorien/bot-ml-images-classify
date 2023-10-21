import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote
import time
import base64
from datetime import datetime
from web import Web
from datahandler import Handler

hd = Handler()

class GoogleSearch(Web):
    def __init__(self):
        super().__init__()
        self.base_url = "https://www.google.com/"
        self.keyword = ''

    def save_img(self, content, file_path):
        with open(file_path, 'wb') as f:
            f.write(content)

    def save_img_64(self, base64_string, file_path):
        header, encoded = base64_string.split(",", 1)
        binary_data = base64.b64decode(encoded)
        with open(file_path, 'wb') as f:
            f.write(binary_data)
        print(f'Sucessfully downloaded {file_path}....')

    def get_dummy(self):
        return int(time.time())

    def get_images_from_search(self, keyword):
        self.keyword = hd.clear_invalid_characters_from_list([keyword],' ')[0]
        url = self.base_url + f"search?q={keyword}&sca_esv=575463815&&tbm=isch&sclient=img"
        print(url)
        self.driver.get(url)

        script = """
        var imgs = document.getElementsByTagName('img')
        var list_img_src = []
        for (let i=0;i<imgs.length;i++){
            list_img_src.push(imgs[i].src)
        }
        console.log(list_img_src)
        return list_img_src
        """
        sources = self.driver.execute_script(script)
        sources = [src for src in sources if len(src) > 0]
        img_64 = [src for src in sources if 'data:image' in src]

        self.create_folder('google_images')
        for i, src in enumerate(img_64):
            self.save_img_64(src, f'google_images/img_{self.keyword}_{i}_{self.get_dummy()}.png')


if __name__ == '__main__':
    go = GoogleSearch()
    go.search('receipts images')
