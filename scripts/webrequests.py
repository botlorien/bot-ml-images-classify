import logging
import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote
import time
import base64
from datetime import datetime
from botlorien_sources.web import Web
from botlorien_sources.datahandler import Handler

hd = Handler()

SCROLL = int(hd.create_file_txt('20', 'config_range_scroll', 'config').strip())


class GoogleSearch(Web):
    def __init__(self):
        super().__init__()
        self.increment_num = 0
        self.base_url = "https://www.google.com/"
        self.keyword = ''
        self.total = None

    def save_img(self, content, file_path, total, pos):
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f'{total}/{pos} - Sucessfully downloaded {file_path}....')

    def save_img_64(self, base64_string):
        pos = self.increment()
        file_path = f'google_images/img_{self.keyword}_{pos}_{self.get_dummy()}.png'
        _, encoded = base64_string.split(",", 1)
        binary_data = base64.b64decode(encoded)
        with open(file_path, 'wb') as f:
            f.write(binary_data)
        print(f'{self.total}/{pos} - Sucessfully downloaded {file_path}....')

    def get_dummy(self):
        return int(time.time())

    def increment(self):
        self.increment_num +=1
        return self.increment_num

    def get_link(self, src):
        try:
            pos = self.increment()
            file_path = f'google_images/img_{self.keyword}_{pos}_{self.get_dummy()}.png'
            resp = requests.get(src)
            self.save_img(resp.content, file_path, self.total, pos)
        except Exception as e:
            logging.exception(e)

    def get_images_from_search(self, keyword):
        self.increment_num = 0
        self.keyword = hd.clear_invalid_characters_from_list([keyword], ' ')[0]
        url = self.base_url + f"search?q={keyword}&sca_esv=575463815&&tbm=isch&sclient=img"
        self.driver.get(url)
        time.sleep(2)

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
        for i in range(1, SCROLL + 1):
            print(f'Scrolling to position {1000 * i}... Total images: {len(sources)}')
            self.driver.execute_script(f'window.scrollTo(0,1000*{i})')
            time.sleep(0.5)
            sources += self.driver.execute_script(script)

        sources = [src for src in sources if len(src) > 0]
        links = list(set([src for src in sources if 'https' in str(src) and not 'favicon' in str(src)]))#[:25]
        print(links)
        sources = list(set([src for src in sources if 'data:image' in src]))#[:5]

        self.total = len(sources) + len(links)
        hd.delete_files_folder('google_images')

        hd.thread_it(20, sources, self.save_img_64)
        hd.thread_it(20, links, self.get_link)

        self.driver.quit()

    def wait(self):
        input()


if __name__ == '__main__':
    go = GoogleSearch()
    go.init_browser(headless=False)
    go.get_images_from_search('Javascript in Python')
