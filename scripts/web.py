import logging
import re
import shutil
import time
import os
import zipfile

from cryptography.fernet import Fernet
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.alert import Alert
from bs4 import BeautifulSoup
from urllib.parse import urlencode, unquote
import pandas as pd
import requests
import asyncio
from pyppeteer import launch


class EmptyValue(Exception):
    pass


class NotFound(Exception):
    pass


class Web:

    def __init__(self):
        self.method = None
        self.driver = None
        self.action = None
        self.session = requests.Session()
        self.cookies = None
        self.token = None
        self.path_to_downloads = 'downloads'

    class TimeoutException(Exception):
        pass

    def time_out(void=None, time_out: int = 20, raise_exception: bool = True):
        """
        Decorator that makes a function repeat its execution until a given timeout limit
        is reached, if necessary. If the function runs without raising exceptions before
        the timeout, it is not repeated.

        :param time_out: Time limit until the function is stopped, defaults to 20
        :type time_out: int, optional
        :param raise_exception: Whether to raise an exception after timeout, defaults to False
        :type raise_exception: bool, optional
        """

        def wrapper(func):
            def inner_wrapper(*args, **kwargs):
                contador_time_out = 0
                ret = False
                error = None
                while contador_time_out < time_out:
                    try:
                        ret = func(*args, **kwargs)
                        break
                    except Exception as e:
                        error = e
                        logging.exception(error)
                        time.sleep(1)
                    contador_time_out += 1

                    if contador_time_out >= time_out and raise_exception:
                        raise error
                return ret

            return inner_wrapper

        return wrapper

    def create_folder(self, path):
        """
        Creates a folder at a given path if it does not exist already.

        :param path: The path where the folder should be created
        :type path: str
        :return: The absolute path of the created folder
        :rtype: str
        """

        if ':' not in path:
            root_path = os.getcwd()
            path = os.path.abspath(root_path + f'/{path}')

        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def create_file_txt(self, text: str = '', name_file: str = '', path_to_save: str = '', subs: bool = False):
        """
        Creates a .txt file at a specified location with provided text. If the file already exists,
        the method can either overwrite it or read its content based on the 'subs' argument.
        The method first ensures that the path to save the file exists, creating it if it doesn't.

        :param text: The text to write into the file. Default is an empty string.
        :type text: str

        :param name_file: The name of the file to be created. Default is an empty string.
        :type name_file: str

        :param path_to_save: The directory where the file should be saved. Default is the current working directory.
        :type path_to_save: str

        :param subs: If True, existing file will be overwritten. If False and file exists, its content will be read. Default is False.
        :type subs: bool

        :return: The text written to the file or read from it.
        :rtype: str

        :Example:

            self._create_file_txt("Hello, world!", "test", "/path/to/save", True) # creates/overwrites a file named "test.txt" with the text "Hello, world!" at "/path/to/save"

        """
        path_to_save = self.create_folder(path_to_save)
        full_path_file = f'{path_to_save}\{name_file}.txt'

        if not os.path.exists(full_path_file) or subs:
            with open(full_path_file, 'w') as f:
                f.write(text)
        else:
            with open(full_path_file, 'r') as f:
                text = f.read()
        return text

    @staticmethod
    def move_file(from_path_file, to_path_file):
        """
        Move a file from the source location to the destination location.

        :param from_path_file: Source file path.
        :type from_path_file: str
        :param to_path_file: Destination file path.
        :type to_path_file: str
        """
        shutil.copy2(from_path_file, to_path_file)

    def wait_download(self, timeout: int = 3600, ext_optional: str = '.txt'):
        # tuple of commom extentions in the project to download
        use_to_ext = ('.xls', '.csv', '.sswweb', '.json', '.pdf', ext_optional)

        # loop over path download until file be found
        contador_while = 0
        while contador_while < timeout:

            # getting the files extentions
            ext_files_list = [str(file_name).split('.')[-1].lower() for file_name in os.listdir(self.path_to_downloads)]
            found = [True for ext in ext_files_list if ext in use_to_ext]
            if True in found:
                break
            contador_while += 1
            time.sleep(1)

    def define_profile_path(self, name_instance):

        appdata_path = os.getenv('APPDATA')
        appdata_dir = os.path.dirname(appdata_path)
        path_store_profile = os.path.abspath(appdata_dir + '/Local/profile_chrome')
        _ = self.create_folder(path_store_profile)

        path_store_profile = os.path.abspath(appdata_dir + f'/Local/profile_chrome/{name_instance}')
        path_store_profile = self.create_folder(path_store_profile)

        profile_path = os.path.abspath(path_store_profile + '/Default')
        profile_path = self.create_file_txt(profile_path, 'profile_path_browser', 'parameters')
        return profile_path

    def init_browser(self, name_instance: str = 'Default', ignore_profile: bool = True, path_to_downloads='downloads',
                     headless: bool = True, driver: str = 'chrome', method: str = 'pypperteer'):
        """
        Initializes the web browser.

        :param driver:
        :param headless:
        :param name_instance: Name of the browser instance, defaults to 'Default'
        :type name_instance: str, optional
        :param ignore_profile: Whether to ignore the profile, defaults to True
        :type ignore_profile: bool, optional
        :param path_to_downloads: Path to the downloads folder, defaults to 'downloads'
        :type path_to_downloads: str, optional
        """
        self.path_to_downloads = self.create_folder(path_to_downloads)
        if not ignore_profile:
            profile_path = self.define_profile_path(name_instance)
        else:
            profile_path = ''
        self.method = method
        self.method = self.create_file_txt(self.method, 'config_method_browser', 'config')
        driver = self.create_file_txt(driver, 'config_driver_browser', 'config')

        if 'selenium' == self.method:
            if driver == 'chrome':
                self.chrome_driver(path_to_downloads, profile_path, ignore_profile, headless)
                self.action = ActionChains(self.driver)
            elif driver == 'edge':
                self.edge_driver(path_to_downloads, profile_path,ignore_profile, headless)
                self.action = ActionChains(self.driver)

        elif 'pypperteer' == self.method:
            asyncio.get_event_loop().run_until_complete(
                self.pypperteer_browser(headless, name_instance, ignore_profile))

    def chromedriver_install(self):
        try:
            ZIP = 'chromedriver.zip'
            resp = requests.get(
                'https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json')
            chromedrivers = resp.json()['channels']['Stable']['downloads']['chromedriver']
            url_win64 = [chromedriver["url"] for chromedriver in chromedrivers if "win64" in chromedriver["platform"]][
                0]
            resp = requests.get(url_win64)
            with open(ZIP, 'wb') as f:
                f.write(resp.content)
            self.create_folder('chromedriver')
            # Open the zip file in read mode
            with zipfile.ZipFile(ZIP, 'r') as zip_ref:
                # Extract all the files
                zip_ref.extractall('chromedriver')
            os.remove(ZIP)
        except PermissionError as e:
            pass
        return os.path.abspath('chromedriver/chromedriver-win64/chromedriver.exe')

    def chrome_driver(self, path_to_downloads, profile_path, ignore_profile, headless):
        options = Options()
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-extensions")

        options.add_experimental_option("prefs", {
            "download.default_directory": path_to_downloads,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "plugins.plugins_list": [{"enabled": False, "name": "Chrome PDF Viewer"}],
            "download.extensions_to_open": "applications/pdf",
            'plugins.always_open_pdf_externally': True
        })
        # Trying to use ChromeDriverManager to get the latest driver
        path_chrome_driver = self.chromedriver_install()

        if len(profile_path) > 0 and not ignore_profile:
            options.add_argument(f"user-data-dir={profile_path}")

        if headless:
            options.add_argument('headless')

        self.driver = webdriver.Chrome(executable_path=path_chrome_driver, options=options)

    def edge_driver(self, path_to_downloads, profile_path, ignore_profile, headless):
        from msedge.selenium_tools import Edge, EdgeOptions
        from webdriver_manager.microsoft import EdgeChromiumDriverManager

        options = EdgeOptions()
        options.use_chromium = True
        if len(profile_path) > 0 and not ignore_profile:
            options.add_argument(f"--user-data-dir={profile_path}")

        if headless:
            # options.add_argument('--headless')
            options.headless = True

        path_to = os.path.abspath('edgedriver/msedgedriver.exe')
        try:
            path_default = EdgeChromiumDriverManager().install()
            self.create_folder('edgedriver')
            self.move_file(path_default, path_to)
        except PermissionError as e:
            pass

        options.add_experimental_option("prefs", {"download.default_directory": f"{path_to_downloads}"})
        self.driver = Edge(executable_path=path_to, options=options)

    async def pypperteer_browser(self, headless, instance, ignore_profile):
        if not ignore_profile:
            self.create_folder('profile_pypperteer')
            path_ = self.create_folder(f'profile_pypperteer/{instance}')
        else:
            path_ = None
        self.driver = await launch(headless=headless,
                                   executablePath=r'C:\Program Files\Google\Chrome\Application\chrome.exe',
                                   userDataDir=path_,
                                   defaultViewport={'width': 1000, 'height': 900}
                                   )

    async def _login_with_pypperteer(self, url, script, time_wait):

        self.page = await self.driver.newPage()

        await self.page.goto(url)

        ret = await self.page.evaluate(f'''() => {{
                {script};
            }}''')

        # await page.click("a[id='5']")
        # await page.close()

        await asyncio.sleep(time_wait)

        # Get cookies from Pyppeteer
        cookies_list = await self.page.cookies()

        # Close the browser
        # await browser.close()

        for cookie in cookies_list:
            self.session.cookies.set(cookie['name'], cookie['value'])

        self.cookies = cookies_list

        return ret

    def login_puppyteer(self, url, script, time_wait: int = 3):
        return asyncio.get_event_loop().run_until_complete(self._login_with_pypperteer(url, script, time_wait))

    async def _get_cookies_pypperteer(self):

        # Get cookies from Pyppeteer
        cookies_list = await self.page.cookies()
        if len(cookies_list) > 0:
            for cookie in cookies_list:
                self.session.cookies.set(cookie['name'], cookie['value'])

            self.cookies = cookies_list
        else:
            raise EmptyValue

    @time_out()
    def get_cookies_pypperteer(self):
        return asyncio.get_event_loop().run_until_complete(self._get_cookies_pypperteer())

    async def _get_html_page_pypperteer(self):

        # Get cookies from Pyppeteer
        content = await self.page.content()
        return content

    @time_out()
    def get_html_page_pypperteer(self):
        return asyncio.get_event_loop().run_until_complete(self._get_html_page_pypperteer())

    async def _execute_script_pypperteer(self, script, time_wait):

        ret = await self.page.evaluate(f'''() => {{
                {script};
            }}''')

        # await page.click("a[id='5']")
        # await page.close()

        await asyncio.sleep(time_wait)

        return ret

    def execute_script_pypperteer(self, script, time_wait: int = 3):
        return asyncio.get_event_loop().run_until_complete(self._execute_script_pypperteer(script, time_wait))

    async def _send_keys_pypperteer(self, element, value_to_send):
        await self.page.focus(element)
        await self.page.type(element, value_to_send)

    def send_keys_pypperteer(self, element, value_to_send):
        return asyncio.get_event_loop().run_until_complete(self._send_keys_pypperteer(element, value_to_send))

    @time_out()
    def get_cookies(self):
        """
        Retrieves cookies from the current session of the browser and stores them in the `session` property of the class.
        The retrieval of cookies is managed by the `time_out` decorator to prevent hanging in case of failure.

        :Example:

            self._get_cookies()

        :raises: EmptyValue if no cookies are retrieved from the current session.
                 TimeoutException if unable to retrieve cookies after the specified timeout period.
        """
        self.cookies = self.driver.get_cookies()

        if len(self.cookies) == 0:
            raise EmptyValue

        for cookie in self.cookies:
            self.session.cookies.set(cookie['name'], cookie['value'])

    @time_out()
    def get_bearer_authorization_from_storage(self):
        """
            Retrieves the bearer authorization token from the SacFlow application.

            This method retrieves the bearer authorization token stored in the local storage of the browser. It executes a JavaScript script to retrieve the token and assigns it to the `self.token` attribute. If the token is `None`, it raises a `NotFound` exception.

            :raises NotFound: If the bearer authorization token is not found.
            :return: The bearer authorization token.
            :rtype: str
        """
        script = """
        var token = localStorage.getItem('token');
        console.log(token);
        return token
        """
        self.token = self.driver.execute_script(script)
        print(self.token)
        if self.token is None:
            raise NotFound

        self.token = f"Bearer {self.token}"
        return self.token

    def convert_query_url_to_dict(self, query, empty_values: bool = True):
        """
        Converts a query URL into a dictionary.

        :param query: The query URL to convert
        :type query: str
        :param empty_values: Whether to empty the values, defaults to True
        :type empty_values: bool, optional
        :return: The dictionary obtained from the query URL
        :rtype: dict
        """
        dict_query = {}
        query_list = query.split('&')
        for param in query_list:
            param_list = param.split('=')
            if not empty_values:
                dict_query[param_list[0]] = param_list[1]
            else:
                dict_query[param_list[0]] = ''
        return dict_query

    def convert_dict_to_query_url(self, dict_query):
        """
        Converts a dictionary into a query URL.

        :param dict_query: The dictionary to convert
        :type dict_query: dict
        :return: The query URL obtained from the dictionary
        :rtype: str
        """
        query_format = []
        for key in dict_query:
            query_format.append(f'{key}={dict_query[key]}')
        return '&'.join(query_format)

    def show_kwargs_possible_values(self, html, query_model):
        """
        Displays the possible keys that can be passed to a function as keyword arguments (kwargs).
        This method takes an HTML document and a query model as arguments. It parses the HTML document
        to extract input tags, and for each key in the query model, it finds the corresponding input tag
        and its next sibling div tag in the HTML document. It then prints the name and value of each parameter
        in the console.

        :param html: The HTML document to be parsed.
        :type html: str

        :param query_model: The query model that contains the keys to be searched in the HTML document.
        :type query_model: str

        :return: None
        :rtype: None

        :Example:

            >>>self.show_kwargs_possible_values("<html>...</html>", "key1=value1&key2=value2")

        :process:
        1. The method converts the query model into a dictionary of parameters.
        2. It finds the input tag corresponding to each key in the parameters dictionary in the HTML document.
        3. If the input tag is found, it finds its next sibling div tag and extracts its text value.
        4. It then prints the name and value of the parameter.
        5. If the input tag is not found, it continues with the next key.
        """

        # Create a BeautifulSoup object
        soup = BeautifulSoup(html, 'html.parser')

        additional_params = self.convert_query_url_to_dict(query_model)
        print('"""')
        print('    :param kwargs:')
        print('    Possible keys include:')
        for key in additional_params:
            try:
                # Find the input tag
                input_tag = soup.find('input', attrs={'name': key})

                if input_tag is not None:
                    # Find the div tag
                    div_tag = input_tag.find_next_sibling('div')

                    # Extract the query parameter name and value
                    param_name = input_tag['name']
                    param_value = div_tag.text.strip().replace('&nbsp;', '').replace(' ', '')

                    # Build the query string
                    query = f'        - {param_name}: {param_value}'

                    print(query)
            except KeyError:
                pass
        print('    :type kwargs: dict')
        print('    :return: None')
        print('"""')

    def update_query_values(self, html, query_model, attribute: str = 'id', **kwargs):
        """
        Update the values in the query string of the query model.

        :param html: The HTML content of the page.
        :type html: str
        :param query_model: The query model string.
        :type query_model: str
        :param attribute: The attribute to search for in the input tags.
        :type attribute: str
        :param kwargs: Additional key-value pairs to update in the query string.
        :type kwargs: dict
        :return: The updated query string.
        :rtype: str
        """
        soup = BeautifulSoup(html, 'html.parser')

        # Find all input tags in the HTML
        inputs = soup.find_all('input')

        # Build a dictionary with the ids and values from the input tags
        params = {input_tag.get(attribute): input_tag.get('value') for input_tag in inputs}

        self.show_kwargs_possible_values(html, query_model)

        additional_params = self.convert_query_url_to_dict(query_model)
        for key in additional_params:
            try:
                additional_params[key] = params[key]
            except KeyError:
                pass
        for key in kwargs:
            additional_params[key] = kwargs[key]

        # Use urlencode to create the query string
        query = urlencode(additional_params)

        return query

    def scrap_tags_from_xml(self, html, tag_inicio, tag_final):
        """
        Scrap tags from XML content in the HTML.

        :param html: The HTML content.
        :type html: str
        :param tag_inicio: The starting tag to search for.
        :type tag_inicio: str
        :param tag_final: The ending tag to search for.
        :type tag_final: str
        :return: The list of found tags.
        :rtype: list
        """
        content = html
        tag_inicio_xml = '<xml'
        tag_final_xml = '</xml>'
        idx_inicio = content.find(tag_inicio_xml)
        idx_final = content.find(tag_final_xml) + len(tag_final_xml)
        xml_user = content[idx_inicio:idx_final]

        # Find number of rows
        rows = xml_user.split('<r>')

        founds = []
        for row in rows:
            idx_inicio = row.find(tag_inicio)
            idx_final = row.find(tag_final)

            if -1 not in (idx_inicio, idx_final):
                found = row[idx_inicio + 4:idx_final]
                founds.append(found)
        return founds

    @time_out()
    def click_id_el(self, id):

        script = f"""
        return document.getElementById('{id}').click()
        """
        self.driver.execute_script(script)

    @time_out()
    def get_el_by_id(self, id):
        script = f"""
        return document.getElementById('{id}')
        """
        return self.driver.execute_script(script)

    @time_out()
    def get_els_by_tag(self, tag):
        script = f"""
        return document.getElementsByTagName('{tag}')
        """
        return self.driver.execute_script(script)

    @time_out()
    def get_el_by_xpath(self, xpath):
        return self.driver.find_element(By.XPATH, xpath)

    def quit(self):
        self.driver.quit()
