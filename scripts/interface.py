import logging
import time
import sys
from io import StringIO
from datetime import datetime
import customtkinter
from PIL import Image
from tkinter import filedialog
from tkcalendar import *
import string
import os
from cryptography.fernet import Fernet
import scripts.globals as gb
from functools import partial
import threading as td

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

class EmptyValue(Exception):
    pass


class Restarting(Exception):
    pass


class Cript:
    """
    A class used for the encryption and decryption of data, which uses Fernet symmetric encryption.

    :Example:

        >>> cript = Cript()
        >>> encrypted_data = cript.cripto_data("some sensitive data")
        >>> original_data = cript.decripto_data(encrypted_data)

    """

    def __init__(self):
        """
        Constructor method.
        Initializes the Cript object with None as the key.
        """
        self.key = None

    @staticmethod
    def create_folder(path):
        """
        Creates a folder at the specified path.

        :param path: Path to the folder to be created.
        :type path: str
        :return: Absolute path to the folder.
        :rtype: str
        """
        if ':' not in path:
            root_path = os.getcwd()
            path = os.path.abspath(root_path + f'/{path}')

        if not os.path.exists(path):
            os.mkdir(path)
        return path

    @staticmethod
    def create_path_to_save():
        """
        Creates a path for storing the key file.

        :return: Path to the key file.
        :rtype: str
        """
        appdata_path = os.getenv('APPDATA')
        appdata_dir = os.path.dirname(appdata_path)
        path_store_key = os.path.abspath(appdata_dir + '/Local/cvl')
        return path_store_key

    def generate_secret_key(self):
        """
        Generates a Fernet key and saves it into a file.

        :return: None
        """
        key = Fernet.generate_key()
        path_store_key = self.create_path_to_save()
        self.create_folder(path_store_key)
        if not os.path.exists(f"{path_store_key}/h.b"):
            with open(f"{path_store_key}/h.b", "wb") as key_file:
                key_file.write(key)

    def load_key(self):
        """
        Loads the Fernet key from a file.

        :return: None
        """
        path_store_key = self.create_path_to_save()
        self.create_folder(path_store_key)
        with open(f"{path_store_key}/h.b", "rb") as key_file:
            self.key = key_file.read()

    def salting(self, dados):
        """
        Salts the input data by interspersing it with ASCII characters.

        :param dados: Data to be salted.
        :type dados: str
        :return: Salted data.
        :rtype: str
        """
        salted = ''
        salt = string.ascii_uppercase + string.ascii_lowercase
        for i, c in enumerate(dados[::-1]):
            index = salt.find(c)
            salted += c + salt[i] if index == -1 else salt[index - 7] + salt[i]
        return salted

    def d_salt(self, salted):
        """
        Removes the salt from the input data.

        :param salted: Salted data.
        :type salted: str
        :return: Original unsalted data.
        :rtype: str
        """
        original_dados = ''
        salt = string.ascii_uppercase + string.ascii_lowercase
        for i, c in enumerate(salted[::-1]):
            index = salt.find(c)
            index_char_alt_range = salt[-7:].find(c)
            d_salted = salt[index + 7] if index_char_alt_range == -1 else salt[index_char_alt_range]
            if i % 2 != 0:
                original_dados += c if index == -1 else d_salted
        return original_dados

    def cripto_data(self, dados):
        """
        Encrypts and salts the input data.

        :param dados: Data to be encrypted.
        :type dados: str
        :return: Encrypted data.
        :rtype: str
        """
        self.generate_secret_key()
        if self.key is None:
            self.load_key()

        salted = self.salting(dados)
        salted = salted.encode()
        cipher_suite = Fernet(self.key)
        cipher_text = cipher_suite.encrypt(salted)
        dado_text = str(cipher_text.decode())
        return dado_text

    def decripto_data(self, dado_text):
        """
        Decrypts the input data and removes the salt.

        :param dado_text: Data to be decrypted.
        :type dado_text: str
        :return: Decrypted data.
        :rtype: str
        """
        if self.key is None:
            self.load_key()
        cipher_suite = Fernet(self.key)
        plain_text = cipher_suite.decrypt(dado_text.encode())
        salted = str(plain_text.decode())
        original_dado = self.d_salt(salted)
        return original_dado


customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"


class Interface:
    """
    A class that provides an interface for entering and storing encrypted credentials.

    :Example:

        >>> interface = Interface()
        >>> interface.ask_credentials()
        >>> credentials = interface.load_credentials()

    """

    def __init__(self, appname: str = 'BotLórien',user:str=''):
        """
        Constructor method.
        Initializes the SswInterface object with several tkinter widgets and an instance of the SswCript class.
        """
        super().__init__()

        self.calendar_window = None
        gb.init_root().title(appname)
        gb.init_root().geometry("+100+100")
        self.user = user
        self.appname = appname
        self.add_user_image = None
        self.chat_image = None
        self.home_image = None
        self.logo_image = self._load_image(os.path.join(PARENT_DIR,'assets/logo_bot.png'))
        self.home_image = self._load_image(os.path.join(PARENT_DIR,'assets/home.png'), 30, 30)
        self.config_image = self._load_image(os.path.join(PARENT_DIR,'assets/configuracoes.png'), 30, 30)
        self.options_image = self._load_image(os.path.join(PARENT_DIR,'assets/options.png'), 30, 30)
        self.greeting_image = self._load_image(os.path.join(PARENT_DIR,'assets/robo_hi.png'), 150, 150)
        self.last_button_state = False
        self.scaling_optionemenu = None
        self.scaling_label = None
        self.appearance_mode_optionemenu = None
        self.appearance_mode_label = None
        self.frame_3_button = None
        self.frame_2_button = None
        self.navigation_frame_label = None
        self.navigation_frame = None
        self.home_button = None
        self.entry_login = None
        self.label_login = None
        self.button_login = None
        self.credentials = None
        self.messagebox_window = None
        self.label_image = None
        self.label_messabox = None
        self.button_messagebox = None
        self._textbox_value = None
        self.last_terminal_value = None
        self.terminal_value = None
        self.textbox_config = None
        self.img_frame = None
        self.counter_img = 0
        self.name_bot = self.create_file_txt('','config_bot_name','config')

        self.path_credentials = os.path.join(os.getcwd(), f'parameters/{self.appname}_credentials.bot')

        self.cript = Cript()

    @staticmethod
    def init_root():
        global ROOT
        try:
            customtkinter.CTkFrame(ROOT)
        except Exception as e:
            ROOT = customtkinter.CTk()
            return ROOT
        else:
            return ROOT

    @staticmethod
    def _create_folder(path):
        """
        Creates a folder at the specified path if it does not already exist.

        :param path: Path to the folder to be created.
        :type path: str
        :return: Absolute path to the folder.
        :rtype: str
        """
        if ':' not in path:
            root_path = os.getcwd()
            path = os.path.abspath(root_path + f'/{path}')

        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def _ui_navigation_frame(self):
        self.navigation_frame = customtkinter.CTkFrame(gb.init_root(), width=140, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)

    def _ui_home_frame(self):
        self.home_frame = customtkinter.CTkFrame(gb.init_root(), corner_radius=0, fg_color="transparent")
        self.home_frame.grid_rowconfigure((0, 1), weight=1)
        self.home_frame.grid_columnconfigure(0, weight=1)

    def _ui_greeting_frame(self):
        self.greeting_frame = customtkinter.CTkFrame(gb.init_root(), corner_radius=0, fg_color="transparent")
        self.greeting_frame.grid_columnconfigure(0, weight=1)
        self.greeting_frame.grid_rowconfigure((0, 1), weight=1)

    def _ui_img_frame(self):
        self.img_frame = customtkinter.CTkFrame(gb.init_root(), corner_radius=0, fg_color="transparent")
        self.img_frame.grid_rowconfigure((0, 1), weight=1)
        self.img_frame.grid_columnconfigure(0, weight=1)
        self.img_menu_frame = customtkinter.CTkFrame(gb.init_root(), corner_radius=0, fg_color="transparent")
        self.img_menu_frame.grid_rowconfigure((0, 1), weight=1)
        self.img_menu_frame.grid_columnconfigure(0, weight=1)
        # self.img_frame.grid_rowconfigure((0,1,2), weight=1)
        #self.img_frame.grid_rowconfigure(3, weight=1)

    def _ui_options_frame(self):
        self.options_frame = customtkinter.CTkFrame(gb.init_root(), corner_radius=0, fg_color="transparent")
        self.options_frame.grid_columnconfigure(0, weight=1)

    def _ui_config_frame(self):
        self.config_frame = customtkinter.CTkFrame(gb.init_root(), corner_radius=0, fg_color="transparent")
        self.config_frame.grid_columnconfigure(0, weight=1)

    def _ui_navigation_label(self):
        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text=f"  {self.appname}",
                                                             image=self.logo_image,
                                                             compound="left",
                                                             font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

    def _ui_home_label(self):
        text = self.get_day_greetings()
        self.select_frame_by_name("greeting")
        self.home_frame_label_text = customtkinter.CTkLabel(self.greeting_frame, text=text,
                                                            font=customtkinter.CTkFont(size=60, weight="bold"))
        self.home_frame_label_text.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.home_frame_label_img = customtkinter.CTkLabel(self.greeting_frame,
                                                           text='',
                                                           image=self.greeting_image)
        self.home_frame_label_img.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")

        gb.init_root().after(3000, lambda: self.select_frame_by_name("homepage"))

    def _ui_img_label(self,text, folder_imgs,**kwargs):
        self.select_frame_by_name("images")
        width = kwargs.pop('width') if 'width' in kwargs else 150
        height = kwargs.pop('height') if 'height' in kwargs else 150
        self.list_imgs = []
        self.list_path_imgs = []
        for img in os.listdir(folder_imgs):
            if img.split('.')[-1].lower() not in ('png','jpg','jpeg'):
                continue
            self.list_path_imgs.append(f'{folder_imgs}/{img}')
            self.list_imgs.append(self._load_image(f'{folder_imgs}/{img}', width, height,**kwargs))
        self.current_img = self.list_imgs[0]
        self.img_frame_label_text = customtkinter.CTkLabel(self.img_frame, text=text,
                                                            font=customtkinter.CTkFont(size=60, weight="bold"))
        self.img_frame_label_text.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.img_frame_label_img = customtkinter.CTkLabel(self.img_frame,
                                                           text='',
                                                           image=self.current_img)
        self.img_frame_label_img.grid(row=1, column=0, padx=20, pady=20, sticky="ew")
        self.img_frame_label_img.grid_rowconfigure(1, weight=1)

    def next_img(self):
        self.counter_img +=1
        print(self.counter_img)
        self.counter_img = self.counter_img if abs(self.counter_img) < len(self.list_imgs) else 0
        self.current_img = self.list_imgs[self.counter_img]
        gb.init_root().after(10, lambda: self.img_frame_label_img.configure(image=self.current_img))

    def back_img(self):
        self.counter_img -=1
        print(self.counter_img)
        self.counter_img = self.counter_img if abs(self.counter_img) < len(self.list_imgs) else 0
        self.current_img = self.list_imgs[self.counter_img]
        gb.init_root().after(10, lambda: self.img_frame_label_img.configure(image=self.current_img))

    def _ui_options_label(self):
        self.options_frame_label = customtkinter.CTkLabel(self.options_frame, text=f"  Opções\n{self.name_bot}",
                                                          image=self.options_image,
                                                          compound="left",
                                                          font=customtkinter.CTkFont(size=30, weight="bold"))
        self.options_frame_label.grid(row=0, column=0, padx=20, pady=20)

    def _ui_config_label(self):
        self.config_frame_label = customtkinter.CTkLabel(self.config_frame, text=f"  Configurações\n{self.name_bot}",
                                                         image=self.config_image,
                                                         compound="left",
                                                         font=customtkinter.CTkFont(size=30, weight="bold"))
        self.config_frame_label.grid(row=0, column=0, padx=20, pady=20)

    def _ui_navigation_buttons(self):
        self.home_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=50,
                                                   border_spacing=10,
                                                   text="Home",
                                                   fg_color="transparent", text_color=("gray10", "gray90"),
                                                   hover_color=("gray70", "gray30"),
                                                   image=self.home_image,
                                                   compound="left",
                                                   command=self.home_button_event)
        self.home_button.grid(row=1, column=0, sticky="ew", padx=5, pady=20)

        self.options_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=50,
                                                      border_spacing=10, text="Options",
                                                      fg_color="transparent", text_color=("gray10", "gray90"),
                                                      hover_color=("gray70", "gray30"),
                                                      image=self.options_image,
                                                      compound="left",
                                                      command=self.options_button_event)
        self.options_button.grid(row=2, column=0, sticky="ew", padx=5, pady=20)

        self.config_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=50,
                                                     border_spacing=10, text="Config",
                                                     fg_color="transparent", text_color=("gray10", "gray90"),
                                                     hover_color=("gray70", "gray30"),
                                                     image=self.config_image,
                                                     compound="left",
                                                     command=self.config_button_event)
        self.config_button.grid(row=3, column=0, sticky="ew", padx=5, pady=20)

    def __ui_navigation_options_menu(self):
        self.appearance_mode_label = customtkinter.CTkLabel(self.navigation_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.navigation_frame,
                                                                       values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))

        self.scaling_label = customtkinter.CTkLabel(self.navigation_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.navigation_frame,
                                                               values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

    def _ui_config_options_menu(self):
        self.label_config_options_menu1 = customtkinter.CTkLabel(self.config_frame,
                                                                 text='Parameters',
                                                                 font=customtkinter.CTkFont(size=20, weight="bold"),
                                                                 padx=30,
                                                                 anchor="w")
        self.label_config_options_menu1.grid(row=1, column=0, sticky="nsew")
        self.scrollable_label_config_frame1 = ScrollableLabelConfigFrame(master=self.config_frame,
                                                                         width=600,
                                                                         fg_color="transparent",
                                                                         command_edit=self._config_parameters_button_options_menu_edit_event,
                                                                         command_remove=self._config_parameters_button_options_menu_remove_event,
                                                                         corner_radius=0,
                                                                         )
        self.scrollable_label_config_frame1.grid(row=2, column=0, padx=0, pady=0, sticky="nsew")

        self.label_config_options_menu2 = customtkinter.CTkLabel(self.config_frame,
                                                                 text='Config',
                                                                 font=customtkinter.CTkFont(size=20, weight="bold"),
                                                                 padx=30,
                                                                 anchor="w")
        self.label_config_options_menu2.grid(row=3, column=0, sticky="nsew")
        self.scrollable_label_config_frame2 = ScrollableLabelConfigFrame(master=self.config_frame,
                                                                         width=600,
                                                                         fg_color="transparent",
                                                                         command_edit=self._config_config_button_options_menu_edit_event,
                                                                         command_remove=self._config_config_button_options_menu_remove_event,
                                                                         corner_radius=0,
                                                                         )
        self.scrollable_label_config_frame2.grid(row=4, column=0, padx=0, pady=0, sticky="nsew")
        self._create_folder('parameters')
        for file in os.listdir('parameters'):  # add items with images
            self.scrollable_label_config_frame1.add_item(file, image=None, ignore_extention_file='.bot')
        self._create_folder('config')
        for file in os.listdir('config'):  # add items with images
            self.scrollable_label_config_frame2.add_item(file, image=None, ignore_extention_file='.csv')

    def _ui_home_options_menu_textbox(self):
        self.home_options_textbox_frame = customtkinter.CTkFrame(self.home_frame,
                                                                 corner_radius=0,
                                                                 fg_color="transparent"
                                                                 )
        self.home_options_textbox_frame.grid(row=1, column=0, sticky="nsew")
        self.home_options_textbox_frame.grid_columnconfigure(0, weight=1)
        self.home_options_textbox_frame.grid_rowconfigure(0, weight=1)
        self.textbox_home_options = customtkinter.CTkTextbox(self.home_options_textbox_frame)
        self.textbox_home_options.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")

    def update_home_options_menu_textbox_value(self):
        output_redirector = OutputRedirector(self.textbox_home_options)
        sys.stdout = output_redirector

    def _ui_home_options_menu(self, *args, **kwargs):
        list_name_home_buttons = kwargs.pop('list_name_home_buttons') if 'list_name_home_buttons' in kwargs else None
        list_func_home_buttons = kwargs.pop('list_func_home_buttons') if 'list_func_home_buttons' in kwargs else None

        self._ui_home_options_menu_textbox()

        frame_buttons = customtkinter.CTkScrollableFrame if len(list_name_home_buttons) > 2 else customtkinter.CTkFrame
        self.home_options_buttons_frame = frame_buttons(self.home_frame,
                                                        corner_radius=0,
                                                        fg_color="transparent",
                                                        width=500)
        self.home_options_buttons_frame.grid(row=0, column=0, sticky="nsew")
        self.home_options_buttons_frame.columnconfigure(0, weight=1)

        self.label_home_options_menu = customtkinter.CTkLabel(self.home_options_buttons_frame,
                                                              text=f'Home\n{self.name_bot}',
                                                              image=self.home_image,
                                                              compound='left',
                                                              font=customtkinter.CTkFont(size=30, weight="bold"),
                                                              padx=30)
        self.label_home_options_menu.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

        if len(list_func_home_buttons) != len(list_name_home_buttons):
            list_func_home_buttons = [partial(print, button) for button in range(len(list_name_home_buttons))]

        self.home_options_buttons = list(range(len(list_name_home_buttons)))
        self.bot_image = self._load_image(os.path.join(PARENT_DIR,'assets/robot.png'))  # bot_walk
        for i, param in enumerate(list_name_home_buttons):
            last_row = i + 2
            self.home_options_buttons[i] = customtkinter.CTkButton(self.home_options_buttons_frame,
                                                                   text=param,
                                                                   image=self.bot_image,
                                                                   command=partial(self.home_options_menu_event,
                                                                                   list_func_home_buttons[i], i),
                                                                   font=customtkinter.CTkFont(size=15, weight="bold"),
                                                                   corner_radius=10,
                                                                   height=60 if len(
                                                                       list_name_home_buttons) > 2 else 100,
                                                                   width=500)
            self.home_options_buttons[i].grid(row=last_row, column=0, padx=(20, 20), pady=(20, 20), sticky="ew")
            self.home_options_buttons_frame.rowconfigure(last_row, weight=1)

    def _ui_img_options_menu(self, *args, **kwargs):
        list_name_img_buttons = kwargs.pop('list_name_img_buttons') if 'list_name_img_buttons' in kwargs else None
        list_func_img_buttons = kwargs.pop('list_func_img_buttons') if 'list_func_img_buttons' in kwargs else None

        frame_buttons = customtkinter.CTkScrollableFrame if len(list_name_img_buttons) > 2 else customtkinter.CTkFrame
        self.img_options_buttons_frame = frame_buttons(self.img_menu_frame,
                                                        corner_radius=0,
                                                        fg_color="transparent",
                                                        width=500)
        self.img_options_buttons_frame.grid(row=3, column=0, sticky="nsew")
        self.img_options_buttons_frame.columnconfigure(0, weight=1)
        #self.img_options_buttons_frame.rowconfigure(2, weight=0)

        if len(list_func_img_buttons) != len(list_name_img_buttons):
            list_func_img_buttons = [partial(print, button) for button in range(len(list_name_img_buttons))]

        self.img_options_buttons = list(range(len(list_name_img_buttons)))
        self.bot_image = self._load_image(os.path.join(PARENT_DIR,'assets/robot.png'))  # bot_walk
        for i, param in enumerate(list_name_img_buttons):
            last_row = i + 3
            self.img_options_buttons[i] = customtkinter.CTkButton(self.img_options_buttons_frame,
                                                                   text=param,
                                                                   image=self.bot_image,
                                                                   command=partial(self.img_options_menu_event,
                                                                                   list_func_img_buttons[i], i),
                                                                   font=customtkinter.CTkFont(size=15, weight="bold"),
                                                                   corner_radius=10,
                                                                   height=10 if len(
                                                                       list_name_img_buttons) > 2 else 10,
                                                                   width=500)
            self.img_options_buttons[i].grid(row=last_row, column=0, padx=(10, 10), pady=(10, 10), sticky="nsew")
            # self.img_options_buttons_frame.rowconfigure(last_row, weight=1)

    def _config_home_options_menu_buttons_standard(self, pos):
        self.last_button_state = False
        gb.init_root().after(1000, lambda: self.home_options_buttons[pos].configure(fg_color=self.default_fg_color))
        gb.init_root().after(1000, lambda: self.home_options_buttons[pos].configure(hover_color=self.default_hv_color))
        gb.init_root().after(1000, lambda: self.home_options_buttons[pos].configure(state='normal'))

    def _config_img_options_menu_buttons_standard(self, pos):
        self.last_button_state = False
        gb.init_root().after(10, lambda: self.img_options_buttons[pos].configure(fg_color=self.default_fg_color))
        gb.init_root().after(10, lambda: self.img_options_buttons[pos].configure(hover_color=self.default_hv_color))
        gb.init_root().after(10, lambda: self.img_options_buttons[pos].configure(state='normal'))

    def exec_func(self, func, pos):
        try:
            func()
            self.messagebox('Processo Finalizado!', 'info')
        except Exception as e:
            logging.exception(e)
            self.messagebox(f'Erro!\n{str(e)}', 'warning')

        self._config_home_options_menu_buttons_standard(pos)

    def exec_func_imgs(self, func, pos):
        try:
            func()
            #self.messagebox('Processo Finalizado!', 'info')
        except Exception as e:
            logging.exception(e)
            #self.messagebox(f'Erro!\n{str(e)}', 'warning')

        self._config_img_options_menu_buttons_standard(pos)

    def home_options_menu_event(self, func, pos):
        if not self.last_button_state:
            self.last_button_state = True
            self.default_fg_color = self.home_options_buttons[pos].cget('fg_color')
            self.default_hv_color = self.home_options_buttons[pos].cget('hover_color')
            self.home_options_buttons[pos].configure(fg_color='red')
            self.home_options_buttons[pos].configure(hover_color='red')
            self.home_options_buttons[pos].configure(state='disable')
            self.home_options_buttons[pos].update()
            self.update_home_options_menu_textbox_value()
            self.home_thread = td.Thread(target=lambda: self.exec_func(func, pos))
            self.home_thread.start()

    def img_options_menu_event(self, func, pos):
        if not self.last_button_state:
            self.last_button_state = True
            self.default_fg_color = self.img_options_buttons[pos].cget('fg_color')
            self.default_hv_color = self.img_options_buttons[pos].cget('hover_color')
            self.img_options_buttons[pos].configure(fg_color='red')
            self.img_options_buttons[pos].configure(hover_color='red')
            self.img_options_buttons[pos].configure(state='disable')
            self.img_options_buttons[pos].update()
            self.img_thread = td.Thread(target=lambda: self.exec_func_imgs(func, pos))
            self.img_thread.start()

    def _ui_navigation(self):
        # create sidebar frame with widgets
        self._ui_navigation_frame()
        self._ui_navigation_label()
        self._ui_navigation_buttons()
        self.__ui_navigation_options_menu()

    def _ui_home(self, *args, **kwargs):
        self._ui_home_frame()
        self._ui_greeting_frame()
        self._ui_home_label()
        self._ui_home_options_menu(self, *args, **kwargs)

    def _ui_options(self):
        self._ui_options_frame()
        self._ui_options_label()

    def _ui_config(self):
        self._ui_config_frame()
        self._ui_config_label()
        self._ui_config_options_menu()

    def _ui_work_frames(self, *args, **kwargs):
        self._ui_config()
        self._ui_options()
        self._ui_home(*args, **kwargs)
        self.select_frame_by_name("greeting")

    def _ui_show_imgs(self,text,folder_imgs,list_name_img_buttons,list_func_img_buttons,**kwargs):
        self._ui_navigation()
        self._ui_home_frame()
        self._ui_options_frame()
        self._ui_config_frame()
        self._ui_greeting_frame()
        self._ui_img_frame()
        self._ui_img_label(text,folder_imgs,**kwargs)
        self._ui_img_options_menu(list_name_img_buttons=list_name_img_buttons,
                                  list_func_img_buttons=list_func_img_buttons)

    def ui(self, list_name_home_buttons, list_func_home_buttons):
        # configure grid layout (4x4)
        gb.init_root().geometry(f"{1100}x{580}+100+100")
        gb.init_root().grid_columnconfigure(1, weight=1)
        gb.init_root().grid_columnconfigure((2, 3), weight=0)
        gb.init_root().grid_rowconfigure((0, 1, 2, 3), weight=1)
        self._ui_navigation()
        self._ui_work_frames(list_name_home_buttons=list_name_home_buttons,
                             list_func_home_buttons=list_func_home_buttons)

        self.loop()

    def ui_show_imgs(self,text, folder_imgs,list_name_img_buttons,list_func_img_buttons,**kwargs):
        # configure grid layout (4x4)
        gb.init_root().geometry(f"{1100}x{580}+100+100")
        gb.init_root().grid_columnconfigure(1, weight=1)
        gb.init_root().grid_columnconfigure((2, 3), weight=0)
        gb.init_root().grid_rowconfigure((0,3), weight=1)
        self.home_options_textbox_frame.grid_forget()
        self.home_options_buttons_frame.grid_forget()
        self._ui_show_imgs(text,folder_imgs,list_name_img_buttons,list_func_img_buttons,**kwargs)
        self.loop()

    def _config_parameters_button_options_menu_remove_event(self, item):
        print(f"Removed item: {item}")
        path_item = f'parameters/{item}'
        os.remove(path_item)
        self.scrollable_label_config_frame1.remove_item(path_item)
        if '.bot' in item:
            self.restart_ui()
        gb.init_root().after(100, self._ui_config_options_menu)

    def _config_parameters_button_options_menu_edit_event(self, item):
        path_item = f'parameters/{item}'
        with open(path_item, 'r') as f:
            read_file = f.read()
        self.show_textbox('Config', item, read_file)
        if self._textbox_value is not None:
            self.create_file(path_item, self._textbox_value)
        self.close_textbox()


    def _config_config_button_options_menu_remove_event(self, item):
        path_item = f'config/{item}'
        os.remove(path_item)
        self.scrollable_label_config_frame2.remove_item(path_item)
        gb.init_root().after(100, self._ui_config_options_menu)

    def _config_config_button_options_menu_edit_event(self, item):
        path_item = f'config/{item}'
        with open(path_item, 'rb') as f:
            read_file = f.read().decode('ISO-8859-1')
        self.show_textbox('Config', item, read_file)
        print(self._textbox_value)
        if self._textbox_value is not None:
            print(self._textbox_value)
            self.create_file(path_item, self._textbox_value)
        self.close_textbox()


    def restart_ui(self):
        sys.stdout = sys.__stdout__
        gb.init_root().destroy()
        raise Restarting

    @staticmethod
    def create_file(file_path, value):
        with open(file_path, 'w') as f:
            f.write(value)

    def create_file_txt(self, text: str = '', name_file: str = '', path_to_save: str = '', subs: bool = False):
        path_to_save = self._create_folder(path_to_save)
        full_path_file = f'{path_to_save}\{name_file}.txt'

        if not os.path.exists(full_path_file) or subs:
            with open(full_path_file, 'w') as f:
                f.write(text)
        else:
            with open(full_path_file, 'r') as f:
                text = f.read()
        return text

    def show_textbox(self, title: str = 'Textbox', label: str = 'Textbox', innertext: str = ''):
        self.textbox_window = customtkinter.CTkToplevel(gb.init_root())
        self.textbox_window.title(title)
        self.textbox_window.attributes("-topmost", True)

        self.textbox_window.geometry("300x300+170+170")
        self.textbox_window.rowconfigure(2, weight=1)
        self.textbox_window.columnconfigure(0, weight=1)

        self.label_textbox = customtkinter.CTkLabel(self.textbox_window, text=label,
                                                    font=customtkinter.CTkFont(size=20, weight="bold"))
        self.label_textbox.grid(row=1, column=0, padx=20, pady=(20, 10), sticky="nsew")

        self.textbox_config = customtkinter.CTkTextbox(self.textbox_window, width=200)
        self.textbox_config.grid(row=2, column=0, padx=20, pady=20, sticky="nsew")

        self._insert_value_textbox(innertext)

        self.button_textbox = customtkinter.CTkButton(self.textbox_window, text='Ok',
                                                      command=self._get_value_textbox)
        self.button_textbox.grid(row=3, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        try:
            self.textbox_window.wait_window()
        except RuntimeError as e:
            logging.exception(e)

        return self.get_textbox_value()

        # try:
        #     self.textbox_window.mainloop()
        # except RuntimeError as e:
        #     logging.exception(e)

    def _insert_value_textbox(self, text):
        self.textbox_config.insert("0.0", text)

    def _get_value_textbox(self):
        self._textbox_value = str(self.textbox_config.get("1.0", "end")).strip()  # end-1c
        print(self._textbox_value)
        self.close_textbox()

    def get_textbox_value(self):
        return self._textbox_value

    def close_textbox(self):
        #self.textbox_window.quit()
        self.textbox_window.destroy()
        #gb.init_root().after(1000, lambda: gb.init_root().update())

    def input_file(self):
        self.file_dialog_value = filedialog.askopenfile(mode='r')  # filetypes=[('Python Files', '*.py')]
        if self.file_dialog_value:
            filepath = os.path.abspath(self.file_dialog_value.name)
            print(filepath)
            return filepath
        else:
            return None

    def select_choice(self, title: str = 'Select', label: str = 'Options', list_options:list=[]):
        self.choice_window = customtkinter.CTkToplevel(gb.init_root())
        self.choice_window.title(title)
        self.choice_window.attributes("-topmost", True)

        self.choice_window.geometry("300x300+170+170")
        self.choice_window.rowconfigure(2, weight=1)
        self.choice_window.columnconfigure(0, weight=1)

        self.choice_label = customtkinter.CTkLabel(self.choice_window,
                                                   text=label,
                                                   font=customtkinter.CTkFont(size=20, weight="bold"))
        self.choice_label.grid(row=1, column=0, padx=20, pady=(20, 10))
        self.choice_optionemenu = customtkinter.CTkOptionMenu(self.choice_window,
                                                               values=list_options,
                                                               command=self.choice_event)
        self.choice_optionemenu.grid(row=2, column=0, padx=20, pady=(10, 20))

        self.button_choice = customtkinter.CTkButton(self.choice_window, text='Ok',
                                                      command=self.close_choice)
        self.button_choice.grid(row=3, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        try:
            self.choice_window.wait_window()
        except RuntimeError as e:
            logging.exception(e)

        return self.get_choice_selected()

    def choice_event(self, option: str):
        self.choice_selected = option

    def get_choice_selected(self):
        return self.choice_selected

    def close_choice(self):
        self.choice_window.destroy()

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.home_button.configure(fg_color=("gray75", "gray25") if name == "homepage" else "transparent")
        self.options_button.configure(fg_color=("gray75", "gray25") if name == "options" else "transparent")
        self.config_button.configure(fg_color=("gray75", "gray25") if name == "config" else "transparent")

        # show selected frame
        if name == "homepage":
            self.home_frame.grid(row=0, rowspan=4, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()
        if name == "greeting":
            self.greeting_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.greeting_frame.grid_forget()
        if name == "images":
            self.img_frame.grid(row=0, column=1, sticky="nsew")
            self.img_menu_frame.grid(row=3, column=1, sticky="nsew")
        else:
            if self.img_frame is not None:
                self.img_frame.grid_forget()
                self.img_menu_frame.grid_forget()
        if name == "options":
            self.options_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.options_frame.grid_forget()
        if name == "config":
            self.config_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.config_frame.grid_forget()

    def home_button_event(self):
        if self.img_frame is None:
            self.select_frame_by_name("homepage")
        else:
            self.select_frame_by_name("images")

    def options_button_event(self):
        self._ui_options()
        self.select_frame_by_name("options")

    def config_button_event(self):
        self._ui_config()
        self.select_frame_by_name("config")

    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
        print("CTkInputDialog:", dialog.get_input())

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def open_input_dialog(self, title: str = 'Type something', text: str = "Type in a number:"):
        dialog = customtkinter.CTkInputDialog(text=text, title=title)
        return dialog.get_input()

    def get_day_greetings(self):
        current_hour = datetime.now().hour
        if current_hour < 12:
            greeting = f'Bom dia {self.user}!\nComo você está?'
        elif 12 <= current_hour < 18:
            greeting = f'Boa tarde {self.user}!\nComo você está?'
        else:
            greeting = f'Boa noite {self.user}!\nComo você está?'
        return greeting

    def _close_calendar(self):
        datai = self.calendario_inicio.get()
        dataf = self.calendario_fim.get()
        self.data_inicio = datai
        self.data_fim = dataf
        print(f'Periodo Selecionado {datai} a {dataf}')
        self.calendar_window.destroy()

    def get_calendar_value(self):
        return self.data_inicio, self.data_fim

    def get_calendar(self):
        self.calendar_window = customtkinter.CTkToplevel(gb.init_root())
        self.calendar_window.title(self.appname)
        self.calendar_window.attributes("-topmost", True)
        self.calendar_window.geometry("+100+100")

        msg = customtkinter.CTkLabel(self.calendar_window, text='Definir Período',
                                     font=customtkinter.CTkFont(size=25, weight="bold"))
        msg.grid(row=1, column=0, padx=10, pady=10, sticky='ew')

        msg = customtkinter.CTkLabel(self.calendar_window, text='Inicio do período',
                                     font=customtkinter.CTkFont(size=15, weight="bold"))
        msg.grid(row=2, column=0, padx=10, pady=10, sticky='ns')

        self.calendario_inicio = DateEntry(self.calendar_window, year=datetime.today().year, locale='pt_br', anchor="n",
                                           justify='center', height=30)
        self.calendario_inicio.grid(row=3, column=0, padx=10, pady=10,
                                    sticky='ns')

        msg = customtkinter.CTkLabel(self.calendar_window, text='Final do período',
                                     font=customtkinter.CTkFont(size=15, weight="bold"))
        msg.grid(row=4, column=0, padx=10, pady=10, sticky='ns')

        self.calendario_fim = DateEntry(self.calendar_window, year=datetime.today().year, locale='pt_br', anchor="n",
                                        justify='center', height=30)
        self.calendario_fim.grid(row=5, column=0, padx=10, pady=10, sticky='ns')

        button = customtkinter.CTkButton(self.calendar_window, text="Avançar",
                                         command=self._close_calendar, width=200)
        button.grid(row=6, column=0, padx=30, pady=(15, 15))

        self.calendar_window.grid_columnconfigure(0, weight=1)

        try:
            self.calendar_window.wait_window()
        except RuntimeError as e:
            logging.exception(e)

        return self.get_calendar_value()

    def ask_credentials(self, list_params: list = None):
        """
        Presents an interface for entering credentials.

        :return: None
        """
        # creating a interface the insertions das credentials
        self.credentials_window = customtkinter.CTkToplevel(gb.init_root())
        self.credentials_window.title(self.appname)
        self.credentials_window.attributes("-topmost", True)
        self.credentials_window.geometry("+100+100")

        self.entry_frame = customtkinter.CTkFrame(self.credentials_window)
        self.entry_frame.grid(row=0, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        image = self._load_image(os.path.join(PARENT_DIR,f'assets/{self.appname}_logo.png'), 30, 30)
        self.label_login = customtkinter.CTkLabel(self.entry_frame, text="Insira suas credenciais!",
                                                  font=customtkinter.CTkFont(size=18, weight="bold"),
                                                  image=image,
                                                  compound='left'
                                                  )
        self.label_login.grid(row=1, column=0, padx=(20, 20), pady=(20, 10))

        self.entry_login = [customtkinter.CTkEntry(self.entry_frame) for _ in range(len(list_params))]

        last_row = 2
        for i, param in enumerate(list_params):
            last_row = i + 2
            self.entry_login[i] = customtkinter.CTkEntry(self.entry_frame, placeholder_text=param, width=70,
                                                         show="*" if 'senha' in param or 'password' in param else None)
            self.entry_login[i].grid(row=last_row, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.button_login = customtkinter.CTkButton(self.entry_frame, text='Avançar', command=self._button_login_event)
        self.button_login.grid(row=last_row + 1, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        try:
            self.credentials_window.mainloop()
        except RuntimeError as e:
            logging.exception(e)

    def _button_login_event(self):
        """
        Encrypts the entered credentials and stores them if they are valid.

        :return: None
        """
        self.credentials = [self.cript.cripto_data(str(entry.get())) if len(entry.get()) > 0 else None for entry in
                            self.entry_login]
        if None in self.credentials:
            self.messagebox('Insira sua credenciais\n corretamente!')
        else:
            self._store_credentials()

        # self.hide_window()
        # self.quit()
        self.credentials_window.quit()
        self.credentials_window.destroy()

    def _store_credentials(self):
        """
        Stores the encrypted credentials in a file.

        :return: None
        """
        self._create_folder('parameters')
        self.credentials = ' ;'.join(self.credentials)
        with open(self.path_credentials, 'w') as file:
            file.write(self.credentials)

    def messagebox(self, message, type_alert: str = 'warning'):
        """
        Presents a messagebox with a given message and alert type.

        :param message: Message to be displayed.
        :type message: str
        :param type_alert: Alert type (either 'warning' or 'info').
        :type type_alert: str
        :return: None
        """
        self.messagebox_window = customtkinter.CTkToplevel(gb.init_root())
        self.messagebox_window.title(type_alert.upper())
        self.messagebox_window.attributes("-topmost", True)

        self.messagebox_window.geometry("300x300+550+240")
        self.messagebox_window.rowconfigure(1, weight=1)
        self.messagebox_window.columnconfigure(0, weight=1)

        image = self._load_image(os.path.join(PARENT_DIR,'assets/warning.png')) if type_alert == 'warning' \
            else self._load_image(os.path.join(PARENT_DIR,'assets/info.png'))
        self.label_image = customtkinter.CTkLabel(self.messagebox_window,
                                                  text='',
                                                  image=image,
                                                  compound='left')
        self.label_image.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nsew")

        self.label_messabox = customtkinter.CTkLabel(self.messagebox_window, text=message,
                                                     font=customtkinter.CTkFont(size=20, weight="bold"))
        self.label_messabox.grid(row=1, column=0, padx=20, pady=(20, 10), sticky="nsew")

        self.button_messagebox = customtkinter.CTkButton(self.messagebox_window, text='Ok',
                                                         command=self.messagebox_window.destroy)
        self.button_messagebox.grid(row=2, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        try:
            self.messagebox_window.mainloop()
        except RuntimeError as e:
            logging.exception(e)

    def _load_image(self, path, width: int = 50, height: int = 50, max_size:int=0):
        """
        Loads an image from a specified path and resizes it to a specified width and height.

        :param path_image: Path to the image file.
        :type path_image: str
        :param width: Width to resize the image to.
        :type width: int
        :param height: Height to resize the image to.
        :type height: int
        :return: Resized image.
        :rtype: customtkinter.CTkImage
        """
        # load and create background image
        #Image.open(path)
        if width is None and height is None:
            with Image.open(path) as img:
                # Get image dimensions
                width, height = img.size
                print(width, height)
                if max_size >0:
                    width, height = self._resize_img_proportional(max_size,width, height)
                    print(width, height )

        with Image.open(path) as img:
            img_obg = img.copy()
        return customtkinter.CTkImage(img_obg, size=(width, height))

    def _resize_img_proportional(self,max_size,width, height):
        # Calculate the aspect ratio
        aspect_ratio = height/width#width / height

        # Determine new dimensions
        # if width > height:
        #     new_width = max_size
        #     new_height = int(max_size / aspect_ratio)
        # else:
        new_height = max_size
        new_width = int(new_height / aspect_ratio)
        return new_width, new_height

    def load_credentials(self):
        """
        Loads the encrypted credentials from a file and decrypts them.

        :return: Decrypted credentials.
        :rtype: list of str
        """
        if os.path.exists(self.path_credentials):
            with open(self.path_credentials, 'r') as file:
                self.credentials = file.read().split(';')
            for i, param in enumerate(self.credentials):
                self.credentials[i] = self.cript.decripto_data(param)
            return self.credentials
        else:
            return None

    def hide_window(self):
        """
        Hides the main tkinter window.

        :return: None
        """
        gb.init_root().withdraw()

    def show_window(self):
        """
        Shows the main tkinter window.

        :return: None
        """
        gb.init_root().iconify()

    def deconify_window(self):
        """
        Shows the main tkinter window.

        :return: None
        """
        gb.init_root().deiconify()

    def quit(self):
        #gb.init_root().quit()
        gb.init_root().destroy()

    def loop(self):
        """
        Starts the tkinter event loop.

        :return: None
        """
        try:
            gb.init_root().mainloop()
        except RuntimeError as e:
            logging.exception(e)


class ScrollableLabelConfigFrame(customtkinter.CTkScrollableFrame):
    def __init__(self, master, command_edit=None, command_remove=None, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)

        self.command_edit = command_edit
        self.command_remove = command_remove
        self.label_list = []
        self.button_edit_list = []
        self.button_remove_list = []

    def add_item(self, item, image=None, **kwargs):
        ignore_extention_file = kwargs['ignore_extention_file'] if 'ignore_extention_file' in kwargs else ''

        label = customtkinter.CTkLabel(self, text=item, image=image, compound="left", padx=5, anchor="w")

        config_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        button_edit = customtkinter.CTkButton(config_frame, text="Editar", width=100, height=24)
        button_remove = customtkinter.CTkButton(config_frame, text="Remover", width=100, height=24)

        if self.command_edit is not None:
            button_edit.configure(command=lambda: self.command_edit(item))
        if self.command_remove is not None:
            button_remove.configure(command=lambda: self.command_remove(item))

        label.grid(row=len(self.label_list), column=0, pady=(0, 10), sticky="w")
        config_frame.grid(row=len(self.button_edit_list), column=1, sticky="nsew")
        button_edit.grid(row=len(self.button_edit_list), column=0, pady=(0, 10),
                         padx=5) if ignore_extention_file not in item else None
        button_remove.grid(row=len(self.button_remove_list), column=1, pady=(0, 10), padx=5)

        self.label_list.append(label)
        self.button_edit_list.append(button_edit)
        self.button_remove_list.append(button_remove)

    def remove_item(self, item):
        for label, button_edit, button_remove in zip(self.label_list, self.button_edit_list, self.button_remove_list):
            if item == label.cget("text"):
                label.destroy()
                button_edit.destroy()
                button_remove.destroy()
                self.label_list.remove(label)
                self.button_edit_list.remove(button_edit)
                self.button_remove_list.remove(button_remove)
                return


class OutputRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.output = StringIO()

    def write(self, message):
        message = str(message)
        self.output.write(message)
        self.text_widget.insert('end', message)
        self.text_widget.see('end')


ui = Interface()

if __name__ == '__main__':
    pass