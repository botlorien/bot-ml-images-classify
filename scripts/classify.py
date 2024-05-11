import os
import time
from scripts.interface import ui
from scripts.datahandler import Handler

hd = Handler()

max_size_height = int(hd.create_file_txt('300',
                                         'config_max_size_height',
                                         'config').strip())


class Classify:

    def __init__(self):
        self.ui = ui
        self.classified_imgs = {'correct': [], 'incorrect': [], 'ignored': []}
        self.counter = 0
        self.status_show_incorrect = False

    def _corret(self):
        """
        Method to classify the current image as correct.
        """
        if (len(self.classified_imgs['correct']) + len(self.classified_imgs['incorrect']) + len(
                self.classified_imgs['ignored'])) < len(
                self.ui.list_path_imgs):
            path_img = self.ui.list_path_imgs[self.ui.counter_img]
            print(path_img)
            file_ = os.path.basename(path_img)
            print(file_)
            self.classified_imgs['correct'].append(path_img)
            hd.move_file(path_img, os.path.abspath(f'targets/correct/{file_}'))
            self.ui.next_img()
        else:
            print('Classificação finalizada!')
            print(len(self.classified_imgs['correct']), self.classified_imgs['correct'])
            self.ui.messagebox('Todas as imagens \njá foram classificadas!', 'info')
            self.main_ui()

    def _incorret(self):
        """
        Method to classify the current image as incorrect.
        """
        if (len(self.classified_imgs['correct']) + len(self.classified_imgs['incorrect']) + len(
                self.classified_imgs['ignored'])) < len(
                self.ui.list_path_imgs):
            path_img = self.ui.list_path_imgs[self.ui.counter_img]
            file_ = os.path.basename(path_img)
            self.classified_imgs['incorrect'].append(path_img)
            hd.move_file(path_img, os.path.abspath(f'targets/incorrect/{file_}'))
            self.ui.next_img()
        else:
            print('Classificação finalizada!')
            print(len(self.classified_imgs['incorrect']), self.classified_imgs['incorrect'])
            self.ui.messagebox('Todas as imagens \njá foram classificadas!', 'info')
            self.main_ui()

    def _ignore(self):
        """
        Method to classify the current image as incorrect.
        """
        if (len(self.classified_imgs['correct']) + len(self.classified_imgs['incorrect']) + len(
                self.classified_imgs['ignored'])) < len(
                self.ui.list_path_imgs):
            path_img = self.ui.list_path_imgs[self.ui.counter_img]
            file_ = os.path.basename(path_img)
            self.classified_imgs['ignored'].append(path_img)
            hd.move_file(path_img, os.path.abspath(f'targets/ignored/{file_}'))
            self.ui.next_img()
        else:
            print('Classificação finalizada!')
            print(len(self.classified_imgs['ignored']), self.classified_imgs['ignored'])
            self.ui.messagebox('Todas as imagens \njá foram classificadas!', 'info')
            self.main_ui()

    def next(self):
        """
        Method to navigate to the next image and update the navigation status and counter accordingly.
        """
        self.ui.next_img()
        self.counter += 1
        if self.counter >= len(self.ui.list_path_imgs) and self.status_show_incorrect:
            self.counter = 0
            self.status_show_incorrect = False
            self.main_ui()
        elif self.counter >= len(self.ui.list_path_imgs) and not self.status_show_incorrect:
            self.status_show_incorrect = True
            self.counter = 0
            self.show_receipts('incorrect')

    def back(self):
        """
        Method to navigate to the previous image and update the navigation status and counter accordingly.
        """
        self.ui.back_img()
        self.counter += 1
        if self.counter >= len(self.ui.list_path_imgs) and self.status_show_incorrect:
            self.counter = 0
            self.status_show_incorrect = False
            self.main_ui()
        elif self.counter >= len(self.ui.list_path_imgs) and not self.status_show_incorrect:
            self.status_show_incorrect = True
            self.counter = 0
            self.show_receipts('incorrect')
            time.sleep(10)

    def classify_receipts(self):
        """
        Method to classify receipts by showing images and providing classification options.
        """
        path_ = 'google_images'
        hd.create_folder('targets')
        hd.delete_files_folder('targets/correct')
        hd.delete_files_folder('targets/incorrect')
        hd.delete_files_folder('targets/ignored')
        if os.path.exists(path_) and len(os.listdir(path_)) > 0:
            list_name_img_buttons = ['CERTO', 'ERRADO', 'IGNORAR']
            list_func_img_buttons = [self._corret, self._incorret, self._ignore]
            self.ui.ui_show_imgs("Imagem",
                                 path_,
                                 list_name_img_buttons,
                                 list_func_img_buttons,
                                 width=None,
                                 height=None,
                                 max_size=max_size_height)
        else:
            self.ui.messagebox('Sem Imagens')
            time.sleep(10)

    def show_receipts(self, folder: str = 'correct'):
        """
        Method to show classified receipts from the specified folder.

        :param folder: The folder to show receipts from, defaults to 'correct'.
        """
        path_ = f'classified imgs/{folder}'
        if os.path.exists(path_) and len(os.listdir(path_)) > 0:
            list_name_img_buttons = ['Avançar', 'Voltar']
            list_func_img_buttons = [self.next, self.back]
            self.ui.ui_show_imgs(folder,
                                 path_,
                                 list_name_img_buttons,
                                 list_func_img_buttons,
                                 width=None,
                                 height=None,
                                 max_size=max_size_height)
        else:
            # self.ui.messagebox('Sem Imagens Corretas')
            if 'correct' == folder:
                self.show_receipts('incorrect')

    def clear_classification(self):
        """
        Clears the current classification data.
        """
        self.classified_imgs = {'correct': [], 'incorrect': [], 'ignored': []}

    def get_classification(self):
        """
        Retrieves the current classification data.

        :returns: A dictionary containing the current classification data.
        """
        return self.classified_imgs

    def set_main_ui(self, main_ui):
        """
        Sets the main UI for the Classify class.

        :param main_ui: The main UI to be set.
        """
        self.main_ui = main_ui


if __name__ == '__main__':
    cl = Classify()
    cl.classify_receipts()
