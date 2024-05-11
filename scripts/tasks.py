import time
from scripts import dataprocessing as dp
from scripts.webrequests import GoogleSearch
from scripts.interface import ui
from scripts.datahandler import Handler
import os
from scripts.classify import Classify

hd = Handler()
cl = Classify()
go = GoogleSearch()

headless = hd.create_file_json({'headless': False},
                               'config_headless_mode',
                               'config')['headless']


def download_images_from_google(main_ui):
    go.init_browser(headless=headless, method='selenium', driver='chrome')
    hd.delete_files_folder('google_images')
    cl.set_main_ui(main_ui)
    time.sleep(1)
    key_word = ui.open_input_dialog('Search', 'Digite o texto de busca')
    go.get_images_from_search(key_word)


def classify_customer_receipts(main_ui):
    """
    Initiates the receipt classification process for a selected customer.

    :param main_ui: The main UI instance to be used in the Classify class.
    """
    cl.set_main_ui(main_ui)
    cl.clear_classification()
    cl.classify_receipts()


def store_classification():
    cl.get_classification()
    dp.store_data_imgs_with_target('targets/correct', 0)
    dp.store_data_imgs_with_target('targets/incorrect', 1)


def train_models():
    dp.train_models()
    dp.print_score()


def test_models(main_ui):
    cl.set_main_ui(main_ui)
    modelos = [''] + list(os.listdir('models'))
    try:
        model = ui.select_choice('Modelos', 'Selecione um modelo', modelos)
    except Exception as e:
        model = ''
    result = dp.test_model(model)
    if result is not None:
        dp.order_imgs_result(*result)
        cl.show_receipts()


def clear_database():
    """
    Truncate the database
    """
    dp.truncate_table()


if __name__ == '__main__':
    # download_ssw_550_report_for_customer()
    # process_ssw_550_report()
    # get_imgs_from_customers()
    # train_test_models_all_customers()
    # order_receipts_to_nce_folder()
    # send_info_by_whats()
    download_all_customers_ssw_550_report()
