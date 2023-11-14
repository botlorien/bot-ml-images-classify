# import the tasks modules here!
import logging
import tasks as ts
from interface import ui
from datahandler import Handler

hd = Handler()


class Tasks:

    def __init__(self, app):
        """
        Initializes the Tasks object.

        :param app: The application object for the bot
        """
        self.app = app
        self.app.set_bot_name('Bot_nce_get_ssw_receipts_ml_classify_and_send_whats')
        self.app.set_bot_description(
            'This bot download ssw receipts classify by machine learning and send the result by whatsapp')
        self.app.set_bot_version('1.1.0')
        self.app.init_bot()

        # init here every task result as needle
        self.task1_result = None
        self.task2_result = None

    @hd.time_out(3)
    def init_credentials(self):
        """
        Initializes the credentials for the bot.

        It logs into the SSW using the credentials, sets up a browser, and then
        sets the user in the app based on the credentials.
        """
        pass

    # copy this structure for every task of the bot
    def download_images_from_google(self):
        task_name = 'download_images_from_google'
        task_description = 'download_images_from_google'
        task_function = ts.download_images_from_google

        self.app.set_task(task_name, task_description)
        self.task2_result = self.app.execute_bot_task(task_function, self.main_ui)

    def classify_customer(self):
        """
        Prompt the user to classify the receipts.
        """
        task_name = 'classify_customer'
        task_description = 'This task prompt to user classify the receipts'
        task_function = ts.classify_customer_receipts

        self.app.set_task(task_name, task_description)
        self.task1_result = self.app.execute_bot_task(task_function, self.main_ui)

    def store_classification(self):
        """
        Gets the classification and stores it on a local database.
        """
        task_name = 'store_classification'
        task_description = 'This task gets the classification and store on local database'
        task_function = ts.store_classification

        self.app.set_task(task_name, task_description)
        self.task1_result = self.app.execute_bot_task(task_function)

    def train_models(self):
        """
        Trains a model using base data and then classifies new data.
        """
        task_name = 'train_models'
        task_description = 'This task loads the base data to train a model and then classify the new data'
        task_function = ts.train_models

        self.app.set_task(task_name, task_description)
        self.task2_result = self.app.execute_bot_task(task_function)

    def test_models(self):
        """
        Trains a model using base data and then classifies new data.
        """
        task_name = 'test_models'
        task_description = 'test_models'
        task_function = ts.test_models

        self.app.set_task(task_name, task_description)
        self.task2_result = self.app.execute_bot_task(task_function, self.main_ui)

    def clear_database(self):
        """
        Trains a model using base data and then classifies new data.
        """
        task_name = 'clear_database'
        task_description = 'clear_database'
        task_function = ts.clear_database

        self.app.set_task(task_name, task_description)
        self.task2_result = self.app.execute_bot_task(task_function)

    def send_info_by_whats(self):
        """
        Send whatsapp information to responsibles.
        """
        task_name = 'send_info_by_whats'
        task_description = 'This task send a summary information to responsibles'
        task_function = ts.send_info_by_whats

        self.app.set_task(task_name, task_description)
        self.task2_result = self.app.execute_bot_task(task_function)

    def baixar_comprovantes(self):
        self.download_images_from_google()

    def main_ui(self):
        """
        Defines the main user interface with button names and their associated functions.
        """
        buttons_name = [
            'BAIXAR IMAGENS',
            'CLASSIFICAR IMAGENS',
            'GRAVAR CLASSIFICAÇÃO',
            'TREINAR MODELOS',
            'AUDITAR IMAGENS',
            'LIMPAR DADOS'
        ]
        buttons_func = [
            self.baixar_comprovantes,
            self.classify_customer,
            self.store_classification,
            self.train_models,
            self.test_models,
            self.clear_database

        ]
        ui.ui(buttons_name, buttons_func)


def _main_ui_free():
    """
    Defines the main user interface with button names and their associated functions.
    """
    buttons_name = [
        'BAIXAR IMAGENS',
        'CLASSIFICAR IMAGENS',
        'GRAVAR CLASSIFICAÇÃO',
        'TREINAR MODELOS',
        'AUDITAR IMAGENS',
        'LIMPAR DADOS'
    ]
    buttons_func = [
        lambda: ts.download_images_from_google(_main_ui_free),
        lambda: ts.classify_customer_receipts(_main_ui_free),
        lambda: ts.store_classification(),
        lambda: ts.train_models(),
        lambda: ts.test_models(_main_ui_free),
        lambda: ts.clear_database()

    ]
    ui.ui(buttons_name, buttons_func)


def main_ui(app):
    """
    Provides an interactive interface for the user to work with the bot.

    :param app: The application object used for bot interactions.
    """
    while True:
        tasks = Tasks(app)
        # tasks.init_credentials()
        tasks.main_ui()


def main_ui_free():
    while True:
        _main_ui_free()


def main(app):
    """
    Executes all tasks for all customers in a batch process.

    :param app: The application object used for bot interactions.
    """
    tasks = Tasks(app)
    tasks.init_credentials()

    # ... another task here!


if __name__ == '__main__':
    pass
