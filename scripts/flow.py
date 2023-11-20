from scripts import tasks as ts
from botlorien_sources.interface import ui
from botlorien_sources.datahandler import Handler
from botlorien_sources.app import BotApp

hd = Handler()

app = BotApp()

app.set_bot(
    bot_name='Bot_ml_images_classify',
    bot_description='This bot download google receipts and classify by machine learning',
    bot_version='2.0.0',
    bot_department='Machine Learning'
)


@app.task
def download_images_from_google():
    """Downloads imagens from google images"""
    ts.download_images_from_google(_main_ui)


@app.task
def classify_customer():
    """Prompts a user interface to classify the images"""
    ts.classify_customer_receipts(_main_ui)


@app.task
def store_classification():
    """Store the classified images on local database or csv"""
    ts.store_classification()


@app.task
def train_models():
    """Train some models with the stored classification"""
    ts.train_models()


@app.task
def test_models():
    """Test the chosen model"""
    ts.test_models(_main_ui)


@app.task
def clear_database():
    """Truncate de database"""
    ts.clear_database()


def _main_ui():
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
        download_images_from_google,
        classify_customer,
        store_classification,
        train_models,
        test_models,
        clear_database

    ]
    ui.ui(buttons_name, buttons_func)


def main_ui():
    while True:
        _main_ui()


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


def main_ui_free():
    while True:
        _main_ui_free()


def main():
    pass


if __name__ == '__main__':
    print(__path__)
