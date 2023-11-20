import os
from botlorien_sources.datahandler import Handler
from botlorien_sources.database import Postgresql
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from imblearn.over_sampling import RandomOverSampler  # pip install -U imbalanced-learn
import pickle
import logging
from lazypredict.Supervised import LazyClassifier


hd = Handler()

img_dict = {}
qtd_img_list = []

env = {
    'dbtable': 'bot_ml_receipts_classify',
    'chunk_img': 10000,
    'dir_train_test': 'google_images',
    'store_db': False,
    'threads': 1,
}
env = hd.create_file_json(env, 'config_env', 'config')

DB_TABLE_NAME = env['dbtable']
CHUNK_IMGS = env['chunk_img']
DIR_TRAIN_TEST = env['dir_train_test']
STORE_DB = env['store_db']
THREADS = env['threads']

if STORE_DB:
    db = Postgresql()

hd.create_folder(DIR_TRAIN_TEST)


class PathNotExists(Exception):
    """
    A custom exception class to indicate that a specified path does not exist.
    """
    pass


def get_opened_img(path_img):
    with Image.open(path_img) as img_obj:
        img_obj = img_obj.resize((150, 150))
        img_array = np.array(img_obj)
        # Check if the image is grayscale
        if len(img_array.shape) == 2:
            # Image is already grayscale
            return img_array
        else:
            # Convert to grayscale
            return img_array[:, :, 0]


def get_lists_opened_imgs(path_img):
    global img_dict, qtd_img_list
    img_obj = get_opened_img(path_img)
    qtd_img_list.append(1)
    img_dict[os.path.join(os.getcwd(), path_img)] = np.array([img_obj.ravel()])


def reshape_img_list():
    list_dict = list(img_dict.values())
    img_list_reshaped = np.vstack(list_dict)
    return img_list_reshaped


def load_imgs(folder: str = DIR_TRAIN_TEST):
    global img_dict, qtd_img_list
    img_dict = {}
    qtd_img_list = []
    images = [f'{folder}/{img}' for img in os.listdir(folder) if '.' in img and os.path.exists(f'{folder}/{img}')]
    images = images[:CHUNK_IMGS] if len(images) > CHUNK_IMGS else images
    hd.thread_it(THREADS, images, get_lists_opened_imgs)
    img_list_reshaped = reshape_img_list()
    labels_img_list = list(img_dict.keys())
    return img_list_reshaped, labels_img_list


def get_model(model):
    if len(model) == 0:
        model_score = hd.import_file('models scores\score.csv')
        print(model_score)
        model = model_score.loc[0, 'Model'] + '.sav'
    print(model)
    with open(f'models/{model}', 'rb') as file_:
        loaded_model = pickle.load(file_)
    return loaded_model


def normalize(img_list_reshaped):
    img_list_reshaped = img_list_reshaped / 16.0
    return img_list_reshaped


def process_results_and_generate_report(result, labels_img_list):
    label_result = ['CORRETO' if int(res) == 0 else 'ERRADO' for res in result]
    print(label_result)
    imgs_correct = [labels_img_list[i] for i, res in enumerate(result) if int(res) == 0]
    imgs_incorrect = [labels_img_list[i] for i, res in enumerate(result) if int(res) == 1]
    df_result2 = pd.DataFrame()
    df_result2['imgs'] = labels_img_list
    df_result2['result'] = label_result
    hd.to_csv(df_result2, 'Relacao imgs auditadas', 'relatorios auditoria')
    return imgs_correct, imgs_incorrect


def store_data_imgs_with_target(folder, target, database: bool = STORE_DB):
    if os.listdir(folder):
        img_list_reshaped, labels_img_list = load_imgs(folder)
        list_target = [target for _ in labels_img_list]
        labels_img_list = [label.split('/')[-1] for label in labels_img_list]

        df_data_to_db = pd.DataFrame()
        df_data_to_db['imgs'] = [str(';'.join([str(val) for val in lista])) for lista in img_list_reshaped]
        df_data_to_db['class'] = list_target
        df_data_to_db['labels'] = labels_img_list

        print(df_data_to_db)
        if database:
            db.to_postgresql(df_data_to_db.astype('str'), DB_TABLE_NAME)
        else:
            hd.create_folder('data')
            df_data_to_db.to_csv('data/dataset.csv', sep=',', mode='a', header=None, index=False)


def truncate_table():
    if STORE_DB:
        sql = f"""
        TRUNCATE TABLE {DB_TABLE_NAME};
        """
        db.execute_script(sql)
    else:
        os.remove('data/dataset.csv')


def get_table_train():
    if STORE_DB:
        sql = f"""
        SELECT * FROM {DB_TABLE_NAME}
        """
        tabela = db.execute_script(sql)
        colunas = db.select_columns(DB_TABLE_NAME, False).replace('"', '')

        tabela = pd.DataFrame(tabela)

        tabela.columns = colunas.split(' ,')
        print(tabela)
    else:
        tabela = pd.read_csv('data/dataset.csv', sep=',', header=None)
        tabela.columns = ['imgs', 'class', 'labels']

    df_imgs_data = tabela.loc[:, 'imgs'].str.split(';', expand=True).astype('float')
    df_imgs_target = tabela.loc[:, 'class'].astype('float')
    df_imgs_name_img = tabela.loc[:, 'labels'].astype('str')

    return df_imgs_data, df_imgs_target, df_imgs_name_img


def generate_models_scores_reports(score, predictions, y_test):
    """
    This function calculates various metrics (accuracy, precision, recall, and confusion matrix)
    for each model and saves these metrics along with scores and predictions to CSV files.

    :param score: DataFrame containing the scores for each model.
    :param predictions: DataFrame containing the predictions made by each model.
    :param y_test: Array or list containing the true labels.

    :return: None
    """
    hd.delete_files_folder('models scores')
    score = pd.DataFrame(score)
    predictions = pd.DataFrame(predictions)
    more_scores = pd.DataFrame()
    for i, name_model in enumerate(predictions.columns):
        print('#' * 10 + name_model + '#' * 10)
        y_pred = predictions[name_model].tolist()
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        print('Confusion matrix', confusion)
        print('Accuracy:', accuracy)
        print('Precision:', precision)
        print('Recall:', recall)

        more_scores.loc[i, 'model'] = name_model
        more_scores.loc[i, 'Confusion matrix'] = str(confusion)
        more_scores.loc[i, 'Accuracy'] = accuracy
        more_scores.loc[i, 'Precision'] = precision
        more_scores.loc[i, 'Recall'] = recall

    score.to_csv('models scores/score.csv', sep=';')
    predictions.to_csv('models scores/predictions.csv', sep=';')
    more_scores.to_csv('models scores/more_scores.csv', sep=';')


def train_models():
    result = get_table_train()
    if result is not None:
        try:
            df_imgs_data, df_imgs_target, _ = result
            print(df_imgs_target.value_counts())
            ros = RandomOverSampler(random_state=42)
            df_imgs_data, df_imgs_target = ros.fit_resample(df_imgs_data, df_imgs_target)
            print(df_imgs_target.value_counts())
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(df_imgs_data, df_imgs_target, test_size=0.2,
                                                                random_state=42)
            # Normalize the pixel values
            X_train = X_train / 16.0
            X_test = X_test / 16.0

            # fit all models
            clf = LazyClassifier(predictions=True)
            score, predictions = clf.fit(X_train, X_test, y_train, y_test)

            print(score)
            print(predictions)

            generate_models_scores_reports(score, predictions, y_test)

            models = clf.provide_models(X_train, X_test, y_train, y_test)

            hd.create_folder('models')
            for name, model in models.items():
                filename = f'models/{name}.sav'
                pickle.dump(model, open(filename, 'wb'))
            return True
        except Exception as e:
            logging.exception(e)
    else:
        return False


def test_model(model: str = ''):
    hd.delete_files_folder('relatorios auditoria')
    if len(os.listdir(DIR_TRAIN_TEST)) > 0:
        img_list_reshaped, labels_img_list = load_imgs()
        loaded_model = get_model(model)
        data_test = normalize(img_list_reshaped)
        result = loaded_model.predict(data_test)
        print(result)
        return process_results_and_generate_report(result, labels_img_list)


def move_file(path_from, path_to):
    if os.path.exists(path_from):
        hd.move_file(path_from, path_to)
        os.remove(path_from)


def order_imgs_result(imgs_correct, imgs_incorrect):
    hd.delete_files_folder('classified imgs')
    path_correct = hd.create_folder('classified imgs/correct')
    path_incorrect = hd.create_folder('classified imgs/incorrect')
    for img in imgs_correct:
        name_img = os.path.basename(img)
        move_file(img, os.path.join(path_correct, name_img))
    for img in imgs_incorrect:
        name_img = os.path.basename(img)
        move_file(img, os.path.join(path_incorrect, name_img))


def max_len_val_table(table):
    max_ = 0
    for col in table.columns:
        max_val = max([len(str(val)) for val in table[col].tolist()])
        if max_val > max_:
            max_ = max_val
    return max_


def print_list_as_table(list_vals, max_len):
    list_vals = [hd.add_left_value(val, max_len, ' ') for val in list_vals]
    print('|'.join(list_vals))


def print_score():
    model_score = hd.import_file('MODELS SCORES\score.csv')
    max_len = max_len_val_table(model_score)
    print_list_as_table(model_score.columns, max_len)
    for row in model_score.itertuples(index=False):
        print_list_as_table(row, max_len)


if __name__ == '__main__':
    from botlorien_sources.database import teste_path
    teste_path()
