import os
import time
from datahandler import Handler
import requests
import pandas as pd
from database import Postgresql
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle
import logging
from lazypredict.Supervised import LazyClassifier, LazyRegressor
import threading as td
from functools import partial

hd = Handler()
db = Postgresql()


class PathNotExists(Exception):
    """
    A custom exception class to indicate that a specified path does not exist.
    """
    pass


def truncate_table():
    name_table = 'bot_ml_receipts_classify'
    sql = f"""
    TRUNCATE TABLE {name_table};
    """
    tabela = db.execute_script(sql)


def get_table_train_from_db():
    name_table = 'bot_ml_receipts_classify'
    sql = f"""
    SELECT * FROM {name_table}
    """
    tabela = db.execute_script(sql)
    colunas = db.select_columns(name_table, False).replace('"', '')

    tabela = pd.DataFrame(tabela)
    print(tabela)
    if len(tabela) > 0:
        tabela.columns = colunas.split(' ,')
        print(tabela)

        df_imgs_data = tabela.loc[:, 'imgs'].str.split(';', expand=True).astype('float')
        df_imgs_target = tabela.loc[:, 'class'].astype('float')
        df_imgs_name_img = tabela.loc[:, 'labels'].astype('str')

        return df_imgs_data, df_imgs_target, df_imgs_name_img


def compare_classification_in_db_with_folder():
    df_imgs_data, df_imgs_target, df_imgs_name_img = get_table_train_from_db()
    hd.delete_files_folder('verification')
    for i, _class in enumerate(df_imgs_target):
        img = df_imgs_name_img[i]
        path_ = hd.create_folder(f'verification/class_{_class}') + f'/{img}'
        hd.move_file(f'google_images/{img}', path_)


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
    result = get_table_train_from_db()
    if result is not None and len(os.listdir('google_images')) > 0:
        try:
            df_imgs_data, df_imgs_target, _ = result
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


def _get_lists_opened_imgs(folder, path_imgs, qtde_imgs, img_list, names_img_list):
    """
    This function processes an image: resizes it, creates a thumbnail, and updates lists with image data and image names.

    :param folder: The folder where the image is located.
    :param path_imgs: The path of the image file.
    :param qtde_imgs: A numpy array that keeps track of the number of images processed.
    :param img_list: A numpy array that stores data of processed images.
    :param names_img_list: A numpy array that stores paths of processed images.

    :return: Updated img_list, names_img_list, and qtde_imgs.
    """
    img = path_imgs.split('/')[-1]
    path_imgs = path_imgs if len(folder) == 0 else f'{folder}/{path_imgs}'
    folder = folder if len(folder) > 0 else '/'.join(path_imgs.split('/')[:-1])
    print(path_imgs)
    try:
        img_obj = Image.open(f'{path_imgs}')
        new_size = (200, 200)
        img_obj = img_obj.resize(new_size)
        img_obj.thumbnail((150, 150), Image.LANCZOS)

        hd.create_folder(f'{folder}/redimensionados')
        img_obj.save(f'{folder}/redimensionados/{img}')
        img_cv = cv2.imread(f'{folder}/redimensionados/{img}', cv2.IMREAD_UNCHANGED)[:, :, 0]
        qtde_imgs = np.append(qtde_imgs, 1)

        dim_img = img_cv.shape[0] * img_cv.shape[1]
        img_cv = img_cv.reshape(dim_img)
        img_list = np.append(img_list, [img_cv])
        names_img_list = np.append(names_img_list, os.path.join(os.getcwd(), path_imgs))
    except Exception as e:
        logging.exception(e)
    return img_list, names_img_list, qtde_imgs


def _load_imgs():
    img_list = np.array([])
    class_list = np.array([])
    names_img_list = np.array([])
    qtde_imgs = np.array([])
    dim_img = 150 * 150
    folder = 'google_images'
    if os.path.exists(folder):
        for p, img in enumerate(os.listdir(folder)):
            if '.' not in img:
                continue
            img_list, names_img_list, qtde_imgs = _get_lists_opened_imgs(folder, img, qtde_imgs, img_list,
                                                                         names_img_list)
        img_list_reshaped = img_list.reshape(int(qtde_imgs.sum()), dim_img)

        return img_list_reshaped, names_img_list
    else:
        raise PathNotExists


def _load_imgs_classified(dict_classification):
    img_list = np.array([])
    class_list = []
    names_img_list = np.array([])
    qtde_imgs = np.array([])
    dim_img = 150 * 150
    folder = ''
    for i, class_ in enumerate(dict_classification.keys()):
        for p, img in enumerate(dict_classification[class_]):
            print(class_)
            print(img)
            if '.' not in img:
                continue
            img_list, names_img_list, qtde_imgs = _get_lists_opened_imgs(folder, img, qtde_imgs, img_list,
                                                                         names_img_list)
            print(img_list)
            print(names_img_list)
            print(qtde_imgs)
            class_list.append(i)

    img_list_reshaped = img_list.reshape(int(qtde_imgs.sum()), dim_img)

    return img_list_reshaped, names_img_list, class_list


def order_list_target(list_labels, dict_classification):
    """
    This function orders the list of labels based on the classification dictionary.

    :param list_labels: A list of labels (image file paths).
    :param dict_classification: A dictionary containing classification information.
    :return: A list of target values based on the classification dictionary.
    """
    list_target = []
    for label in list_labels:
        label = os.path.abspath(label)
        for i, class_ in enumerate(dict_classification.keys()):
            label_class = [label_c for label_c in dict_classification[class_] if os.path.abspath(label_c) in label]
            if len(label_class) > 0:
                list_target.append(i)
                break
    print(list_target)
    print(list_labels)
    return list_target


def generate_table_classification_to_store(dict_classification):
    list_data, list_labels, list_target = _load_imgs_classified(dict_classification)
    print(list_labels)
    print(list_target)
    list_labels = [label.split('/')[-1] for label in list_labels]
    df_imgs_data = pd.DataFrame(list_data)
    df_imgs_target = pd.DataFrame(list_target)
    df_imgs_name_img = pd.DataFrame(list_labels)

    df_imgs_data.columns = [str(coluna) for coluna in df_imgs_data.columns]
    df_imgs_target.columns = [str(coluna) for coluna in df_imgs_target.columns]
    df_imgs_name_img.columns = [str(coluna) for coluna in df_imgs_name_img.columns]

    hd.create_folder('dados')

    df_imgs_data.to_csv('dados/df_imgs_data.csv', sep=';')
    df_imgs_target.to_csv('dados/df_imgs_target.csv', sep=';')
    df_imgs_name_img.to_csv('dados/df_imgs_name_img.csv', sep=';')

    df_data_to_db = pd.DataFrame()
    df_imgs_data = df_imgs_data.astype('str')
    df_imgs_target = df_imgs_target.astype('str')
    df_imgs_name_img = df_imgs_name_img.astype('str')

    df_data_to_db['imgs'] = [str(';'.join(lista)) for lista in df_imgs_data.values.tolist()]
    df_data_to_db['class'] = [str(';'.join(lista)) for lista in df_imgs_target.values.tolist()]
    df_data_to_db['labels'] = [str(';'.join(lista)) for lista in df_imgs_name_img.values.tolist()]

    return df_data_to_db.astype('str')


def max_len_val_table(table):
    max_ = 0
    for col in table.columns:
        max_val = max([len(str(val)) for val in table[col].tolist()])
        if max_val > max_:
            max_ = max_val
    return max_


def print_list_as_table(list_vals, max_len):
    list_vals = [hd.add_left_space(val, max_len) for val in list_vals]
    print('|'.join(list_vals))


def print_score():
    model_score = hd.import_file('MODELS SCORES\score.csv')
    max_len = max_len_val_table(model_score)
    print_list_as_table(model_score.columns, max_len)
    for row in model_score.itertuples(index=False):
        print_list_as_table(row, max_len)


def test_model(model: str = ''):
    hd.delete_files_folder('relatorios auditoria')
    if len(os.listdir('google_images')) > 0:
        img_list_reshaped, names_img_list = _load_imgs()

        if len(model) <= 0:
            model_score = hd.import_file('MODELS SCORES\score.csv')
            print(model_score)
            model = model_score.loc[0, 'Model'] + '.sav'
        print(model)

        # load the saved model
        with open(f'models/{model}', 'rb') as file_:
            loaded_model = pickle.load(file_)

        # make a prediction using the loaded model
        news = img_list_reshaped / 16.0
        df_imgs_data_test = pd.DataFrame(news)
        df_imgs_data_test.columns = [str(coluna) for coluna in df_imgs_data_test.columns]
        result = loaded_model.predict(df_imgs_data_test)
        print(result)

        label_result = ['CORRETO' if int(res) == 0 else 'ERRADO' for res in result]
        print(label_result)

        imgs_correct = [names_img_list[i] for i, res in enumerate(result) if int(res) == 0]
        imgs_incorrect = [names_img_list[i] for i, res in enumerate(result) if int(res) == 1]

        df_result2 = pd.DataFrame()

        df_result2['imgs'] = names_img_list
        df_result2['result'] = label_result

        hd.to_csv(df_result2, 'Relacao imgs auditadas', 'relatorios auditoria')

        return imgs_correct, imgs_incorrect


def order_imgs_result(imgs_correct, imgs_incorrect):
    """
    Organizes images based on the prediction results (correct or incorrect) into separate folders.

    :param customer: The customer's name as a string.
    :param imgs_correct: A list of paths to correctly classified images.
    :param imgs_incorrect: A list of paths to incorrectly classified images.
    :return: None
    """
    hd.create_folder('classified imgs')
    hd.delete_files_folder('classified imgs/')
    path_correct = hd.create_folder('classified imgs/correct')
    path_incorrect = hd.create_folder('classified imgs/incorrect')
    for img in imgs_correct:
        name_img = img.split('/')[-1]
        hd.move_file(img, f'{path_correct}/{name_img}')
    for img in imgs_incorrect:
        name_img = img.split('/')[-1]
        hd.move_file(img, f'{path_incorrect}/{name_img}')


def generate_info_to_whats():
    path_corrects = 'classified imgs/correct'
    path_incorrects = 'classified imgsincorrect'
    if os.path.exists(path_corrects) and os.path.exists(path_incorrects):
        corretos = len(os.listdir(path_corrects))
        errados = len(os.listdir(path_incorrects))

        sct = f"""📍 *Comprovantes Auditados* 📍
            *Corretos: {corretos}*
            *Errados: {errados}*
            """
        sct = sct.replace('\t', '').replace('  ', '')
        return sct


def create_folders_nce(path):
    try:
        if not os.path.exists(path):
            os.mkdir(path)
        return path
    except Exception as e:
        logging.exception(e)


if __name__ == '__main__':
    print_score()
