import time
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os

import pandas as pd

from werkzeug.utils import secure_filename
from models import RandomForestMSE, GradientBoostingMSE
import re
import plotly.express as px
import numpy as np
from wtforms import StringField, SubmitField, validators
from flask_wtf import FlaskForm
import pandas as pd
from flask_wtf.file import FileField

global PREDICTOR
global TARGET_NAME
global READY_TO_SEND
PREDICTOR = None
TARGET_NAME = None
READY_TO_SEND = False


def my_validators(type):
    def integer_validate(form, field):
        if (not field.data) or not field.data.isdecimal():
            raise validators.ValidationError("Целое число введено некорректно")

    def float_validate(form, field):
        if form.model_name == 'random_forest' and field.id == 'lr':
            pass
        else:
            try:
                float(field.data)
            except:
                raise validators.ValidationError("Число введено некорректно")

    def train_file_validate(form, field):
        if (not field.data) or not re.search('^.*\.(csv)$', secure_filename(field.data.filename)):
            raise validators.ValidationError("Тренировочные данные некорректны")

    def valid_file_validate(form, field):
        if field.data and not re.search('^.*\.(csv)$', secure_filename(field.data.filename)):
            raise validators.ValidationError("Валидационные данные некорректны")

    def not_empty(form, field):
        if not field.data:
            raise validators.ValidationError("Валидационные данные некорректны")

    if type == 'int':
        return integer_validate
    if type == 'float':
        return float_validate
    if type == 'train':
        return train_file_validate
    if type == 'test':
        return valid_file_validate
    if type == 'empty':
        return not_empty


class ModelParams(FlaskForm):
    n_estimators = StringField("Количество деревьев: ", [my_validators('int')])
    lr = StringField("Темп обучения: ", [my_validators('float')])
    max_depth = StringField("Максимальная глубина дерева: ", [my_validators('int')])
    feature_subsample_size = StringField("Размер подвыборки: ", [my_validators('float')])
    train_data = FileField("Данные для обучения: ", [my_validators('train')])
    valid_data = FileField("Данные для валидации (опционально): ", [my_validators('test')])
    target_name = StringField("Название целевой колонки", [my_validators('empty')])  # validators.InputRequired()
    submit_data = SubmitField("Обучить")

    def __init__(self):
        super(ModelParams, self).__init__()
        self.model = None
        self.is_ok = True
        self.model_name = None

    def setup_and_fit(self):
        if self.model_name == 'random_forest':
            self.model = RandomForestMSE(
                self.get_n_estimators(), self.get_max_depth(),
                self.get_feature_subsample_size()
            )
        else:
            self.model = GradientBoostingMSE(
                self.get_n_estimators(), self.get_lr(),
                self.get_max_depth(), self.get_feature_subsample_size()
            )
        train_data = pd.read_csv('./src/storage/train.csv')
        if self.valid_data.data:
            val_data = pd.read_csv('./src/storage/validation.csv')
            if not self.target_name.data in train_data.columns:
                self.is_ok = False
                raise ValueError
            self.model.fit(
                train_data.drop(self.target_name.data, axis=1).values, train_data[self.target_name.data].values,
                val_data.drop(self.target_name.data, axis=1).values, val_data[self.target_name.data].values
            )
            fig1 = px.line(y=self.model.history['train_score'], x=np.arange(len(self.model.history['train_score'])),
                           labels={'x': 'Номер итерации (количество деревьев)', 'y': 'RMSE'}
                           )
            fig1.update_layout(title_text='График зависимости RMSE на тренировочной выборке', title_x=0.5)
            fig1.write_html('./src/templates/train_plt.html')
            fig1 = px.line(y=self.model.history['val_score'], x=np.arange(len(self.model.history['val_score'])),
                           labels={'x': 'Номер итерации (количество деревьев)', 'y': 'RMSE'}
                           )
            fig1.update_layout(title_text='График зависимости RMSE на валидационной выборке', title_x=0.5)
            fig1.write_html('./src/templates/val_plt.html')
        else:
            self.model.fit(
                train_data.drop(self.target_name.data, axis=1).values, train_data[self.target_name.data].values
            )
            fig1 = px.line(y=self.model.history['train_score'], x=np.arange(len(self.model.history['train_score'])),
                           labels={'x': 'Номер итерации (количество деревьев)', 'y': 'RMSE'}
                           )
            fig1.update_layout(title_text='График зависимости RMSE на тренировочной выборке', title_x=0.5)
            fig1.write_html('./src/templates/train_plt.html')
        global PREDICTOR, TARGET_NAME
        PREDICTOR = self.model
        TARGET_NAME = self.target_name.data

    def get_history(self):
        return self.model.history

    def get_n_estimators(self):
        return int(self.n_estimators.data)

    def get_lr(self):
        return float(self.lr.data)

    def get_max_depth(self):
        return float(self.max_depth.data)

    def get_feature_subsample_size(self):
        return float(self.feature_subsample_size.data)


class FileSharer(FlaskForm):
    data_file = FileField("Данные для предсказания: ", [my_validators('train')])  # my_validators('train')
    submit = SubmitField("Получить предсказание")


app = Flask(__name__, template_folder='templates')
app.config["SECRET_KEY"] = '571ebf8e13ca209536c29be68d435c00'

MODEL = None


@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST' and 'model' in request.form:
        MODEL = request.form['model']
        return redirect(url_for('choose_model', model=MODEL))
    return render_template('main.html')


@app.route('/<model>', methods=['GET', 'POST'])
def choose_model(model):
    if request.method == 'POST':
        if ("exit" in request.form and (request.form["exit"] == "Назад" or request.form["exit"] == "Выход")):
            return redirect(url_for('main_page'))
        if "make_predictions" in request.form:
            return redirect(url_for('make_predictions'))
    params = ModelParams()
    if model == 'boosting':
        params.model_name = 'boosting'
    else:
        params.model_name = 'random_forest'
    if params.validate_on_submit():
        TR_FILE, TS_FILE = params.train_data.data, params.valid_data.data
        TR_FILE.save("./src/storage/train.csv")
        if TS_FILE:
            TS_FILE.save("./src/storage/validation.csv")

        try:
            loaded_file_cols = pd.read_csv("./src/storage/train.csv").columns
            if params.target_name.data not in loaded_file_cols:
                raise ValueError
            # fit model here
            params.setup_and_fit()
        except:
            params.is_ok = False
        return render_template('results.html', form=params)
    return render_template('param_choice.html', form=params)


@app.route('/predict', methods=["GET", "POST"])
def make_predictions():
    global READY_TO_SEND
    if request.method == "POST" and "exit" in request.form:
        READY_TO_SEND = False
        return redirect(url_for('main_page'))
    file_data = FileSharer()
    if file_data.validate_on_submit():
        file_request = file_data.data_file.data
        file_request.save("./src/storage/test.csv")
        to_predict = pd.read_csv("./src/storage/test.csv")
        predictions = PREDICTOR.predict(to_predict.values)
        pd.DataFrame({'target': predictions}).to_csv("./src/storage/prediction.csv", index=False)
        READY_TO_SEND = True
        print(os.getcwd())
        return send_from_directory("./storage", "prediction.csv", as_attachment=True)
    return render_template('predict.html', form=file_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

