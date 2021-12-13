# EVMProject
Мой докерхаб с сервером https://hub.docker.com/r/beccohov/evmproject
<h1>Как запустить сервер?</h1>
<ul>
  <li>
  Клонировать этот репозиторий
  </li>
  <li>
  Скачать докер образ из докерхаба или собрать локальный из докерфайла (build.sh из корневой директории)
  </li>
  <li>
    Запустить run.sh  (из корневой директории) - обратите внимание, что этот файл сделан для работы с докерхабом. Если запускать локально, то можно использовать коментарий в этом файле
  </li>
</ul>

<h1>Как использовать приложение</h1>
<ul>
  <li>
  На главной странице выбрать модель и нажать "далее"
  </li>
  <li>
  Выбрать параметры и прикрепить файлы (тренировочный обязателен), указать правильный terget (название целевой колонки)
  </li>
  <li>
    Выбрать "обучить" и ждать, пока не загрузится страница. Когда загрузится - модель обучена. На странице отображены все графики rmse ан трейне и валидации (если было передано) и параметры модели. Если нажать "получить предсказания", то откроется страница, где нужно загрузить файл с тестом. При нажитии "Предсказать" будет загружен файл с файлом предсказанием на компьютер с сервера.
  </li>
  <li>
    В репозитории содержатся данные для примера - train.csv, test.csv (target == price). Данные не обработыны предварительно и перемешаны, поэтому модель на них работает не очень качественно, но зато работает и можно протестировать.
  </li>
</ul>
