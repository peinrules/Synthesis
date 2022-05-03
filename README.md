# Synthesis

Инструкция по запуску:
* Положите файл с данными в одну папку с файлом stocks_clusterization.py;
* Для запуска вам (вероятно) понадобится установить некоторые библиотеки. Ниже прикладываю инструкцию;
    - https://pypi.org/project/tslearn/
    - https://scikit-learn.org/stable/install.html
    - https://numpy.org/install/
    - https://pandas.pydata.org/pandas-docs/dev/getting_started/install.html
* Запуск осуществляется командой: `python3 stocks_clusterization.py -f <filename>`, где `<filename>` есть название файла с данными (например `data.csv`);
* После отработки програмы результат будет сохранён в файл `clusters_<filename>`.
