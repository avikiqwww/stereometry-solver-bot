"""
Фильтр «стереометрия / не стереометрия»: TF-IDF + метод опорных векторов (SVM).

Файл пока пустой: пишем кусками вместе и обсуждаем каждый кусок.

Что нужно написать (порядок обсуждается):
  1. Загрузка данных и разбиение (train / test, группы почти-дубликатов).
  2. Pipeline: предобработка -> TfidfVectorizer -> SVM.
  3. Обучение и метрики (accuracy, precision, recall, F1, ROC-AUC, матрица ошибок).
  4. Подбор гиперпараметров для линейного и полиномиального ядра
     (C, degree, gamma, coef0) и разбор их важности (теплокарта, доли дисперсии).
  5. Интерфейс классификатора: fit, predict, save, load.

Структура папки:
  svm_classifier.py   сам классификатор (pipeline, fit, predict, save, load)
  training/           скрипты обучения итоговой модели
  experiments/        эксперименты: сравнение ядер, подбор гиперпараметров, их важность
  models/             сохранённые модели (папка в .gitignore, в репозиторий не попадает)

Данные: data/processed/filter_dataset.jsonl (подвыборка: tfidf_subset_500.csv).
Предобработка: classifiers/common/text_morph.py (preprocess_lemma_no_task_words).
"""
