# Batch Gradient Descend with Spark

Программа решает задачу поиска коэффициентов линейной регрессии, минимизируя функцию стоимости с помощью метода градиентного спуска. Ставится проблема обработки большой выборки точек, котороые невозможно хранить в оперативной памяти одной машины. Вычисление функции стоимости может быть поделено на части и выполняться параллельно. Для параллельной работы с исходными данными используется фреймворк Apache Spark. 


### Ссылки

* Apache Spark [https://spark.apache.org/](https://spark.apache.org/)
* Документация  [https://spark.apache.org/docs/latest/](https://spark.apache.org/docs/latest/)
* Описание параллельной реализации алгоритма [http://www.cse.buffalo.edu/faculty/miller/Courses/CSE633/Li-Hui-Fall-2012-CSE633.pdf](http://www.cse.buffalo.edu/faculty/miller/Courses/CSE633/Li-Hui-Fall-2012-CSE633.pdf)
