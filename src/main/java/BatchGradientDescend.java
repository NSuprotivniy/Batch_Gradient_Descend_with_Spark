import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.util.DoubleAccumulator;

/**
 * В классе реализованы методы поиска коэффициентов линейной регрессии методом градиентного спуска.
 *
 * Метод градиентного спуска минимизирует функцию стоимости, используя вектор коэффициентов,
 * найденный на предыдущем шаге. В качестве начальной оценки параметров используется единичный вектор.
 * Необходимым условием нахождения локального минимума является уменьшение изменения значения функции стоимости.
 * Алгоритм заканчивает свою работу, если изменение становится меньше допустимого значения.
 * В противном случае, если значение изменения растёт, то алгоритм отработает за заданное максимальное число шагов.
 *
 */
public class BatchGradientDescend {

    BatchGradientDescend() {

    }

    /**
     * Функция поиска коэффициентов линейной регрессии с помощью фреймворка Spark.
     * Выполняется обработка выборки точек, распределённых по вычислительным машинам кластера.
     * На каждой машине вычисляется часть очередного приближения коэффициента регрессии,
     * затем все части складываются и вычисляется коэффициент, который сохраняется на драйвере.
     *
     * Когда все коэффициенты подсчитаны, вычисляется значение функции стоимости также с использованием
     * рабочих машин и выборки точек, хранящейся на них.
     *
     * @param jsc                   контекст spark программы.
     * @param curve                 ссылка на выборку точек линейной регрессии.
     * @param sample_size           количество точек в выборке.
     * @param dimension             размерность вектора зависимых аргументов x.
     * @param learning_rate         скорость поиска минимума.
     * @param max_iteration         максимальное количество шагов работы алгоритма.
     * @param convergence_criteria  допустимое значение погрешности результата.
     *
     * @return                      вектор коэффициентов линейной регрессии или null.
     */
    public static Double[] runWithSpark(JavaSparkContext jsc,
                                     JavaPairRDD<Double[], Double> curve,
                                     int sample_size,
                                     int dimension,
                                     double learning_rate,
                                     int max_iteration,
                                     double convergence_criteria) {

        // Последнее значение функции ошибки. Нальное значение инициализируется как наибольшее.
        Double total_error = Double.MAX_VALUE;

        // Инициализация начального приближения параметров регрессиии: единичный вектор.
        // Одна копия (params_driver) хранится на драйвере, другая (params_broadcast) на каждой рабочей машине кластера.
        Double[] params_driver = new Double[dimension + 1];
        for (int i = 0; i < params_driver.length; i++) params_driver[i] = 1.0;
        Broadcast<Double[]> params_broadcast = jsc.broadcast(params_driver);


        // За ограниченное количество итераций.
        for (int iteration = 0; iteration < max_iteration; iteration++) {

            // Ссылка на распределённый по рабочим машинам вектор коэффициентов.
            Broadcast<Double[]> params_broadcast_1 = params_broadcast;

            // Для каждого коэффициента регрессии.
            for (int j = 0; j < dimension + 1; j++) {

                // Сохраняем индекс вычисляемого коэффициента на рабочих машинах.
                Broadcast<Integer> derivative_num_broadcast = jsc.broadcast(j);

                // Сохраняем значение коэффициента на рабочих значеним.
                // После подсчётов аккумулятор будет хранить новое значение коэффициента.
                DoubleAccumulator params_elem_acum = jsc.sc().doubleAccumulator();
                params_elem_acum.setValue(params_driver[j]);

                // Находим значение коэффициента регрессии.
                curve.foreach((tuple) -> {

                    Double[] x = tuple._1();
                    Double y = tuple._2();

                    Integer derivative_num_node = derivative_num_broadcast.getValue();
                    Double[] params_node = params_broadcast_1.getValue();


                    Double h = params_node[dimension];

                    for (int i = 0; i < dimension; i++) {
                        h += params_node[i] * x[i];
                    }

                    Double J;
                    if (derivative_num_node == dimension) {
                        J = (y - h);
                    } else {
                        J = (y - h) * x[derivative_num_node];
                    }

                    params_elem_acum.add(J);
                });

                params_driver[j] += learning_rate * params_elem_acum.value() / sample_size;
            }

            // Сохраняем вычисленный вектор коэффициентов на рабочих машинах для
            // получения значения функции стоимости и использования на следующем шаге.
            params_broadcast = jsc.broadcast(params_driver);

            // Ссылка на распределённый по рабочим машинам вектор коэффициентов.
            Broadcast<Double[]> params_broadcast_2 = params_broadcast;

            // Аккумулятор накапливает
            DoubleAccumulator total_error_accum = jsc.sc().doubleAccumulator();


            // Вычисление значения функции стоимости.
            curve.foreach((tuple) -> {

                Double[] x = tuple._1();
                Double y = tuple._2();

                Double[] params_node = params_broadcast_2.getValue();
                Double h = params_node[dimension] - y;

                for (int i = 0; i < dimension; i++) {
                    h += params_node[i] * x[i];
                }

                total_error_accum.add(h * h);

            });

            Double error = Math.sqrt(total_error_accum.value());

            // Сравнение изменения значения функции стоимости с пороговым значением.
            if (Math.abs(total_error - error) < convergence_criteria) {
                System.out.println("DONE. Iteration " + iteration);
                break;
            }
            else {
                total_error = error;
            }

        }

        return params_driver;
    }

    /**
     * Функция поиска коэффициентов линейной регрессии с помощью градиентного метода.
     * Вычисления производятся однопоточно.
     *
     * @param y                     вектор значений объясняемой переменной.
     * @param x                     таблица векторов значений зависимых аргументов.
     * @param dimension             размерность вектора зависимых аргументов x.
     * @param learning_rate         скорость поиска минимума.
     * @param max_iteration         максимально допустимое число шагов работы алгоритма.
     * @param convergence_criteria  допустимое значение погрешности результата.
     *
     * @return                      вектор коэффициентов линейной регрессии.
     */
    public static Double[] runWithSingleThread(Double[] y,
                                               Double[][] x,
                                               int dimension,
                                               double learning_rate,
                                               int max_iteration,
                                               double convergence_criteria) {

        // Количество точек в выборке.
        long sample_size = y.length;

        // Инициализация вектора параметров линейной регрессии: единичный вектор.
        Double[] params = new Double[dimension + 1];
        for (int i = 0; i < dimension + 1; i++) {
            params[i] = 1.0;
        }

        // Изменение функции стоимости. По этому значению производится минимизация.
        // Инициализируется максимально большим значением.
        double total_error = Double.MAX_VALUE;

        for (int iteration = 0; iteration < max_iteration; iteration++) {

            // Вычисление очередного приближения вектора параметров регрессии.
            for (int i = 0; i < dimension + 1; i++) {
                double cost = cost_function_derivative(y, x, params, i);
                params[i] += learning_rate * cost / sample_size;
            }

            // Изменение полученное на текущем шаге.
            double current_error = cost_function(y, x, params);

            // Если изменения меньше заданного порога, то результат принимается.
            if (Math.abs(current_error - total_error) < convergence_criteria) {
                System.out.println("DONE. Iteration " + iteration);
                break;
            } else {
                total_error = current_error;
            }

        }

        return params;

    }


    /**
     * Вычисляет значение функции стоимости.
     * Среднеквадратичное отклонение вектора y от скалярного произведения {x, 1}
     * на вектор параметров линейной регрессии.
     *
     * @param y         вектор значений объясняемой переменной.
     * @param x         таблица векторов значений зависимых аргументов.
     * @param params    вектор коэффициентов линейной регрессии.
     *
     * @return          значение функции стоимости.
     */
    public static double cost_function(Double[] y, Double[][] x, Double[] params) {

        double result = 0.0;

        int sample_size = y.length;
        int dim = x.length;

        for (int i = 0; i < sample_size; i++) {

            double error_add = params[params.length - 1] - y[i];

            for (int j = 0; j < dim; j++) {
                error_add += params[j] * x[j][i];
            }

            result += error_add * error_add;
        }

        return Math.sqrt(result);
    }

    /**
     * Вычисляет значение частной производной от функции стоимости.
     * Сумма разностей вектора y и скалярного произведения {x, 1} на вектор параметров линейной регрессии,
     * умноженных на коэффициент x, по которой производится дифференцирование.
     *
     * @param y                 вектор значений объясняемой переменной.
     * @param x                 таблица векторов значений зависимых аргументов.
     * @param params            вектор коэффициентов линейной регрессии.
     * @param derivative_num    номер аргумента, по которому производится дифференцирование.
     *
     * @return                  значение частной производной от функции стоимости.
     */
    private static double cost_function_derivative(Double[] y, Double[][] x, Double[] params, int derivative_num) {

        int sample_size = y.length;
        int dim = x.length;

        double result = 0.0;

        for (int i = 0; i < sample_size; i++) {

            double cost_function_addition = y[i] - params[params.length - 1];

            for (int j = 0; j < dim; j++) {
                cost_function_addition -= params[j] * x[j][i];
            }

            if (derivative_num == dim) {
                result += cost_function_addition;
            }
            else {
                result += cost_function_addition * x[derivative_num][i];
            }


        }

        return result;
    }


}
