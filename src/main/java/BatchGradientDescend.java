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

    public void runWithSpark() {

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
            for (int i = 0; i < dimension; i++) {
                double cost = cost_function_derivative(y, x, params, i);
                params[i] += learning_rate * cost / sample_size;
            }

            // Изменение полученное на текущем шаге.
            double current_error = cost_function(y, x, params);

            // Если изменения меньше заданного порога, то результат принимается.
            if (Math.sqrt(current_error - total_error) < convergence_criteria) {
                System.out.println("DONE. Iteration " + iteration);
                break;
            } else {
                total_error = current_error;
            }

            System.out.println(current_error);

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
    private static double cost_function(Double[] y, Double[][] x, Double[] params) {

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

            result += cost_function_addition * x[derivative_num][i];
        }

        return result;
    }


}
