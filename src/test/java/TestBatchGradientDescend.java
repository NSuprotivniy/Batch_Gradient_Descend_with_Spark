import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Тестирование фукнкций нахождения параметров линейной регрессии.
 */
public class TestBatchGradientDescend {

    // Параметры выборки.
    private Double[] y;      // вектор значений объясняемой переменной.
    private Double[][] x;    // таблица векторов значений зависимых аргументов.
    private Double[] params; // вектор коэффициентов исходной функции.
    private int dimension;   // размерность вектора зависимых аргументов x.
    private int sample_size; // количество точек в выборке.

    // Параметры градиаентного спуска.
    double learning_rate = 1.0;             // скорость поиска минимума.
    int max_iteration = 1024;               // максимальное количество шагов работы алгоритма.
    double convergence_criteria = 0.0001;   // допустимое значение погрешности результата.



    /**
     * Инициализация параметров исходных данных и самих исходных данных.
     */
    @Before
    public void init_curve_params() {

        dimension = 3;
        sample_size = 10_000;
        y = new Double[sample_size];
        x = new Double[dimension][];
        params = new Double[dimension + 1];

        for (int i = 0; i < dimension; i++) {
            x[i] = new Double[sample_size];
        }

        double error_range = 10;
        double data_range = 100;

        noise_function(error_range, data_range);

    }


    /**
     * Тестирование однопоточной реализации алгоритма нахождения коэффициентов линейной регрессии.
     * По заданным в init_curve_params() исходным данным строится вектор параметров регрессии.
     * Добавляется 100 точных значений в выборку и строится новый вектор параметров.
     * Производится проверка, что полученный вектор лучше приближает искомый.
     */
    @Test
    public void test_runWithSingleThread() {

        // Поиск коэффициентов регрессии.
        Double[] regression_params = BatchGradientDescend.runWithSingleThread(y, x, dimension, learning_rate, max_iteration, convergence_criteria );

        // Задаём размер новой выборки и массивы под неё.
        int new_sample_size = sample_size + 100;

        Double[][] new_x = new Double[dimension][];
        Double[]   new_y = new Double[new_sample_size];

        System.arraycopy(y, 0, new_y, 0, sample_size);

        // Генератор новых случайных точек.
        Random generator = new Random();
        double data_range = 100;

        // Генерация новых точек.
        for (int j = 0; j < dimension; j++) {
            new_x[j] = new Double[new_sample_size];
            System.arraycopy(x[j], 0, new_x[j], 0, sample_size);

            for (int i = sample_size; i < new_sample_size; i++) {
                new_x[j][i] = generator.nextDouble() % data_range;
            }
        }

        // Расчёт точного значения зависимого аргумента, без внесения шумов.
        for (int i = sample_size; i < new_sample_size; i++) {
            new_y[i] = params[dimension];
            for (int j = 0; j < dimension; j++) {
                new_y[i] += new_x[j][i] * params[j];
            }
        }

        // Поиск коэффициентов регрессии для новой выборки.
        Double[] new_regression_params = BatchGradientDescend.runWithSingleThread(new_y, new_x, dimension, learning_rate, max_iteration, convergence_criteria );

        // Среднеквадратичное отклонение исходных значений параметров от коэффициентов полученных по первой выборке.
        double average_square_real_params_and_regression_params = 0.0;
        // Среднеквадратичное отклонение исходных значений параметров от коэффициентов полученных по более точной выборке.
        double average_square_real_params_and_new_regression_params = 0.0;

        for (int i = 0; i < dimension + 1; i++) {
            average_square_real_params_and_regression_params += Math.pow(params[i] - regression_params[i], 2);
            average_square_real_params_and_new_regression_params += Math.pow(params[i] - new_regression_params[i], 2);
        }

        System.out.println(average_square_real_params_and_regression_params);
        System.out.println(average_square_real_params_and_new_regression_params);

        Assert.assertTrue("New approximation should be more acurate",
                average_square_real_params_and_regression_params >= average_square_real_params_and_new_regression_params);

    }

    /**
     * Тестирование многопоточной реализации поиска коэффициентов линейной регрессии с использованием spark.
     * По заданным в init_curve_params() исходным данным строится вектор параметров регрессии c помощью spark.
     * По этим же данным строится вектор с помощью однопоточной реализации.
     * Два результата сравниваются между собой через среднеквадратичное отклонение.
     */
    @Test
    public void test_runWithSpark() {

        // Задаём spark контест.
        SparkConf conf = new SparkConf().setAppName("test_app").setMaster("local[*]");
        JavaSparkContext jsc = new JavaSparkContext(conf);

        // Сохраним выборку в RDD.
        JavaPairRDD<Double[], Double> curve = spark_RDD(jsc);

        // Поиск коэффициентов регрессии с помощью spark.
        Double[] spark_params = BatchGradientDescend.runWithSpark(jsc, curve, sample_size, dimension, learning_rate, max_iteration, convergence_criteria);
        // Поиск коэффициентов регрессии однопоточно.
        Double[] single_threaded_params = BatchGradientDescend.runWithSingleThread(y, x, dimension, learning_rate, max_iteration, convergence_criteria );

        // Среднеквадратичное отклонение векторов, полученноых с помощью spark реализации и однопоточной реализации.
        double average_square = 0.0;

        for (int i = 0; i < dimension + 1; i++) {
            average_square += Math.pow(spark_params[i] - single_threaded_params[i], 2);
        }

        average_square = Math.sqrt(average_square);

        System.out.println(average_square);

        // Точность совпадения отклонений.
        double accuracy = 0.1;

        Assert.assertTrue("Approximation accuracy by params getting with spark and with single should be same",
                average_square < accuracy);

    }


    /**
     * Сохраняет выборку в spark RDD в виде пары колекции пар <x[], y>,
     * каждая из которых описывает одну точку.
     *
     * @param jsc   spark контекст.
     * @return      ссылка на выборку точек линейной регрессии в spark RDD.
     */
    private JavaPairRDD<Double[], Double> spark_RDD(JavaSparkContext jsc) {

        List<Tuple2<Double[], Double>> curve_list = new ArrayList<>(sample_size);

        for (int i = 0; i < sample_size; i++) {
            Double[] vector = new Double[dimension];
            for (int j = 0; j < dimension; j++) {
                vector[j] = x[j][i];
            }
            Tuple2<Double[], Double> tuple = new Tuple2<>(vector, y[i]);
            curve_list.add(tuple);
        }

        JavaRDD<Tuple2<Double[], Double>> rdd = jsc.parallelize(curve_list);

        return JavaPairRDD.fromJavaRDD(rdd);
    }

    /**
     * Генерация случайной выборки со случайным разбросом и шумовыми эффектами.
     */
    private void noise_function(double error_range, double data_range) {

        Random generator = new Random();
        Random noise = new Random();


        for (int j = 0; j < dimension; j++) {
            for (int i = 0; i < sample_size; i++) {
                x[j][i] = generator.nextDouble() % data_range;
            }
        }

        for (int j = 0; j < dimension + 1; j++) {
            params[j] = generator.nextDouble() % data_range;
        }

        for (int i = 0; i < sample_size; i++) {
            y[i] = params[dimension];
            for (int j = 0; j < dimension; j++) {
                y[i] += x[j][i] * params[j] + noise.nextDouble() % error_range;
            }
        }
    }
}
