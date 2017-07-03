package bench;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


/**
 * Класс генерирует выборку векторов по заданным параметрам.
 * К каждой точке прибовляется шумовой коэффициент.
 * Данные возвращаются через RDD Spark контекста.
 */
public class Helper {

    // Параметры выборки.
    private Double[] y;      // вектор значений объясняемой переменной.
    private Double[][] x;    // таблица векторов значений зависимых аргументов.
    private Double[] params; // вектор коэффициентов исходной функции.
    private int dimension;   // размерность вектора зависимых аргументов x.
    private int sample_size; // количество точек в выборке.

    double error_range = 10;
    double data_range = 100;

    Helper(int dimension, int sample_size) {

        this.dimension = dimension;
        this.sample_size = sample_size;

        init_curve_params();

    }

    public void init_curve_params() {

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


    /**
     * Сохраняет выборку в spark RDD в виде пары колекции пар <x[], y>,
     * каждая из которых описывает одну точку.
     *
     * @param jsc   spark контекст.
     * @return      ссылка на выборку точек линейной регрессии в spark RDD.
     */
    public JavaPairRDD<Double[], Double> spark_RDD(JavaSparkContext jsc) {

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
}
