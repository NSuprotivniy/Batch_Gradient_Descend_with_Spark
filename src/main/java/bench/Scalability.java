package bench;

import java.util.concurrent.TimeUnit;

import Main.BatchGradientDescend;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;


/**
 * Выполняется замер средней времени работы градиентного спуска на Spark.
 * Для заданного количества элементов в выборке, размерности вектора аргументов
 * генерируется зашумлённая выборка и сохраняется в RDD Spark контекста.
 *
 * Устанавливается Master URL, скорость спуска, максимальное число итераций, точность результата.
 *
 * Производится прогрев Java машины на 5 итерациях, затем 5 итераций для замера.
 */



@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(1)
public class Scalability {


    int sample_size = 1_000;                // Размер выборки.
    int dimension = 3;                      // Размерность вектора аргументов.

    double learning_rate = 1.0;             // Скорость спуска градиентного метода.
    int max_iteration = 1000;               // Максимальное число итераций.
    double convergence_criteria = 0.0001;   // Точность результата.

    Helper helper;



    JavaSparkContext jsc;                   // Spark контекст.
    JavaPairRDD<Double[], Double> data;     // RDD, в которой хранится выборка.



    @Setup(value = Level.Trial)
    public void setUpTrial() {

        // Генерация зашумлённой выборки.
        helper = new Helper(dimension, sample_size);


    }

    @Setup(value = Level.Invocation)
    public void setUpInvocation() {
        
        String masterURL = "local[1]";

        SparkConf conf = new SparkConf().setAppName("Batch Gradient Descend").setMaster(masterURL);
        jsc = new JavaSparkContext(conf);

        data = helper.spark_RDD(jsc);



    }

    @Benchmark
    public void measureBatchGradientDescend() {
        BatchGradientDescend.runWithSpark(jsc, data, sample_size, dimension, learning_rate, max_iteration, convergence_criteria);

        jsc.stop();
        jsc.close();
    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(Scalability.class.getSimpleName())
                .build();

        new Runner(opt).run();
    }
}

