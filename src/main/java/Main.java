import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

/**
 * Created by nsuprotivniy on 01.07.17.
 */
public class Main {

    public static void main(String[] args) {

        String sample_file_name = args[0];
        String config_file_name = args[1];
        String masterURLs = args[2];
        double learning_rate = Double.parseDouble(args[3]);
        double convergence_criteria = Double.parseDouble(args[4]);
        int max_iteration = Integer.parseInt(args[5]);


        SparkConf conf = new SparkConf().setAppName("Batch Gradient Descend").setMaster(masterURLs);
        JavaSparkContext jsc = new JavaSparkContext(conf);

        DataConfig dataConfig = new DataConfig(config_file_name);
        Broadcast<DataConfig> dataConfigBroadcast = jsc.broadcast(dataConfig);

        JavaRDD<String> lines = jsc.textFile(sample_file_name);
        JavaPairRDD<Double[], Double> curve = lines.mapToPair(s -> {

            int dim = dataConfigBroadcast.value().dim;

            String[] split_string = s.split(" ");
            Double y = Double.parseDouble(split_string[split_string.length - 1]);
            Double[] x = new Double[dim];

            for (int i = 0; i < dim; i++) {
                x[i] = Double.parseDouble(split_string[i]);
            }

            return new Tuple2<Double[], Double>(x, y);

        });

        curve.persist(StorageLevel.MEMORY_ONLY());
        curve.collect().forEach(tuple -> { for(Double d : tuple._1()) System.out.print(d + " "); System.out.println(tuple._2()); });

        Double[] batch_params = BatchGradientDescend.runWithSpark(jsc, curve, dataConfig.sample_size, dataConfig.dim, learning_rate, max_iteration, convergence_criteria);

        for (Double param : batch_params) {
            System.out.println(param);
        }

    }
}
