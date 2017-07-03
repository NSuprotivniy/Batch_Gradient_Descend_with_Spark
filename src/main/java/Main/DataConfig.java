package Main;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.Scanner;

/**
 * Класс-хранилище основных параметров выборки точек линейной регрессии.
 */
public class DataConfig implements Serializable {

    public long sample_size; // количество точек в выборке.
    public int dim;          // размерность вектора зависимых аргументов x.

    /**
     * Чтение информации из переменных.
     *
     * @param sample_size количество точек в выборке.
     * @param dim         размерность вектора зависимых аргументов x.
     */
    DataConfig(long sample_size, int dim) {
        this.sample_size = sample_size;
        this.dim = dim;
    }

    /**
     * Чтение информации из конфигурационного файла.
     *
     * @param config_file путь к конфигурационному файлу.
     */
    DataConfig(String config_file) {

        try(Scanner scanner = new Scanner(new File(config_file))) {
            sample_size = scanner.nextLong();
            dim = scanner.nextInt();
        }
        catch (IOException e) {
            System.out.println("Cannot open file");
        }
    }
}