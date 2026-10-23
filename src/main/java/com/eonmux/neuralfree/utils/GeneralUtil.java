package com.eonmux.neuralfree.utils;

import com.eonmux.neuralfree.datatype.structures.*;
import org.apache.logging.log4j.Level;

import java.io.BufferedReader;
import java.io.FileReader;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class GeneralUtil {

    public enum Transpose {
        NO,
        YES,
        CONJUGATE
    }

    private static final Pattern nameWithoutExtension = Pattern.compile("(?<=.)\\.[^.]+$");
    private static final Pattern nameWithoutPath = Pattern.compile(".*[/\\\\\\\\]");

    public static String getFileNameWithoutPath(String filePath) {
        return nameWithoutPath.matcher(filePath).replaceAll("");
    }

    public static String getFileNameWithoutExtension(String filePath) {
        return nameWithoutExtension.matcher(filePath).replaceAll("");
    }

    public static String getFileAsString(String filePath) {
        StringBuilder contentBuilder = new StringBuilder();
        try {
            URL url = new URL("file:///" + filePath);
            try (BufferedReader br = new BufferedReader(new FileReader(url.getPath()))) {
                String sCurrentLine;
                while ((sCurrentLine = br.readLine()) != null) {
                    contentBuilder.append(sCurrentLine).append("\n");
                }
            }
        } catch (Exception e) {
            LoggingUtil.getLogger(GeneralUtil.class).log(Level.ERROR, e);
        }
        return contentBuilder.toString();
    }

    public static String getResourceFileAsString(String filePath) {
        URL url = Thread.currentThread().getContextClassLoader().getResource(filePath);
        if (url == null) {
            LoggingUtil.getLogger(GeneralUtil.class).error("Resource not found: {}", filePath);
            return "";
        }
        StringBuilder contentBuilder = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(url.getPath()))) {
            String sCurrentLine;
            while ((sCurrentLine = br.readLine()) != null) {
                contentBuilder.append(sCurrentLine).append("\n");
            }
        } catch (Exception e) {
            LoggingUtil.getLogger(GeneralUtil.class).log(Level.ERROR, e);
        }
        return contentBuilder.toString();
    }

    public static double distanceBetweenPoints(Vector3F x1, Vector3F x2) {
        return Math.sqrt(Math.pow(x2.getF1() - x1.getF1(), 2d) + Math.pow(x2.getF2() - x1.getF2(), 2d) + Math.pow(x2.getF3() - x1.getF3(), 2d));
    }

    public static double distanceBetweenPoints(Vector2F x1, Vector2F x2) {
        return Math.sqrt(Math.pow(x2.getF1() - x1.getF1(), 2d) + Math.pow(x2.getF2() - x1.getF2(), 2d));
    }

    public static boolean isNumeric(String string) {
        return string.matches("-?\\d+([.]\\d+)?");
    }

    public static <K, V> Map<K, V> zipToMap(List<K> keys, List<V> values) {
        return IntStream.range(0, keys.size()).boxed()
                .collect(toMap(keys::get, values::get, false));
    }

    public static <K, V> Map<K, V> zipToMap(K[] keys, V[] values) {
        return IntStream.range(0, keys.length).boxed()
                .collect(toMap(Arrays.asList(keys)::get, Arrays.asList(values)::get, false));
    }

    public static <T, K, U> Collector<T, ?, Map<K, U>> toMap(Function<? super T, ? extends K> keyMapper, Function<? super T, ? extends U> valueMapper, boolean suppressWarn) {
        return Collectors.collectingAndThen(
                Collectors.toList(),
                list -> {
                    Map<K, U> map = new LinkedHashMap<>();
                    list.forEach(item -> {
                        K key = keyMapper.apply(item);
                        if (map.containsKey(key) && !suppressWarn) {
                            LoggingUtil.getLogger(GeneralUtil.class).warn("Duplicate key: {}", key);
                        }
                        U value = valueMapper.apply(item);
                        map.put(key, value);
                    });
                    return map;
                }
        );
    }

    public static <K> ArrayList<K> cleanArray(K[] array) {
        ArrayList<K> list = new ArrayList<>();
        for (K elem : array) {
            if (elem != null) {
                list.add(elem);
            }
        }
        return list;
    }

    public static String quote(String s) {
        return new StringBuilder()
                .append('\'')
                .append(s)
                .append('\'')
                .toString();
    }

    public static byte[] IntToBytes(int value) {
        ByteBuffer buffer = ByteBuffer.allocate(4);
        buffer.putInt(value);
        return buffer.array();
    }

    public static byte[] LongToBytes(long value) {
        ByteBuffer buffer = ByteBuffer.allocate(8);
        buffer.putLong(value);
        return buffer.array();
    }

    public static byte[] FloatToBytes(float value) {
        ByteBuffer buffer = ByteBuffer.allocate(4);
        buffer.putFloat(value);
        return buffer.array();
    }

    public static byte[] floatsToBytes(float[] values) {
        ByteBuffer buffer = ByteBuffer.allocate(4 * values.length);

        for (float value : values) {
            buffer.putFloat(value);
        }

        return buffer.array();
    }

    public static byte[] floatsToBytes(FloatDataSet values) {
        ByteBuffer buffer = ByteBuffer.allocate(4 * values.size());

        for (float value : values.toArray()) {
            buffer.putFloat(value);
        }

        return buffer.array();
    }

    public static void dumpBuffer(FloatBuffer buffer) {
        float[] temp = new float[buffer.capacity()];
        buffer.position(0);
        buffer.get(temp);
        System.out.println("FloatBuffer contents: [");
        for (Float f : temp) {
            System.out.println(f + ",");
        }
        System.out.println("]");
    }

    public static float[] convertCollectionToFloatArray(Collection<Float> floatSet) {
        float[] floatArray = new float[floatSet.size()];
        int index = 0;
        for (Float floatValue : floatSet) {
            floatArray[index++] = floatValue;
        }
        return floatArray;
    }
}
