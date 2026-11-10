package com.eonmux.neuralfree.demo.tools;

import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * One completed measurement of {@code nftool bench}, llama.cpp llama-bench's {@code test}
 * ({@code tools/llama-bench/llama-bench.cpp}): the run's shape and engine settings, the model facts, the
 * RFC 3339 time of the run and the nanosecond samples of the repetitions. Beyond the reference's fields a row
 * carries {@code tg_readout}, what each timed generated token read back ({@link BenchReadout}): the
 * {@code --tg-readout} of a test that generates tokens through the decode path, {@code logits} for one that
 * generates none or runs the embedding mode, whose steps read the logits row whatever the flag says. The
 * statistics follow the reference: {@code avg} is the arithmetic mean, {@code stdev} the sample standard deviation
 * ({@code sqrt(sq_sum / (n-1) - mean^2 * n / (n-1))}, 0 for one sample); each sample's throughput is
 * {@code 1e9 * n_tokens / sample_ns} and {@code avg_ts}/{@code stddev_ts} average those throughputs.
 * Immutable.
 */
final class BenchTest {

    /** A field of the output rows: STRING, BOOL, INT or FLOAT, llama-bench's {@code field_type}. */
    enum FieldType {
        STRING,
        BOOL,
        INT,
        FLOAT
    }

    private static final DateTimeFormatter UTC = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss'Z'")
            .withZone(ZoneOffset.UTC);

    /** The output fields, in llama-bench's order; the settings the engine does not have are left out. */
    private static final String[] FIELDS = {
        "model_filename", "model_type", "model_size", "model_n_params", "backends",
        "n_batch", "n_ubatch", "n_threads", "n_ctx", "type_k", "type_v",
        "n_gpu_layers", "n_cpu_moe", "split_mode", "main_gpu", "no_kv_offload",
        "devices", "tensor_split", "tensor_buft_overrides", "load_mode", "embeddings", "tg_readout",
        "n_prompt", "n_gen", "n_depth", "test_time",
        "avg_ns", "stddev_ns", "avg_ts", "stddev_ts",
    };

    final BenchInstance inst;
    final String modelType;
    final long modelSize;
    final long modelNParams;
    final String backends;
    final int nCtx;
    final String tgReadout;
    final String testTime;
    final long[] samplesNs;

    BenchTest(BenchInstance inst, String modelType, long modelSize, long modelNParams, String backends,
              int nCtx, BenchReadout readout, long[] samplesNs) {
        this.inst = inst;
        this.modelType = modelType;
        this.modelSize = modelSize;
        this.modelNParams = modelNParams;
        this.backends = backends;
        this.nCtx = nCtx;
        this.tgReadout = inst.nGen > 0 && !inst.embeddings ? readout.toString() : BenchReadout.LOGITS.toString();
        this.testTime = UTC.format(Instant.now());
        this.samplesNs = samplesNs.clone();
    }

    /** The output field names, in order. */
    static String[] fields() {
        return FIELDS.clone();
    }

    /** The value type a field carries, for the JSON, Markdown and SQL printers. */
    static FieldType fieldType(String field) {
        switch (field) {
            case "avg_ts":
            case "stddev_ts":
                return FieldType.FLOAT;
            case "no_kv_offload":
            case "embeddings":
                return FieldType.BOOL;
            case "model_filename":
            case "model_type":
            case "backends":
            case "type_k":
            case "type_v":
            case "split_mode":
            case "devices":
            case "tensor_split":
            case "tensor_buft_overrides":
            case "load_mode":
            case "tg_readout":
            case "test_time":
                return FieldType.STRING;
            default:
                for (String f : FIELDS) {
                    if (f.equals(field)) {
                        return FieldType.INT;
                    }
                }
                return FieldType.STRING;
        }
    }

    /** Mean of the samples, 0 when there are none (llama.cpp {@code avg}). */
    long avgNs() {
        return avg(samplesNs);
    }

    /** Sample standard deviation of the samples, 0 for fewer than two (llama.cpp {@code stdev}). */
    long stdevNs() {
        return stdev(samplesNs);
    }

    /** Per-sample throughputs in tokens per second, one per sample. */
    double[] getTs() {
        double[] ts = new double[samplesNs.length];
        for (int i = 0; i < samplesNs.length; i++) {
            ts[i] = 1e9 * inst.nTokens() / samplesNs[i];
        }
        return ts;
    }

    /** Mean of the throughputs, 0 when there are none. */
    double avgTs() {
        return avg(getTs());
    }

    /** Sample standard deviation of the throughputs, 0 for fewer than two. */
    double stdevTs() {
        return stdev(getTs());
    }

    /** The field values of this test, in {@link #fields()} order, as the flat formats print them. */
    String[] values() {
        List<String> v = new ArrayList<>();
        v.add(inst.model);
        v.add(modelType);
        v.add(Long.toString(modelSize));
        v.add(Long.toString(modelNParams));
        v.add(backends);
        v.add(Integer.toString(inst.nBatch));
        v.add(Integer.toString(inst.nUbatch));
        v.add(Integer.toString(inst.nThreads));
        v.add(Integer.toString(nCtx));
        v.add(inst.typeK.name().toLowerCase(Locale.ROOT));
        v.add(inst.typeV.name().toLowerCase(Locale.ROOT));
        v.add(Integer.toString(inst.ngl));
        v.add(Integer.toString(inst.nCpuMoe));
        v.add(inst.splitMode.llamaName());
        v.add(Integer.toString(inst.mainGpu));
        v.add(inst.noKvOffload ? "1" : "0");
        v.add(inst.devicesString());
        v.add(inst.tensorSplitString());
        v.add(inst.tensorOverridesString());
        v.add(inst.loadMode.llamaName());
        v.add(inst.embeddings ? "1" : "0");
        v.add(tgReadout);
        v.add(Integer.toString(inst.nPrompt));
        v.add(Integer.toString(inst.nGen));
        v.add(Integer.toString(inst.nDepth));
        v.add(testTime);
        v.add(Long.toString(avgNs()));
        v.add(Long.toString(stdevNs()));
        v.add(doubleValue(avgTs()));
        v.add(doubleValue(stdevTs()));
        return v.toArray(new String[0]);
    }

    /** The fields and values of this test, in order. */
    Map<String, String> map() {
        Map<String, String> m = new LinkedHashMap<>();
        String[] f = fields();
        String[] v = values();
        for (int i = 0; i < f.length; i++) {
            m.put(f[i], v[i]);
        }
        return m;
    }

    /** A double as llama.cpp's {@code std::to_string} prints one: six decimal places, a plain point. */
    static String doubleValue(double v) {
        return String.format(Locale.ROOT, "%.6f", v);
    }

    private static long avg(long[] v) {
        if (v.length == 0) {
            return 0;
        }
        long sum = 0;
        for (long x : v) {
            sum += x;
        }
        return sum / v.length;
    }

    /** llama.cpp's integer stdev, computed on the square sums; rounds to a long like the template does. */
    private static long stdev(long[] v) {
        if (v.length <= 1) {
            return 0;
        }
        double mean = avg(v);
        double sqSum = 0;
        for (long x : v) {
            sqSum += (double) x * x;
        }
        return Math.round(Math.sqrt(sqSum / (v.length - 1) - mean * mean * v.length / (v.length - 1.0)));
    }

    private static double avg(double[] v) {
        if (v.length == 0) {
            return 0;
        }
        double sum = 0;
        for (double x : v) {
            sum += x;
        }
        return sum / v.length;
    }

    private static double stdev(double[] v) {
        if (v.length <= 1) {
            return 0;
        }
        double mean = avg(v);
        double sqSum = 0;
        for (double x : v) {
            sqSum += x * x;
        }
        return Math.sqrt(sqSum / (v.length - 1) - mean * mean * v.length / (v.length - 1.0));
    }
}
