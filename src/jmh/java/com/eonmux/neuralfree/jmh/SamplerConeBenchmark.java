package com.eonmux.neuralfree.jmh;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.util.Arrays;
import java.util.Comparator;
import java.util.concurrent.TimeUnit;

/**
 * The same chains as {@link SamplerChainBenchmark} over a device top-k readout instead of the whole row
 * ({@link Sampler#applyTopK}): the chain's cone of the row's best candidates in rank order, what a
 * {@code decodeTopKOnDevice} step hands the host.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 2)
@Fork(1)
public class SamplerConeBenchmark {

    @Param({"greedy", "default"})
    public String chain;

    private Sampler sampler;
    private int[] ids;
    private float[] values;

    @Setup
    public void setUp() {
        sampler = SamplerChainBenchmark.chain(chain);
        int cone = sampler.coneWidth();
        if (cone <= 0) {
            throw new IllegalStateException("the " + chain + " chain has no device cone");
        }
        float[] row = SamplerChainBenchmark.logits(SyntheticVocab.SIZE, 11);
        Integer[] order = new Integer[row.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, Comparator.<Integer>comparingDouble(i -> -row[i]).thenComparingInt(i -> i));
        ids = new int[cone];
        values = new float[cone];
        for (int i = 0; i < cone; i++) {
            ids[i] = order[i];
            values[i] = row[order[i]];
        }
    }

    @Benchmark
    public int cone() {
        return sampler.applyTopK(ids, values).selectedId();
    }
}
