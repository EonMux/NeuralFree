package com.eonmux.neuralfree.jmh;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
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

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * The host sampler chain over one Qwen-sized logits row ({@link Sampler#apply}), the work a whole-row readout
 * leaves to the host after each decode step: {@code greedy} (argmax), {@code default} (llama.cpp's chain: top-k 40,
 * top-p 0.95, min-p 0.05, temperature 0.8), {@code min-p} (min-p 0.05 alone over the whole row) and
 * {@code penalties} (the default chain behind repeat, presence and frequency penalties over a 64-token history).
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 2)
@Fork(1)
public class SamplerChainBenchmark {

    @Param({"greedy", "default", "min-p", "penalties"})
    public String chain;

    private Sampler sampler;
    private float[] logits;

    /** The chain a benchmark parameter names, seeded so every run draws the same tokens. */
    static Sampler chain(String name) {
        SamplerConfig config;
        switch (name) {
            case "greedy":
                config = SamplerConfig.greedyDecoding();
                break;
            case "default":
                config = SamplerConfig.defaults().withSeed(7);
                break;
            case "min-p":
                config = SamplerConfig.builder().topK(0).topP(1.0f).minP(0.05f).seed(7).build();
                break;
            case "penalties":
                config = SamplerConfig.defaults().toBuilder().repeatPenalty(1.1f).presencePenalty(0.5f)
                        .frequencyPenalty(0.5f).penaltyLastN(64).seed(7).build();
                break;
            default:
                throw new IllegalArgumentException("unknown chain " + name);
        }
        return Sampler.create(config);
    }

    /** A logits row of {@code size} entries: a Gaussian body with a few dominant candidates. */
    static float[] logits(int size, long seed) {
        Random random = new Random(seed);
        float[] row = new float[size];
        for (int i = 0; i < size; i++) {
            row[i] = (float) (random.nextGaussian() * 2.0);
        }
        for (int i = 0; i < 8; i++) {
            row[random.nextInt(size)] = 14.0f - i;
        }
        return row;
    }

    @Setup
    public void setUp() {
        sampler = chain(chain);
        logits = logits(SyntheticVocab.SIZE, 11);
        Random history = new Random(13);
        for (int i = 0; i < 64; i++) {
            sampler.accept(history.nextInt(SyntheticVocab.SIZE));
        }
    }

    @Benchmark
    public int row() {
        return sampler.apply(logits).selectedId();
    }
}
