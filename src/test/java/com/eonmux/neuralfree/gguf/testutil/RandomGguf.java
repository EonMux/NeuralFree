package com.eonmux.neuralfree.gguf.testutil;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.SplittableRandom;

/**
 * A temporary GGUF of F32 tensors for mixer-level fixtures, opened as {@link CpuWeights} with its
 * {@link TensorCatalog}: random matrices in {@code [-scale, scale)} and constant vectors. Closing it closes the
 * weights and the parser and deletes the file.
 */
public final class RandomGguf implements AutoCloseable {

    private final Path path;
    private final GGUFFileParser parser;
    private final CpuWeights weights;

    private RandomGguf(Path path) throws IOException {
        this.path = path;
        this.parser = new GGUFFileParser(path.toString());
        this.weights = CpuWeights.open(parser);
    }

    public static Builder builder(String arch, long seed) {
        return new Builder(arch, seed);
    }

    public CpuWeights weights() {
        return weights;
    }

    public TensorCatalog catalog() {
        return TensorCatalog.of(parser);
    }

    @Override
    public void close() throws IOException {
        try {
            weights.close();
        } finally {
            try {
                parser.close();
            } finally {
                Files.deleteIfExists(path);
            }
        }
    }

    /** Collects the tensors of the file. */
    public static final class Builder {

        private final GgufTestWriter writer;
        private final SplittableRandom rng;

        private Builder(String arch, long seed) {
            this.writer = new GgufTestWriter().architecture(arch);
            this.rng = new SplittableRandom(seed);
        }

        /** A tensor of shape {@code ne} with values uniform in {@code [-scale, scale)}. */
        public Builder random(String name, float scale, long... ne) {
            float[] v = new float[count(ne)];
            for (int i = 0; i < v.length; i++) {
                v[i] = (float) ((2 * rng.nextDouble() - 1) * scale);
            }
            writer.tensorFromFloats(name, GGMLType.F32, ne, v);
            return this;
        }

        /** A tensor of shape {@code ne} with every value {@code value}. */
        public Builder constant(String name, float value, long... ne) {
            float[] v = new float[count(ne)];
            java.util.Arrays.fill(v, value);
            writer.tensorFromFloats(name, GGMLType.F32, ne, v);
            return this;
        }

        public RandomGguf open() throws IOException {
            return new RandomGguf(writer.writeTempFile());
        }

        private static int count(long[] ne) {
            long n = 1;
            for (long d : ne) {
                n *= d;
            }
            return Math.toIntExact(n);
        }
    }
}
