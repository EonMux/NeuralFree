package com.eonmux.neuralfree.gguf.hf.st;

import java.util.Objects;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The file names of a pre-sharded vLLM checkpoint, saved by {@code ShardedStateLoader.save_model} and loaded by
 * vLLM's {@code sharded_state} and {@code runai_streamer_sharded} formats.
 *
 * <p>The tensor-parallel-only layout is {@code model-rank-<rank>-part-<part>.safetensors}; with pipeline parallelism
 * it is {@code model-pp-<pp>-tp-<tp>-part-<part>.safetensors}. Every worker writes one or more part files holding
 * disjoint tensors of its own state dict; one worker is identified by its pipeline and tensor-parallel ranks.</p>
 */
public final class VllmShardFiles {

    private static final Pattern TP_ONLY = Pattern.compile("model-rank-(\\d{1,9})-part-(\\d{1,9})\\.safetensors");
    private static final Pattern PP_AND_TP =
            Pattern.compile("model-pp-(\\d{1,9})-tp-(\\d{1,9})-part-(\\d{1,9})\\.safetensors");

    private VllmShardFiles() {
    }

    /** One pre-sharded file: its worker's pipeline and tensor-parallel ranks, and its part index. */
    public static final class ShardFile {
        private final int pipeline;
        private final int tensor;
        private final int part;

        ShardFile(int pipeline, int tensor, int part) {
            this.pipeline = pipeline;
            this.tensor = tensor;
            this.part = part;
        }

        /** The pipeline-parallel rank; 0 in the tensor-parallel-only layout. */
        public int pipeline() {
            return pipeline;
        }

        /** The tensor-parallel rank. */
        public int tensor() {
            return tensor;
        }

        /** The part index within the worker's state dict. */
        public int part() {
            return part;
        }

        @Override
        public String toString() {
            return "pp=" + pipeline + ", tp=" + tensor + ", part=" + part;
        }
    }

    /** Whether {@code fileName} is a pre-sharded vLLM checkpoint file. */
    public static boolean isPreSharded(String fileName) {
        return parse(fileName).isPresent();
    }

    /**
     * The worker and part {@code fileName} names, or empty when it is not a pre-sharded checkpoint file.
     *
     * @throws ArithmeticException when a rank does not fit an {@code int} (it is nine digits at most, so it cannot)
     */
    public static Optional<ShardFile> parse(String fileName) {
        Objects.requireNonNull(fileName, "fileName");
        Matcher m = PP_AND_TP.matcher(fileName);
        if (m.matches()) {
            return Optional.of(new ShardFile(Integer.parseInt(m.group(1)), Integer.parseInt(m.group(2)),
                    Integer.parseInt(m.group(3))));
        }
        m = TP_ONLY.matcher(fileName);
        if (m.matches()) {
            return Optional.of(new ShardFile(0, Integer.parseInt(m.group(1)), Integer.parseInt(m.group(2))));
        }
        return Optional.empty();
    }
}
