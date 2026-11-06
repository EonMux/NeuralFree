package com.eonmux.neuralfree.gguf.hf.config;

import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;

import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * The {@code dual_chunk_attention_config} of a flattened configuration (Qwen2.5-1M, vLLM
 * {@code DualChunkRotaryEmbedding} and the dual-chunk flash-attention backend): the chunk and local sizes, the
 * trained context length of the logit scaling factor and the sparse-attention knobs vLLM reads.
 *
 * <p>vLLM defaults {@code chunk_size} to 8192 and {@code local_size} to 1024, and enables the vertical-slash sparse
 * attention when the model directory carries a {@code sparse_attention_config.json} or the config names a
 * {@code sparse_attention_config}. NeuralFree implements the dense dual-chunk attention only; those sparse forms are
 * refused here by name, as is a malformed block or a {@code sparse_attention_enabled: true}.
 * Immutable.
 */
public final class HfDualChunk {

    /** The configuration key. */
    public static final String KEY = "dual_chunk_attention_config";
    /** File vLLM loads the per-head sparse selection from. */
    public static final String SPARSE_FILE = "sparse_attention_config.json";

    private final int chunkSize;
    private final int localSize;
    private final int originalMaxPositionEmbeddings;
    private final int sparseAttentionThreshold;
    private final int sparseAttentionLastQ;

    private HfDualChunk(int chunkSize, int localSize, int originalMaxPositionEmbeddings,
                        int sparseAttentionThreshold, int sparseAttentionLastQ) {
        this.chunkSize = chunkSize;
        this.localSize = localSize;
        this.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings;
        this.sparseAttentionThreshold = sparseAttentionThreshold;
        this.sparseAttentionLastQ = sparseAttentionLastQ;
    }

    /**
     * The dual-chunk block of {@code cfg}, empty when the configuration has none (a JSON {@code null} counts as
     * absent).
     *
     * @throws GgufFormatException naming the key for a malformed block or an enabled sparse attention
     */
    public static Optional<HfDualChunk> of(HfConfig cfg) {
        Objects.requireNonNull(cfg, "cfg");
        Object raw = cfg.flat().get(KEY);
        if (raw == null) {
            return Optional.empty();
        }
        String source = cfg.source();
        if (!(raw instanceof Map)) {
            throw new GgufFormatException(source + ": '" + KEY + "' must be an object, got "
                    + HfConfig.typeName(raw));
        }
        Map<String, Object> params = HfConfig.asObject(raw);
        refuseSparse(cfg, params, source);
        int chunk = integer(source, params, "chunk_size", 8192);
        int local = integer(source, params, "local_size", 1024);
        int orig = integer(source, params, "original_max_position_embeddings", 0);
        int threshold = integer(source, params, "sparse_attention_threshold", 32768);
        int lastQ = integer(source, params, "sparse_attention_last_q", 64);
        if (chunk <= 0 || local <= 0 || chunk <= local) {
            throw new GgufFormatException(source + ": '" + KEY + "' needs 0 < local_size < chunk_size, got "
                    + local + " / " + chunk);
        }
        if (orig < 0) {
            throw new GgufFormatException(source + ": '" + KEY + ".original_max_position_embeddings' must be >= 0,"
                    + " got " + orig);
        }
        if (threshold <= 0 || lastQ <= 0) {
            throw new GgufFormatException(source + ": '" + KEY + "' needs a positive sparse_attention_threshold and"
                    + " sparse_attention_last_q, got " + threshold + " / " + lastQ);
        }
        return Optional.of(new HfDualChunk(chunk, local, orig, threshold, lastQ));
    }

    /**
     * Refuses the sparse forms: vLLM turns the vertical-slash sparse attention on when the model directory holds
     * {@code sparse_attention_config.json} (or the block carries a {@code sparse_attention_config}), unless the
     * block says {@code sparse_attention_enabled: false} explicitly.
     */
    private static void refuseSparse(HfConfig cfg, Map<String, Object> params, String source) {
        Object explicit = params.get("sparse_attention_enabled");
        if (explicit != null && !(explicit instanceof Boolean)) {
            throw new GgufFormatException(source + ": '" + KEY + ".sparse_attention_enabled' must be a boolean, got "
                    + HfConfig.typeName(explicit));
        }
        if (Boolean.FALSE.equals(explicit)) {
            return;
        }
        Object sparseConfig = params.get("sparse_attention_config");
        boolean hasFile = cfg.dir() != null && cfg.dir().resolve(SPARSE_FILE).toFile().isFile();
        boolean enabled = Boolean.TRUE.equals(explicit) || sparseConfig != null || hasFile;
        if (enabled) {
            String cause = sparseConfig != null ? "sparse_attention_config" : "sparse_attention_enabled";
            throw new GgufFormatException(source + ": '" + KEY + "." + cause + "': vertical-slash sparse dual-chunk"
                    + " attention is not supported; NeuralFree runs the dense dual-chunk attention only");
        }
    }

    private static int integer(String source, Map<String, Object> params, String key, int dflt) {
        Object v = params.get(key);
        if (v == null) {
            return dflt;
        }
        long l = HfConfig.toLong(source, KEY + "." + key, v);
        if (l < 0 || l > HfConfig.MAX_WIDTH) {
            throw new GgufFormatException(source + ": '" + KEY + "." + key + "' value " + l + " is outside [0, "
                    + HfConfig.MAX_WIDTH + "]");
        }
        return (int) l;
    }

    /** The effective chunk size ({@code chunk_size}). */
    public int chunkSize() { return chunkSize; }
    /** The effective local size ({@code local_size}). */
    public int localSize() { return localSize; }
    /** The trained context length of the logit scaling factor; 0 disables it. */
    public int originalMaxPositionEmbeddings() { return originalMaxPositionEmbeddings; }
    /** {@code sparse_attention_threshold} (carried for the GGUF metadata; sparse attention itself is refused). */
    public int sparseAttentionThreshold() { return sparseAttentionThreshold; }
    /** {@code sparse_attention_last_q} (carried for the GGUF metadata; sparse attention itself is refused). */
    public int sparseAttentionLastQ() { return sparseAttentionLastQ; }

    /** Writes the NeuralFree GGUF keys of this block, after the converter's own keys. */
    public void writeTo(HfMetadataBuilder kv) {
        kv.set(GgufKeys.ATTENTION_DUAL_CHUNK_CHUNK_SIZE, chunkSize);
        kv.set(GgufKeys.ATTENTION_DUAL_CHUNK_LOCAL_SIZE, localSize);
        kv.set(GgufKeys.ATTENTION_DUAL_CHUNK_ORIGINAL_MAX_POSITION_EMBEDDINGS, originalMaxPositionEmbeddings);
        kv.set(GgufKeys.ATTENTION_DUAL_CHUNK_SPARSE_ENABLED, Boolean.FALSE);
        kv.set(GgufKeys.ATTENTION_DUAL_CHUNK_SPARSE_THRESHOLD, sparseAttentionThreshold);
        kv.set(GgufKeys.ATTENTION_DUAL_CHUNK_SPARSE_LAST_Q, sparseAttentionLastQ);
    }

    @Override
    public String toString() {
        return "HfDualChunk{chunk=" + chunkSize + ", local=" + localSize + ", orig="
                + originalMaxPositionEmbeddings + '}';
    }
}
