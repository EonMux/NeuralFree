package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * A LoRA adapter loaded from a GGUF adapter file ({@link LoraAdapterLoader}, the llama.cpp adapter format,
 * src/llama-adapter.cpp:151-427): one {@link Pair} per adapted base tensor, the adapter's {@code alpha}, its task
 * metadata (jina-embeddings-v3 task adapters) and the aLoRA invocation tokens.
 *
 * <p>A projection {@code y = W x} becomes {@code y = W x + scale * B (A x)} (llm_graph_context::build_lora_mm,
 * llama-graph.cpp:1648-1664), an expert product {@code y_p = W[e_p] x_p} becomes
 * {@code y_p + scale * B[e_p] (A[e_p] x_p)} (build_lora_mm_id, :1734-1746) and a token embedding row
 * {@code e = E[id]} becomes {@code e + scale * B A[id]} (:2694-2709), with {@code scale = userScale * alpha / rank},
 * or {@code userScale} when alpha is 0 (llama-adapter.h:53-57). {@link #delta} binds one pair to a user scale.
 * Immutable.</p>
 */
public final class LoraAdapter {

    private final String path;
    private final String arch;
    private final float alpha;
    private final String taskName;
    private final String promptPrefix;
    private final int[] invocationTokens;
    private final Map<String, Pair> pairs;
    private final Map<String, String> metadata;

    LoraAdapter(String path, String arch, float alpha, String taskName, String promptPrefix, int[] invocationTokens,
                Map<String, Pair> pairs, Map<String, String> metadata) {
        this.path = path;
        this.arch = arch;
        this.alpha = alpha;
        this.taskName = taskName;
        this.promptPrefix = promptPrefix;
        this.invocationTokens = invocationTokens == null ? new int[0] : invocationTokens.clone();
        this.pairs = Collections.unmodifiableMap(new LinkedHashMap<>(pairs));
        this.metadata = Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    /** The file the adapter was read from. */
    public String path() { return path; }
    /** {@code general.architecture} of the adapter, equal to the base model's. */
    public String arch() { return arch; }
    /** {@code adapter.lora.alpha}, 0 when absent. */
    public float alpha() { return alpha; }
    /** {@code adapter.lora.task_name}, or null. */
    public String taskName() { return taskName; }
    /** {@code adapter.lora.prompt_prefix}, or null. */
    public String promptPrefix() { return promptPrefix; }
    /** {@code adapter.alora.invocation_tokens} (empty for a plain LoRA). */
    public int[] invocationTokens() { return invocationTokens.clone(); }
    /** True for an activated LoRA, whose delta applies only from its invocation tokens on. */
    public boolean isActivated() { return invocationTokens.length > 0; }
    /** Scalar metadata of the adapter file as strings (llama_adapter_meta_*). */
    public Map<String, String> metadata() { return metadata; }
    /** Names of the adapted base tensors, in file order. */
    public Set<String> targets() { return pairs.keySet(); }

    /** The pair adapting base tensor {@code target}, or null. */
    public Pair pair(String target) {
        return pairs.get(target);
    }

    /** The effective scale of {@code target} for {@code userScale} (llama-adapter.h:53-57). */
    public float scale(String target, float userScale) {
        Pair p = requirePair(target);
        return alpha != 0.0f ? userScale * alpha / p.rank : userScale;
    }

    /**
     * The delta of {@code target} bound to {@code userScale}: {@code B} is pre-multiplied by the scale.
     *
     * @throws IllegalArgumentException when the adapter does not adapt {@code target}
     */
    public Delta delta(String target, float userScale) {
        Pair p = requirePair(target);
        float s = scale(target, userScale);
        float[] b = p.b.clone();
        for (int i = 0; i < b.length; i++) {
            b[i] *= s;
        }
        return new Delta(p, b);
    }

    private Pair requirePair(String target) {
        Pair p = pairs.get(Objects.requireNonNull(target, "target"));
        if (p == null) {
            throw new IllegalArgumentException("LoRA adapter " + path + " does not adapt " + target);
        }
        return p;
    }

    @Override
    public String toString() {
        return "LoraAdapter{" + path + ", " + pairs.size() + " tensors, alpha=" + alpha
                + (taskName != null ? ", task=" + taskName : "")
                + (invocationTokens.length > 0 ? ", aLoRA" : "") + '}';
    }

    /** The product a pair adapts. */
    public enum Target {
        /** A dense projection {@code W {in, out}}. */
        PROJECTION,
        /** A 3-D expert tensor {@code W {in, out, n_expert}} of {@code ggml_mul_mat_id}. */
        EXPERTS,
        /** A token-embedding table {@code E {n_embd, n_vocab}}. */
        EMBEDDING
    }

    /**
     * The decoded {@code lora_a} / {@code lora_b} of one base tensor, row-major F32:
     * <ul>
     *   <li>a projection {@code W {in, out}}: {@code A} is {@code rank} rows of {@code in} floats, {@code B} is
     *       {@code out} rows of {@code rank} floats;</li>
     *   <li>an expert tensor {@code W {in, out, n_expert}}: the same per expert, stacked by expert ({@code A[e][r]} at
     *       row {@code e*rank + r}, {@code B[e][o]} at row {@code e*out + o});</li>
     *   <li>a token embedding {@code E {n_embd, n_vocab}}: {@code A} is {@code n_vocab} rows of {@code rank} floats
     *       and {@code B} is {@code n_embd} rows of {@code rank} floats.</li>
     * </ul>
     * Immutable.
     */
    public static final class Pair {
        final String target;
        final Target kind;
        final int inDim;
        final int outDim;
        final int rank;
        final int experts;
        final float[] a;
        final float[] b;

        /** A projection ({@code embedding} false) or token-embedding pair. */
        Pair(String target, boolean embedding, int inDim, int outDim, int rank, float[] a, float[] b) {
            this(target, embedding ? Target.EMBEDDING : Target.PROJECTION, inDim, outDim, rank, 1, a, b);
        }

        Pair(String target, Target kind, int inDim, int outDim, int rank, int experts, float[] a, float[] b) {
            if (inDim < 1 || outDim < 1 || rank < 1 || experts < 1) {
                throw new IllegalArgumentException("LoRA pair of " + target + " needs positive sizes, got in " + inDim
                        + ", out " + outDim + ", rank " + rank + ", experts " + experts);
            }
            long aLen = kind == Target.EMBEDDING ? (long) inDim * rank : (long) experts * rank * inDim;
            long bLen = (long) experts * outDim * rank;
            if (a.length != aLen || b.length != bLen) {
                throw new IllegalArgumentException("LoRA pair of " + target + " holds " + a.length + " / " + b.length
                        + " values, expected " + aLen + " / " + bLen);
            }
            this.target = target;
            this.kind = kind;
            this.inDim = inDim;
            this.outDim = outDim;
            this.rank = rank;
            this.experts = experts;
            this.a = a;
            this.b = b;
        }

        public String target() { return target; }
        public Target kind() { return kind; }
        /** True for a token-embedding table, adapted per looked-up row. */
        public boolean embedding() { return kind == Target.EMBEDDING; }
        /** Projection input width, or the vocabulary size of an embedding table. */
        public int inDim() { return inDim; }
        /** Projection output width, or the embedding width. */
        public int outDim() { return outDim; }
        public int rank() { return rank; }
        /** Expert count of an expert pair, 1 otherwise. */
        public int experts() { return experts; }

        @Override
        public String toString() {
            return "LoraAdapter.Pair{" + target + ", " + kind + ", " + inDim + " -> " + outDim + ", rank " + rank
                    + (experts > 1 ? ", " + experts + " experts" : "") + '}';
        }
    }

    /**
     * One pair bound to a scale. {@link #a()} and {@link #scaledB()} are the F32 arrays the device paths upload; the
     * CPU products use {@link CpuMatVec#PLAIN} over F32 views of {@code A} and the scaled {@code B}, built on first
     * use. The bound values are immutable.
     */
    public static final class Delta {
        private final Pair pair;
        private final float[] scaledB;
        private volatile Views views;

        Delta(Pair pair, float[] scaledB) {
            this.pair = pair;
            this.scaledB = scaledB;
        }

        public Pair pair() { return pair; }
        /** {@code A}, row-major (shared, read only). */
        public float[] a() { return pair.a; }
        /** {@code scale * B}, row-major (shared, read only). */
        public float[] scaledB() { return scaledB; }

        /**
         * {@code y[t] += scale * B (A x[t])} for {@code nTokens} rows of a projection.
         *
         * @param tmp scratch of at least {@code tmpOff + nTokens * rank} floats, overwritten
         * @throws IllegalStateException for an expert or embedding pair
         */
        public void add(int nTokens, float[] x, int xOff, int xStride, float[] tmp, int tmpOff, float[] y, int yOff,
                        int yStride) {
            require(Target.PROJECTION, "add");
            Views v = views();
            CpuMatVec.PLAIN.matmul(v.a, x, xOff, xStride, tmp, tmpOff, pair.rank, nTokens);
            for (int t = 0; t < nTokens; t++) {
                CpuMatVec.PLAIN.matvecAdd(v.b, tmp, tmpOff + t * pair.rank, y, yOff + t * yStride);
            }
        }

        /**
         * {@code y[p] += scale * B[ids[p]] (A[ids[p]] x_p)} for the {@code pairs} pairs {@code p = t*k + j}: the input
         * row of pair {@code p} at {@code xOff + (tokenRows ? p / k : p) * inDim}, its output row at
         * {@code yOff + p * outDim}.
         *
         * @param tmp scratch of at least {@code rank + outDim} floats, overwritten
         * @throws IllegalStateException    for a projection or embedding pair
         * @throws IllegalArgumentException for an expert id outside the pair's experts
         */
        public void addExperts(int pairs, int k, int[] ids, float[] x, int xOff, boolean tokenRows, float[] tmp,
                               float[] y, int yOff) {
            require(Target.EXPERTS, "addExperts");
            Views v = views();
            int r = pair.rank;
            int out = pair.outDim;
            for (int p = 0; p < pairs; p++) {
                int e = ids[p];
                if (e < 0 || e >= pair.experts) {
                    throw new IllegalArgumentException("expert " + e + " of pair " + p + " is outside the "
                            + pair.experts + " experts of the LoRA pair of " + pair.target);
                }
                int row = tokenRows ? p / k : p;
                CpuMatVec.PLAIN.matvecRows(v.a, (long) e * r, r, x, xOff + row * pair.inDim, tmp, 0);
                CpuMatVec.PLAIN.matvecRows(v.b, (long) e * out, out, tmp, 0, tmp, r);
                int base = yOff + p * out;
                for (int o = 0; o < out; o++) {
                    y[base + o] += tmp[r + o];
                }
            }
        }

        /**
         * {@code y[t] += scale * B A[ids[idsOff + t]]} for {@code n} embedded rows (llama-graph.cpp:2694-2709).
         *
         * @throws IllegalStateException    for a projection or expert pair
         * @throws IllegalArgumentException for an id outside the table
         */
        public void addEmbedding(int n, int[] ids, int idsOff, float[] y, int yOff, int yStride) {
            require(Target.EMBEDDING, "addEmbedding");
            Views v = views();
            for (int t = 0; t < n; t++) {
                int id = ids[idsOff + t];
                if (id < 0 || id >= pair.inDim) {
                    throw new IllegalArgumentException("token " + id + " is outside the " + pair.inDim
                            + " rows of the LoRA table of " + pair.target);
                }
                CpuMatVec.PLAIN.matvecAdd(v.b, pair.a, id * pair.rank, y, yOff + t * yStride);
            }
        }

        /** {@link #addEmbedding(int, int[], int, float[], int, int)} over every id of {@code ids}. */
        public void addEmbedding(int[] ids, float[] y, int yOff, int yStride) {
            addEmbedding(ids.length, ids, 0, y, yOff, yStride);
        }

        private void require(Target kind, String method) {
            if (pair.kind != kind) {
                throw new IllegalStateException(pair.target + " is a " + pair.kind + " pair; " + method + " serves "
                        + kind + " pairs");
            }
        }

        private Views views() {
            Views v = views;
            if (v == null) {
                CpuTensorRef a = pair.kind == Target.EMBEDDING ? null
                        : f32(pair.a, (long) pair.experts * pair.rank, pair.inDim);
                v = new Views(a, f32(scaledB, (long) pair.experts * pair.outDim, pair.rank));
                views = v;
            }
            return v;
        }

        private static CpuTensorRef f32(float[] values, long rows, long cols) {
            ByteBuffer buf = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            buf.asFloatBuffer().put(values);
            return CpuTensorRef.of(GGMLType.F32, buf, rows, cols);
        }

        /** The F32 views of {@code A} (null for a table) and of the scaled {@code B}. */
        private static final class Views {
            final CpuTensorRef a;
            final CpuTensorRef b;

            Views(CpuTensorRef a, CpuTensorRef b) {
                this.a = a;
                this.b = b;
            }
        }
    }
}
