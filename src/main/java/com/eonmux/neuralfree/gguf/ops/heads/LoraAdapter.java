package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;

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
    private final String digest;
    private final long contentLength;

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
        long[] length = {0L};
        this.digest = digestOf(arch, alpha, this.invocationTokens, this.metadata, this.pairs, length);
        this.contentLength = length[0];
    }

    /**
     * The content digest of the adapter, the persistent identity, as 64 hex characters together with
     * {@link #contentLength}: a SHA-256 over the canonical view metadata ({@code general.architecture},
     * {@code adapter.lora.alpha} — which folds a PEFT {@code use_rslora} adapter's alpha at open, the task and
     * aLoRA metadata and every remaining metadata scalar, in sorted key order), the target mapping and tensor
     * table (target name, pair kind, dimensions and rank per adapted tensor) and the exact tensor bytes the
     * forward pass computes from — the decoded F32 halves of every pair. The digest needs no file re-read
     * after load.
     */
    public String digest() {
        return digest;
    }

    /** The number of bytes the digest covers. */
    public long contentLength() {
        return contentLength;
    }

    private static String digestOf(String arch, float alpha, int[] tokens, Map<String, String> metadata,
                                   Map<String, Pair> pairs, long[] lengthOut) {
        MessageDigest sha;
        try {
            sha = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available", e);
        }
        byte[] tag = bytes("lora-adapter.n1");
        sha.update(tag);
        long length = tag.length;
        length += hash(sha, bytes(arch));
        length += hash(sha, f32(new float[]{alpha}));
        length += hash(sha, u32(tokens));
        Map<String, String> sorted = new TreeMap<>();
        for (Map.Entry<String, String> e : metadata.entrySet()) {
            if (e.getKey() != null && e.getValue() != null) {
                sorted.put(e.getKey(), e.getValue());
            }
        }
        for (Map.Entry<String, String> e : sorted.entrySet()) {
            length += hash(sha, bytes(e.getKey()));
            length += hash(sha, bytes(e.getValue()));
        }
        for (Pair p : pairs.values()) {
            length += hash(sha, bytes(p.target()));
            length += hash(sha, s32(p.kind().ordinal()));
            length += hash(sha, u32(new int[]{p.inDim(), p.outDim(), p.rank(), p.experts()}));
            length += hash(sha, f32(p.a));
            length += hash(sha, f32(p.b));
        }
        lengthOut[0] = length;
        return hex(sha.digest());
    }

    /** Hashes {@code values} and counts them. */
    private static long hash(MessageDigest sha, byte[] values) {
        sha.update(values);
        return values.length;
    }

    private static byte[] bytes(String s) {
        byte[] b = s.getBytes(StandardCharsets.UTF_8);
        byte[] out = new byte[4 + b.length];
        out[0] = (byte) b.length;
        out[1] = (byte) (b.length >>> 8);
        out[2] = (byte) (b.length >>> 16);
        out[3] = (byte) (b.length >>> 24);
        System.arraycopy(b, 0, out, 4, b.length);
        return out;
    }

    private static byte[] u32(int[] values) {
        byte[] out = new byte[4 * values.length];
        for (int i = 0; i < values.length; i++) {
            int v = values[i];
            int o = i * 4;
            out[o] = (byte) v;
            out[o + 1] = (byte) (v >>> 8);
            out[o + 2] = (byte) (v >>> 16);
            out[o + 3] = (byte) (v >>> 24);
        }
        return out;
    }

    private static byte[] s32(int value) {
        return u32(new int[]{value});
    }

    private static byte[] f32(float[] values) {
        ByteBuffer buf = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        buf.asFloatBuffer().put(values);
        return buf.array();
    }

    private static String hex(byte[] digest) {
        StringBuilder sb = new StringBuilder(digest.length * 2);
        for (byte b : digest) {
            sb.append(String.format(Locale.ROOT, "%02x", b));
        }
        return sb.toString();
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
    /**
     * The start of the last occurrence of the invocation tokens in {@code tokens}, the first row an activated LoRA
     * applies to (llama.cpp's backward scan of {@code launch_slot_with_task}); -1 when they do not occur or the
     * adapter is not activated.
     */
    public int invocationStart(int[] tokens) {
        Objects.requireNonNull(tokens, "tokens");
        int n = invocationTokens.length;
        if (n == 0) {
            return -1;
        }
        for (int start = tokens.length - n; start >= 0; start--) {
            boolean match = true;
            for (int i = 0; i < n && match; i++) {
                match = tokens[start + i] == invocationTokens[i];
            }
            if (match) {
                return start;
            }
        }
        return -1;
    }

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
        return new Delta(p, alpha, b, userScale);
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

    /** The model a pair adapts: the text model's weights, or the vision/connector tensors of its mmproj file. */
    public enum Side {
        /** A tensor of the text model's forward pass. */
        TEXT,
        /** A tensor of the vision encoder or its connector (the mmproj file beside the text model). */
        VISION
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
        final Side side;
        final int inDim;
        final int outDim;
        final int rank;
        final int experts;
        final float[] a;
        final float[] b;

        /** A projection ({@code embedding} false) or token-embedding pair of the text model. */
        Pair(String target, boolean embedding, int inDim, int outDim, int rank, float[] a, float[] b) {
            this(target, embedding ? Target.EMBEDDING : Target.PROJECTION, Side.TEXT, inDim, outDim, rank, 1, a, b);
        }

        /** A text-model pair of {@code kind}. */
        Pair(String target, Target kind, int inDim, int outDim, int rank, int experts, float[] a, float[] b) {
            this(target, kind, Side.TEXT, inDim, outDim, rank, experts, a, b);
        }

        /** A pair of {@code side} and {@code kind}. */
        Pair(String target, Target kind, Side side, int inDim, int outDim, int rank, int experts, float[] a,
             float[] b) {
            this.side = Objects.requireNonNull(side, "side");
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
        /** The model whose weights this pair adapts. */
        public Side side() { return side; }
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
            return "LoraAdapter.Pair{" + target + ", " + kind + (side == Side.VISION ? ", vision" : "") + ", "
                    + inDim + " -> " + outDim + ", rank " + rank
                    + (experts > 1 ? ", " + experts + " experts" : "") + '}';
        }
    }

    /**
     * One pair bound to a scale. {@link #a()} and {@link #scaledB()} are the F32 arrays the device paths upload; the
     * CPU products use {@link CpuMatVec#PLAIN} over F32 views of {@code A} and the scaled {@code B}, built on first
     * use and dropped by {@link #rescale}. The pair and {@code alpha} are immutable; the bound scale is an engine
     * thread's mutability.
     */
    public static final class Delta {
        private final Pair pair;
        private final float alpha;
        private volatile float[] scaledB;
        private volatile Views views;
        private volatile float userScale;

        Delta(Pair pair, float alpha, float[] scaledB, float userScale) {
            this.pair = pair;
            this.alpha = alpha;
            this.scaledB = scaledB;
            this.userScale = userScale;
        }

        public Pair pair() { return pair; }
        /** {@code A}, row-major (shared, read only). */
        public float[] a() { return pair.a; }
        /** {@code scale * B}, row-major (shared, read only). */
        public float[] scaledB() { return scaledB; }

        /**
         * Rewrites {@code scaledB} to the effective scale of {@code userScale} × B in place and drops the cached
         * device views, so the next product rebuilds them from the new bytes. Same as the engine's rescale between
         * steps: it runs on the engine thread between two steps, never while a step dispatches.
         */
        public void rescale(float newUserScale) {
            if (!Float.isFinite(newUserScale)) {
                throw new IllegalArgumentException("LoRA pair " + pair.target + " cannot rescale to the non-finite"
                        + " scale " + newUserScale);
            }
            float s = alpha != 0.0f ? newUserScale * alpha / pair.rank : newUserScale;
            float[] b = pair.b.clone();
            for (int i = 0; i < b.length; i++) {
                b[i] *= s;
            }
            this.userScale = newUserScale;
            this.views = null;
            this.scaledB = b;
        }

        /** The user scale the delta's {@code scaledB} is bound to. */
        public float userScale() {
            return userScale;
        }

        /**
         * {@link #add(int, float[], int, int, float[], int, float[], int, int)} at scale 1.
         */
        public void add(int nTokens, float[] x, int xOff, int xStride, float[] tmp, int tmpOff, float[] y, int yOff,
                        int yStride) {
            addScaled(1.0f, nTokens, x, xOff, xStride, tmp, tmpOff, y, yOff, yStride);
        }

        /**
         * {@code y[t] += scale * (scale * B) (A x[t])} for {@code nTokens} rows of a projection: the effective
         * scale of {@code userScale} times {@code factor} ({@code factor} 1.0 adds the bound delta as bound; a
         * rescaled slot's rows run it at their slot key's factor of it).
         *
         * @param tmp scratch of at least {@code tmpOff + nTokens * rank} floats, overwritten
         * @throws IllegalStateException     for an expert or embedding pair
         * @throws IllegalArgumentException  for a non-finite factor
         */
        public void addScaled(float factor, int nTokens, float[] x, int xOff, int xStride, float[] tmp, int tmpOff,
                              float[] y, int yOff, int yStride) {
            require(Target.PROJECTION, "add");
            if (!Float.isFinite(factor)) {
                throw new IllegalArgumentException("LoRA pair " + pair.target + " needs a finite factor, got "
                        + factor);
            }
            Views v = views();
            if (factor == 1.0f) {
                CpuMatVec.PLAIN.matmul(v.a, x, xOff, xStride, tmp, tmpOff, pair.rank, nTokens);
                for (int t = 0; t < nTokens; t++) {
                    CpuMatVec.PLAIN.matvecAdd(v.b, tmp, tmpOff + t * pair.rank, y, yOff + t * yStride);
                }
                return;
            }
            CpuMatVec.PLAIN.matmul(v.a, x, xOff, xStride, tmp, tmpOff, pair.rank, nTokens);
            for (int t = 0; t < nTokens; t++) {
                scaledMatvecAdd(v.b, tmp, tmpOff + t * pair.rank, factor, y, yOff + t * yStride);
            }
        }

        /** {@code y[yOff + o] += factor * (b[o] . tmp[tmpOff ..])} over the pair's outputs. */
        private void scaledMatvecAdd(CpuTensorRef b, float[] tmp, int tmpOff, float factor, float[] y, int yOff) {
            int r = pair.rank;
            int out = pair.outDim;
            for (int o = 0; o < out; o++) {
                ByteBuffer row = b.rowBuffer(o);
                int off = b.rowOffset(o);
                float acc = 0.0f;
                for (int j = 0; j < r; j++) {
                    acc += row.getFloat(off + 4 * j) * tmp[tmpOff + j];
                }
                y[yOff + o] += factor * acc;
            }
        }

        /**
         * {@link #addExperts(int, int, int[], float[], int, boolean, float[], float[], int)} at scale 1.
         */
        public void addExperts(int pairs, int k, int[] ids, float[] x, int xOff, boolean tokenRows, float[] tmp,
                               float[] y, int yOff) {
            addExpertsScaled(1.0f, pairs, k, ids, x, xOff, tokenRows, tmp, y, yOff);
        }

        /**
         * {@code y[p] += factor * (scale * B[ids[p]]) (A[ids[p]] x_p)} for the {@code pairs} pairs
         * {@code p = t*k + j}: the input row of pair {@code p} at {@code xOff + (tokenRows ? p / k : p) * inDim},
         * its output row at {@code yOff + p * outDim}.
         *
         * @param tmp scratch of at least {@code rank + outDim} floats, overwritten
         * @throws IllegalStateException    for a projection or embedding pair
         * @throws IllegalArgumentException for an expert id outside the pair's experts or a non-finite factor
         */
        public void addExpertsScaled(float factor, int pairs, int k, int[] ids, float[] x, int xOff, boolean tokenRows,
                                     float[] tmp, float[] y, int yOff) {
            require(Target.EXPERTS, "addExperts");
            if (!Float.isFinite(factor)) {
                throw new IllegalArgumentException("LoRA pair " + pair.target + " needs a finite factor, got "
                        + factor);
            }
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
                    y[base + o] += factor * tmp[r + o];
                }
            }
        }

        /**
         * {@link #addEmbedding(int, int[], int, float[], int, int)} at scale 1.
         */
        public void addEmbedding(int n, int[] ids, int idsOff, float[] y, int yOff, int yStride) {
            addEmbeddingScaled(1.0f, n, ids, idsOff, y, yOff, yStride);
        }

        /**
         * {@code y[t] += factor * (scale * B) A[ids[idsOff + t]]} for {@code n} embedded rows
         * (llama-graph.cpp:2694-2709); {@code factor} is the delta's slot's factor of the bound scale.
         *
         * @throws IllegalStateException    for a projection or expert pair
         * @throws IllegalArgumentException for an id outside the table or a non-finite factor
         */
        public void addEmbeddingScaled(float factor, int n, int[] ids, int idsOff, float[] y, int yOff, int yStride) {
            require(Target.EMBEDDING, "addEmbedding");
            if (!Float.isFinite(factor)) {
                throw new IllegalArgumentException("LoRA pair " + pair.target + " needs a finite factor, got "
                        + factor);
            }
            Views v = views();
            for (int t = 0; t < n; t++) {
                int id = ids[idsOff + t];
                if (id < 0 || id >= pair.inDim) {
                    throw new IllegalArgumentException("token " + id + " is outside the " + pair.inDim
                            + " rows of the LoRA table of " + pair.target);
                }
                for (int o = 0; o < pair.outDim; o++) {
                    ByteBuffer bRow = v.b.rowBuffer(o);
                    int bOff = v.b.rowOffset(o);
                    float acc = 0.0f;
                    for (int j = 0; j < pair.rank; j++) {
                        acc += bRow.getFloat(bOff + 4 * j) * pair.a[id * pair.rank + j];
                    }
                    y[yOff + t * yStride + o] += factor * acc;
                }
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
