package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.ScratchWidths;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Preallocated working arrays of one model's CPU forward pass, the twin of {@code GpuScratch}.
 *
 * <ul>
 *   <li><b>Well-known slots</b> are allocated once at construction for {@code ubatch} rows ({@code [t][width]}) and
 *       never change: float slots {@link #X}, {@link #BRANCH}, {@link #NORMED}, {@link #Q}, {@link #K}, {@link #V},
 *       {@link #ATTN_OUT}, {@link #FFN_GATE_UP}, {@link #FFN_OUT}, {@link #LOGITS}, {@link #INPUT_EMBD}, and int
 *       slots {@link #TOKEN_IDS} and {@link #POS}. Widths follow {@link #forModel}.</li>
 *   <li><b>Temporaries</b> ({@link #temp}, {@link #intTemp}) are named working arrays of the blocks, requested at
 *       bind time. A request is served by the current array of that name when it is large enough; a larger request
 *       allocates a new array that becomes the current one, and earlier holders keep theirs. Holders of one name
 *       may overwrite each other's content between block calls, so a temporary carries data only within one call.</li>
 * </ul>
 *
 * <p>Blocks take every per-step array from here, so a step allocates nothing. Not thread-safe: one forward pass at a
 * time; blocks shard work inside a call with {@code CpuParallel}.</p>
 */
public final class CpuScratch {

    /** Residual stream (width: widest stream, {@code n_embd} or {@code hc * n_embd}). */
    public static final String X = "X";
    /** Output of the current mixer or FFN branch before the residual (stream width). */
    public static final String BRANCH = "BRANCH";
    /** Normalized block input (stream width). */
    public static final String NORMED = "NORMED";
    /** Query projection, or a fused Q/K/V or gated-Q projection (widest projection). */
    public static final String Q = "Q";
    /** Key projection (widest projection). */
    public static final String K = "K";
    /** Value projection (widest projection). */
    public static final String V = "V";
    /** Attention or mixer core output before the output projection (widest projection). */
    public static final String ATTN_OUT = "ATTN_OUT";
    /** Gate and up projections of the FFN, or the selected experts' intermediates. */
    public static final String FFN_GATE_UP = "FFN_GATE_UP";
    /** FFN output before the residual, or the selected experts' outputs before their weighted sum. */
    public static final String FFN_OUT = "FFN_OUT";
    /** Logits of the output rows ({@code vocab} floats per row). */
    public static final String LOGITS = "LOGITS";
    /** Embedding-stage inputs: padded input embeddings and per-layer embedding inputs. */
    public static final String INPUT_EMBD = "INPUT_EMBD";
    /** Token ids then token types of the step (2 ints per token). */
    public static final String TOKEN_IDS = "TOKEN_IDS";
    /** Positions of the step (4 ints per token, one per rope section). */
    public static final String POS = "POS";

    /** Every well-known float slot name (unmodifiable). */
    public static final List<String> FLOAT_SLOTS = Collections.unmodifiableList(Arrays.asList(
            X, BRANCH, NORMED, Q, K, V, ATTN_OUT, FFN_GATE_UP, FFN_OUT, LOGITS, INPUT_EMBD));
    /** Every well-known int slot name (unmodifiable). */
    public static final List<String> INT_SLOTS = Collections.unmodifiableList(Arrays.asList(TOKEN_IDS, POS));

    private final int ubatch;
    private final Map<String, float[]> slots;
    private final Map<String, int[]> intSlots;
    private final Map<String, float[]> temps = new HashMap<>();
    private final Map<String, int[]> intTemps = new HashMap<>();
    private long allocatedBytes;

    private CpuScratch(int ubatch, Map<String, Long> floatSizes, Map<String, Long> intSizes) {
        this.ubatch = ubatch;
        Map<String, float[]> f = new LinkedHashMap<>();
        for (Map.Entry<String, Long> e : floatSizes.entrySet()) {
            f.put(e.getKey(), new float[arrayLength(e.getKey(), e.getValue())]);
            allocatedBytes += 4L * e.getValue();
        }
        Map<String, int[]> i = new LinkedHashMap<>();
        for (Map.Entry<String, Long> e : intSizes.entrySet()) {
            i.put(e.getKey(), new int[arrayLength(e.getKey(), e.getValue())]);
            allocatedBytes += 4L * e.getValue();
        }
        this.slots = Collections.unmodifiableMap(f);
        this.intSlots = Collections.unmodifiableMap(i);
    }

    /**
     * Scratch for a model run with at most {@code ubatch} tokens per step (also for models without memory).
     *
     * <p>The float slots take the per-token widths of {@link ScratchWidths}, the sizing shared with
     * {@code GpuScratch}; {@code TOKEN_IDS} holds 2 and {@code POS} 4 ints per token.</p>
     *
     * @throws IllegalArgumentException when {@code ubatch < 1} or a slot exceeds one Java array (reduce the ubatch)
     */
    public static CpuScratch forModel(ModelSpec spec, int ubatch) {
        Objects.requireNonNull(spec, "spec");
        checkUbatch(ubatch);
        Map<String, Long> ints = new LinkedHashMap<>();
        ints.put(TOKEN_IDS, ScratchWidths.rows(ubatch, ScratchWidths.TOKEN_ID_INTS));
        ints.put(POS, ScratchWidths.rows(ubatch, ScratchWidths.POS_INTS));
        return new CpuScratch(ubatch, modelSlotFloats(spec, ubatch), ints);
    }

    /** Scratch with explicitly sized slots (well-known or custom names). */
    public static Builder builder(int ubatch) {
        checkUbatch(ubatch);
        return new Builder(ubatch);
    }

    /** Builder of a scratch with explicit slot sizes. */
    public static final class Builder {
        private final int ubatch;
        private final Map<String, Long> floats = new LinkedHashMap<>();
        private final Map<String, Long> ints = new LinkedHashMap<>();

        private Builder(int ubatch) {
            this.ubatch = ubatch;
        }

        /**
         * Adds a float slot of {@code n} elements.
         *
         * @throws IllegalArgumentException for an empty or repeated name or a non-positive size
         */
        public Builder slot(String name, long n) {
            declare(name, n);
            floats.put(name, n);
            return this;
        }

        /**
         * Adds an int slot of {@code n} elements.
         *
         * @throws IllegalArgumentException for an empty or repeated name or a non-positive size
         */
        public Builder intSlot(String name, long n) {
            declare(name, n);
            ints.put(name, n);
            return this;
        }

        private void declare(String name, long n) {
            if (name == null || name.isEmpty()) {
                throw new IllegalArgumentException("scratch slot name is empty");
            }
            if (n <= 0) {
                throw new IllegalArgumentException("scratch slot " + name + " needs a positive size, got " + n);
            }
            if (floats.containsKey(name) || ints.containsKey(name)) {
                throw new IllegalArgumentException("scratch slot " + name + " is declared twice");
            }
        }

        /** Allocates every slot. */
        public CpuScratch build() {
            return new CpuScratch(ubatch, floats, ints);
        }
    }

    // ------------------------------------------------------------------ sizing

    static Map<String, Long> modelSlotFloats(ModelSpec spec, int ubatch) {
        ScratchWidths w = ScratchWidths.of(spec);
        Map<String, Long> m = new LinkedHashMap<>();
        m.put(X, ScratchWidths.rows(ubatch, w.stream()));
        m.put(BRANCH, ScratchWidths.rows(ubatch, w.stream()));
        m.put(NORMED, ScratchWidths.rows(ubatch, w.stream()));
        m.put(Q, ScratchWidths.rows(ubatch, w.projection()));
        m.put(K, ScratchWidths.rows(ubatch, w.projection()));
        m.put(V, ScratchWidths.rows(ubatch, w.projection()));
        m.put(ATTN_OUT, ScratchWidths.rows(ubatch, w.projection()));
        m.put(FFN_GATE_UP, ScratchWidths.rows(ubatch, w.gateUp()));
        m.put(FFN_OUT, ScratchWidths.rows(ubatch, w.ffnOut()));
        m.put(LOGITS, ScratchWidths.rows(ubatch, w.logits()));
        m.put(INPUT_EMBD, ScratchWidths.rows(ubatch, w.input()));
        return m;
    }

    private static int arrayLength(String name, long n) {
        if (n > Integer.MAX_VALUE - 8L) {
            throw new IllegalArgumentException("scratch slot " + name + " needs " + n
                    + " elements, more than one Java array holds; reduce the ubatch (neuralfree.ubatch)");
        }
        return (int) n;
    }

    private static void checkUbatch(int ubatch) {
        if (ubatch < 1) {
            throw new IllegalArgumentException("ubatch must be at least 1, got " + ubatch);
        }
    }

    // ------------------------------------------------------------------ access

    /** Maximum tokens per step. */
    public int ubatch() {
        return ubatch;
    }

    /**
     * A float slot declared at construction.
     *
     * @throws IllegalArgumentException when no float slot has this name
     */
    public float[] slot(String name) {
        float[] s = slots.get(name);
        if (s == null) {
            throw new IllegalArgumentException("no float scratch slot '" + name + "' (slots: " + slots.keySet() + ")");
        }
        return s;
    }

    /**
     * An int slot declared at construction.
     *
     * @throws IllegalArgumentException when no int slot has this name
     */
    public int[] ints(String name) {
        int[] s = intSlots.get(name);
        if (s == null) {
            throw new IllegalArgumentException("no int scratch slot '" + name + "' (slots: " + intSlots.keySet()
                    + ")");
        }
        return s;
    }

    /** True when a float or int slot has this name. */
    public boolean has(String name) {
        return slots.containsKey(name) || intSlots.containsKey(name);
    }

    /** Names of the float slots in declaration order (unmodifiable). */
    public Set<String> slotNames() {
        return slots.keySet();
    }

    /**
     * A float temporary of at least {@code n} elements under {@code name}.
     *
     * @throws IllegalArgumentException for an empty name or a negative size
     */
    public float[] temp(String name, int n) {
        checkTemp(name, n);
        float[] t = temps.get(name);
        if (t == null || t.length < n) {
            t = new float[n];
            temps.put(name, t);
            allocatedBytes += 4L * n;
        }
        return t;
    }

    /**
     * An int temporary of at least {@code n} elements under {@code name}.
     *
     * @throws IllegalArgumentException for an empty name or a negative size
     */
    public int[] intTemp(String name, int n) {
        checkTemp(name, n);
        int[] t = intTemps.get(name);
        if (t == null || t.length < n) {
            t = new int[n];
            intTemps.put(name, t);
            allocatedBytes += 4L * n;
        }
        return t;
    }

    private static void checkTemp(String name, int n) {
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("scratch temporary name is empty");
        }
        if (n < 0) {
            throw new IllegalArgumentException("scratch temporary " + name + " needs a non-negative size, got " + n);
        }
    }

    /** Bytes allocated by this scratch so far (slots and temporaries, including superseded temporaries). */
    public long allocatedBytes() {
        return allocatedBytes;
    }

    @Override
    public String toString() {
        return "CpuScratch{ubatch=" + ubatch + ", slots=" + slots.keySet() + ", ints=" + intSlots.keySet()
                + ", temps=" + temps.keySet() + ", bytes=" + allocatedBytes + "}";
    }
}
