package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;

import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Reads and validates the {@code prism.hadamard.*} metadata of a PRISM Hadamard-folded model and binds one
 * {@link HadamardSpec} per affected tensor, as llama.cpp does in {@code load_hparams_hadamard}
 * (llama.cpp src/llama-model.cpp:2052-2213) and {@code load_tensors_hadamard} (llama.cpp
 * src/llama-model.cpp:2217-2380).
 *
 * <p>A folded weight {@code W_f = W*D*H} needs its matmul input transformed by {@code H*(D*x)}
 * ({@link Result#forward()}); a latent lookup table needs its looked-up rows transformed by {@code D*(H*z)}
 * ({@link Result#inverse()}). With {@code prism.hadamard.tied_output} the token embedding appears in both
 * maps: forward for its use as the output head, inverse for the embedding lookup.
 *
 * <p>Every malformed or unsupported input throws {@link GgufFormatException} with llama.cpp's message.
 * Block sizes above {@link HadamardSpec#MAX_KERNEL_BLOCK} are valid here; GPU callers check
 * {@link HadamardSpec#kernelSupported()}.
 */
public final class PrismHadamardLoader {

    /** Tensor name of the token embedding. */
    public static final String TOKEN_EMBD = "token_embd.weight";
    /** Tensor name of the untied output head. */
    public static final String OUTPUT = "output.weight";

    private static final String TRANSFORM = "normalized-sylvester-walsh-hadamard";
    private static final String AXIS = "input-last-dimension";
    private static final String SIGN_MODE_IDENTITY = "identity";
    private static final String SIGN_MODE_EXPLICIT = "explicit";

    /**
     * Architectures whose graphs route every folded weight through the transform (llama.cpp
     * src/llama-model.cpp:2129-2141).
     */
    private static final Set<Arch> VERIFIED_ARCHS =
            EnumSet.of(Arch.LLAMA, Arch.QWEN3, Arch.QWEN3MOE, Arch.QWEN35, Arch.QWEN35MOE, Arch.QWEN3NEXT);

    /** Foldable per-layer weight kinds (llama.cpp src/llama-model.cpp:2146-2152). */
    private static final String[] FOLDABLE_KINDS = {
        "attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate", "attn_output",
        "ffn_gate", "ffn_up", "ffn_down",
        "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps",
        "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
        "ssm_out",
    };

    /** Dimensions of a model tensor ({@code ne[0]} first), or null when the model has no such tensor. */
    @FunctionalInterface
    public interface TensorShapes {
        long[] dims(String tensorName);
    }

    /** Bound transforms of one model. Immutable. */
    public static final class Result {
        private static final Result NONE = new Result(Collections.<String, HadamardSpec>emptyMap(),
                Collections.<String, HadamardSpec>emptyMap(), false);

        private final Map<String, HadamardSpec> forward;
        private final Map<String, HadamardSpec> inverse;
        private final boolean tiedOutput;

        private Result(Map<String, HadamardSpec> forward, Map<String, HadamardSpec> inverse, boolean tiedOutput) {
            this.forward = forward;
            this.inverse = inverse;
            this.tiedOutput = tiedOutput;
        }

        /** Folded weight name to the transform of its matmul input. Unmodifiable. */
        public Map<String, HadamardSpec> forward() {
            return forward;
        }

        /** Latent lookup table name to the transform of its looked-up rows. Unmodifiable. */
        public Map<String, HadamardSpec> inverse() {
            return inverse;
        }

        /** True when the output head is the latent token embedding ({@code prism.hadamard.tied_output}). */
        public boolean tiedOutput() {
            return tiedOutput;
        }

        /** True when the model has no PRISM transform. */
        public boolean isEmpty() {
            return forward.isEmpty() && inverse.isEmpty();
        }
    }

    private PrismHadamardLoader() {
    }

    /**
     * Folded weight name to the transform of its matmul input; the forward half of
     * {@link #load(GgufKv, TensorShapes, int, int)}, which documents the arguments.
     */
    public static Map<String, HadamardSpec> parse(GgufKv kv, TensorShapes shapes, int ssmDtRank, int ssmNGroup) {
        return load(kv, shapes, ssmDtRank, ssmNGroup).forward();
    }

    /**
     * Reads the PRISM metadata and binds every transform.
     *
     * @param kv        model metadata
     * @param shapes    tensor dimensions of the model (input width {@code ne[0]}, sign-vector selection)
     * @param ssmDtRank {@code ssm_dt_rank} (value heads of a gated delta net), used only with
     *                  {@code prism.hadamard.gdn_v_grouped}
     * @param ssmNGroup {@code ssm_n_group} (key heads of a gated delta net), used only with
     *                  {@code prism.hadamard.gdn_v_grouped}
     * @return the bound transforms; empty when the model carries no {@code prism.hadamard.version}
     * @throws GgufFormatException when the metadata is malformed, unsupported for the architecture, or does
     *                             not match the model's tensors
     */
    public static Result load(GgufKv kv, TensorShapes shapes, int ssmDtRank, int ssmNGroup) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(shapes, "shapes");
        final boolean tiedOutput = kv.bool(GgufKeys.PRISM_HADAMARD_TIED_OUTPUT, false);
        final long version = kv.u32Long(GgufKeys.PRISM_HADAMARD_VERSION, 0L);
        final boolean hasVersion = kv.has(GgufKeys.PRISM_HADAMARD_VERSION);

        Map<String, Integer> weightBlocks = new LinkedHashMap<>();
        Map<String, Integer> inverseBlocks = new LinkedHashMap<>();
        Map<Integer, float[]> signData = new LinkedHashMap<>();
        boolean gdnVGrouped = false;

        if (hasVersion) {
            if (version != 1 && version != 2) {
                throw fail("unsupported prism.hadamard.version: %d", version);
            }
            if ((version == 2) != tiedOutput) {
                throw fail("prism.hadamard version 2 requires tied_output=true; version 1 forbids it");
            }
            if (tiedOutput && shapes.dims(OUTPUT) != null) {
                throw fail("prism.hadamard.tied_output requires output.weight to be absent");
            }
            final int blockSize = readBlockSize(kv);
            checkFixedString(kv, GgufKeys.PRISM_HADAMARD_TRANSFORM, TRANSFORM, "unsupported prism.hadamard.transform: %s");
            checkFixedString(kv, GgufKeys.PRISM_HADAMARD_AXIS, AXIS, "unsupported prism.hadamard.axis: %s");
            final String signMode = kv.strRequired(GgufKeys.PRISM_HADAMARD_SIGN_MODE);
            if (!SIGN_MODE_IDENTITY.equals(signMode) && !SIGN_MODE_EXPLICIT.equals(signMode)) {
                throw fail("unsupported prism.hadamard.sign_mode: %s", signMode);
            }
            final String[] weightNames = requiredStrings(kv, GgufKeys.PRISM_HADAMARD_WEIGHT_NAMES);
            if (weightNames.length == 0) {
                throw fail("prism.hadamard.weight_names is empty");
            }
            if (SIGN_MODE_EXPLICIT.equals(signMode)) {
                readSigns(kv, blockSize, signData);
            }
            gdnVGrouped = kv.bool(GgufKeys.PRISM_HADAMARD_GDN_V_GROUPED, false);

            final Arch arch = Arch.fromGgufName(kv.archName());
            if (!VERIFIED_ARCHS.contains(arch)) {
                throw fail("prism.hadamard: arch '%s' is not verified to apply the activation transform to all folded weights",
                        kv.archName());
            }
            for (String name : weightNames) {
                if (!isFoldableWeight(name)) {
                    throw fail("prism.hadamard: weight '%s' is not on a verified Hadamard-aware matmul path", name);
                }
                if (weightBlocks.putIfAbsent(name, blockSize) != null) {
                    throw fail("duplicate prism.hadamard weight: %s", name);
                }
            }
            final String[] inverseNames = kv.strArray(GgufKeys.PRISM_HADAMARD_INVERSE_WEIGHT_NAMES);
            if (inverseNames != null) {
                for (String name : inverseNames) {
                    if (!TOKEN_EMBD.equals(name)) {
                        throw fail("prism.hadamard: weight '%s' is not a verified inverse-after-lookup table", name);
                    }
                    if (weightBlocks.containsKey(name) || inverseBlocks.putIfAbsent(name, blockSize) != null) {
                        throw fail("duplicate prism.hadamard inverse weight: %s", name);
                    }
                }
            }
        }

        if (tiedOutput) {
            if (version != 2) {
                throw fail("prism.hadamard.tied_output requires version 2");
            }
            Integer latent = inverseBlocks.get(TOKEN_EMBD);
            if (latent == null) {
                throw fail("prism.hadamard.tied_output requires a latent token embedding");
            }
            weightBlocks.putIfAbsent(TOKEN_EMBD, latent);
        } else if (inverseBlocks.containsKey(TOKEN_EMBD) && shapes.dims(OUTPUT) == null) {
            throw fail("a tied Hadamard output requires version 2 and tied_output=true");
        }

        if (weightBlocks.isEmpty() && inverseBlocks.isEmpty()) {
            return Result.NONE;
        }
        Map<String, HadamardSpec> forward = bind(weightBlocks, HadamardSpec.SignOrder.BEFORE_ROTATION, shapes, signData,
                gdnVGrouped, ssmDtRank, ssmNGroup);
        Map<String, HadamardSpec> inverse = bind(inverseBlocks, HadamardSpec.SignOrder.AFTER_ROTATION, shapes, signData,
                gdnVGrouped, ssmDtRank, ssmNGroup);
        return new Result(forward, inverse, tiedOutput);
    }

    /**
     * True for {@code output.weight} and {@code blk.<n>.<kind>.weight} with a foldable kind
     * (llama.cpp src/llama-model.cpp:2145-2174).
     */
    public static boolean isFoldableWeight(String name) {
        if (OUTPUT.equals(name)) {
            return true;
        }
        if (!name.startsWith("blk.")) {
            return false;
        }
        int pos = 4;
        while (pos < name.length() && Character.isDigit(name.charAt(pos)) && name.charAt(pos) < 128) {
            pos++;
        }
        if (pos == 4 || pos >= name.length() || name.charAt(pos) != '.') {
            return false;
        }
        String rest = name.substring(pos + 1);
        for (String kind : FOLDABLE_KINDS) {
            if (rest.equals(kind + ".weight")) {
                return true;
            }
        }
        return false;
    }

    /** {@code prism.hadamard.block_size}: a positive power of two (llama.cpp src/llama-model.cpp:2079-2081). */
    private static int readBlockSize(GgufKv kv) {
        if (!kv.has(GgufKeys.PRISM_HADAMARD_BLOCK_SIZE)) {
            throw fail("required metadata key '%s' is missing", GgufKeys.PRISM_HADAMARD_BLOCK_SIZE);
        }
        long b = kv.u32Long(GgufKeys.PRISM_HADAMARD_BLOCK_SIZE, 0L);
        if (b == 0 || (b & (b - 1)) != 0 || b > (1L << 30)) {
            throw fail("invalid prism.hadamard.block_size: %d", b);
        }
        return (int) b;
    }

    private static void checkFixedString(GgufKv kv, String key, String expected, String message) {
        String v = kv.strRequired(key);
        if (!expected.equals(v)) {
            throw fail(message, v);
        }
    }

    private static String[] requiredStrings(GgufKv kv, String key) {
        String[] v = kv.strArray(key);
        if (v == null) {
            throw fail("required metadata key '%s' is missing", key);
        }
        return v;
    }

    private static int[] requiredInts(GgufKv kv, String key) {
        int[] v = kv.i32Array(key);
        if (v == null) {
            throw fail("required metadata key '%s' is missing", key);
        }
        return v;
    }

    /** Explicit sign vectors, one per input width (llama.cpp src/llama-model.cpp:2095-2124). */
    private static void readSigns(GgufKv kv, int blockSize, Map<Integer, float[]> signData) {
        final int[] widths = requiredInts(kv, GgufKeys.PRISM_HADAMARD_SIGN_WIDTHS);
        final int[] values = requiredInts(kv, GgufKeys.PRISM_HADAMARD_SIGN_VALUES);
        if (widths.length == 0) {
            throw fail("prism.hadamard.sign_mode is explicit but sign_widths is empty");
        }
        int off = 0;
        for (int width : widths) {
            if (width <= 0 || width % blockSize != 0 || (long) off + width > values.length) {
                throw fail("invalid prism.hadamard sign width: %d", width);
            }
            if (signData.containsKey(width)) {
                throw fail("duplicate prism.hadamard sign width: %d", width);
            }
            float[] vec = new float[width];
            for (int i = 0; i < width; i++) {
                int v = values[off + i];
                if (v != 1 && v != -1) {
                    throw fail("prism.hadamard sign values must be +/-1");
                }
                vec[i] = v;
            }
            signData.put(width, vec);
            off += width;
        }
        if (off != values.length) {
            throw fail("prism.hadamard.sign_values length mismatch");
        }
    }

    /** One spec per named tensor (llama.cpp src/llama-model.cpp:2235-2375). */
    private static Map<String, HadamardSpec> bind(Map<String, Integer> blocks, HadamardSpec.SignOrder order,
                                                  TensorShapes shapes, Map<Integer, float[]> signData,
                                                  boolean gdnVGrouped, int ssmDtRank, int ssmNGroup) {
        if (blocks.isEmpty()) {
            return Collections.emptyMap();
        }
        Map<String, HadamardSpec> out = new LinkedHashMap<>();
        for (Map.Entry<String, Integer> e : blocks.entrySet()) {
            final String name = e.getKey();
            final int blockSize = e.getValue();
            final long[] dims = shapes.dims(name);
            if (dims == null || dims.length == 0) {
                throw fail("prism.hadamard weight not found: %s", name);
            }
            final long ne0 = dims[0];
            if (ne0 <= 0 || ne0 % blockSize != 0) {
                throw fail("prism.hadamard block size %d does not divide input dimension %d for %s", blockSize, ne0, name);
            }
            if (ne0 > Integer.MAX_VALUE) {
                throw fail("prism.hadamard input dimension %d of %s exceeds the supported maximum", ne0, name);
            }
            final int width = (int) ne0;
            float[] signs = null;
            if (!signData.isEmpty()) {
                signs = signData.get(width);
                if (signs == null) {
                    throw fail("prism.hadamard has no sign vector for width %d (%s)", width, name);
                }
            }
            int permHd = 0;
            int permNk = 0;
            int permRep = 0;
            if (gdnVGrouped && name.contains(".ssm_out.")) {
                // ssm_out of a gated delta net reads its value heads in tiled order (llama.cpp
                // src/llama-model.h:682-690)
                final int nV = ssmDtRank;
                final int nK = ssmNGroup;
                if (nK <= 0 || nV <= 0 || nV % nK != 0 || width % nV != 0) {
                    throw fail("prism.hadamard: bad GDN head geometry for %s", name);
                }
                permHd = width / nV;
                permNk = nK;
                permRep = nV / nK;
            }
            final HadamardSpec.SignOrder so = signs == null ? HadamardSpec.SignOrder.NONE : order;
            out.put(name, HadamardSpec.of(blockSize, so, signs, permHd, permNk, permRep));
        }
        return Collections.unmodifiableMap(out);
    }

    private static GgufFormatException fail(String fmt, Object... args) {
        return new GgufFormatException(String.format(Locale.ROOT, fmt, args));
    }
}
