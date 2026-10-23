package com.eonmux.neuralfree.gguf.ops.audio;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;

/**
 * Validated geometry of a wavtokenizer-dec model (llama.cpp src/models/wavtokenizer-dec.cpp), shared by its two
 * executors {@link GpuWavTokenizer} and {@link CpuWavTokenizer}, which implement {@link Backend}.
 *
 * <p>The decoder maps codec codes to STFT features (llama.cpp {@code t_embd}, wavtokenizer-dec.cpp:118-262):
 * <ol>
 *   <li>{@code token_embd} rows {@code [T][E]}, transposed to the channel-major signal {@code [E][T]};</li>
 *   <li>{@code conv1d} ({@code K = 7}, {@code E -> C}) plus bias;</li>
 *   <li>the posnet: ResNet blocks (layers 0, 1, 3, 4: GroupNorm, SiLU, conv K=3, GroupNorm, SiLU, conv K=3,
 *       residual), one single-head self-attention block (layer 2: GroupNorm, 1x1 q/k/v, softmax over keys with
 *       scale {@code 1/sqrt(C)}, 1x1 output, residual) and a final GroupNorm (layer 5);</li>
 *   <li>LayerNorm per token ({@code token_embd_norm});</li>
 *   <li>the ConvNeXt blocks: depthwise conv K=7, LayerNorm per token, {@code pw1} + bias, GELU (tanh form,
 *       {@code ggml_gelu}), {@code pw2} + bias, per-channel {@code gamma}, residual;</li>
 *   <li>LayerNorm per token ({@code output_norm}) and the {@code output} projection plus bias:
 *       {@code [T][nOut]} features.</li>
 * </ol>
 * The whole code sequence is one non-causal batch (llama-context.cpp:1914 requires {@code n_ubatch >= n_tokens}),
 * so an executor decodes at most its bind-time capacity of codes per call.
 *
 * <p>Tensor shapes follow the loader (wavtokenizer-dec.cpp:9-113) exactly, with ggml dimension order
 * ({@code ne[0]} first); a mismatch fails at construction with the tensor name. Immutable.
 */
public final class WavTokenizerDecoder {

    /** Executes the decoder on one device. */
    public interface Backend extends AutoCloseable {

        /**
         * Decodes one code sequence.
         *
         * @param codes codec codes, each in {@code [0, nCodes())}
         * @return {@code [codes.length][nOut()]} STFT feature rows (fresh arrays owned by the caller)
         * @throws IllegalArgumentException when the sequence is empty, longer than the capacity or holds a code
         *                                  outside the codebook
         */
        float[][] decode(int[] codes);

        /** Largest code sequence one call decodes. */
        int capacity();

        /** The geometry this executor was bound to. */
        WavTokenizerDecoder decoder();

        @Override
        void close();
    }

    /** Role of a posnet layer (wavtokenizer-dec.cpp:33-73). */
    public enum PosnetKind {
        /** GroupNorm, SiLU, conv, GroupNorm, SiLU, conv, residual. */
        RESNET,
        /** GroupNorm and single-head self-attention with 1x1 projections, residual. */
        ATTENTION,
        /** GroupNorm only (tensor name {@code attn_norm}). */
        NORM
    }

    /** Kernel width of {@code conv1d} (wavtokenizer-dec.cpp:14). */
    public static final int INPUT_CONV_K = 7;
    /** Kernel width of the posnet ResNet convolutions (wavtokenizer-dec.cpp:42, 48). */
    public static final int POSNET_CONV_K = 3;
    /** Kernel width of the ConvNeXt depthwise convolutions (wavtokenizer-dec.cpp:90). */
    public static final int CONVNEXT_DW_K = 7;
    /** Posnet layers llama.cpp defines (wavtokenizer-dec.cpp:73 aborts beyond them). */
    public static final int MAX_POSNET_LAYERS = 6;
    /** Posnet layer that holds the attention block. */
    public static final int POSNET_ATTENTION_LAYER = 2;

    private final int nCodes;
    private final int nEmbd;
    private final int width;
    private final int nFf;
    private final int nOut;
    private final int posnetLayers;
    private final int convnextLayers;
    private final int nGroups;
    private final float groupEps;
    private final float normEps;

    private WavTokenizerDecoder(int nCodes, int nEmbd, int width, int nFf, int nOut, int posnetLayers,
                                int convnextLayers, int nGroups, float groupEps, float normEps) {
        this.nCodes = nCodes;
        this.nEmbd = nEmbd;
        this.width = width;
        this.nFf = nFf;
        this.nOut = nOut;
        this.posnetLayers = posnetLayers;
        this.convnextLayers = convnextLayers;
        this.nGroups = nGroups;
        this.groupEps = groupEps;
        this.normEps = normEps;
    }

    /**
     * Geometry of a resolved wavtokenizer-dec model.
     *
     * @throws IllegalArgumentException when the model is another architecture
     * @throws GgufFormatException      when a tensor is missing, a shape differs from the loader's, or the
     *                                  hparams disagree with the tensors
     */
    public static WavTokenizerDecoder of(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        return of(spec.hparams(), spec.tensors());
    }

    /**
     * Geometry from hparams and tensors: the tensors fix every width, the hparams supply the layer counts and
     * norm parameters and must agree with the tensor widths ({@code features_length}, {@code embedding_length},
     * posnet and ConvNeXt widths, and {@code feed_forward_length} when there are ConvNeXt layers, which the loader
     * checks against {@code pw1}, wavtokenizer-dec.cpp:96).
     *
     * @throws IllegalArgumentException when {@code hp} is another architecture
     * @throws GgufFormatException      as {@link #fromTensors}, or when the hparams disagree with the tensors
     */
    public static WavTokenizerDecoder of(ModelHparams hp, TensorCatalog tc) {
        Objects.requireNonNull(hp, "hparams");
        if (hp.arch() != Arch.WAVTOKENIZER_DEC) {
            throw new IllegalArgumentException("the wavtokenizer decoder needs a " + Arch.WAVTOKENIZER_DEC.ggufName()
                    + " model, got " + hp.arch().ggufName());
        }
        WavTokenizerDecoder d = fromTensors(tc, hp.posnetLayers(), hp.convnextLayers(), hp.nNormGroups(),
                hp.normGroupEps(), hp.normEps());
        agree("features_length", hp.nEmbd(), d.nEmbd);
        agree("embedding_length", hp.nEmbdOut(), d.nOut);
        agree("posnet.embedding_length", hp.posnetEmbd(), d.width);
        agree("convnext.embedding_length", hp.convnextEmbd(), d.width);
        if (d.convnextLayers > 0) {
            agree("feed_forward_length", hp.nFf(0), d.nFf);
        }
        return d;
    }

    /**
     * Geometry from the tensors alone, with the layer counts and norm parameters given explicitly.
     *
     * @param posnetLayers   {@code posnet.block_count}, at most {@value #MAX_POSNET_LAYERS}
     * @param convnextLayers {@code convnext.block_count}
     * @param nGroups        {@code attention.group_norm_groups}
     * @param groupEps       {@code attention.group_norm_epsilon}
     * @param normEps        {@code attention.layer_norm_epsilon}
     * @throws GgufFormatException when a count or norm parameter is invalid, a tensor is missing or a shape differs
     *                             from the loader's
     */
    public static WavTokenizerDecoder fromTensors(TensorCatalog tc, int posnetLayers, int convnextLayers,
                                                  int nGroups, float groupEps, float normEps) {
        Objects.requireNonNull(tc, "tensors");
        if (posnetLayers < 0 || posnetLayers > MAX_POSNET_LAYERS) {
            throw new GgufFormatException("wavtokenizer-dec: posnet block count " + posnetLayers + " is outside [0, "
                    + MAX_POSNET_LAYERS + "] (wavtokenizer-dec.cpp:73)");
        }
        if (convnextLayers < 0) {
            throw new GgufFormatException("wavtokenizer-dec: convnext block count " + convnextLayers + " is negative");
        }
        if (nGroups <= 0) {
            throw new GgufFormatException("wavtokenizer-dec: group norm group count must be positive, got " + nGroups);
        }
        checkEps("group norm", groupEps);
        checkEps("layer norm", normEps);

        TensorRef tokEmbd = tc.require(TensorNames.weight(TensorNames.TOKEN_EMBD));
        int nEmbd = dim(tokEmbd, 0);
        int nCodes = dim(tokEmbd, 1);
        TensorRef conv = tc.require(TensorNames.weight(TensorNames.CONV1D));
        int width = dim(conv, 2);
        shape(conv, INPUT_CONV_K, nEmbd, width);
        shape(tc.require(TensorNames.bias(TensorNames.CONV1D)), 1, width);

        for (int il = 0; il < posnetLayers; il++) {
            checkPosnetLayer(tc, il, width);
        }
        shape(tc.require(TensorNames.weight(TensorNames.TOKEN_EMBD_NORM)), width);
        shape(tc.require(TensorNames.bias(TensorNames.TOKEN_EMBD_NORM)), width);

        int nFf = -1;
        for (int il = 0; il < convnextLayers; il++) {
            int layerFf = checkConvnextLayer(tc, il, width);
            if (nFf >= 0 && layerFf != nFf) {
                throw new GgufFormatException("wavtokenizer-dec: convnext layer " + il + " has " + layerFf
                        + " hidden units, layer 0 has " + nFf + " (llama.cpp uses one n_ff)");
            }
            nFf = layerFf;
        }
        shape(tc.require(TensorNames.weight(TensorNames.OUTPUT_NORM)), width);
        shape(tc.require(TensorNames.bias(TensorNames.OUTPUT_NORM)), width);
        TensorRef output = tc.require(TensorNames.weight(TensorNames.OUTPUT));
        int nOut = dim(output, 1);
        shape(output, width, nOut);
        shape(tc.require(TensorNames.bias(TensorNames.OUTPUT)), nOut);
        return new WavTokenizerDecoder(nCodes, nEmbd, width, Math.max(nFf, 0), nOut, posnetLayers, convnextLayers,
                nGroups, groupEps, normEps);
    }

    private static void checkPosnetLayer(TensorCatalog tc, int il, int c) {
        switch (posnetKind(il)) {
            case RESNET:
                vector(tc, posnet(il, TensorNames.POS_NET_NORM1, false), c);
                vector(tc, posnet(il, TensorNames.POS_NET_NORM1, true), c);
                shape(tc.require(posnet(il, TensorNames.POS_NET_CONV1, false)), POSNET_CONV_K, c, c);
                vector(tc, posnet(il, TensorNames.POS_NET_CONV1, true), c);
                vector(tc, posnet(il, TensorNames.POS_NET_NORM2, false), c);
                vector(tc, posnet(il, TensorNames.POS_NET_NORM2, true), c);
                shape(tc.require(posnet(il, TensorNames.POS_NET_CONV2, false)), POSNET_CONV_K, c, c);
                vector(tc, posnet(il, TensorNames.POS_NET_CONV2, true), c);
                return;
            case ATTENTION:
                vector(tc, posnet(il, TensorNames.POS_NET_ATTN_NORM, false), c);
                vector(tc, posnet(il, TensorNames.POS_NET_ATTN_NORM, true), c);
                for (String p : new String[]{TensorNames.POS_NET_ATTN_Q, TensorNames.POS_NET_ATTN_K,
                        TensorNames.POS_NET_ATTN_V, TensorNames.POS_NET_ATTN_OUT}) {
                    shape(tc.require(posnet(il, p, false)), 1, c, c);
                    vector(tc, posnet(il, p, true), c);
                }
                return;
            case NORM:
                vector(tc, posnet(il, TensorNames.POS_NET_ATTN_NORM, false), c);
                vector(tc, posnet(il, TensorNames.POS_NET_ATTN_NORM, true), c);
                return;
            default:
                throw new IllegalStateException("unhandled posnet kind at layer " + il);
        }
    }

    /** Validates one ConvNeXt layer and returns its hidden width. */
    private static int checkConvnextLayer(TensorCatalog tc, int il, int c) {
        shape(tc.require(convnext(il, TensorNames.CONVNEXT_DW, false)), CONVNEXT_DW_K, 1, c);
        shape(tc.require(convnext(il, TensorNames.CONVNEXT_DW, true)), 1, c);
        shape(tc.require(convnext(il, TensorNames.CONVNEXT_NORM, false)), c);
        shape(tc.require(convnext(il, TensorNames.CONVNEXT_NORM, true)), c);
        TensorRef pw1 = tc.require(convnext(il, TensorNames.CONVNEXT_PW1, false));
        int ff = dim(pw1, 1);
        shape(pw1, c, ff);
        shape(tc.require(convnext(il, TensorNames.CONVNEXT_PW1, true)), ff);
        shape(tc.require(convnext(il, TensorNames.CONVNEXT_PW2, false)), ff, c);
        shape(tc.require(convnext(il, TensorNames.CONVNEXT_PW2, true)), c);
        shape(tc.require(convnext(il, TensorNames.CONVNEXT_GAMMA, false)), c);
        return ff;
    }

    // ------------------------------------------------------------------ names

    /**
     * Role of posnet layer {@code il} (wavtokenizer-dec.cpp:33-73).
     *
     * @throws IllegalArgumentException outside {@code [0, MAX_POSNET_LAYERS)}
     */
    public static PosnetKind posnetKind(int il) {
        switch (il) {
            case 0:
            case 1:
            case 3:
            case 4:
                return PosnetKind.RESNET;
            case POSNET_ATTENTION_LAYER:
                return PosnetKind.ATTENTION;
            case 5:
                return PosnetKind.NORM;
            default:
                throw new IllegalArgumentException("unknown posnet layer " + il + " (wavtokenizer-dec.cpp:73)");
        }
    }

    /** {@code posnet.<il>.<base>.weight} or {@code .bias}. */
    public static String posnet(int il, String base, boolean bias) {
        return TensorNames.layerName(TensorNames.STACK_POSNET, il,
                bias ? TensorNames.bias(base) : TensorNames.weight(base));
    }

    /** {@code convnext.<il>.<base>.weight} or {@code .bias}. */
    public static String convnext(int il, String base, boolean bias) {
        return TensorNames.layerName(TensorNames.STACK_CONVNEXT, il,
                bias ? TensorNames.bias(base) : TensorNames.weight(base));
    }

    // ------------------------------------------------------------------ calls

    /**
     * Validates one code sequence for an executor of the given capacity.
     *
     * @throws IllegalArgumentException when the sequence is empty, longer than {@code capacity}, or holds a code
     *                                  outside {@code [0, nCodes())}
     */
    public void checkCodes(int[] codes, int capacity) {
        Objects.requireNonNull(codes, "codes");
        if (codes.length == 0) {
            throw new IllegalArgumentException("wavtokenizer-dec needs at least one code");
        }
        if (codes.length > capacity) {
            throw new IllegalArgumentException(codes.length + " codes exceed the decoder capacity of " + capacity
                    + ": the decoder is non-causal and needs the whole sequence in one batch (llama-context.cpp:1914),"
                    + " raise neuralfree.ubatch");
        }
        for (int t = 0; t < codes.length; t++) {
            if (codes[t] < 0 || codes[t] >= nCodes) {
                throw new IllegalArgumentException("code " + codes[t] + " at " + t + " is outside the codebook [0, "
                        + nCodes + ")");
            }
        }
    }

    /**
     * Validates a bind-time capacity: positive, with every {@code [T][width]} intermediate inside the 32-bit
     * element range of the kernels.
     *
     * @throws IllegalArgumentException otherwise
     */
    public void checkCapacity(int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("decoder capacity must be positive, got " + capacity);
        }
        long widest = Math.max(Math.max(nEmbd, width), Math.max(nFf, nOut));
        if (widest * capacity > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("decoder capacity " + capacity + " x width " + widest
                    + " exceeds the 32-bit element range");
        }
    }

    /** Codebook size ({@code token_embd} rows). */
    public int nCodes() { return nCodes; }
    /** Code embedding width {@code E} ({@code features_length}). */
    public int nEmbd() { return nEmbd; }
    /** Posnet and ConvNeXt width {@code C}. */
    public int width() { return width; }
    /** ConvNeXt hidden width ({@code pw1} rows); 0 without ConvNeXt layers. */
    public int nFf() { return nFf; }
    /** Feature width ({@code embedding_length}, {@code output} rows). */
    public int nOut() { return nOut; }
    public int posnetLayers() { return posnetLayers; }
    public int convnextLayers() { return convnextLayers; }
    /** GroupNorm group count ({@code n_norm_groups}). */
    public int nGroups() { return nGroups; }
    /** GroupNorm epsilon ({@code f_norm_group_eps}). */
    public float groupEps() { return groupEps; }
    /** LayerNorm epsilon ({@code f_norm_eps}). */
    public float normEps() { return normEps; }

    /** Posnet attention softmax scale {@code 1/sqrt(C)} in float arithmetic (wavtokenizer-dec.cpp:187). */
    public float attentionScale() {
        return 1.0f / (float) Math.sqrt(width);
    }

    @Override
    public String toString() {
        return "WavTokenizerDecoder{codes=" + nCodes + ", E=" + nEmbd + ", C=" + width + ", ff=" + nFf + ", out="
                + nOut + ", posnet=" + posnetLayers + ", convnext=" + convnextLayers + ", groups=" + nGroups + '}';
    }

    // ------------------------------------------------------------------ validation helpers

    private static void agree(String key, int hparam, int tensors) {
        if (hparam != tensors) {
            throw new GgufFormatException("wavtokenizer-dec: " + key + " is " + hparam + " but the tensors give "
                    + tensors);
        }
    }

    private static void checkEps(String what, float eps) {
        if (!(eps >= 0f) || Float.isInfinite(eps)) {
            throw new GgufFormatException("wavtokenizer-dec: " + what + " epsilon must be finite and non-negative, got "
                    + eps);
        }
    }

    private static void vector(TensorCatalog tc, String name, int c) {
        shape(tc.require(name), 1, c);
    }

    /** {@code ne[i]} of a tensor as a positive int. */
    private static int dim(TensorRef r, int i) {
        long v = r.ne(i);
        if (v <= 0 || v > Integer.MAX_VALUE) {
            throw new GgufFormatException("wavtokenizer-dec: tensor " + r.name() + " has dimension " + i + " = " + v);
        }
        return (int) v;
    }

    /** The llama.cpp loader's exact shape check: listed dims match, every later dim is 1. */
    private static void shape(TensorRef r, long... ne) {
        boolean ok = trailingOnes(r, ne.length);
        for (int i = 0; i < ne.length && ok; i++) {
            ok = r.ne(i) == ne[i];
        }
        if (!ok) {
            StringBuilder want = new StringBuilder("{");
            for (int i = 0; i < ne.length; i++) {
                want.append(i == 0 ? "" : ", ").append(ne[i]);
            }
            throw new GgufFormatException("wavtokenizer-dec: tensor " + r.name() + " has shape " + r
                    + ", the loader expects " + want.append('}'));
        }
    }

    private static boolean trailingOnes(TensorRef r, int from) {
        for (int i = from; i < r.rank(); i++) {
            if (r.ne(i) != 1) {
                return false;
            }
        }
        return true;
    }
}
