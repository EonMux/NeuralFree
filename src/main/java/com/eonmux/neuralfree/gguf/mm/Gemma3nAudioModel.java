package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;

import static com.eonmux.neuralfree.gguf.mm.ClipShapes.shape;

/**
 * Validated geometry of the Gemma 3n audio encoder ({@code clip.audio.projector_type gemma3na}): the Universal
 * Speech Model of Hugging Face transformers {@code Gemma3nAudioEncoder} and the {@code Gemma3nMultimodalEmbedder}
 * that maps its rows into the text model (models/gemma3n/modeling_gemma3n.py). llama.cpp declares the type
 * without a graph, so the tensor names are those its GGUF tensor map gives the Hugging Face checkpoint:
 * <ul>
 *   <li>the sub-sample conv projection: {@code a.conv1d.{0,1}.weight} (3x3, stride 2, no bias) with their
 *       cumulative group norm scales {@code a.conv1d.{0,1}.norm.weight}, and {@code a.pre_encode.out.weight} over
 *       the {@code [freq x channel]} flatten;</li>
 *   <li>per conformer block: the start feed-forward ({@code ffn_norm}, {@code ffn_up}, {@code ffn_down},
 *       {@code ffn_post_norm}), the chunked local attention ({@code ln1}, {@code attn_q/k/v}, {@code per_dim_scale},
 *       {@code linear_pos}, {@code attn_out}, {@code ln2}), the light conv ({@code conv_norm}, {@code conv_pw1},
 *       {@code conv_dw}, {@code norm_conv}, {@code conv_pw2}), the end feed-forward (the {@code _1} tensors) and
 *       {@code layer_pre_norm};</li>
 *   <li>the embedder: {@code mm.a.soft_emb_norm}, {@code mm.a.input_projection}, and the hard-token path
 *       {@code mm.a.embedding} / {@code mm.a.hard_emb_norm} that gives the padding row.</li>
 * </ul>
 * The channel counts, FFN width, light-conv kernel and embedding vocabulary come from the tensor shapes; the
 * attention chunking, logit cap, strides, norm epsilons and reduction are the {@code Gemma3nAudioConfig} defaults,
 * which the converter does not write. Every tensor is resolved by name and shape-checked; a mismatch fails at
 * construction with the tensor named. Immutable.
 */
public final class Gemma3nAudioModel {

    /** Rows one audio window occupies in the prompt ({@code audio_soft_tokens_per_image}). */
    public static final int TOKENS_PER_WINDOW = 188;
    /** Queries per attention block ({@code conf_attention_chunk_size}). */
    static final int ATTN_CHUNK = 12;
    /** Keys before the query a block attends ({@code conf_attention_context_left - 1}). */
    static final int MAX_PAST = 12;
    /** Attention logit soft cap ({@code conf_attention_logit_cap}). */
    static final float LOGIT_CAP = 50f;
    /** Every n-th conformer row reaches the embedder ({@code conf_reduction_factor}). */
    static final int REDUCTION = 4;
    /** Kernel and stride of both sub-sample convolutions ({@code sscp_conv_kernel_size / stride_size}). */
    static final int SSCP_KERNEL = 3;
    static final int SSCP_STRIDE = 2;
    /** Epsilon of the cumulative group norms ({@code sscp_conv_group_norm_eps}). */
    static final float GROUP_NORM_EPS = 1e-3f;
    /** Weight of each feed-forward branch on the residual ({@code conf_residual_weight}). */
    static final float RESIDUAL_WEIGHT = 0.5f;
    /** Clamp applied around the conformer sub-layers ({@code gradient_clipping}). */
    static final float GRADIENT_CLIP = 1e10f;

    private final ClipHparams hparams;
    private final int nMel;
    private final int c0;
    private final int c1;
    private final int f0;
    private final int f1;
    private final int ff;
    private final int convKernel;
    private final int vocab;

    private Gemma3nAudioModel(ClipHparams hparams, int c0, int c1, int ff, int convKernel, int vocab) {
        this.hparams = hparams;
        this.nMel = hparams.audio().nMelBins();
        this.c0 = c0;
        this.c1 = c1;
        this.f0 = convOut(nMel);
        this.f1 = convOut(f0);
        this.ff = ff;
        this.convKernel = convKernel;
        this.vocab = vocab;
    }

    /**
     * The geometry of {@code hp} over the tensors of {@code tc}.
     *
     * @throws IllegalArgumentException for a hidden width the relative position signal cannot split (odd) or the
     *                                  heads cannot divide, or a tensor shape that differs from the graph's
     * @throws MissingTensorException   when a required tensor is absent
     */
    static Gemma3nAudioModel resolve(ClipHparams hp, TensorCatalog tc) {
        Objects.requireNonNull(hp, "hparams");
        Objects.requireNonNull(tc, "tensors");
        int h = hp.nEmbd();
        if (h % 2 != 0 || h % hp.nHead() != 0) {
            throw new IllegalArgumentException("projector type 'gemma3na' needs an even clip.audio.embedding_length"
                    + " that clip.audio.attention.head_count divides, got " + h + " / " + hp.nHead());
        }
        TensorRef conv0 = tc.require("a.conv1d.0.weight");
        TensorRef conv1 = tc.require("a.conv1d.1.weight");
        int c0 = (int) conv0.ne(3);
        int c1 = (int) conv1.ne(3);
        shape(tc, conv0, SSCP_KERNEL, SSCP_KERNEL, 1, c0);
        shape(tc, conv1, SSCP_KERNEL, SSCP_KERNEL, c0, c1);
        shape(tc, tc.require("a.conv1d.0.norm.weight"), c0);
        shape(tc, tc.require("a.conv1d.1.norm.weight"), c1);
        int f1 = convOut(convOut(hp.audio().nMelBins()));
        shape(tc, tc.require("a.pre_encode.out.weight"), (long) f1 * c1, h);
        int ff = (int) tc.require(block(0, "ffn_up.weight")).ne(1);
        int kernel = (int) tc.require(block(0, "conv_dw.weight")).ne(0);
        for (int il = 0; il < hp.nLayer(); il++) {
            checkBlock(tc, hp, il, ff, kernel);
        }
        TensorRef embedding = tc.require("mm.a.embedding.weight");
        int vocab = (int) embedding.ne(1);
        shape(tc, embedding, h, vocab);
        shape(tc, tc.require("mm.a.hard_emb_norm.weight"), h);
        shape(tc, tc.require("mm.a.soft_emb_norm.weight"), h);
        shape(tc, tc.require("mm.a.input_projection.weight"), h, hp.projectionDim());
        return new Gemma3nAudioModel(hp, c0, c1, ff, kernel, vocab);
    }

    private static void checkBlock(TensorCatalog tc, ClipHparams hp, int il, int ff, int kernel) {
        int h = hp.nEmbd();
        for (String norm : new String[]{"ffn_norm", "ffn_post_norm", "ln1", "ln2", "conv_norm", "norm_conv",
                "ffn_norm_1", "ffn_post_norm_1", "layer_pre_norm"}) {
            shape(tc, tc.require(block(il, norm + ".weight")), h);
        }
        for (String suffix : new String[]{"", "_1"}) {
            shape(tc, tc.require(block(il, "ffn_up" + suffix + ".weight")), h, ff);
            shape(tc, tc.require(block(il, "ffn_down" + suffix + ".weight")), ff, h);
        }
        for (String square : new String[]{"attn_q", "attn_k", "attn_v", "attn_out", "linear_pos", "conv_pw2"}) {
            shape(tc, tc.require(block(il, square + ".weight")), h, h);
        }
        shape(tc, tc.require(block(il, "per_dim_scale")), h / hp.nHead());
        shape(tc, tc.require(block(il, "conv_pw1.weight")), h, 2L * h);
        shape(tc, tc.require(block(il, "conv_dw.weight")), kernel, 1, h);
    }

    /** The name of tensor {@code suffix} of conformer block {@code il}. */
    static String block(int il, String suffix) {
        return "a.blk." + il + "." + suffix;
    }

    /** The output size of a kernel-3, stride-2 convolution padded by one on each side. */
    static int convOut(int size) {
        return (size + 2 - SSCP_KERNEL) / SSCP_STRIDE + 1;
    }

    /**
     * Conformer rows of a mel window of {@code nFrames} frames: the two stride-2 convolutions halve the frames
     * (rounding up), the reduction keeps every fourth row.
     */
    static int encoderRows(int nFrames) {
        int t = convOut(convOut(nFrames));
        return (t + REDUCTION - 1) / REDUCTION;
    }

    /** The hparams the geometry was resolved from. */
    ClipHparams hparams() {
        return hparams;
    }

    /** Mel bins of the input ({@code clip.audio.num_mel_bins}). */
    int nMel() {
        return nMel;
    }

    /** Output channels of the first sub-sample convolution. */
    int c0() {
        return c0;
    }

    /** Output channels of the second sub-sample convolution. */
    int c1() {
        return c1;
    }

    /** Frequency bins after the first sub-sample convolution. */
    int f0() {
        return f0;
    }

    /** Frequency bins after the second sub-sample convolution. */
    int f1() {
        return f1;
    }

    /** Width of the conformer feed-forward blocks. */
    int ff() {
        return ff;
    }

    /** Taps of the causal depthwise convolution of the light-conv module. */
    int convKernel() {
        return convKernel;
    }

    /** Rows of the hard-token embedding table; the last one is the padding token. */
    int vocab() {
        return vocab;
    }

    @Override
    public String toString() {
        return "Gemma3nAudioModel{n_mel=" + nMel + ", sscp=" + c0 + "/" + c1 + ", hidden=" + hparams.nEmbd()
                + ", heads=" + hparams.nHead() + ", layers=" + hparams.nLayer() + ", ff=" + ff + ", conv_k="
                + convKernel + ", vocab=" + vocab + ", out=" + hparams.projectionDim() + "}";
    }
}
