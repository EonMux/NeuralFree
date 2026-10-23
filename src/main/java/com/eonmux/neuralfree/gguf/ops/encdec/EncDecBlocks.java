package com.eonmux.neuralfree.gguf.ops.encdec;

import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec.BiasMode;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;

/**
 * Block factories of the encoder-decoder family, exposed to the block catalogs: the T5 decoder cross-attention of
 * {@link MixerKind#CROSS_ATTENTION_T5} layers ({@link GpuCrossAttentionMixer}, {@link CpuCrossAttentionMixer}). The
 * family has no FFN blocks and no layer extras; the T5 encoder stack and the decoder self-attention run on the plain
 * attention mixer with the bias table of {@link T5RelativeBias}.
 *
 * <p>The T5 decoder layer (llama.cpp src/models/t5.cpp:133-215) is, with {@code x} the layer input:
 * <pre>
 *   sa  = wo   @ attn(Q = wq @ rms(x),  K/V = self cache,  T5 causal bias)       t5.cpp:136-152
 *   h   = x + sa                                                                  t5.cpp:154
 *   ca  = wo_c @ attn(Q = wq_c @ rms_c(h), K/V = wk_c/wv_c @ encoder output)     t5.cpp:160-182
 *   y   = h + ca, then the FFN                                                    t5.cpp:212
 * </pre>
 * The layer executor runs the self-attention through the ATTENTION factory on the layer's attention view, adds its
 * residual, applies {@code cross_attn_norm} and calls the {@link MixerKind#CROSS_ATTENTION_T5} mixer with
 * {@code in = rms_c(h)}; the mixer writes {@code ca} to {@code out} and the executor adds the residual. The cross K/V
 * are computed once per prompt from the encoder output by {@code encode(...)} into the layer's cross cache
 * ({@code GpuMemory.crossKv}, {@code CpuMemory.crossKv}).</p>
 */
public final class EncDecBlocks {

    /** CPU int temporary holding the query positions of a step. */
    static final String CPU_TEMP_QPOS = "encdec.q_pos";

    static final String W_CROSS_Q = TensorNames.weight(TensorNames.DEC_CROSS_ATTN_Q);
    static final String W_CROSS_K = TensorNames.weight(TensorNames.DEC_CROSS_ATTN_K);
    static final String W_CROSS_V = TensorNames.weight(TensorNames.DEC_CROSS_ATTN_V);
    static final String W_CROSS_O = TensorNames.weight(TensorNames.DEC_CROSS_ATTN_OUT);

    private EncDecBlocks() {
    }

    // ================================================================== factory exposure

    /** GPU mixer of {@link MixerKind#CROSS_ATTENTION_T5}. */
    public static List<GpuMixerFactory> gpuMixers() {
        return Collections.singletonList(gpu(GpuCrossAttentionMixer::new));
    }

    /** CPU mixer of {@link MixerKind#CROSS_ATTENTION_T5}. */
    public static List<CpuMixerFactory> cpuMixers() {
        return Collections.singletonList(cpu(CpuCrossAttentionMixer::new));
    }

    /** The encoder-decoder family has no FFN blocks. */
    public static List<GpuFfnFactory> gpuFfns() {
        return Collections.emptyList();
    }

    /** The encoder-decoder family has no FFN blocks. */
    public static List<CpuFfnFactory> cpuFfns() {
        return Collections.emptyList();
    }

    /** The encoder-decoder family has no layer extras. */
    public static List<GpuLayerExtraFactory> gpuExtras() {
        return Collections.emptyList();
    }

    /** The encoder-decoder family has no layer extras. */
    public static List<CpuLayerExtraFactory> cpuExtras() {
        return Collections.emptyList();
    }

    private static GpuMixerFactory gpu(Supplier<GpuMixer> ctor) {
        return new GpuMixerFactory() {
            @Override
            public MixerKind kind() {
                return MixerKind.CROSS_ATTENTION_T5;
            }

            @Override
            public GpuMixer create(LayerSpec layer) {
                checkLayer(layer);
                return ctor.get();
            }

            @Override
            public String toString() {
                return "EncDecBlocks.gpu(" + MixerKind.CROSS_ATTENTION_T5 + ")";
            }
        };
    }

    private static CpuMixerFactory cpu(Supplier<CpuMixer> ctor) {
        return new CpuMixerFactory() {
            @Override
            public MixerKind kind() {
                return MixerKind.CROSS_ATTENTION_T5;
            }

            @Override
            public CpuMixer create(LayerSpec layer) {
                checkLayer(layer);
                return ctor.get();
            }

            @Override
            public String toString() {
                return "EncDecBlocks.cpu(" + MixerKind.CROSS_ATTENTION_T5 + ")";
            }
        };
    }

    // ================================================================== layer checks

    /**
     * Checks that a layer is a T5 decoder layer whose cross-attention this family runs: kind
     * {@link MixerKind#CROSS_ATTENTION_T5} with a cached, bidirectional cross-attention without position bias, RoPE or
     * a feature of other architectures (Q/K norm, gates, sinks, softcap, temperature, clamping, value scale, V norm,
     * fused QKV, sliding window, value experts). The decoder self-attention is checked by the attention factory that
     * runs it.
     *
     * @throws IllegalArgumentException      when the layer is not a T5 decoder layer
     * @throws UnsupportedOperationException when the cross-attention uses a feature T5 does not have
     */
    static void checkLayer(LayerSpec layer) {
        if (layer == null || layer.mixerKind() != MixerKind.CROSS_ATTENTION_T5 || layer.crossAttention() == null) {
            throw new IllegalArgumentException(MixerKind.CROSS_ATTENTION_T5 + " mixer cannot run layer "
                    + (layer == null ? "null" : describe(layer) + " (" + layer.mixerKind() + ")"));
        }
        AttentionSpec cross = layer.crossAttention();
        checkPlain(layer, "cross-attention", cross);
        if (cross.biasMode() != BiasMode.NONE || cross.causal() || !cross.usesMemory()) {
            throw unsupported(layer, "cross-attention with bias " + cross.biasMode() + ", causal " + cross.causal()
                    + ", usesMemory " + cross.usesMemory() + " (T5 cross-attention is unbiased and bidirectional"
                    + " over the cached encoder output)");
        }
    }

    private static void checkPlain(LayerSpec layer, String what, AttentionSpec a) {
        if (a.linearReplace() || a.fusedQkv() || a.valueExperts() != null) {
            throw unsupported(layer, what + " with a fused, linear-replace or expert projection");
        }
        if (!a.noRope()) {
            throw unsupported(layer, what + " with RoPE");
        }
        if (a.qkNorm() != QkNormKind.NONE || a.vNorm() || a.gate() != AttnGateKind.NONE) {
            throw unsupported(layer, what + " with Q/K norm " + a.qkNorm() + ", V norm " + a.vNorm() + " or gate "
                    + a.gate());
        }
        if (a.sinks() || a.softcap() != 0f || a.temperature().enabled() || a.clampKqv() != 0f
                || a.valueScale() != 0f) {
            throw unsupported(layer, what + " with sinks, softcap, temperature, KQV clamp or value scale");
        }
        if (a.effectiveSwaType() != SwaType.NONE) {
            throw unsupported(layer, what + " with a sliding window " + a.effectiveSwaType());
        }
    }

    private static UnsupportedOperationException unsupported(LayerSpec layer, String what) {
        return new UnsupportedOperationException("T5 decoder layer " + describe(layer) + ": " + what
                + " is not part of T5 cross-attention");
    }

    /** Layer name used in messages. */
    static String describe(LayerSpec layer) {
        return layer.stack() + "." + layer.il();
    }

    /**
     * Checks a projection's shape: {@code cols} inputs and {@code rows} outputs.
     *
     * @throws IllegalStateException naming the tensor when it differs
     */
    static void checkMatrix(LayerSpec layer, String name, long cols, long rows, long wantCols, long wantRows) {
        if (cols != wantCols || rows != wantRows) {
            throw new IllegalStateException("T5 decoder layer " + describe(layer) + ": tensor " + name + " is "
                    + cols + " x " + rows + " (in x out), expected " + wantCols + " x " + wantRows);
        }
    }

    /**
     * Narrows a size to the kernels' {@code int} range.
     *
     * @throws IllegalStateException naming {@code what} when it does not fit
     */
    static int toInt(long v, String what) {
        if (v < 0 || v > Integer.MAX_VALUE) {
            throw new IllegalStateException(what + " = " + v + " is outside the kernels' 32-bit range");
        }
        return (int) v;
    }
}
