package com.eonmux.neuralfree.gguf.ops.mla;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.ops.attn.AttnExtMath;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MlaSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Block factories of the multi-head latent attention family: the MLA mixer of deepseek2,
 * deepseek2-ocr's MLA files, mistral4, minicpm3, plm, kimi-linear, kimi-k3, bailingmoe3, the sliding-window layers of
 * dots3note, hy_v4 checkpoints without a lightning indexer, the dense NextN blocks of deepseek32 and glm-dsa
 * (llama-model.cpp:2830-2836, glm-dsa.cpp:519-526) and the other {@link MixerKind#MLA} layers ({@link GpuMlaMixer},
 * {@link CpuMlaMixer}). The family has no FFN blocks and no layer extras.
 *
 * <p>Both twins follow the llama.cpp MLA graph (models/deepseek2.cpp:493-620): Q from {@code attn_q} or the low-rank
 * {@code attn_q_a -> rms(attn_q_a_norm) -> attn_q_b}; {@code attn_kv_a_mqa} gives the latent {@code kv_cmpr}
 * (rms {@code attn_kv_a_norm}) and the shared rotary key {@code k_pe}. With a latent cache the K row is
 * {@code [kv_cmpr | k_pe]} and V aliases its first {@code kv_lora} values; {@code attn_k_b} is absorbed into Q per
 * head and {@code attn_v_b} expands the latent attention output per head. With a decompressed cache K and V are
 * stored per head ({@code [k_nope | k_pe]} and {@code v}). {@link Geometry} holds the shapes, strides and Hadamard
 * blocks both twins share.</p>
 */
public final class MlaBlocks {

    private MlaBlocks() {
    }

    // ================================================================== factory exposure

    /** GPU mixer of the {@link MixerKind#MLA} kind. */
    public static List<GpuMixerFactory> gpuMixers() {
        return Collections.singletonList(new GpuMixerFactory() {
            @Override
            public MixerKind kind() {
                return MixerKind.MLA;
            }

            @Override
            public GpuMixer create(LayerSpec layer) {
                checkKind(layer);
                String why = GpuMlaMixer.unsupported(layer);
                if (why != null) {
                    throw new UnsupportedOperationException(why);
                }
                return new GpuMlaMixer();
            }

            @Override
            public String toString() {
                return "MlaBlocks.gpu(MLA)";
            }
        });
    }

    /** CPU mixer of the {@link MixerKind#MLA} kind. */
    public static List<CpuMixerFactory> cpuMixers() {
        return Collections.singletonList(new CpuMixerFactory() {
            @Override
            public MixerKind kind() {
                return MixerKind.MLA;
            }

            @Override
            public CpuMixer create(LayerSpec layer) {
                checkKind(layer);
                String why = unsupportedCommon(layer);
                if (why != null) {
                    throw new UnsupportedOperationException(why);
                }
                return new CpuMlaMixer();
            }

            @Override
            public String toString() {
                return "MlaBlocks.cpu(MLA)";
            }
        });
    }

    /** The MLA family has no FFN blocks. */
    public static List<GpuFfnFactory> gpuFfns() {
        return Collections.emptyList();
    }

    /** The MLA family has no FFN blocks. */
    public static List<CpuFfnFactory> cpuFfns() {
        return Collections.emptyList();
    }

    /** The MLA family has no layer extras. */
    public static List<GpuLayerExtraFactory> gpuExtras() {
        return Collections.emptyList();
    }

    /** The MLA family has no layer extras. */
    public static List<CpuLayerExtraFactory> cpuExtras() {
        return Collections.emptyList();
    }

    // ================================================================== shared checks

    /**
     * @throws IllegalArgumentException when the layer is not an MLA layer
     */
    static void checkKind(LayerSpec layer) {
        Objects.requireNonNull(layer, "layer");
        if (layer.mixerKind() != MixerKind.MLA || layer.mla() == null) {
            throw new IllegalArgumentException("layer " + layer.il() + " is a " + layer.mixerKind()
                    + " layer, not an MLA layer");
        }
    }

    /**
     * Why neither twin can run this layer, or null. MLA graphs read no ALiBi or T5 bias, gate the output only through
     * their own {@code attn_gate} projection, normalize at most the shared rotary key ({@link #kPeNorm}), and the
     * decoders always keep their latent or per-head cache. Per-head attention sinks join the softmax as in every
     * {@code build_attn} (hy-v4.cpp:325-327).
     */
    static String unsupportedCommon(LayerSpec layer) {
        AttentionSpec a = layer.attention();
        if (!layer.usesKv()) {
            return "layer " + layer.il() + ": MLA without KV memory has no implementation; the MLA graphs are "
                    + "decoders that always cache K/V";
        }
        if (a != null && a.biasMode() != AttentionSpec.BiasMode.NONE) {
            return "layer " + layer.il() + ": MLA attention with a " + a.biasMode() + " position bias has no "
                    + "implementation";
        }
        if (gateKind(layer) == AttnGateKind.Q_INTERLEAVED) {
            return "layer " + layer.il() + ": an MLA layer cannot carry a gate interleaved with Q";
        }
        return kPeNormRefusal(layer);
    }

    /**
     * True when the layer RMS-normalizes the shared rotary key {@code k_pe} with {@code attn_k_norm}
     * ({@code n_rot} values, eps {@code f_norm_rms_eps}) before its RoPE: the only Q/K norm of the MLA graphs,
     * present on every dots3note layer, sliding-window and DSA alike (models/dots3note.cpp:90-91, 330-332). The MLA
     * attention facet records it as an {@link QkNormKind#RMS_HEAD} norm applied before RoPE. The graph passes no bias
     * to this norm, so the facet's {@link AttentionSpec#qkNormBias()} (add a bias the file carries) has nothing to
     * add. Shared by the MLA and DSA sparse-MLA mixers.
     *
     * @throws UnsupportedOperationException for any other Q/K norm on an MLA layer
     */
    public static boolean kPeNorm(LayerSpec layer) {
        Objects.requireNonNull(layer, "layer");
        String why = kPeNormRefusal(layer);
        if (why != null) {
            throw new UnsupportedOperationException(why);
        }
        AttentionSpec a = layer.attention();
        return a != null && a.qkNorm() == QkNormKind.RMS_HEAD;
    }

    /** Why the layer's Q/K norm is not the MLA {@code k_pe} norm, or null when it is that norm or none. */
    private static String kPeNormRefusal(LayerSpec layer) {
        AttentionSpec a = layer.attention();
        if (a == null || a.qkNorm() == QkNormKind.NONE) {
            return null;
        }
        String where = "layer " + layer.il() + ": ";
        if (a.qkNorm() != QkNormKind.RMS_HEAD) {
            return where + "an MLA layer normalizes only its shared rotary key (RMS with attn_k_norm), not a "
                    + a.qkNorm() + " Q/K norm";
        }
        if (a.qkNormAfterRope()) {
            return where + "the MLA k_pe norm runs before RoPE (dots3note.cpp:330-332), the layer asks for it after"
                    + " RoPE";
        }
        if (layer.mla().headKRope() <= 0) {
            return where + "the MLA k_pe norm needs rotary dims, the layer has none";
        }
        return null;
    }

    /** True when the layer adds its per-head {@code attn_sinks} logits to the softmax denominator. */
    static boolean sinks(LayerSpec layer) {
        AttentionSpec a = layer.attention();
        return a != null && a.sinks();
    }

    /** Output gate of the layer ({@link AttnGateKind#NONE} without an attention facet). */
    static AttnGateKind gateKind(LayerSpec layer) {
        AttentionSpec a = layer.attention();
        return a == null ? AttnGateKind.NONE : a.gate();
    }

    /** Query temperature of the layer, or null when it has none (models/deepseek2.cpp:589-593). */
    static AttentionSpec.Temperature temperature(LayerSpec layer) {
        AttentionSpec a = layer.attention();
        return a != null && a.temperature().enabled() ? a.temperature() : null;
    }

    /**
     * RoPE of the layer's rotary dims, or null when the MLA layer applies none ({@link MlaSpec#ropeApplied()} false,
     * kimi-linear / kimi-k3).
     *
     * @throws IllegalStateException when the MLA spec applies RoPE but the layer carries no {@link RopeSpec}
     */
    static RopeSpec rope(LayerSpec layer) {
        if (!layer.mla().ropeApplied()) {
            return null;
        }
        RopeSpec r = layer.rope();
        if (r == null) {
            throw new IllegalStateException("layer " + layer.il() + ": the MLA layer applies RoPE to its "
                    + layer.mla().headKRope() + " rotary dims but the layer spec has no RoPE");
        }
        return r;
    }

    /** Logit softcap of the layer (0 without an attention facet). */
    static float softcap(LayerSpec layer) {
        AttentionSpec a = layer.attention();
        return a == null ? 0.0f : a.softcap();
    }

    /** Gate values per token of a gated layer: one per head, or one per output element. */
    static int gateWidth(LayerSpec layer) {
        MlaSpec m = layer.mla();
        return AttnExtMath.gateWidth(gateKind(layer), m.nHead(), m.headV());
    }

    /**
     * First cache row a query at {@code pos} may attend: 0 for a ring (visibility comes from the stored positions)
     * or a cache without a window, else the window's first visible position (rows hold their own positions).
     */
    static int kvStart(KvLayout layout, int pos) {
        KvWindow w = layout.window();
        return layout.ring() ? 0 : w.firstVisible(pos);
    }

    /**
     * One past the last cache row that may hold a key for queries up to {@code lastPos}.
     *
     * @throws IllegalArgumentException when a non-ring cache cannot hold {@code lastPos}
     */
    static int kvEnd(KvLayout layout, int lastPos) {
        if (lastPos < 0) {
            throw new IllegalArgumentException("query position " + lastPos + " is negative");
        }
        long end = lastPos + 1L;
        if (layout.ring()) {
            return (int) Math.min(layout.rows(), end);
        }
        if (end > layout.rows()) {
            throw new IllegalArgumentException("position " + lastPos + " is beyond the KV cache of " + layout.rows()
                    + " rows");
        }
        return (int) end;
    }

    /** Narrows a size to the kernels' {@code int} range. @throws IllegalStateException otherwise */
    static int toInt(long v, String what) {
        if (v < 0 || v > Integer.MAX_VALUE) {
            throw new IllegalStateException(what + " = " + v + " is outside the kernels' 32-bit range");
        }
        return (int) v;
    }

    /** Checks a projection's shape {@code [cols] -> [rows]}. @throws IllegalStateException otherwise */
    static void checkShape(String name, long cols, long rows, long wantCols, long wantRows) {
        if (cols != wantCols || rows != wantRows) {
            throw new IllegalStateException(name + " is " + cols + " x " + rows + " (cols x rows), the MLA layer "
                    + "needs " + wantCols + " x " + wantRows);
        }
    }

    /** Checks that a vector weight is F32 of {@code n} values. @throws IllegalStateException otherwise */
    static void checkF32Vector(String name, GGMLType type, long numElements, long n) {
        if (type != GGMLType.F32 || numElements != n) {
            throw new IllegalStateException(name + " must be an F32 vector of " + n + " values, got " + type + " with "
                    + numElements);
        }
    }

    // ================================================================== geometry

    /**
     * Shapes, cache strides, Hadamard blocks and context-shift geometry of one MLA layer bound to its cache, shared by
     * the GPU and CPU twins. Immutable.
     *
     * <ul>
     *   <li>Latent cache: query heads of {@code dQ = kv_lora + n_rot} values attend one KV head whose row is
     *       {@code [kv_cmpr | k_pe]}; V is the row's first {@code dV = kv_lora} values (llama-graph.cpp:3323-3366
     *       views V at offset 0 of K).</li>
     *   <li>Decompressed cache: {@code n_head} KV heads of K {@code [k_nope | k_pe]} ({@code dQ = head_k}) and V
     *       ({@code dV = head_v}).</li>
     *   <li>Hadamard rotation comes only from the layer's {@link RotationPlan} in the memory plan: Q and K are rotated
     *       in blocks of its K size, V in blocks of its V size, and the attention output is un-rotated
     *       (llama-graph.cpp:3225-3271). The planner leaves latent caches unrotated, as llama.cpp's K-only
     *       {@code build_attn} never rotates them (llama-graph.cpp:3294-3376), and rotates decompressed caches like
     *       any {@code build_attn_inp_kv} cache. A rotated latent row would rotate as a whole in K blocks that also
     *       tile the {@code kv_lora} prefix, so the aliased V and the attention output stay whole blocks.</li>
     *   <li>Context shift rotates the rotary dims {@code [nOffs, nOffs + n_rot)} of every K head
     *       (llama-kv-cache.cpp:2040-2068: the view at {@code n_embd_nope} of each head), undoing and re-applying the
     *       K rotation around the RoPE when the rows are rotated.</li>
     * </ul>
     */
    static final class Geometry {
        final int nHead;
        final int nope;
        final int rope;
        final int headK;
        final int headV;
        final int kvLora;
        final int qLora;
        /** {@code kv_lora + n_rot}: the rows of {@code attn_kv_a_mqa} and the latent cache row. */
        final int kvaWidth;
        /** {@code n_head * head_k}: one token's projected query. */
        final int qWidth;
        final boolean latent;
        final boolean absorbed;
        /** Query (and key) head width the attention reads. */
        final int dQ;
        /** Value head width the attention reads and writes. */
        final int dV;
        final int nHeadKv;
        final int kRowStride;
        final int kHeadStride;
        final int vRowStride;
        final int vHeadStride;
        /** Hadamard block applied to Q and K, 0 when they are stored unrotated. */
        final int rotK;
        /** Hadamard block applied to V and the attention output, 0 when unrotated. */
        final int rotV;
        final float kqScale;
        final int shiftHeads;
        final int shiftHeadStride;
        final int shiftRowStride;
        final int shiftOffs;

        private Geometry(MlaSpec m, RotationPlan rotation) {
            nHead = m.nHead();
            nope = m.headKNope();
            rope = m.headKRope();
            headK = m.headK();
            headV = m.headV();
            kvLora = m.kvLoraRank();
            qLora = m.qLoraRank();
            kvaWidth = m.latentRowWidth();
            qWidth = toInt((long) nHead * headK, "MLA query width");
            latent = m.latentCache();
            absorbed = m.absorbed();
            kqScale = m.kqScale();
            if (latent) {
                dQ = kvaWidth;
                dV = kvLora;
                nHeadKv = 1;
                kRowStride = kvaWidth;
                kHeadStride = kvaWidth;
                vRowStride = kvaWidth;
                vHeadStride = kvaWidth;
                rotK = rotation.nRotK();
                rotV = 0;
                shiftHeads = 1;
                shiftHeadStride = kvaWidth;
                shiftRowStride = kvaWidth;
                shiftOffs = kvLora;
            } else {
                dQ = headK;
                dV = headV;
                nHeadKv = nHead;
                kRowStride = qWidth;
                kHeadStride = headK;
                vRowStride = toInt((long) nHead * headV, "MLA value width");
                vHeadStride = headV;
                rotK = rotation.nRotK();
                rotV = rotation.nRotV();
                shiftHeads = nHead;
                shiftHeadStride = headK;
                shiftRowStride = qWidth;
                shiftOffs = nope;
            }
        }

        /**
         * Geometry of an MLA layer bound to its cache and the cache's rotation from the memory plan.
         *
         * @param rotation the layer's {@code LayerMemoryPlan.rotation()}
         * @throws IllegalStateException when the cache rows do not have the layer's latent or per-head shape, a
         *                               decompressed cache stores K and V in different types (one flash kernel
         *                               reads both), or the rotation blocks do not tile the rows they rotate
         */
        static Geometry of(LayerSpec layer, KvLayout layout, RotationPlan rotation) {
            MlaSpec m = layer.mla();
            Objects.requireNonNull(layout, "layout");
            Objects.requireNonNull(rotation, "rotation");
            if (m.latentCache()) {
                if (!layout.vAliasesK() || layout.widthK() != m.latentRowWidth() || layout.widthV() != m.kvLoraRank()) {
                    throw new IllegalStateException("layer " + layer.il() + ": the latent MLA cache needs rows of "
                            + m.latentRowWidth() + " with V aliasing the first " + m.kvLoraRank() + ", got " + layout);
                }
            } else {
                long wk = (long) m.nHead() * m.headK();
                long wv = (long) m.nHead() * m.headV();
                if (layout.vAliasesK() || layout.widthK() != wk || layout.widthV() != wv) {
                    throw new IllegalStateException("layer " + layer.il() + ": the decompressed MLA cache needs K rows of "
                            + wk + " and V rows of " + wv + ", got " + layout);
                }
                if (layout.dtypeK() != layout.dtypeV()) {
                    throw new IllegalStateException("layer " + layer.il() + ": the flash kernels read K and V of one "
                            + "type, the decompressed MLA cache stores K as " + layout.dtypeK() + " and V as "
                            + layout.dtypeV() + "; set neuralfree.kv.typeK and neuralfree.kv.typeV alike");
                }
            }
            checkRotation(layer.il(), m, rotation);
            return new Geometry(m, rotation);
        }

        /**
         * Checks that the planned rotation can be applied as planned: K blocks tiling the K head (and, for a latent
         * row, its {@code kv_lora} prefix, which V and the attention output alias), V blocks tiling the V head, no V
         * rotation of a latent cache (V is a view of K), and blocks within the Hadamard kernel's limit.
         */
        private static void checkRotation(int il, MlaSpec m, RotationPlan rotation) {
            int bk = rotation.nRotK();
            int bv = rotation.nRotV();
            String where = "layer " + il + ": the memory plan's " + rotation + " ";
            if (m.latentCache()) {
                if (bv > 0) {
                    throw new IllegalStateException(where + "rotates V of a latent MLA cache, whose V is a view of K");
                }
                if (bk > 0 && (m.latentRowWidth() % bk != 0 || m.kvLoraRank() % bk != 0)) {
                    throw new IllegalStateException(where + "does not tile the latent row of " + m.latentRowWidth()
                            + " and its kv_lora prefix of " + m.kvLoraRank());
                }
            } else {
                if (bk > 0 && m.headK() % bk != 0) {
                    throw new IllegalStateException(where + "does not tile the K head of " + m.headK());
                }
                if (bv > 0 && m.headV() % bv != 0) {
                    throw new IllegalStateException(where + "does not tile the V head of " + m.headV());
                }
            }
            if (bk > HadamardSpec.MAX_KERNEL_BLOCK || bv > HadamardSpec.MAX_KERNEL_BLOCK) {
                throw new IllegalStateException(where + "exceeds the Hadamard kernel block of "
                        + HadamardSpec.MAX_KERNEL_BLOCK);
            }
        }

        /** Rows of the step's Q, K or attention output: one per (token, query head). */
        int headRows(int nTokens) {
            return nTokens * nHead;
        }

        /** {@code rot_k} the context shift undoes and re-applies around the RoPE: the K block, 0 when unrotated. */
        int shiftRotK() {
            return rotK;
        }

        /**
         * Why the K rows of this cache cannot be shifted, or null: multi-position RoPE (llama.cpp get_can_shift,
         * llama-kv-cache.cpp:1213-1222), rotated rows of a type whose shift kernel does not un-rotate, or a geometry
         * the {@code LLM_KV_RopeShift_*} kernels reject.
         */
        String shiftBlocker(RopeSpec rope, KvLayout layout) {
            if (rope == null) {
                return null;
            }
            if (RopeMath.multiSection(rope.type())) {
                return "multi-position RoPE (" + rope.type() + ") cannot be shifted (llama-kv-cache.cpp:1213-1222)";
            }
            if (rotK > 0 && !layout.dtypeK().quantized()) {
                return "the " + layout.dtypeK() + " cache holds Hadamard-rotated K rows; the shift kernels undo the"
                        + " rotation only in quantized (block-type) caches";
            }
            try {
                RopeMath.shiftType(rope.type());
                RopeKernels.validateKvShift(layout.dtypeK(), rope.nDims(), shiftOffs, layout.rows(), shiftHeads,
                        shiftHeadStride, shiftRowStride, shiftRotK());
                return null;
            } catch (IllegalArgumentException e) {
                return e.getMessage();
            }
        }

        @Override
        public String toString() {
            return "MlaGeometry{" + (latent ? "latent" : "decompressed") + ", nHead=" + nHead + ", nope=" + nope
                    + ", rope=" + rope + ", kvLora=" + kvLora + ", headV=" + headV + ", dQ=" + dQ + ", dV=" + dV
                    + ", rotK=" + rotK + ", rotV=" + rotV + '}';
        }
    }
}
