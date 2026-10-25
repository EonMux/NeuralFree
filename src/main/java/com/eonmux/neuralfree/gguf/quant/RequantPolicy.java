package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.OnlineQuant;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.PlannedTensor;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/**
 * The eligibility and the producer of online weight quantization (QZ-1): {@code --quant-online} quantizes float
 * matmul weights at load, with llama-quantize's eligibility rules (LC/src/llama-quant.cpp:295-396
 * {@code tensor_allows_quantization}) and the row quantizers of {@link OnlineQuantizers}. The registry is the one
 * seam PAR QN-4 extends; {@code OnlineQuant.NONE} requantizes nothing.
 */
public final class RequantPolicy {

    /** Requantization keeps nothing below rank 2 ({@code ggml_n_dims(tensor) < 2}, llama-quant.cpp:300). */
    private static final int MIN_RANK = 2;

    /**
     * The names requantization keeps away from, in the order of llama-quant.cpp:308-377: the norm tensors, the
     * expert gating and routing tensors, the very small projections (altup, laurel, per_layer_model_proj), the
     * Mamba and Kimi convolutions, MiniMax's and GLM's tiny indexer, hc, ssm and MLA-a/k_b/v_b projection weights
     * (GLM5-next's arch gate of llama-quant.cpp:340-355 is applied as one name list) and RWKV's small time-mix
     * weights (:362-377).
     */
    private static final String[] FORBIDDEN = {
        "_norm.weight", "ffn_gate_inp.weight", "ffn_gate_tid2eid.weight", "altup", "laurel",
        "per_layer_model_proj", "ssm_conv1d", "shortconv.conv.weight", "indexer.k_proj.weight",
        "indexer.q_proj.weight", "hc_", "indexer.attn_q_b", "indexer.attn_k", "indexer.proj",
        "indexer_compressor_gate", "indexer_compressor_ape", "ssm_f_a.weight", "ssm_f_b.weight",
        "ssm_g_a.weight", "ssm_g_b.weight", "ssm_beta.weight", "attn_kv_a_mqa.weight", "attn_k_b.weight",
        "attn_v_b.weight", "time_mix_first.weight", "time_mix_w0.weight", "time_mix_w1.weight",
        "time_mix_w2.weight", "time_mix_v0.weight", "time_mix_v1.weight", "time_mix_v2.weight",
        "time_mix_a0.weight", "time_mix_a1.weight", "time_mix_a2.weight", "time_mix_g1.weight",
        "time_mix_g2.weight", "time_mix_decay_w1.weight", "time_mix_decay_w2.weight",
        "time_mix_lerp_fused.weight",
    };

    /** The exact names llama-quant.cpp compares, :327-328 (BERT tokens) and :357-360 (decision head). */
    private static final String[] FORBIDDEN_EXACT = {"pos_embd.weight", "token_types.weight"};

    private RequantPolicy() {
    }

    /**
     * Whether the GGUF tensor {@code ggufName}, currently stored as {@code current} and target at {@code target},
     * is requantized online (llama-quant.cpp:295-396 over the current type and the plan's rank): its type is float
     * (F32, F16 or BF16), its rank is at least 2, {@code ne[0]} is a whole number of blocks of {@code target}, and
     * its name passes the ported name rules: it ends with {@code weight} and holds none of the forbidden names.
     */
    public static boolean eligible(String ggufName, GGMLType current, long[] ne, GGMLType target) {
        Objects.requireNonNull(target, "target");
        if (current != GGMLType.F32 && current != GGMLType.F16 && current != GGMLType.BF16) {
            return false;
        }
        if (ne == null || ne.length < MIN_RANK) {
            return false;
        }
        if (target.getBlockSize() <= 0 || ne[0] % target.getBlockSize() != 0) {
            return false;
        }
        return nameEligible(ggufName);
    }

    /**
     * The name rules in one shot: the name ends with {@code weight} (llama-quant.cpp:305) and holds none of the
     * forbidden spellings (:308-377).
     */
    static boolean nameEligible(String ggufName) {
        if (ggufName == null || !ggufName.endsWith("weight")) {
            return false;
        }
        for (String forbidden : FORBIDDEN_EXACT) {
            if (ggufName.equals(forbidden)) {
                return false;
            }
        }
        for (String forbidden : FORBIDDEN) {
            if (ggufName.contains(forbidden)) {
                return false;
            }
        }
        return !ggufName.startsWith("decision.");
    }

    /**
     * Step QZ-1 of the Hugging Face planning pipeline: {@code src} requantized, or {@code src} itself when the
     * option carries no quantizer ({@code NONE}), {@code src} is not a plain float source (an already-quantized
     * checkpoint's tensors are never requantized), the planned type {@code current} is not float or the tensor is
     * not {@link #eligible(String, GGMLType, long[], GGMLType)}.
     */
    public static RowSource replace(String ggufName, RowSource src, GGMLType current, OnlineQuant target) {
        Objects.requireNonNull(src, "src");
        if (target == null || target == OnlineQuant.NONE || !src.format().isFloat()) {
            return src;
        }
        OnlineQuantizer quantizer = OnlineQuantizers.get(target);
        long[] ne = PlannedTensor.neOf(src.shape());
        if (!eligible(ggufName, current, ne, quantizer.type())) {
            return src;
        }
        return new QuantizedRows(ggufName, src, quantizer);
    }

    // ------------------------------------------------------------------ the producer

    /**
     * The float source's rows, each row's stored floats dequantized to f32 and quantized into the target type's row
     * (llama-quant.cpp:273-283, the f32 repack the requantizer feeds its row quantizers from). Immutable, stateless
     * between calls and thread-safe within its scratch per call; row-local.
     */
    private static final class QuantizedRows implements RowSource {

        /** The scratch a chunk of rows keeps under, in bytes (per call). */
        private static final int CHUNK_BYTES = 1 << 20;

        private final String ggufName;
        private final RowSource src;
        private final OnlineQuantizer quantizer;
        private final ElementFormat format;
        private final int inner;

        QuantizedRows(String ggufName, RowSource src, OnlineQuantizer quantizer) {
            this.ggufName = Objects.requireNonNull(ggufName, "ggufName");
            this.src = Objects.requireNonNull(src, "src");
            this.quantizer = Objects.requireNonNull(quantizer, "quantizer");
            long[] shape = src.shape();
            long innerL = shape.length == 0 ? 1 : shape[shape.length - 1];
            if (innerL > Integer.MAX_VALUE) {
                throw new IllegalArgumentException(ggufName + ": a row of " + innerL
                        + " values does not fit one int-sized requantization row");
            }
            this.inner = (int) innerL;
            if (inner % quantizer.type().getBlockSize() != 0) {
                throw new IllegalArgumentException(ggufName + ": a row of " + inner + " values is not a whole"
                        + " number of " + quantizer.type() + " blocks of " + quantizer.type().getBlockSize());
            }
            this.format = ElementFormat.of(quantizer.type());
        }

        @Override
        public String describe() {
            return ggufName + " quantized to " + quantizer.type() + " at load (from " + src.describe() + ")";
        }

        @Override
        public ElementFormat format() {
            return format;
        }

        @Override
        public long[] shape() {
            return src.shape();
        }

        @Override
        public long rows() {
            return src.rows();
        }

        @Override
        public long rowBytes() {
            return format.rowBytes(inner);
        }

        @Override
        public boolean rowLocal() {
            return true;
        }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            int inRowBytes = Math.toIntExact(src.rowBytes());
            int outRowBytes = Math.toIntExact(rowBytes());
            float[] x = new float[inner];
            byte[] row = new byte[outRowBytes];
            int chunk = Math.max(1, CHUNK_BYTES / Math.max(inRowBytes, 1));
            for (long r = first; r < first + count; r += chunk) {
                int rows = (int) Math.min(chunk, first + count - r);
                ByteBuffer in = ByteBuffer.allocate(rows * inRowBytes).order(ByteOrder.LITTLE_ENDIAN);
                src.readRows(r, rows, in);
                in.flip();
                for (int i = 0; i < rows; i++) {
                    decode(in, inRowBytes, x);
                    quantizer.quantizeRow(x, 0, inner, row, 0);
                    dst.put(row);
                }
            }
        }

        /** Decodes one stored float row of {@code rowBytes} bytes into {@code x}. */
        private void decode(ByteBuffer in, int rowBytes, float[] x) {
            int limit = in.position() + rowBytes;
            GGMLType t = src.format().ggmlType();
            for (int i = 0; i < x.length; i++) {
                switch (t) {
                    case F32:
                        x[i] = in.getFloat();
                        break;
                    case F16:
                        x[i] = FpBlockDecoders.fp16ToFp32(in.getShort() & 0xFFFF);
                        break;
                    case BF16:
                        x[i] = FpBlockDecoders.bf16ToFp32(in.getShort() & 0xFFFF);
                        break;
                    case F64:
                        x[i] = (float) in.getDouble();
                        break;
                    default:
                        throw new IllegalStateException(ggufName + ": no float decode of the source's " + t);
                }
            }
            if (in.position() != limit) {
                throw new IllegalStateException(ggufName + ": the stored float row read " + (in.position() - limit)
                        + " bytes short");
            }
        }

        @Override
        public String toString() {
            return describe() + ", " + Arrays.toString(src.shape());
        }
    }
}
