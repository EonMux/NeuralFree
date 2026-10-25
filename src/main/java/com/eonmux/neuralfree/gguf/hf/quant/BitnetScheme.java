package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.io.safetensors.StDType;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;

/**
 * Offline BitNet b1.58 weights ({@link QuantFormat#BITNET}) rewritten into TQ2_0: {@code weight} U8
 * {@code [N/4, K]} holds four ternary codes per byte, output row {@code n = i * (N/4) + r} taking
 * {@code (weight[r][k] >> 2i) & 3} as code {@code c}, and {@code w = (c - 1) / weight_scale} with a per-tensor
 * {@code weight_scale} (llama.cpp conversion/base.py:338-349, 462-469).
 *
 * <p>The source is TQ2_0 with {@code d = 1} and the inverse scale as its {@link RowSource#outputScale()}, so the
 * ternary codes stay exact; {@link RowSource#foldOutputScale()} gives TQ2_0 with {@code d = fp16(1 / scale)} (F32
 * values under exact quantization, or when fp16 cannot hold {@code 1 / scale}). Rows whose length is not a multiple
 * of 256 take F32 values {@code (c - 1) / scale}.</p>
 */
public final class BitnetScheme {

    private BitnetScheme() {
    }

    /** True for the formats this scheme rewrites. */
    public static boolean handles(QuantFormat f) {
        return f == QuantFormat.BITNET;
    }

    /**
     * Replaces each module's {@code weight} and {@code weight_scale} in {@code in} by one virtual weight source, as
     * {@link IntGroupScheme#rewriteModules} does.
     *
     * @throws IOException              when a tensor is missing, has the wrong dtype or shape, or holds invalid values
     * @throws IllegalArgumentException when {@code args} is not a format of this scheme
     */
    public static HfTensorIndex rewriteModules(QuantArgs args, List<String> modules, HfConfig cfg, HfTensorIndex in,
                                               ModelSourceOptions o) throws IOException {
        IntRewriteBatch.requireFormat(args, handles(args.format()), BitnetScheme.class);
        String checkpoint = cfg.dir().toString();
        return IntRewriteBatch.rewrite(args, modules, cfg, in, o, (b, m) -> ternary(b, m, checkpoint), "weight");
    }

    private static RowSource ternary(IntRewriteBatch b, IntModuleTensors m, String checkpoint) throws IOException {
        RowSource w = m.take("weight");
        m.requireDtype(w, "weight", StDType.U8, StDType.I8);
        long[] ws = m.tail(w, 0, 2, "weight");
        int k = m.width(ws[1], "weight");
        long quarter = ws[0];
        if (quarter <= 0 || quarter > Long.MAX_VALUE / 4) {
            throw m.invalid(m.tensorName("weight") + " has " + quarter + " packed rows; expected a positive count");
        }
        RowSource sc = m.take("weight_scale");
        float[] scale = m.perExpertFloats(sc, 0, "weight_scale");
        float s = scale[0];
        if (!Float.isFinite(s) || !Float.isFinite(1f / s)) {
            throw m.invalid(m.tensorName("weight_scale") + " is " + s + "; expected a finite scale with a finite"
                    + " inverse");
        }
        boolean exact = b.options().exactQuant();
        if (k % QK_K != 0) {
            b.warn("float fallback " + GGMLType.F32 + ": K " + k + " is not a multiple of " + QK_K, m.name());
            return new TernarySource(m.name(), checkpoint, w, quarter, k, s, GGMLType.F32, false, exact);
        }
        return new TernarySource(m.name(), checkpoint, w, quarter, k, s, GGMLType.TQ2_0, true, exact);
    }

    /**
     * The rows of one BitNet weight: TQ2_0 codes with {@code d = 1} and an output scale, TQ2_0 with the inverse scale
     * folded into {@code d}, or float values {@code (c - 1) / scale}. Immutable and thread-safe.
     */
    static final class TernarySource implements RowSource {
        private final String module;
        private final String checkpoint;
        private final RowSource packed;
        private final long quarter;
        private final int k;
        private final float scale;
        private final GGMLType target;
        private final boolean outputScaled;
        private final boolean exact;
        private final ElementFormat format;
        private final long rowBytes;

        TernarySource(String module, String checkpoint, RowSource packed, long quarter, int k, float scale,
                      GGMLType target, boolean outputScaled, boolean exact) {
            this.module = module;
            this.checkpoint = checkpoint;
            this.packed = packed;
            this.quarter = quarter;
            this.k = k;
            this.scale = scale;
            this.target = target;
            this.outputScaled = outputScaled;
            this.exact = exact;
            this.format = ElementFormat.of(target);
            this.rowBytes = format.rowBytes(k);
        }

        @Override
        public String describe() {
            return module + " " + QuantFormat.BITNET + " b2 g" + k + " -> " + target;
        }

        @Override
        public ElementFormat format() {
            return format;
        }

        @Override
        public long[] shape() {
            return new long[]{4 * quarter, k};
        }

        @Override
        public long rows() {
            return 4 * quarter;
        }

        @Override
        public long rowBytes() {
            return rowBytes;
        }

        @Override
        public boolean rowLocal() {
            return true;
        }

        @Override
        public float[] outputScale() {
            return outputScaled ? new float[]{1f / scale} : null;
        }

        @Override
        public RowSource foldOutputScale() {
            if (!outputScaled) {
                throw new UnsupportedOperationException(describe() + " has no output scale to fold");
            }
            float d = FpBlockDecoders.roundToFp16(1f / scale);
            boolean fp16Holds = Float.isFinite(d) && d != 0;
            return new TernarySource(module, checkpoint, packed, quarter, k, scale,
                    exact || !fp16Holds ? GGMLType.F32 : GGMLType.TQ2_0, false, exact);
        }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            if (first < 0 || count < 0 || first + count > rows()) {
                throw new IllegalArgumentException(describe() + ": rows [" + first + ", " + (first + count)
                        + ") lie outside [0, " + rows() + ")");
            }
            long bytes = count * rowBytes;
            if (dst.remaining() < bytes) {
                throw new IllegalArgumentException(describe() + ": " + count + " rows need " + bytes + " bytes, "
                        + dst.remaining() + " remain");
            }
            ByteBuffer out = dst.slice().order(ByteOrder.LITTLE_ENDIAN);
            int chunk = Math.max(1, (1 << 24) / k);
            int done = 0;
            int[] q = new int[k];
            while (done < count) {
                long n = first + done;
                int shift = (int) (n / quarter) * 2;
                long r0 = n % quarter;
                int c = (int) Math.min(Math.min(count - done, chunk), quarter - r0);
                ByteBuffer in = IntPacking.rows(packed, r0, c);
                for (int i = 0; i < c; i++) {
                    for (int j = 0; j < k; j++) {
                        q[j] = (in.get(i * k + j) >>> shift) & 3;
                    }
                    encodeRow(q, out, (int) ((done + i) * rowBytes));
                }
                done += c;
            }
            dst.position(dst.position() + (int) bytes);
        }

        private void encodeRow(int[] q, ByteBuffer out, int off) {
            if (target == GGMLType.TQ2_0) {
                float d = outputScaled ? 1f : FpBlockDecoders.roundToFp16(1f / scale);
                for (int j0 = 0, b = 0; j0 < k; j0 += QK_K, b++) {
                    IntPacking.putTq2_0(out, off + b * target.getTypeSize(), d, q, j0);
                }
                return;
            }
            for (int j = 0; j < k; j++) {
                IntPacking.putFloat(out, off, j, target, (q[j] - 1) / scale);
            }
        }

        @Override
        public String toString() {
            return describe() + " in '" + checkpoint + "'";
        }
    }
}
