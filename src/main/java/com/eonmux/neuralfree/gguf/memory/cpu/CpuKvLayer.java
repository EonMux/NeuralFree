package com.eonmux.neuralfree.gguf.memory.cpu;

import com.eonmux.neuralfree.gguf.memory.KvBlockCodec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvVendorCodec;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceStepInputs;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Heap KV cache of one owning layer. K and V rows are stored in their {@link KvDType}: F32 as {@code float[]}, F16 and
 * BF16 as {@code short[]} (IEEE half / bfloat16 bits, round to nearest even), the FP8 types as {@code byte[]} OCP FP8
 * codes ({@link KvBlockCodec#fp8Code}), the block types (Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0, Q5_1) as {@code byte[]} ggml
 * blocks quantized by the reference quantizers of {@link KvBlockCodec}, the per-token-head types as one signed code
 * per value (two per byte for INT4) plus an F32 side row of one scale per KV head, and the vendor formats through
 * {@link KvVendorCodec}. A full-row write of a per-token-head dtype derives each head's scale
 * ({@link KvBlockCodec#perTokenHeadScale}); a slice write reuses the row's stored scales, so a context shift rotates
 * the stored codes and keeps the scale. Reads decode the codes and multiply each head by its row scale, so callers see
 * the stored values. A cache whose K and V are both {@code INT8_PER_TOKEN_HEAD} also exposes its stored codes and
 * scales through {@link #int8Rows()}, which the fused int8 attention reads instead of dequantizing. Caches whose
 * element count exceeds a Java array are split into row chunks. Reads may run concurrently with each other; writes and
 * row moves are made by the thread that owns the step.
 *
 * <p>Every {@link KvDType.SidePlane} of the K and V dtypes has one {@link Rows} here, indexed by the plane's
 * {@link KvDType.SidePlane#widthFor(int, int) width}, and travels with its rows through copies, shifts and
 * transfers. The vendor formats decode from those planes: NVFP4 multiplies each 16-value group by its E4M3 scale,
 * the TurboQuant key stream reconstructs the centroid vector, norm-corrects it, applies the inverse Hadamard
 * rotation and scales it by the row's vector norm, the TurboQuant value stream applies its affine scale and zero,
 * and the DeepSeek MLA latent rows decode their NoPE tiles (E4M3 or E2M1) under their tile scales beside the
 * unscaled RoPE tail.</p>
 */
public final class CpuKvLayer implements KvAccessor {

    private final int layer;
    private final KvLayout layout;
    private final int group;
    private final int nHeadKv;
    private final int headDimK;
    private final int headDimV;
    private final Rows k;
    private final Rows v;
    /** One row set per K side plane, in {@link KvDType#sidePlanes()} order. */
    private final List<Rows> kPlanes;
    /** One row set per V side plane; empty when V aliases K. */
    private final List<Rows> vPlanes;
    /** Raw code view of an int8 cache; null when K or V is stored in another dtype. */
    private final KvInt8Rows int8Rows;

    private CpuKvLayer(int layer, KvLayout layout, int group, int nHeadKv, int headDimK, int headDimV) {
        this.layer = layer;
        this.layout = layout;
        this.group = group;
        this.nHeadKv = nHeadKv;
        this.headDimK = headDimK;
        this.headDimV = headDimV;
        this.k = new Rows(layout.dtypeK(), layout.rows(), layout.widthK());
        this.v = layout.vAliasesK() ? null : new Rows(layout.dtypeV(), layout.rows(), layout.widthV());
        this.kPlanes = allocatePlanes(layout.dtypeK(), layout.widthK(), layout.rows());
        this.vPlanes = v == null ? new ArrayList<Rows>() : allocatePlanes(layout.dtypeV(), layout.widthV(), layout.rows());
        this.int8Rows = v != null && layout.dtypeK() == KvDType.INT8_PER_TOKEN_HEAD
                && layout.dtypeV() == KvDType.INT8_PER_TOKEN_HEAD ? new Int8Rows() : null;
    }

    private List<Rows> allocatePlanes(KvDType dtype, int width, int rows) {
        List<Rows> out = new ArrayList<>();
        for (KvDType.SidePlane plane : dtype.sidePlanes()) {
            out.add(new Rows(plane.dtype(), rows, plane.widthFor(width, nHeadKv)));
        }
        return out;
    }

    /**
     * Allocates a zeroed cache. A latent layout ({@link KvLayout#vAliasesK()}) has one head of
     * {@code (widthK, widthV)}; otherwise the widths must equal {@code nHeadKv * headDimK} and
     * {@code nHeadKv * headDimV}.
     *
     * @param group KV group addressing the rows, or -1 for a cache not addressed by a group
     * @throws IllegalArgumentException when the head shape does not match the layout widths
     */
    public static CpuKvLayer allocate(int layer, KvLayout layout, int group, int nHeadKv, int headDimK, int headDimV) {
        Objects.requireNonNull(layout, "layout");
        if (layout.vAliasesK()) {
            if (nHeadKv != 1 || headDimK != layout.widthK() || headDimV != layout.widthV()) {
                throw new IllegalArgumentException("latent KV cache of layer " + layer + " has one head of ("
                        + layout.widthK() + ", " + layout.widthV() + "), got " + nHeadKv + " x (" + headDimK + ", "
                        + headDimV + ")");
            }
        } else if (nHeadKv <= 0 || (long) nHeadKv * headDimK != layout.widthK()
                || (long) nHeadKv * headDimV != layout.widthV()) {
            throw new IllegalArgumentException("KV cache of layer " + layer + " stores rows of K " + layout.widthK()
                    + " / V " + layout.widthV() + ", which " + nHeadKv + " heads of (" + headDimK + ", " + headDimV
                    + ") do not fill");
        }
        return new CpuKvLayer(layer, layout, group, nHeadKv, headDimK, headDimV);
    }

    /** A single-head key-only cache (indexer, MSA and compressed rows): V reads the K row and cannot be written. */
    static CpuKvLayer keyOnly(int layer, KvLayout latentLayout, int group) {
        return allocate(layer, latentLayout, group, 1, latentLayout.widthK(), latentLayout.widthV());
    }

    @Override
    public int layer() {
        return layer;
    }

    @Override
    public KvLayout layout() {
        return layout;
    }

    @Override
    public int group() {
        return group;
    }

    @Override
    public int nHeadKv() {
        return nHeadKv;
    }

    @Override
    public int headDimK() {
        return headDimK;
    }

    @Override
    public int headDimV() {
        return headDimV;
    }

    // ================================================================== reads

    @Override
    public void readK(int row, int head, float[] dst, int off) {
        checkHead(head);
        readRow(layout.dtypeK(), k, kPlanes, row, head, headDimK, dst, off);
    }

    @Override
    public void readKSlice(int row, int from, int n, float[] dst, int off) {
        if (from < 0 || n < 0 || from > layout.widthK() - n) {
            throw new IndexOutOfBoundsException("values [" + from + ", " + (from + n) + ") outside a K row of "
                    + layout.widthK());
        }
        if (from % headDimK == 0 && n == headDimK) {
            readK(row, from / headDimK, dst, off);
            return;
        }
        float[] full = new float[layout.widthK()];
        for (int h = 0; h < nHeadKv; h++) {
            readK(row, h, full, h * headDimK);
        }
        System.arraycopy(full, from, dst, off, n);
    }

    @Override
    public void readV(int row, int head, float[] dst, int off) {
        checkHead(head);
        if (v == null) {
            readRow(layout.dtypeK(), k, kPlanes, row, head, headDimV, dst, off);
        } else {
            readRow(layout.dtypeV(), v, vPlanes, row, head, headDimV, dst, off);
        }
    }

    /** Decodes head {@code head} of {@code row} of a cache of {@code dtype} with its side planes. */
    private void readRow(KvDType dtype, Rows rows, List<Rows> planes, int row, int head, int headElems, float[] dst,
                         int off) {
        if (!dtype.vendorFormat()) {
            rows.read(row, head * headElems, headElems, dst, off);
            applyScale(planes, 0, row, head, dst, off, headElems);
            return;
        }
        if (dtype == KvDType.TQ_K8_KEYS) {
            rows.read(row, head * headElems, headElems, dst, off);
            return;
        }
        if (dtype.nvfp4()) {
            int groups = headElems / KvVendorCodec.NVFP4_GROUP;
            int groupBase = head * groups;
            int byteBase = head * (headElems / 2);
            byte[] stored = rows.storedRow(row);
            for (int i = 0; i < headElems; i++) {
                float sf = planes.get(0).value(row, groupBase + i / KvVendorCodec.NVFP4_GROUP);
                dst[off + i] = KvVendorCodec.e2m1Value(KvVendorCodec.nvfp4Code(stored, byteBase, i)) * sf;
            }
            return;
        }
        if (dtype.turboQuantValue()) {
            boolean three = dtype == KvDType.TQ_3BIT_VALUES;
            int bits = three ? 3 : 4;
            Rows codes = rows;
            float scale = planes.get(0).value(row, head);
            float zero = planes.get(1).value(row, head);
            byte[] stored = codes.storedRow(row);
            int byteBase = head * codeBytes(headElems, bits);
            for (int i = 0; i < headElems; i++) {
                dst[off + i] = KvVendorCodec.turboQuantCode(stored, byteBase, i, bits) * scale + zero;
            }
            return;
        }
        if (dtype.turboQuantMse()) {
            readTurboQuantMse(rows, planes, row, head, headElems, dst, off);
            return;
        }
        if (dtype.latentFormat()) {
            if (head != 0) {
                throw new IllegalArgumentException("layer " + layer + ": a " + dtype + " latent row has one head, got "
                        + head);
            }
            float[] full = new float[KvDType.DS_MLA_ROW_VALUES];
            readLatent(dtype, rows, planes, row, full, 0);
            System.arraycopy(full, 0, dst, off, Math.min(headElems, KvDType.DS_MLA_ROW_VALUES));
            return;
        }
        throw new IllegalArgumentException("layer " + layer + ": no CPU decode for KV type " + dtype);
    }

    /** Bytes of {@code n} values of a {@code bits}-bit TurboQuant stream. */
    private static int codeBytes(int n, int bits) {
        return bits == 4 ? n / 2 : n / 8 * 3;
    }

    /**
     * Reconstructs one TurboQuant key head: the centroid vector of the row's indices, norm-corrected as vLLM's
     * {@code NORM_CORRECTION} path does, rotated back by the normalized Hadamard matrix and scaled by the vector norm.
     */
    private void readTurboQuantMse(Rows rows, List<Rows> planes, int row, int head, int headElems, float[] dst,
                                   int off) {
        int bits = rows.dtype() == KvDType.TQ_3BIT_KEYS ? 3 : 4;
        byte[] stored = rows.storedRow(row);
        int byteBase = head * codeBytes(headElems, bits);
        float[] centroids = KvVendorCodec.centroids(headElems, bits);
        int[] indices = new int[headElems];
        for (int i = 0; i < headElems; i++) {
            indices[i] = KvVendorCodec.turboQuantCode(stored, byteBase, i, bits);
        }
        float[] centroidsOfRow = new float[headElems];
        float[] norm = new float[1];
        KvVendorCodec.centroidsWithNorm(indices, headElems, centroids, true, centroidsOfRow, 0, norm);
        float[] rotated = new float[headElems];
        KvVendorCodec.hadamard(centroidsOfRow, 0, headElems, rotated, 0);
        float vecNorm = planes.get(0).value(row, head);
        for (int i = 0; i < headElems; i++) {
            dst[off + i] = rotated[i] * vecNorm;
        }
    }

    /** Decodes a DeepSeek MLA latent row ({@code fp8_ds_mla} or {@code nvfp4_ds_mla}) into its 576 values. */
    private void readLatent(KvDType dtype, Rows rows, List<Rows> planes, int row, float[] dst, int off) {
        byte[] stored = rows.storedRow(row);
        if (dtype == KvDType.FP8_DS_MLA) {
            float[] scales = new float[4];
            for (int i = 0; i < 4; i++) {
                scales[i] = planes.get(0).value(row, i);
            }
            KvVendorCodec.fp8DsMlaLoad(stored, 0, scales, 0, dst, off);
            return;
        }
        float[] sf = new float[32];
        for (int i = 0; i < sf.length; i++) {
            sf[i] = planes.get(0).value(row, i);
        }
        KvVendorCodec.nvfp4DsMlaLoad(stored, 0, sf, 0, dst, off);
    }

    /** Multiplies the {@code n} decoded values of head {@code head} by the plane's stored scale. */
    private static void applyScale(List<Rows> planes, int plane, int row, int head, float[] dst, int off, int n) {
        if (planes.isEmpty()) {
            return;
        }
        float s = planes.get(plane).value(row, head);
        for (int i = 0; i < n; i++) {
            dst[off + i] *= s;
        }
    }

    // ================================================================== writes

    @Override
    public void writeK(int row, float[] src, int off) {
        writeRow(layout.dtypeK(), k, kPlanes, row, src, off, headDimK);
    }

    @Override
    public void writeKSlice(int row, int from, int n, float[] src, int off) {
        if (from < 0 || n < 0 || from > layout.widthK() - n) {
            throw new IndexOutOfBoundsException("values [" + from + ", " + (from + n) + ") outside a K row of "
                    + layout.widthK());
        }
        KvDType dtype = layout.dtypeK();
        if (dtype.quantized() && (from % dtype.blockSize() != 0 || n % dtype.blockSize() != 0)) {
            throw new IllegalArgumentException(dtype + " rows are written in whole blocks of " + dtype.blockSize()
                    + " values, got values [" + from + ", " + (from + n) + ")");
        }
        float[] full = new float[layout.widthK()];
        for (int h = 0; h < nHeadKv; h++) {
            readK(row, h, full, h * headDimK);
        }
        System.arraycopy(src, off, full, from, n);
        writeK(row, full, 0);
    }

    @Override
    public void writeV(int row, float[] src, int off) {
        if (v == null) {
            throw new UnsupportedOperationException("V of the latent cache of layer " + layer + " is part of the K row");
        }
        writeRow(layout.dtypeV(), v, vPlanes, row, src, off, headDimV);
    }

    /** Stores a full row of {@code dtype} and derives its side planes. */
    private void writeRow(KvDType dtype, Rows rows, List<Rows> planes, int row, float[] src, int off, int headElems) {
        if (!dtype.vendorFormat()) {
            if (planes.isEmpty()) {
                rows.write(row, src, off);
            } else {
                storePerTokenHead(rows, planes.get(0), row, src, off, headElems);
            }
            return;
        }
        if (dtype.nvfp4()) {
            storeNvfp4(rows, planes.get(0), row, src, off, headElems, dtype == KvDType.NVFP4_4OVER6);
            return;
        }
        if (dtype.turboQuantValue()) {
            boolean three = dtype == KvDType.TQ_3BIT_VALUES;
            storeTurboQuantValues(rows, planes, row, src, off, headElems, three ? 7 : 15, three ? 3 : 4);
            return;
        }
        if (dtype.turboQuantKey()) {
            storeTurboQuantKeys(dtype, rows, planes, row, src, off, headElems);
            return;
        }
        if (dtype.latentFormat()) {
            storeLatent(dtype, rows, planes, row, src, off);
            return;
        }
        throw new IllegalArgumentException("layer " + layer + ": no CPU store for KV type " + dtype);
    }

    @Override
    public KvInt8Rows int8Rows() {
        return int8Rows;
    }

    /** The stored codes and scales of an {@code INT8_PER_TOKEN_HEAD} K and V pair. */
    private final class Int8Rows implements KvInt8Rows {

        @Override
        public int nHeadKv() {
            return nHeadKv;
        }

        @Override
        public int headDimK() {
            return headDimK;
        }

        @Override
        public int headDimV() {
            return headDimV;
        }

        @Override
        public void readKCodes(int row, int head, byte[] dst, int off) {
            checkHead(head);
            k.readCodes(row, head * headDimK, headDimK, dst, off);
        }

        @Override
        public void readVCodes(int row, int head, byte[] dst, int off) {
            checkHead(head);
            v.readCodes(row, head * headDimV, headDimV, dst, off);
        }

        @Override
        public float kScale(int row, int head) {
            checkHead(head);
            return kPlanes.get(0).value(row, head);
        }

        @Override
        public float vScale(int row, int head) {
            checkHead(head);
            return vPlanes.get(0).value(row, head);
        }

        @Override
        public String toString() {
            return "KvInt8Rows{layer=" + layer + ", " + nHeadKv + "x(" + headDimK + ", " + headDimV + ")}";
        }
    }

    /**
     * Stores a full row of a per-token-head cache: each head's values get the scale of their absmax over qmax, then
     * the codes of {@code value / scale}.
     */
    private static void storePerTokenHead(Rows rows, Rows scales, int row, float[] src, int off, int headElems) {
        KvDType type = rows.dtype();
        int heads = rows.width() / headElems;
        byte[] codes = new byte[headElems];
        for (int h = 0; h < heads; h++) {
            int base = off + h * headElems;
            float scale = KvBlockCodec.perTokenHeadScale(type, src, base, headElems);
            scales.setValue(row, h, scale);
            for (int e = 0; e < headElems; e++) {
                codes[e] = (byte) KvBlockCodec.perTokenHeadCode(type, src[base + e], scale);
            }
            rows.writeCodes(row, h * headElems, headElems, codes);
        }
    }

    /** NVFP4: one E4M3 scale per 16-value group and E2M1 codes packed two per byte. */
    private void storeNvfp4(Rows rows, Rows sf, int row, float[] src, int off, int headElems, boolean fourOverSix) {
        int heads = rows.width() / headElems;
        byte[] codes = new byte[headElems / 2];
        for (int h = 0; h < heads; h++) {
            int base = off + h * headElems;
            int groups = headElems / KvVendorCodec.NVFP4_GROUP;
            for (int g = 0; g < groups; g++) {
                int at = base + g * KvVendorCodec.NVFP4_GROUP;
                float sfValue = KvVendorCodec.nvfp4GroupScale(src, at, KvVendorCodec.NVFP4_GROUP, fourOverSix);
                sf.setValue(row, h * groups + g, sfValue);
                KvVendorCodec.nvfp4Pack(src, at, KvVendorCodec.NVFP4_GROUP, sfValue, codes,
                        g * (KvVendorCodec.NVFP4_GROUP / 2));
            }
            rows.writeStoredBytes(row, h * (headElems / 2), codes, 0, headElems / 2);
        }
    }

    /** TurboQuant value stream: per-head affine scale and zero, uniform 3- or 4-bit codes. */
    private void storeTurboQuantValues(Rows rows, List<Rows> planes, int row, float[] src, int off, int headElems,
                                       int levels, int bits) {
        int heads = rows.width() / headElems;
        int[] codes = new int[headElems];
        float[] scale = new float[1];
        float[] zero = new float[1];
        byte[] packed = new byte[codeBytes(headElems, bits)];
        for (int h = 0; h < heads; h++) {
            KvVendorCodec.turboQuantValueCodes(src, off + h * headElems, headElems, levels, codes, scale, zero);
            planes.get(0).setValue(row, h, scale[0]);
            planes.get(1).setValue(row, h, zero[0]);
            Arrays.fill(packed, (byte) 0);
            KvVendorCodec.turboQuantPack(codes, headElems, bits, packed, 0);
            rows.writeStoredBytes(row, h * codeBytes(headElems, bits), packed, 0, codeBytes(headElems, bits));
        }
    }

    /**
     * TurboQuant key stream: E4M3 codes for the FP8 stream; for the Lloyd-Max streams, the index of
     * {@code Hadamard(x / ||x||)} per coordinate and the vector norm in its per-head plane.
     */
    private void storeTurboQuantKeys(KvDType dtype, Rows rows, List<Rows> planes, int row, float[] src, int off,
                                     int headElems) {
        int heads = rows.width() / headElems;
        if (dtype == KvDType.TQ_K8_KEYS) {
            byte[] codes = new byte[headElems];
            for (int h = 0; h < heads; h++) {
                for (int i = 0; i < headElems; i++) {
                    codes[i] = (byte) KvVendorCodec.e4m3Code(src[off + h * headElems + i]);
                }
                rows.writeCodes(row, h * headElems, headElems, codes);
            }
            return;
        }
        int bits = dtype == KvDType.TQ_3BIT_KEYS ? 3 : 4;
        float[] midpoints = KvVendorCodec.midpoints(headElems, bits);
        float[] unit = new float[headElems];
        float[] rotated = new float[headElems];
        int[] codes = new int[headElems];
        byte[] packed = new byte[codeBytes(headElems, bits)];
        for (int h = 0; h < heads; h++) {
            int base = off + h * headElems;
            float sum = 0.0f;
            for (int i = 0; i < headElems; i++) {
                sum += src[base + i] * src[base + i];
            }
            float norm = (float) Math.sqrt(sum);
            for (int i = 0; i < headElems; i++) {
                unit[i] = src[base + i] / (norm + 1e-8f);
            }
            KvVendorCodec.hadamard(unit, 0, headElems, rotated, 0);
            for (int i = 0; i < headElems; i++) {
                codes[i] = KvVendorCodec.centroidIndex(rotated[i], midpoints);
            }
            planes.get(0).setValue(row, h, norm);
            Arrays.fill(packed, (byte) 0);
            KvVendorCodec.turboQuantPack(codes, headElems, bits, packed, 0);
            rows.writeStoredBytes(row, h * codeBytes(headElems, bits), packed, 0, codeBytes(headElems, bits));
        }
    }

    /** Stores a DeepSeek MLA latent row of 576 values and its per-tile or per-group scales. */
    private void storeLatent(KvDType dtype, Rows rows, List<Rows> planes, int row, float[] src, int off) {
        if (rows.width() != KvDType.DS_MLA_ROW_VALUES) {
            throw new IllegalArgumentException("layer " + layer + ": " + dtype + " rows hold "
                    + KvDType.DS_MLA_ROW_VALUES + " values, the cache row holds " + rows.width());
        }
        byte[] stored = new byte[(int) rows.rowBytes()];
        if (dtype == KvDType.FP8_DS_MLA) {
            float[] scales = new float[4];
            KvVendorCodec.fp8DsMlaStore(src, off, stored, 0, scales, 0);
            for (int i = 0; i < 4; i++) {
                planes.get(0).setValue(row, i, scales[i]);
            }
        } else {
            float[] sf = new float[32];
            KvVendorCodec.nvfp4DsMlaStore(src, off, stored, 0, sf, 0);
            for (int i = 0; i < sf.length; i++) {
                planes.get(0).setValue(row, i, sf[i]);
            }
        }
        rows.writeStoredRow(row, stored);
    }

    // ================================================================== rows and planes

    /**
     * Copies rows {@code [row0, row0 + n)} of K ({@code v} false) or V as their stored little-endian bytes
     * ({@link KvLayout#rowBytesK()} / {@link KvLayout#rowBytesV()} per row) into {@code dst[0..]}.
     *
     * @throws IllegalArgumentException for V of a latent cache, whose V is part of the K row
     */
    public void readRowBytes(boolean v, int row0, int n, byte[] dst) {
        rowsOf(v).readBytes(row0, n, Objects.requireNonNull(dst, "dst"));
    }

    /**
     * Stores {@code src[0, n * rowBytes)}, little-endian stored bytes, as rows {@code [row0, row0 + n)} of K or V;
     * the bits land unchanged, with no requantization.
     *
     * @throws IllegalArgumentException for V of a latent cache, whose V is part of the K row
     */
    public void writeRowBytes(boolean v, int row0, int n, byte[] src) {
        rowsOf(v).writeBytes(row0, n, Objects.requireNonNull(src, "src"));
    }

    private Rows rowsOf(boolean value) {
        if (!value) {
            return k;
        }
        if (v == null) {
            throw new IllegalArgumentException("V of the latent cache of layer " + layer + " is part of the K row");
        }
        return v;
    }

    /** Number of side planes of K ({@code v} false) or V. */
    public int sidePlaneCount(boolean value) {
        return planesOf(value).size();
    }

    /** Bytes of one side-plane row of K ({@code v} false) or V. */
    public long sidePlaneRowBytes(boolean value, int plane) {
        return planesOf(value).get(plane).rowBytes();
    }

    /**
     * Copies rows {@code [row0, row0 + n)} of side plane {@code plane} of K ({@code v} false) or V into
     * {@code dst[0, n * sidePlaneRowBytes)} as little-endian stored values.
     */
    public void readSideBytes(boolean value, int plane, int row0, int n, byte[] dst) {
        planesOf(value).get(plane).readBytes(row0, n, dst);
    }

    /** Stores little-endian side-plane values into rows {@code [row0, row0 + n)} of side plane {@code plane}. */
    public void writeSideBytes(boolean value, int plane, int row0, int n, byte[] src) {
        planesOf(value).get(plane).writeBytes(row0, n, src);
    }

    private List<Rows> planesOf(boolean value) {
        if (value && v == null) {
            throw new IllegalArgumentException("V of the latent cache of layer " + layer + " is part of the K row");
        }
        return value ? vPlanes : kPlanes;
    }

    /** Applies a context shift's row moves to K, V and their side planes. */
    void moveRows(DeviceStepInputs.KvShift shift) {
        k.moveRows(shift);
        if (v != null) {
            v.moveRows(shift);
        }
        for (Rows plane : kPlanes) {
            plane.moveRows(shift);
        }
        for (Rows plane : vPlanes) {
            plane.moveRows(shift);
        }
    }

    @Override
    public void copyRows(int srcRow, int dstRow, int n) {
        k.copyRows(srcRow, dstRow, n);
        if (v != null) {
            v.copyRows(srcRow, dstRow, n);
        }
        for (Rows plane : kPlanes) {
            plane.copyRows(srcRow, dstRow, n);
        }
        for (Rows plane : vPlanes) {
            plane.copyRows(srcRow, dstRow, n);
        }
    }

    /** Zeroes every row. */
    void clear() {
        k.clear();
        if (v != null) {
            v.clear();
        }
        for (Rows plane : kPlanes) {
            plane.clear();
        }
        for (Rows plane : vPlanes) {
            plane.clear();
        }
    }

    /**
     * Copies every stored bit of {@code src}'s rows into this cache's rows (no requantization).
     *
     * @throws IllegalArgumentException when the two caches do not share their row storage shapes
     */
    void copyAllFrom(CpuKvLayer src) {
        if ((v == null) != (src.v == null)) {
            throw new IllegalArgumentException("cache of layer " + layer + " (" + layout + ") cannot take the rows of "
                    + src.layout);
        }
        k.copyAllFrom(src.k);
        if (v != null) {
            v.copyAllFrom(src.v);
        }
        if (kPlanes.size() != src.kPlanes.size() || vPlanes.size() != src.vPlanes.size()) {
            throw new IllegalArgumentException("cache of layer " + layer + " cannot take the side rows of "
                    + src.layout);
        }
        for (int i = 0; i < kPlanes.size(); i++) {
            kPlanes.get(i).copyAllFrom(src.kPlanes.get(i));
        }
        for (int i = 0; i < vPlanes.size(); i++) {
            vPlanes.get(i).copyAllFrom(src.vPlanes.get(i));
        }
    }

    private void checkHead(int head) {
        if (head < 0 || head >= nHeadKv) {
            throw new IndexOutOfBoundsException("head " + head + " outside [0, " + nHeadKv + ") of layer " + layer);
        }
    }

    @Override
    public String toString() {
        return "CpuKvLayer{il=" + layer + ", group=" + group + ", heads=" + nHeadKv + "x(" + headDimK + ", " + headDimV
                + "), " + layout + '}';
    }

    // ================================================================== row storage

    /** Rows of one dtype, chunked so no array exceeds the JVM limit. */
    static final class Rows {

        private static final int MAX_ARRAY = Integer.MAX_VALUE - 8;

        private final KvDType dtype;
        private final int rows;
        private final int width;
        private final int unitsPerRow;
        private final int rowsPerChunk;
        private final float[][] f32;
        private final short[][] h16;
        private final byte[][] blocks;

        Rows(KvDType dtype, int rows, int width) {
            this.dtype = Objects.requireNonNull(dtype, "dtype");
            if (rows <= 0 || width <= 0 || !dtype.supportsWidth(width)) {
                throw new IllegalArgumentException(dtype + " rows of width " + width + " x " + rows + " are invalid");
            }
            this.rows = rows;
            this.width = width;
            this.unitsPerRow = dtype.byteStored() ? (int) dtype.rowBytes(width) : width;
            this.rowsPerChunk = Math.max(1, MAX_ARRAY / unitsPerRow);
            int chunks = (int) (((long) rows + rowsPerChunk - 1) / rowsPerChunk);
            this.f32 = dtype == KvDType.F32 ? new float[chunks][] : null;
            this.h16 = dtype == KvDType.F16 || dtype == KvDType.BF16 ? new short[chunks][] : null;
            this.blocks = dtype.byteStored() ? new byte[chunks][] : null;
            for (int c = 0; c < chunks; c++) {
                int n = Math.min(rowsPerChunk, rows - c * rowsPerChunk) * unitsPerRow;
                if (f32 != null) {
                    f32[c] = new float[n];
                } else if (h16 != null) {
                    h16[c] = new short[n];
                } else {
                    blocks[c] = new byte[n];
                }
            }
        }

        int rows() {
            return rows;
        }

        int width() {
            return width;
        }

        KvDType dtype() {
            return dtype;
        }

        /** One value of a side row: F32, F16 or BF16, or the decoded E4M3 code of an NVFP4 scale plane. */
        float value(int row, int index) {
            checkRow(row);
            if (index < 0 || index >= width) {
                throw new IndexOutOfBoundsException("value " + index + " outside a row of " + width);
            }
            int c = row / rowsPerChunk;
            int at = (row - c * rowsPerChunk) * unitsPerRow + index;
            if (f32 != null) {
                return f32[c][at];
            }
            if (h16 != null) {
                return dtype == KvDType.F16 ? FpBlockDecoders.fp16ToFp32(h16[c][at])
                        : FpBlockDecoders.bf16ToFp32(h16[c][at]);
            }
            return KvBlockCodec.fp8Value(dtype, blocks[c][at]);
        }

        /** Stores one value of a side row, rounding into the plane's dtype. */
        void setValue(int row, int index, float v) {
            checkRow(row);
            if (index < 0 || index >= width) {
                throw new IndexOutOfBoundsException("value " + index + " outside a row of " + width);
            }
            int c = row / rowsPerChunk;
            int at = (row - c * rowsPerChunk) * unitsPerRow + index;
            if (f32 != null) {
                f32[c][at] = v;
            } else if (h16 != null) {
                h16[c][at] = (short) (dtype == KvDType.F16 ? FpBlockDecoders.fp32ToFp16(v)
                        : KvBlockCodec.fp32ToBf16(v));
            } else {
                blocks[c][at] = (byte) KvBlockCodec.fp8Code(dtype, v);
            }
        }

        /** The raw stored bytes of one row (block, FP8 and vendor types). */
        byte[] storedRow(int row) {
            checkRow(row);
            if (blocks == null) {
                throw new IllegalStateException(dtype + " rows are not byte-stored");
            }
            int c = row / rowsPerChunk;
            int at = (row - c * rowsPerChunk) * unitsPerRow;
            return Arrays.copyOfRange(blocks[c], at, at + unitsPerRow);
        }

        /** Stores {@code n} raw bytes at byte {@code byteOff} of one row, the other bytes unchanged. */
        void writeStoredBytes(int row, int byteOff, byte[] src, int srcOff, int n) {
            checkRow(row);
            if (blocks == null || byteOff < 0 || n < 0 || byteOff > unitsPerRow - n) {
                throw new IndexOutOfBoundsException("bytes [" + byteOff + ", " + ((long) byteOff + n)
                        + ") outside a row of " + unitsPerRow + " bytes");
            }
            int c = row / rowsPerChunk;
            System.arraycopy(src, srcOff, blocks[c], (row - c * rowsPerChunk) * unitsPerRow + byteOff, n);
        }

        /** Stores raw little-endian row bytes; {@code src.length} must equal the row's stored size. */
        void writeStoredRow(int row, byte[] src) {
            checkRow(row);
            if (blocks == null || src.length != unitsPerRow) {
                throw new IllegalArgumentException(dtype + " rows hold " + unitsPerRow + " bytes, got "
                        + src.length);
            }
            int c = row / rowsPerChunk;
            int at = (row - c * rowsPerChunk) * unitsPerRow;
            System.arraycopy(src, 0, blocks[c], at, unitsPerRow);
        }

        /**
         * Stores the codes of values {@code [from, from + n)} of a cache row. Each code is one byte for FP8, INT8,
         * NVFP4 and the packed TurboQuant streams (whose byte order the caller packs) and one nibble for INT4 (value
         * {@code j} in the low nibble when {@code j} is even); a nibble write keeps the other value's nibble.
         */
        void writeCodes(int row, int from, int n, byte[] codes) {
            checkRow(row);
            checkSlice(from, n);
            if (n > codes.length) {
                throw new IllegalArgumentException(n + " values need " + n + " codes, got " + codes.length);
            }
            int c = row / rowsPerChunk;
            byte[] a = blocks[c];
            int base = (row - c * rowsPerChunk) * unitsPerRow;
            if (dtype == KvDType.INT4_PER_TOKEN_HEAD) {
                for (int i = 0; i < n; i++) {
                    int value = from + i;
                    int at = base + (value >> 1);
                    int code = codes[i] & 0xF;
                    if ((value & 1) == 0) {
                        a[at] = (byte) ((a[at] & 0xF0) | code);
                    } else {
                        a[at] = (byte) ((a[at] & 0x0F) | (code << 4));
                    }
                }
            } else {
                System.arraycopy(codes, 0, a, base + from, n);
            }
        }

        /**
         * Copies the stored codes of values {@code [from, from + n)} of a per-token-head row into
         * {@code dst[off..]}: one signed byte per value for INT8 and one unsigned byte for FP8.
         *
         * @throws IllegalStateException when the rows are not a per-token-head type or store two values per byte
         */
        void readCodes(int row, int from, int n, byte[] dst, int off) {
            if (!dtype.perTokenHead() || dtype == KvDType.INT4_PER_TOKEN_HEAD) {
                throw new IllegalStateException(dtype + " rows hold no one-byte code per value");
            }
            checkRow(row);
            checkSlice(from, n);
            if (off < 0 || n > dst.length - off) {
                throw new IndexOutOfBoundsException("destination [" + off + ", " + (off + n) + ") outside "
                        + dst.length);
            }
            int c = row / rowsPerChunk;
            int base = (row - c * rowsPerChunk) * unitsPerRow;
            System.arraycopy(blocks[c], base + from, dst, off, n);
        }

        /** Dequantizes values {@code [from, from + n)} of {@code row} into {@code dst[off..]}. */
        void read(int row, int from, int n, float[] dst, int off) {
            checkRow(row);
            checkSlice(from, n);
            if (off < 0 || n > dst.length - off) {
                throw new IndexOutOfBoundsException("destination [" + off + ", " + (off + n) + ") outside "
                        + dst.length);
            }
            int c = row / rowsPerChunk;
            int base = (row - c * rowsPerChunk) * unitsPerRow;
            switch (dtype) {
                case F32:
                    System.arraycopy(f32[c], base + from, dst, off, n);
                    break;
                case F16: {
                    short[] a = h16[c];
                    for (int i = 0; i < n; i++) {
                        dst[off + i] = FpBlockDecoders.fp16ToFp32(a[base + from + i]);
                    }
                    break;
                }
                case BF16: {
                    short[] a = h16[c];
                    for (int i = 0; i < n; i++) {
                        dst[off + i] = FpBlockDecoders.bf16ToFp32(a[base + from + i]);
                    }
                    break;
                }
                case FP8_E4M3:
                case FP8_E5M2: {
                    byte[] a = blocks[c];
                    for (int i = 0; i < n; i++) {
                        dst[off + i] = KvBlockCodec.fp8Value(dtype, a[base + from + i]);
                    }
                    break;
                }
                case TQ_K8_KEYS: {
                    byte[] a = blocks[c];
                    for (int i = 0; i < n; i++) {
                        dst[off + i] = KvBlockCodec.fp8Value(KvDType.FP8_E4M3, a[base + from + i]);
                    }
                    break;
                }
                case FP8_PER_TOKEN_HEAD:
                case INT8_PER_TOKEN_HEAD: {
                    byte[] a = blocks[c];
                    for (int i = 0; i < n; i++) {
                        dst[off + i] = KvBlockCodec.perTokenHeadValue(dtype, a[base + from + i]);
                    }
                    break;
                }
                case INT4_PER_TOKEN_HEAD: {
                    byte[] a = blocks[c];
                    for (int i = 0; i < n; i++) {
                        int value = from + i;
                        int packed = a[base + (value >> 1)] & 0xFF;
                        dst[off + i] = KvBlockCodec.perTokenHeadValue(dtype,
                                (value & 1) == 0 ? packed & 0xF : packed >>> 4);
                    }
                    break;
                }
                default: {
                    final int bs = dtype.blockSize();
                    int i = 0;
                    while (i < n) {
                        int e = from + i;
                        int j = e % bs;
                        int run = Math.min(bs - j, n - i);
                        KvBlockCodec.dequantize(dtype, blocks[c], base + (e / bs) * dtype.blockBytes(), j, run, dst,
                                off + i);
                        i += run;
                    }
                    break;
                }
            }
        }

        /** Stores a full row of {@code width} values from {@code src[off..]}. */
        void write(int row, float[] src, int off) {
            writeSlice(row, 0, width, src, off);
        }

        /**
         * Stores values {@code [from, from + n)} of {@code row} from {@code src[off..]}; the other values keep their
         * stored bits. A block type needs {@code from} and {@code n} on block boundaries.
         *
         * @throws IllegalArgumentException for a block-type slice that splits a block
         */
        void writeSlice(int row, int from, int n, float[] src, int off) {
            if (dtype.perTokenHead() || dtype.vendorFormat()) {
                throw new IllegalStateException(dtype + " rows are written through CpuKvLayer's format store");
            }
            checkRow(row);
            checkSlice(from, n);
            if (off < 0 || n > src.length - off) {
                throw new IndexOutOfBoundsException("source [" + off + ", " + (off + n) + ") outside " + src.length);
            }
            int c = row / rowsPerChunk;
            int base = (row - c * rowsPerChunk) * unitsPerRow;
            switch (dtype) {
                case F32:
                    System.arraycopy(src, off, f32[c], base + from, n);
                    break;
                case F16: {
                    short[] a = h16[c];
                    for (int i = 0; i < n; i++) {
                        a[base + from + i] = (short) FpBlockDecoders.fp32ToFp16(src[off + i]);
                    }
                    break;
                }
                case BF16: {
                    short[] a = h16[c];
                    for (int i = 0; i < n; i++) {
                        a[base + from + i] = (short) KvBlockCodec.fp32ToBf16(src[off + i]);
                    }
                    break;
                }
                case FP8_E4M3:
                case FP8_E5M2: {
                    byte[] a = blocks[c];
                    for (int i = 0; i < n; i++) {
                        a[base + from + i] = (byte) KvBlockCodec.fp8Code(dtype, src[off + i]);
                    }
                    break;
                }
                default: {
                    final int bs = dtype.blockSize();
                    if (from % bs != 0 || n % bs != 0) {
                        throw new IllegalArgumentException(dtype + " rows are written in whole blocks of " + bs
                                + " values, got values [" + from + ", " + (from + n) + ")");
                    }
                    for (int b = 0; b < n / bs; b++) {
                        KvBlockCodec.quantize(dtype, src, off + b * bs, blocks[c],
                                base + (from / bs + b) * dtype.blockBytes());
                    }
                    break;
                }
            }
        }

        private void checkSlice(int from, int n) {
            if (from < 0 || n < 0 || from > width - n) {
                throw new IndexOutOfBoundsException("values [" + from + ", " + (from + n) + ") outside a row of "
                        + width);
            }
        }

        /** Copies every chunk of {@code src}, which has this storage's dtype, row count and width. */
        void copyAllFrom(Rows src) {
            if (src.dtype != dtype || src.rows != rows || src.width != width) {
                throw new IllegalArgumentException(dtype + " rows of " + rows + " x " + width + " cannot take "
                        + src.dtype + " rows of " + src.rows + " x " + src.width);
            }
            int chunks = f32 != null ? f32.length : h16 != null ? h16.length : blocks.length;
            for (int c = 0; c < chunks; c++) {
                if (f32 != null) {
                    System.arraycopy(src.f32[c], 0, f32[c], 0, f32[c].length);
                } else if (h16 != null) {
                    System.arraycopy(src.h16[c], 0, h16[c], 0, h16[c].length);
                } else {
                    System.arraycopy(src.blocks[c], 0, blocks[c], 0, blocks[c].length);
                }
            }
        }

        /** Zeroes every row. */
        void clear() {
            int chunks = f32 != null ? f32.length : h16 != null ? h16.length : blocks.length;
            for (int c = 0; c < chunks; c++) {
                if (f32 != null) {
                    Arrays.fill(f32[c], 0f);
                } else if (h16 != null) {
                    Arrays.fill(h16[c], (short) 0);
                } else {
                    Arrays.fill(blocks[c], (byte) 0);
                }
            }
        }

        /**
         * Executes a context shift's row moves: downward moves row by row in ascending order (each row is read
         * before any move writes it); otherwise the moved rows are gathered first and then scattered.
         */
        void moveRows(DeviceStepInputs.KvShift shift) {
            if (shift.runCount() == 0) {
                return;
            }
            if (!shift.needsStaging()) {
                for (int i = 0; i < shift.runCount(); i++) {
                    for (int r = 0; r < shift.runRows(i); r++) {
                        copyRow(shift.runSource(i) + r, shift.runTarget(i) + r);
                    }
                }
                return;
            }
            Object[] staged = new Object[shift.movedRows()];
            int at = 0;
            for (int i = 0; i < shift.runCount(); i++) {
                for (int r = 0; r < shift.runRows(i); r++) {
                    staged[at++] = rowCopy(shift.runSource(i) + r);
                }
            }
            at = 0;
            for (int i = 0; i < shift.runCount(); i++) {
                for (int r = 0; r < shift.runRows(i); r++) {
                    putRow(shift.runTarget(i) + r, staged[at++]);
                }
            }
        }

        /** Bytes of one stored row in the dtype's little-endian layout ({@code KvDType.rowBytes}). */
        int rowBytes() {
            return f32 != null ? unitsPerRow * Float.BYTES : h16 != null ? unitsPerRow * Short.BYTES : unitsPerRow;
        }

        /** Copies rows {@code [row0, row0 + n)} as their stored little-endian bytes into {@code dst[0..]}. */
        void readBytes(int row0, int n, byte[] dst) {
            checkByteRange(row0, n, dst.length);
            int rb = rowBytes();
            for (int i = 0; i < n; i++) {
                int row = row0 + i;
                Object a = chunk(row);
                int o = offset(row);
                ByteBuffer out = ByteBuffer.wrap(dst, i * rb, rb).order(ByteOrder.LITTLE_ENDIAN);
                if (f32 != null) {
                    out.asFloatBuffer().put((float[]) a, o, unitsPerRow);
                } else if (h16 != null) {
                    out.asShortBuffer().put((short[]) a, o, unitsPerRow);
                } else {
                    System.arraycopy(a, o, dst, i * rb, rb);
                }
            }
        }

        /** Stores {@code src[0, n * rowBytes)}, little-endian stored bytes, as rows {@code [row0, row0 + n)}. */
        void writeBytes(int row0, int n, byte[] src) {
            checkByteRange(row0, n, src.length);
            int rb = rowBytes();
            for (int i = 0; i < n; i++) {
                int row = row0 + i;
                Object a = chunk(row);
                int o = offset(row);
                ByteBuffer in = ByteBuffer.wrap(src, i * rb, rb).order(ByteOrder.LITTLE_ENDIAN);
                if (f32 != null) {
                    in.asFloatBuffer().get((float[]) a, o, unitsPerRow);
                } else if (h16 != null) {
                    in.asShortBuffer().get((short[]) a, o, unitsPerRow);
                } else {
                    System.arraycopy(src, i * rb, a, o, rb);
                }
            }
        }

        private void checkByteRange(int row0, int n, int bytes) {
            if (n < 0 || row0 < 0 || row0 > rows - n) {
                throw new IndexOutOfBoundsException("rows [" + row0 + ", " + ((long) row0 + n)
                        + ") outside a cache of " + rows + " rows");
            }
            if ((long) n * rowBytes() > bytes) {
                throw new IllegalArgumentException(n + " rows of " + rowBytes() + " bytes do not fit an array of "
                        + bytes + " bytes");
            }
        }

        /** Moves {@code n} whole rows starting at {@code srcRow} onto the run at {@code dstRow} (a {@code PageCopy}). */
        void copyRows(int srcRow, int dstRow, int n) {
            if (n < 0 || srcRow < 0 || dstRow < 0 || srcRow + n > rows || dstRow + n > rows) {
                throw new IndexOutOfBoundsException("copy of rows [" + srcRow + ", " + (srcRow + n) + ") onto ["
                        + dstRow + ", " + (dstRow + n) + ") outside a cache of " + rows + " rows");
            }
            if (srcRow < dstRow) {
                for (int i = n - 1; i >= 0; i--) {
                    copyRow(srcRow + i, dstRow + i);
                }
            } else {
                for (int i = 0; i < n; i++) {
                    copyRow(srcRow + i, dstRow + i);
                }
            }
        }

        private void copyRow(int src, int dst) {
            putRow(dst, chunk(src), offset(src));
        }

        private Object rowCopy(int row) {
            Object a = chunk(row);
            int o = offset(row);
            if (f32 != null) {
                return Arrays.copyOfRange((float[]) a, o, o + unitsPerRow);
            }
            if (h16 != null) {
                return Arrays.copyOfRange((short[]) a, o, o + unitsPerRow);
            }
            return Arrays.copyOfRange((byte[]) a, o, o + unitsPerRow);
        }

        private void putRow(int row, Object data) {
            putRow(row, data, 0);
        }

        private void putRow(int row, Object src, int srcOff) {
            System.arraycopy(src, srcOff, chunk(row), offset(row), unitsPerRow);
        }

        private Object chunk(int row) {
            checkRow(row);
            int c = row / rowsPerChunk;
            return f32 != null ? f32[c] : h16 != null ? h16[c] : blocks[c];
        }

        private int offset(int row) {
            return (row % rowsPerChunk) * unitsPerRow;
        }

        private void checkRow(int row) {
            if (row < 0 || row >= rows) {
                throw new IndexOutOfBoundsException("row " + row + " outside a cache of " + rows + " rows");
            }
        }
    }
}
