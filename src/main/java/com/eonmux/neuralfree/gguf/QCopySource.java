package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.quant.RowQuantizers;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * A quantized copy of a target model's own projection matrices, the target view of two-level verification
 * ({@code neuralfree.draft.tlv}, SpecPilot's TLV, arXiv 2611.02501): the copy restates every main-stack
 * attention and feed-forward matrix and the output head that the copy undercuts — round-to-nearest at
 * {@code neuralfree.draft.tlv.bits}, 6 bits as {@link GGMLType#Q6_K} rows (6.5625 bits per weight, whole
 * 256-value blocks) or 8 as {@link GGMLType#Q8_0} (8.5 bits, whole 32-value blocks) — and hands every other
 * tensor through verbatim, so a model opened over the copy runs the same graph through the existing int8 and
 * K-quant matvec kernels and streams only what the copy saves. Embeddings, norms, biases and per-tensor scale
 * siblings stay exact: the copy only replaces what a verifier's matmuls read, and it never decides a token.
 *
 * <p>A matrix is restated when it is a stored {@code F32}/{@code F16}/{@code BF16} rank-2 tensor whose row
 * length is a whole number of the copy type's blocks and whose bytes the copy strictly undercuts; anything else
 * (an already-quantized matrix, a row length between blocks) keeps its stored type. The copy is only worth its
 * cost when it streams far fewer bytes than the target — a {@code BF16}/{@code F16}/F32 target — so a target
 * that already streams no more than the copy (8-bit or denser storage at {@code bits=8}, denser than Q6_K at
 * {@code bits=6}) is refused by name, as is a target with routed experts, whose expert tables would dominate
 * the verify pass either way.
 *
 * <p>The tensor table is planned from metadata alone; the quantized bytes are built when a store or region
 * first reads them, so a context fit can measure the copy without encoding anything. The wrapper holds no
 * resources: the target source stays with its owner, the stores and regions it hands out close what they
 * opened, and the payloads are byte arrays. Immutable after construction; the payload build is synchronized.
 */
public final class QCopySource implements ModelSource {

    private static final Logger LOGGER = LogManager.getLogger(QCopySource.class);

    /** The projection matrices of one main-stack layer the copy restates, by their tensor suffix. */
    private static final List<String> LAYER_MATRICES = Arrays.asList(
            TensorNames.weight(TensorNames.ATTN_Q), TensorNames.weight(TensorNames.ATTN_K),
            TensorNames.weight(TensorNames.ATTN_V), TensorNames.weight(TensorNames.ATTN_QKV),
            TensorNames.weight(TensorNames.ATTN_OUT), TensorNames.weight(TensorNames.FFN_GATE),
            TensorNames.weight(TensorNames.FFN_UP), TensorNames.weight(TensorNames.FFN_DOWN));

    /** Routed-expert tensors whose presence refuses the copy (the Q copy covers dense stacks). */
    private static final List<String> ROUTED_EXPERTS = Arrays.asList(
            TensorNames.weight(TensorNames.FFN_GATE_EXPS), TensorNames.weight(TensorNames.FFN_UP_EXPS),
            TensorNames.weight(TensorNames.FFN_GATE_UP_EXPS), TensorNames.weight(TensorNames.FFN_DOWN_EXPS),
            TensorNames.weight(TensorNames.FFN_GATE_INP));

    /** GGUF tensor-data alignment the copied offsets follow. */
    private static final long ALIGNMENT = 32;

    private final ModelSource target;
    private final List<GGUFTensorInfo> tensors;
    private final TreeMap<Long, GGUFTensorInfo> copiedByOffset = new TreeMap<>();
    private final List<Copied> plan;
    private volatile Map<Long, byte[]> payloads;

    private QCopySource(ModelSource target, List<GGUFTensorInfo> tensors, List<Copied> plan) {
        this.target = target;
        this.tensors = tensors;
        this.plan = plan;
        for (Copied c : plan) {
            copiedByOffset.put(c.info.offset(), c.info);
        }
    }

    /** One copied tensor: its table entry and the quantized bytes to build over the target's regions. */
    private static final class Copied {
        private final GGUFTensorInfo info;
        private final GGUFTensorInfo source;
        private final GGMLType sourceType;
        private final int cols;
        private final int rows;

        Copied(GGUFTensorInfo info, GGUFTensorInfo source, GGMLType sourceType, int cols, int rows) {
            this.info = info;
            this.source = source;
            this.sourceType = sourceType;
            this.cols = cols;
            this.rows = rows;
        }

        /** The tensor's quantized bytes: one decoded target row at a time into the copy's row quantizer. */
        byte[] build(TensorRegions regions) throws IOException {
            long bytes = source.sizeInBytes();
            if (regions.contiguousRunBytes(source) != bytes) {
                throw new IllegalArgumentException("tensor " + source.name() + " of the target is stored in"
                        + " pieces, which a quantized copy cannot restate");
            }
            CpuTensorRef ref = CpuTensorRef.of(sourceType, regions.region(source, 0L, bytes), rows, cols);
            int rowBytes = RowQuantizers.bytesPerRow(info.type(), cols);
            byte[] out = new byte[(int) bytes];
            float[] row = new float[cols];
            for (int r = 0; r < rows; r++) {
                ref.decodeRow(r, row, 0);
                RowQuantizers.quantizeRow(info.type(), row, 0, out, r * rowBytes, cols, null);
            }
            return out;
        }
    }

    /**
     * The metadata-only size of the copy of {@code target} at {@code bits}, for a context fit or a placement
     * budget that reserves the copy beside the target's weights; 0 when no matrix is eligible or the target is
     * one the attach refuses with its own message, so a load never fails for it here.
     *
     * @throws IllegalArgumentException naming {@code neuralfree.draft.tlv.bits} for a width that is not 6 or 8
     */
    public static long plannedBytes(ModelSource target, int bits) {
        Objects.requireNonNull(target, "target");
        if (bits != 6 && bits != 8) {
            throw new IllegalArgumentException(SpeculativeOptions.PROP_TLV_BITS + " must be 6 (Q6_K rows) or 8"
                    + " (Q8_0 rows), got " + bits);
        }
        try {
            return planOpen(target, copyType(bits)).bytes;
        } catch (UnsupportedOperationException e) {
            return 0L;
        }
    }

    /**
     * The quantized copy of {@code target} at {@code bits}; the caller keeps the ownership of {@code target}.
     *
     * @param bits the copy's bits per weight, 6 ({@code Q6_K}) or 8 ({@code Q8_0})
     * @throws IllegalArgumentException      naming {@code neuralfree.draft.tlv.bits} for a width that is not 6
     *                                       or 8
     * @throws UnsupportedOperationException naming {@code neuralfree.draft.tlv} when the target carries routed
     *                                       experts or streams no more than the copy would
     */
    public static QCopySource open(ModelSource target, int bits) {
        Objects.requireNonNull(target, "target");
        if (bits != 6 && bits != 8) {
            throw new IllegalArgumentException(SpeculativeOptions.PROP_TLV_BITS + " must be 6 (Q6_K rows) or 8"
                    + " (Q8_0 rows), got " + bits);
        }
        Plan plan = planOpen(target, copyType(bits));
        return new QCopySource(target, plan.tensors, plan.copied);
    }

    /** The copy's GGML type at {@code bits}. */
    private static GGMLType copyType(int bits) {
        return bits == 6 ? GGMLType.Q6_K : GGMLType.Q8_0;
    }

    /** The copy's total bytes: its restated matrices' quantized sizes. */
    public long plannedBytes() {
        long bytes = 0;
        for (Copied c : plan) {
            bytes += c.info.sizeInBytes();
        }
        return bytes;
    }

    /** One planning pass: the restated tensor table, the copied tensors and their total bytes. */
    private static final class Plan {
        private final List<GGUFTensorInfo> tensors;
        private final List<Copied> copied;
        private final long bytes;

        Plan(List<GGUFTensorInfo> tensors, List<Copied> copied, long bytes) {
            this.tensors = tensors;
            this.copied = copied;
            this.bytes = bytes;
        }
    }

    /**
     * Plans the copy of {@code target} at the copy type: the eligible matrices in file order (each trunk
     * layer's attention and feed-forward matrices, then the output head), each at its aligned offset behind
     * the target's data, plus the restated tensor table and the copy's total bytes.
     *
     * @throws UnsupportedOperationException naming {@code neuralfree.draft.tlv} for a routed-expert target or
     *                                       one the copy does not undercut
     */
    private static Plan planOpen(ModelSource target, GGMLType qType) {
        TensorCatalog catalog = TensorCatalog.of(target);
        String refusal = routedExpertsRefusal(catalog);
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        long next = ALIGNMENT;
        for (GGUFTensorInfo info : target.getTensors()) {
            next = Math.max(next, align(info.offset() + info.sizeInBytes()));
        }
        List<Copied> copied = new ArrayList<>();
        Map<String, GGUFTensorInfo> restated = new LinkedHashMap<>();
        for (int il = 0; il < catalog.blockCount(); il++) {
            for (String suffix : LAYER_MATRICES) {
                TensorRef ref = catalog.layer(il, suffix);
                if (ref == null) {
                    continue;
                }
                Copied copy = copyOf(catalog.info(ref), qType, next);
                if (copy != null) {
                    copied.add(copy);
                    restated.put(copy.info.name(), copy.info);
                    next = align(next + copy.info.sizeInBytes());
                }
            }
        }
        TensorRef head = catalog.get(TensorNames.weight(TensorNames.OUTPUT));
        if (head != null) {
            GGUFTensorInfo info = catalog.info(head);
            if (!restated.containsKey(info.name())) {
                Copied copy = copyOf(info, qType, next);
                if (copy != null) {
                    copied.add(copy);
                    restated.put(info.name(), copy.info);
                    next = align(next + copy.info.sizeInBytes());
                }
            }
        }
        if (copied.isEmpty()) {
            throw new UnsupportedOperationException(SpeculativeOptions.PROP_TLV + " builds a quantized copy of"
                    + " the target's projection matrices, but no matrix of " + target.getFilePath()
                    + " streams more than the copy would: every eligible matrix is stored at or below the copy's "
                    + qType + " rate already, or no matrix's rows fit the copy's blocks; lower "
                    + SpeculativeOptions.PROP_TLV_BITS + ", or run without the copy");
        }
        long bytes = 0;
        for (Copied c : copied) {
            bytes += c.info.sizeInBytes();
        }
        List<GGUFTensorInfo> tensors = new ArrayList<>();
        for (GGUFTensorInfo info : target.getTensors()) {
            GGUFTensorInfo swap = restated.get(info.name());
            tensors.add(swap != null ? swap : info);
        }
        LOGGER.info("quantized target view of {}: {} matrices restated as {} rows ({} bits), {} bytes to"
                        + " stream", target.getFilePath(), copied.size(), qType, bits(qType), bytes);
        return new Plan(Collections.unmodifiableList(tensors), Collections.unmodifiableList(copied), bytes);
    }

    /** The bit width a copy type stands for, for messages. */
    private static int bits(GGMLType qType) {
        return qType == GGMLType.Q6_K ? 6 : 8;
    }

    /** Why the copy refuses {@code catalog}'s target, or null: a routed-expert table would dominate it. */
    private static String routedExpertsRefusal(TensorCatalog catalog) {
        for (int il = 0; il < catalog.blockCount(); il++) {
            for (String suffix : ROUTED_EXPERTS) {
                TensorRef ref = catalog.layer(il, suffix);
                if (ref != null) {
                    return SpeculativeOptions.PROP_TLV + " quantizes a dense target's projection matrices, but "
                            + ref.name() + " holds routed experts, whose tables the copy neither restates nor"
                            + " undercuts; run two-level verification on a dense model";
                }
            }
        }
        return null;
    }

    /**
     * The copied tensor restating {@code source} at {@code offset}, or null when the copy does not undercut
     * it: a stored {@code F32}/{@code F16}/{@code BF16} rank-2 matrix whose row length takes whole copy blocks
     * and whose bytes the copy strictly reduces.
     */
    private static Copied copyOf(GGUFTensorInfo source, GGMLType qType, long offset) {
        if (source.rank() != 2 || !isFloat(source.type()) || source.dim(0) > Integer.MAX_VALUE - 8
                || source.dim(1) > Integer.MAX_VALUE - 8) {
            return null;
        }
        int cols = (int) source.dim(0);
        int rows = (int) source.dim(1);
        if (cols % qType.getBlockSize() != 0) {
            return null;
        }
        long copiedBytes = (long) RowQuantizers.bytesPerRow(qType, cols) * rows;
        if (copiedBytes >= source.sizeInBytes()) {
            return null;
        }
        GGUFTensorInfo info = new GGUFTensorInfo(source.name(), source.dims(), qType, offset);
        return new Copied(info, source, source.type(), cols, rows);
    }

    private static boolean isFloat(GGMLType type) {
        return type == GGMLType.F32 || type == GGMLType.F16 || type == GGMLType.BF16;
    }

    // ------------------------------------------------------------------ payloads

    private static long align(long offset) {
        long rem = offset % ALIGNMENT;
        return rem == 0 ? offset : offset + ALIGNMENT - rem;
    }

    private Map<Long, byte[]> ensurePayloads() throws IOException {
        Map<Long, byte[]> built = payloads;
        if (built != null) {
            return built;
        }
        synchronized (this) {
            if (payloads == null) {
                payloads = buildPayloads();
            }
            return payloads;
        }
    }

    private Map<Long, byte[]> buildPayloads() throws IOException {
        Map<Long, byte[]> out = new LinkedHashMap<>();
        try (TensorRegions regions = target.openRegions(TensorReadMode.MMAP)) {
            for (Copied c : plan) {
                out.put(c.info.offset(), c.build(regions));
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ ModelSource

    @Override
    public ModelFormat format() {
        return target.format();
    }

    @Override
    public String getFilePath() {
        return target.getFilePath();
    }

    @Override
    public Map<String, Object> getMetadata() {
        return target.getMetadata();
    }

    @Override
    public GGUFValueType getValueType(String key) {
        return target.getValueType(key);
    }

    @Override
    public GGUFValueType getArrayElementType(String key) {
        return target.getArrayElementType(key);
    }

    @Override
    public String getString(String key) {
        return target.getString(key);
    }

    @Override
    public List<GGUFTensorInfo> getTensors() {
        return tensors;
    }

    @Override
    public long getTensorCount() {
        return tensors.size();
    }

    @Override
    public WeightStore createMappedStore(StoreUse use) throws IOException {
        return new CopiedStore(target.createMappedStore(use));
    }

    @Override
    public TensorRegions openRegions(TensorReadMode mode) throws IOException {
        return new CopiedRegions(target.openRegions(mode));
    }

    /**
     * Closes nothing: the copied payloads are byte arrays and the target source stays with its owner, whose
     * close already covers it. The stores and regions this source hands out close what they opened.
     */
    @Override
    public void close() {
    }

    /** True when the byte range lies inside a copied tensor. */
    private boolean isCopied(long offset, int size) {
        Map.Entry<Long, GGUFTensorInfo> e = copiedByOffset.floorEntry(offset);
        return e != null && offset >= e.getKey() && offset + size <= e.getKey() + e.getValue().sizeInBytes();
    }

    /** The slice of a copied tensor's payload the byte range covers. */
    private byte[] payload(long offset, int size) throws IOException {
        Map<Long, byte[]> built = ensurePayloads();
        long start = copiedByOffset.floorEntry(offset).getKey();
        return Arrays.copyOfRange(built.get(start), (int) (offset - start), (int) (offset - start) + size);
    }

    /** The store of a copy source: copied tensors from their payloads, the target's from the mapped store. */
    private final class CopiedStore implements WeightStore {
        private final WeightStore store;

        CopiedStore(WeightStore store) {
            this.store = store;
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst) throws IOException {
            readTensor(tensorOffset, size, dst, 0);
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) throws IOException {
            if (isCopied(tensorOffset, size)) {
                System.arraycopy(payload(tensorOffset, size), 0, dst, dstOffset, size);
                return;
            }
            store.readTensor(tensorOffset, size, dst, dstOffset);
        }

        @Override
        public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) throws IOException {
            if (isCopied(tensorOffset, size)) {
                dst.put(payload(tensorOffset, size));
                return;
            }
            store.readTensorDirect(tensorOffset, size, dst);
        }

        @Override
        public void close() throws IOException {
            store.close();
        }
    }

    /** The regions of a copy source: copied tensors from their payloads, the target's from the mapped regions. */
    private final class CopiedRegions implements TensorRegions {
        private final TensorRegions regions;

        CopiedRegions(TensorRegions regions) {
            this.regions = regions;
        }

        @Override
        public ByteBuffer region(GGUFTensorInfo info, long byteOff, long length) throws IOException {
            if (isCopied(info.offset(), (int) info.sizeInBytes())) {
                ByteBuffer b = ByteBuffer.wrap(ensurePayloads().get(info.offset()));
                b.position((int) byteOff);
                b.limit((int) (byteOff + length));
                return b.slice().order(ByteOrder.LITTLE_ENDIAN);
            }
            return regions.region(info, byteOff, length);
        }

        @Override
        public long contiguousRunBytes(GGUFTensorInfo info) {
            return isCopied(info.offset(), (int) info.sizeInBytes()) ? info.sizeInBytes()
                    : regions.contiguousRunBytes(info);
        }

        @Override
        public String description() {
            return regions.description() + " with the quantized target view";
        }

        @Override
        public void close() throws IOException {
            regions.close();
        }
    }
}
