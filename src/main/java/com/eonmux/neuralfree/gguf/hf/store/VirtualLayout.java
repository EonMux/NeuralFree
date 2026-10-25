package com.eonmux.neuralfree.gguf.hf.store;

import com.eonmux.neuralfree.gguf.DataSegment;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainer;
import com.eonmux.neuralfree.gguf.hf.tensor.PlannedTensor;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorPlan;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The virtual tensor-data address space of a Hugging Face model source. The checkpoint files occupy
 * {@code [0, P)} as their {@link DataSegment}s (file {@code f} at {@code V_f = Σ_{g<f} align64(dataLength_g)}); a
 * planned tensor whose source is stored unchanged ({@link RowSource#directVirtualOffset()} {@code >= 0} and the stored
 * type is the planned type) keeps that offset, which need not be aligned. Every other tensor is produced: it gets a
 * 64-byte-aligned slot in {@code [P, totalBytes())}, in plan order, starting at
 * {@code P = V_last + align64(dataLength_last)}. Immutable.
 */
public final class VirtualLayout {

    /** Alignment of the checkpoint files and of every produced slot. */
    public static final long ALIGNMENT = 64;

    private final List<DataSegment> segments;
    private final List<GGUFTensorInfo> tensors;
    private final ProducedSegments produced;
    private final long producedBase;
    private final long totalBytes;

    private VirtualLayout(List<DataSegment> segments, List<GGUFTensorInfo> tensors, ProducedSegments produced,
                          long producedBase, long totalBytes) {
        this.segments = segments;
        this.tensors = tensors;
        this.produced = produced;
        this.producedBase = producedBase;
        this.totalBytes = totalBytes;
    }

    /** The layout of {@code plan} over the files of {@code ckpt}; see {@link #of(List, TensorPlan)}. */
    public static VirtualLayout of(CheckpointContainer ckpt, TensorPlan plan) {
        return of(Objects.requireNonNull(ckpt, "ckpt").segments(), plan);
    }

    /**
     * The layout of {@code plan} over the checkpoint {@code segments}.
     *
     * @throws IllegalArgumentException when the segments are out of order or overlap
     * @throws IllegalStateException    when a direct tensor's bytes do not lie inside one segment, or its stored size
     *                                  differs from its planned size
     * @throws ArithmeticException      when an offset overflows
     */
    public static VirtualLayout of(List<DataSegment> segments, TensorPlan plan) {
        Objects.requireNonNull(segments, "segments");
        Objects.requireNonNull(plan, "plan");
        List<DataSegment> segs = Collections.unmodifiableList(new ArrayList<>(segments));
        long base = producedBase(segs);
        List<GGUFTensorInfo> infos = new ArrayList<>(plan.tensors().size());
        List<PlannedTensor> derived = new ArrayList<>();
        List<Long> derivedOffsets = new ArrayList<>();
        long next = base;
        for (PlannedTensor t : plan.tensors()) {
            long direct = directOffset(t);
            long offset;
            if (direct >= 0) {
                checkDirect(t, direct, segs);
                offset = direct;
            } else {
                offset = next;
                derived.add(t);
                derivedOffsets.add(offset);
                next = align(Math.addExact(offset, t.byteSize()));
            }
            infos.add(new GGUFTensorInfo(t.ggufName(), t.ne(), t.type(), offset));
        }
        long total = derived.isEmpty() ? base : Math.addExact(derivedOffsets.get(derived.size() - 1),
                derived.get(derived.size() - 1).byteSize());
        ProducedSegments ps = ProducedSegments.of(derived, derivedOffsets, base, total);
        return new VirtualLayout(segs, Collections.unmodifiableList(infos), ps, base, total);
    }

    /** {@code P}: the end of the last segment, aligned to {@link #ALIGNMENT}; 0 without segments. */
    static long producedBase(List<DataSegment> segs) {
        long end = 0;
        for (int i = 0; i < segs.size(); i++) {
            DataSegment s = Objects.requireNonNull(segs.get(i), "segment " + i);
            if (s.virtualBase() < end) {
                throw new IllegalArgumentException("data segment " + i + " (" + s + ") starts before the end " + end
                        + " of the previous one");
            }
            end = align(s.virtualEnd());
        }
        return end;
    }

    /** The stored offset of {@code t}, or -1 when it is produced. */
    private static long directOffset(PlannedTensor t) {
        RowSource src = t.source();
        long off = src.directVirtualOffset();
        return off >= 0 && src.format().ggmlType() == t.type() ? off : -1;
    }

    private static void checkDirect(PlannedTensor t, long offset, List<DataSegment> segs) {
        GGUFTensorInfo probe = new GGUFTensorInfo(t.ggufName(), t.ne(), t.type(), offset);
        if (probe.sizeInBytes() != t.byteSize()) {
            throw new IllegalStateException(t.ggufName() + ": " + probe.sizeInBytes() + " stored bytes differ from the"
                    + " planned " + t.byteSize());
        }
        for (DataSegment s : segs) {
            if (s.contains(offset, t.byteSize())) {
                return;
            }
        }
        throw new IllegalStateException(t.ggufName() + ": stored bytes [" + offset + ", +" + t.byteSize()
                + ") lie inside no checkpoint file");
    }

    /** {@code v} rounded up to {@link #ALIGNMENT}. */
    static long align(long v) {
        long r = v % ALIGNMENT;
        return r == 0 ? v : Math.addExact(v, ALIGNMENT - r);
    }

    /** The checkpoint segments, covering {@code [0, producedBase())}. */
    public List<DataSegment> segments() {
        return segments;
    }

    /** The tensor infos in plan order, with virtual offsets. */
    public List<GGUFTensorInfo> tensors() {
        return tensors;
    }

    /** The produced tensors and their slots. */
    public ProducedSegments produced() {
        return produced;
    }

    /** {@code P}: where the produced tensors start. */
    public long producedBase() {
        return producedBase;
    }

    /** The end of the address space: the end of the last produced tensor, or {@code P} when none is produced. */
    public long totalBytes() {
        return totalBytes;
    }

    @Override
    public String toString() {
        return "VirtualLayout{" + tensors.size() + " tensors, " + segments.size() + " files, " + produced.size()
                + " produced from " + producedBase + ", end " + totalBytes + "}";
    }
}
