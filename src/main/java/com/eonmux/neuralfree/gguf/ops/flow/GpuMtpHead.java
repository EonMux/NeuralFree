package com.eonmux.neuralfree.gguf.ops.flow;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuLayerExecutor;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.ops.hc.GpuHcSite;
import com.eonmux.neuralfree.gguf.ops.hc.HcSubBlock;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopePositions;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Head on a target's device: rows are named scratch temporaries, work is enqueued on the target's queue. */
final class GpuMtpHead extends MtpHead {
    private static final String PREFIX = "flow.draft.";

    private final GpuForwardContext ctx;
    private final GpuBlockRegistry registry;
    private final GpuWeights draft;
    private final GpuOps ops;
    private final GpuScratch scratch;
    private final ComputeQueue queue;
    private final Map<String, GpuRows> rows = new HashMap<>();

    GpuMtpHead(GpuForwardContext ctx, GpuBlockRegistry registry, GpuWeights draft) {
        super(Objects.requireNonNull(ctx, "target context").spec());
        this.ctx = ctx;
        this.registry = Objects.requireNonNull(registry, "registry");
        this.draft = draft;
        this.ops = ctx.ops();
        this.scratch = ctx.scratch();
        this.queue = ctx.queue();
    }

    private static final class GpuRows extends Rows {
        final ScratchSlot slot;

        GpuRows(String name, int width, ScratchSlot slot) {
            super(name, width);
            this.slot = slot;
        }

        GpuBuffer buf() { return slot.buffer(); }

        int off() { return slot.offset(); }
    }

    private GpuRows cast(Rows r) {
        if (!(r instanceof GpuRows) || rows.get(r.name()) != r) {
            throw new IllegalArgumentException(r + " does not belong to this head");
        }
        return (GpuRows) r;
    }

    private GpuWeights weights(Source s) {
        if (s == Source.TARGET) {
            return ctx.weights();
        }
        if (draft == null) {
            throw new IllegalStateException("this head has no drafter weights");
        }
        return draft;
    }

    private GpuWeight weight(Source s, String name) {
        return weights(s).get(name);
    }

    @Override
    public int ubatch() {
        return scratch.ubatch();
    }

    @Override
    public Rows rows(String name, int width) {
        checkOpen();
        GpuRows r = rows.get(name);
        if (r != null) {
            if (r.width() != width) {
                throw new IllegalArgumentException("rows " + name + " already hold " + r.width()
                        + " floats per row");
            }
            return r;
        }
        if (width < 1) {
            throw new IllegalArgumentException("rows " + name + " need a positive width, got " + width);
        }
        r = new GpuRows(name, width, scratch.temp(PREFIX + name, (long) ubatch() * width));
        rows.put(name, r);
        return r;
    }

    @Override
    public boolean has(Source s, String name) {
        return (s == Source.TARGET || draft != null) && weights(s).getOrNull(name) != null;
    }

    @Override
    public long[] shape(Source s, String name) {
        return weight(s, name).ne();
    }

    @Override
    public float[] vector(Source s, String name) {
        GpuWeight w = weight(s, name);
        if (w.type() != GGMLType.F32 || !w.isSingleShard()) {
            throw new IllegalArgumentException(name + " must be a single-buffer F32 tensor, got " + w);
        }
        long n = w.rows() * w.cols();
        if (n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(name + " has " + n + " elements");
        }
        float[] out = new float[(int) n];
        release(queue.enqueueReadBuffer(w.shard(0), 0L, n * Float.BYTES, out));
        queue.finish();
        return out;
    }

    @Override
    public int[] integers(Source s, String name) {
        GpuWeight w = weight(s, name);
        requireIntegerType(name, w.type());
        if (!w.isSingleShard()) {
            throw new IllegalArgumentException(name + " must be one buffer, got " + w);
        }
        long bytes = w.sizeInBytes();
        if (bytes > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(name + " has " + bytes + " bytes");
        }
        byte[] raw = new byte[(int) bytes];
        release(queue.enqueueReadBuffer(w.shard(0), 0L, bytes, raw));
        queue.finish();
        int size = w.type() == GGMLType.I32 ? 4 : 8;
        return toInts(name, w.type(), ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN), bytes / size);
    }

    @Override
    public void write(Rows dst, float[] src, int nRows) {
        GpuRows d = cast(dst);
        checkRows(d, nRows, ubatch());
        long n = (long) nRows * d.width();
        if (src.length < n) {
            throw new IllegalArgumentException("source holds " + src.length + " floats, " + n + " needed");
        }
        release(queue.enqueueWriteBuffer(d.buf(), (long) d.off() * Float.BYTES, n * Float.BYTES, src));
    }

    @Override
    public void read(Rows src, int nRows, float[] dst) {
        GpuRows r = cast(src);
        checkRows(r, nRows, ubatch());
        long n = (long) nRows * r.width();
        readFloats(r.buf(), r.off(), n, dst);
    }

    @Override
    public void readRow(Rows src, int row, float[] dst) {
        GpuRows r = cast(src);
        checkRows(r, row + 1, ubatch());
        readFloats(r.buf(), r.off() + (long) row * r.width(), r.width(), dst);
    }

    private void readFloats(GpuBuffer buf, long off, long n, float[] dst) {
        if (dst.length < n) {
            throw new IllegalArgumentException("destination holds " + dst.length + " floats, " + n + " needed");
        }
        release(queue.enqueueReadBuffer(buf, off * Float.BYTES, n * Float.BYTES, dst));
        queue.finish();
    }

    @Override
    public void embed(Source s, String table, int[] tokens, int n, Rows out) {
        GpuRows o = cast(out);
        checkRows(o, n, ubatch());
        GpuWeight w = weight(s, table);
        checkWidth(o, w.cols(), table);
        ops.embed(w, Arrays.copyOf(tokens, n), o.buf(), o.off());
    }

    @Override
    public void norm(NormKind kind, Source s, String weight, float eps, Rows in, Rows out, int n) {
        GpuRows i = cast(in);
        GpuRows o = cast(out);
        checkRows(i, n, ubatch());
        checkWidth(o, i.width(), "norm of " + i.name());
        GpuWeight w = weight == null ? null : weight(s, weight);
        ops.norm(kind, w, null, eps, i.buf(), i.off(), o.buf(), o.off(), n, i.width());
    }

    @Override
    public void concat(Rows a, Rows b, Rows out, int n) {
        GpuRows x = cast(a);
        GpuRows y = cast(b);
        GpuRows o = cast(out);
        checkRows(o, n, ubatch());
        checkWidth(o, (long) x.width() + y.width(), "concat of " + x.name() + " and " + y.name());
        ops.copy2d(x.buf(), x.off(), x.width(), o.buf(), o.off(), o.width(), n, x.width());
        ops.copy2d(y.buf(), y.off(), y.width(), o.buf(), o.off() + x.width(), o.width(), n, y.width());
    }

    @Override
    public void project(Source s, String weight, Rows in, Rows out, int n) {
        GpuRows i = cast(in);
        GpuRows o = cast(out);
        checkRows(i, n, ubatch());
        GpuWeight w = weight(s, weight);
        checkWidth(i, w.cols(), "input of " + weight);
        checkWidth(o, w.rows(), weight);
        ops.matmul(w, i.buf(), i.off(), o.buf(), o.off(), n);
    }

    @Override
    public void add(Rows a, Rows b, Rows out, int n) {
        GpuRows x = cast(a);
        GpuRows y = cast(b);
        GpuRows o = cast(out);
        checkRows(o, n, ubatch());
        checkWidth(x, o.width(), "add into " + o.name());
        checkWidth(y, o.width(), "add into " + o.name());
        ops.add(x.buf(), x.off(), y.buf(), y.off(), o.buf(), o.off(), n * o.width());
    }

    @Override
    public void scale(Rows x, float s, int n) {
        GpuRows r = cast(x);
        checkRows(r, n, ubatch());
        ops.scale(r.buf(), r.off(), n * r.width(), s);
    }

    /**
     * One {@code LLM_RowNorm} over the {@code n * streams} parts; part {@code r} uses weight row
     * {@code r % weightRows}. The weight is addressed by its element count, so a per-stream weight may be stored
     * {@code [part, streams]} or flat, as on the host.
     */
    @Override
    public void normStreams(NormKind kind, Source s, String weight, float eps, Rows in, Rows out, int n,
                            int streams) {
        GpuRows i = cast(in);
        GpuRows o = cast(out);
        checkRows(i, n, ubatch());
        checkWidth(o, i.width(), "norm of " + i.name());
        int part = streamPart(i, streams);
        int nRows = Math.multiplyExact(n, streams);
        GpuWeight w = weight == null ? null : weight(s, weight);
        int wRows = 1;
        if (w != null) {
            if (w.type() != GGMLType.F32 || !w.isSingleShard()) {
                throw new IllegalArgumentException(weight + " must be a single-buffer F32 tensor, got " + w);
            }
            wRows = streamWeightRows(weight, w.rows() * w.cols(), part, streams);
        }
        int local = ops.local();
        ops.enqueue(GpuOps.ROW_NORM, new long[]{(long) nRows * local}, new long[]{local},
                NormKernels.rowNormArgs(kind, eps, nRows, part, part, i.buf(), i.off(), o.buf(), o.off(),
                        w != null ? w.shard(0) : null, null, wRows, ops.placeholders().readOnlyFloats()));
    }

    /** {@code streams} copies of {@code a} and one strided copy of every part of {@code b}. */
    @Override
    public void concatStreams(Rows a, Rows b, Rows out, int n, int streams) {
        GpuRows x = cast(a);
        GpuRows y = cast(b);
        GpuRows o = cast(out);
        checkRows(o, n, ubatch());
        int partB = streamPart(y, streams);
        int part = x.width() + partB;
        checkWidth(o, (long) streams * part, "stream concat of " + x.name() + " and " + y.name());
        for (int p = 0; p < streams; p++) {
            ops.copy2d(x.buf(), x.off(), x.width(), o.buf(), o.off() + p * part, o.width(), n, x.width());
        }
        ops.copy2d(y.buf(), y.off(), partB, o.buf(), o.off() + x.width(), part, Math.multiplyExact(n, streams),
                partB);
    }

    @Override
    public void projectStreams(Source s, String weight, Rows in, Rows out, int n, int streams) {
        GpuRows i = cast(in);
        GpuRows o = cast(out);
        checkRows(i, n, ubatch());
        GpuWeight w = weight(s, weight);
        checkWidth(i, (long) streams * w.cols(), "input of " + weight + " over " + streams + " streams");
        checkWidth(o, (long) streams * w.rows(), weight + " over " + streams + " streams");
        ops.matmul(w, i.buf(), i.off(), o.buf(), o.off(), Math.multiplyExact(n, streams));
    }

    @Override
    public Collapse collapse(Source s, HcSubBlock site) {
        checkOpen();
        GpuHcSite bound = own(new GpuHcSite(ctx, weights(s), Objects.requireNonNull(site, "site")));
        return (streams, out, n) -> {
            GpuRows i = cast(streams);
            GpuRows o = cast(out);
            checkRows(i, n, ubatch());
            checkWidth(i, site.streamWidth(), "input of " + site);
            checkWidth(o, site.nEmbd(), site.toString());
            bound.collapse(n, i.buf(), i.off(), o.buf(), o.off());
        };
    }

    @Override
    public void logits(Source s, String weight, Rows in, int row, float[] dst) {
        GpuRows i = cast(in);
        checkRows(i, row + 1, ubatch());
        GpuWeight w = weight(s, weight);
        checkWidth(i, w.cols(), "input of " + weight);
        if (w.rows() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(weight + " has " + w.rows() + " rows");
        }
        int vocab = (int) w.rows();
        ScratchSlot out = scratch.temp(PREFIX + "logits", vocab);
        ops.matvec(w, i.buf(), i.off() + row * i.width(), out.buffer(), out.offset(), 1);
        readFloats(out.buffer(), out.offset(), vocab, dst);
    }

    @Override
    public Memory ownMemory(ModelSpec spec, MemoryPlan plan) {
        checkOpen();
        DeviceModelMemory m = DeviceModelMemory.allocate(ctx.backend(), queue, plan,
                spec.hparams().mropeTextPositions(), null);
        return own(new GpuOwnMemory(m, spec.hparams().mropeTextPositions()));
    }

    @Override
    public Memory targetCaches(ModelSpec spec, MemoryPlan plan) {
        checkOpen();
        GpuMemory targetMemory = ctx.memory();
        DeviceModelMemory m = DeviceModelMemory.allocate(ctx.backend(), queue, plan,
                spec.hparams().mropeTextPositions(), targetMemory);
        return own(new GpuViewMemory(new SharedGpuMemory(m, targetMemory), spec.hparams().mropeTextPositions()));
    }

    private abstract class GpuDraftMemory extends Memory {
        private final MropeTextPositions textPositions;

        GpuDraftMemory(boolean owns, MropeTextPositions textPositions) {
            super(owns);
            this.textPositions = textPositions;
        }

        abstract GpuMemory gpu();

        @Override
        public ModelMemory memory() {
            return gpu();
        }

        @Override
        void begin(int pos0, int n, int[] tokens) {
            gpu().stepInputs().upload(pos0, n);
            writeTokenIds(tokens, n);
            writePositions(textPositions, pos0, n);
        }
    }

    private final class GpuOwnMemory extends GpuDraftMemory {
        private final DeviceModelMemory m;

        GpuOwnMemory(DeviceModelMemory m, MropeTextPositions textPositions) {
            super(true, textPositions);
            this.m = m;
        }

        @Override
        GpuMemory gpu() { return m; }
    }

    private final class GpuViewMemory extends GpuDraftMemory {
        private final SharedGpuMemory m;

        GpuViewMemory(SharedGpuMemory m, MropeTextPositions textPositions) {
            super(false, textPositions);
            this.m = m;
        }

        @Override
        GpuMemory gpu() { return m; }
    }

    private void writeTokenIds(int[] tokens, int n) {
        scratch.recordStepTokens(tokens, n);
        ScratchSlot slot = scratch.slot(ScratchSlots.TOKEN_IDS);
        ByteBuffer b = ByteBuffer.allocate(n * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int t = 0; t < n; t++) {
            b.putInt(tokens[t]);
        }
        release(queue.enqueueWriteBuffer(slot.buffer(), (long) slot.offset() * Float.BYTES,
                (long) n * Integer.BYTES, b.array()));
    }

    /** The step's four-section positions into {@link ScratchSlots#POS}, as the prefill driver writes them. */
    private void writePositions(MropeTextPositions textPositions, int pos0, int n) {
        int count = MropeTextPositions.SECTIONS * n;
        int[] pos = new int[count];
        RopePositions.fillStep(textPositions, pos0, n, pos);
        ScratchSlot slot = scratch.slot(ScratchSlots.POS);
        ByteBuffer b = ByteBuffer.allocate(count * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < count; i++) {
            b.putInt(pos[i]);
        }
        release(queue.enqueueWriteBuffer(slot.buffer(), (long) slot.offset() * Float.BYTES,
                (long) count * Integer.BYTES, b.array()));
    }

    private GpuForwardContext context(ModelSpec spec, Source s, Memory memory) {
        if (!(memory instanceof GpuDraftMemory)) {
            throw new IllegalArgumentException(memory + " was not made by this head");
        }
        GpuForwardContext.Builder b = GpuForwardContext.builder(ctx.kernels(), ops, weights(s), spec)
                .memory(((GpuDraftMemory) memory).gpu())
                .observer(ctx.observer());
        if (s == Source.TARGET) {
            b.weightSource(ctx.weightSource());
        }
        return b.build();
    }

    @Override
    public Stack stack(ModelSpec spec, Source s, Memory memory, List<LayerSpec> layers) {
        checkOpen();
        GpuLayerExecutor ex = GpuLayerExecutor.forStack(context(spec, s, memory), registry, layers);
        int width = layers.get(0).residual().streamWidth();
        if (ex.outputWidth() != width) {
            ex.close();
            throw new IllegalArgumentException("draft stack maps rows of " + width + " to " + ex.outputWidth());
        }
        ScratchSlot x = scratch.slot(ScratchSlots.X);
        return own(new Stack() {
            @Override
            public void run(int pos0, int n, int[] tokens, Rows rows) {
                GpuRows r = cast(rows);
                checkRows(r, n, ubatch());
                checkWidth(r, width, "draft stack");
                memory.begin(pos0, n, tokens);
                ops.copy(r.buf(), r.off(), x.buffer(), x.offset(), n * width);
                ex.run(pos0, n, n == 1 ? GpuLayerExecutor.StepMode.DECODE : GpuLayerExecutor.StepMode.PREFILL,
                        GpuLayerExecutor.TapSink.NONE);
                ops.copy(x.buffer(), x.offset(), r.buf(), r.off(), n * width);
                memory.end(pos0, n, tokens);
            }

            @Override
            public void close() {
                ex.close();
            }
        });
    }

    @Override
    public Blocks blocks(ModelSpec spec, Source s, Memory memory, LayerSpec layer) {
        checkOpen();
        GpuForwardContext c = context(spec, s, memory);
        GpuMixer mixer = registry.mixer(layer);
        GpuFfnBlock ffn;
        try {
            mixer.bind(c, layer);
            ffn = registry.ffn(layer);
        } catch (RuntimeException e) {
            mixer.close();
            throw e;
        }
        try {
            ffn.bind(c, layer);
        } catch (RuntimeException e) {
            ffn.close();
            mixer.close();
            throw e;
        }
        return own(new Blocks() {
            @Override
            public void mixer(int pos0, int n, int[] tokens, Rows in, Rows out) {
                GpuRows i = cast(in);
                GpuRows o = cast(out);
                checkRows(i, n, ubatch());
                memory.begin(pos0, n, tokens);
                if (n == 1) {
                    mixer.decode(pos0, i.slot, o.slot);
                } else {
                    mixer.prefill(pos0, n, i.slot, o.slot);
                }
                memory.end(pos0, n, tokens);
            }

            @Override
            public void ffn(int n, Rows in, Rows out) {
                GpuRows i = cast(in);
                GpuRows o = cast(out);
                checkRows(i, n, ubatch());
                ffn.forward(n, i.slot, o.slot);
            }

            @Override
            public void close() {
                try {
                    ffn.close();
                } finally {
                    mixer.close();
                }
            }
        });
    }

    private static void release(GpuEvent e) {
        if (e != null) {
            e.close();
        }
    }

    @Override
    public String toString() {
        return "GpuMtpHead{" + target().arch().ggufName() + ", ubatch=" + ubatch()
                + (draft != null ? ", drafter weights" : "") + '}';
    }
}
