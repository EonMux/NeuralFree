package com.eonmux.neuralfree.gguf.ops.flow;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuBlockRegistry;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnBlock;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuLayerExecutor;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.ops.hc.CpuHcSite;
import com.eonmux.neuralfree.gguf.ops.hc.HcSubBlock;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.ops.rope.RopePositions;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Head on a target's host runtime: rows are float arrays, layers run on the target's scratch. */
final class CpuMtpHead extends MtpHead {
    private final CpuForwardContext ctx;
    private final CpuBlockRegistry registry;
    private final CpuWeights draft;
    private final CpuScratch scratch;
    private final Map<String, CpuRows> rows = new HashMap<>();

    CpuMtpHead(CpuForwardContext ctx, CpuBlockRegistry registry, CpuWeights draft) {
        super(Objects.requireNonNull(ctx, "target context").spec());
        this.ctx = ctx;
        this.registry = Objects.requireNonNull(registry, "registry");
        this.draft = draft;
        this.scratch = ctx.scratch();
    }

    private static final class CpuRows extends Rows {
        final float[] data;

        CpuRows(String name, int width, float[] data) {
            super(name, width);
            this.data = data;
        }
    }

    private CpuRows cast(Rows r) {
        if (!(r instanceof CpuRows) || rows.get(r.name()) != r) {
            throw new IllegalArgumentException(r + " does not belong to this head");
        }
        return (CpuRows) r;
    }

    private CpuWeights weights(Source s) {
        if (s == Source.TARGET) {
            return ctx.weights();
        }
        if (draft == null) {
            throw new IllegalStateException("this head has no drafter weights");
        }
        return draft;
    }

    /** The target's configured kernels for its tensors, the drafter file's own for drafter tensors. */
    private CpuMatVec matVec(Source s) {
        return s == Source.TARGET ? ctx.matVec() : weights(s).matVec();
    }

    @Override
    public int ubatch() {
        return scratch.ubatch();
    }

    @Override
    public Rows rows(String name, int width) {
        checkOpen();
        CpuRows r = rows.get(name);
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
        r = new CpuRows(name, width, new float[Math.multiplyExact(ubatch(), width)]);
        rows.put(name, r);
        return r;
    }

    @Override
    public boolean has(Source s, String name) {
        return (s == Source.TARGET || draft != null) && weights(s).has(name);
    }

    @Override
    public long[] shape(Source s, String name) {
        CpuWeights w = weights(s);
        return w.describe(w.get(name)).ne();
    }

    @Override
    public float[] vector(Source s, String name) {
        return weights(s).floats(name).clone();
    }

    @Override
    public int[] integers(Source s, String name) {
        CpuWeights w = weights(s);
        CpuTensorRef t = w.get(name);
        TensorRef d = w.describe(t);
        requireIntegerType(name, d.type());
        int size = d.type() == GGMLType.I32 ? 4 : 8;
        long rowsCount = t.rows();
        long cols = t.cols();
        ByteBuffer all = ByteBuffer.allocate(Math.toIntExact(rowsCount * cols * size))
                .order(ByteOrder.LITTLE_ENDIAN);
        for (long r = 0; r < rowsCount; r++) {
            ByteBuffer b = t.rowBuffer(r).duplicate().order(ByteOrder.LITTLE_ENDIAN);
            int off = t.rowOffset(r);
            for (long c = 0; c < cols * size; c++) {
                all.put(b.get(off + (int) c));
            }
        }
        return toInts(name, d.type(), all, rowsCount * cols);
    }

    @Override
    public void write(Rows dst, float[] src, int nRows) {
        CpuRows d = cast(dst);
        checkRows(d, nRows, ubatch());
        System.arraycopy(src, 0, d.data, 0, nRows * d.width());
    }

    @Override
    public void read(Rows src, int nRows, float[] dst) {
        CpuRows r = cast(src);
        checkRows(r, nRows, ubatch());
        System.arraycopy(r.data, 0, dst, 0, nRows * r.width());
    }

    @Override
    public void readRow(Rows src, int row, float[] dst) {
        CpuRows r = cast(src);
        checkRows(r, row + 1, ubatch());
        System.arraycopy(r.data, row * r.width(), dst, 0, r.width());
    }

    @Override
    public void embed(Source s, String table, int[] tokens, int n, Rows out) {
        CpuRows o = cast(out);
        checkRows(o, n, ubatch());
        CpuWeights w = weights(s);
        CpuTensorRef t = w.get(table);
        checkWidth(o, t.cols(), table);
        for (int i = 0; i < n; i++) {
            if (tokens[i] < 0 || tokens[i] >= t.rows()) {
                throw new IllegalArgumentException("token id " + tokens[i] + " is outside the " + t.rows()
                        + " rows of " + table);
            }
            w.embedRow(t, tokens[i], o.data, i * o.width());
        }
    }

    @Override
    public void norm(NormKind kind, Source s, String weight, float eps, Rows in, Rows out, int n) {
        CpuRows i = cast(in);
        CpuRows o = cast(out);
        checkRows(i, n, ubatch());
        checkWidth(o, i.width(), "norm of " + i.name());
        float[] w = weight == null ? null : weights(s).floats(weight);
        if (w != null && w.length != i.width()) {
            throw new IllegalArgumentException(weight + " has " + w.length + " values, rows hold " + i.width());
        }
        CpuOps.norm(kind, eps, i.data, 0, o.data, 0, n, i.width(), w, null);
    }

    @Override
    public void concat(Rows a, Rows b, Rows out, int n) {
        CpuRows x = cast(a);
        CpuRows y = cast(b);
        CpuRows o = cast(out);
        checkRows(o, n, ubatch());
        checkWidth(o, (long) x.width() + y.width(), "concat of " + x.name() + " and " + y.name());
        for (int t = 0; t < n; t++) {
            System.arraycopy(x.data, t * x.width(), o.data, t * o.width(), x.width());
            System.arraycopy(y.data, t * y.width(), o.data, t * o.width() + x.width(), y.width());
        }
    }

    @Override
    public void project(Source s, String weight, Rows in, Rows out, int n) {
        CpuRows i = cast(in);
        CpuRows o = cast(out);
        checkRows(i, n, ubatch());
        CpuWeights w = weights(s);
        CpuTensorRef t = w.get(weight);
        checkWidth(i, t.cols(), "input of " + weight);
        checkWidth(o, t.rows(), weight);
        matVec(s).matmul(t, i.data, 0, o.data, 0, n);
    }

    @Override
    public void add(Rows a, Rows b, Rows out, int n) {
        CpuRows x = cast(a);
        CpuRows y = cast(b);
        CpuRows o = cast(out);
        checkRows(o, n, ubatch());
        checkWidth(x, o.width(), "add into " + o.name());
        checkWidth(y, o.width(), "add into " + o.name());
        CpuOps.add(x.data, 0, y.data, 0, o.data, 0, n * o.width());
    }

    @Override
    public void scale(Rows x, float s, int n) {
        CpuRows r = cast(x);
        checkRows(r, n, ubatch());
        CpuOps.scale(r.data, 0, n * r.width(), s);
    }

    @Override
    public void normStreams(NormKind kind, Source s, String weight, float eps, Rows in, Rows out, int n,
                            int streams) {
        CpuRows i = cast(in);
        CpuRows o = cast(out);
        checkRows(i, n, ubatch());
        checkWidth(o, i.width(), "norm of " + i.name());
        int part = streamPart(i, streams);
        float[] w = weight == null ? null : weights(s).floats(weight);
        int wRows = w == null ? 1 : streamWeightRows(weight, w.length, part, streams);
        NormMath.rowNorm(kind, eps, Math.multiplyExact(n, streams), part, part, wRows, i.data, 0, w, null, o.data, 0);
    }

    @Override
    public void concatStreams(Rows a, Rows b, Rows out, int n, int streams) {
        CpuRows x = cast(a);
        CpuRows y = cast(b);
        CpuRows o = cast(out);
        checkRows(o, n, ubatch());
        int partB = streamPart(y, streams);
        int part = x.width() + partB;
        checkWidth(o, (long) streams * part, "stream concat of " + x.name() + " and " + y.name());
        for (int t = 0; t < n; t++) {
            for (int p = 0; p < streams; p++) {
                int dst = t * o.width() + p * part;
                System.arraycopy(x.data, t * x.width(), o.data, dst, x.width());
                System.arraycopy(y.data, t * y.width() + p * partB, o.data, dst + x.width(), partB);
            }
        }
    }

    @Override
    public void projectStreams(Source s, String weight, Rows in, Rows out, int n, int streams) {
        CpuRows i = cast(in);
        CpuRows o = cast(out);
        checkRows(i, n, ubatch());
        CpuTensorRef t = weights(s).get(weight);
        checkWidth(i, (long) streams * t.cols(), "input of " + weight + " over " + streams + " streams");
        checkWidth(o, (long) streams * t.rows(), weight + " over " + streams + " streams");
        matVec(s).matmul(t, i.data, 0, o.data, 0, Math.multiplyExact(n, streams));
    }

    @Override
    public Collapse collapse(Source s, HcSubBlock site) {
        checkOpen();
        CpuForwardContext c = s == Source.TARGET ? ctx
                : CpuForwardContext.builder(target(), weights(s), scratch).matVec(matVec(s)).build();
        CpuHcSite bound = new CpuHcSite(c, Objects.requireNonNull(site, "site"));
        return (streams, out, n) -> {
            CpuRows i = cast(streams);
            CpuRows o = cast(out);
            checkRows(i, n, ubatch());
            checkWidth(i, site.streamWidth(), "input of " + site);
            checkWidth(o, site.nEmbd(), site.toString());
            bound.collapse(n, i.data, 0, o.data, 0);
        };
    }

    @Override
    public void logits(Source s, String weight, Rows in, int row, float[] dst) {
        CpuRows i = cast(in);
        checkRows(i, row + 1, ubatch());
        CpuWeights w = weights(s);
        CpuTensorRef t = w.get(weight);
        checkWidth(i, t.cols(), "input of " + weight);
        if (dst.length < t.rows()) {
            throw new IllegalArgumentException("destination holds " + dst.length + " floats, " + t.rows()
                    + " needed");
        }
        matVec(s).matvec(t, i.data, row * i.width(), dst, 0);
    }

    @Override
    public Memory ownMemory(ModelSpec spec, MemoryPlan plan) {
        checkOpen();
        return own(new CpuDraftMemory(true, HeapModelMemory.allocate(plan, spec, null), spec));
    }

    @Override
    public Memory targetCaches(ModelSpec spec, MemoryPlan plan) {
        checkOpen();
        CpuMemory target = ctx.memory();
        HeapModelMemory m = HeapModelMemory.allocate(plan, spec, target);
        return own(new CpuDraftMemory(false, new SharedCpuMemory(m, target), spec));
    }

    private final class CpuDraftMemory extends Memory {
        private final CpuMemory m;
        private final ModelSpec spec;

        CpuDraftMemory(boolean owns, CpuMemory m, ModelSpec spec) {
            super(owns);
            this.m = m;
            this.spec = spec;
        }

        @Override
        public ModelMemory memory() {
            return m;
        }

        @Override
        void begin(int pos0, int n, int[] tokens) {
            m.beginStep(pos0, n);
            RopePositions.fillStep(spec.hparams().mropeTextPositions(), pos0, n, scratch.ints(CpuScratch.POS));
            System.arraycopy(tokens, 0, scratch.ints(CpuScratch.TOKEN_IDS), 0, n);
        }
    }

    private CpuForwardContext context(ModelSpec spec, Source s, Memory memory) {
        if (!(memory instanceof CpuDraftMemory)) {
            throw new IllegalArgumentException(memory + " was not made by this head");
        }
        return CpuForwardContext.builder(spec, weights(s), scratch)
                .memory(((CpuDraftMemory) memory).m)
                .matVec(matVec(s))
                .build();
    }

    @Override
    public Stack stack(ModelSpec spec, Source s, Memory memory, List<LayerSpec> layers) {
        checkOpen();
        if (layers.isEmpty()) {
            throw new IllegalArgumentException(spec.arch().ggufName() + ": a draft stack needs at least one layer");
        }
        CpuForwardContext c = context(spec, s, memory);
        List<CpuLayerExecutor> executors = new ArrayList<>(layers.size());
        try {
            for (LayerSpec l : layers) {
                executors.add(new CpuLayerExecutor(registry, c, l));
            }
        } catch (RuntimeException e) {
            for (CpuLayerExecutor ex : executors) {
                ex.close();
            }
            throw e;
        }
        int width = executors.get(0).streamWidth();
        return own(new Stack() {
            @Override
            public void run(int pos0, int n, int[] tokens, Rows rows) {
                CpuRows r = cast(rows);
                checkRows(r, n, ubatch());
                checkWidth(r, width, "draft stack");
                memory.begin(pos0, n, tokens);
                for (CpuLayerExecutor ex : executors) {
                    ex.forward(pos0, n, r.data);
                }
                memory.end(pos0, n, tokens);
            }

            @Override
            public void close() {
                for (CpuLayerExecutor ex : executors) {
                    ex.close();
                }
            }
        });
    }

    @Override
    public Blocks blocks(ModelSpec spec, Source s, Memory memory, LayerSpec layer) {
        checkOpen();
        CpuForwardContext c = context(spec, s, memory);
        CpuMixer mixer = registry.mixer(layer);
        CpuFfnBlock ffn;
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
                CpuRows i = cast(in);
                CpuRows o = cast(out);
                checkRows(i, n, ubatch());
                memory.begin(pos0, n, tokens);
                if (n == 1) {
                    mixer.decode(pos0, i.data, o.data);
                } else {
                    mixer.prefill(pos0, n, i.data, o.data);
                }
                memory.end(pos0, n, tokens);
            }

            @Override
            public void ffn(int n, Rows in, Rows out) {
                CpuRows i = cast(in);
                CpuRows o = cast(out);
                checkRows(i, n, ubatch());
                ffn.forward(n, i.data, o.data);
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

    @Override
    public String toString() {
        return "CpuMtpHead{" + target().arch().ggufName() + ", ubatch=" + ubatch()
                + (draft != null ? ", drafter weights" : "") + '}';
    }
}
