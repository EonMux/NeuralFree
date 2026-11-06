package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.attn.AttnKernels;
import com.eonmux.neuralfree.gguf.ops.attn.FlashAttnTiling;
import com.eonmux.neuralfree.gguf.ops.flow.FlowKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.vision.VisionKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.UnaryOperator;

/**
 * The GPU executor of the shared vision graphs: every {@link VisionOps} op on an OpenCL, CUDA or Vulkan backend
 * through {@link GpuOps} and the LLM kernel library, the counterpart of {@link CpuVisionOps}. A block is a scratch
 * temporary named per encode call ({@link #begin} restarts the names, so the buffers of one call are reused by the
 * next); matmuls run in the weights' device types, norms through {@code LLM_RowNorm}, activations and gates
 * through {@code LLM_Unary} / {@code LLM_GLU}, reorders as {@code LLM_GatherRows} of whole {@code n_embd} chunks
 * by the index tables of {@link PixelShuffle#chunkSources} (plus {@code LLM_PermuteCols} for the channel-major
 * conv unfolds), the depthwise conv as nine gathered taps scaled per channel, attention as bidirectional
 * {@code LLM_FlashAttn_F32}, and the tower is {@link GpuVitTower}. Host rows the graphs assemble are written
 * through the queue's staged writes; a {@link #download} waits for the queue. Not thread-safe; one encode at a
 * time.
 */
final class GpuVisionOps implements VisionOps {

    /** A block of rows in one scratch temporary, {@code [n][width]} from element 0. */
    static final class GpuBlock implements Block {
        final ScratchSlot slot;
        private final int n;
        private final int width;

        GpuBlock(ScratchSlot slot, int n, int width) {
            if (n <= 0 || width <= 0 || (long) n * width > slot.floats()) {
                throw new IllegalArgumentException(n + " rows of " + width + " do not fit the " + slot.floats()
                        + " floats of " + slot.name());
            }
            this.slot = slot;
            this.n = n;
            this.width = width;
        }

        @Override
        public int n() {
            return n;
        }

        @Override
        public int width() {
            return width;
        }

        GpuBuffer buffer() {
            return slot.buffer();
        }

        int size() {
            return n * width;
        }
    }

    /** The buffers and row strides of one attention: Q, K, V, the output and the key positions. */
    static final class AttnBuffers {
        final GpuBuffer q;
        final int qStride;
        final GpuBuffer k;
        final GpuBuffer v;
        final int kvStride;
        final GpuBuffer out;
        final int outStride;
        final GpuBuffer pos;

        AttnBuffers(GpuBuffer q, int qStride, GpuBuffer k, GpuBuffer v, int kvStride, GpuBuffer out, int outStride,
                    GpuBuffer pos) {
            this.q = q;
            this.qStride = qStride;
            this.k = k;
            this.v = v;
            this.kvStride = kvStride;
            this.out = out;
            this.outStride = outStride;
            this.pos = pos;
        }
    }

    private static final String TEMP = "vision.";

    private final ClipVisionModel model;
    private final ClipHparams hp;
    private final GpuOps ops;
    private final ComputeQueue queue;
    private final GpuScratch scratch;
    private final GpuPlaceholders placeholders;
    private final int local;
    private final GpuVisionWeights w;
    private final int smemFloats;
    private final int computeUnits;
    private final GpuVitTower tower;
    private final Map<String, float[]> hostCopies = new HashMap<>();
    private final Map<String, FlashAttnTiling> plans = new HashMap<>();
    private int seq;

    /**
     * Binds the executor over the resolved device tensors {@code w} on {@code ops}' queue.
     *
     * @param smemFloats   flash-attention shared floats of the device
     * @param computeUnits compute units of the device
     */
    GpuVisionOps(ClipVisionModel model, GpuOps ops, GpuVisionWeights w, int smemFloats, int computeUnits) {
        this.model = Objects.requireNonNull(model, "model");
        this.hp = model.hparams();
        this.ops = Objects.requireNonNull(ops, "ops");
        this.queue = ops.queue();
        this.scratch = ops.scratch();
        this.placeholders = ops.placeholders();
        this.local = ops.local();
        this.w = Objects.requireNonNull(w, "weights");
        this.smemFloats = smemFloats;
        this.computeUnits = computeUnits;
        this.tower = new GpuVitTower(this, w);
    }

    /** Starts an encode call: the blocks of the previous call are released for reuse. */
    void begin() {
        seq = 0;
    }

    GpuOps gpu() {
        return ops;
    }

    GpuPlaceholders placeholders() {
        return placeholders;
    }

    int local() {
        return local;
    }

    /**
     * The tiled prefill flash-attention plan of {@code nHead} query heads over {@code nHeadKv} KV heads of width
     * {@code dh} on this device.
     *
     * @throws UnsupportedOperationException when the device has no tiled plan for the shape
     */
    FlashAttnTiling plan(int nHead, int nHeadKv, int dh) {
        String key = nHead + "/" + nHeadKv + "/" + dh;
        FlashAttnTiling p = plans.get(key);
        if (p == null) {
            p = attentionPlan(nHead, nHeadKv, dh, Math.max(1, model.nPos()), smemFloats, computeUnits);
            if (p.plan() != FlashAttnTiling.Plan.TILED) {
                throw new UnsupportedOperationException("no tiled flash-attention plan for " + nHead + " heads of"
                        + " width " + dh + " (" + smemFloats + " shared floats): " + p);
            }
            plans.put(key, p);
        }
        return p;
    }

    /** The prefill plan of an attention shape ({@link FlashAttnTiling#prefill}). */
    static FlashAttnTiling attentionPlan(int nHead, int nHeadKv, int dh, int nKvMax, int smemFloats,
                                         int computeUnits) {
        return FlashAttnTiling.prefill(nHead, nHeadKv, dh, dh, nKvMax, smemFloats, computeUnits);
    }

    // ------------------------------------------------------------------ model

    @Override
    public ClipVisionModel model() {
        return model;
    }

    @Override
    public ClipHparams hp() {
        return hp;
    }

    // ------------------------------------------------------------------ weights

    @Override
    public boolean hasMatrix(String name) {
        return w.matrixOrNull(name) != null;
    }

    @Override
    public boolean hasVector(String name) {
        return w.vectorOrNull(name) != null;
    }

    @Override
    public float[] hostVector(String name) {
        return hostCopy(w.vector(name));
    }

    @Override
    public float[] hostVectorOrNull(String name) {
        GpuWeight v = w.vectorOrNull(name);
        return v == null ? null : hostCopy(v);
    }

    @Override
    public float[] hostWeight(TensorRef ref) {
        return ref == null ? null : hostCopy(w.vectorOf(ref));
    }

    /** The values of a one-row F32 weight, read from the device once and kept. */
    private float[] hostCopy(GpuWeight v) {
        float[] cached = hostCopies.get(v.name());
        if (cached == null) {
            cached = read(v.shard(0), Math.toIntExact(v.cols()));
            hostCopies.put(v.name(), cached);
        }
        return cached;
    }

    // ------------------------------------------------------------------ host transfer

    @Override
    public Block upload(float[] rows, int width) {
        if (width <= 0 || rows.length == 0 || rows.length % width != 0) {
            throw new IllegalArgumentException(rows.length + " values do not split into rows of " + width);
        }
        GpuBlock b = alloc(rows.length / width, width);
        writeFloats(b.buffer(), 0, rows);
        return b;
    }

    @Override
    public Block view(Block src, int n, int width) {
        GpuBlock in = block(src);
        if ((long) n * width != in.size()) {
            throw new IllegalArgumentException(n + " rows of " + width + " do not cover " + in.size() + " values");
        }
        return new GpuBlock(in.slot, n, width);
    }

    @Override
    public float[] download(Block b) {
        GpuBlock in = block(b);
        return read(in.buffer(), in.size());
    }

    @Override
    public int mark() {
        return seq;
    }

    @Override
    public void release(int mark) {
        if (mark < 0 || mark > seq) {
            throw new IllegalArgumentException("allocation mark " + mark + " is outside [0, " + seq + "]");
        }
        seq = mark;
    }

    // ------------------------------------------------------------------ encoder input

    @Override
    public Block patchRows(ClipImage image, int pw, int ph) {
        return patchEmbed(ClipPatches.raster(image, hp.patchSize(), pw, ph), pw * ph);
    }

    @Override
    public Block patchEmbed(float[] patches, int nPatches) {
        int outWidth = Math.toIntExact(w.patchEmbd.rows());
        int kernelWidth = (int) w.patchEmbd.cols();
        int rowWidth = nPatches <= 0 ? -1 : patches.length / nPatches;
        if (nPatches <= 0 || (rowWidth != kernelWidth && rowWidth != 2 * kernelWidth)
                || (long) nPatches * rowWidth != patches.length) {
            throw new IllegalArgumentException(patches.length + " patch values are not " + nPatches + " rows of "
                    + kernelWidth);
        }
        GpuBlock out = alloc(nPatches, outWidth);
        if (rowWidth == 2 * kernelWidth) {
            patchEmbedPair(patches, nPatches, kernelWidth, out);
        } else {
            ScratchSlot in = temp("patches", patches.length);
            writeFloats(in.buffer(), 0, patches);
            ops.matmul(w.patchEmbd, in.buffer(), 0, out.buffer(), 0, nPatches);
            if (w.patchEmbd1 != null) {
                ScratchSlot second = temp("patches.frame1", out.size());
                ops.matmul(w.patchEmbd1, in.buffer(), 0, second.buffer(), 0, nPatches);
                ops.add(second.buffer(), 0, out.buffer(), 0, out.buffer(), 0, out.size());
            }
        }
        biasAdd(out.buffer(), w.patchBias, nPatches, outWidth);
        return out;
    }

    /**
     * The two-frame input of {@code build_inp_temporal_merge} on the GPU: the pair rows de-interleaved on the
     * host, kernel 0 over the first frames' buffer, kernel 1 over the second frames', the outputs added. Requires
     * the second kernel.
     */
    private void patchEmbedPair(float[] patches, int nPatches, int kernelWidth, GpuBlock out) {
        if (w.patchEmbd1 == null) {
            throw new IllegalArgumentException("this encoder carries no temporal patch kernel"
                    + " (v.patch_embd.weight.1), so it cannot encode a frame pair");
        }
        float[] first = new float[nPatches * kernelWidth];
        float[] second = new float[nPatches * kernelWidth];
        for (int t = 0; t < nPatches; t++) {
            System.arraycopy(patches, t * 2 * kernelWidth, first, t * kernelWidth, kernelWidth);
            System.arraycopy(patches, t * 2 * kernelWidth + kernelWidth, second, t * kernelWidth, kernelWidth);
        }
        ScratchSlot firstBuf = temp("patches.frame0", first.length);
        writeFloats(firstBuf.buffer(), 0, first);
        ScratchSlot secondBuf = temp("patches.frame1", second.length);
        writeFloats(secondBuf.buffer(), 0, second);
        ScratchSlot secondRows = temp("patches.frame1.rows", out.size());
        ops.matmul(w.patchEmbd, firstBuf.buffer(), 0, out.buffer(), 0, nPatches);
        ops.matmul(w.patchEmbd1, secondBuf.buffer(), 0, secondRows.buffer(), 0, nPatches);
        ops.add(secondRows.buffer(), 0, out.buffer(), 0, out.buffer(), 0, out.size());
    }

    @Override
    public void addHost(Block dst, float[] rows) {
        GpuBlock d = block(dst);
        if (rows.length != d.size()) {
            throw new IllegalArgumentException("the add needs " + d.size() + " values, got " + rows.length);
        }
        ScratchSlot staged = temp("add", rows.length);
        writeFloats(staged.buffer(), 0, rows);
        ops.add(d.buffer(), 0, staged.buffer(), 0, d.buffer(), 0, d.size());
    }

    @Override
    public void addBlock(Block dst, Block src) {
        GpuBlock d = block(dst);
        GpuBlock s = block(src);
        if (s.size() != d.size()) {
            throw new IllegalArgumentException("the add needs " + d.size() + " values, got " + s.size());
        }
        ops.add(d.buffer(), 0, s.buffer(), 0, d.buffer(), 0, d.size());
    }

    @Override
    public Block insertRow(Block src, int at, float[] values) {
        GpuBlock in = block(src);
        if (values.length != in.width || at < 0 || at > in.n) {
            throw new IllegalArgumentException("a row of " + values.length + " values at " + at + " does not fit "
                    + in.n + " rows of " + in.width);
        }
        GpuBlock out = alloc(in.n + 1, in.width);
        if (at > 0) {
            ops.copy(in.buffer(), 0, out.buffer(), 0, at * in.width);
        }
        writeFloats(out.buffer(), at * in.width, values);
        if (at < in.n) {
            ops.copy(in.buffer(), at * in.width, out.buffer(), (at + 1) * in.width, (in.n - at) * in.width);
        }
        return out;
    }

    @Override
    public void addGathered(Block dst, TensorRef table, int[] idx, int width) {
        GpuBlock d = block(dst);
        GpuWeight t = w.vectorOf(Objects.requireNonNull(table, "table"));
        int tableRows = Math.toIntExact(t.cols() / width);
        if (t.cols() % width != 0 || idx.length * width != d.size()) {
            throw new IllegalArgumentException("the " + t.cols() + " values of " + table.name() + " gathered by "
                    + idx.length + " rows of " + width + " do not fit " + d.size() + " values");
        }
        requireRows(idx, tableRows, table.name());
        ScratchSlot rows = temp("gathered", d.size());
        gather(idx, t.shard(0), rows.buffer(), width);
        ops.add(d.buffer(), 0, rows.buffer(), 0, d.buffer(), 0, d.size());
    }

    @Override
    public Block gatherRows(Block src, int[] idx, int width) {
        GpuBlock in = block(src);
        if (in.size() % width != 0) {
            throw new IllegalArgumentException(in.size() + " values do not split into rows of " + width);
        }
        requireRows(idx, in.size() / width, "the gathered block");
        GpuBlock out = alloc(idx.length, width);
        gather(idx, in.buffer(), out.buffer(), width);
        return out;
    }

    // ------------------------------------------------------------------ algebra

    @Override
    public Block linear(String matrix, Block x, String biasName) {
        return project(w.matrix(matrix), x, biasName);
    }

    @Override
    public Block linearKernel(String matrix, int outRows, int inWidth, Block x, String biasName) {
        return project(w.kernelMatrix(matrix, outRows, inWidth), x, biasName);
    }

    private Block project(GpuWeight weight, Block x, String biasName) {
        GpuBlock in = block(x);
        if (in.width != weight.cols()) {
            throw new IllegalArgumentException(weight.name() + " reads rows of " + weight.cols() + ", the block's"
                    + " rows hold " + in.width);
        }
        int outWidth = Math.toIntExact(weight.rows());
        GpuBlock out = alloc(in.n, outWidth);
        GpuBuffer bias = biasName == null ? null : w.bufferOrNull(biasName);
        projectBiased(weight, bias, in.buffer(), out.buffer(), in.n, outWidth);
        GpuProjectionDeltas.Site site = siteOf(weight, in.width, outWidth);
        if (site != null) {
            site.apply(in.n, in.buffer(), 0, in.width, out.buffer(), 0, outWidth);
        }
        return out;
    }

    /**
     * The LoRA site of a connector matrix, refused by name when an adapter adapts it but the product's shape
     * differs from the tensor's (a conv kernel read as another im2col matrix).
     */
    private GpuProjectionDeltas.Site siteOf(GpuWeight weight, int inWidth, int outWidth) {
        GpuProjectionDeltas.Site site = w.site(weight.name());
        if (site == null || site.isEmpty() || (site.inDim() == inWidth && site.outDim() == outWidth)) {
            return site;
        }
        throw new UnsupportedOperationException("LoRA tensor " + weight.name() + " is read as a " + outWidth + " x "
                + inWidth + " im2col kernel, not its " + site.outDim() + " x " + site.inDim() + " tensor shape;"
                + " the connector cannot apply the delta to this product");
    }

    @Override
    public void norm(Block x, NormKind kind, String weight, String bias, float eps, int rows, int width) {
        normRows(block(x), kind, w.vector(weight), bias == null ? null : w.vectorOrNull(bias), eps, rows, width);
    }

    @Override
    public void normByRef(Block x, NormKind kind, TensorRef weight, TensorRef bias, float eps, int rows,
                          int width) {
        normRows(block(x), kind, w.vectorOf(weight), w.vectorOf(bias), eps, rows, width);
    }

    private void normRows(GpuBlock x, NormKind kind, GpuWeight weight, GpuWeight bias, float eps, int rows,
                          int width) {
        if ((long) rows * width > x.size()) {
            throw new IllegalArgumentException(rows + " rows of " + width + " exceed the block's " + x.size()
                    + " values");
        }
        ops.norm(kind, weight, bias, eps, x.buffer(), 0, x.buffer(), 0, rows, width);
    }

    @Override
    public void activate(Block x, ActOp op) {
        GpuBlock b = block(x);
        ops.unary(op, ActParams.NONE, b.buffer(), 0, b.buffer(), 0, b.size());
    }

    @Override
    public void glu(Block gate, Block up, GluOp op, int nRows, int width) {
        glu(gate, up, op, ActParams.NONE, nRows, width);
    }

    @Override
    public void glu(Block gate, Block up, GluOp op, ActParams params, int nRows, int width) {
        GpuBlock g = block(gate);
        GpuBlock u = block(up);
        ops.glu(op, params, g.buffer(), 0, g.width, u.buffer(), 0, u.width, u.buffer(), 0, u.width, nRows, width);
    }

    @Override
    public Block columns(Block src, int from, int width) {
        GpuBlock in = block(src);
        if (from < 0 || width <= 0 || from + width > in.width) {
            throw new IllegalArgumentException("columns [" + from + ", " + (from + width) + ") are outside rows of "
                    + in.width);
        }
        GpuBlock out = alloc(in.n, width);
        ops.copy2d(in.buffer(), from, in.width, out.buffer(), 0, width, in.n, width);
        return out;
    }

    @Override
    public void rope(Block x, int nHeads, RopePlan plan) {
        tower.rope(block(x), nHeads, plan);
    }

    @Override
    public void mulRowScale(Block x, float[] scale) {
        GpuBlock b = block(x);
        if (scale.length != b.width) {
            throw new IllegalArgumentException("the row scale needs " + b.width + " values, got " + scale.length);
        }
        ScratchSlot s = temp("rowscale", scale.length);
        writeFloats(s.buffer(), 0, scale);
        rowScale(b.buffer(), s.buffer(), b.n, b.width);
    }

    @Override
    public void hardswish(Block x) {
        GpuBlock b = block(x);
        ScratchSlot gate = temp("hardswish", b.size());
        ops.copy(b.buffer(), 0, gate.buffer(), 0, b.size());
        hardsigmoid(gate.buffer(), b.size());
        ops.glu(GluOp.MUL, ActParams.NONE, gate.buffer(), 0, b.size(), b.buffer(), 0, b.size(), b.buffer(), 0,
                b.size(), 1, b.size());
    }

    @Override
    public void hardsigmoid(Block x) {
        GpuBlock b = block(x);
        hardsigmoid(b.buffer(), b.size());
    }

    /** {@code min(max(x / 6 + 1/2, 0), 1)} in place: {@code ggml_hardsigmoid}. */
    private void hardsigmoid(GpuBuffer x, int n) {
        ops.unary(ActOp.SCALE_BIAS, ActParams.builder().scaleBias(1f / 6f, 0.5f).build(), x, 0, x, 0, n);
        ops.unary(ActOp.CLAMP, ActParams.builder().clamp(0f, 1f).build(), x, 0, x, 0, n);
    }

    // ------------------------------------------------------------------ reorders

    @Override
    public Block mergePermute(Block src, int nEmbd, int pw, int ph, int s) {
        return chunkGather(src, nEmbd, pw * ph, s, ids -> PixelShuffle.mergePermute(ids, 1, pw, ph, s));
    }

    @Override
    public Block shuffleInternvl(Block src, int nEmbd, int pw, int ph, int s) {
        return chunkGather(src, nEmbd, pw * ph, s, ids -> PixelShuffle.internvl(ids, 1, pw, ph, s));
    }

    @Override
    public Block shuffleLlama4(Block src, int nEmbd, int pw, int ph, int s) {
        return chunkGather(src, nEmbd, pw * ph, s, ids -> PixelShuffle.llama4(ids, 1, pw, ph, s));
    }

    @Override
    public Block unfold(Block src, int nEmbd, int pw, int ph, int s) {
        PixelShuffle.requireMultiple(pw, ph, s, "conv");
        return unfoldWindows(src, nEmbd, pw, pw / s, ph / s, s);
    }

    @Override
    public Block unfoldWindows(Block src, int nEmbd, int pw, int ow, int oh, int s) {
        int nRows = block(src).size() / nEmbd;
        GpuBlock patchMajor = block(chunkGather(src, nEmbd, nRows, s,
                ids -> PixelShuffle.unfoldWindows(ids, 1, pw, ow, oh, s)));
        int[] perm = PixelShuffle.channelMajor(nEmbd, s);
        ScratchSlot table = temp("unfold.perm", perm.length);
        writeInts(table.buffer(), perm);
        GpuBlock out = alloc(patchMajor.n, patchMajor.width);
        ops.permuteCols(table.buffer(), patchMajor.buffer(), 0, out.buffer(), 0, out.n, out.width);
        return out;
    }

    @Override
    public Block im2col(Block src, int nEmbd, int pw, int ph, int k, int stride, int pad) {
        return channelMajorWindows(src, nEmbd, k, ids -> PixelShuffle.im2col(ids, 1, pw, ph, k, stride, pad));
    }

    @Override
    public Block im2colPadded(Block src, int nEmbd, int pw, int ph, int k, int stride, int padTop, int padLeft,
                              int outW, int outH) {
        return channelMajorWindows(src, nEmbd, k,
                ids -> PixelShuffle.im2colPadded(ids, 1, pw, ph, k, stride, padTop, padLeft, outW, outH));
    }

    @Override
    public Block unfoldPadded(Block src, int nEmbd, int pw, int ph, int s) {
        return channelMajorWindows(src, nEmbd, s, ids -> PixelShuffle.unfoldPadded(ids, 1, pw, ph, s));
    }

    /**
     * The {@code k x k} windows {@code windows} lays out over the chunk ids (a pad reads zero), gathered patch-major
     * and permuted to the channel-major im2col order.
     */
    private Block channelMajorWindows(Block src, int nEmbd, int k, UnaryOperator<float[]> windows) {
        int nRows = block(src).size() / nEmbd;
        GpuBlock patchMajor = block(chunkGather(src, nEmbd, nRows, k, windows));
        int[] perm = PixelShuffle.channelMajor(nEmbd, k);
        ScratchSlot table = temp("im2col.perm", perm.length);
        writeInts(table.buffer(), perm);
        GpuBlock out = alloc(patchMajor.n, patchMajor.width);
        ops.permuteCols(table.buffer(), patchMajor.buffer(), 0, out.buffer(), 0, out.n, out.width);
        return out;
    }

    /**
     * {@code src}'s {@code n_embd} chunks reordered by {@code reorder} (run over the chunk ids, {@link
     * PixelShuffle#chunkSources}) into rows of {@code s * s} chunks; a pad entry reads a zero row.
     */
    private Block chunkGather(Block src, int nEmbd, int nRows, int s, UnaryOperator<float[]> reorder) {
        GpuBlock in = block(src);
        if ((long) nRows * nEmbd != in.size()) {
            throw new IllegalArgumentException("the reorder reads " + nRows + " rows of " + nEmbd + ", the block"
                    + " holds " + in.size() + " values");
        }
        int[] idx = PixelShuffle.chunkSources(nRows, reorder);
        GpuBuffer source = in.buffer();
        if (hasPad(idx)) {
            ScratchSlot padded = temp("reorder.pad", (long) (nRows + 1) * nEmbd);
            ops.copy(in.buffer(), 0, padded.buffer(), 0, in.size());
            ops.fill(padded.buffer(), (long) in.size() * Float.BYTES, (long) nEmbd * Float.BYTES, 0);
            for (int i = 0; i < idx.length; i++) {
                idx[i] = idx[i] < 0 ? nRows : idx[i];
            }
            source = padded.buffer();
        }
        int group = s * s;
        GpuBlock out = alloc(idx.length / group, nEmbd * group);
        gather(idx, source, out.buffer(), nEmbd);
        return out;
    }

    private static boolean hasPad(int[] idx) {
        for (int i : idx) {
            if (i < 0) {
                return true;
            }
        }
        return false;
    }

    @Override
    public Block concatChannels(List<Block> blocks, int nRows, int width) {
        int k = blocks.size();
        GpuBlock out = alloc(nRows, width * k);
        for (int i = 0; i < k; i++) {
            GpuBlock b = block(blocks.get(i));
            if (b.size() != nRows * width) {
                throw new IllegalArgumentException("concat part " + i + " holds " + b.size() + " values, not "
                        + nRows + " rows of " + width);
            }
            ops.copy2d(b.buffer(), 0, width, out.buffer(), i * width, width * k, nRows, width);
        }
        return out;
    }

    /** {@code ggml_upscale} nearest by {@code f}: one gathered row per output row ({@code LLM_GatherRows}). */
    @Override
    public Block upscaleNearest(Block src, int width, int height, int channels, int f) {
        GpuBlock in = block(src);
        if (in.width != channels || in.n != width * height || f <= 0) {
            throw new IllegalArgumentException("a " + width + "x" + height + " map of " + channels + " channels"
                    + " needs " + width * height + " rows of " + channels + ", the block holds " + in.n + " rows of "
                    + in.width);
        }
        int outW = width * f;
        int outH = height * f;
        int[] idx = new int[outW * outH];
        for (int oy = 0; oy < outH; oy++) {
            for (int ox = 0; ox < outW; ox++) {
                idx[oy * outW + ox] = (oy / f) * width + ox / f;
            }
        }
        return gatherRows(src, idx, channels);
    }

    /**
     * {@code ggml_conv_2d_dw} 3x3 with zero padding 1, through the depthwise kernel of
     * {@link VisionKernels#CONV2D_DEPTHWISE}.
     */
    @Override
    public Block depthwise3x3(Block x, int width, int height, TensorRef weight, int stride) {
        return depthwise(x, width, height, weight, 3, stride, 1, 1, (width - 1) / stride + 1,
                (height - 1) / stride + 1, null);
    }

    /**
     * {@code ggml_conv_2d_dw} with a {@code k x k} kernel ({@code LLM_Conv2dDepthwise}): one launch, the kernel's
     * F32 weight and the optional bias read as device buffers.
     */
    @Override
    public Block depthwise(Block x, int width, int height, TensorRef weight, int k, int stride, int padTop,
                           int padLeft, int outW, int outH, TensorRef bias) {
        GpuBlock in = block(x);
        int d = in.width;
        if (in.n != width * height) {
            throw new IllegalArgumentException("a " + width + "x" + height + " feature map needs " + width * height
                    + " rows, got " + in.n);
        }
        GpuWeight kernel = w.vectorOf(weight);
        if (kernel.cols() != (long) k * k * d) {
            throw new IllegalArgumentException(weight.name() + " holds " + kernel.cols() + " values, not a " + k
                    + "x" + k + " kernel per channel of " + d);
        }
        VisionKernels.checkDepthwise(width, height, outW, outH, d, k, stride, stride, padTop, padLeft);
        GpuBlock out = alloc(outW * outH, d);
        ops.enqueue(VisionKernels.CONV2D_DEPTHWISE, new long[]{VisionKernels.elementGlobal((long) outW * outH * d)},
                new long[]{VisionKernels.LOCAL}, width, height, outW, outH, d, k, stride, stride, padTop, padLeft,
                0, 0, in.buffer(), kernel.shard(0), biasBuffer(bias, d), out.buffer());
        return out;
    }

    /** The F32 buffer of projector vector {@code bias}, or a zeroed scratch row when absent. */
    private GpuBuffer biasBuffer(TensorRef bias, int width) {
        if (bias == null) {
            ScratchSlot zero = temp("bias.zero", width);
            ops.fill(zero.buffer(), 0, (long) width * Float.BYTES, 0);
            return zero.buffer();
        }
        GpuWeight b = w.vectorOf(bias);
        if (b.cols() != width) {
            throw new IllegalArgumentException(bias.name() + " holds " + b.cols() + " values, not one per channel"
                    + " (" + width + ")");
        }
        return b.shard(0);
    }

    /** {@code ggml_pool_2d} average over whole windows ({@code LLM_Pool2dAvg}). */
    @Override
    public Block avgPool(Block x, int width, int height, int channels, int kyH, int kw, int outW, int outH) {
        GpuBlock in = block(x);
        if (in.width != channels || in.n != width * height) {
            throw new IllegalArgumentException("a " + width + "x" + height + " map of " + channels + " channels"
                    + " needs " + width * height + " rows of " + channels + ", the block holds " + in.n + " rows of "
                    + in.width);
        }
        VisionKernels.checkPool(width, height, outW, outH, channels, kyH, kw);
        GpuBlock out = alloc(outW * outH, channels);
        ops.enqueue(VisionKernels.POOL_2D_AVG, new long[]{VisionKernels.elementGlobal((long) outW * outH * channels)},
                new long[]{VisionKernels.LOCAL}, width, height, outW, outH, channels, kyH, kw, 0, 0, in.buffer(),
                out.buffer());
        return out;
    }

    /**
     * {@code convnext_grn} on the device: the per-channel sums of squares as one pool over the map read as one
     * column ({@code W} 1, {@code H} the rows), their roots and mean read back to the host for the small scale,
     * then the row scale and the optional bias on the device.
     */
    @Override
    public void grn(Block x, int rows, int width, TensorRef weight, TensorRef bias) {
        GpuBlock b = block(x);
        if (b.width != width || (long) rows * width > b.size() || rows <= 0) {
            throw new IllegalArgumentException("the response norm reads " + rows + " rows of " + width + ", the"
                    + " block holds " + b.n + " rows of " + b.width);
        }
        ScratchSlot squares = temp("grn.squares", (long) rows * width);
        ops.copy(b.buffer(), 0, squares.buffer(), 0, rows * width);
        ops.glu(GluOp.MUL, ActParams.NONE, squares.buffer(), 0, width, b.buffer(), 0, width, squares.buffer(), 0,
                width, rows, width);
        VisionKernels.checkPool(1, rows, 1, 1, width, rows, 1);
        ScratchSlot means = temp("grn.means", width);
        ops.enqueue(VisionKernels.POOL_2D_AVG, new long[]{VisionKernels.elementGlobal(width)},
                new long[]{VisionKernels.LOCAL}, 1, rows, 1, 1, width, rows, 1, 0, 0, squares.buffer(),
                means.buffer());
        float[] channelMeans = read(means.buffer(), width);
        float[] scale = new float[width];
        double sum = 0;
        for (int c = 0; c < width; c++) {
            scale[c] = (float) Math.sqrt((double) channelMeans[c] * rows);
            sum += scale[c];
        }
        float m = Math.max((float) (sum / width), Yasa2Graph.GRN_EPS);
        float[] affine = weight == null ? null : hostWeight(weight);
        for (int c = 0; c < width; c++) {
            // x + (x * N) * w: the identity term rides in the row scale
            scale[c] = 1f + scale[c] / m * (affine == null ? 1f : affine[c]);
        }
        ScratchSlot factors = temp("grn.scale", width);
        writeFloats(factors.buffer(), 0, scale);
        rowScale(b.buffer(), factors.buffer(), rows, width);
        if (bias != null) {
            biasAdd(b.buffer(), biasBuffer(bias, width), rows, width);
        }
    }

    @Override
    public Block attend(Block q, int qRowOff, int nq, Block k, Block v, int kvRowOff, int nk, int kvWidth,
                        int nHead, int kvGroup, int dh, float scale) {
        GpuBlock qb = block(q);
        GpuBlock kb = block(k);
        GpuBlock vb = block(v);
        int qWidth = nHead * dh;
        if (qb.width != qWidth || kb.width != kvWidth || vb.width != kvWidth || kvGroup <= 0
                || nHead % kvGroup != 0 || qRowOff + nq > qb.n || kvRowOff + nk > kb.n || kvRowOff + nk > vb.n) {
            throw new IllegalArgumentException("attention of " + nq + " queries at row " + qRowOff + " over " + nk
                    + " keys at row " + kvRowOff + " does not fit the blocks (" + nHead + " heads of " + dh + ")");
        }
        FlashAttnTiling plan = plan(nHead, nHead / kvGroup, dh);
        GpuBlock out = alloc(nq, qWidth);
        ScratchSlot pos = positions(Math.max(nq, kvRowOff + nk));
        AttnBuffers bufs = new AttnBuffers(qb.buffer(), qWidth, kb.buffer(), vb.buffer(), kvWidth, out.buffer(),
                qWidth, pos.buffer());
        flash(plan, bufs, qRowOff, nq, 0, kvRowOff, kvRowOff + nk, scale);
        return out;
    }

    @Override
    public Block attendWindows(Block q, Block k, Block v, int n, int kvWidth, int nHead, int kvGroup, int dh,
                               float scale, VisionRows.KeyRanges windows) {
        Objects.requireNonNull(windows, "windows");
        GpuBlock qb = block(q);
        GpuBlock kb = block(k);
        GpuBlock vb = block(v);
        int qWidth = nHead * dh;
        if (qb.width != qWidth || kb.width != kvWidth || vb.width != kvWidth || kvGroup <= 0
                || nHead % kvGroup != 0 || n > qb.n || n > kb.n || n > vb.n) {
            throw new IllegalArgumentException("window attention over " + n + " rows does not fit the blocks ("
                    + nHead + " heads of " + dh + ")");
        }
        List<int[]> runs = VisionRows.keyRuns(windows, n);
        FlashAttnTiling plan = plan(nHead, nHead / kvGroup, dh);
        GpuBlock out = alloc(n, qWidth);
        ScratchSlot pos = positions(n);
        AttnBuffers bufs = new AttnBuffers(qb.buffer(), qWidth, kb.buffer(), vb.buffer(), kvWidth, out.buffer(),
                qWidth, pos.buffer());
        for (int[] run : runs) {
            flash(plan, bufs, run[0], run[1] - run[0], run[0], run[2], run[3], scale);
        }
        return out;
    }

    // ------------------------------------------------------------------ tower

    @Override
    public Block tower(Block x, int nTokens, NormKind norm, TowerOptions opts) {
        return tower.forward(block(x), nTokens, norm, opts);
    }

    /**
     * The SAM backbone on the device ({@code models/deepseekocr.cpp build_sam} over {@link SamTensors}): the
     * patch conv as its im2col matmul, the absolute position grid (bicubic-resized when its side differs), the
     * blocks (channel layer norm, the fused biased QKV, the windowed or whole-grid attention of
     * {@link VisionKernels#REL_POS_ATTN} with its decomposed relative-position tables, the output projection and
     * the MLP, each with its residual; the window partition and its inverse as gathered rows over a zero pad
     * row), then the neck (1x1 conv, channel layer norm, 3x3 conv, channel layer norm) and the two stride-2 3x3
     * convs as im2col matmuls.
     *
     * @throws UnsupportedOperationException when the encoder graph carries no SAM backbone
     * @throws IllegalArgumentException      when the image is not square or not a patch multiple (named)
     */
    @Override
    public Block sam(ClipImage image) {
        VisionFamilyHparams.Sam sam = hp.visionFamily().sam();
        if (sam == null) {
            throw new UnsupportedOperationException("the encoder graph " + model.encoderGraph()
                    + " has no SAM backbone");
        }
        int patch = samPatch();
        int g = samSide(image, patch);
        int e = sam.nEmbd();
        int heads = sam.nHead();
        int dh = e / heads;
        int n = g * g;
        float[] patches = ClipPatches.raster(image, patch, g, g);
        GpuBlock x = (GpuBlock) linearKernel(SamTensors.PATCH, e, patch * patch * 3,
                upload(patches, patch * patch * 3), SamTensors.PATCH_BIAS);
        float[] pos = hostVector(SamTensors.POS);
        int side = (int) Math.round(Math.sqrt((double) pos.length / e));
        addHost(x, side == g ? pos : Glm4vGraph.resizeBicubic(pos, e, side, g, g));
        for (int il = 0; il < sam.nLayer(); il++) {
            samBlock(sam, x, g, il, e, heads, dh, sam.ffnOp().unary());
        }
        return samNeck(x, g, e);
    }

    /** The side of the pixel square one SAM patch row reads, from the patch kernel's shape. */
    private int samPatch() {
        TensorRef patch = model.projectorMatrices().get(SamTensors.PATCH);
        VisionFamilyHparams.Sam sam = hp.visionFamily().sam();
        long values = 1;
        for (int i = 0; i < 4; i++) {
            values *= patch.ne(i);
        }
        return (int) Math.round(Math.sqrt((double) values / (3L * sam.nEmbd())));
    }

    /** The patch grid side of a square image whose side is a patch multiple. */
    private static int samSide(ClipImage image, int patch) {
        if (image.width() != image.height() || image.width() % patch != 0) {
            throw new IllegalArgumentException("the SAM backbone needs a square image whose side is a multiple of"
                    + " the patch size " + patch + ", got " + image.width() + "x" + image.height());
        }
        return image.width() / patch;
    }

    /** One SAM block in place over the {@code g x g} rows of {@code x}: norm, attention, norm, MLP. */
    private void samBlock(VisionFamilyHparams.Sam sam, GpuBlock x, int g, int il, int e, int heads, int dh,
                          ActOp act) {
        int n = g * g;
        String p = SamTensors.block(il, "");
        boolean global = SamBackbone.isGlobal(il);
        int win = sam.window();
        int nw = (g + win - 1) / win;
        int side = global ? g : win;
        int groups = global ? 1 : nw * nw;
        int rows = groups * side * side;
        // the pre-norm runs in grid order: the partition's zero pad rows must stay zero
        ScratchSlot normed = temp("sam.ln", (long) n * e);
        ops.norm(NormKind.LAYER, w.vector(p + "pre_ln.weight"), w.vectorOrNull(p + "pre_ln.bias"), SamBackbone.EPS,
                x.buffer(), 0, normed.buffer(), 0, n, e);
        GpuBlock cur = global ? new GpuBlock(normed, n, e) : partition(new GpuBlock(normed, n, e), e, g, win);
        GpuBlock qkv = (GpuBlock) linear(p + "attn.qkv.weight", cur, p + "attn.qkv.bias");
        GpuBlock q = alloc(rows, e);
        GpuBlock k = alloc(rows, e);
        GpuBlock v = alloc(rows, e);
        ops.copy2d(qkv.buffer(), 0, 3 * e, q.buffer(), 0, e, rows, e);
        ops.copy2d(qkv.buffer(), e, 3 * e, k.buffer(), 0, e, rows, e);
        ops.copy2d(qkv.buffer(), 2 * e, 3 * e, v.buffer(), 0, e, rows, e);
        ScratchSlot rw = relTable(p + "attn.pos_w.weight", dh, side);
        ScratchSlot rh = relTable(p + "attn.pos_h.weight", dh, side);
        VisionKernels.checkRelPosAttn(groups, side, heads, dh);
        int l = attnLocal();
        GpuBlock attn = alloc(rows, e);
        ops.enqueue(VisionKernels.REL_POS_ATTN,
                new long[]{VisionKernels.attnWorkItems(groups, heads, side, l)}, new long[]{l}, groups, side, heads,
                dh, (float) (1.0 / Math.sqrt(dh)), e, e, e, 0, 0, 0,
                q.buffer(), k.buffer(), v.buffer(), rw.buffer(), rh.buffer(), attn.buffer());
        GpuBlock out = (GpuBlock) linear(p + "attn.out.weight", attn, p + "attn.out.bias");
        addBlock(x, global ? out : unpartition(out, e, g, win));
        ScratchSlot normed2 = temp("sam.ln2", (long) n * e);
        ops.norm(NormKind.LAYER, w.vector(p + "post_ln.weight"), w.vectorOrNull(p + "post_ln.bias"), SamBackbone.EPS,
                x.buffer(), 0, normed2.buffer(), 0, n, e);
        GpuBlock up = (GpuBlock) linear(p + "mlp.lin1.weight", new GpuBlock(normed2, n, e), p + "mlp.lin1.bias");
        ops.unary(act, ActParams.NONE, up.buffer(), 0, up.buffer(), 0, up.size());
        GpuBlock down = (GpuBlock) linear(p + "mlp.lin2.weight", up, p + "mlp.lin2.bias");
        addBlock(x, down);
    }

    /** The local size of {@link VisionKernels#REL_POS_ATTN} on this device. */
    private int attnLocal() {
        return VisionKernels.attnLocal(local);
    }

    /** The relative-position rows of one axis, linearly resized when their length differs (a device upload). */
    private ScratchSlot relTable(String name, int dh, int side) {
        float[] table = SamWindows.relTable(hostVector(name), dh, side);
        ScratchSlot slot = temp("sam.rel." + name, table.length);
        writeFloats(slot.buffer(), 0, table);
        return slot;
    }

    /**
     * The {@code g x g} grid cut into {@code win x win} windows (zero-padded at the right and bottom): window
     * rows gathered over a copy of the block plus one zero row.
     */
    private GpuBlock partition(GpuBlock x, int e, int g, int win) {
        int nw = (g + win - 1) / win;
        int rows = nw * nw * win * win;
        ScratchSlot padded = temp("sam.pad", (long) (x.n + 1) * e);
        ops.copy(x.buffer(), 0, padded.buffer(), 0, x.size());
        ops.fill(padded.buffer(), (long) x.size() * Float.BYTES, (long) e * Float.BYTES, 0);
        GpuBlock out = alloc(rows, e);
        gather(SamWindows.partitionIndex(g, win), padded.buffer(), out.buffer(), e);
        return out;
    }

    /** The inverse of {@link #partition}: the windows back to the {@code g x g} grid. */
    private GpuBlock unpartition(GpuBlock windows, int e, int g, int win) {
        GpuBlock out = alloc(g * g, e);
        gather(SamWindows.unpartitionIndex(g, win), windows.buffer(), out.buffer(), e);
        return out;
    }

    /** The neck and the two stride-2 convs over the {@code g x g} rows of {@code x}. */
    private GpuBlock samNeck(GpuBlock x, int g, int e) {
        int c = neckChannels(SamTensors.NECK_0, e);
        int n = g * g;
        GpuBlock cur = (GpuBlock) linearKernel(SamTensors.NECK_0, c, e, x, null);
        ops.norm(NormKind.LAYER, w.vector(SamTensors.NECK_1 + ".weight"), w.vector(SamTensors.NECK_1 + ".bias"),
                SamBackbone.EPS, cur.buffer(), 0, cur.buffer(), 0, n, c);
        cur = (GpuBlock) linearKernel(SamTensors.NECK_2, c, c * 9,
                im2colPadded(cur, c, g, g, SamTensors.CONV_KERNEL, 1, 1, 1, g, g), null);
        ops.norm(NormKind.LAYER, w.vector(SamTensors.NECK_3 + ".weight"), w.vector(SamTensors.NECK_3 + ".bias"),
                SamBackbone.EPS, cur.buffer(), 0, cur.buffer(), 0, n, c);
        int k = SamTensors.CONV_KERNEL;
        int g2 = PixelShuffle.convOut(g, k, 2, 1);
        int c2 = neckChannels(SamTensors.NET_2, c);
        cur = (GpuBlock) linearKernel(SamTensors.NET_2, c2, c * k * k,
                im2colPadded(cur, c, g, g, k, 2, 1, 1, g2, g2), null);
        int g3 = PixelShuffle.convOut(g2, k, 2, 1);
        int c3 = neckChannels(SamTensors.NET_3, c2);
        return (GpuBlock) linearKernel(SamTensors.NET_3, c3, c2 * k * k,
                im2colPadded(cur, c2, g2, g2, k, 2, 1, 1, g3, g3), null);
    }

    /** The output channels of a SAM conv kernel: the kernel's fourth dim. */
    private int neckChannels(String name, int in) {
        TensorRef r = projectorMatrix(name);
        if (r == null || r.ne(3) <= 0 || r.ne(3) > Integer.MAX_VALUE || r.ne(2) != in) {
            throw new IllegalArgumentException("the SAM backbone conv " + name + " does not read " + in
                    + " channels");
        }
        return (int) r.ne(3);
    }

    /** The projector matrix {@code name} of the encoder graph, or null when it resolved none. */
    private TensorRef projectorMatrix(String name) {
        return model.projectorMatrices().get(name);
    }

    // ------------------------------------------------------------------ device helpers

    /** The block {@code b} of this executor. */
    GpuBlock block(Block b) {
        if (!(b instanceof GpuBlock)) {
            throw new IllegalArgumentException("the block does not belong to the GPU vision encoder");
        }
        return (GpuBlock) b;
    }

    /** A new block of the current encode call. */
    GpuBlock alloc(int n, int width) {
        return new GpuBlock(temp("b" + seq++, (long) n * width), n, width);
    }

    /** The scratch temporary of {@code role}, reused by every later request for the same role. */
    ScratchSlot temp(String role, long floats) {
        ScratchSlot s = scratch.temp(TEMP + role, floats);
        if (s.offset() != 0) {
            throw new IllegalStateException("scratch temporary " + s.name() + " does not start at element 0");
        }
        return s;
    }

    /** The key positions {@code 0 .. n-1} of a bidirectional attention. */
    ScratchSlot positions(int n) {
        ScratchSlot pos = temp("attn.pos", n);
        int[] v = new int[n];
        for (int i = 0; i < n; i++) {
            v[i] = i;
        }
        writeInts(pos.buffer(), v);
        return pos;
    }

    /**
     * Bidirectional flash attention of query rows {@code [qBegin, qBegin + nq)} over key rows {@code [kvBegin,
     * kvEnd)} into output rows {@code [outRow, outRow + nq)}, one direct (unsplit) launch.
     */
    void flash(FlashAttnTiling plan, AttnBuffers b, int qBegin, int nq, int outRow, int kvBegin, int kvEnd,
               float scale) {
        flash(plan, b, qBegin, nq, outRow, kvBegin, kvEnd, scale, false);
    }

    /**
     * {@link #flash(FlashAttnTiling, AttnBuffers, int, int, int, int, int, float)}, masking every key whose position
     * ({@code b.pos}) is past the query's when {@code causal}.
     */
    void flash(FlashAttnTiling plan, AttnBuffers b, int qBegin, int nq, int outRow, int kvBegin, int kvEnd,
               float scale, boolean causal) {
        FlashAttnTiling.Split split = plan.prefillSplit(nq, kvEnd - kvBegin, 0);
        Object[] fa = new Object[AttnKernels.FA_ARG_COUNT];
        fa[AttnKernels.ARG_FA_N_TOKENS] = nq;
        fa[AttnKernels.ARG_FA_N_HEAD] = plan.nHead();
        fa[AttnKernels.ARG_FA_N_HEAD_KV] = plan.nHeadKv();
        fa[AttnKernels.ARG_FA_D_K] = plan.dK();
        fa[AttnKernels.ARG_FA_D_V] = plan.dV();
        fa[AttnKernels.ARG_FA_BQ] = plan.bq();
        fa[AttnKernels.ARG_FA_BK] = plan.bk();
        fa[AttnKernels.ARG_FA_G_TILE] = plan.gTile();
        fa[AttnKernels.ARG_FA_N_SPLIT] = split.count();
        fa[AttnKernels.ARG_FA_SPLIT_LEN] = split.length();
        fa[AttnKernels.ARG_FA_KV_START] = kvBegin;
        fa[AttnKernels.ARG_FA_KV_END] = kvEnd;
        fa[AttnKernels.ARG_FA_Q_ROW_STRIDE] = b.qStride;
        fa[AttnKernels.ARG_FA_OUT_ROW_STRIDE] = b.outStride;
        fa[AttnKernels.ARG_FA_K_ROW_STRIDE] = b.kvStride;
        fa[AttnKernels.ARG_FA_K_HEAD_STRIDE] = plan.dK();
        fa[AttnKernels.ARG_FA_V_ROW_STRIDE] = b.kvStride;
        fa[AttnKernels.ARG_FA_V_HEAD_STRIDE] = plan.dV();
        fa[AttnKernels.ARG_FA_SCALE] = scale;
        fa[AttnKernels.ARG_FA_SOFTCAP] = 0.0f;
        fa[AttnKernels.ARG_FA_MAX_BIAS] = 0.0f;
        fa[AttnKernels.ARG_FA_FLAGS] = AttnKernels.attnFlags(causal, false, true);
        fa[AttnKernels.ARG_FA_SWA_TYPE] = SwaType.NONE.code();
        fa[AttnKernels.ARG_FA_N_SWA] = 0;
        fa[AttnKernels.ARG_FA_BIAS_MODE] = AttentionSpec.BiasMode.NONE.code();
        fa[AttnKernels.ARG_FA_N_BUCKETS] = 0;
        fa[AttnKernels.ARG_FA_MAX_DISTANCE] = 0;
        fa[AttnKernels.ARG_FA_Q_OFF] = qBegin * b.qStride;
        fa[AttnKernels.ARG_FA_OUT_OFF] = outRow * b.outStride;
        fa[AttnKernels.ARG_FA_BUF_Q] = b.q;
        fa[AttnKernels.ARG_FA_BUF_K] = b.k;
        fa[AttnKernels.ARG_FA_BUF_V] = b.v;
        fa[AttnKernels.ARG_FA_BUF_Q_POS] = b.pos;
        fa[AttnKernels.ARG_FA_BUF_KV_POS] = b.pos;
        fa[AttnKernels.ARG_FA_BUF_SINKS] = placeholders.readOnlyFloats();
        fa[AttnKernels.ARG_FA_BUF_REL_BIAS] = placeholders.readOnlyFloats();
        fa[AttnKernels.ARG_FA_BUF_OUT] = b.out;
        fa[AttnKernels.ARG_FA_BUF_PARTIAL] = placeholders.scratchFloats();
        fa[AttnKernels.ARG_FA_BUF_KV_SCALES] = placeholders.kvScalesOnes();
        ops.enqueue(AttnKernels.flashAttn(KvDType.F32), plan.prefillGlobal(nq, split.count()),
                FlashAttnTiling.prefillLocal(), fa);
    }

    /** {@code dst [n][width] = W . src + b} (no bias when {@code bias} is null). */
    void projectBiased(GpuWeight weight, GpuBuffer bias, GpuBuffer src, GpuBuffer dst, int n, int width) {
        ops.matmul(weight, src, 0, dst, 0, n);
        biasAdd(dst, bias, n, width);
    }

    /** {@code data[r*dim + d] += bias[d]} for {@code r < rows}; nothing for a null bias. */
    void biasAdd(GpuBuffer data, GpuBuffer bias, int rows, int dim) {
        if (bias == null) {
            return;
        }
        ops.enqueue(GpuOps.BATCH_BIAS_ADD, ActKernels.global1d(dim, local), new long[]{local}, dim, rows, bias,
                data);
    }

    /** {@code x[r][c] *= scale[c]} over {@code rows} rows of {@code width} ({@code LLM_MulRowScale}). */
    void rowScale(GpuBuffer x, GpuBuffer scale, int rows, int width) {
        ops.enqueue(NormKernels.MUL_ROW_SCALE, ActKernels.global1d((long) rows * width, local), new long[]{local},
                NormKernels.mulRowScaleArgs(rows, width, scale, width, x));
    }

    /** {@code dst row r = src row idx[r]} for rows of {@code rowLen} floats ({@code LLM_GatherRows}). */
    void gather(int[] idx, GpuBuffer src, GpuBuffer dst, int rowLen) {
        ScratchSlot table = temp("gather.idx", idx.length);
        writeInts(table.buffer(), idx);
        ops.enqueue(FlowKernels.GATHER_ROWS, new long[]{FlowKernels.gatherGlobal(idx.length, rowLen)},
                new long[]{FlowKernels.GATHER_LOCAL},
                idx.length, rowLen, rowLen, rowLen, 0, 0, table.buffer(), src, dst);
    }

    private static void requireRows(int[] idx, int rows, String what) {
        for (int i : idx) {
            if (i < 0 || i >= rows) {
                throw new IllegalArgumentException("row " + i + " is outside the " + rows + " rows of " + what);
            }
        }
    }

    void writeFloats(GpuBuffer dst, int offFloats, float[] values) {
        closeEvent(queue.enqueueWriteBuffer(dst, (long) offFloats * Float.BYTES, (long) values.length * Float.BYTES,
                values));
    }

    void writeInts(GpuBuffer dst, int[] values) {
        byte[] bytes = new byte[values.length * Integer.BYTES];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().put(values);
        closeEvent(queue.enqueueWriteBuffer(dst, 0, bytes.length, bytes));
    }

    /** Reads {@code n} floats after every enqueued step has run. */
    private float[] read(GpuBuffer src, int n) {
        float[] host = new float[n];
        closeEvent(queue.enqueueReadBuffer(src, 0, (long) n * Float.BYTES, host));
        queue.finish();
        return host;
    }

    private static void closeEvent(GpuEvent e) {
        if (e != null) {
            e.close();
        }
    }
}
