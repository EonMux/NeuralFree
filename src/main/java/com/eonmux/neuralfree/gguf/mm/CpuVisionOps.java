package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

/**
 * The CPU executor of the shared vision graphs: every {@link VisionOps} op is float-array algebra over
 * {@code CpuOps} / {@code CpuMatVec} (the blocks are array views, so uploads and downloads are free and the
 * reorders run in place), and the tower is the shared {@link CpuVitTower}. Package-private executor support.
 */
final class CpuVisionOps implements VisionOps {

    /** Rows of a block as one flat array; the CPU executor mutates arrays in place. */
    private static final class ArrayBlock implements Block {
        final float[] rows;
        final int width;

        ArrayBlock(float[] rows, int width) {
            if (rows.length % width != 0) {
                throw new IllegalArgumentException(rows.length + " values do not split into rows of " + width);
            }
            this.rows = rows;
            this.width = width;
        }

        @Override
        public int n() {
            return rows.length / width;
        }

        @Override
        public int width() {
            return width;
        }
    }

    private final ClipVit vit;
    private final ClipVisionWeights w;
    /** The SAM backbone of the DeepSeek-OCR encoders; null for the other families. */
    private final SamBackbone sam;
    /** The LoRA site of every projector matrix the connector can adapt, by matrix name; empty without deltas. */
    private final Map<String, CpuProjectionDeltas.Site> sites;

    private CpuVisionOps(ClipVisionModel model, ClipVisionWeights w, CpuProjectionDeltas deltas) {
        this.vit = new ClipVit(model, w);
        this.w = w;
        this.sam = DeepseekOcrGraph.runsSam(model.encoderGraph()) ? new SamBackbone(model.hparams(), w) : null;
        Map<String, CpuProjectionDeltas.Site> served = new HashMap<>();
        if (deltas != null) {
            for (Map.Entry<String, TensorRef> e : model.projectorMatrices().entrySet()) {
                CpuTensorRef ref = w.matrixOrNull(e.getKey());
                if (ref != null) {
                    served.put(e.getKey(), deltas.serveModel(e.getValue().name(), ref, w.matVec));
                }
            }
        }
        this.sites = Collections.unmodifiableMap(served);
    }

    /**
     * Binds the CPU executor over {@code model} with the weights of {@code floats} / {@code refs}; matmuls run
     * through {@code matVec} in the tensors' stored types. The tower and connector matrices are
     * served to {@code deltas} when it is non-null, so a LoRA adapter's vision pairs apply to their products.
     */
    static CpuVisionOps bind(ClipVisionModel model, CpuMatVec matVec, Function<String, float[]> floats,
                             Function<String, CpuTensorRef> refs, CpuProjectionDeltas deltas) {
        Objects.requireNonNull(model, "model");
        return new CpuVisionOps(model, new ClipVisionWeights(model, matVec, floats, refs, deltas), deltas);
    }

    ClipVisionWeights weights() {
        return w;
    }

    @Override
    public ClipVisionModel model() {
        return vit.model;
    }

    @Override
    public ClipHparams hp() {
        return vit.hp;
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
        return w.vector(name);
    }

    @Override
    public float[] hostVectorOrNull(String name) {
        return w.vectorOrNull(name);
    }

    @Override
    public float[] hostWeight(TensorRef ref) {
        if (ref == null) {
            return null;
        }
        return vit.w.floats(ref);
    }

    // ------------------------------------------------------------------ host transfer

    @Override
    public Block upload(float[] rows, int width) {
        return new ArrayBlock(rows, width);
    }

    @Override
    public Block view(Block src, int n, int width) {
        ArrayBlock in = array(src);
        if (n * width != in.rows.length) {
            throw new IllegalArgumentException(n + " rows of " + width + " do not cover " + in.rows.length
                    + " values");
        }
        return new ArrayBlock(in.rows, width);
    }

    @Override
    public float[] download(Block b) {
        return array(b).rows;
    }

    @Override
    public int mark() {
        return 0;
    }

    @Override
    public void release(int mark) {
        // CPU blocks are heap arrays the collector reclaims once unreachable
    }

    private static ArrayBlock array(Block b) {
        if (!(b instanceof ArrayBlock)) {
            throw new IllegalArgumentException("the block does not belong to the CPU vision encoder");
        }
        return (ArrayBlock) b;
    }

    // ------------------------------------------------------------------ encoder input

    @Override
    public Block patchRows(ClipImage image, int pw, int ph) {
        return new ArrayBlock(vit.patchRows(image, pw, ph), vit.hp.nEmbd());
    }

    @Override
    public Block patchEmbed(float[] patches, int nPatches) {
        return new ArrayBlock(vit.patchEmbed(patches, nPatches),
                Math.toIntExact(vit.w.patchEmbd.rows()));
    }

    @Override
    public void addHost(Block dst, float[] rows) {
        float[] d = array(dst).rows;
        if (rows.length != d.length) {
            throw new IllegalArgumentException("the add needs " + d.length + " values, got " + rows.length);
        }
        VisionRows.addPrefix(rows, d, d.length);
    }

    @Override
    public void addBlock(Block dst, Block src) {
        float[] d = array(dst).rows;
        float[] s = array(src).rows;
        if (s.length != d.length) {
            throw new IllegalArgumentException("the add needs " + d.length + " values, got " + s.length);
        }
        VisionRows.addInto(s, d);
    }

    @Override
    public Block insertRow(Block src, int at, float[] values) {
        ArrayBlock in = array(src);
        if (values.length != in.width) {
            throw new IllegalArgumentException("the row needs " + in.width + " values, got " + values.length);
        }
        float[] out = new float[in.rows.length + in.width];
        System.arraycopy(in.rows, 0, out, 0, at * in.width);
        System.arraycopy(values, 0, out, at * in.width, in.width);
        System.arraycopy(in.rows, at * in.width, out, (at + 1) * in.width, in.rows.length - at * in.width);
        return new ArrayBlock(out, in.width);
    }

    @Override
    public void addGathered(Block dst, TensorRef table, int[] idx, int width) {
        ArrayBlock d = array(dst);
        float[] pos = hostWeight(table);
        if (pos.length % width != 0) {
            throw new IllegalArgumentException("the position table holds " + pos.length + " values, not rows of "
                    + width);
        }
        int posRows = pos.length / width;
        for (int t = 0; t < idx.length; t++) {
            int src = idx[t] * width;
            if (idx[t] < 0 || idx[t] >= posRows) {
                throw new IllegalArgumentException("position row " + idx[t] + " is outside the " + posRows
                        + " rows of " + table.name());
            }
            for (int c = 0; c < width; c++) {
                d.rows[t * width + c] += pos[src + c];
            }
        }
    }

    @Override
    public Block gatherRows(Block src, int[] idx, int width) {
        return new ArrayBlock(VisionRows.gatherRows(array(src).rows, idx, width), width);
    }

    // ------------------------------------------------------------------ algebra

    @Override
    public Block linear(String matrix, Block x, String biasName) {
        ArrayBlock in = array(x);
        float[] bias = biasName == null ? null : w.vectorOrNull(biasName);
        CpuTensorRef weight = w.matrix(matrix);
        return new ArrayBlock(vit.linear(weight, in.rows, in.n(), bias, sites.get(matrix)), outWidth(matrix));
    }

    @Override
    public Block linearKernel(String matrix, int outRows, int inWidth, Block x, String biasName) {
        ArrayBlock in = array(x);
        CpuTensorRef kernel = CpuVitTower.matrix(w.matrix(matrix), outRows, inWidth);
        float[] bias = biasName == null ? null : w.vectorOrNull(biasName);
        return new ArrayBlock(vit.linear(kernel, in.rows, in.n(), bias, kernelSite(matrix, outRows, inWidth)),
                outRows);
    }

    /**
     * The LoRA site of a conv kernel read through its im2col matrix: null when the connector cannot adapt it, and
     * refused by name when an adapter does adapt it but the reshaped product's shape differs from the tensor's.
     */
    private CpuProjectionDeltas.Site kernelSite(String matrix, int outRows, int inWidth) {
        CpuProjectionDeltas.Site site = sites.get(matrix);
        if (site == null || site.isEmpty() || (site.inDim() == inWidth && site.outDim() == outRows)) {
            return site;
        }
        throw new UnsupportedOperationException("LoRA tensor " + matrix + " is read as a " + outRows + " x "
                + inWidth + " im2col kernel, not its " + site.outDim() + " x " + site.inDim() + " tensor shape;"
                + " the connector cannot apply the delta to this product");
    }

    private int outWidth(String matrix) {
        return Math.toIntExact(w.matrix(matrix).rows());
    }

    @Override
    public void norm(Block x, NormKind kind, String weight, String bias, float eps, int rows, int width) {
        CpuVitTower.normRows(array(x).rows, array(x).rows, rows, width, w.vector(weight),
                bias == null ? null : w.vectorOrNull(bias), kind, eps);
    }

    @Override
    public void normByRef(Block x, NormKind kind, TensorRef weight, TensorRef bias, float eps, int rows,
                          int width) {
        CpuVitTower.normRows(array(x).rows, array(x).rows, rows, width,
                weight == null ? null : hostWeight(weight), bias == null ? null : hostWeight(bias), kind, eps);
    }

    @Override
    public void activate(Block x, ActOp op) {
        VisionRows.activate(op, array(x).rows);
    }

    @Override
    public void glu(Block gate, Block up, GluOp op, int nRows, int width) {
        glu(gate, up, op, ActParams.NONE, nRows, width);
    }

    @Override
    public void glu(Block gate, Block up, GluOp op, ActParams params, int nRows, int width) {
        ArrayBlock g = array(gate);
        ArrayBlock u = array(up);
        if ((long) nRows * g.width > g.rows.length || (long) nRows * u.width > u.rows.length || width > g.width
                || width > u.width) {
            throw new IllegalArgumentException(nRows + " rows of " + width + " do not fit the gate and up blocks");
        }
        CpuOps.gluRows(op, params, nRows, width, g.rows, 0, g.width, u.rows, 0, u.width, u.rows, 0, u.width);
    }

    @Override
    public Block columns(Block src, int from, int width) {
        ArrayBlock in = array(src);
        if (from < 0 || width <= 0 || from + width > in.width) {
            throw new IllegalArgumentException("columns [" + from + ", " + (from + width) + ") are outside rows of "
                    + in.width);
        }
        int n = in.n();
        float[] out = new float[n * width];
        for (int t = 0; t < n; t++) {
            System.arraycopy(in.rows, t * in.width + from, out, t * width, width);
        }
        return new ArrayBlock(out, width);
    }

    @Override
    public void rope(Block x, int nHeads, RopePlan plan) {
        ArrayBlock in = array(x);
        if (nHeads <= 0 || in.width != nHeads * vit.model.dHead()) {
            throw new IllegalArgumentException("rows of " + in.width + " are not " + nHeads + " heads of "
                    + vit.model.dHead());
        }
        posHook(plan).apply(in.rows, in.n(), nHeads);
    }

    @Override
    public void mulRowScale(Block x, float[] scale) {
        VisionRows.mulCols(array(x).rows, scale);
    }

    @Override
    public void hardswish(Block x) {
        float[] a = array(x).rows;
        for (int i = 0; i < a.length; i++) {
            a[i] = a[i] * Math.min(Math.max(a[i] + 3f, 0f), 6f) / 6f;
        }
    }

    @Override
    public void hardsigmoid(Block x) {
        float[] a = array(x).rows;
        for (int i = 0; i < a.length; i++) {
            a[i] = Math.min(Math.max(a[i] + 3f, 0f), 6f) / 6f;
        }
    }

    // ------------------------------------------------------------------ reorders

    @Override
    public Block mergePermute(Block src, int nEmbd, int pw, int ph, int s) {
        return new ArrayBlock(PixelShuffle.mergePermute(array(src).rows, nEmbd, pw, ph, s), nEmbd * s * s);
    }

    @Override
    public Block shuffleInternvl(Block src, int nEmbd, int pw, int ph, int s) {
        return new ArrayBlock(PixelShuffle.internvl(array(src).rows, nEmbd, pw, ph, s), nEmbd * s * s);
    }

    @Override
    public Block shuffleLlama4(Block src, int nEmbd, int pw, int ph, int s) {
        return new ArrayBlock(PixelShuffle.llama4(array(src).rows, nEmbd, pw, ph, s), nEmbd * s * s);
    }

    @Override
    public Block unfold(Block src, int nEmbd, int pw, int ph, int s) {
        return new ArrayBlock(PixelShuffle.unfold(array(src).rows, nEmbd, pw, ph, s), nEmbd * s * s);
    }

    @Override
    public Block im2col(Block src, int nEmbd, int pw, int ph, int k, int stride, int pad) {
        return new ArrayBlock(PixelShuffle.im2col(array(src).rows, nEmbd, pw, ph, k, stride, pad), nEmbd * k * k);
    }

    @Override
    public Block unfoldPadded(Block src, int nEmbd, int pw, int ph, int s) {
        return new ArrayBlock(PixelShuffle.unfoldPadded(array(src).rows, nEmbd, pw, ph, s), nEmbd * s * s);
    }

    @Override
    public Block unfoldWindows(Block src, int nEmbd, int pw, int ow, int oh, int s) {
        return new ArrayBlock(PixelShuffle.unfoldWindows(array(src).rows, nEmbd, pw, ow, oh, s), nEmbd * s * s);
    }

    @Override
    public Block im2colPadded(Block src, int nEmbd, int pw, int ph, int k, int stride, int padTop, int padLeft,
                              int outW, int outH) {
        return new ArrayBlock(PixelShuffle.im2colPadded(array(src).rows, nEmbd, pw, ph, k, stride, padTop, padLeft,
                outW, outH), nEmbd * k * k);
    }

    @Override
    public Block upscaleNearest(Block src, int width, int height, int channels, int f) {
        return new ArrayBlock(MobileNetV5Maps.upscaleNearest(array(src).rows, width, height, channels, f), channels);
    }

    @Override
    public Block concatChannels(List<Block> blocks, int nRows, int width) {
        List<float[]> arrays = new java.util.ArrayList<>(blocks.size());
        for (Block b : blocks) {
            arrays.add(array(b).rows);
        }
        return new ArrayBlock(VisionRows.concatChannels(arrays, nRows, width), width * blocks.size());
    }

    @Override
    public Block depthwise3x3(Block x, int width, int height, TensorRef weight, int stride) {
        ArrayBlock in = array(x);
        int d = in.width();
        if (in.n() != width * height) {
            throw new IllegalArgumentException("a " + width + "x" + height + " feature map needs " + width * height
                    + " rows, got " + in.n());
        }
        float[] kernel = hostWeight(weight);
        if (kernel.length != 9 * d) {
            throw new IllegalArgumentException(weight.name() + " holds " + kernel.length + " values, not a 3x3"
                    + " kernel per channel of " + d);
        }
        return new ArrayBlock(MobileVlmGraph.depthwise3x3(in.rows, width, height, d, kernel, stride), d);
    }

    @Override
    public Block depthwise(Block x, int width, int height, TensorRef weight, int k, int stride, int padTop,
                           int padLeft, int outW, int outH, TensorRef bias) {
        ArrayBlock in = array(x);
        int d = in.width();
        if (in.n() != width * height) {
            throw new IllegalArgumentException("a " + width + "x" + height + " feature map needs " + width * height
                    + " rows, got " + in.n());
        }
        float[] kernel = hostWeight(weight);
        if (kernel.length != k * k * d) {
            throw new IllegalArgumentException(weight.name() + " holds " + kernel.length + " values, not a " + k
                    + "x" + k + " kernel per channel of " + d);
        }
        float[] out = MobileNetV5Maps.depthwise(in.rows, width, height, d, kernel, k, stride, padTop, padLeft, outW,
                outH);
        VisionRows.addBias(out, bias == null ? null : hostWeight(bias));
        return new ArrayBlock(out, d);
    }

    @Override
    public Block avgPool(Block x, int width, int height, int channels, int kyH, int kw, int outW, int outH) {
        ArrayBlock in = array(x);
        if (in.width() != channels || in.n() != width * height) {
            throw new IllegalArgumentException("a " + width + "x" + height + " map of " + channels + " channels"
                    + " needs " + width * height + " rows of " + channels + ", the block holds " + in.n() + " rows"
                    + " of " + in.width());
        }
        return new ArrayBlock(MobileNetV5Maps.avgPool(in.rows, width, height, channels, kyH, kw, outW, outH),
                channels);
    }

    @Override
    public void grn(Block x, int rows, int width, TensorRef weight, TensorRef bias) {
        ArrayBlock in = array(x);
        if (in.width() != width || in.n() < rows || rows <= 0) {
            throw new IllegalArgumentException("the response norm reads " + rows + " rows of " + width + ", the"
                    + " block holds " + in.n() + " rows of " + in.width());
        }
        float[] a = in.rows;
        double[] sq = new double[width];
        for (int t = 0; t < rows; t++) {
            int o = t * width;
            for (int c = 0; c < width; c++) {
                sq[c] += (double) a[o + c] * a[o + c];
            }
        }
        float[] g = new float[width];
        double mean = 0;
        for (int c = 0; c < width; c++) {
            g[c] = (float) Math.sqrt(sq[c]);
            mean += g[c];
        }
        float m = Math.max((float) (mean / width), Yasa2Graph.GRN_EPS);
        float[] wv = weight == null ? null : hostWeight(weight);
        float[] bv = bias == null ? null : hostWeight(bias);
        float[] scale = new float[width];
        for (int c = 0; c < width; c++) {
            scale[c] = g[c] / m * (wv == null ? 1f : wv[c]);
        }
        for (int t = 0; t < rows; t++) {
            int o = t * width;
            for (int c = 0; c < width; c++) {
                float v = a[o + c] * scale[c];
                if (bv != null) {
                    v += bv[c];
                }
                a[o + c] += v;
            }
        }
    }

    @Override
    public Block attend(Block q, int qRowOff, int nq, Block k, Block v, int kvRowOff, int nk, int kvWidth,
                        int nHead, int kvGroup, int dh, float scale) {
        float[] out = VisionRows.attend(array(q).rows, qRowOff, nq, array(k).rows, array(v).rows, kvRowOff, nk,
                kvWidth, nHead, kvGroup, dh, scale, null);
        return new ArrayBlock(out, nHead * dh);
    }

    @Override
    public Block sam(ClipImage image) {
        if (sam == null) {
            throw new UnsupportedOperationException("the encoder graph " + vit.model.encoderGraph()
                    + " has no SAM backbone");
        }
        SamBackbone.Output out = sam.forward(image);
        return new ArrayBlock(out.rows, out.channels);
    }

    @Override
    public Block attendWindows(Block q, Block k, Block v, int n, int kvWidth, int nHead, int kvGroup, int dh,
                               float scale, VisionRows.KeyRanges windows) {
        Objects.requireNonNull(windows, "windows");
        VisionRows.keyRuns(windows, n);
        float[] out = VisionRows.attend(array(q).rows, 0, n, array(k).rows, array(v).rows, 0, n, kvWidth, nHead,
                kvGroup, dh, scale, windows);
        return new ArrayBlock(out, nHead * dh);
    }

    // ------------------------------------------------------------------ tower

    @Override
    public Block tower(Block x, int nTokens, NormKind norm, TowerOptions opts) {
        ArrayBlock in = array(x);
        CpuVitTower.Options mapped = CpuVitTower.Options.NONE;
        if (opts.postNorm) {
            mapped = mapped.withPostNorm();
        }
        if (opts.windows != null) {
            mapped = mapped.withLayerWindows(opts.windows);
        }
        if (opts.causal) {
            mapped = mapped.withLayerWindows(il -> VisionRows.CAUSAL);
        }
        if (opts.rope != null) {
            mapped = mapped.withPos(posHook(opts.rope));
        }
        if (opts.nLayers >= 0) {
            mapped = mapped.withLayers(opts.nLayers);
        }
        if (opts.tap != null) {
            mapped = mapped.withTap((done, rows, n) -> opts.tap.observe(done, new ArrayBlock(rows, in.width()), n));
        }
        if (!opts.wholeTower()) {
            mapped = mapped.withBlocks(opts.from(), opts.nLayers);
        }
        if (!Float.isNaN(opts.attnScale)) {
            mapped = mapped.withAttnScale(opts.attnScale);
        }
        if (opts.valueNorm) {
            mapped = mapped.withValueNorm();
        }
        if (opts.sinks) {
            mapped = mapped.withSinks();
        }
        return new ArrayBlock(w.tower.forward(in.rows, nTokens, norm, mapped), in.width());
    }

    /**
     * The CPU hook of a rope plan: the qwen vision m-rope, the NEOX segments, the two-halves rope of clip.cpp, or
     * its NEOX-mode variant; the hook rotates every head of the rows it is given, Q and grouped K alike.
     */
    private CpuVitTower.PosHook posHook(RopePlan rope) {
        if (rope.qwenPositions != null) {
            return vit.qwenRope(rope.qwenPositions, rope.qwenBase);
        }
        if (rope.neox != null) {
            return vit.neoxRope(rope.neox, rope.neoxBase);
        }
        int dh = vit.model.dHead();
        int[] posA = rope.posA;
        int[] posB = rope.posB;
        double base = rope.base;
        if (rope.neoxMode) {
            return (x, n, nHeads) -> VisionRope.neoxHalves(x, posA, posB, n, nHeads, dh, base);
        }
        boolean interleave = rope.interleaveFreq;
        return (x, n, nHeads) -> VisionRope.twoHalves(x, posA, posB, n, nHeads, dh, base, interleave);
    }
}
