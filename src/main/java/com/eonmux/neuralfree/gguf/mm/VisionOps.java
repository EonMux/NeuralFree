package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.List;
import java.util.Objects;
import java.util.function.IntFunction;

/**
 * The executor primitives of the clip vision graphs: the one surface the per-family graphs
 * ({@link LlavaGraph}, {@link InternVlGraph} and the rest) are written against, run by the CPU executor
 * ({@link CpuVisionOps}, float arrays over {@code CpuOps} / {@code CpuMatVec}) and the GPU executor
 * ({@link GpuVisionOps}, {@code GpuOps} and the LLM kernels). A {@link Block} is a device-resident row block
 * {@code [n][width]}, row {@code t} at {@code t * width}; the graph logic, the tower description
 * ({@link TowerOptions}) and the rotary plan ({@link RopePlan}) are shared verbatim, and each executor maps every
 * op onto its own device. Weights the graphs touch by registry name resolve per executor ({@code CpuTensorRef}
 * views or resident device tensors); tower tensors the graphs pass as {@link TensorRef}s. The small rows a graph
 * assembles on the host (position tables, class tokens, framing rows) move through {@link #upload},
 * {@link #addHost}, {@link #insertRow} and {@link #hostWeight}. Package-private executor support.
 */
interface VisionOps {

    /** Device-resident rows {@code [n][width]}, row {@code t} at {@code t * width}. */
    interface Block {
        /** The row count. */
        int n();

        /** The row width in values. */
        int width();
    }

    /** Observer of the residual stream between tower blocks ({@link TowerOptions#withTap}). */
    interface LayerTap {
        /**
         * Called with the stream after {@code done} blocks (0: after the pre-norm, before any block); {@code rows}
         * is the live block, so an observer that keeps it copies it ({@link #download}) and must not modify it.
         */
        void observe(int done, Block rows, int nTokens);
    }

    /**
     * The rotary plan of the tower's attention ({@link TowerOptions#withRope}), or of one {@link #rope} call: the
     * qwen vision m-rope, the two half-head rotations of clip.cpp {@code build_rope_2d}, or NEOX rotations of
     * fixed dim segments of every head ({@link NeoxSegment}).
     */
    final class RopePlan {
        final int[] qwenPositions;
        final float qwenBase;
        final int[] posA;
        final int[] posB;
        final double base;
        final boolean interleaveFreq;
        final List<NeoxSegment> neox;
        final float neoxBase;
        final boolean neoxMode;

        private RopePlan(int[] qwenPositions, float qwenBase, int[] posA, int[] posB, double base,
                         boolean interleaveFreq, List<NeoxSegment> neox, float neoxBase, boolean neoxMode) {
            this.qwenPositions = qwenPositions;
            this.qwenBase = qwenBase;
            this.posA = posA;
            this.posB = posB;
            this.base = base;
            this.interleaveFreq = interleaveFreq;
            this.neox = neox;
            this.neoxBase = neoxBase;
            this.neoxMode = neoxMode;
        }

        /** The qwen vision m-rope over {@code positions} ({@code 4 * nPatches} ints, {@code [y | x | y | x]}). */
        static RopePlan qwen(int[] positions) {
            return qwen(positions, QwenVisionGrid.ROPE_FREQ_BASE);
        }

        /** {@link #qwen(int[])} at frequency base {@code base}. */
        static RopePlan qwen(int[] positions, float base) {
            if (!(base > 0f)) {
                throw new IllegalArgumentException("the rope frequency base must be positive, got " + base);
            }
            return new RopePlan(Objects.requireNonNull(positions, "positions"), base, null, null, 0, false, null,
                    0f, false);
        }

        /**
         * clip.cpp {@code build_rope_2d}: {@code posA} on the first head half, {@code posB} on the second, each half
         * a normal-mode rope (adjacent pairs).
         */
        static RopePlan twoHalves(int[] posA, int[] posB, double base, boolean interleaveFreq) {
            return new RopePlan(null, 0f, Objects.requireNonNull(posA, "posA"), Objects.requireNonNull(posB, "posB"),
                    base, interleaveFreq, null, 0f, false);
        }

        /**
         * One {@code ggml_rope_ext} NEOX call per segment at frequency base {@code base}, the segment's offset set
         * by {@code ggml_rope_set_offset}; the dims outside every segment stay unrotated.
         */
        static RopePlan neox(List<NeoxSegment> segments, float base) {
            if (segments.isEmpty() || !(base > 0f)) {
                throw new IllegalArgumentException("a NEOX rope plan needs segments and a positive base, got "
                        + segments.size() + " segments at base " + base);
            }
            return new RopePlan(null, 0f, null, null, 0, false,
                    java.util.Collections.unmodifiableList(new java.util.ArrayList<>(segments)), base, false);
        }

        /**
         * The Gemma 4 vision rope (models/gemma4v.cpp): {@code posA} on the first head half, {@code posB} on the
         * second, each half a NEOX-mode rope over its {@code d_head / 2} dims (pairs {@code (i, i + d_head / 4)}).
         */
        static RopePlan neoxHalves(int[] posA, int[] posB, double base) {
            return new RopePlan(null, 0f, Objects.requireNonNull(posA, "posA"), Objects.requireNonNull(posB, "posB"),
                    base, false, null, 0f, true);
        }
    }

    /**
     * NEOX rotation of dims {@code [offset, offset + nDims)} of every head: dim {@code offset + i} pairs with
     * {@code offset + i + nDims / 2} at angle {@code positions[t] * base^(-2i / nDims)}. Immutable.
     */
    final class NeoxSegment {
        final int offset;
        final int nDims;
        final int[] positions;

        NeoxSegment(int offset, int nDims, int[] positions) {
            if (offset < 0 || offset % 2 != 0 || nDims <= 0 || nDims % 2 != 0) {
                throw new IllegalArgumentException("a NEOX segment needs an even offset and an even positive width,"
                        + " got " + offset + " and " + nDims);
            }
            this.offset = offset;
            this.nDims = nDims;
            this.positions = Objects.requireNonNull(positions, "positions").clone();
        }
    }

    /**
     * The per-call description of the shared ViT tower: the blocks to run, windows, rope, layout, mask, stream
     * taps, the attention scale, the value norm and the attention sinks.
     */
    final class TowerOptions {
        /** Every block, no windows, no rope, the pre-norm layout, bidirectional attention, no tap, the
         *  {@code 1/sqrt(d_head)} scale. */
        static final TowerOptions NONE = new TowerOptions(-1, null, null, false, false, null, -1, Float.NaN,
                false, false);

        final int nLayers;
        final IntFunction<VisionRows.KeyRanges> windows;
        final RopePlan rope;
        final boolean postNorm;
        final boolean causal;
        final LayerTap tap;
        /** First block of a segment ({@link #withBlocks}); -1 for the whole tower with its pre- and post-norm. */
        final int firstBlock;
        /** The softmax scale, NaN for {@code 1/sqrt(d_head)}. */
        final float attnScale;
        final boolean valueNorm;
        final boolean sinks;

        private TowerOptions(int nLayers, IntFunction<VisionRows.KeyRanges> windows, RopePlan rope, boolean postNorm,
                             boolean causal, LayerTap tap, int firstBlock, float attnScale, boolean valueNorm,
                             boolean sinks) {
            this.nLayers = nLayers;
            this.windows = windows;
            this.rope = rope;
            this.postNorm = postNorm;
            this.causal = causal;
            this.tap = tap;
            this.firstBlock = firstBlock;
            this.attnScale = attnScale;
            this.valueNorm = valueNorm;
            this.sinks = sinks;
        }

        /** Runs only the first {@code n} blocks (the llava graph stops at its deepest feature layer). */
        TowerOptions withLayers(int n) {
            if (n < 0) {
                throw new IllegalArgumentException("block count " + n + " is negative");
            }
            return new TowerOptions(n, windows, rope, postNorm, causal, tap, firstBlock, attnScale, valueNorm,
                    sinks);
        }

        /**
         * Runs the blocks {@code [from, to)} as one segment of a tower the graph interleaves with its own steps: no
         * pre-norm and no post-norm (the graph applies them), the window function and the tap indexed by the
         * global block index.
         */
        TowerOptions withBlocks(int from, int to) {
            if (from < 0 || to < from) {
                throw new IllegalArgumentException("block range [" + from + ", " + to + ") is invalid");
            }
            return new TowerOptions(to, windows, rope, postNorm, causal, tap, from, attnScale, valueNorm, sinks);
        }

        /**
         * The key range of every query row per layer index (a block-diagonal window attention); the function
         * returns null for a layer whose rows attend to all keys.
         *
         * @throws IllegalStateException when the options are causal, which no window combines with
         */
        TowerOptions withWindows(IntFunction<VisionRows.KeyRanges> ranges) {
            if (causal) {
                throw new IllegalStateException("a causal tower attends to every earlier row; it takes no windows");
            }
            return new TowerOptions(nLayers, Objects.requireNonNull(ranges, "windows"), rope, postNorm, causal, tap,
                    firstBlock, attnScale, valueNorm, sinks);
        }

        /** Rotates Q and K of every block with {@code plan}. */
        TowerOptions withRope(RopePlan plan) {
            return new TowerOptions(nLayers, windows, Objects.requireNonNull(plan, "rope"), postNorm, causal, tap,
                    firstBlock, attnScale, valueNorm, sinks);
        }

        /** The CogVLM block layout: attention and FFN read the stream unnormalized, their outputs are normed
         * before each residual add. */
        TowerOptions withPostNorm() {
            return new TowerOptions(nLayers, windows, rope, true, causal, tap, firstBlock, attnScale, valueNorm,
                    sinks);
        }

        /**
         * Row {@code t} of every block attends to rows {@code [0, t]} only: the causal text tower of a CLIP dual
         * encoder (HF {@code CLIPTextTransformer}).
         *
         * @throws IllegalStateException when the options carry windows
         */
        TowerOptions withCausal() {
            if (windows != null) {
                throw new IllegalStateException("a windowed tower cannot also attend causally");
            }
            return new TowerOptions(nLayers, null, rope, postNorm, true, tap, firstBlock, attnScale, valueNorm,
                    sinks);
        }

        /** Taps the residual stream before the first block and after every block (feature layers, deepstack). */
        TowerOptions withTap(LayerTap observer) {
            return new TowerOptions(nLayers, windows, rope, postNorm, causal,
                    Objects.requireNonNull(observer, "tap"), firstBlock, attnScale, valueNorm, sinks);
        }

        /** The softmax scale {@code scale} instead of {@code 1/sqrt(d_head)} (Gemma 4 attends at 1). */
        TowerOptions withAttnScale(float scale) {
            if (!(scale > 0f) || Float.isInfinite(scale)) {
                throw new IllegalArgumentException("attention scale " + scale + " is not positive and finite");
            }
            return new TowerOptions(nLayers, windows, rope, postNorm, causal, tap, firstBlock, scale, valueNorm,
                    sinks);
        }

        /** RMS-normalizes every value head without a weight before the attention (Gemma 4). */
        TowerOptions withValueNorm() {
            return new TowerOptions(nLayers, windows, rope, postNorm, causal, tap, firstBlock, attnScale, true,
                    sinks);
        }

        /**
         * Adds every block's per-head attention sink ({@link ClipLayer#attnSinks()}) to the softmax denominator
         * ({@code ggml_soft_max_add_sinks}); every block run must carry them.
         */
        TowerOptions withSinks() {
            return new TowerOptions(nLayers, windows, rope, postNorm, causal, tap, firstBlock, attnScale, valueNorm,
                    true);
        }

        /** The first block to run. */
        int from() {
            return Math.max(firstBlock, 0);
        }

        /** True when the call runs the whole tower, its pre- and post-norm included. */
        boolean wholeTower() {
            return firstBlock < 0;
        }
    }

    // ------------------------------------------------------------------ model

    /** The bound vision model: hparams, tensor geometry and the resolved tower and projector tensors. */
    ClipVisionModel model();

    /** The hparams of {@link #model()}. */
    ClipHparams hp();

    // ------------------------------------------------------------------ weights

    /** True when the projector graph resolved a matrix {@code name} (the file carries the optional tensor). */
    boolean hasMatrix(String name);

    /** True when the projector graph resolved a vector {@code name}. */
    boolean hasVector(String name);

    /**
     * A host copy of the projector vector {@code name}, the framing rows the graph assembles
     * ({@code v.token_embd.img_break}, begin/end rows).
     *
     * @throws IllegalStateException when the projector graph resolved no such vector
     */
    float[] hostVector(String name);

    /** A host copy of the projector vector {@code name}, or null when the file does not carry it. */
    float[] hostVectorOrNull(String name);

    /**
     * A host copy of a tower tensor (position tables, class tokens, conv weights), decoded when stored
     * otherwise. Small tables only; the block math reads device weights.
     */
    float[] hostWeight(TensorRef ref);

    // ------------------------------------------------------------------ host transfer

    /** Uploads {@code rows} as a block of rows {@code width} values wide. */
    Block upload(float[] rows, int width);

    /** The same values of {@code src} viewed as {@code n} rows of {@code width} (a merge-group reshape). */
    Block view(Block src, int n, int width);

    /** Downloads every value of {@code b}. */
    float[] download(Block b);

    /** The allocation mark of the blocks made so far in this encode call ({@link #release}). */
    int mark();

    /**
     * Returns every block made after {@code mark} to the executor for reuse by the blocks made next; the caller
     * reads none of them afterwards (a graph that loops over blocks keeps its device memory to one block's worth).
     */
    void release(int mark);

    // ------------------------------------------------------------------ encoder input

    /** The patches of {@code image} on the {@code pw x ph} grid, projected to {@code [pw * ph][n_embd]} rows. */
    Block patchRows(ClipImage image, int pw, int ph);

    /**
     * Projects the {@code nPatches} rows of a {@link ClipPatches} matrix to {@code [nPatches][n_embd]} rows,
     * with the patch bias, and the qwen second-frame kernel's product added when the file has one.
     */
    Block patchEmbed(float[] patches, int nPatches);

    /** {@code dst += rows}, the host rows of {@code dst}'s extent. */
    void addHost(Block dst, float[] rows);

    /** {@code dst += src}, two blocks of the same extent. */
    void addBlock(Block dst, Block src);

    /** A new block of {@code n + 1} rows: {@code src} with {@code values} inserted as row {@code at}. */
    Block insertRow(Block src, int at, float[] values);

    /** {@code dst += table[idx[t]]} for every row {@code t}: the bucketed position rows of the resampler graph. */
    void addGathered(Block dst, TensorRef table, int[] idx, int width);

    /** Output row {@code i} is input row {@code idx[i]}, each {@code width} values. */
    Block gatherRows(Block src, int[] idx, int width);

    // ------------------------------------------------------------------ algebra

    /** {@code [n][out]} rows: the matrix {@code name} times {@code x}, plus the bias {@code biasName} when given. */
    Block linear(String matrix, Block x, String biasName);

    /**
     * {@link #linear} over a conv kernel read as its im2col matmul: the kernel's bytes viewed as the
     * {@code [outRows][inWidth]} matrix (the HunyuanVL, GLM-4V and GLM-Edge convs).
     */
    Block linearKernel(String matrix, int outRows, int inWidth, Block x, String biasName);

    /** Norms {@code rows} rows of {@code width} values of {@code x} in place. */
    void norm(Block x, NormKind kind, String weight, String bias, float eps, int rows, int width);

    /** Norms {@code rows} rows of {@code width} values of {@code x} in place, the tower tensors by reference. */
    void normByRef(Block x, NormKind kind, TensorRef weight, TensorRef bias, float eps, int rows, int width);

    /** The unary op over every value of {@code x}, in place. */
    void activate(Block x, ActOp op);

    /** {@code up[t][i] = f(gate[t][i]) * up[t][i]} over {@code nRows} rows of {@code width}. */
    void glu(Block gate, Block up, GluOp op, int nRows, int width);

    /** {@link #glu} with the op's parameters ({@code ggml_swiglu_clamp}'s limit). */
    void glu(Block gate, Block up, GluOp op, ActParams params, int nRows, int width);

    /** A new block of the columns {@code [from, from + width)} of every row of {@code src} (a ggml view). */
    Block columns(Block src, int from, int width);

    /**
     * Rotates the rows of {@code x}, {@code nHeads} heads of {@code d_head} each, in place by {@code plan} (the
     * rope a graph applies to its own Q and K outside the shared tower).
     */
    void rope(Block x, int nHeads, RopePlan plan);

    /** {@code x[t][i] *= scale[i]} over every whole row ({@code ggml_mul} by a row vector). */
    void mulRowScale(Block x, float[] scale);

    /** {@code ggml_hardswish}: {@code x * min(max(x + 3, 0), 6) / 6} over every value, in place. */
    void hardswish(Block x);

    /** {@code ggml_hardsigmoid}: {@code min(max(x + 3, 0), 6) / 6} over every value, in place. */
    void hardsigmoid(Block x);

    // ------------------------------------------------------------------ reorders

    /** clip.cpp {@code build_patch_merge_permute} (Idefics3, Cohere2V, Kimi-VL, dots, PaddleOCR, LFM2). */
    Block mergePermute(Block src, int nEmbd, int pw, int ph, int s);

    /** The InternVL shuffle. */
    Block shuffleInternvl(Block src, int nEmbd, int pw, int ph, int s);

    /** The Llama 4 shuffle. */
    Block shuffleLlama4(Block src, int nEmbd, int pw, int ph, int s);

    /** Every {@code s x s} window unfolded channel-major (a stride-{@code s} conv's im2col, padded grid). */
    Block unfold(Block src, int nEmbd, int pw, int ph, int s);

    /** The first {@code ow x oh} windows of a {@code pw}-wide grid unfolded (unpadded conv drops the rest). */
    Block unfoldWindows(Block src, int nEmbd, int pw, int ow, int oh, int s);

    /**
     * The im2col of {@code ggml_conv_2d} over a {@code pw x ph} grid of {@code n_embd} rows: every {@code k x k}
     * window at stride {@code stride} with zero padding {@code pad}, unfolded channel-major, raster order over the
     * {@link PixelShuffle#convOut output grid} (the Step3-VL downsamplers).
     */
    Block im2col(Block src, int nEmbd, int pw, int ph, int k, int stride, int pad);

    /**
     * {@link #im2col} whose windows start {@code padTop} / {@code padLeft} cells before the grid (a SAME-padded
     * conv's asymmetric padding), laid out over the caller's {@code outW x outH} output grid.
     */
    Block im2colPadded(Block src, int nEmbd, int pw, int ph, int k, int stride, int padTop, int padLeft, int outW,
                       int outH);

    /**
     * Every {@code s x s} window of a {@code pw x ph} grid zero-padded at the right and bottom to a multiple of
     * {@code s}, unfolded channel-major ({@link PixelShuffle#unfoldPadded}; the DeepSeek-V4 aligner).
     */
    Block unfoldPadded(Block src, int nEmbd, int pw, int ph, int s);

    /** Output row {@code t} is row {@code t} of every block in turn ({@code ggml_concat} along the channels). */
    Block concatChannels(List<Block> blocks, int nRows, int width);

    /**
     * The map upscaled by the whole factor {@code f} per side, nearest ({@code ggml_upscale}; the MobileNetV5
     * fusion adapter).
     */
    Block upscaleNearest(Block src, int width, int height, int channels, int f);

    /**
     * {@code ggml_conv_2d_dw} with a {@code k x k} kernel at stride {@code stride}: weight value
     * {@code (c * k + ky) * k + kx} multiplies pixel {@code (x * stride + kx - padLeft, y * stride + ky - padTop)}
     * of channel {@code c}, taps outside the {@code width x height} map reading zero; the optional bias adds per
     * channel. Returns the {@code [outW * outH][channels]} rows of the {@code outW x outH} output grid.
     */
    Block depthwise(Block x, int width, int height, TensorRef weight, int k, int stride, int padTop, int padLeft,
                    int outW, int outH, TensorRef bias);

    /**
     * {@code ggml_pool_2d} average over the whole {@code kyH x kw}-cell windows at stride {@code kyH}, {@code kw}
     * and no padding: {@code [outH * outW][channels]} rows, raster order (the MobileNetV5 fusion adapter, the
     * yasa2 token pool and the column reduction of the response normalization).
     */
    Block avgPool(Block x, int width, int height, int channels, int kyH, int kw, int outW, int outH);

    /**
     * {@code convnext_grn} in place over the {@code rows} pixel rows of {@code width} channels:
     * {@code G_c = ||x_c||_2} over the rows, {@code N_c = G_c / max(mean_c G, 1e-6)},
     * {@code x + (x * N) * weight + bias} with the absent affine terms left out.
     */
    void grn(Block x, int rows, int width, TensorRef weight, TensorRef bias);

    /**
     * {@code ggml_conv_2d_dw} with a 3x3 kernel, zero padding 1, on a {@code width x height} feature map of
     * raster rows.
     */
    Block depthwise3x3(Block x, int width, int height, TensorRef weight, int stride);

    /**
     * Multi-head softmax attention: {@code nq} query rows of {@code nHead * dh} at row offset {@code qRowOff},
     * {@code nk} key and value rows of {@code kvWidth} at row offset {@code kvRowOff}, query head {@code h}
     * reading KV head {@code h / kvGroup}. Every query row attends to all keys.
     */
    Block attend(Block q, int qRowOff, int nq, Block k, Block v, int kvRowOff, int nk, int kvWidth, int nHead,
                 int kvGroup, int dh, float scale);

    /**
     * The DeepSeek-OCR SAM backbone over a square image ({@link SamBackbone}): {@code side * side} raster rows of
     * the backbone's output channels.
     *
     * @throws UnsupportedOperationException when the encoder graph carries no SAM backbone
     * @throws IllegalArgumentException      when the image is not square or not a patch multiple (named)
     */
    Block sam(ClipImage image);

    /**
     * {@link #attend} of {@code n} query rows over the same {@code n} key and value rows, each query row over its
     * own key range of {@code windows} (a block-diagonal window attention).
     *
     * @throws IllegalArgumentException when a key range leaves the {@code n} rows or the blocks do not fit
     */
    Block attendWindows(Block q, Block k, Block v, int n, int kvWidth, int nHead, int kvGroup, int dh, float scale,
                        VisionRows.KeyRanges windows);

    // ------------------------------------------------------------------ tower

    /**
     * The shared ViT tower over {@code nTokens} token-major rows in place: optional pre-norm, then per block
     * norm 1, attention (the tower's Q/K/V, the optional Q/K norms, the rope of {@code opts}, softmax at
     * {@code 1/sqrt(d_head)} over the layer's key windows, bidirectional or causal as {@code opts} selects, the
     * output projection, the optional layer scale {@code ls1}), the residual, norm 2, the FFN (the encoder's
     * activation, or the routed experts of a MoE block), the optional layer scale {@code ls2} and the residual (the
     * post-norm layout of CogVLM when {@code opts} selects it), and the optional tap; then the optional post-norm.
     */
    Block tower(Block x, int nTokens, NormKind norm, TowerOptions opts);
}
