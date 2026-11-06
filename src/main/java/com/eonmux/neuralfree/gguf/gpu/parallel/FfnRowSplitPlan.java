package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.quant.InputTransform;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The feed-forward row split of a model ({@code neuralfree.split.mode=row}, llama.cpp {@code LLAMA_SPLIT_MODE_ROW},
 * vLLM's MLP tensor parallelism): every eligible trunk layer's dense FFN splits over the ranks on the hidden
 * dimension. Rank {@code r} holds its range of the gate and up projection rows with the matching slice of their bias
 * and scale vectors, and the matching column slice of the down projection, so each rank computes {@code
 * down_r(act(gate_r x, up_r x))} and an all-reduce sums the partials. An even split gives every rank {@code n_ff /
 * ranks} neurons; llama.cpp's {@code --tensor-split} shares size the ranges in proportion to their values, rounded
 * down to whole device blocks, with the last rank taking the remainder. The attention, the norms, the embedding, the
 * output head and every ineligible layer stay whole on the main device; llama.cpp's row split instead shards the
 * attention heads too (see {@code split.mode=tensor}), which this feed-forward-only seam does not.
 *
 * <p>A layer is eligible when its FFN kind is {@code DENSE}, it has no feature that reads the whole hidden width at
 * once (gemma3n's activation sparsity selects a global top-k over {@code n_ff}, bitnet's {@code ffn_sub_norm}
 * normalizes the whole row), its stack is the main trunk, {@code n_ff} divides into {@code ranks} whole neuron
 * groups aligned with the device block size (or, with explicit shares, {@code n_ff} is a whole number of blocks and
 * every rank's share holds at least one), and none of its projection weights carries a PRISM
 * input transform (the transforms are built for the full matmul width). Ineligible layers keep their full FFN; a
 * model where no layer is eligible is refused. The plan is immutable.</p>
 */
public final class FfnRowSplitPlan {

    /** The split FFN of one layer: its tensor suffixes and the hidden rows each rank holds. */
    public static final class Layer {
        private final String stack;
        private final int weightLayer;
        private final int il;
        private final int nIn;
        private final int nOut;
        private final int nFf;
        private final int[] starts;
        private final int[] sizes;
        private final String up;
        private final String gate;
        private final String down;
        private final boolean fusedUp;
        private final String upBias;
        private final String gateBias;
        private final String downBias;
        private final String downScale;
        private final String actScales;

        Layer(String stack, int weightLayer, int il, int nIn, int nOut, int nFf, int[] starts, int[] sizes,
              String up, String gate, String down, boolean fusedUp, String upBias, String gateBias, String downBias,
              String downScale, String actScales) {
            this.stack = stack;
            this.weightLayer = weightLayer;
            this.il = il;
            this.nIn = nIn;
            this.nOut = nOut;
            this.nFf = nFf;
            this.starts = starts.clone();
            this.sizes = sizes.clone();
            this.up = up;
            this.gate = gate;
            this.down = down;
            this.fusedUp = fusedUp;
            this.upBias = upBias;
            this.gateBias = gateBias;
            this.downBias = downBias;
            this.downScale = downScale;
            this.actScales = actScales;
        }

        public String stack() { return stack; }
        public int weightLayer() { return weightLayer; }
        /** Slot index of the layer, for messages. */
        public int il() { return il; }
        /** Input width of the FFN: the stream width the up projection reads. */
        public int nIn() { return nIn; }
        /** Output width of the FFN: the down projection's rows. */
        public int nOut() { return nOut; }
        /** The layer's hidden width {@code n_ff}. */
        public int nFf() { return nFf; }
        /** Hidden rows the first rank holds; equal to every rank's count in an even split. */
        public int shard() { return sizes[0]; }
        /** Hidden rows rank {@code rank} holds. */
        public int shard(int rank) {
            return sizes[rank];
        }
        /** The widest shard over the ranks, for buffer sizing. */
        public int maxShard() {
            int max = 0;
            for (int size : sizes) {
                max = Math.max(max, size);
            }
            return max;
        }
        /** Whether every rank holds the same number of hidden rows. */
        public boolean even() {
            for (int size : sizes) {
                if (size != sizes[0]) {
                    return false;
                }
            }
            return true;
        }
        /** First hidden row rank {@code rank} holds. */
        public int rowStart(int rank) { return starts[rank]; }
        /** Tensor suffix of the up projection ({@code ffn_up.weight}). */
        public String up() { return up; }
        /** Tensor suffix of the gate projection, or null when the gate is fused into the up projection. */
        public String gate() { return gate; }
        /** Tensor suffix of the down projection ({@code ffn_down.weight}). */
        public String down() { return down; }
        /** True when the up projection holds the gate and up halves fused ({@code 2 * n_ff} rows). */
        public boolean fusedUp() { return fusedUp; }
        /** Tensor suffix of the up projection's bias over the hidden width, or null. */
        public String upBias() { return upBias; }
        /** Tensor suffix of the gate projection's bias over the hidden width, or null. */
        public String gateBias() { return gateBias; }
        /** Tensor suffix of the down projection's bias over the output width, or null. */
        public String downBias() { return downBias; }
        /** Tensor suffix of the down projection's scalar output scale, or null. */
        public String downScale() { return downScale; }
        /** Tensor suffix of the per-neuron activation scales (mpt), or null. */
        public String actScales() { return actScales; }

        /** The full tensor name of a suffix of this layer. */
        public String tensor(String suffix) {
            return TensorNames.layerName(stack, weightLayer, suffix);
        }

        @Override
        public String toString() {
            return stack + "." + il() + " (n_ff " + nFf + ", shards " + java.util.Arrays.toString(sizes) + ")";
        }
    }

    private final int ranks;
    private final Map<String, Layer> byTensor;
    private final Map<String, Layer> byLayer;
    private final List<Layer> layers;
    private final List<String> ineligible;

    private FfnRowSplitPlan(int ranks, Map<String, Layer> byTensor, Map<String, Layer> byLayer, List<Layer> layers,
                            List<String> ineligible) {
        this.ranks = ranks;
        this.byTensor = byTensor;
        this.byLayer = byLayer;
        this.layers = layers;
        this.ineligible = ineligible;
    }

    /** Number of ranks the hidden dimension splits over: the main device plus its peers. */
    public int ranks() {
        return ranks;
    }

    /** The split layers, in model order. Unmodifiable. */
    public List<Layer> layers() {
        return Collections.unmodifiableList(layers);
    }

    /** The split layer a tensor belongs to, or null when the tensor is not split. */
    public Layer layerOf(String tensorName) {
        return byTensor.get(tensorName);
    }

    /** The split layer of a model layer, or null when the layer keeps its full FFN. */
    public Layer layerOf(LayerSpec layer) {
        return byLayer.get(layerKey(layer));
    }

    /** True when a tensor uploads as per-rank shards instead of one whole copy on the main device. */
    public boolean splits(String tensorName) {
        return byTensor.containsKey(tensorName);
    }

    /** Why a layer keeps its full FFN, in model order ({@code <stack>.<il>: <reason>}). Unmodifiable. */
    public List<String> ineligible() {
        return Collections.unmodifiableList(ineligible);
    }

    /**
     * Plans the split of {@code spec}'s trunk over {@code ranks} ranks, reading each tensor's device type from
     * {@code uploader}'s plans: a column slice of the down projection must keep whole device blocks.
     *
     * @throws UnsupportedOperationException when no layer is eligible; the message names the first disqualifier
     */
    public static FfnRowSplitPlan of(ModelSpec spec, int ranks, GpuWeightUploader uploader) {
        return of(spec, ranks, uploader, new double[0]);
    }

    /**
     * Plans the split of {@code spec}'s trunk over {@code ranks} ranks with the layer shares of llama.cpp's
     * {@code --tensor-split}: rank {@code r} holds the hidden rows of its share of the width, rounded down to whole
     * device blocks, and the last rank takes the remainder. An empty {@code shares} divides the width evenly. The
     * shares must hold one value per rank.
     *
     * @throws UnsupportedOperationException when no layer is eligible; the message names the first disqualifier
     * @throws IllegalArgumentException      for fewer than two ranks, a share count that does not match the ranks, or
     *                                       a non-positive share
     */
    public static FfnRowSplitPlan of(ModelSpec spec, int ranks, GpuWeightUploader uploader, double[] shares) {
        if (ranks < 2) {
            throw new IllegalArgumentException("a row split needs at least two ranks, got " + ranks);
        }
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(uploader, "uploader");
        Objects.requireNonNull(shares, "shares");
        if (shares.length > 0) {
            if (shares.length != ranks) {
                throw new IllegalArgumentException("the tensor split gives " + shares.length + " share(s) for "
                        + ranks + " rank(s)");
            }
            for (int i = 0; i < shares.length; i++) {
                if (!(shares[i] > 0.0) || !Double.isFinite(shares[i])) {
                    throw new IllegalArgumentException("tensor split share " + i + " must be a finite value above 0,"
                            + " got " + shares[i]);
                }
            }
        }
        List<Layer> layers = new ArrayList<>();
        Map<String, Layer> byLayer = new LinkedHashMap<>();
        List<String> ineligible = new ArrayList<>();
        String firstDisqualifier = null;
        for (LayerSpec l : spec.layers()) {
            if (l.ffnKind() != FfnKind.DENSE) {
                String reason = "the " + l.ffnKind() + " FFN has no hidden-dimension split";
                ineligible.add(describe(l) + ": " + reason);
                if (firstDisqualifier == null) {
                    firstDisqualifier = describe(l) + ": " + reason;
                }
                continue;
            }
            String reason = eligible(l, l.ffn(), ranks, uploader, shares);
            if (reason != null) {
                ineligible.add(describe(l) + ": " + reason);
                if (firstDisqualifier == null) {
                    firstDisqualifier = describe(l) + ": " + reason;
                }
                continue;
            }
            if (byLayer.containsKey(layerKey(l))) {
                // a loop slot reuses its physical layer's tensors; one split serves every slot
                continue;
            }
            Layer layer = layer(l, l.ffn(), ranks, uploader, shares);
            layers.add(layer);
            byLayer.put(layerKey(l), layer);
        }
        if (layers.isEmpty()) {
            throw new UnsupportedOperationException("split-mode row splits the dense feed-forward projections over"
                    + " the ranks, but every " + spec.arch().ggufName() + " layer is ineligible"
                    + (firstDisqualifier == null ? "" : "; the first disqualifier is " + firstDisqualifier));
        }
        return new FfnRowSplitPlan(ranks, byTensor(layers), byLayer, layers, ineligible);
    }

    private static String layerKey(LayerSpec l) {
        return l.stack() + ":" + l.weightLayer();
    }

    private static String describe(LayerSpec l) {
        return l.stack() + "." + l.il();
    }

    /** Why the layer keeps its full FFN under the split, or null when it splits. */
    private static String eligible(LayerSpec l, FfnSpec ffn, int ranks, GpuWeightUploader uploader,
                                   double[] shares) {
        if (l.norm() != null && l.norm().ffnSubNorm()) {
            return "ffn_sub_norm normalizes the whole hidden row, which a split breaks";
        }
        if (ffn.activationSparsityStdMul() > 0f) {
            return "activation sparsity selects a global top-k over the hidden width, which a split breaks";
        }
        if (!TensorNames.STACK_BLK.equals(l.stack())) {
            return "only the main trunk splits; the " + l.stack() + " stack keeps its full FFN";
        }
        GpuWeightUploader.TensorPlan down = plan(l, ffn.downWeight(), uploader);
        GpuWeightUploader.TensorPlan up = plan(l, ffn.upWeight(), uploader);
        if (down == null || up == null) {
            return "the model has no tensor " + (down == null ? fullName(l, ffn.downWeight())
                    : fullName(l, ffn.upWeight()));
        }
        if (up.ne().length < 2 || down.ne().length < 2) {
            return "the projections are not matrices (" + up.ne().length + " and " + down.ne().length
                    + " dimension(s))";
        }
        int nFf = ffn.nFf();
        long upRows = up.ne()[1];
        if (upRows != nFf && upRows != 2L * nFf) {
            return "ffn_up has " + upRows + " rows, which is neither the spec's n_ff " + nFf
                    + " nor a fused gate-and-up " + (2L * nFf);
        }
        if (down.ne()[0] != nFf) {
            return "ffn_down reads " + down.ne()[0] + " columns, but the spec's n_ff is " + nFf;
        }
        if (upRows == nFf) {
            GpuWeightUploader.TensorPlan gate = plan(l, ffn.gateWeight(), uploader);
            if (gate != null && (gate.ne().length < 2 || gate.ne()[1] != nFf)) {
                return "ffn_gate has " + (gate.ne().length < 2 ? "no rows" : gate.ne()[1] + " rows")
                        + ", but the spec's n_ff is " + nFf;
            }
        }
        long block = down.deviceType().getBlockSize();
        if (shares.length == 0) {
            if (nFf % (ranks * block) != 0) {
                return "n_ff " + nFf + " does not divide into " + ranks + " rank shard(s) aligned to whole "
                        + down.deviceType() + " blocks of " + block + " element(s)";
            }
        } else {
            if (nFf % block != 0) {
                return "n_ff " + nFf + " is not a whole number of " + down.deviceType() + " blocks of " + block
                        + " element(s)";
            }
            if (boundaries(nFf, ranks, (int) block, shares) == null) {
                return "the tensor split shares leave a rank without " + block + " whole hidden block(s) of n_ff "
                        + nFf;
            }
        }
        InputTransformLoader.Result transforms = uploader.transforms();
        if (transforms != null && !transforms.isEmpty()) {
            for (String suffix : new String[]{ffn.upWeight(), ffn.gateWeight(), ffn.downWeight()}) {
                GpuWeightUploader.TensorPlan p = plan(l, suffix, uploader);
                if (p != null) {
                    InputTransform t = transforms.forward(p.name());
                    if (t != null && !t.isIdentity()) {
                        return "the PRISM input transform of " + p.name() + " is built for the full matmul width";
                    }
                }
            }
            HadamardSpec h = transforms.hadamard(fullName(l, ffn.downWeight()));
            if (h != null) {
                return "the Hadamard rotation of " + fullName(l, ffn.downWeight())
                        + " is built for the full hidden width";
            }
        }
        return null;
    }

    /**
     * The hidden-row boundaries of each rank: the evenly divided shards when {@code shares} is empty, otherwise the
     * shares' cumulative fractions of {@code nFf} rounded down to whole device blocks with the last rank taking the
     * remainder. Returns null when a share would leave a rank empty.
     */
    private static int[][] boundaries(int nFf, int ranks, int block, double[] shares) {
        int[] starts = new int[ranks];
        int[] sizes = new int[ranks];
        if (shares.length == 0) {
            int each = nFf / ranks;
            for (int rank = 0; rank < ranks; rank++) {
                starts[rank] = rank * each;
                sizes[rank] = each;
            }
            return new int[][]{starts, sizes};
        }
        double sum = 0.0;
        for (double share : shares) {
            sum += share;
        }
        int low = 0;
        double cumulative = 0.0;
        for (int rank = 0; rank < ranks; rank++) {
            cumulative += shares[rank];
            int high = rank == ranks - 1 ? nFf
                    : (int) Math.floor(nFf * cumulative / sum / block) * block;
            starts[rank] = low;
            sizes[rank] = high - low;
            low = high;
        }
        for (int size : sizes) {
            if (size <= 0) {
                return null;
            }
        }
        return new int[][]{starts, sizes};
    }

    private static GpuWeightUploader.TensorPlan plan(LayerSpec l, String suffix, GpuWeightUploader uploader) {
        return uploader.plan(fullName(l, suffix));
    }

    private static String fullName(LayerSpec l, String suffix) {
        return TensorNames.layerName(l.stack(), l.weightLayer(), suffix);
    }

    /** The split layer: its projection and companion tensor suffixes that the file actually has. */
    private static Layer layer(LayerSpec l, FfnSpec ffn, int ranks, GpuWeightUploader uploader, double[] shares) {
        int nFf = ffn.nFf();
        GpuWeightUploader.TensorPlan downPlan = plan(l, ffn.downWeight(), uploader);
        int[][] bounds = boundaries(nFf, ranks, (int) downPlan.deviceType().getBlockSize(), shares);
        if (bounds == null) {
            throw new IllegalStateException("layer " + describe(l) + " passed eligibility without a valid split");
        }
        int[] starts = bounds[0];
        int[] sizes = bounds[1];
        GpuWeightUploader.TensorPlan up = plan(l, ffn.upWeight(), uploader);
        GpuWeightUploader.TensorPlan down = plan(l, ffn.downWeight(), uploader);
        boolean fused = up.ne()[1] == 2L * nFf;
        String gate = ffn.gateWeight();
        if (fused) {
            gate = null;
        } else if (plan(l, gate, uploader) == null) {
            gate = null;
        }
        String role = ffn.role().up();
        return new Layer(l.stack(), l.weightLayer(), l.il(), (int) up.ne()[0], (int) down.ne()[1], nFf, starts, sizes,
                ffn.upWeight(), gate, ffn.downWeight(), fused,
                vector(l, TensorNames.bias(role), uploader),
                gate == null ? null : vector(l, TensorNames.bias(ffn.role().gate()), uploader),
                vector(l, TensorNames.bias(ffn.role().down()), uploader),
                vector(l, TensorNames.scale(ffn.role().down()), uploader),
                vector(l, TensorNames.FFN_ACT + ".scales", uploader));
    }

    /** A companion tensor the file has (an F32 vector over the hidden or output width), or null. */
    private static String vector(LayerSpec l, String suffix, GpuWeightUploader uploader) {
        return plan(l, suffix, uploader) != null ? suffix : null;
    }

    private static Map<String, Layer> byTensor(List<Layer> layers) {
        Map<String, Layer> byTensor = new LinkedHashMap<>();
        for (Layer layer : layers) {
            for (String suffix : new String[]{layer.up, layer.gate, layer.down, layer.upBias, layer.gateBias,
                    layer.downBias, layer.downScale, layer.actScales}) {
                if (suffix != null) {
                    byTensor.put(layer.tensor(suffix), layer);
                }
            }
        }
        return Collections.unmodifiableMap(byTensor);
    }
}
