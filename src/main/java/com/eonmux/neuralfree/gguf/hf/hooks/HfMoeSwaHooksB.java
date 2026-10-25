package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerJson;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Hooks of the sliding-window / sink MoE converter classes, part B: {@code MimoV2Model} (LC/conversion/mimo.py:18-403,
 * with the FP8 qkv dequant mimo.py:34-141 and the MXFP4 experts mimo.py:186-244) and {@code BeamModel}
 * (LC/conversion/beam.py:14-141). The Flint/Afmoe/Kolibri-1, Cohere-2 and Granite-SWA rows are in
 * {@link HfMoeSwaHooksA}, the Laguna/Maple/Mellum/ExaoneMoE/SmallThinker tail in {@link HfMoeSwaHooksC} and the
 * Step3.5/MuseSpark classes in {@link HfMoeSwaHooksD}.
 *
 * <p>Every method is a public static hook reference that subclasses compose where the converter calls
 * {@code super()}.</p>
 */
public final class HfMoeSwaHooksB {

    /** The block size of the FP8 scales (mimo.py:37, bs = 128). */
    private static final int BS = 128;

    /** The checkpoint name of an MiMo qkv weight. */
    private static final Pattern MIMO_QKV = Pattern.compile(
            "^model\\.(?:mtp\\.)?layers\\.(\\d{1,9})\\.self_attn\\.qkv_proj\\.weight$");

    private static final Pattern MIMO_MTP = Pattern.compile("^model\\.mtp\\.layers\\.(\\d{1,9})\\.(.+)$");

    /** The tensors of step3.py:284-286 that are squeezed to their layout shape. */
    private static final Set<String> STEP35_SQUEEZE = Collections.unmodifiableSet(new java.util.LinkedHashSet<>(
            Arrays.asList(".self_attn.g_proj.weight", ".moe.gate.weight", ".moe.up_proj.weight",
                    ".moe.gate_proj.weight", ".moe.down_proj.weight")));

    /** The NoPE query-scale constants beam.py:21-23 (reference values, not in config.json). */
    public static final double BEAM_TEMP_SCALE = 0.1;
    public static final int BEAM_TEMP_LENGTH = 2048;

    /** The per-load expert collector holder key of the B classes. */
    private static final String EXPERTS_STATE = HfMoeSwaHooksB.class.getName() + ".experts";

    private HfMoeSwaHooksB() {
    }

    // ------------------------------------------------------------------ shared helpers

    /** A little-endian F32 in-memory source a hook writes. */
    static RowSource f32(float... values) {
        return new FloatArraySource(values);
    }

    /** The float array source of {@link #f32} and the rope factor arrays. */
    static final class FloatArraySource implements RowSource {
        private final float[] values;
        private final long[] shape;

        FloatArraySource(float... values) {
            this(values, new long[]{values.length});
        }

        FloatArraySource(float[] values, long[] shape) {
            this.values = values;
            this.shape = shape;
        }

        @Override
        public String describe() {
            return "hook f32 " + Arrays.toString(shape);
        }

        @Override
        public ElementFormat format() {
            return ElementFormat.of(GGMLType.F32);
        }

        @Override
        public long[] shape() {
            return shape;
        }

        @Override
        public long rows() {
            return shape.length <= 1 ? 1 : shape[0];
        }

        @Override
        public long rowBytes() {
            return shape.length == 0 ? 4L : 4L * shape[shape.length - 1];
        }

        @Override
        public boolean rowLocal() {
            return true;
        }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) {
            dst.order(ByteOrder.LITTLE_ENDIAN);
            int perRow = (int) (shape[shape.length - 1]);
            for (int i = 0; i < count; i++) {
                long base = (first + i) * perRow;
                for (int j = 0; j < perRow; j++) {
                    dst.putFloat(values[(int) (base + j)]);
                }
            }
        }
    }

    /** {@code stringList} of a config list. */
    static List<String> stringList(HfConfig cfg, String name) {
        Object v = cfg.find(name).orElse(null);
        List<String> out = new ArrayList<>();
        if (v instanceof List) {
            for (Object o : (List<?>) v) {
                out.add(String.valueOf(o));
            }
        }
        return out;
    }

    /** {@code stringList} of an already materialized list. */
    private static List<String> stringListOf(Object v) {
        List<String> out = new ArrayList<>();
        if (v instanceof List) {
            for (Object o : (List<?>) v) {
                out.add(String.valueOf(o));
            }
        }
        return out;
    }

    /** The {@code bool} list of {@code name} when present, or null (shared with part A). */
    private static List<Boolean> flagList(HfConfig cfg, String name) {
        return HfMoeSwaHooksA.flagList(cfg, name);
    }

    /** The per-load expert collector over the gate/up/down projections. */
    private static HfHookSupport.ExpertCollector experts(HfLoadContext ctx, int nExperts) {
        HfMoeSwaHooksA.Holder holder = ctx.state(EXPERTS_STATE, HfMoeSwaHooksA.Holder.class,
                HfMoeSwaHooksA.Holder::new);
        synchronized (holder) {
            if (holder.collector == null) {
                holder.collector = new HfHookSupport.ExpertCollector(nExperts, HfMoeSwaHooksA.GATE_UP_DOWN,
                        HfHookSupport.MergedName.SAME_PREFIX);
            }
            return holder.collector;
        }
    }

    // ------------------------------------------------------------------ MimoV2Model

    /**
     * {@code MimoV2Model.__init__}'s block count (mimo.py:26-33): the trunk plus its 3 exported MTP layers
     * ({@code _n_nextn}), the layer count alone for a {@code --no-mtp} load.
     */
    public static int mimoBlocks(HfLoadContext ctx) {
        return HfQwenHooks.mtpBlockCount(ctx);
    }

    /**
     * {@code MimoV2Model.dequant_model}'s FP8 {@code qkv_proj} rewrites (mimo.py:98-141), as the class's index hook:
     * with a fp8 {@code quantization_config}, every {@code qkv_proj.weight} gets the row dequant of its
     * {@code weight_scale_inv} before the quant rewrite sees it, then the scale tensor is consumed.
     */
    public static HfTensorIndex mimoIndex(HfLoadContext ctx, HfTensorIndex index) {
        Object qc = ctx.config().raw().raw("quantization_config");
        if (!(qc instanceof Map) || !"fp8".equals(((Map<?, ?>) qc).get("quant_method"))) {
            return index;
        }
        Map<String, RowSource> replace = new HashMap<>();
        Set<String> remove = new java.util.LinkedHashSet<>();
        int trunk = ctx.config().blockCount();
        for (String name : index.names()) {
            Matcher m = MIMO_QKV.matcher(name);
            if (!m.matches() || !index.has(name + "_scale_inv")) {
                continue;
            }
            int bid = Integer.parseInt(m.group(1));
            boolean isMtp = name.startsWith("model.mtp.layers.");
            int slotBid = isMtp ? bid + trunk : bid;
            boolean isSwa = isMtp || hybridFlag(ctx.config(), slotBid);
            int nKv = ctx.config().requireInt(isSwa ? "swa_num_key_value_heads" : "num_key_value_heads");
            int vhd = ctx.config().requireInt("v_head_dim");
            replace.put(name, new Fp8QkvWeight(name, index.get(name), index.get(name + "_scale_inv"),
                    ctx.config().requireInt("num_attention_heads"), nKv,
                    ctx.config().requireInt("head_dim"), vhd));
            remove.add(name + "_scale_inv");
        }
        return replace.isEmpty() ? index : index.withReplaced(replace, remove);
    }

    /** One of the hybrid layer pattern's SWA flags (bid sized to the trunk). */
    private static boolean hybridFlag(HfConfig cfg, int bid) {
        List<Integer> hybrid = cfg.findIntList(Collections.emptyList(), "hybrid_layer_pattern");
        return bid < hybrid.size() && hybrid.get(bid) == 1;
    }

    /**
     * The dequantized, TP-regrouped FP8 {@code qkv_proj} rows
     * (mimo.py:143-172),
     * lazily read from the U8 weight bytes and the F32 scale rows: the weight row of a TP-sharded scale block is
     * {@code scale(rank * blocks_per_rank + rr / bs, block_col)}, and the per-rank {@code [Q|K|V]} rows are regrouped
     * into the fused un-sharded order.
     */
    static final class Fp8QkvWeight implements RowSource {
        private final String name;
        private final RowSource weight;
        private final RowSource scales;
        private final int rows;
        private final int k;
        private final int tp;
        private final int qPer;
        private final int kPer;
        private final int vPer;
        private final int rowsPerRank;
        private final int blocksPerRank;

        Fp8QkvWeight(String name, RowSource weight, RowSource scales, int nQ, int nKv, int hd, int vhd) {
            this.name = name;
            this.weight = weight;
            this.scales = scales;
            long[] w = weight.shape();
            if (w.length != 2 || w[0] > Integer.MAX_VALUE || w[1] > Integer.MAX_VALUE) {
                throw new GgufFormatException(name + ": a qkv_proj weight must be a 2-dimensional matrix, got "
                        + Arrays.toString(w));
            }
            this.k = (int) w[1];
            int qSize = nQ * hd;
            int kSize = nKv * hd;
            int vSize = nKv * vhd;
            int total = Math.addExact(qSize, Math.addExact(kSize, vSize));
            if ((int) w[0] != total) {
                throw new GgufFormatException(name + ": qkv_proj weight rows " + w[0] + " != q+k+v " + total);
            }
            this.qPer = qSize;
            this.kPer = kSize;
            this.vPer = vSize;
            int cand = 0;
            long scaleRows = scales.shape()[0];
            for (int c : new int[]{8, 4}) {
                if (total % c == 0) {
                    int rpr = total / c;
                    if (scaleRows == (long) c * ((rpr + BS - 1) / BS)) {
                        cand = c;
                        break;
                    }
                }
            }
            if (cand == 0) {
                throw new GgufFormatException(name + ": cannot detect TP - scale_inv rows " + scaleRows
                        + ", q+k+v " + total);
            }
            this.tp = cand;
            this.rowsPerRank = total / cand;
            this.blocksPerRank = (rowsPerRank + BS - 1) / BS;
            this.rows = total;
        }

        @Override
        public String describe() {
            return name + " dequantized (" + rows + "x" + k + ", tp=" + tp + ")";
        }

        @Override
        public ElementFormat format() {
            return ElementFormat.of(GGMLType.F32);
        }

        @Override
        public long[] shape() {
            return new long[]{rows, k};
        }

        @Override
        public long rows() {
            return rows;
        }

        @Override
        public long rowBytes() {
            return 4L * k;
        }

        @Override
        public boolean rowLocal() {
            return weight.rowLocal() && scales.rowLocal();
        }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            dst.order(ByteOrder.LITTLE_ENDIAN);
            byte[] wRow = new byte[k];
            ByteBuffer wBuf = ByteBuffer.wrap(wRow).order(ByteOrder.LITTLE_ENDIAN);
            float[] scaleRowBuf = new float[(k + BS - 1) / BS];
            List<List<ByteBuffer>> scaleCache = scaleCache();
            for (int i = 0; i < count; i++) {
                int outRow = (int) (first + i);
                long oldRow = oldRowOf(outRow);
                wBuf.clear();
                weight.readRows(oldRow, 1, wBuf);
                wBuf.flip();
                ByteBuffer scaleBytes = scaleRow(scaleCache, scaleRowOf(oldRow));
                for (int j = 0; j < k; j++) {
                    float scale = scaleBytes.getFloat((j / BS) * 4);
                    dst.putFloat(FpBlockDecoders.e4m3ToFp32(wRow[j] & 0xFF) * scale);
                }
            }
        }

        /** The fused-order output row's row in the sharded checkpoint order. */
        private long oldRowOf(int outRow) {
            if (outRow < tp * qPer) {
                return (long) (outRow / qPer) * rowsPerRank + outRow % qPer;
            }
            if (outRow < tp * (qPer + kPer)) {
                int d = outRow - tp * qPer;
                return (long) (d / kPer) * rowsPerRank + qPer + d % kPer;
            }
            int d = outRow - tp * (qPer + kPer);
            return (long) (d / vPer) * rowsPerRank + (long) qPer + kPer + d % vPer;
        }

        /** The per-row scale block-row of a sharded weight row (mimo.py:74-79). */
        private int scaleRowOf(long oldRow) {
            int rr = (int) (oldRow % rowsPerRank);
            int rank = (int) (oldRow / rowsPerRank);
            return rank * blocksPerRank + rr / BS;
        }

        /** The scale rows of the current read batch, deduplicated. */
        private List<List<ByteBuffer>> scaleCache() {
            return new ArrayList<>();
        }

        private ByteBuffer scaleRow(List<List<ByteBuffer>> cache, int scaleRow) throws IOException {
            int nb = (k + BS - 1) / BS;
            ByteBuffer buf = ByteBuffer.allocate(4 * nb).order(ByteOrder.LITTLE_ENDIAN);
            scales.readRows(scaleRow, 1, buf);
            buf.flip();
            return buf;
        }
    }

    /**
     * {@code MimoV2Model.set_gguf_parameters} (mimo.py:143-175) after the generic metadata: the KV head count array
     * over the MTP-extended {@code hybrid_layer_pattern}, the sliding window and pattern, the value width, the routed
     * experts, the partial-rotary rope width, the RMS epsilon, the optional value scale and NextN.
     */
    public static void mimoMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        int trunk = ctx.config().blockCount();
        int nNextn = Math.max(0, ctx.blockCount() - trunk);
        List<Integer> hybrid = new ArrayList<>(cfg.findIntList(Collections.emptyList(), "hybrid_layer_pattern"));
        while (hybrid.size() < trunk) {
            hybrid.add(0);
        }
        for (int i = 0; i < nNextn; i++) {
            hybrid.add(1);
        }
        int kvTrunk = cfg.requireInt("num_key_value_heads");
        int kvSwa = cfg.requireInt("swa_num_key_value_heads");
        List<Integer> kvArr = new ArrayList<>(hybrid.size());
        List<Boolean> pattern = new ArrayList<>(hybrid.size());
        for (Integer flag : hybrid) {
            kvArr.add(flag == 1 ? kvSwa : kvTrunk);
            pattern.add(flag == 1);
        }
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, kvArr);
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, cfg.requireInt("sliding_window"));
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, cfg.requireInt("v_head_dim"));
        kv.set(GgufKeys.EXPERT_COUNT, cfg.requireInt("n_routed_experts"));
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, cfg.requireInt("moe_intermediate_size"));
        double partial = cfg.findFloat(1.0, "partial_rotary_factor");
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, (int) Math.floor(cfg.requireInt("head_dim") * partial));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, cfg.findFloat(1e-5, "layernorm_epsilon"));
        Object vScale = cfg.find("attention_value_scale").orElse(null);
        if (vScale instanceof Number) {
            kv.set(GgufKeys.ATTENTION_VALUE_SCALE, ((Number) vScale).doubleValue());
        }
        if (nNextn > 0) {
            kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, nNextn);
        }
    }

    /** {@code MimoV2Model.filter_tensors} (mimo.py:319-337): the {@code attention_sink} tensors gain {@code .weight}. */
    public static final HfNameFilter MIMO_FILTER = (name, base) ->
            base.apply(name.contains("attention_sink") && !name.endsWith(".weight") ? name + ".weight" : name);

    /**
     * {@code MimoV2Model.modify_tensors} (mimo.py:339-395): the MTP prefix renames to the appended layer slots, the
     * per-expert stacking, everything else mapped.
     */
    public static void mimoTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        String name = hfName;
        Matcher m = MIMO_MTP.matcher(name);
        if (m.matches()) {
            int nextn = ctx.blockCount() - ctx.config().blockCount();
            int idx = Integer.parseInt(m.group(1));
            if (idx >= nextn) {
                throw new GgufFormatException("MTP layer " + idx + " is outside the " + nextn + " MTP layers");
            }
            name = "model.layers." + (ctx.config().blockCount() + idx) + "." + m.group(2);
        }
        if (name.contains(".mlp.experts.")) {
            HfHookSupport.layerOf(name);
            if (!experts(ctx, ctx.config().requireInt("n_routed_experts")).offer(ctx, name, out)) {
                throw new GgufFormatException("Unprocessed experts: ['" + name + "']");
            }
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, name, src, out);
    }

    /** {@code MimoV2Model.prepare_tensors} (mimo.py:397-403): leftover experts are invalid. */
    public static void mimoExtra(HfLoadContext ctx, TensorSink out) {
        HfMoeSwaHooksA.requireOrphans(ctx, EXPERTS_STATE);
    }

    // ------------------------------------------------------------------ BeamModel

    /**
     * {@code BeamModel.__init__} (beam.py:25-58): the layer-type and MLP-type array coverage, at least one
     * sliding-attention layer, the leading dense block and the attention-bias rejection.
     *
     * @throws GgufFormatException for any raised value
     */
    public static HfConfig beamFixup(HfConfig cfg) {
        int n = cfg.blockCount();
        List<String> types = stringList(cfg, "layer_types");
        if (types.size() != n) {
            throw new GgufFormatException(cfg.source() + ": Beam: layer_types must have num_hidden_layers entries");
        }
        boolean anySliding = false;
        for (String t : types) {
            if (!"sliding_attention".equals(t) && !"full_attention".equals(t)) {
                throw new GgufFormatException(cfg.source() + ": Beam: unknown layer type in " + types);
            }
            anySliding |= "sliding_attention".equals(t);
        }
        if (!anySliding) {
            throw new GgufFormatException(cfg.source() + ": Beam: expected at least one sliding_attention layer");
        }
        List<String> mlp = stringList(cfg, "mlp_layer_types");
        if (mlp.size() != n) {
            throw new GgufFormatException(cfg.source() + ": Beam: mlp_layer_types must have num_hidden_layers entries");
        }
        int dense = 0;
        while (dense < n && "dense".equals(mlp.get(dense))) {
            dense++;
        }
        for (int i = dense; i < n; i++) {
            if (!"sparse".equals(mlp.get(i))) {
                throw new GgufFormatException(cfg.source()
                        + ": Beam: llama.cpp supports dense MLP layers only as a leading block");
            }
        }
        if (cfg.findBool(false, "attention_bias")) {
            throw new GgufFormatException(cfg.source() + ": Beam: attention_bias=True is not supported");
        }
        return cfg;
    }

    /** The beam dense lead (beam.py:46-52), the {@code mlp_layer_types} leading dense prefix. */
    public static int beamLead(HfConfig cfg) {
        List<String> mlp = stringList(cfg, "mlp_layer_types");
        int dense = 0;
        int n = cfg.blockCount();
        while (dense < n && dense < mlp.size() && "dense".equals(mlp.get(dense))) {
            dense++;
        }
        return dense;
    }

    /**
     * {@code BeamModel.set_gguf_parameters} (beam.py:64-99) after the generic metadata: the rope width, the sliding
     * window and its {@code layer_types} pattern, the NoPE query-scale constants as the attention temperature keys,
     * the dense lead, the expert FFN width, the sigmoid gating, the routing norm and scale and the optional final
     * softcap.
     */
    public static void beamMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, cfg.requireInt("head_dim"));
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, cfg.requireInt("sliding_window"));
        List<Boolean> pattern = flagList(cfg, "layer_types");
        if (pattern == null) {
            throw new GgufFormatException(cfg.source() + ": Beam: layer_types must name each layer");
        }
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
        kv.set(GgufKeys.ATTENTION_TEMPERATURE_SCALE, BEAM_TEMP_SCALE);
        kv.set(GgufKeys.ATTENTION_TEMPERATURE_LENGTH, BEAM_TEMP_LENGTH);
        kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, beamLead(cfg));
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, cfg.requireInt("moe_intermediate_size"));
        HfMoeSwaHooksA.gatingSigmoid(kv);
        kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, cfg.findBool(true, "norm_topk_prob"));
        Object scale = cfg.find("routed_scaling_factor").orElse(null);
        if (!(scale instanceof Number)) {
            throw new GgufFormatException(cfg.source() + ": 'routed_scaling_factor' must be a number, got " + scale);
        }
        kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, ((Number) scale).doubleValue());
        Object softcap = cfg.find("final_logit_softcapping").orElse(null);
        if (softcap instanceof Number) {
            kv.set(GgufKeys.FINAL_LOGIT_SOFTCAPPING, ((Number) softcap).doubleValue());
        }
    }

    /** {@code BeamModel.filter_tensors} (beam.py:101-105): the attention gate bias gains its projection name. */
    public static final HfNameFilter BEAM_FILTER = (name, base) ->
            base.apply(name.endsWith(".self_attn.gate_bias")
                    ? name.replace(".self_attn.gate_bias", ".self_attn.gate_proj.bias") : name);

    /**
     * {@code BeamModel.modify_tensors} (beam.py:107-133): the routed experts of a sparse layer stack per projection, an
     * expert tensor of a leading dense layer is an error, everything else maps.
     */
    public static void beamTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.contains(".mlp.experts.")) {
            int lead = beamLead(ctx.config());
            if (HfHookSupport.layerOf(hfName) < lead) {
                throw new GgufFormatException("Beam: expert tensor " + hfName + " in a dense layer");
            }
            if (!experts(ctx, HfMoeSwaHooksA.expertCount(ctx.config())).offer(ctx, hfName, out)) {
                throw new GgufFormatException("Unprocessed experts: ['" + hfName + "']");
            }
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** {@code BeamModel.prepare_tensors} (beam.py:135-141): leftover experts are invalid. */
    public static void beamExtra(HfLoadContext ctx, TensorSink out) {
        HfMoeSwaHooksA.requireOrphans(ctx, EXPERTS_STATE);
    }

    /**
     * {@code BeamModel.set_vocab} (beam.py:60-75): the GPT-2 BPE vocabulary and the {@code <|return|>} eot token id
     * found among the tokenizer's added tokens.
     */
    public static Map<String, Object> beamVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre));
        int eot = addedTokenId(ctx, "<|return|>");
        if (eot >= 0) {
            md.put(GgufKeys.TOKENIZER_EOT_ID, eot);
        }
        return Collections.unmodifiableMap(md);
    }

    /** The id of {@code content} among the tokenizer's added tokens, or -1. */
    static int addedTokenId(HfLoadContext ctx, String content) throws IOException {
        HfTokenizerJson tj = ctx.tokenizerFiles().readTokenizerJson();
        if (tj == null) {
            return -1;
        }
        for (HfTokenizerJson.AddedToken t : tj.addedTokens()) {
            if (content.equals(t.content())) {
                return t.id();
            }
        }
        return -1;
    }
}
