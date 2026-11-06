package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ColumnTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;

/**
 * The linear-attention classes built on the Qwen3.5 mixture: {@code Qwen4ExpTextModel} (LC/conversion/qwen4exp.py:
 * 18-222), whose per-layer n-gram PLE table is stored in shards indexed by {@code shard_<i>}, and the MiniCPM-V 4.6
 * / 4.7 text towers (LC/conversion/minicpm.py:121-252) of architecture QWEN35, or QWEN35MOE for 4.7 whose
 * {@code text_config.model_type} is {@code qwen3_5_moe_text} (minicpm.py:224-226). Qwen4Exp concatenates the PLE
 * shards by index into the F32 {@code per_layer_token_embd.weight}, reads the int64 PLE hash constants exactly and
 * writes them as INT64 keys, concatenates the MTP {@code fc_embedding}/{@code fc_hidden} into
 * {@code nextn.eh_proj}, and splits the QSA indexer's {@code index_qk_proj} into {@code q}/{@code k}. Both classes
 * inherit the Qwen3.5 tensor chain ({@link HfQwenLinearHooks}).
 */
public final class HfLinearHooksB {

    /** Qwen4Exp's MTP head tensors the shared Qwen remapper does not know (qwen4exp.py:32-36). */
    private static final Map<String, String> MTP_EXTRA;
    private static final String MTP_FC_EMBEDDING = "nextn_fc_embedding";
    private static final String MTP_FC_HIDDEN = "nextn_fc_hidden";

    static {
        Map<String, String> names = new HashMap<>();
        names.put("fc_embedding", MTP_FC_EMBEDDING);
        names.put("fc_hidden", MTP_FC_HIDDEN);
        names.put("hyper_connection_mixer", "nextn_hc_head");
        MTP_EXTRA = Collections.unmodifiableMap(names);
    }

    /** Converter tensor names ({@code MODEL_TENSOR}) of the tensors the Qwen4Exp hook builds itself. */
    private static final String INDEXER_Q_PROJ = "INDEXER_Q_PROJ";
    private static final String INDEXER_K_PROJ = "INDEXER_K_PROJ";
    private static final String NEXTN_EH_PROJ = "NEXTN_EH_PROJ";
    private static final String WEIGHT_SUFFIX = ".weight";

    private static final String PLE_SHARDS_STATE = "qwen4exp.pleShards";
    private static final String PLE_CONST_STATE = "qwen4exp.pleConstants";
    private static final String MTP_FC_STATE = "qwen4exp.mtpFc";
    private static final String QWEN35_MOE_TEXT = "qwen3_5_moe_text";
    private static final String ZYXT = "zyxt";
    private static final int MAX_HASH_ELEMENTS = 4096;
    private static final int INT64_BYTES = 8;

    private static final String[] NORM_PLUS_ONE = {
            ".ple.norm_key.weight", ".ple.norm_query.weight", ".ple.norm_conv.weight",
            ".indexer.q_layernorm.weight", ".indexer.k_layernorm.weight"};

    private HfLinearHooksB() {
    }

    /** A resolver to {@link Arch#QWEN4EXP}. */
    public static HfArchResolver qwen4ExpArch() {
        return HfArchResolver.of(Arch.QWEN4EXP);
    }

    /** A resolver to {@link Arch#QWEN35} (the MiniCPM-V 4.6 text tower). */
    public static HfArchResolver miniCpmV46Arch() {
        return HfArchResolver.of(Arch.QWEN35);
    }

    /**
     * The MiniCPM-V 4.7 architecture: QWEN35MOE when {@code text_config.model_type} is {@value #QWEN35_MOE_TEXT}
     * (minicpm.py:224-226), QWEN35 otherwise.
     */
    public static Arch miniCpmV47Arch(HfConfig cfg, com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex index) {
        Object modelType = cfg.text().raw("model_type");
        return QWEN35_MOE_TEXT.equals(modelType) ? Arch.QWEN35MOE : Arch.QWEN35;
    }

    // ------------------------------------------------------------------ MiniCPM-V 4.6 / 4.7 text

    /**
     * The MiniCPM-V 4.6 text drops (minicpm.py:126-135): the projector merger before the inherited filters, since
     * the merger tensors belong to the mmproj file.
     */
    static HfNameFilter miniCpmV46Filter() {
        return (name, base) -> {
            String t = base.apply(name);
            if (t == null) {
                return null;
            }
            if (t.startsWith("model.merger.") || t.startsWith("mtp")) {
                return null;
            }
            return t;
        };
    }

    /** The MiniCPM-V 4.7 text drops (minicpm.py:237-242): only the MTP tensors, the merger stays. */
    static HfNameFilter miniCpmV47Filter() {
        return (name, base) -> {
            String t = base.apply(name);
            if (t == null) {
                return null;
            }
            return t.startsWith("mtp") ? null : t;
        };
    }

    /** The MiniCPM-V 4.6 text metadata: {@link HfQwenLinearHooks#qwen35Meta}. */
    public static void miniCpm4_6Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfQwenLinearHooks.qwen35Meta(ctx, kv, plan);
    }

    /** The MiniCPM-V 4.7 text metadata, adding the ZYXT rope section order (minicpm.py:227-229). */
    public static void miniCpm4_7Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfQwenLinearHooks.qwen35Meta(ctx, kv, plan);
        kv.set(GgufKeys.ROPE_SECTION_ORDER, ZYXT);
    }

    // ------------------------------------------------------------------ Qwen4Exp

    /**
     * {@code Qwen4ExpTextModel.filter_tensors} (qwen4exp.py:45-55), composed over {@code HfQwenHooks.mtpFilter}:
     * {@code mtp.fc_embedding} and {@code mtp.fc_hidden} move to {@code model.layers.<n>.nextn_fc_<part>}; every
     * other name goes through the Qwen MTP mixin then the inherited filters.
     */
    public static HfNameFilter qwen4ExpFilter(HfLoadContext ctx) {
        int n = ctx.config().blockCount();
        boolean keep = ctx.options().includeMtp();
        HfNameFilter mtp = HfQwenHooks.mtpFilter(ctx);
        return (name, base) -> {
            String[] parts = name.split("\\.", 3);
            if (parts.length == 3 && "mtp".equals(parts[0]) && MTP_EXTRA.containsKey(parts[1])) {
                if (!keep) {
                    return null;
                }
                return "model.layers." + n + "." + MTP_EXTRA.get(parts[1]) + "." + parts[2];
            }
            return mtp.filter(name, base);
        };
    }

    /** Qwen4Exp's block count over the MTP mixin (qwen4exp.py:29-36). */
    public static int qwen4ExpBlockCount(HfLoadContext ctx) {
        return HfQwenHooks.mtpBlockCount(ctx);
    }

    /** {@code Qwen4ExpTextModel.set_gguf_parameters} (qwen4exp.py:72-115). */
    public static void qwen4ExpMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfQwenLinearHooks.qwen35Meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        int nLayer = cfg.requireInt("num_hidden_layers");
        kv.set(GgufKeys.HYPER_CONNECTION_COUNT, cfg.requireInt("hc_count"));
        kv.set(GgufKeys.HYPER_CONNECTION_LOW_RANK, cfg.requireInt("hc_lowrank"));
        kv.set(GgufKeys.ATTENTION_INDEXER_HEAD_COUNT, cfg.requireInt("indexer_n_heads"));
        kv.set(GgufKeys.ATTENTION_INDEXER_KEY_LENGTH, cfg.requireInt("indexer_head_dim"));
        kv.set(GgufKeys.ATTENTION_INDEXER_TOP_K, cfg.requireInt("indexer_budget"));
        double ratio = cfg.findFloat("indexer_compress_ratio")
                .orElseThrow(() -> new GgufFormatException(cfg.source()
                        + ": required key 'indexer_compress_ratio' is missing"));
        Object rawLayerTypes = cfg.find("layer_types").orElse(null);
        if (!(rawLayerTypes instanceof java.util.List)
                || !(((java.util.List<?>) rawLayerTypes).size() == nLayer)) {
            throw new GgufFormatException(cfg.source() + ": 'layer_types' must have one entry per hidden layer");
        }
        List<Double> ratios = new ArrayList<>(ctx.blockCount());
        for (int i = 0; i < nLayer; i++) {
            ratios.add("full_attention".equals(((java.util.List<?>) rawLayerTypes).get(i)) ? ratio : 0.0);
        }
        for (int i = nLayer; i < ctx.blockCount(); i++) {
            ratios.add(ratio);
        }
        kv.set(GgufKeys.ATTENTION_COMPRESS_RATIOS, ratios);
        PleConstants ple = ple(ctx);
        if (ple.layers.isEmpty() || isMtpOnly(ctx)) {
            return;
        }
        kv.set(GgufKeys.PLE_LAYERS, ple.layers);
        kv.set(GgufKeys.PLE_NGRAM_SIZE, cfg.requireInt("ngram_size"));
        kv.set(GgufKeys.PLE_HEADS_PER_NGRAM, cfg.requireInt("heads_per_ngram"));
        kv.set(GgufKeys.PLE_CONV_KERNEL, cfg.requireInt("ple_conv_kernel_size"));
        kv.set(GgufKeys.PLE_EOS_TOKEN_ID, eosTokenId(cfg));
        OptionalInt image = cfg.findInt("image_token_id");
        if (image.isPresent()) {
            kv.set(GgufKeys.PLE_IMAGE_TOKEN_ID, image.getAsInt());
        }
        if (ple.rowDim > 0) {
            kv.set(GgufKeys.EMBEDDING_LENGTH_PER_LAYER, ple.rowDim);
        }
        writeI64(kv, GgufKeys.PLE_LAYER_MULTIPLIERS, ple.multipliers);
        writeI64(kv, GgufKeys.PLE_HEAD_OFFSETS, ple.headOffsets);
        writeI64(kv, GgufKeys.PLE_HEAD_VOCAB_SIZES, ple.headSizes);
    }

    /** True when this file is an MTP-only export, which never reads PLE (qwen4exp.py:94-96). */
    private static boolean isMtpOnly(HfLoadContext ctx) {
        return !ctx.options().includeMtp();
    }

    /**
     * The per-load PLE facts (qwen4exp.py:57-70, :107-108, :131-140): the three int64 constant tensors read
     * exactly in row order at tensor time, the row dimension of the shards, and the configured layer ids (1-based;
     * empty means this file has no n-gram table).
     */
    private static final class PleConstants {
        final List<Integer> layers = new ArrayList<>();
        final long[] multipliers = new long[0];
        final long[] headOffsets = new long[0];
        final long[] headSizes = new long[0];
        final int rowDim;

        PleConstants(HfLoadContext ctx, HfConfig cfg, PleState state) {
            List<Integer> pleLayers = cfg.findIntList("ple_layer_ids").orElse(Collections.emptyList());
            for (int id : pleLayers) {
                layers.add(id - 1);
            }
            this.rowDim = state.rowDim;
        }
    }

    /** The per-load PLE state: the constant sources, read once per tensor, and the shard row dimension. */
    private static final class PleState {
        long[] multipliers = new long[0];
        long[] headOffsets = new long[0];
        long[] headSizes = new long[0];
        int rowDim;
        boolean resolved;

        void stash(String hfName, RowSource src) {
            if (hfName.endsWith("ple_embedding.layer_multipliers")) {
                multipliers = read(src);
            } else if (hfName.endsWith("ple_embedding.ngram_heads_offsets")) {
                headOffsets = read(src);
            } else {
                headSizes = read(src);
            }
        }

        private long[] read(RowSource src) {
            long len = 1;
            for (long d : src.shape()) {
                len = Math.multiplyExact(len, d);
            }
            if (len > MAX_HASH_ELEMENTS) {
                throw new GgufFormatException(src.describe() + ": PLE constant tensors of " + len
                        + " elements are unsupported (limit " + MAX_HASH_ELEMENTS + ")");
            }
            ByteBuffer buf = ByteBuffer.allocate((int) Math.multiplyExact(len, INT64_BYTES))
                    .order(ByteOrder.LITTLE_ENDIAN);
            try {
                src.readRows(0, (int) src.rows(), buf);
            } catch (IOException e) {
                throw new IllegalStateException("PLE constant " + src.describe() + " is unreadable: "
                        + e.getMessage(), e);
            }
            buf.flip();
            long[] values = new long[(int) len];
            for (int i = 0; i < values.length; i++) {
                values[i] = buf.getLong();
            }
            return values;
        }
    }

    private static PleState pleState(HfLoadContext ctx) {
        return ctx.state(PLE_CONST_STATE, PleState.class, PleState::new);
    }

    /** {@code _read_hash_constants} + the ple layer ids (qwen4exp.py:94-115). */
    private static PleConstants ple(HfLoadContext ctx) {
        PleState s = pleState(ctx);
        synchronized (s) {
            return new PleConstants(ctx, ctx.config(), s);
        }
    }

    /** The primary EOS token id; a list resets the n-gram hash on its last entry (qwen4exp.py:121-128). */
    private static int eosTokenId(HfConfig cfg) {
        Object v = cfg.find("eos_token_id").orElse(null);
        if (v instanceof List && !((List<?>) v).isEmpty()) {
            Object last = ((List<?>) v).get(((List<?>) v).size() - 1);
            if (last instanceof Integer || last instanceof Long) {
                return ((Number) last).intValue();
            }
            throw new GgufFormatException(cfg.source() + ": eos_token_id entries must be integers, got " + last);
        }
        if (v instanceof Integer || v instanceof Long) {
            return ((Number) v).intValue();
        }
        throw new GgufFormatException(cfg.source()
                + ": eos_token_id is required: the PLE hash resets its n-grams on it");
    }

    private static void writeI64(HfMetadataBuilder kv, String template, long[] values) {
        List<Long> list = new ArrayList<>(values.length);
        for (long v : values) {
            list.add(v);
        }
        kv.setTyped(kv.key(template), GGUFValueType.ARRAY, GGUFValueType.INT64,
                Collections.unmodifiableList(list));
    }

    /** {@code Qwen4ExpTextModel.modify_tensors} (qwen4exp.py:132-175). */
    public static void qwen4ExpTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.endsWith("ple_embedding.layer_multipliers")
                || hfName.endsWith("ple_embedding.ngram_heads_offsets")
                || hfName.endsWith("ple_embedding.ngram_heads_vocab_sizes")) {
            pleState(ctx).stash(hfName, src);
            out.drop(hfName, "Qwen4Exp PLE hash constants fold into the ple.* metadata keys");
            return;
        }
        if (hfName.contains(".ngram_embedding.shard_")) {
            placePleShard(ctx, hfName, src, out);
            return;
        }
        if (hfName.endsWith("." + MTP_FC_EMBEDDING + ".weight")
                || hfName.endsWith("." + MTP_FC_HIDDEN + ".weight")) {
            String kind = hfName.endsWith("." + MTP_FC_EMBEDDING + ".weight") ? MTP_FC_EMBEDDING : MTP_FC_HIDDEN;
            Handoff fc = ctx.state(MTP_FC_STATE, Handoff.class, Handoff::new);
            synchronized (fc) {
                fc.part(ctx, hfName, kind, src, out);
            }
            return;
        }
        if (hfName.contains(".indexer.index_qk_proj.weight")) {
            HfConfig cfg = ctx.config();
            int nQ = cfg.requireInt("indexer_n_heads") * cfg.requireInt("indexer_head_dim");
            int bid = HfHookSupport.layerOf(hfName);
            if (src.rows() <= nQ) {
                throw new GgufFormatException("tensor '" + hfName + "' has " + src.rows()
                        + " rows: indexer_n_heads * indexer_head_dim = " + nQ + " Q rows and K rows must follow");
            }
            out.add(ctx.names().ggufName(INDEXER_Q_PROJ, bid, WEIGHT_SUFFIX),
                    RowTransforms.sliceRows(src, 0, nQ));
            out.add(ctx.names().ggufName(INDEXER_K_PROJ, bid, WEIGHT_SUFFIX),
                    RowTransforms.sliceRows(src, nQ, src.rows() - nQ));
            return;
        }
        String name = hfName;
        RowSource s = src;
        for (String normName : NORM_PLUS_ONE) {
            if (name.endsWith(normName)) {
                s = HfHookSupport.normPlusOne(s);
                break;
            }
        }
        if (name.endsWith(".ple.conv1d.weight")) {
            s = ElementTransforms.squeezeConv(s);
        }
        HfQwenLinearHooks.vReorderTensors(ctx, name, s, HfSharedInputMerges.sink(ctx, name, out));
    }

    /**
     * {@code Qwen4ExpTextModel.prepare_tensors} (qwen4exp.py:250-262): neither MTP projection half, nor a merged
     * weight half, nor a PLE shard may be left over once the checkpoint is planned.
     */
    public static void qwen4ExpExtra(HfLoadContext ctx, TensorSink out) {
        Handoff fc = ctx.state(MTP_FC_STATE, Handoff.class, Handoff::new);
        synchronized (fc) {
            if (!fc.parts.isEmpty()) {
                throw new GgufFormatException("MTP projection missing its other half: " + fc.parts.keySet());
            }
        }
        HfSharedInputMerges.requireComplete(ctx);
        Shards shards = ctx.state(PLE_SHARDS_STATE, Shards.class, Shards::new);
        synchronized (shards) {
            int nParts = ctx.config().findInt("split_ngram_parts").orElse(0);
            if (!shards.byIndex.isEmpty() && shards.byIndex.size() != nParts) {
                throw new GgufFormatException("got " + shards.byIndex.size() + " PLE embedding shards, expected "
                        + nParts);
            }
        }
    }

    /**
     * {@code place_ple_shard} (qwen4exp.py:212-239): the shards concatenate by index into one F32 tensor of
     * {@code per_layer_token_embd.weight}.
     */
    private static void placePleShard(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        int idx;
        int dot = hfName.lastIndexOf(".shard_");
        int start = dot + ".shard_".length();
        int end = hfName.indexOf('.', start);
        String digits = hfName.substring(start, end < 0 ? hfName.length() : end);
        if (digits.isEmpty() || digits.length() > 9) {
            throw new GgufFormatException("tensor '" + hfName + "': invalid PLE shard index '" + digits + "'");
        }
        for (int i = 0; i < digits.length(); i++) {
            char c = digits.charAt(i);
            if (c < '0' || c > '9') {
                throw new GgufFormatException("tensor '" + hfName + "': invalid PLE shard index '" + digits + "'");
            }
        }
        idx = Integer.parseInt(digits);
        int nParts = ctx.config().requireInt("split_ngram_parts");
        if (nParts <= 0 || nParts > 256) {
            throw new GgufFormatException(ctx.config().source() + ": 'split_ngram_parts' must be in [1, 256]");
        }
        Shards shards = ctx.state(PLE_SHARDS_STATE, Shards.class, Shards::new);
        synchronized (shards) {
            if (shards.byIndex.containsKey(idx)) {
                throw new GgufFormatException("tensor '" + hfName + "': PLE shard " + idx + " repeats");
            }
            long[] shape = src.shape();
            if (shape.length < 2) {
                throw new GgufFormatException("tensor '" + hfName + "' must be a matrix, got rank " + shape.length);
            }
            int rowDim = (int) shape[shape.length - 1];
            shards.byIndex.put(idx, ElementTransforms.dequant(src, GGMLType.F32));
            shards.rowDim = Math.max(shards.rowDim, rowDim);
            if (shards.byIndex.size() < nParts) {
                return;
            }
            List<RowSource> ordered = new ArrayList<>(nParts);
            for (int i = 0; i < nParts; i++) {
                RowSource part = shards.byIndex.get(i);
                if (part == null) {
                    throw new GgufFormatException("PLE embedding shard " + i + " is missing (tensor '" + hfName
                            + "')");
                }
                long partRowDim = part.shape()[part.shape().length - 1];
                if (partRowDim != rowDim) {
                    throw new GgufFormatException("PLE shard " + i + " has row dim " + partRowDim
                            + ", expected " + rowDim);
                }
                ordered.add(part);
            }
            pleState(ctx).rowDim = Math.max(pleState(ctx).rowDim, rowDim);
            out.add(TensorNames.weight(TensorNames.PER_LAYER_TOKEN_EMBD),
                    RowTransforms.concatRows(Collections.unmodifiableList(ordered)));
        }
    }

    /** The indexed PLE shards of one load. */
    private static final class Shards {
        final Map<Integer, RowSource> byIndex = new LinkedHashMap<>();
        int rowDim;
    }

    /** The two MTP projection halves buffered until both arrive (qwen4exp.py:148-154). */
    private static final class Handoff {
        final Map<String, RowSource> parts = new LinkedHashMap<>();

        void part(HfLoadContext ctx, String hfName, String kind, RowSource src, TensorSink out) throws IOException {
            parts.put(kind, src);
            out.consume(hfName);
            if (parts.size() < 2) {
                return;
            }
            RowSource eh = ColumnTransforms.concatCols(java.util.Arrays.asList(
                    parts.get(MTP_FC_EMBEDDING), parts.get(MTP_FC_HIDDEN)));
            parts.clear();
            int bid = HfLoadContext.bidOf(hfName);
            if (bid < 0) {
                throw new GgufFormatException("tensor '" + hfName + "' has no layer number for its MTP head");
            }
            out.add(ctx.names().ggufName(NEXTN_EH_PROJ, bid, WEIGHT_SUFFIX), eh);
        }
    }
}
