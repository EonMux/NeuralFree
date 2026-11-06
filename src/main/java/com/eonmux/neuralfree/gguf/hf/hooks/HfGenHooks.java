package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.gguf.tokenizer.hf.SentencePieceModel;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The audio-generator hooks: LC/conversion/pockettts.py {@code PocketTTSModel}
 * (:81-174), LC/conversion/qwen3tts.py {@code Qwen3TTSTalkerModel} (:41-185) and LC/conversion/wavtokenizer.py
 * {@code WavTokenizerDecModel} (:13-46). Each public static mirrors one converter method: the {@code *Meta} the
 * class's {@code set_gguf_parameters}, {@code *Tensors} its {@code modify_tensors}, {@code *Vocab} its
 * {@code set_vocab}, and the {@code *Filter}/{@code *Fixup} its {@code filter_tensors}/{@code __init__} where it has
 * one; a step the class runs through {@code super()} composes the parent hook the same way.
 *
 * <p>None of these checkpoints has name-map rows of its own (the tensor-naming tables list no {@code flow_lm.*},
 * {@code text_embedding}, {@code codec_embedding} or {@code posnet.*} patterns), so the tensor hooks build their
 * GGUF names directly with {@code ctx.names().ggufName(...)}. The PocketTTS hparams the converter derives from tensor
 * shapes ({@code _load_hparams}) are synthesized by the checkpoint detector before the hooks run; the Qwen3 TTS talker
 * config replaces {@code text_config} in {@code qwen3TtsFixup}; the WavTokenizer decoder writes no vocabulary.</p>
 */
public final class HfGenHooks {

    private HfGenHooks() {
    }

    // ------------------------------------------------------------------ shared

    /** The full float32 contents of {@code src}, for the folds and matvecs these hooks run (pockettts.py, qwen3tts.py). */
    static float[] floats(RowSource src, String name) throws IOException {
        RowSource f = ElementTransforms.convert(src, GGMLType.F32);
        long total = f.rows() * f.rowBytes();
        if (total > Integer.MAX_VALUE - 8) {
            throw new GgufFormatException("tensor '" + name + "' is too large to read as float32");
        }
        ByteBuffer b = ByteBuffer.allocate((int) total).order(ByteOrder.LITTLE_ENDIAN);
        f.readRows(0, (int) f.rows(), b);
        float[] values = new float[Math.toIntExact(total / 4L)];
        b.flip();
        b.asFloatBuffer().get(values);
        return values;
    }

    /** An F32 one-row tensor of {@code values}, appended to an embedding table. */
    private static RowSource oneRow(String describe, float... values) {
        return RopeTensors.fromFloats(new long[]{values.length}, values);
    }

    // ------------------------------------------------------------------ PocketTTSModel

    /**
     * {@code PocketTTSModel.set_gguf_parameters} (pockettts.py:81): the generic text-model keys. The hparams the
     * converter derives from tensor shapes ({@code _load_hparams}, pockettts.py:49-76) are already in
     * {@code ctx.config()} — the checkpoint detector synthesizes them — so the generic row writes everything.
     */
    public static void pocketTtsMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfGenericMetadata.apply(ctx, kv);
    }

    /**
     * {@code PocketTTSModel.modify_tensors} (pockettts.py:122-152): the {@code flow_lm.*} tensors map onto the
     * POCKETTTS arch (the mimi and flow-net tensors go to the mmproj, which this text entry does not plan); the
     * conditioner embedding is replaced by the extended table ({@link #pocketTtsEmbedding}) and the fused
     * {@code self_attn.in_proj} splits into Q/K/V ({@code _LAYER_TENSOR_MAP}, pockettts.py:84-90).
     */
    public static void pocketTtsTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (!hfName.startsWith("flow_lm.")) {
            out.drop(hfName, "the mimi and flow-net tensors belong to the PocketTTS mmproj");
            return;
        }
        if (hfName.equals("flow_lm.conditioner.embed.weight")) {
            out.add(ctx.names().ggufName("TOKEN_EMBD", -1, ".weight"), pocketTtsEmbedding(ctx));
            return;
        }
        if (hfName.startsWith("flow_lm.out_norm.")) {
            String suffix = "." + hfName.substring(hfName.lastIndexOf('.') + 1);
            out.add(ctx.names().ggufName("OUTPUT_NORM", -1, suffix), src);
            return;
        }
        if (hfName.startsWith("flow_lm.transformer.layers.")) {
            int bid = ctx.bid(hfName);
            if (bid < 0) {
                throw new GgufFormatException("tensor '" + hfName + "' has no layer number");
            }
            String key = hfName.substring(hfName.indexOf("layers." + bid + ".") + ("layers." + bid + ".").length());
            key = key.substring(0, key.lastIndexOf('.'));
            if (key.equals("self_attn.in_proj")) {
                RowSource[] qkv = RowTransforms.splitRows(src, src.shape()[0] / 3, src.shape()[0] / 3,
                        src.shape()[0] / 3);
                out.add(ctx.names().ggufName("ATTN_Q", bid, ".weight"), qkv[0]);
                out.add(ctx.names().ggufName("ATTN_K", bid, ".weight"), qkv[1]);
                out.add(ctx.names().ggufName("ATTN_V", bid, ".weight"), qkv[2]);
                return;
            }
            String tensor = pocketLayerTensor(key);
            if (tensor != null) {
                out.add(ctx.names().ggufName(tensor, bid, ".weight"), src);
                return;
            }
        }
        out.drop(hfName, "the PocketTTS text arch has no tensor for this name");
    }

    /** The POCKETTTS tensor of a {@code flow_lm} layer part (pockettts.py:84-90), else null. */
    private static String pocketLayerTensor(String key) {
        switch (key) {
            case "norm1": return "ATTN_NORM";
            case "norm2": return "FFN_NORM";
            case "self_attn.out_proj": return "ATTN_OUT";
            case "linear1": return "FFN_UP";
            case "linear2": return "FFN_DOWN";
            default: return null;
        }
    }

    /**
     * The extended token embedding (pockettts.py:163-174): the conditioner rows, then the {@code bos_before_voice}
     * latent (when the pack has it) and the {@code audio_bos} row {@code input_linear @ bos_emb} appended, in the
     * order {@code _embd_table} concatenates them.
     */
    private static RowSource pocketTtsEmbedding(HfLoadContext ctx) throws IOException {
        HfTensorIndex index = ctx.index();
        RowSource embed = index.get("flow_lm.conditioner.embed.weight");
        List<RowSource> rows = new ArrayList<>();
        rows.add(embed);
        if (index.has("flow_lm.bos_before_voice")) {
            rows.add(ElementTransforms.reshape(index.get("flow_lm.bos_before_voice"), 1, -1));
        }
        float[] bosEmb = floats(index.get("flow_lm.bos_emb"), "flow_lm.bos_emb");
        float[] inputLinear = floats(index.get("flow_lm.input_linear.weight"), "flow_lm.input_linear.weight");
        int cols = bosEmb.length;
        float[] audioBos = new float[cols];
        for (int c = 0; c < cols; c++) {
            float acc = 0f;
            for (int i = 0; i < bosEmb.length; i++) {
                acc += bosEmb[i] * inputLinear[c * bosEmb.length + i];
            }
            audioBos[c] = acc;
        }
        rows.add(oneRow("audio_bos", audioBos));
        return rows.size() == 1 ? rows.get(0) : RowTransforms.concatRows(rows);
    }

    /**
     * {@code PocketTTSModel.set_vocab} (pockettts.py:92-120): the Unigram SentencePiece model written as the UGM
     * (t5) tokenizer with its normalizer flags, and the last rows — not sentencepiece pieces — renamed to the learned
     * vectors of {@code _extra_tokens} (pockettts.py:154-161) as CONTROL tokens of score -1000.
     */
    public static Map<String, Object> pocketTtsVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = ctx.tokenizerFiles();
        if (files.hasGgufTokenizer()) {
            return HfVocabWriters.fromGguf(files);
        }
        java.nio.file.Path model = files.file(HfTokenizerFiles.TOKENIZER_MODEL);
        if (model == null) {
            throw new GgufFormatException("the checkpoint has no " + HfTokenizerFiles.TOKENIZER_MODEL);
        }
        SentencePieceModel sp = SentencePieceModel.read(model);
        if (sp.trainerSpec().modelType() != SentencePieceModel.ModelType.UNIGRAM) {
            throw new GgufFormatException(model + ": the PocketTTS tokenizer must be a Unigram SentencePiece model");
        }
        int nVocab = ctx.config().requireInt("vocab_size");
        String[] texts = new String[nVocab];
        int[] types = new int[nVocab];
        float[] scores = new float[nVocab];
        for (int id = 0; id < nVocab; id++) {
            if (id < sp.pieceCount()) {
                texts[id] = sp.piece(id);
                scores[id] = sp.score(id);
                types[id] = sp.type(id) == SentencePieceModel.PieceType.CONTROL ? 3 : 1;
            } else {
                texts[id] = "[PAD" + id + "]";
                scores[id] = -10000f;
                types[id] = 2;
            }
        }
        String[] extra = ctx.index().has("flow_lm.bos_before_voice")
                ? new String[]{"<|pad|>", "<|bos_before_voice|>", "<|audio_bos|>"}
                : new String[]{"<|pad|>", "<|audio_bos|>"};
        for (int i = 0; i < extra.length; i++) {
            int id = nVocab - extra.length + i;
            texts[id] = extra[i];
            types[id] = 3;
            scores[id] = -1000f;
        }
        Map<String, Object> md = new LinkedHashMap<>();
        md.put(GgufKeys.TOKENIZER_MODEL, "t5");
        md.put(GgufKeys.TOKENIZER_PRE, "default");
        md.put(GgufKeys.TOKENIZER_LIST, Arrays.asList(texts));
        List<Float> s = new ArrayList<>(nVocab);
        List<Integer> t = new ArrayList<>(nVocab);
        for (int id = 0; id < nVocab; id++) {
            s.add(scores[id]);
            t.add(types[id]);
        }
        md.put(GgufKeys.TOKENIZER_SCORES, s);
        md.put(GgufKeys.TOKENIZER_TOKEN_TYPE, t);
        SentencePieceModel.NormalizerSpec n = sp.normalizerSpec();
        md.put(GgufKeys.TOKENIZER_ADD_PREFIX, n.addDummyPrefix());
        md.put(GgufKeys.TOKENIZER_REMOVE_EXTRA_WS, n.removeExtraWhitespaces());
        byte[] charsmap = n.precompiledCharsmap();
        if (charsmap.length > 0) {
            List<Integer> bytes = new ArrayList<>(charsmap.length);
            for (byte x : charsmap) {
                bytes.add(x & 0xFF);
            }
            md.put(GgufKeys.TOKENIZER_PRECOMPILED_CHARSMAP, bytes);
        }
        md.put(GgufKeys.TOKENIZER_ADD_BOS, false);
        md.put(GgufKeys.TOKENIZER_ADD_EOS, false);
        return Collections.unmodifiableMap(md);
    }

    // ------------------------------------------------------------------ Qwen3TTSTalkerModel

    /** The five text-projection tensors folded into the embedding (qwen3tts.py:44-50). */
    private static final List<String> QWEN3_TTS_TEXT_PROJ = Collections.unmodifiableList(Arrays.asList(
            "model.text_embedding.weight",
            "text_projection.linear_fc1.weight",
            "text_projection.linear_fc1.bias",
            "text_projection.linear_fc2.weight",
            "text_projection.linear_fc2.bias"));

    /**
     * {@code Qwen3TTSTalkerModel.__init__} (qwen3tts.py:56-66): the talker config becomes the text config, its
     * {@code vocab_size} replaced by {@code text_vocab_size}; the talker's keys are flattened in over the root (so
     * they win, as the converter's {@code hparams["text_config"] = talker_config} does).
     */
    public static HfConfig qwen3TtsFixup(HfConfig cfg) {
        JsonView talker = cfg.raw().optObj("talker_config");
        if (talker.isEmpty()) {
            throw new GgufFormatException(cfg.source() + ": a Qwen3 TTS talker requires a 'talker_config' object");
        }
        Object tvs = talker.raw("text_vocab_size");
        if (!(tvs instanceof Number) || tvs instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": 'talker_config.text_vocab_size' is required");
        }
        Map<String, Object> text = new LinkedHashMap<>();
        for (Object k : talker.keys()) {
            text.put(k.toString(), talker.raw(k.toString()));
        }
        text.put("vocab_size", toLongValue(cfg.source() + ": key 'talker_config.vocab_size'", tvs));
        Map<String, Object> flat = new LinkedHashMap<>(cfg.flat());
        flat.putAll(text);
        return cfg.withFlat(flat);
    }

    /**
     * {@code Qwen3TTSTalkerModel.set_gguf_parameters} (qwen3tts.py:121-136): the generic keys (the fixup has made the
     * talker config the text config), then the suppress list of the extended vocabulary — every id past the last 1024
     * rows except the codec EOS —, the codec EOS as the model EOS, and no added EOS (qwen3tts.py:129-136).
     */
    public static void qwen3TtsMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        int textVocab = cfg.requireInt("vocab_size");
        int codecVocab = cfg.requireInt("text_vocab_size");
        int codecEos = cfg.requireInt("codec_eos_token_id");
        long total = (long) textVocab + codecVocab;
        List<Integer> suppress = new ArrayList<>();
        for (long i = total - 1024; i < total; i++) {
            if (i != (long) textVocab + codecEos) {
                suppress.add((int) i);
            }
        }
        kv.set(GgufKeys.TOKENIZER_SUPPRESS_TOKENS, suppress);
        kv.set(GgufKeys.TOKENIZER_EOS_ID, (int) (textVocab + codecEos));
        kv.set(GgufKeys.TOKENIZER_ADD_EOS, false);
    }

    /**
     * {@code Qwen3TTSTalkerModel.filter_tensors} (qwen3tts.py:138-146): only the {@code talker.*} tensors survive
     * (the {@code talker.code_predictor.*} are dropped), the {@code talker.} prefix removed before the parent filter
     * runs ({@code super().filter_tensors}).
     */
    public static HfNameFilter qwen3TtsFilter(HfLoadContext ctx) {
        return (name, base) -> {
            if (!name.startsWith("talker.") || name.startsWith("talker.code_predictor.")) {
                return null;
            }
            return base.apply(name.substring("talker.".length()));
        };
    }

    /**
     * {@code Qwen3TTSTalkerModel.modify_tensors} (qwen3tts.py:154-185): the {@code codec_embedding} rows are appended
     * to the folded text embedding to form the extended token embedding, the {@code codec_head} is the output, the
     * five text-projection tensors are folded into the embedding (the MLP applied once all five are buffered), and
     * the rest map through the Qwen3 tensor hook ({@code super().modify_tensors}).
     */
    public static void qwen3TtsTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.equals("model.codec_embedding.weight")) {
            Map<String, RowSource> st = ctx.state("qwen3tts", Map.class, LinkedHashMap::new);
            st.put("codec", src);
            emitQwen3TtsEmbedding(ctx, st, out);
            return;
        }
        if (hfName.equals("codec_head.weight")) {
            out.add(ctx.names().ggufName("OUTPUT", -1, ".weight"), src);
            return;
        }
        if (QWEN3_TTS_TEXT_PROJ.contains(hfName)) {
            Map<String, RowSource> st = ctx.state("qwen3tts", Map.class, LinkedHashMap::new);
            st.put(hfName, src);
            if (st.size() >= 5) {
                st.put("folded", foldQwen3TtsTextEmbedding(ctx, st));
                emitQwen3TtsEmbedding(ctx, st, out);
            }
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** {@code torch.cat([folded_text_embed, codec_embed], dim=0)} once both halves are buffered (qwen3tts.py:148-152). */
    private static void emitQwen3TtsEmbedding(HfLoadContext ctx, Map<String, RowSource> st, TensorSink out) {
        RowSource folded = st.get("folded");
        RowSource codec = st.get("codec");
        if (folded == null || codec == null || out.isConsumed("qwen3tts.embed")) {
            return;
        }
        out.add(ctx.names().ggufName("TOKEN_EMBD", -1, ".weight"),
                RowTransforms.concatRows(Arrays.asList(folded, codec)));
        out.consume("qwen3tts.embed");
    }

    /**
     * The folded text embedding (qwen3tts.py:173-181): {@code fc2(act_fn(fc1(embed)))} with the configuration's
     * {@code hidden_act} activation, computed over the buffered tensors.
     */
    private static RowSource foldQwen3TtsTextEmbedding(HfLoadContext ctx, Map<String, RowSource> st) throws IOException {
        String act = ctx.config().findStringOr("silu", "hidden_act");
        RowSource hidden = linear(st, st.get("text_projection.linear_fc1.weight"),
                st.get("text_projection.linear_fc1.bias"), st.get("model.text_embedding.weight"));
        hidden = activation(hidden, act);
        return linear(st, st.get("text_projection.linear_fc2.weight"),
                st.get("text_projection.linear_fc2.bias"), hidden);
    }

    /** A dense layer {@code W @ x + b} (x a single row) evaluated on the float contents of its inputs. */
    private static RowSource linear(Map<String, RowSource> st, RowSource w, RowSource b, RowSource x) throws IOException {
        RowSource[] needed = {w, b, x};
        for (RowSource r : needed) {
            if (r == null) {
                throw new GgufFormatException("the Qwen3 TTS text projection is incomplete; all five tensors"
                        + " are required");
            }
        }
        float[] wf = floats(w, "text projection weight");
        float[] bf = floats(b, "text projection bias");
        float[] xf = floats(x, "text projection input");
        long inDim = wf.length / Math.max(1, bf.length);
        float[] out = new float[bf.length];
        for (int o = 0; o < out.length; o++) {
            float acc = bf[o];
            for (long i = 0; i < inDim; i++) {
                acc += xf[(int) i] * wf[Math.toIntExact(o * inDim + i)];
            }
            out[o] = acc;
        }
        return oneRow("text projection", out);
    }

    /** The {@code hidden_act} element-wise activation (qwen3tts.py:36-40, 173-181). */
    private static RowSource activation(RowSource src, String act) throws IOException {
        float[] v = floats(src, "activation");
        for (int i = 0; i < v.length; i++) {
            switch (act) {
                case "gelu":
                    v[i] = (float) (0.5 * v[i] * (1.0 + Math.tanh(0.7978845608028654 * (v[i]
                            + 0.044715 * Math.pow(v[i], 3)))));
                    break;
                case "relu":
                    v[i] = Math.max(0f, v[i]);
                    break;
                case "silu":
                default:
                    v[i] = (float) (v[i] / (1.0 + Math.exp(-v[i])));
                    break;
            }
        }
        return oneRow("activation " + act, v);
    }

    /**
     * {@code Qwen3TTSTalkerModel.set_vocab} (qwen3tts.py:87-119): the text tokenizer with the codec vocabulary
     * appended. With a SentencePiece model the text half is the llama SPM list; otherwise the GPT-2 BPE base is used
     * and the chat template dropped (qwen3tts.py:118-119). The codec rows are the {@code _codec_token_names}
     * (qwen3tts.py:71-85) as CONTROL tokens of score 0.
     */
    public static Map<String, Object> qwen3TtsVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = ctx.tokenizerFiles();
        if (files.hasGgufTokenizer()) {
            return HfVocabWriters.fromGguf(files);
        }
        HfVocabWriters.Input in = HfHookSupport.vocabInput(ctx);
        List<String> codecNames = qwen3TtsCodecNames(ctx, ctx.config().requireInt("text_vocab_size"));
        Map<String, Object> md;
        if (HfVocabWriters.canSentencePiece(in)) {
            md = new LinkedHashMap<>(HfVocabWriters.sentencePiece(in, pre));
        } else {
            md = new LinkedHashMap<>(HfVocabWriters.gpt2(in, pre));
            md.remove(GgufKeys.TOKENIZER_CHAT_TEMPLATE);
        }
        appendCodec(md, codecNames);
        return Collections.unmodifiableMap(md);
    }

    /** The codec token names (qwen3tts.py:71-85): {@code <|codec_i|>} rows, the {@code *_id} talker fields renamed. */
    private static List<String> qwen3TtsCodecNames(HfLoadContext ctx, int n) {
        List<String> names = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            names.add("<|codec_" + i + "|>");
        }
        JsonView talker = ctx.config().raw().optObj("talker_config");
        for (Object key : talker.keys()) {
            String k = key.toString();
            if (!k.endsWith("_id")) {
                continue;
            }
            String prefix = k.substring(0, k.length() - 3);
            Object val = talker.raw(k);
            if (val instanceof Integer || val instanceof Long) {
                int id = ((Number) val).intValue();
                if (id >= 0 && id < n) {
                    names.set(id, "<|" + prefix + "|>");
                }
            } else if (val instanceof Map) {
                for (Map.Entry<?, ?> e : ((Map<?, ?>) val).entrySet()) {
                    if (e.getValue() instanceof Integer || e.getValue() instanceof Long) {
                        int id = ((Number) e.getValue()).intValue();
                        if (id >= 0 && id < n) {
                            names.set(id, "<|" + prefix + "_" + e.getKey() + "|>");
                        }
                    }
                }
            }
        }
        return names;
    }

    /** Appends the codec rows (CONTROL, score 0) to the text token list. */
    private static void appendCodec(Map<String, Object> md, List<String> codecNames) {
        Object list = md.get(GgufKeys.TOKENIZER_LIST);
        if (!(list instanceof List)) {
            throw new GgufFormatException("the Qwen3 TTS text tokenizer has no token list to extend");
        }
        List<?> base = (List<?>) list;
        List<Object> tokens = new ArrayList<>(base.size() + codecNames.size());
        tokens.addAll(base);
        tokens.addAll(codecNames);
        md.put(GgufKeys.TOKENIZER_LIST, tokens);
        Object types = md.get(GgufKeys.TOKENIZER_TOKEN_TYPE);
        if (types instanceof List) {
            List<Integer> tt = new ArrayList<>((List<Integer>) types);
            for (int i = 0; i < codecNames.size(); i++) {
                tt.add(3);
            }
            md.put(GgufKeys.TOKENIZER_TOKEN_TYPE, tt);
        }
        Object scores = md.get(GgufKeys.TOKENIZER_SCORES);
        if (scores instanceof List) {
            List<Float> sc = new ArrayList<>((List<Float>) scores);
            for (int i = 0; i < codecNames.size(); i++) {
                sc.add(0f);
            }
            md.put(GgufKeys.TOKENIZER_SCORES, sc);
        }
    }

    // ------------------------------------------------------------------ WavTokenizerDecModel

    /**
     * {@code WavTokenizerDecModel.set_gguf_parameters} (wavtokenizer.py:32-46): the generic keys (the class calls
     * {@code super()}), then the decoder's feature width, feed-forward width, group-norm parameters and the
     * posnet/convnext geometry, and the bidirectional attention.
     */
    public static void wavTokenizerDecMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.FEATURES_LENGTH, (int) cfg.findLong(0, "n_embd_features"));
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, (int) cfg.findLong(0, "n_ff"));
        kv.set(GgufKeys.ATTENTION_GROUPNORM_EPS, cfg.findFloat(1e-5, "group_norm_epsilon"));
        kv.set(GgufKeys.ATTENTION_GROUPNORM_GROUPS, (int) cfg.findLong(0, "group_norm_groups"));
        kv.set(GgufKeys.POSNET_EMBEDDING_LENGTH, (int) subInt(cfg, "posnet", "n_embd"));
        kv.set(GgufKeys.POSNET_BLOCK_COUNT, (int) subInt(cfg, "posnet", "n_layer"));
        kv.set(GgufKeys.CONVNEXT_EMBEDDING_LENGTH, (int) subInt(cfg, "convnext", "n_embd"));
        kv.set(GgufKeys.CONVNEXT_BLOCK_COUNT, (int) subInt(cfg, "convnext", "n_layer"));
        kv.set(GgufKeys.ATTENTION_CAUSAL, false);
    }

    /** A required int of the nested configuration object {@code <path>.<key>} (wavtokenizer.py:40-44). */
    private static long subInt(HfConfig cfg, String path, String key) {
        JsonView sub = cfg.raw().optObj(path);
        Object v = sub.raw(key);
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": required key '" + path + "." + key + "' is missing");
        }
        return toLongValue(cfg.source() + ": key '" + path + "." + key + "'", v);
    }

    /** {@code v} as an exact {@code long}; integral doubles convert, other types fail (HfConfig.toLong). */
    private static long toLongValue(String where, Object v) {
        if (v instanceof Long) {
            return (Long) v;
        }
        if (v instanceof Integer) {
            return (Integer) v;
        }
        if (v instanceof java.math.BigInteger && ((java.math.BigInteger) v).bitLength() < Long.SIZE) {
            return ((java.math.BigInteger) v).longValue();
        }
        if (v instanceof Double) {
            double d = (Double) v;
            if (d == Math.rint(d) && !Double.isInfinite(d) && d >= -0x1p63 && d < 0x1p63) {
                return (long) d;
            }
        }
        throw new GgufFormatException(where + " must be an integer, got " + v);
    }

    /**
     * {@code WavTokenizerDecModel.filter_tensors} (wavtokenizer.py:16-27): the codebook statistics
     * ({@code cluster_size}, {@code embed_avg} and {@code inited}) are dropped; the rest run through the parent
     * filter ({@code super().filter_tensors}).
     */
    public static HfNameFilter wavTokenizerDecFilter(HfLoadContext ctx) {
        return (name, base) -> {
            if (name.endsWith("codebook.cluster_size") || name.endsWith("codebook.embed_avg")
                    || name.endsWith("codebook.inited")) {
                return null;
            }
            return base.apply(name);
        };
    }

    /** {@code WavTokenizerDecModel.set_vocab} (wavtokenizer.py:29-30): no vocabulary ({@code _set_vocab_none}). */
    public static Map<String, Object> wavTokenizerDecVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = ctx.tokenizerFiles();
        if (files.hasGgufTokenizer()) {
            return HfVocabWriters.fromGguf(files);
        }
        return HfVocabWriters.none(HfHookSupport.vocabInput(ctx), pre);
    }
}
