package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.resolve.TensorClaims;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Takes the KV-cache scales out of a quantized checkpoint's tensor index and turns them into the GGUF KV-scale
 * tensors ({@code blk.N.attn_k.kv_scale}, {@code blk.N.attn_v.kv_scale}, {@code blk.N.attn_q.q_scale},
 * {@code blk.N.attn_prob_scale}; F32). It recognizes every name form vLLM remaps (weight_utils.py:1705-1814,
 * base_config.py:229-250, and the {@code output_scale} mappers of fp8.py and quark.py):
 * <ul>
 *   <li>{@code *.k_scale}, {@code *.v_scale} and {@code *.q_scale}, including {@code self_attn.{k,v}_proj.{k,v}_scale},
 *       {@code self_attn.qkv_proj.{k,v}_scale}, {@code self_attn.qkqkv_proj.{k,v}_scale} and
 *       {@code mixer.{k,v}_proj.{k,v}_scale};</li>
 *   <li>the deprecated {@code *.kv_scale}, which sets the key scale and the value scale the layer does not set
 *       itself;</li>
 *   <li>{@code *.self_attn.q.scale}, {@code *.self_attn.k_cache.scale} and {@code *.self_attn.v_cache.scale};</li>
 *   <li>{@code *.{k,v,q}_proj.output_scale} (compressed-tensors, Quark and FP8 KV-cache schemes);</li>
 *   <li>{@code *.prob_output_scale} and {@code *.prob_scale};</li>
 *   <li>{@code *.{k,v,q}_zero_point}, which must be zero (an asymmetric KV cache has no GGUF slot).</li>
 * </ul>
 * A scale holds one value or one per head ({@code [n_head_kv]} for keys and values, {@code [n_head]} for queries, the
 * compressed-tensors {@code attn_head} strategy, also stored as {@code [heads, 1, 1]}); every value must be finite and
 * positive.
 */
public final class KvScaleCollector {

    private static final Logger LOG = LogManager.getLogger(KvScaleCollector.class);

    /** The GGUF tensor a scale becomes. */
    public enum Kind {
        K(TensorNames.ATTN_K + "." + TensorClaims.KV_SCALE),
        V(TensorNames.ATTN_V + "." + TensorClaims.KV_SCALE),
        Q(TensorNames.ATTN_Q + "." + TensorClaims.Q_SCALE),
        PROB(TensorClaims.ATTN_PROB_SCALE);

        private final String suffix;

        Kind(String suffix) {
            this.suffix = suffix;
        }

        /** The GGUF name of this scale in layer {@code bid}. */
        public String ggufName(int bid) {
            return TensorNames.layerName(bid, suffix);
        }

        static Kind of(char c) {
            return c == 'k' ? K : c == 'v' ? V : Q;
        }
    }

    /** One collected scale. Immutable. */
    public static final class Scale {
        private final Kind kind;
        private final int layer;
        private final String hfName;
        private final RowSource source;

        Scale(Kind kind, int layer, String hfName, RowSource source) {
            this.kind = kind;
            this.layer = layer;
            this.hfName = hfName;
            this.source = source;
        }

        /** The GGUF tensor of the scale in block {@link #layer()}. */
        public String ggufName() { return kind.ggufName(layer); }

        /** Which GGUF scale tensor of a block this is. */
        public Kind kind() { return kind; }

        /** The block: the layer number of {@link #hfName()}, or the block {@link #atLayer} gave. */
        public int layer() { return layer; }

        /** The checkpoint tensor it comes from. */
        public String hfName() { return hfName; }

        /** The F32 values, {@code [1]} or one per head. */
        public RowSource source() { return source; }

        /**
         * The scale in GGUF block {@code bid}: the block of its tensor once the architecture's name filter has renamed
         * it (appended draft layers, for example, follow the main layers).
         */
        public Scale atLayer(int bid) {
            if (bid < 0) {
                throw new IllegalArgumentException("KV-cache scale " + hfName + ": block " + bid + " is negative");
            }
            return bid == layer ? this : new Scale(kind, bid, hfName, source);
        }

        @Override
        public String toString() {
            return ggufName() + " <- " + hfName;
        }
    }

    /** The index without the KV-cache tensors, and the scales. Immutable. */
    public static final class Result {
        private final HfTensorIndex index;
        private final List<Scale> scales;
        private final Set<String> outputScaleModules;

        Result(HfTensorIndex index, List<Scale> scales, Set<String> outputScaleModules) {
            this.index = index;
            this.scales = scales;
            this.outputScaleModules = outputScaleModules;
        }

        /** The index without every recognized KV-cache tensor. */
        public HfTensorIndex index() { return index; }

        /** The scales, in checkpoint order. Unmodifiable. */
        public List<Scale> scales() { return scales; }

        /** The modules whose {@code output_scale} was a KV-cache scale. Unmodifiable. */
        public Set<String> outputScaleModules() { return outputScaleModules; }
    }

    private static final Pattern QKV_SCALE = Pattern.compile("\\.([qkv])_scale$");
    private static final Pattern QKV_ZERO = Pattern.compile("\\.([qkv])_zero_point$");
    private static final Pattern Q_DOT_SCALE = Pattern.compile("\\.self_attn\\.q\\.scale$");
    private static final Pattern CACHE_SCALE = Pattern.compile("\\.self_attn\\.([kv])_cache\\.scale$");
    private static final Pattern PROJ_OUTPUT = Pattern.compile("\\.([qkv])_proj\\.output_scale$");
    private static final Pattern PROB = Pattern.compile("\\.prob(?:_output)?_scale$");
    private static final String KV_SCALE = ".kv_scale";
    private static final String OUTPUT_SCALE = ".output_scale";

    private KvScaleCollector() {
    }

    /**
     * Collects the KV-cache scales of {@code in}. Scales are told apart by their kind and their layer: the tensor
     * name up to its block number, so that {@code model.layers.0} and an appended {@code mtp.layers.0} are different
     * layers.
     *
     * @throws GgufFormatException when a scale has no layer number, the wrong element count, a non-finite or
     *                             non-positive value, two tensors give one scale, or a zero point is not zero
     * @throws IOException         when a scale cannot be read
     */
    public static Result collect(HfConfig cfg, HfTensorIndex in) throws IOException {
        Objects.requireNonNull(cfg, "cfg");
        Objects.requireNonNull(in, "in");
        OptionalInt nHead = cfg.findInt("num_attention_heads", "n_head", "n_heads");
        OptionalInt nKv = cfg.findInt("num_key_value_heads", "n_head_kv", "num_kv_heads");
        int heads = nHead.orElse(-1);
        int kvHeads = nKv.isPresent() ? nKv.getAsInt() : heads;
        Map<String, Scale> scales = new LinkedHashMap<>();
        List<String> fused = new ArrayList<>();
        Set<String> remove = new LinkedHashSet<>();
        Set<String> modules = new LinkedHashSet<>();
        for (String name : in.names()) {
            if (name.endsWith(KV_SCALE)) {
                fused.add(name);
                remove.add(name);
                continue;
            }
            Matcher z = QKV_ZERO.matcher(name);
            if (z.find()) {
                requireZero(name, in.get(name));
                remove.add(name);
                continue;
            }
            Kind kind = kindOf(name);
            if (kind == null) {
                continue;
            }
            if (PROJ_OUTPUT.matcher(name).find()) {
                modules.add(name.substring(0, name.length() - OUTPUT_SCALE.length()));
            }
            put(scales, scale(name, kind, in.get(name), kind == Kind.PROB ? 1 : kind == Kind.Q ? heads : kvHeads));
            remove.add(name);
        }
        for (String name : fused) {
            Scale k = scale(name, Kind.K, in.get(name), kvHeads);
            put(scales, k);
            String v = key(Kind.V, name);
            if (!scales.containsKey(v)) {
                scales.put(v, new Scale(Kind.V, k.layer(), name, k.source()));
            }
        }
        if (!scales.isEmpty()) {
            LOG.info("KV-cache scales: {} tensor(s) from {} checkpoint tensor(s)", scales.size(), remove.size());
        }
        HfTensorIndex out = remove.isEmpty() ? in : in.withReplaced(Collections.emptyMap(), remove);
        return new Result(out, Collections.unmodifiableList(new ArrayList<>(scales.values())),
                Collections.unmodifiableSet(modules));
    }

    /** The scale {@code name} gives, or null when it is not a KV-cache scale. */
    static Kind kindOf(String name) {
        Matcher m = PROJ_OUTPUT.matcher(name);
        if (m.find()) {
            return Kind.of(m.group(1).charAt(0));
        }
        if (PROB.matcher(name).find()) {
            return Kind.PROB;
        }
        if (Q_DOT_SCALE.matcher(name).find()) {
            return Kind.Q;
        }
        m = CACHE_SCALE.matcher(name);
        if (m.find()) {
            return Kind.of(m.group(1).charAt(0));
        }
        m = QKV_SCALE.matcher(name);
        if (m.find()) {
            return Kind.of(m.group(1).charAt(0));
        }
        return null;
    }

    private static void put(Map<String, Scale> scales, Scale s) {
        Scale prev = scales.putIfAbsent(key(s.kind(), s.hfName()), s);
        if (prev != null) {
            throw new GgufFormatException("KV-cache scale " + s.ggufName() + " is given by both " + prev.hfName()
                    + " and " + s.hfName());
        }
    }

    /** The identity of the {@code kind} scale of the layer of {@code name}: the name through its block number. */
    private static String key(Kind kind, String name) {
        int end = 0;
        for (String part : name.split("\\.", -1)) {
            end += part.length();
            if (!part.isEmpty() && part.chars().allMatch(c -> c >= '0' && c <= '9')) {
                break;
            }
            end++;
        }
        return kind + "@" + name.substring(0, Math.min(end, name.length()));
    }

    private static Scale scale(String name, Kind kind, RowSource src, int perHead) throws IOException {
        int bid = HfLoadContext.bidOf(name);
        if (bid < 0) {
            throw new GgufFormatException("KV-cache scale " + name + " has no layer number");
        }
        float[] v = values(name, src);
        if (v.length != 1 && (perHead <= 0 || v.length != perHead)) {
            throw new GgufFormatException("KV-cache scale " + name + " has " + v.length + " values; expected 1"
                    + (kind == Kind.PROB || perHead <= 0 ? "" : " or " + perHead + " (one per "
                    + (kind == Kind.Q ? "query" : "key/value") + " head)"));
        }
        for (int i = 0; i < v.length; i++) {
            if (!Float.isFinite(v[i]) || v[i] <= 0f) {
                throw new GgufFormatException("KV-cache scale " + name + " holds " + v[i] + " at " + i
                        + "; scales must be finite and positive");
            }
        }
        return new Scale(kind, bid, name, RopeTensors.fromFloats(new long[]{v.length}, v));
    }

    private static void requireZero(String name, RowSource src) throws IOException {
        float[] v = values(name, src);
        for (int i = 0; i < v.length; i++) {
            if (v[i] != 0f) {
                throw new GgufFormatException("KV-cache zero point " + name + " holds " + v[i] + " at " + i
                        + "; an asymmetric KV cache is not supported");
            }
        }
    }

    /** The values of a small tensor as F32. */
    static float[] values(String name, RowSource src) throws IOException {
        long n = 1;
        for (long d : src.shape()) {
            n *= d;
            if (n > HfConfig.MAX_HEADS) {
                throw new GgufFormatException("KV-cache tensor " + name + " of shape " + Arrays.toString(src.shape())
                        + " has more than " + HfConfig.MAX_HEADS + " values");
            }
        }
        RowSource f;
        try {
            f = ElementTransforms.convert(src, GGMLType.F32);
        } catch (IllegalArgumentException e) {
            throw new GgufFormatException("KV-cache tensor " + name + " is stored as " + src.format()
                    + ", which has no float value", e);
        }
        ByteBuffer b = ByteBuffer.allocate((int) (f.rows() * f.rowBytes())).order(ByteOrder.LITTLE_ENDIAN);
        f.readRows(0, (int) f.rows(), b);
        b.flip();
        float[] out = new float[(int) n];
        for (int i = 0; i < out.length; i++) {
            out[i] = b.getFloat();
        }
        return out;
    }
}
