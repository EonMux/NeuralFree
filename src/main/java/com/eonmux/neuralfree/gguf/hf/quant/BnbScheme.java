package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfQuantConfig;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.quant.Fp4Packing.Geometry;
import com.eonmux.neuralfree.gguf.hf.quant.Fp4Packing.Module;
import com.eonmux.neuralfree.gguf.hf.quant.Fp4Packing.Notes;
import com.eonmux.neuralfree.gguf.hf.quant.Fp4Packing.Produced;
import com.eonmux.neuralfree.gguf.hf.quant.Fp4Packing.Rewrite;
import com.eonmux.neuralfree.gguf.hf.quant.Fp4Packing.RowWriter;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.QuantTables;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Rewrites bitsandbytes modules into their GGUF targets.
 * <ul>
 *   <li>4-bit NF4/FP4: the weight is U8 {@code [ceil(N*K/2), 1]}, the row-major flattened codes with the even element
 *       in the high nibble, with {@code weight.absmax} per {@code blocksize} elements (U8 codes of a second-level
 *       quantization under double quant, folded in f32 exactly as bitsandbytes), {@code weight.quant_map} and the
 *       JSON {@code weight.quant_state.bitsandbytes__<type>}. Target NF4/FP4_BNB (absmax replicated per 64-run) when
 *       {@code blocksize % 64 == 0} and {@code K % 64 == 0} and the stored map is within 1e-4 of the codebook;
 *       otherwise the float fallback decoded through the stored map.</li>
 *   <li>int8: I8 {@code [N, K]} with {@code SCB [N]}: I8_B128 with {@code d = SCB[n] / 127} when
 *       {@code K % 128 == 0}; Q8_0 with an fp16 {@code d} (or F32 under {@code exactQuant}) when {@code K % 32 == 0};
 *       otherwise the float fallback.</li>
 *   <li>A float weight without its quantization state was saved before bitsandbytes quantized it at load: it loads
 *       as stored.</li>
 * </ul>
 */
public final class BnbScheme {

    private static final Logger LOG = LogManager.getLogger(BnbScheme.class);

    /** Largest {@code quant_state} JSON tensor, in bytes. */
    static final int MAX_QUANT_STATE_BYTES = 64 * 1024;

    private static final Set<String> STATE_KEYS = Set.of("quant_type", "blocksize", "dtype", "shape",
            "nested_blocksize", "nested_dtype", "nested_offset");
    private static final Set<String> FLOAT_DTYPES = Set.of("float32", "float16", "bfloat16");
    private static final String STATE = "weight.quant_state.bitsandbytes__";
    private static final float MAP_TOLERANCE = 1e-4f;
    private static final int RUN = 64;
    private static final int B128 = 128;
    private static final int Q8 = 32;

    private BnbScheme() {
    }

    /** True for {@link QuantFormat#NF4_BNB}, {@link QuantFormat#FP4_BNB} and {@link QuantFormat#INT8_BNB}. */
    public static boolean handles(QuantFormat f) {
        return f == QuantFormat.NF4_BNB || f == QuantFormat.FP4_BNB || f == QuantFormat.INT8_BNB;
    }

    /**
     * Rewrites every module of {@code modules} (HF module prefixes) in {@code in}.
     *
     * @throws GgufFormatException when the configuration stores 4-bit codes in another dtype than uint8, or a
     *                             module's tensors or quantization state are missing or inconsistent
     * @throws IOException         when a tensor cannot be read
     */
    public static HfTensorIndex rewriteModules(QuantArgs args, List<String> modules, HfConfig cfg, HfTensorIndex in,
                                               ModelSourceOptions o) throws IOException {
        Objects.requireNonNull(args, "args");
        Objects.requireNonNull(modules, "modules");
        Objects.requireNonNull(cfg, "cfg");
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(o, "o");
        if (!handles(args.format())) {
            throw new IllegalArgumentException("BnbScheme does not handle " + args.format());
        }
        boolean fourBit = args.format() != QuantFormat.INT8_BNB;
        if (fourBit && !modules.isEmpty()) {
            Object storage = HfQuantConfig.of(cfg).view().raw("bnb_4bit_quant_storage");
            if (storage != null && !"uint8".equals(storage)) {
                throw new GgufFormatException("unsupported quant_method 'bitsandbytes': bnb_4bit_quant_storage '"
                        + storage + "' is not supported; only uint8 is");
            }
        }
        Rewrite rw = new Rewrite();
        Notes notes = new Notes();
        for (String module : modules) {
            Module m = new Module(in, module);
            if (fourBit) {
                fourBit(args, m, in, o, rw, notes);
            } else {
                int8(args, m, in, o, rw, notes);
            }
        }
        notes.log(LOG);
        return rw.apply(in);
    }

    private static boolean isFloat(StDType t) {
        return t == StDType.F32 || t == StDType.F16 || t == StDType.BF16;
    }

    /** The weight of a module, or null when it is a float weight saved before quantization (it loads as stored). */
    private static RowSource weight(Module m, Notes notes, String stateSuffix) {
        RowSource w = m.get("weight");
        if (w == null) {
            throw new GgufFormatException("bitsandbytes module " + m.prefix() + " has no weight tensor");
        }
        if (isFloat(Fp4Packing.dtype(w, m.name("weight"))) && !m.has(stateSuffix)) {
            notes.info("bitsandbytes modules saved before quantization load as stored; --quant-online quantizes"
                    + " them", m.prefix());
            return null;
        }
        return w;
    }

    // ------------------------------------------------------------------ 4-bit

    /** The parsed {@code quant_state} of one 4-bit weight. Immutable. */
    static final class QuantState {
        final String quantType;
        final int blocksize;
        final String dtype;
        final long n;
        final long k;
        final boolean nested;
        final int nestedBlocksize;
        final float nestedOffset;

        private QuantState(String quantType, int blocksize, String dtype, long n, long k, boolean nested,
                           int nestedBlocksize, float nestedOffset) {
            this.quantType = quantType;
            this.blocksize = blocksize;
            this.dtype = dtype;
            this.n = n;
            this.k = k;
            this.nested = nested;
            this.nestedBlocksize = nestedBlocksize;
            this.nestedOffset = nestedOffset;
        }
    }

    /**
     * Parses a {@code quant_state.bitsandbytes__<type>} tensor: U8 UTF-8 JSON of at most
     * {@link #MAX_QUANT_STATE_BYTES}, with only the bitsandbytes keys.
     */
    static QuantState parseState(RowSource t, String name) throws IOException {
        if (Fp4Packing.dtype(t, name) != StDType.U8) {
            throw new GgufFormatException("bitsandbytes quant state " + name + " must be U8");
        }
        long bytes = Fp4Packing.numel(t.shape());
        if (bytes > MAX_QUANT_STATE_BYTES) {
            throw new GgufFormatException("bitsandbytes quant state " + name + " has " + bytes + " bytes; at most "
                    + MAX_QUANT_STATE_BYTES);
        }
        String text;
        try {
            text = StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(ByteBuffer.wrap(Fp4Packing.readAll(t, name))).toString();
        } catch (CharacterCodingException e) {
            throw new GgufFormatException("bitsandbytes quant state " + name + " is not UTF-8", e);
        }
        Object parsed;
        try {
            parsed = Json.parse(text, Json.Limits.STRICT);
        } catch (JsonException e) {
            throw new GgufFormatException("bitsandbytes quant state " + name + ": " + e.getMessage(), e);
        }
        if (!(parsed instanceof Map)) {
            throw new GgufFormatException("bitsandbytes quant state " + name + " is not a JSON object");
        }
        Map<?, ?> s = (Map<?, ?>) parsed;
        for (Object key : s.keySet()) {
            if (!STATE_KEYS.contains(key)) {
                throw new GgufFormatException("bitsandbytes quant state " + name + " has the unknown key '" + key
                        + "'");
            }
        }
        String type = string(s, "quant_type", name);
        int blocksize = (int) integer(s, "blocksize", name, 1, 1 << 20);
        String dtype = string(s, "dtype", name);
        if (!FLOAT_DTYPES.contains(dtype)) {
            throw new GgufFormatException("bitsandbytes quant state " + name + ": dtype '" + dtype + "' is not one of "
                    + FLOAT_DTYPES);
        }
        Object shape = s.get("shape");
        if (!(shape instanceof List) || ((List<?>) shape).size() != 2) {
            throw new GgufFormatException("bitsandbytes quant state " + name + ": shape " + shape
                    + " is not [N, K]");
        }
        long n = asLong(((List<?>) shape).get(0), name + " shape[0]", 1, Integer.MAX_VALUE);
        long k = asLong(((List<?>) shape).get(1), name + " shape[1]", 1, Integer.MAX_VALUE / 8);
        boolean nested = s.containsKey("nested_blocksize") || s.containsKey("nested_dtype")
                || s.containsKey("nested_offset");
        int nestedBlocksize = 0;
        float offset = 0f;
        if (nested) {
            nestedBlocksize = (int) integer(s, "nested_blocksize", name, 1, 1 << 20);
            String nestedDtype = string(s, "nested_dtype", name);
            if (!nestedDtype.equals("float32")) {
                throw new GgufFormatException("bitsandbytes quant state " + name + ": nested_dtype '" + nestedDtype
                        + "' is not float32");
            }
            Object off = s.get("nested_offset");
            if (!(off instanceof Number) || !Double.isFinite(((Number) off).doubleValue())) {
                throw new GgufFormatException("bitsandbytes quant state " + name + ": nested_offset " + off
                        + " is not a finite number");
            }
            offset = ((Number) off).floatValue();
        }
        return new QuantState(type, blocksize, dtype, n, k, nested, nestedBlocksize, offset);
    }

    private static String string(Map<?, ?> s, String key, String name) {
        Object v = s.get(key);
        if (!(v instanceof String)) {
            throw new GgufFormatException("bitsandbytes quant state " + name + ": " + key + " is " + v
                    + "; expected a string");
        }
        return (String) v;
    }

    private static long integer(Map<?, ?> s, String key, String name, long min, long max) {
        return asLong(s.get(key), name + " " + key, min, max);
    }

    private static long asLong(Object v, String where, long min, long max) {
        if (!(v instanceof Long || v instanceof Integer) || ((Number) v).longValue() < min
                || ((Number) v).longValue() > max) {
            throw new GgufFormatException("bitsandbytes quant state " + where + " is " + v + "; expected an integer"
                    + " in [" + min + ", " + max + "]");
        }
        return ((Number) v).longValue();
    }

    private static boolean powerOfTwo(long v) {
        return v > 0 && (v & (v - 1)) == 0;
    }

    private static void fourBit(QuantArgs args, Module m, HfTensorIndex in, ModelSourceOptions o, Rewrite rw,
                                Notes notes) throws IOException {
        boolean fp4 = args.format() == QuantFormat.FP4_BNB;
        String type = fp4 ? "fp4" : "nf4";
        String stateSuffix = STATE + type;
        RowSource w = weight(m, notes, stateSuffix);
        if (w == null) {
            return;
        }
        String wName = m.name("weight");
        if (Fp4Packing.dtype(w, wName) != StDType.U8) {
            throw new GgufFormatException("bitsandbytes 4-bit weight " + wName + " has dtype "
                    + Fp4Packing.dtype(w, wName).headerName() + "; only uint8 storage is supported");
        }
        String stName = m.name(stateSuffix);
        if (stName == null) {
            throw new GgufFormatException("bitsandbytes module " + m.prefix() + " has no " + m.prefix() + "."
                    + stateSuffix);
        }
        QuantState qs = parseState(in.get(stName), stName);
        if (!qs.quantType.equals(type)) {
            throw new GgufFormatException("bitsandbytes quant state " + stName + " has quant_type '" + qs.quantType
                    + "'; expected '" + type + "'");
        }
        if (!powerOfTwo(qs.blocksize) || qs.blocksize < 32 || qs.blocksize > 4096) {
            throw new GgufFormatException("bitsandbytes quant state " + stName + ": blocksize " + qs.blocksize
                    + " is not one of 32, 64, ..., 4096");
        }
        long elements = Math.multiplyExact(qs.n, qs.k);
        long[] ws = w.shape();
        if (ws.length != 2 || ws[1] != 1 || ws[0] != (elements + 1) / 2) {
            throw new GgufFormatException("bitsandbytes 4-bit weight " + wName + " has shape " + Arrays.toString(ws)
                    + "; expected [" + (elements + 1) / 2 + ", 1] for [" + qs.n + ", " + qs.k + "]");
        }
        long blocks = (elements + qs.blocksize - 1) / qs.blocksize;
        Absmax absmax = absmax(m, in, qs, blocks);
        String mapName = m.name("weight.quant_map");
        float[] map = quantMap(in, m, mapName);
        float maxDiff = 0f;
        boolean bitEqual = true;
        for (int i = 0; i < 16; i++) {
            float ref = fp4 ? QuantTables.fp4BnbCode(i) : QuantTables.nf4Code(i);
            bitEqual &= Float.floatToIntBits(ref) == Float.floatToIntBits(map[i]);
            maxDiff = Math.max(maxDiff, Math.abs(ref - map[i]));
        }
        Geometry g = Geometry.of(new long[]{qs.n, qs.k}, wName);
        GGMLType target = fp4 ? GGMLType.FP4_BNB : GGMLType.NF4;
        StDType fb = Fp4Packing.fallbackType(floatDtype(qs.dtype), o);
        Produced src;
        if (!(maxDiff <= MAP_TOLERANCE)) {
            notes.fallback("bitsandbytes 4-bit weights whose quant_map differs from the " + type + " codebook by more"
                    + " than 1e-4 load as the float fallback", m.prefix());
            src = floats(args, m, w, g, qs, absmax, map, fb);
        } else if (!bitEqual && o.exactQuant()) {
            src = floats(args, m, w, g, qs, absmax, map, StDType.F32);
        } else if (qs.blocksize % RUN == 0 && qs.k % RUN == 0) {
            src = Produced.of(Fp4Packing.describe(m.prefix(), args.format(), args.bits(), qs.blocksize, target),
                    ElementFormat.of(target), g.shape, false, bnb4Writer(w, g, qs, absmax));
        } else {
            notes.fallback("bitsandbytes 4-bit weights whose 64-element runs cross rows or blocks (K % 64 != 0 or"
                    + " blocksize 32) load as the float fallback", m.prefix());
            src = floats(args, m, w, g, qs, absmax, map, fb);
        }
        rw.put(m.outputName("weight"), src);
        rw.consume(stName);
        rw.consume(mapName);
        rw.consume(m, "weight.absmax", "weight.nested_absmax", "weight.nested_quant_map");
    }

    /** The safetensors dtype of a validated {@code quant_state} torch dtype name. */
    private static StDType floatDtype(String torchDtype) {
        switch (torchDtype) {
            case "float32": return StDType.F32;
            case "float16": return StDType.F16;
            default: return StDType.BF16;
        }
    }

    private static float[] quantMap(HfTensorIndex in, Module m, String mapName) throws IOException {
        if (mapName == null) {
            throw new GgufFormatException("bitsandbytes module " + m.prefix() + " has no " + m.prefix()
                    + ".weight.quant_map");
        }
        RowSource t = in.get(mapName);
        if (Fp4Packing.dtype(t, mapName) != StDType.F32 || Fp4Packing.numel(t.shape()) != 16) {
            throw new GgufFormatException("bitsandbytes quant_map " + mapName + " must be F32 [16]");
        }
        return Fp4Packing.readScales(t, mapName, false);
    }

    /** The absmax source of one weight: F32 values, or U8 codes with their second-level state. Immutable. */
    private static final class Absmax {
        final RowSource values;
        final String name;
        final RowSource nestedAbsmax;
        final String nestedName;
        final float[] nestedMap;

        Absmax(RowSource values, String name, RowSource nestedAbsmax, String nestedName, float[] nestedMap) {
            this.values = values;
            this.name = name;
            this.nestedAbsmax = nestedAbsmax;
            this.nestedName = nestedName;
            this.nestedMap = nestedMap;
        }
    }

    private static Absmax absmax(Module m, HfTensorIndex in, QuantState qs, long blocks) throws IOException {
        String aName = m.name("weight.absmax");
        if (aName == null) {
            throw new GgufFormatException("bitsandbytes module " + m.prefix() + " has no " + m.prefix()
                    + ".weight.absmax");
        }
        RowSource a = in.get(aName);
        StDType at = Fp4Packing.dtype(a, aName);
        if (Fp4Packing.numel(a.shape()) != blocks) {
            throw new GgufFormatException("bitsandbytes absmax " + aName + " has " + Fp4Packing.numel(a.shape())
                    + " values; expected " + blocks + " for " + qs.n * qs.k + " elements in blocks of "
                    + qs.blocksize);
        }
        if (!qs.nested) {
            if (at != StDType.F32) {
                throw new GgufFormatException("bitsandbytes absmax " + aName + " has dtype " + at.headerName()
                        + "; expected F32 without double quantization");
            }
            return new Absmax(a, aName, null, null, null);
        }
        String nName = m.name("weight.nested_absmax");
        String nmName = m.name("weight.nested_quant_map");
        if (at != StDType.U8 || nName == null || nmName == null) {
            throw new GgufFormatException("bitsandbytes double quantization of " + m.prefix() + " needs U8 "
                    + aName + ", weight.nested_absmax and weight.nested_quant_map");
        }
        RowSource nested = in.get(nName);
        long nestedBlocks = (blocks + qs.nestedBlocksize - 1) / qs.nestedBlocksize;
        if (Fp4Packing.dtype(nested, nName) != StDType.F32 || Fp4Packing.numel(nested.shape()) != nestedBlocks) {
            throw new GgufFormatException("bitsandbytes nested absmax " + nName + " must be F32 with " + nestedBlocks
                    + " values for " + blocks + " blocks of " + qs.nestedBlocksize);
        }
        RowSource nm = in.get(nmName);
        if (Fp4Packing.dtype(nm, nmName) != StDType.F32 || Fp4Packing.numel(nm.shape()) != 256) {
            throw new GgufFormatException("bitsandbytes nested quant_map " + nmName + " must be F32 [256]");
        }
        return new Absmax(a, aName, nested, nName, Fp4Packing.readScales(nm, nmName, false));
    }

    /**
     * Every absmax of a weight in f32; under double quantization
     * {@code nested_quant_map[q[b]] * nested_absmax[b / nested_blocksize] + nested_offset}.
     */
    static float[] foldAbsmax(byte[] codes, float[] nestedMap, float[] nestedAbsmax, int nestedBlocksize,
                              float offset) {
        float[] out = new float[codes.length];
        for (int b = 0; b < codes.length; b++) {
            out[b] = nestedMap[codes[b] & 0xFF] * nestedAbsmax[b / nestedBlocksize] + offset;
        }
        return out;
    }

    private static float[] absmaxValues(Absmax a, QuantState qs) throws IOException {
        if (a.nestedAbsmax == null) {
            return Fp4Packing.readScales(a.values, a.name, false);
        }
        return foldAbsmax(Fp4Packing.readAll(a.values, a.name), a.nestedMap,
                Fp4Packing.readScales(a.nestedAbsmax, a.nestedName, false), qs.nestedBlocksize, qs.nestedOffset);
    }

    /** The packed bytes covering elements {@code [first * K, (first + count) * K)}, and the first byte's index. */
    private static byte[] packedRows(RowSource w, long k, long first, int count) throws IOException {
        long b0 = first * k / 2;
        long b1 = ((first + count) * k + 1) / 2;
        return Fp4Packing.readRows(w, b0, Math.toIntExact(b1 - b0));
    }

    private static RowWriter bnb4Writer(RowSource w, Geometry g, QuantState qs, Absmax a) {
        return (first, count, dst) -> {
            float[] absmax = absmaxValues(a, qs);
            byte[] codes = new byte[RUN];
            int step = Fp4Packing.chunkRows(g.k);
            for (int done = 0; done < count; done += step) {
                int c = Math.min(step, count - done);
                long f = first + done;
                byte[] p = packedRows(w, g.k, f, c);
                long byteBase = f * g.k / 2;
                for (int r = 0; r < c; r++) {
                    for (int col = 0; col < g.k; col += RUN) {
                        long e0 = (f + r) * g.k + col;
                        int off = (int) (e0 / 2 - byteBase);
                        for (int i = 0; i < RUN; i++) {
                            codes[i] = (byte) Fp4Packing.highFirst(p, off, i);
                        }
                        Fp4Packing.putBnb4(dst, absmax[(int) (e0 / qs.blocksize)], codes, 0);
                    }
                }
            }
        };
    }

    /**
     * The dequantized weight {@code map[code] * absmax} in {@code type}. The absmax vector is a single row, so each
     * read decodes it once and then works through the requested rows in chunks.
     */
    private static Produced floats(QuantArgs args, Module m, RowSource w, Geometry g, QuantState qs, Absmax a,
                                   float[] map, StDType type) {
        RowWriter writer = (first, count, dst) -> {
            float[] absmax = absmaxValues(a, qs);
            int step = Fp4Packing.chunkRows(g.k);
            float[] out = new float[Math.toIntExact((long) Math.min(step, count) * g.k)];
            for (int done = 0; done < count; done += step) {
                int c = Math.min(step, count - done);
                long f = first + done;
                byte[] p = packedRows(w, g.k, f, c);
                long byteBase = f * g.k / 2;
                for (int r = 0; r < c; r++) {
                    for (int col = 0; col < g.k; col++) {
                        long e = (f + r) * g.k + col;
                        int b = p[(int) (e / 2 - byteBase)];
                        int code = (e & 1) == 0 ? (b >>> 4) & 0x0F : b & 0x0F;
                        out[r * g.k + col] = map[code] * absmax[(int) (e / qs.blocksize)];
                    }
                }
                for (int i = 0, n = c * g.k; i < n; i++) {
                    Fp4Packing.putFloat(dst, type, out[i]);
                }
            }
        };
        return Produced.of(Fp4Packing.describe(m.prefix(), args.format(), args.bits(), qs.blocksize, type),
                ElementFormat.of(type), g.shape, false, writer);
    }

    // ------------------------------------------------------------------ int8

    private static void int8(QuantArgs args, Module m, HfTensorIndex in, ModelSourceOptions o, Rewrite rw,
                             Notes notes) throws IOException {
        RowSource w = weight(m, notes, "SCB");
        if (w == null) {
            return;
        }
        String wName = m.name("weight");
        if (Fp4Packing.dtype(w, wName) != StDType.I8) {
            throw new GgufFormatException("bitsandbytes int8 weight " + wName + " has dtype "
                    + Fp4Packing.dtype(w, wName).headerName() + "; expected I8");
        }
        Geometry g = Geometry.of(w.shape(), wName);
        if (g.experts != 0) {
            throw new GgufFormatException("bitsandbytes int8 weight " + wName + " must be [N, K]");
        }
        String sName = m.name("SCB");
        if (sName == null) {
            throw new GgufFormatException("bitsandbytes int8 module " + m.prefix() + " has no " + m.prefix() + ".SCB");
        }
        RowSource scb = in.get(sName);
        if (Fp4Packing.dtype(scb, sName) != StDType.F32 || Fp4Packing.numel(scb.shape()) != g.n) {
            throw new GgufFormatException("bitsandbytes SCB " + sName + " must be F32 [" + g.n + "]");
        }
        String fName = m.name("weight_format");
        if (fName != null) {
            byte[] fmt = Fp4Packing.readAll(in.get(fName), fName);
            if (fmt.length != 1 || fmt[0] != 0) {
                throw new GgufFormatException("bitsandbytes weight_format " + fName + " is not the row-major format 0");
            }
            rw.consume(fName);
        }
        float[] scales = Fp4Packing.readScales(scb, sName, false);
        float[] d = new float[scales.length];
        for (int i = 0; i < d.length; i++) {
            d[i] = scales[i] / 127f;
        }
        Produced src;
        if (g.k % B128 == 0) {
            src = Produced.of(Fp4Packing.describe(m.prefix(), args.format(), 8, QuantArgs.CHANNEL,
                    GGMLType.I8_B128), ElementFormat.of(GGMLType.I8_B128), g.shape, w.rowLocal(),
                    int8Writer(w, g, d, B128, false));
        } else if (g.k % Q8 == 0 && !o.exactQuant()) {
            notes.fallback("bitsandbytes int8 weights with K % 128 != 0 load as Q8_0 with fp16 scales",
                    m.prefix());
            src = Produced.of(Fp4Packing.describe(m.prefix(), args.format(), 8, QuantArgs.CHANNEL, GGMLType.Q8_0),
                    ElementFormat.of(GGMLType.Q8_0), g.shape, w.rowLocal(), int8Writer(w, g, d, Q8, true));
        } else {
            notes.fallback("bitsandbytes int8 weights with K % 128 != 0 load as F32", m.prefix());
            src = Fp4Packing.floats(Fp4Packing.describe(m.prefix(), args.format(), 8, QuantArgs.CHANNEL,
                    StDType.F32), g.shape, StDType.F32, w.rowLocal(), (first, count, out) -> {
                        byte[] q = Fp4Packing.readRows(w, first, count);
                        for (int i = 0; i < count * g.k; i++) {
                            out[i] = q[i] * d[(int) (first + i / g.k)];
                        }
                    });
        }
        rw.put(m.outputName("weight"), src);
        rw.consume(sName);
    }

    /** I8_B128 ({@code f32 d}) or Q8_0 ({@code f16 d}) blocks of the stored codes, {@code d} per row. */
    private static RowWriter int8Writer(RowSource w, Geometry g, float[] d, int block, boolean half) {
        return (first, count, dst) -> {
            int step = Fp4Packing.chunkRows(g.k);
            for (int done = 0; done < count; done += step) {
                int c = Math.min(step, count - done);
                byte[] q = Fp4Packing.readRows(w, first + done, c);
                for (int r = 0; r < c; r++) {
                    float scale = d[(int) (first + done + r)];
                    for (int col = 0; col < g.k; col += block) {
                        if (half) {
                            Fp4Packing.putU16(dst, FpBlockDecoders.fp32ToFp16(scale));
                        } else {
                            Fp4Packing.putF32(dst, scale);
                        }
                        dst.put(q, r * g.k + col, block);
                    }
                }
            }
        };
    }
}
