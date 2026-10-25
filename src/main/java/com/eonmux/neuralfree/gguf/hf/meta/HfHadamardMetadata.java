package com.eonmux.neuralfree.gguf.hf.meta;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.names.HfTensorFilters;
import com.eonmux.neuralfree.gguf.hf.names.HfTensorNameMap;
import com.eonmux.neuralfree.gguf.hf.quant.HadamardTransforms;
import com.eonmux.neuralfree.gguf.hf.quant.HadamardTransforms.HadamardPlan;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.quant.PrismHadamardLoader;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The model-level metadata the converter writes after the architecture metadata: the PRISM Hadamard contract of a
 * checkpoint packed with {@code hadamard_packing.json} ({@code add_hadamard_metadata}, LC/conversion/base.py:691-840)
 * and the per-tensor {@code prec_a4} flags of ModelOpt mixed-precision checkpoints (base.py:665-675, 1391-1395).
 *
 * <p>The manifest is validated exactly as the converter validates it, in the same order, and additionally the block
 * size must not exceed {@link HadamardTransforms#MAX_BLOCK}, the limit of {@code PrismHadamardLoader}; GPU uploads
 * refuse blocks above {@code HadamardSpec.MAX_KERNEL_BLOCK} themselves. Every violation throws
 * {@link GgufFormatException} {@code "hadamard_packing.json: <reason>"}; a manifest on an architecture without a
 * verified runtime throws {@code "hadamard_packing.json present but <arch> has no verified PRISM runtime"}. The keys
 * are written in the converter's order. {@code prism.hadamard.gdn_v_grouped} is written by the Qwen linear-attention tensor hooks, which
 * decide it while planning.</p>
 *
 * <p>A checkpoint whose weights expect an online Hadamard rotation of their inputs (a compressed-tensors or fp_quant
 * input transform, {@link HadamardPlan}) gets the same keys without a manifest: version 1, the plan's block size and
 * sign mode, and the GGUF names its weights were planned under. Per-expert weights stacked into one tensor name it
 * once, and every expert of that tensor must carry the transform.</p>
 */
public final class HfHadamardMetadata {

    /** The manifest file in the model directory. */
    public static final String MANIFEST = "hadamard_packing.json";
    /** The largest manifest read. */
    public static final long MAX_MANIFEST_BYTES = 16L << 20;

    /** {@code _HADAMARD_ARCHS} (base.py:752-759), equal to the verified set of {@link PrismHadamardLoader}. */
    static final Set<Arch> HADAMARD_ARCHS = Collections.unmodifiableSet(
            EnumSet.of(Arch.LLAMA, Arch.QWEN3, Arch.QWEN3MOE, Arch.QWEN35, Arch.QWEN35MOE, Arch.QWEN3NEXT));

    private static final String KIND = "hadamard-weight-fold";
    private static final String STATUS = "requires-matching-runtime";
    private static final String MANIFEST_TRANSFORM = "normalized-signed-sylvester-walsh-hadamard";
    private static final String TRANSFORM = "normalized-sylvester-walsh-hadamard";
    private static final String AXIS = "input-last-dimension";
    private static final String SIGN_IDENTITY = "identity";
    private static final String SIGN_EXPLICIT = "explicit";
    private static final String ROLE_FOLD = "fold-before-matmul";
    private static final String ROLE_INVERSE = "inverse-after-lookup";
    private static final int TIED_SCHEMA = 3;
    private static final String FOLDED_NAMES_STATE = HfHadamardMetadata.class.getName() + ".foldedNames";
    /** One expert of a per-expert checkpoint tensor: the group prefix, the expert index, the part and the tail. */
    private static final Pattern EXPERT_MEMBER = Pattern.compile("^(.*\\.experts)\\.\\d{1,9}\\.([^.]+)(\\..*)?$");

    private HfHadamardMetadata() {
    }

    /**
     * Writes the PRISM Hadamard keys when the model directory holds {@link #MANIFEST}, then the {@code prec_a4} flags
     * of {@link HfLoadContext#precA4()}.
     *
     * @throws GgufFormatException when the manifest is invalid or the architecture has no verified runtime
     * @throws IOException         when the manifest cannot be read
     */
    public static void apply(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        apply(ctx, kv, plan, null);
    }

    /**
     * Writes the PRISM Hadamard keys of {@link #MANIFEST} when the model directory holds one, else those of
     * {@code online} when it is not null, then the {@code prec_a4} flags of {@link HfLoadContext#precA4()}.
     * {@code ctx.index()} is the filtered index {@code plan} was built from.
     *
     * @param online the online input transform of the checkpoint's weights, in checkpoint names, or null
     * @throws GgufFormatException when the manifest is invalid, both a manifest and an online transform are present,
     *                             the architecture has no verified runtime, or a transformed weight is not planned on a
     *                             verified Hadamard-aware matmul path as a whole tensor
     * @throws IOException         when the manifest cannot be read
     */
    public static void apply(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan, HadamardPlan online)
            throws IOException {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(plan, "plan");
        Map<String, Object> manifest = readManifest(ctx);
        if (manifest != null && online != null) {
            throw fail("the checkpoint also has an online Hadamard input transform (" + online
                    + "); PRISM applies one Hadamard contract per model");
        }
        if (manifest != null) {
            writeHadamard(ctx, kv, plan, manifest);
        } else if (online != null) {
            writeOnline(ctx, kv, plan, online);
        }
        writePrecA4(ctx.precA4(), kv);
    }

    /**
     * The checkpoint tensor names the manifest folds, without validation ({@code hadamard_folded_names},
     * base.py:677-689); empty when the model directory has no manifest. Read once per load and kept in the load
     * state, as the converter caches it. The Qwen linear-attention hooks use it to decide the folded value order.
     *
     * @throws GgufFormatException when the manifest is not valid JSON
     * @throws IOException         when the manifest cannot be read
     */
    public static Set<String> foldedNames(HfLoadContext ctx) throws IOException {
        FoldedNames cache = ctx.state(FOLDED_NAMES_STATE, FoldedNames.class, FoldedNames::new);
        synchronized (cache) {
            if (cache.names == null) {
                cache.names = readFoldedNames(ctx);
            }
            return cache.names;
        }
    }

    private static Set<String> readFoldedNames(HfLoadContext ctx) throws IOException {
        Map<String, Object> manifest = readManifest(ctx);
        if (manifest == null) {
            return Collections.emptySet();
        }
        Set<String> out = new LinkedHashSet<>();
        Object tensors = manifest.get("tensors");
        if (tensors instanceof List) {
            for (Object record : (List<?>) tensors) {
                if (record instanceof Map && ((Map<?, ?>) record).get("name") instanceof String) {
                    out.add((String) ((Map<?, ?>) record).get("name"));
                }
            }
        }
        return Collections.unmodifiableSet(out);
    }

    /** The per-load cache of {@link #foldedNames}. */
    private static final class FoldedNames {
        Set<String> names;
    }

    /** {@code add_tensor_extra_prec_a4} with the names sorted (base.py:1391-1395, gguf_writer.py:524-528). */
    static void writePrecA4(Map<String, Boolean> precA4, HfMetadataBuilder kv) {
        if (precA4.isEmpty()) {
            return;
        }
        TreeMap<String, Boolean> sorted = new TreeMap<>(precA4);
        kv.set(GgufKeys.GENERAL_TENSOR_EXTRA_NAME, new ArrayList<>(sorted.keySet()));
        kv.set(GgufKeys.GENERAL_TENSOR_EXTRA_PREC_A4, new ArrayList<>(sorted.values()));
    }

    // ------------------------------------------------------------------ manifest

    private static Map<String, Object> readManifest(HfLoadContext ctx) throws IOException {
        Path file = ctx.config().sideFile(MANIFEST);
        if (!Files.isRegularFile(file)) {
            return null;
        }
        Object parsed;
        try {
            parsed = Json.parse(file, MAX_MANIFEST_BYTES, Json.Limits.LENIENT);
        } catch (JsonException e) {
            throw fail(e.getMessage());
        }
        if (!(parsed instanceof Map)) {
            throw fail("the manifest is not a JSON object");
        }
        return asMap(parsed);
    }

    private static void writeHadamard(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan,
                                      Map<String, Object> manifest) {
        long schema = schemaVersion(manifest);
        Map<String, Object> transform = transform(manifest);
        int blockSize = blockSize(transform.get("block_size"));
        if (!MANIFEST_TRANSFORM.equals(transform.get("name"))) {
            throw fail("unsupported Hadamard transform: " + transform.get("name"));
        }
        String signMode = signMode(transform);
        List<Long> signWidths = new ArrayList<>();
        List<Long> signValues = new ArrayList<>();
        if (SIGN_EXPLICIT.equals(signMode)) {
            readSigns(manifest.get("signs"), blockSize, signWidths, signValues);
        }
        Object records = manifest.get("tensors");
        if (!(records instanceof List) || ((List<?>) records).isEmpty()) {
            throw fail("the manifest has no folded tensors");
        }
        if (!HADAMARD_ARCHS.contains(ctx.arch())) {
            throw new GgufFormatException(MANIFEST + " present but " + ctx.arch().ggufName()
                    + " has no verified PRISM runtime");
        }
        List<String> weightNames = new ArrayList<>();
        List<String> inverseNames = new ArrayList<>();
        mapRecords(ctx, (List<?>) records, weightNames, inverseNames);
        boolean tiedOutput = tiedOutput(ctx, plan, manifest, schema, weightNames, inverseNames);
        if (tiedOutput) {
            kv.set(GgufKeys.PRISM_HADAMARD_TIED_OUTPUT, true);
        }
        kv.set(GgufKeys.PRISM_HADAMARD_VERSION, tiedOutput ? 2 : 1);
        kv.set(GgufKeys.PRISM_HADAMARD_BLOCK_SIZE, blockSize);
        kv.set(GgufKeys.PRISM_HADAMARD_TRANSFORM, TRANSFORM);
        kv.set(GgufKeys.PRISM_HADAMARD_AXIS, AXIS);
        kv.set(GgufKeys.PRISM_HADAMARD_SIGN_MODE, signMode);
        kv.set(GgufKeys.PRISM_HADAMARD_WEIGHT_NAMES, weightNames);
        if (SIGN_EXPLICIT.equals(signMode)) {
            kv.set(GgufKeys.PRISM_HADAMARD_SIGN_WIDTHS, signWidths);
            kv.set(GgufKeys.PRISM_HADAMARD_SIGN_VALUES, signValues);
        }
        if (!inverseNames.isEmpty()) {
            kv.set(GgufKeys.PRISM_HADAMARD_INVERSE_WEIGHT_NAMES, inverseNames);
        }
        ctx.log("Hadamard contract: H" + blockSize + ", sign_mode=" + signMode + ", " + weightNames.size()
                + " folded weight(s), " + inverseNames.size() + " inverse-lookup");
    }

    // ------------------------------------------------------------------ online input transforms

    private static void writeOnline(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan, HadamardPlan online) {
        if (!HADAMARD_ARCHS.contains(ctx.arch())) {
            throw new GgufFormatException("online Hadamard input transform present but " + ctx.arch().ggufName()
                    + " has no verified PRISM runtime");
        }
        List<String> weightNames = onlineWeightNames(ctx, plan, online);
        if (weightNames.isEmpty()) {
            ctx.log("Hadamard input transform: every transformed weight was dropped by the filters");
            return;
        }
        boolean explicit = SIGN_EXPLICIT.equals(online.signMode());
        for (String g : weightNames) {
            long width = plan.ne(g)[0];
            if (width % online.blockSize() != 0 || (explicit && online.signs((int) width) == null)) {
                throw new GgufFormatException("Hadamard input transform: '" + g + "' has input width " + width
                        + ", which " + (explicit ? "has no sign vector or " : "") + "is not a multiple of block "
                        + online.blockSize());
            }
        }
        kv.set(GgufKeys.PRISM_HADAMARD_VERSION, 1);
        kv.set(GgufKeys.PRISM_HADAMARD_BLOCK_SIZE, online.blockSize());
        kv.set(GgufKeys.PRISM_HADAMARD_TRANSFORM, TRANSFORM);
        kv.set(GgufKeys.PRISM_HADAMARD_AXIS, AXIS);
        kv.set(GgufKeys.PRISM_HADAMARD_SIGN_MODE, online.signMode());
        kv.set(GgufKeys.PRISM_HADAMARD_WEIGHT_NAMES, weightNames);
        if (explicit) {
            kv.set(GgufKeys.PRISM_HADAMARD_SIGN_WIDTHS, online.signWidths());
            kv.set(GgufKeys.PRISM_HADAMARD_SIGN_VALUES, online.signValues());
        }
        ctx.log("Hadamard input transform: H" + online.blockSize() + ", sign_mode=" + online.signMode() + ", "
                + weightNames.size() + " weight(s)");
    }

    /**
     * The planned GGUF names of the transformed weights, in first-seen order. A weight the filters drop is skipped; a
     * kept one must be planned, on a verified matmul path, and every checkpoint tensor planned under the same name
     * must be transformed too.
     */
    private static List<String> onlineWeightNames(HfLoadContext ctx, HfPlanView plan, HadamardPlan online) {
        HfTensorNameMap names = ctx.names();
        HfTensorIndex index = ctx.index();
        Map<String, String> transformed = new LinkedHashMap<>();
        for (String hfName : online.weightNames()) {
            String filtered = HfTensorFilters.filterName(ctx, hfName);
            if (filtered == null || !index.has(filtered)) {
                continue;
            }
            String g = plannedName(names, filtered).orElseThrow(() -> new GgufFormatException(
                    "Hadamard input transform: can not map tensor '" + hfName + "'"));
            if (!plan.has(g)) {
                throw new GgufFormatException("Hadamard input transform: '" + hfName + "' maps to '" + g
                        + "', which is not planned");
            }
            if (!PrismHadamardLoader.isFoldableWeight(g)) {
                throw new GgufFormatException("Hadamard input transform: '" + hfName + "' maps to '" + g
                        + "', which is not on a verified Hadamard-aware matmul path");
            }
            transformed.put(filtered, g);
        }
        Set<String> out = new LinkedHashSet<>(transformed.values());
        for (String n : index.names()) {
            if (transformed.containsKey(n)) {
                continue;
            }
            Optional<String> g = plannedName(names, n);
            if (g.isPresent() && out.contains(g.get())) {
                throw new GgufFormatException("Hadamard input transform: '" + g.get() + "' is planned from '" + n
                        + "', which has no input transform, and from transformed tensors; PRISM rotates whole tensors");
            }
        }
        return new ArrayList<>(out);
    }

    /**
     * The GGUF name {@code name} is planned under: its own mapping, else for one expert of a per-expert tensor the
     * mapping of the merged name {@code <prefix>.experts.<part><tail>} the expert stacks are planned under.
     */
    private static Optional<String> plannedName(HfTensorNameMap names, String name) {
        Optional<HfTensorNameMap.Mapped> m = names.map(name, ".weight", ".bias");
        if (!m.isPresent()) {
            Matcher e = EXPERT_MEMBER.matcher(name);
            if (e.matches()) {
                m = names.map(e.group(1) + "." + e.group(2) + (e.group(3) == null ? "" : e.group(3)), ".weight",
                        ".bias");
            }
        }
        return m.map(HfTensorNameMap.Mapped::ggufName);
    }

    // ------------------------------------------------------------------ manifest validation

    /** {@code schema_version} in {1, 2, 3} and {@code kind} (base.py:702-704). */
    private static long schemaVersion(Map<String, Object> manifest) {
        Long schema = exactInteger(manifest.get("schema_version"));
        if (schema == null || schema < 1 || schema > TIED_SCHEMA || !KIND.equals(manifest.get("kind"))) {
            throw fail("unsupported Hadamard manifest (schema_version " + manifest.get("schema_version")
                    + ", kind " + manifest.get("kind") + ")");
        }
        return schema;
    }

    /** The {@code transform} object of a manifest with the expected {@code status} (base.py:705-710). */
    private static Map<String, Object> transform(Map<String, Object> manifest) {
        if (!STATUS.equals(manifest.get("status"))) {
            throw fail("unexpected Hadamard manifest status: " + manifest.get("status"));
        }
        Object transform = manifest.get("transform");
        if (!(transform instanceof Map)) {
            throw fail("the manifest is missing transform metadata");
        }
        return asMap(transform);
    }

    /** {@code identity} or {@code explicit} (base.py:716-718). */
    private static String signMode(Map<String, Object> transform) {
        Object mode = transform.get("sign_mode");
        if (!SIGN_IDENTITY.equals(mode) && !SIGN_EXPLICIT.equals(mode)) {
            throw fail("unsupported Hadamard sign mode: " + mode);
        }
        return (String) mode;
    }

    /** A positive power of two (base.py:711-713), at most the loader's largest block. */
    private static int blockSize(Object value) {
        Long b = value instanceof Double ? null : exactInteger(value);
        if (b == null || b <= 0 || (b & (b - 1)) != 0 || b > HadamardTransforms.MAX_BLOCK) {
            throw fail("invalid Hadamard block size: " + value + " (a power of two up to "
                    + HadamardTransforms.MAX_BLOCK + ")");
        }
        return (int) (long) b;
    }

    /** The explicit sign table, ordered by width (base.py:720-735). */
    private static void readSigns(Object signsValue, int blockSize, List<Long> widths, List<Long> values) {
        if (!(signsValue instanceof Map) || ((Map<?, ?>) signsValue).isEmpty()) {
            throw fail("explicit sign mode requires a signs table");
        }
        TreeMap<Long, Object> byWidth = new TreeMap<>();
        for (Map.Entry<String, Object> e : asMap(signsValue).entrySet()) {
            long width = parseWidth(e.getKey());
            if (byWidth.put(width, e.getValue()) != null) {
                throw fail("sign width " + width + " is listed twice");
            }
        }
        for (Map.Entry<Long, Object> e : byWidth.entrySet()) {
            long width = e.getKey();
            if (width <= 0 || width % blockSize != 0) {
                throw fail("sign width " + width + " must be positive and a multiple of block size " + blockSize);
            }
            if (!(e.getValue() instanceof List) || ((List<?>) e.getValue()).size() != width) {
                throw fail("invalid sign vector for width " + width);
            }
            List<Long> vec = new ArrayList<>((int) width);
            for (Object v : (List<?>) e.getValue()) {
                Long s = v instanceof Number ? exactInteger(v) : null;
                if (s == null || (s != 1 && s != -1)) {
                    throw fail("invalid sign vector for width " + width);
                }
                vec.add(s);
            }
            widths.add(width);
            values.addAll(vec);
        }
    }

    private static long parseWidth(String key) {
        String k = key.trim();
        if (!k.matches("[+-]?[0-9]{1,18}")) {
            throw fail("sign width '" + key + "' is not an integer");
        }
        return Long.parseLong(k);
    }

    /** Filters and maps every tensor record (base.py:770-799). */
    private static void mapRecords(HfLoadContext ctx, List<?> records, List<String> weightNames,
                                   List<String> inverseNames) {
        HfTensorNameMap names = ctx.names();
        for (Object recordValue : records) {
            if (!(recordValue instanceof Map) || !(((Map<?, ?>) recordValue).get("name") instanceof String)) {
                throw fail("the manifest has an invalid tensor record");
            }
            Map<String, Object> record = asMap(recordValue);
            String name = (String) record.get("name");
            Long axis = exactInteger(record.get("axis"));
            if (axis == null || axis != -1) {
                throw fail("unsupported Hadamard tensor axis for '" + name + "'");
            }
            Object role = record.containsKey("role") ? record.get("role") : ROLE_FOLD;
            if (!ROLE_FOLD.equals(role) && !ROLE_INVERSE.equals(role)) {
                throw fail("unsupported Hadamard tensor role for '" + name + "': " + role);
            }
            String filtered = HfTensorFilters.filterName(ctx, name);
            if (filtered == null) {
                throw fail("Hadamard tensor is filtered out: '" + name + "'");
            }
            String mapped = mapName(names, filtered);
            if (ROLE_INVERSE.equals(role)) {
                if (!PrismHadamardLoader.TOKEN_EMBD.equals(mapped)) {
                    throw fail("Hadamard tensor '" + name + "' maps to '" + mapped
                            + "', which is not a verified inverse-after-lookup table");
                }
                inverseNames.add(mapped);
            } else {
                if (!PrismHadamardLoader.isFoldableWeight(mapped)) {
                    throw fail("Hadamard tensor '" + name + "' maps to '" + mapped
                            + "', which is not on a verified Hadamard-aware matmul path");
                }
                weightNames.add(mapped);
            }
        }
    }

    private static String mapName(HfTensorNameMap names, String filtered) {
        try {
            return names.mapOrThrow(filtered);
        } catch (GgufFormatException e) {
            throw fail(e.getMessage());
        }
    }

    /**
     * {@code tied_output} and its consistency with the schema and the checkpoint (base.py:815-828); a tied output
     * also requires that no {@code output.weight} is planned, which the runtime requires (PrismHadamardLoader).
     */
    private static boolean tiedOutput(HfLoadContext ctx, HfPlanView plan, Map<String, Object> manifest, long schema,
                                      List<String> weightNames, List<String> inverseNames) {
        Object tiedValue = manifest.containsKey("tied_output") ? manifest.get("tied_output") : Boolean.FALSE;
        if (!(tiedValue instanceof Boolean) || (schema == TIED_SCHEMA) != (Boolean) tiedValue) {
            throw fail("Hadamard schema 3 requires tied_output=true; older schemas forbid it");
        }
        boolean tied = (Boolean) tiedValue;
        boolean tieEmbeddings = ctx.config().findBool(false, "tie_word_embeddings");
        if (tied) {
            if (!inverseNames.equals(Collections.singletonList(PrismHadamardLoader.TOKEN_EMBD))) {
                throw fail("tied Hadamard output requires one latent token embedding");
            }
            if (!tieEmbeddings) {
                throw fail("tied Hadamard output requires tie_word_embeddings=true");
            }
            if (weightNames.contains(PrismHadamardLoader.OUTPUT) || hasOutputHead(ctx)
                    || plan.has(PrismHadamardLoader.OUTPUT)) {
                throw fail("tied Hadamard output must not carry a separate output head");
            }
        } else if (inverseNames.contains(PrismHadamardLoader.TOKEN_EMBD) && tieEmbeddings) {
            throw fail("a tied latent embedding requires Hadamard schema 3 and tied_output=true");
        }
        return tied;
    }

    private static boolean hasOutputHead(HfLoadContext ctx) {
        HfTensorNameMap names = ctx.names();
        for (String name : ctx.index().names()) {
            HfTensorNameMap.Mapped m = names.map(name, ".weight", ".bias").orElse(null);
            if (m != null && PrismHadamardLoader.OUTPUT.equals(m.ggufName())) {
                return true;
            }
        }
        return false;
    }

    /** An exact integer value of a JSON number (integral doubles included), or null. */
    private static Long exactInteger(Object v) {
        if (v instanceof Long || v instanceof Integer) {
            return ((Number) v).longValue();
        }
        if (v instanceof BigInteger) {
            return ((BigInteger) v).bitLength() < Long.SIZE ? ((BigInteger) v).longValue() : null;
        }
        if (v instanceof Double) {
            double d = (Double) v;
            return d == Math.rint(d) && Math.abs(d) < 0x1p53 ? (long) d : null;
        }
        return null;
    }

    private static GgufFormatException fail(String reason) {
        return new GgufFormatException(MANIFEST + ": " + reason);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asMap(Object v) {
        return (Map<String, Object>) v;
    }
}
