package com.eonmux.neuralfree.gguf.hf.meta;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfRope;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalDouble;
import java.util.OptionalLong;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * {@code TextModel.set_gguf_parameters} (LC/conversion/base.py:1606-1720): the architecture metadata every text model
 * class writes before its own keys, driven row by row by {@link HfHparamAliases} in table order, so the keys come out
 * in the converter's order. The table supplies the keys, their types and the aliases read (first present wins, a
 * present {@code null} ends the search, as {@code find_hparam} does); this class implements each rule kind.
 *
 * <p>The RoPE rows follow the converter: the parameters are {@code rope_parameters.full_attention} or the whole
 * {@code rope_parameters} ({@link HfRope#global()}). {@code linear} with a factor writes type and factor; {@code yarn}
 * with a factor writes type, factor, the required original context length and the optional truncate, extrapolation
 * factor, attention factor and beta fast/slow; {@code su}/{@code longrope} writes the type only; {@code dynamic} and
 * {@code llama3} write nothing; any other type logs a warning. {@code rope_parameters.mrope_section} is padded with
 * zeros to four sections. The {@code mirror} rows are {@code TextModel.__init__}'s alias normalization, which
 * {@link HfRope} performs when the configuration is loaded; their aliases are checked against {@link HfRope}'s.</p>
 *
 * <p>{@code general.*} keys are not written here; the model source writes them.</p>
 */
public final class HfGenericMetadata {

    private static final Logger LOG = LogManager.getLogger(HfGenericMetadata.class);

    private static final Pattern ROPE_LITERAL = Pattern.compile("rope_type == '([^']+)'");
    private static final Pattern SCORE_LITERAL = Pattern.compile("score_func == '([^']+)'");
    private static final Pattern ENUM_VALUE =
            Pattern.compile("gguf\\.(RopeScalingType|ExpertGatingFuncType)\\.([A-Z]+)");
    private static final Pattern ERROR_VALUE = Pattern.compile("ValueError\\(f'([^']*)\\{([a-z_]+)}([^']*)'\\)");
    private static final String MROPE_SECTION = "mrope_section";
    private static final int MROPE_SECTIONS = 4;

    /** {@code gguf.ExpertGatingFuncType} (LC/gguf-py/gguf/constants.py:6285-6288). */
    private static final Map<String, Integer> GATING_FUNC = Map.of("SOFTMAX", 1, "SIGMOID", 2, "SQRTSOFTPLUS", 4);

    /** The scaling a RoPE type selects in {@code set_gguf_parameters}. */
    private enum RopeBranch {
        NONE, LINEAR, YARN, LONGROPE
    }

    /** The locals of {@code set_gguf_parameters}. Mutable, one per call. */
    private static final class State {
        final HfLoadContext ctx;
        final HfConfig cfg;
        HfRope.Params ropeParams;
        String ropeType;
        RopeBranch branch = RopeBranch.NONE;
        Object scoreFunc;
        boolean scoreMatched;

        State(HfLoadContext ctx) {
            this.ctx = ctx;
            this.cfg = ctx.config();
        }
    }

    private HfGenericMetadata() {
    }

    /**
     * Writes the generic text-model metadata of {@code ctx}'s configuration into {@code kv}.
     *
     * @throws GgufFormatException   when a required value is missing or a value has the wrong type or range, or the
     *                               expert score function is not supported
     * @throws IllegalStateException when the table holds a row this class does not implement (a build defect)
     */
    public static void apply(HfLoadContext ctx, HfMetadataBuilder kv) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(kv, "kv");
        State st = new State(ctx);
        for (HfHparamAliases.Row r : HfHparamAliases.rows()) {
            switch (r.head()) {
                case LOCAL:
                    local(st, r);
                    break;
                case MIRROR:
                    checkMirror(r);
                    break;
                case REQUIRED:
                case OPTIONAL:
                    write(st, kv, r);
                    break;
                case ALWAYS:
                    always(st, kv, r);
                    break;
                case REJECT:
                    reject(st, r);
                    break;
                default:
                    throw r.defect("unhandled rule head " + r.head());
            }
        }
    }

    // ------------------------------------------------------------------ locals and mirrors

    private static void local(State st, HfHparamAliases.Row r) {
        switch (r.target()) {
            case "rope_parameters":
                requireAliases(r, "rope_parameters", "rope_scaling");
                break;
            case "rope_params":
                requireAliases(r, "rope_parameters.full_attention", "rope_parameters");
                st.ropeParams = st.cfg.rope().global();
                break;
            case "rope_type":
                requireAliases(r, "rope_params.rope_type");
                st.ropeType = st.ropeParams.type().orElse(null);
                break;
            case "rope_factor":
                requireAliases(r, "rope_params.factor");
                st.branch = ropeBranch(st);
                break;
            case "score_func":
                st.scoreFunc = st.cfg.find(r.aliases().toArray(new String[0])).orElse(null);
                break;
            default:
                throw r.defect("unhandled local '" + r.target() + "'");
        }
    }

    /** The branch {@code set_gguf_parameters} takes for the RoPE type (base.py:1634-1665). */
    private static RopeBranch ropeBranch(State st) {
        String type = st.ropeType;
        if (type == null) {
            return RopeBranch.NONE;
        }
        boolean hasFactor = st.ropeParams.factor().isPresent();
        if ("linear".equals(type) && hasFactor) {
            return RopeBranch.LINEAR;
        }
        if ("yarn".equals(type) && hasFactor) {
            return RopeBranch.YARN;
        }
        if ("su".equals(type) || "longrope".equals(type)) {
            return RopeBranch.LONGROPE;
        }
        if (!"dynamic".equals(type) && !"llama3".equals(type.toLowerCase(Locale.ROOT))) {
            LOG.warn("{}: unknown RoPE type: {}", st.cfg.architecture(), type);
        }
        return RopeBranch.NONE;
    }

    /** Checks that a {@code mirror} row's aliases are the ones {@link HfRope} applies, in its order. */
    private static void checkMirror(HfHparamAliases.Row r) {
        List<String> expected;
        switch (r.target()) {
            case "rope_parameters.sliding_attention.rope_theta":
                expected = HfRope.LOCAL_THETA_ALIASES;
                break;
            case "rope_parameters.rope_theta":
                expected = HfRope.THETA_ALIASES;
                break;
            case "rope_parameters.partial_rotary_factor":
                expected = HfRope.PARTIAL_ROTARY_ALIASES;
                break;
            case "rope_parameters.rope_type":
                expected = List.of("rope_parameters.type");
                break;
            case "rope_parameters.original_max_position_embeddings":
                expected = List.of("original_max_position_embeddings");
                break;
            default:
                throw r.defect("unhandled mirror '" + r.target() + "'");
        }
        if (r.aliases().size() > expected.size() || !expected.subList(0, r.aliases().size()).equals(r.aliases())) {
            throw r.defect("aliases " + r.aliases() + " are not the leading aliases " + expected + " HfRope applies");
        }
    }

    private static void requireAliases(HfHparamAliases.Row r, String... expected) {
        if (!r.aliases().equals(Arrays.asList(expected))) {
            throw r.defect("aliases " + r.aliases() + " differ from " + Arrays.toString(expected));
        }
    }

    // ------------------------------------------------------------------ writes

    private static void write(State st, HfMetadataBuilder kv, HfHparamAliases.Row r) {
        String key = r.template();
        if (GgufKeys.BLOCK_COUNT.equals(key)) {
            kv.set(key, st.ctx.blockCount());
            return;
        }
        if (GgufKeys.ROPE_DIMENSION_SECTIONS.equals(key)) {
            writeMropeSections(st, kv, r);
            return;
        }
        if (r.condition() != null && !conditionHolds(st, r)) {
            return;
        }
        String first = r.aliases().get(0);
        Object value = first.startsWith("rope_params.") || first.startsWith("rope_parameters.")
                ? ropeValue(st, r) : flatValue(st, r);
        if (value == null) {
            if (r.head() == HfHparamAliases.Head.REQUIRED) {
                throw new GgufFormatException(st.cfg.source() + ": required key " + r.aliases() + " is missing");
            }
            return;
        }
        setChecked(st, kv, key, value, r.aliases());
    }

    /** A flat {@code hparams} value through the typed {@link HfConfig} lookups, which apply the configuration caps. */
    private static Object flatValue(State st, HfHparamAliases.Row r) {
        String[] aliases = r.aliases().toArray(new String[0]);
        GgufKeyTable.KeyType t = r.type();
        if (t.any()) {
            Object v = st.cfg.find(aliases).orElse(null);
            return v instanceof List ? v : v == null ? null : (Object) st.cfg.findLong(aliases).getAsLong();
        }
        switch (t.type()) {
            case UINT32:
            case INT32: {
                OptionalLong v = st.cfg.findLong(aliases);
                return v.isPresent() ? (Object) v.getAsLong() : null;
            }
            case FLOAT32: {
                OptionalDouble v = st.cfg.findFloat(aliases);
                return v.isPresent() ? (Object) v.getAsDouble() : null;
            }
            case BOOL:
                return st.cfg.findBool(aliases).orElse(null);
            case STRING:
                return st.cfg.findString(aliases).orElse(null);
            default:
                throw r.defect("unhandled value type " + t + " for flat aliases");
        }
    }

    /**
     * The first present of {@code rope_params.<k>} (the global RoPE parameters) or {@code rope_parameters.<path>}
     * (the whole normalized object) aliases; a present {@code null} ends the search.
     */
    private static Object ropeValue(State st, HfHparamAliases.Row r) {
        for (String alias : r.aliases()) {
            Map<String, Object> scope;
            String path;
            if (alias.startsWith("rope_params.")) {
                scope = st.ropeParams.raw();
                path = alias.substring("rope_params.".length());
            } else if (alias.startsWith("rope_parameters.")) {
                scope = st.cfg.rope().raw();
                path = alias.substring("rope_parameters.".length());
            } else {
                throw r.defect("alias '" + alias + "' mixes RoPE and flat lookups");
            }
            String[] parts = path.split("\\.");
            for (int i = 0; i < parts.length - 1 && scope != null; i++) {
                Object next = scope.get(parts[i]);
                scope = next instanceof Map ? asMap(next) : null;
            }
            if (scope != null && scope.containsKey(parts[parts.length - 1])) {
                return scope.get(parts[parts.length - 1]);
            }
        }
        return null;
    }

    /** {@code rope_parameters.mrope_section}, padded with zeros to four entries and cut to four (base.py:1668-1674). */
    private static void writeMropeSections(State st, HfMetadataBuilder kv, HfHparamAliases.Row r) {
        requireAliases(r, "rope_parameters." + MROPE_SECTION);
        Map<String, Object> top = st.cfg.rope().raw();
        if (!top.containsKey(MROPE_SECTION)) {
            return;
        }
        Object v = top.get(MROPE_SECTION);
        if (!(v instanceof List)) {
            throw new GgufFormatException(st.cfg.source() + ": 'rope_parameters." + MROPE_SECTION
                    + "' must be an array of integers");
        }
        List<Object> sections = new ArrayList<>((List<?>) v);
        while (sections.size() < MROPE_SECTIONS) {
            sections.add(0L);
        }
        setChecked(st, kv, r.template(), sections.subList(0, MROPE_SECTIONS), r.aliases());
    }

    private static void always(State st, HfMetadataBuilder kv, HfHparamAliases.Row r) {
        String key = r.template();
        if (GgufKeys.ATTENTION_CAUSAL.equals(key)) {
            requireAliases(r, "is_causal", "use_bidirectional_attention");
            Map<String, Object> flat = st.cfg.flat();
            if (!"False".equals(r.value())) {
                throw r.defect("unhandled value " + r.value());
            }
            if (Boolean.FALSE.equals(flat.get("is_causal"))
                    || Boolean.TRUE.equals(flat.get("use_bidirectional_attention"))) {
                kv.set(key, false);
            }
            return;
        }
        Matcher m = ENUM_VALUE.matcher(r.value());
        if (!m.matches()) {
            throw r.defect("unhandled value " + r.value());
        }
        if (GgufKeys.ROPE_SCALING_TYPE.equals(key) && "RopeScalingType".equals(m.group(1))) {
            if (conditionHolds(st, r)) {
                kv.set(key, m.group(2).toLowerCase(Locale.ROOT));
            }
        } else if (GgufKeys.EXPERT_GATING_FUNC.equals(key) && "ExpertGatingFuncType".equals(m.group(1))) {
            Integer code = GATING_FUNC.get(m.group(2));
            if (code == null) {
                throw r.defect("unknown ExpertGatingFuncType." + m.group(2));
            }
            if (conditionHolds(st, r)) {
                kv.set(key, code);
                st.scoreMatched = true;
            }
        } else {
            throw r.defect("unhandled always-row for " + key);
        }
    }

    /** The converter's {@code raise ValueError(...)} when no gating function matched the score function. */
    private static void reject(State st, HfHparamAliases.Row r) {
        Matcher m = ERROR_VALUE.matcher(r.value());
        if (!m.matches() || !"score_func".equals(m.group(2))
                || r.condition() == null || !r.condition().startsWith("score_func is not None")) {
            throw r.defect("unhandled reject rule");
        }
        if (st.scoreFunc != null && !st.scoreMatched) {
            throw new GgufFormatException(m.group(1) + st.scoreFunc + m.group(3));
        }
    }

    /**
     * Whether a row's {@code when} condition holds. RoPE conditions select the branch of their last
     * {@code rope_type == '<t>'} literal; score-function conditions match their last {@code score_func == '<f>'}
     * literal.
     */
    private static boolean conditionHolds(State st, HfHparamAliases.Row r) {
        String c = r.condition();
        if (c == null) {
            return true;
        }
        if (c.contains("rope_type")) {
            return st.branch != RopeBranch.NONE && st.branch == branchOf(r, lastGroup(ROPE_LITERAL, c, r));
        }
        if (c.contains("score_func")) {
            return st.scoreFunc != null && lastGroup(SCORE_LITERAL, c, r).equals(st.scoreFunc);
        }
        throw r.defect("unhandled condition '" + c + "'");
    }

    private static RopeBranch branchOf(HfHparamAliases.Row r, String literal) {
        switch (literal) {
            case "linear": return RopeBranch.LINEAR;
            case "yarn": return RopeBranch.YARN;
            case "su":
            case "longrope": return RopeBranch.LONGROPE;
            default: throw r.defect("unhandled RoPE type literal '" + literal + "'");
        }
    }

    private static String lastGroup(Pattern p, String condition, HfHparamAliases.Row r) {
        Matcher m = p.matcher(condition);
        String last = null;
        while (m.find()) {
            last = m.group(1);
        }
        if (last == null) {
            throw r.defect("condition '" + condition + "' has no " + p.pattern() + " literal");
        }
        return last;
    }

    private static void setChecked(State st, HfMetadataBuilder kv, String key, Object value, List<String> aliases) {
        try {
            kv.set(key, value);
        } catch (GgufFormatException e) {
            throw e;
        } catch (IllegalArgumentException e) {
            throw new GgufFormatException(st.cfg.source() + ": " + aliases + ": " + e.getMessage(), e);
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asMap(Object v) {
        return (Map<String, Object>) v;
    }
}
