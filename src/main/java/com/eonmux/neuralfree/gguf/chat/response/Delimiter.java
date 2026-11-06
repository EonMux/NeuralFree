package com.eonmux.neuralfree.gguf.chat.response;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

/**
 * A region anchor of a {@code response_template} (vLLM {@code _compile_anchor}): a regex ({@code <key>_pattern}), one
 * literal or a list of literals ({@code <key>}). Literals keep their strings, so a complete literal match is final
 * unless one literal is a strict prefix of another ({@link #canExtend}). Immutable.
 */
final class Delimiter {

    private final TemplateRegex regex;
    private final List<String> literals;
    private final boolean canExtend;

    private Delimiter(TemplateRegex regex, List<String> literals, boolean canExtend) {
        this.regex = regex;
        this.literals = literals;
        this.canExtend = canExtend;
    }

    /**
     * The anchor {@code spec} declares under {@code literalKey} or {@code patternKey}, or null when it declares neither.
     *
     * @param scope what declares it, for errors ({@code Field 'thinking'})
     * @throws IllegalArgumentException for both keys, a literal that is not a non-empty string or a non-empty list of
     *                                  them, or a pattern that is not a string or does not compile
     */
    static Delimiter of(String scope, Map<?, ?> spec, String literalKey, String patternKey) {
        boolean literal = spec.containsKey(literalKey);
        boolean pattern = spec.containsKey(patternKey);
        if (literal && pattern) {
            throw new IllegalArgumentException(scope + ": cannot specify both '" + literalKey + "' and '"
                    + patternKey + "'");
        }
        if (literal) {
            List<String> literals = literals(scope, literalKey, spec.get(literalKey));
            boolean canExtend = false;
            for (String a : literals) {
                for (String b : literals) {
                    canExtend |= !a.equals(b) && a.startsWith(b);
                }
            }
            return new Delimiter(TemplateRegex.anyOf(literals), Collections.unmodifiableList(literals), canExtend);
        }
        if (pattern) {
            Object raw = spec.get(patternKey);
            if (!(raw instanceof String)) {
                throw new IllegalArgumentException(scope + ": '" + patternKey + "' must be a string");
            }
            return new Delimiter(TemplateRegex.compile((String) raw, scope + ": " + patternKey), null, false);
        }
        return null;
    }

    private static List<String> literals(String scope, String key, Object raw) {
        List<String> out = new ArrayList<>();
        if (raw instanceof String) {
            out.add((String) raw);
        } else if (raw instanceof List) {
            if (((List<?>) raw).isEmpty()) {
                throw new IllegalArgumentException(scope + ": '" + key + "' list must contain at least one literal");
            }
            LinkedHashSet<String> unique = new LinkedHashSet<>();
            for (Object o : (List<?>) raw) {
                if (!(o instanceof String)) {
                    throw new IllegalArgumentException(scope + ": '" + key + "' list must contain only strings");
                }
                unique.add((String) o);
            }
            out.addAll(unique);
        } else {
            throw new IllegalArgumentException(scope + ": '" + key + "' must be a string or list of strings, got "
                    + (raw == null ? "null" : raw.getClass().getSimpleName()));
        }
        if (out.contains("")) {
            throw new IllegalArgumentException(scope + ": '" + key + "' literals cannot be empty strings");
        }
        return out;
    }

    TemplateRegex regex() {
        return regex;
    }

    /** The literal strings, in declaration order, or null for a regex anchor. */
    List<String> literals() {
        return literals;
    }

    /** Whether a complete literal match at the end of the input could still grow into a longer literal. */
    boolean canExtend() {
        return canExtend;
    }

    /**
     * The literal strings of the anchor, or of its regex when the regex spells only literals (vLLM's
     * {@code _literals}); null when the regex matches more than literals.
     */
    List<String> spelled() {
        return literals != null ? literals : LiteralPatterns.expand(regex.source());
    }
}
