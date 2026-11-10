package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Pattern;

/**
 * The rule registry of the grammar builders, the Java port of the {@code _rules} map and
 * {@code _add_rule} of llama.cpp's {@code common_chat_schema_converter}
 * ({@code common/json-schema-to-grammar.cpp:356-369}) and of its {@code format_grammar}
 * ({@code :984-990}).
 *
 * <p>Every grammar builder ({@link JsonSchemaToGbnf},
 * {@link RegexToGbnf} and their callers) writes llama.cpp-style rule text into one builder;
 * {@link #format()} emits the registry as a GBNF grammar text (one {@code name ::= rule} line
 * per entry, entries in name order as llama.cpp's sorted map emits them) and is normally passed
 * to {@link GbnfParser#parse}, which yields the {@link Grammar} the sampling runtime consumes
 * (validation for undefined rules and left recursion included).</p>
 *
 * <p>{@link #addRule} sanitizes the requested name (every run of characters outside
 * {@code [A-Za-z0-9-]} becomes {@code -}, the {@code INVALID_RULE_CHARS_RE} of llama.cpp) and
 * deduplicates: the same name registered again with the same content returns the stored name;
 * a different content gets a numeric suffix ({@code name0}, {@code name1}, ...). Rules can be
 * snapshotted and restored so a failed sub-conversion leaves no partial rules behind (the
 * {@code _visit_pattern} revert of llama.cpp, {@code :390-404}). Immutable once formatted and
 * thread-confined to the thread building the grammar.</p>
 */
public final class GbnfBuilder {

    /**
     * The flexible-whitespace rule text, llama.cpp {@code SPACE_RULE}
     * ({@code common/json-schema-to-grammar.cpp:229}): an empty first alternate, then
     * one space, then one or two newlines followed by blanks, tabs or nothing. Referenced
     * by {@code space} in every JSON rule this builder produces.
     */
    public static final String SPACE_RULE = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

    private static final Pattern INVALID_RULE_CHARS = Pattern.compile("[^a-zA-Z0-9-]+");

    /** Rule definitions in name order; {@code format()} iterates exactly this order. */
    private final TreeMap<String, String> rules = new TreeMap<>();

    /** The attribute text of the budgeted rules ({@code [max_tokens=n]}), by stored name. */
    private final Map<String, String> attributes = new HashMap<>();

    /** A builder seeding {@code space} with the flexible-whitespace rule. */
    public GbnfBuilder() {
        rules.put("space", SPACE_RULE);
    }

    /**
     * A builder seeding {@code space} with an explicit rule text; {@code JsonSchemaToGbnf}
     * passes the compact mode's empty literal here. Does not run through
     * {@link #addRule}: llama.cpp seeds {@code _rules["space"]}
     * ({@code common/json-schema-to-grammar.cpp:817-818}) directly.
     *
     * @param spaceRule the right-hand text of the {@code space} rule; the empty literal {@code ""} is
     *                  the compact mode (whitespace forbidden between JSON tokens)
     */
    public GbnfBuilder(String spaceRule) {
        rules.put("space", spaceRule);
    }

    /**
     * Adds or deduplicates a rule (llama.cpp {@code _add_rule},
     * {@code common/json-schema-to-grammar.cpp:356-369}).
     *
     * @param name       the requested rule name; sanitized before storage
     * @param definition the rule text right of {@code ::=} (may span character sets, literals
     *                   and references; its GBNF validity is checked by {@link #grammar()})
     * @return the stored name, which callers reference in other rules
     */
    public String addRule(String name, String definition) {
        return store(sanitizeRuleName(name), "", definition);
    }

    /**
     * Adds or deduplicates a rule carrying xgrammar's budget attributes ({@code name[max_tokens=n, max_chars=m]},
     * {@link GbnfParser}): its text consumes at most {@code maxTokens} tokens and {@code maxChars} code points.
     *
     * @param maxTokens the token budget, or negative for none
     * @param maxChars  the code-point budget, or negative for none
     * @return the stored rule name
     */
    public String addBudgetedRule(String name, int maxTokens, int maxChars, String definition) {
        List<String> budgets = new ArrayList<>();
        if (maxTokens >= 0) {
            budgets.add("max_tokens=" + maxTokens);
        }
        if (maxChars >= 0) {
            budgets.add("max_chars=" + maxChars);
        }
        String attribute = budgets.isEmpty() ? "" : "[" + String.join(", ", budgets) + "]";
        return store(sanitizeRuleName(name), attribute, definition);
    }

    private String store(String storedName, String attribute, String definition) {
        int suffix = -1;
        while (true) {
            String candidate = suffix < 0 ? storedName : storedName + suffix;
            String existing = rules.get(candidate);
            if (existing == null) {
                rules.put(candidate, definition);
                if (!attribute.isEmpty()) {
                    attributes.put(candidate, attribute);
                }
                return candidate;
            }
            if (existing.equals(definition) && attribute.equals(attributes.getOrDefault(candidate, ""))) {
                return candidate;
            }
            suffix++;
        }
    }

    /**
     * Replaces the seeded {@code space} rule. A converter translating an explicit whitespace pattern calls this
     * before any schema rule references the rule, so every JSON token is separated by the pattern's grammar.
     *
     * @param definition the right-hand text of the {@code space} rule
     */
    void setSpaceRule(String definition) {
        rules.put("space", definition);
    }

    /** Whether {@code name} (already sanitized) is defined. */
    public boolean hasRule(String name) {
        return rules.containsKey(name);
    }

    /** The definition stored under {@code name}, or null. */
    public String rule(String name) {
        return rules.get(name);
    }

    /** A map copy of the current registry, keyed in storage order; never the live map. */
    public Map<String, String> snapshot() {
        return new TreeMap<>(rules);
    }

    /** Restores an earlier {@link #snapshot()}, discarding everything added since. */
    public void restore(Map<String, String> taken) {
        rules.clear();
        rules.putAll(taken);
        attributes.keySet().retainAll(taken.keySet());
    }

    /**
     * The registry as GBNF text (llama.cpp {@code format_grammar},
     * {@code common/json-schema-to-grammar.cpp:984-990}): one {@code name ::= rule} line per
     * entry, in name order.
     */
    public String format() {
        StringBuilder out = new StringBuilder();
        for (Map.Entry<String, String> entry : rules.entrySet()) {
            if (out.length() > 0) {
                out.append('\n');
            }
            out.append(entry.getKey()).append(attributes.getOrDefault(entry.getKey(), "")).append(" ::= ")
                    .append(entry.getValue());
        }
        return out.toString();
    }

    /**
     * Formats the registry and parses it with {@link GbnfParser}, producing the validated
     * {@link Grammar} (the compiled automaton the sampling runtime and lazy triggers consume).
     *
     * @throws GrammarException when the formatted text violates the GBNF grammar (an undefined
     *                          rule, a missing {@code root} rule, left recursion) or a
     *                          {@link GrammarLimits} bound
     */
    public Grammar grammar() {
        return GbnfParser.parse(format());
    }

    /**
     * Converts a {@link ConstraintSpec} to its validated {@link Grammar}, the dispatch the
     * request wiring calls ({@code response_format}, {@code json_schema}):
     * {@link Kind#GBNF GBNF} through {@link GbnfParser}, {@link Kind#LARK LARK} through
     * {@link LarkGrammar}, {@link Kind#JSON_SCHEMA JSON_SCHEMA}
     * and {@link Kind#JSON_OBJECT JSON_OBJECT} through {@link JsonSchemaToGbnf} under the spec's JSON options (the
     * whitespace mode, {@code whitespace_pattern} and {@code disable_additional_properties}), {@link Kind#REGEX
     * REGEX} through {@link RegexToGbnf#compile} and
     * {@link Kind#CHOICE CHOICE} as an alternation of escaped literals.
     *
     * @throws IllegalArgumentException when the constraint is malformed or untranslatable
     */
    public static Grammar grammarOf(ConstraintSpec spec) {
        return grammarOf(spec, null);
    }

    /**
     * {@link #grammarOf(ConstraintSpec)} with the model vocabulary a GBNF grammar's {@code <token>}
     * references resolve through ({@link GbnfParser#parse(String, GrammarVocab)}).
     *
     * @param vocab the vocabulary, or null to reject the {@code <token>} name form
     * @throws IllegalArgumentException when the constraint is malformed or untranslatable
     */
    public static Grammar grammarOf(ConstraintSpec spec, GrammarVocab vocab) {
        switch (spec.kind()) {
            case GBNF:
                return GbnfParser.parse(spec.text(), vocab);
            case LARK:
                return LarkGrammar.compile(spec.text(), spec.whitespace(), vocab);
            case JSON_SCHEMA:
                return JsonSchemaToGbnf.convert(spec.text(), spec.jsonOptions());
            case JSON_OBJECT:
                return GbnfParser.parse(JsonSchemaToGbnf.gbnf("{}", spec.jsonOptions()));
            case REGEX:
                return regexGrammar(spec);
            case CHOICE: {
                GbnfBuilder builder = new GbnfBuilder();
                List<String> literals = new ArrayList<>();
                for (String choice : spec.choices()) {
                    literals.add(JsonSchemaToGbnf.formatLiteral(choice));
                }
                builder.addRule(Grammar.ROOT_RULE, "(" + String.join(" | ", literals) + ")");
                return builder.grammar();
            }
            case STRUCTURAL_TAG:
                return StructuralTag.parse(spec.text(), spec.whitespace(), vocab);
            default:
                throw new IllegalArgumentException("constraint kind " + spec.kind() + " has no compiler");
        }
    }

    /**
     * The grammar of a {@link Kind#REGEX REGEX} spec through {@link RegexToGbnf#compile}; a
     * refusal names the regex constraint and the construct or problem.
     */
    private static Grammar regexGrammar(ConstraintSpec spec) {
        try {
            return RegexToGbnf.compile(spec.text(), Grammar.ROOT_RULE, spec.dotAll());
        } catch (RegexToGbnf.UnsupportedRegexException e) {
            throw new IllegalArgumentException("regex constraint " + spec.text()
                    + " uses an unsupported construct: " + e.getMessage(), e);
        } catch (RegexToGbnf.InvalidRegexException e) {
            throw new IllegalArgumentException("regex constraint " + spec.text()
                    + " is not a valid regex: " + e.getMessage(), e);
        }
    }

    /**
     * Compiles {@code spec} for a sampler: its grammar ({@link #grammarOf}) and the triggers gating it — the spec's
     * own {@link ConstraintSpec#triggers()}, else, for a structural tag, the payload's own triggers and free-text
     * exclusions ({@link StructuralTag#compileConstraint}), else the {@link ConstraintSpec#gate() gate}'s words, the
     * grammar then wrapped behind the gate's prefix ({@link ConstraintGate}).
     *
     * @throws IllegalArgumentException when the constraint is malformed or untranslatable
     */
    public static CompiledConstraint compile(ConstraintSpec spec) {
        return compile(spec, null);
    }

    /**
     * {@link #compile(ConstraintSpec)} resolving {@code <token>} references through {@code vocab}.
     *
     * @param vocab the vocabulary, or null to reject the {@code <token>} name form
     * @throws IllegalArgumentException when the constraint is malformed or untranslatable
     */
    public static CompiledConstraint compile(ConstraintSpec spec, GrammarVocab vocab) {
        if (spec.kind() == ConstraintSpec.Kind.STRUCTURAL_TAG && spec.triggers() == null) {
            CompiledConstraint compiled = StructuralTag.compileConstraint(spec.text(), spec.whitespace(), vocab);
            if (compiled.triggers() == null && spec.gate() != null) {
                return new CompiledConstraint(spec.gate().wrap(compiled.grammar()), spec.gate().triggers());
            }
            return compiled;
        }
        Grammar grammar = grammarOf(spec, vocab);
        GrammarTriggers triggers = spec.triggers();
        if (triggers == null && spec.gate() != null) {
            return new CompiledConstraint(spec.gate().wrap(grammar), spec.gate().triggers());
        }
        return new CompiledConstraint(grammar, triggers);
    }

    /** Sanitizes a rule name, llama.cpp {@code INVALID_RULE_CHARS_RE} ({@code :275}). */
    static String sanitizeRuleName(String name) {
        return INVALID_RULE_CHARS.matcher(name).replaceAll("-");
    }

    /**
     * The repetition suffix of {@code itemRule} for {@code minItems}/{code maxItems}
     * bounds and an optional separator, the port of llama.cpp's {@code build_repetition}
     * ({@code common/json-schema-to-grammar.cpp:18-43}): {@maxItems == 0} yields an empty
     * text, {@code {0,1}} a {@code ?}, the separator-less forms {@code ? * + {m,} {m,n}}
     * depending on the bounds and a separator rule one nested comma-group per bound as
     * llama.cpp builds it. The largest meaningful bound is Integer.MAX_VALUE ({@code *} and
     * {@code +}: {@code has_max == false} in llama.cpp).
     *
     * @param itemRule      the rule text repeated (a rule name or a quoted literal)
     * @param minItems      the lower repetition bound
     * @param maxItems      the upper repetition bound; Integer.MAX_VALUE means unbounded
     * @param separatorRule the separator rule text, empty in llama.cpp's string/number forms
     */
    public static String repetition(String itemRule, int minItems, int maxItems,
            String separatorRule) {
        boolean hasMax = maxItems != Integer.MAX_VALUE;
        if (maxItems == 0) {
            return "";
        }
        if (minItems == 0 && maxItems == 1) {
            return itemRule + "?";
        }
        if (separatorRule.isEmpty()) {
            if (minItems == 1 && !hasMax) {
                return itemRule + "+";
            }
            if (minItems == 0 && !hasMax) {
                return itemRule + "*";
            }
            return itemRule + "{" + minItems + "," + (hasMax ? String.valueOf(maxItems) : "") + "}";
        }
        String result = itemRule + " " + repetition("(" + separatorRule + " " + itemRule + ")",
                minItems == 0 ? 0 : minItems - 1,
                hasMax ? maxItems - 1 : maxItems, "");
        if (minItems == 0) {
            result = "(" + result + ")?";
        }
        return result;
    }
}
