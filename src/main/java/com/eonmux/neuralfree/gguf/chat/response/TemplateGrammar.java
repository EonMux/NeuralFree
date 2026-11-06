package com.eonmux.neuralfree.gguf.chat.response;

import com.eonmux.neuralfree.gguf.chat.ChatTool;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Structural tags for tool calls and answers derived from a response template (vLLM
 * {@code response_template_grammar.py}): xgrammar format trees ({@code sequence}, {@code or}, {@code tag},
 * {@code regex}, {@code json_schema}, {@code any_text}, ...) that hold the output to the template's framing — the
 * reasoning region, then calls in the tool-call field's format with their arguments
 * ({@link TemplateArguments}), or an answer in the content field. The grammar engine compiles them
 * ({@code ConstraintSpec.structuralTag}). A format is a map of the xgrammar JSON; a derivation that the template cannot
 * spell returns null. Stateless.
 */
final class TemplateGrammar {

    /** Any text: the {@code any_text} format with no excludes. Compared by identity where vLLM does. */
    static final Map<String, Object> ANY_TEXT = Collections.unmodifiableMap(anyText(Collections.<String>emptyList()));
    /** At most a newline between parameters; unbounded whitespace would let a model write it until max_tokens. */
    static final Map<String, Object> SEPARATOR = regex("\\n?");

    private static final Pattern REGEX_META = Pattern.compile("[.^$*+?{}\\[\\]\\\\|()]");
    private static final Pattern REPEATED_CLASS = Pattern.compile("\\\\([wS])\\+|\\[\\^((?:\\\\.|[^\\]\\\\])+)\\]\\+");

    private TemplateGrammar() {
    }

    // ================================================================= format builders

    static Map<String, Object> regex(String pattern) {
        return format("regex", "pattern", pattern);
    }

    static Map<String, Object> anyText(List<String> excludes) {
        Map<String, Object> f = format("any_text", "excludes", new ArrayList<>(excludes));
        if (excludes.isEmpty()) f.remove("excludes");
        return f;
    }

    static Map<String, Object> constString(String value) {
        return format("const_string", "value", value);
    }

    static Map<String, Object> constStrings(List<String> values) {
        List<Map<String, Object>> out = new ArrayList<>();
        for (String v : values) out.add(constString(v));
        return oneOf(out);
    }

    static Map<String, Object> sequence(List<Map<String, Object>> elements) {
        return format("sequence", "elements", new ArrayList<>(elements));
    }

    @SafeVarargs
    static Map<String, Object> sequence(Map<String, Object>... elements) {
        return sequence(Arrays.asList(elements));
    }

    static Map<String, Object> oneOf(List<Map<String, Object>> formats) {
        return formats.size() == 1 ? formats.get(0) : format("or", "elements", new ArrayList<>(formats));
    }

    static Map<String, Object> optional(Map<String, Object> content) {
        return format("optional", "content", content);
    }

    static Map<String, Object> star(Map<String, Object> content) {
        return format("star", "content", content);
    }

    static Map<String, Object> tag(String begin, Map<String, Object> content, Object end) {
        Map<String, Object> f = format("tag", "begin", begin);
        f.put("content", content);
        f.put("end", end instanceof List ? new ArrayList<>((List<?>) end) : end);
        return f;
    }

    private static Map<String, Object> format(String type, String key, Object value) {
        Map<String, Object> f = new LinkedHashMap<>();
        f.put("type", type);
        f.put(key, value);
        return f;
    }

    /** The whole structural tag of a format. */
    private static Map<String, Object> structuralTag(Map<String, Object> format) {
        return format("structural_tag", "format", format);
    }

    // ================================================================= delimiters

    /** The strings the assistant message starts after, if literal (vLLM {@code message_start_anchors}). */
    static List<String> messageStartAnchors(ResponseTemplate template) {
        List<String> anchors = template.messageStartAnchors();
        return anchors == null ? Collections.<String>emptyList() : anchors;
    }

    /** Literal closers, empty when the field has none, or null when underivable (vLLM {@code _closers}). */
    static List<String> closers(ResponseTemplate.Field field) {
        if (field.close() == null) return Collections.emptyList();
        return field.closeLiterals() != null ? field.closeLiterals()
                : LiteralPatterns.expand(field.close().regex().source());
    }

    /** The field's openers, with continuations of the message-start anchors (vLLM {@code _openers}). */
    static List<String> openers(ResponseTemplate.Field field, List<String> messageStart) {
        List<String> openers = field.openLiterals() != null || field.open() == null ? field.openLiterals()
                : LiteralPatterns.expand(field.open().regex().source(), messageStart != null, false);
        return continuing(openers, messageStart, field.open() == null ? null : field.open().regex());
    }

    /**
     * {@code literals} and the rest of each one past a message-start anchor it starts with, where the parser finds the
     * opener in that rest (vLLM {@code _continuing}).
     */
    static List<String> continuing(List<String> literals, List<String> anchors, TemplateRegex opener) {
        if (literals == null || literals.isEmpty() || anchors == null || anchors.isEmpty()) return literals;
        LinkedHashSet<String> out = new LinkedHashSet<>(literals);
        for (String literal : literals) {
            for (String anchor : anchors) {
                if (literal.length() <= anchor.length() || !literal.startsWith(anchor)) continue;
                String rest = literal.substring(anchor.length());
                for (int i = 0; i < rest.length(); i++) {
                    if (opener.fullMatchesFrom(rest, i)) {
                        out.add(rest);
                        break;
                    }
                }
            }
        }
        return new ArrayList<>(out);
    }

    /** {@code text} with every regex metacharacter escaped (vLLM {@code _escape}). */
    static String escape(String text) {
        return REGEX_META.matcher(text).replaceAll("\\\\$0");
    }

    /** One character inside a regex class (vLLM {@code _class_escape}). */
    static String classEscape(char c) {
        switch (c) {
            case '\t': return "\\t";
            case '\n': return "\\n";
            case '\r': return "\\r";
            default: return "\\]^-[".indexOf(c) >= 0 ? "\\" + c : String.valueOf(c);
        }
    }

    /** A named group's prefix, body and suffix in {@code pattern}, or null (vLLM {@code _split_group}). */
    static String[] splitGroup(String pattern, String name) {
        String opener = "(?P<" + name + ">";
        int start = pattern.indexOf(opener);
        if (start < 0) return null;
        int depth = 0;
        boolean inClass = false;
        int i = start;
        while (i < pattern.length()) {
            char c = pattern.charAt(i);
            if (c == '\\') {
                i += 2;
                continue;
            }
            if (inClass) {
                inClass = c != ']';
            } else if (c == '[') {
                inClass = true;
            } else if (c == '(') {
                depth++;
            } else if (c == ')') {
                depth--;
                if (depth == 0) {
                    return new String[]{pattern.substring(0, start), pattern.substring(start + opener.length(), i),
                            pattern.substring(i + 1)};
                }
            }
            i++;
        }
        return null;
    }

    /** The opener regex of a call to {@code name}, its prefix continued past the message start (vLLM {@code _call_opener}). */
    static String callOpener(String[] opener, String name, List<String> messageStart) {
        String before = opener[0];
        if (messageStart != null && !before.isEmpty()) {
            List<String> literals = LiteralPatterns.expand(before, true, false);
            // text before a name inside a group is no pattern of its own
            if (literals != null && !literals.isEmpty()) {
                literals = continuing(literals, messageStart, TemplateRegex.compile(before, "call opener"));
                List<String> escaped = new ArrayList<>();
                for (String l : literals) escaped.add(escape(l));
                before = "(?:" + String.join("|", escaped) + ")";
            }
        }
        return before + name + opener[2];
    }

    /**
     * A regex for the names {@code pattern} matches besides {@code names}, or null when {@code pattern} is no
     * negatable character class repeated (vLLM {@code _other_names}).
     */
    static String otherNames(String pattern, List<String> names) {
        if (names.isEmpty()) return pattern;
        Matcher m = REPEATED_CLASS.matcher(pattern);
        if (!m.matches()) return null;
        String negated = m.group(1) != null ? (m.group(1).equals("w") ? "\\W" : "\\s") : m.group(2);
        Trie root = new Trie();
        for (String name : names) {
            Trie node = root;
            for (char c : name.toCharArray()) node = node.children.computeIfAbsent(c, k -> new Trie());
            node.end = true;
        }
        return rest(root, true, negated);
    }

    /** One trie node of {@link #otherNames}: its children in insertion order, and whether a name ends here. */
    private static final class Trie {
        final Map<Character, Trie> children = new LinkedHashMap<>();
        boolean end;
    }

    /** The rest of a name past {@code node} that ends on none of the names. */
    private static String rest(Trie node, boolean first, String negated) {
        List<String> branches = new ArrayList<>();
        branches.add(other(node.children.keySet(), negated) + other(Collections.<Character>emptySet(), negated) + "*");
        for (Map.Entry<Character, Trie> e : node.children.entrySet()) {
            branches.add(escape(String.valueOf(e.getKey())) + rest(e.getValue(), false, negated));
        }
        return "(?:" + String.join("|", branches) + ")" + (first || node.end ? "" : "?");
    }

    private static String other(Set<Character> chars, String negated) {
        StringBuilder out = new StringBuilder("[^").append(negated);
        for (char c : new TreeSet<>(chars)) out.append(classEscape(c));
        return out.append(']').toString();
    }

    // ================================================================= reasoning and answers

    /** The reasoning region, holding at least one character when {@code nonempty} (vLLM {@code _thinking_format}). */
    static Map<String, Object> thinkingFormat(ResponseTemplate.Field thinking, List<String> messageStart,
                                              boolean nonempty) {
        List<String> openers = openers(thinking, messageStart);
        List<String> closers = closers(thinking);
        if (openers == null || openers.isEmpty() || closers == null || closers.isEmpty()) return null;
        Map<String, Object> text = nonempty ? sequence(nonemptyStart(closers), ANY_TEXT) : ANY_TEXT;
        List<Map<String, Object>> tags = new ArrayList<>();
        for (String opener : openers) tags.add(tag(opener, text, closers));
        return oneOf(tags);
    }

    /**
     * The start of text that is not empty and does not begin with one of the closers (vLLM {@code _nonempty_start}):
     * each branch leaves every closer's spelling on a character that starts no closer.
     */
    static Map<String, Object> nonemptyStart(List<String> closers) {
        TreeSet<Character> starts = new TreeSet<>();
        LinkedHashSet<String> prefixes = new LinkedHashSet<>();
        prefixes.add("");
        for (String closer : closers) {
            if (closer.isEmpty()) continue;
            starts.add(closer.charAt(0));
            for (int end = 1; end < closer.length(); end++) prefixes.add(closer.substring(0, end));
        }
        List<String> branches = new ArrayList<>();
        for (String prefix : prefixes) {
            boolean closed = false;
            TreeSet<Character> excluded = new TreeSet<>(starts);
            for (String closer : closers) {
                if (closer.isEmpty()) continue;
                closed |= prefix.contains(closer);
                if (closer.length() > prefix.length() && closer.startsWith(prefix)) {
                    excluded.add(closer.charAt(prefix.length()));
                }
            }
            if (closed) continue;
            StringBuilder b = new StringBuilder(escape(prefix)).append("[^");
            for (char c : excluded) b.append(classEscape(c));
            branches.add(b.append(']').toString());
        }
        return regex(String.join("|", branches));
    }

    /** The answer field holding {@code content}, or free text without it; null when underivable (vLLM {@code _reply_format}). */
    static Map<String, Object> replyFormat(ResponseTemplate.Field answer, List<String> messageStart,
                                           Map<String, Object> content) {
        List<String> openers = openers(answer, messageStart);
        List<String> closers = closers(answer);
        if (openers == null || openers.isEmpty() || closers == null) return null;
        List<Map<String, Object>> elements = new ArrayList<>();
        elements.add(constStrings(openers));
        elements.addAll(answerText(closers, content));
        return sequence(elements);
    }

    /** An answer's text, which cannot close the answer early, then optionally its closer (vLLM {@code _answer_text}). */
    static List<Map<String, Object>> answerText(List<String> closers, Map<String, Object> content) {
        List<Map<String, Object>> elements = new ArrayList<>();
        elements.add(content != null ? content : anyText(closers));
        if (!closers.isEmpty()) elements.add(optional(constStrings(closers)));
        return elements;
    }

    /** The rest of an answer that is already open, or null when underivable (vLLM {@code answer_rest_structural_tag}). */
    static Map<String, Object> answerRestTag(ResponseTemplate.Field answer) {
        List<String> closers = closers(answer);
        return closers == null ? null : structuralTag(sequence(answerText(closers, null)));
    }

    /** Whether the field's opener names the function its value calls (vLLM {@code _names_function_by_opener}). */
    static boolean namesFunctionByOpener(ResponseTemplate.Field field) {
        if (field.open() == null || field.open().literals() != null || field.transformEach()
                || !field.open().regex().groupNames().contains("name") || field.transform() == null) {
            return false;
        }
        Object t = field.transform().template();
        Object function = t instanceof Map ? ((Map<?, ?>) t).get("function") : null;
        return function instanceof Map && "{name}".equals(((Map<?, ?>) function).get("name"))
                && "{content}".equals(((Map<?, ?>) function).get("arguments"));
    }

    // ================================================================= tool calls

    /** The options of {@link #toolCallTag}. */
    static final class CallOptions {
        boolean parallelToolCalls = true;
        ResponseTemplate.Field thinking;
        ResponseTemplate.Field answer;
        Map<String, Object> answerFormat;
        List<String> messageStart;
        boolean requiresReasoning;
        boolean sortedArguments;
        boolean strictArguments;
    }

    /**
     * Native calls to {@code functions} in the format of the tool-call {@code field}, respecting strict arguments,
     * the call count and the reasoning (vLLM {@code tool_call_structural_tag}): the whole structural tag, or null when
     * the template's format cannot be derived. An answer field lets the model answer instead, in the answer format if
     * given; without functions only the answer is derived. When the model may answer, calls may also name any other
     * function the template can spell, with any arguments. At the message start the tag covers the output from its
     * first token, opening with non-empty reasoning when {@code requiresReasoning}.
     *
     * @throws IllegalArgumentException for a tool or parameter name the format cannot encode
     */
    static Map<String, Object> toolCallTag(ResponseTemplate.Field field, List<ChatTool> functions, CallOptions o) {
        if (functions.isEmpty() && o.answer != null) {
            return frame(start -> {
                Map<String, Object> reply = replyFormat(o.answer, start, o.answerFormat);
                return reply == null ? null : new ArrayList<>(Collections.singletonList(reply));
            }, o);
        }
        if (functions.isEmpty() || field == null || !namesFunctionByOpener(field)) return null;
        String[] opener = splitGroup(field.open().regex().source(), "name");
        List<String> closerLiterals = closers(field);
        if (opener == null || closerLiterals == null) return null;
        List<Map<String, Object>> arguments = new ArrayList<>();
        List<Boolean> enforced = new ArrayList<>();
        for (ChatTool function : functions) {
            Map<String, Object> parameters = parameters(function);
            boolean strict = Boolean.TRUE.equals(function.strict());
            Map<String, Object> args = null;
            if (strict || o.strictArguments) {
                try {
                    args = TemplateArguments.format(field, parameters, true, o.sortedArguments);
                } catch (IllegalArgumentException e) {
                    if (strict) throw e;
                }
            }
            enforced.add(args != null);
            if (args == null && !strict) args = TemplateArguments.format(field, parameters, false, false);
            if (args == null) return null;
            arguments.add(args);
        }
        TemplateRegex namePattern = TemplateRegex.compile(opener[1], "tool name pattern");
        List<String> names = new ArrayList<>();
        for (ChatTool function : functions) {
            if (!namePattern.fullMatches(function.name())) {
                throw new IllegalArgumentException("tool name '" + function.name() + "' cannot be encoded in tool calls");
            }
            names.add(escape(function.name()));
        }
        if (o.answer != null) {
            List<String> strictNames = new ArrayList<>();
            for (int i = 0; i < functions.size(); i++) {
                if (enforced.get(i)) strictNames.add(functions.get(i).name());
            }
            String others = otherNames(opener[1], strictNames);
            Map<String, Object> loose = TemplateArguments.format(field, Collections.<String, Object>emptyMap(), false,
                    false);
            if (others != null && loose != null) {
                List<String> keptNames = new ArrayList<>();
                List<Map<String, Object>> keptArguments = new ArrayList<>();
                for (int i = 0; i < names.size(); i++) {
                    if (!enforced.get(i)) continue;
                    keptNames.add(names.get(i));
                    keptArguments.add(arguments.get(i));
                }
                keptNames.add("(?:" + others + ")");
                keptArguments.add(loose);
                names = keptNames;
                arguments = keptArguments;
            }
        }
        Map<String, Object> closers = closerLiterals.isEmpty() ? null : constStrings(closerLiterals);
        List<String> callNames = names;
        List<Map<String, Object>> callArguments = arguments;
        Function<List<String>, Map<String, Object>> call = start -> {
            List<Map<String, Object>> calls = new ArrayList<>();
            for (int i = 0; i < callNames.size(); i++) {
                calls.add(sequence(regex(callOpener(opener, callNames.get(i), start)), callArguments.get(i)));
            }
            return oneOf(calls);
        };
        return frame(start -> {
            List<Map<String, Object>> elements = new ArrayList<>();
            elements.add(call.apply(start));
            if (o.parallelToolCalls && field.repeats()) {
                Map<String, Object> next = call.apply(null);
                elements.add(star(closers != null ? sequence(closers, next) : next));
            }
            // the end of generation can end a call without its closer
            if (closers != null) elements.add(optional(closers));
            if (o.answer != null) {
                Map<String, Object> reply = replyFormat(o.answer, start, o.answerFormat);
                if (reply == null) return null;
                elements = new ArrayList<>(Collections.singletonList(oneOf(Arrays.asList(sequence(elements), reply))));
            }
            return elements;
        }, o);
    }

    /** A function's parameter schema; one that declares none takes no arguments. */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> parameters(ChatTool function) {
        Map<String, Object> p = function.parameters();
        if (p == null || p.isEmpty()) {
            Map<String, Object> none = new LinkedHashMap<>();
            none.put("type", "object");
            none.put("properties", new LinkedHashMap<String, Object>());
            return none;
        }
        return p;
    }

    /**
     * Frames the output after the reasoning, from the first token at the message start (vLLM {@code _structural_tag}):
     * the reasoning is optional unless {@code requiresReasoning} opens the message with non-empty reasoning.
     */
    private static Map<String, Object> frame(Function<List<String>, List<Map<String, Object>>> output, CallOptions o) {
        List<Map<String, Object>> rest = output.apply(null);
        if (rest == null) return null;
        Map<String, Object> think = null;
        if (o.thinking != null) {
            think = thinkingFormat(o.thinking, o.messageStart, o.requiresReasoning);
            if (think == null) return null;
        }
        if (o.messageStart == null) {
            if (think != null) rest.add(0, optional(think));
            return structuralTag(sequence(rest));
        }
        List<Map<String, Object>> afterThinking = new ArrayList<>();
        if (think != null) afterThinking.add(think);
        afterThinking.addAll(rest);
        if (think != null && o.requiresReasoning) return structuralTag(sequence(afterThinking));
        List<Map<String, Object>> start = output.apply(o.messageStart);
        if (start == null) return null;
        Map<String, Object> format = sequence(start);
        if (think != null) format = oneOf(Arrays.asList(sequence(afterThinking), format));
        return structuralTag(Objects.requireNonNull(format));
    }
}
