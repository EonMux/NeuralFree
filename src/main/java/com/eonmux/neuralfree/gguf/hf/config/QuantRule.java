package com.eonmux.neuralfree.gguf.hf.config;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;

import java.util.Locale;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * One normalized quantization rule: the HF modules its {@link ModuleMatcher} selects are stored as its
 * {@link QuantArgs}, or, for an exclusion rule ({@link #excludes()}), are not quantized. Rules are tried in order;
 * the first match decides. Immutable.
 */
public final class QuantRule {

    private final ModuleMatcher matcher;
    private final QuantArgs args;

    private QuantRule(ModuleMatcher matcher, QuantArgs args) {
        this.matcher = Objects.requireNonNull(matcher, "matcher");
        this.args = args;
    }

    /** A rule storing the matched modules as {@code args}. */
    public static QuantRule of(ModuleMatcher matcher, QuantArgs args) {
        return new QuantRule(matcher, Objects.requireNonNull(args, "args"));
    }

    /** A rule leaving the matched modules unquantized. */
    public static QuantRule exclude(ModuleMatcher matcher) {
        return new QuantRule(matcher, null);
    }

    public ModuleMatcher matcher() {
        return matcher;
    }

    /** The arguments of the matched modules; null for an exclusion rule. */
    public QuantArgs args() {
        return args;
    }

    /** Whether the matched modules are not quantized. */
    public boolean excludes() {
        return args == null;
    }

    /** Whether the rule selects HF module {@code module} (for example {@code model.layers.3.self_attn.q_proj}). */
    public boolean matches(String module) {
        return matcher.matches(module);
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof QuantRule && matcher.equals(((QuantRule) o).matcher)
                && Objects.equals(args, ((QuantRule) o).args);
    }

    @Override
    public int hashCode() {
        return Objects.hash(matcher, args);
    }

    @Override
    public String toString() {
        return matcher + " -> " + (args == null ? "unquantized" : args);
    }

    /**
     * Selects HF module names. Regular expressions use Python {@code re} semantics: {@link Kind#REGEX} is
     * {@code re.match} (anchored at the start only) and {@link Kind#REGEX_SEARCH} is {@code re.search}; Python
     * named groups {@code (?P<n>...)} and back-references {@code (?P=n)} are accepted. {@link Kind#GLOB} is
     * {@code fnmatch.fnmatchcase}.
     */
    public static final class ModuleMatcher {

        /** How the pattern is applied. */
        public enum Kind {
            /** Equal to the pattern. */
            EXACT,
            /** {@code re.match}: the pattern matches a prefix of the name. */
            REGEX,
            /** {@code re.search}: the pattern matches anywhere in the name. */
            REGEX_SEARCH,
            /** {@code fnmatch}: shell wildcards over the whole name. */
            GLOB,
            /** The name equals the pattern or ends with {@code "." + pattern}. */
            SUFFIX,
            /** The pattern is contained in the name. */
            SUBSTRING,
            /** The pattern is one of the dotted components of the name, or a dotted prefix of it. */
            COMPONENT,
            /** The name equals the pattern or starts with {@code pattern + "."}. */
            PREFIX,
            /**
             * transformers {@code should_convert_module}: the pattern, as a Python regex, matches from the start of the
             * name up to a module boundary, or the name ends with the pattern.
             */
            MODULE_PATTERN,
            /** A module class name; {@code Linear} selects every quantizable module, other classes none. */
            CLASS
        }

        private final Kind kind;
        private final String pattern;
        private final Pattern regex;

        private ModuleMatcher(Kind kind, String pattern, Pattern regex) {
            this.kind = kind;
            this.pattern = pattern;
            this.regex = regex;
        }

        public static ModuleMatcher exact(String name) {
            return new ModuleMatcher(Kind.EXACT, Objects.requireNonNull(name, "name"), null);
        }

        /**
         * A Python {@code re.match} pattern.
         *
         * @throws GgufFormatException naming the pattern when it does not compile
         */
        public static ModuleMatcher regex(String pythonPattern) {
            return new ModuleMatcher(Kind.REGEX, pythonPattern, compile(pythonPattern));
        }

        /** A Python {@code re.search} pattern. */
        public static ModuleMatcher search(String pythonPattern) {
            return new ModuleMatcher(Kind.REGEX_SEARCH, pythonPattern, compile(pythonPattern));
        }

        /** An {@code fnmatch} pattern. */
        public static ModuleMatcher glob(String glob) {
            return new ModuleMatcher(Kind.GLOB, glob, Pattern.compile(globToRegex(glob), Pattern.DOTALL));
        }

        public static ModuleMatcher suffix(String name) {
            return new ModuleMatcher(Kind.SUFFIX, Objects.requireNonNull(name, "name"), null);
        }

        public static ModuleMatcher substring(String part) {
            return new ModuleMatcher(Kind.SUBSTRING, Objects.requireNonNull(part, "part"), null);
        }

        public static ModuleMatcher component(String name) {
            return new ModuleMatcher(Kind.COMPONENT, Objects.requireNonNull(name, "name"), null);
        }

        public static ModuleMatcher prefix(String name) {
            return new ModuleMatcher(Kind.PREFIX, Objects.requireNonNull(name, "name"), null);
        }

        /**
         * A transformers {@code modules_to_not_convert} entry ({@link Kind#MODULE_PATTERN}).
         *
         * @throws GgufFormatException naming the pattern when it does not compile
         */
        public static ModuleMatcher modulePattern(String pattern) {
            Objects.requireNonNull(pattern, "pattern");
            return new ModuleMatcher(Kind.MODULE_PATTERN, pattern,
                    compile("(?:" + pattern + ")(?:(?<=\\.)|(?=\\.|$))"));
        }

        public static ModuleMatcher className(String className) {
            return new ModuleMatcher(Kind.CLASS, Objects.requireNonNull(className, "className"), null);
        }

        /** Every module ({@code Linear}). */
        public static ModuleMatcher anyLinear() {
            return className("Linear");
        }

        /**
         * A compressed-tensors / Quark style target: {@code re:<regex>} is {@code re.match}, a dotted or lower-case
         * name is exact, and anything else is a class name.
         */
        public static ModuleMatcher target(String target) {
            Objects.requireNonNull(target, "target");
            if (target.startsWith("re:")) {
                return regex(target.substring(3));
            }
            boolean looksLikeClass = !target.isEmpty() && Character.isUpperCase(target.charAt(0))
                    && target.indexOf('.') < 0;
            return looksLikeClass ? className(target) : exact(target);
        }

        /** An ignore-list entry: {@code re:<regex>} is {@code re.match}, anything else is exact. */
        public static ModuleMatcher exactOrRegex(String entry) {
            Objects.requireNonNull(entry, "entry");
            return entry.startsWith("re:") ? regex(entry.substring(3)) : exact(entry);
        }

        public Kind kind() {
            return kind;
        }

        public String pattern() {
            return pattern;
        }

        /** Whether this matcher selects module {@code module}. */
        public boolean matches(String module) {
            switch (kind) {
                case EXACT: return module.equals(pattern);
                case REGEX: return regex.matcher(module).lookingAt();
                case REGEX_SEARCH: return regex.matcher(module).find();
                case GLOB: return regex.matcher(module).matches();
                case SUFFIX: return module.equals(pattern) || module.endsWith("." + pattern);
                case SUBSTRING: return module.contains(pattern);
                case COMPONENT: return module.equals(pattern) || module.startsWith(pattern + ".")
                        || ("." + module + ".").contains("." + pattern + ".");
                case PREFIX: return module.equals(pattern) || module.startsWith(pattern + ".");
                case MODULE_PATTERN: return regex.matcher(module).lookingAt() || module.endsWith(pattern);
                case CLASS: return pattern.equals("Linear");
                default: throw new IllegalStateException("unknown matcher kind " + kind);
            }
        }

        /** {@code text} with every character other than an ASCII letter, digit or underscore escaped (re.escape). */
        public static String escape(String text) {
            StringBuilder out = new StringBuilder(text.length() * 2);
            for (int i = 0; i < text.length(); i++) {
                char c = text.charAt(i);
                boolean plain = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
                if (!plain && c < 0x80) {
                    out.append('\\');
                }
                out.append(c);
            }
            return out.toString();
        }

        private static Pattern compile(String pythonPattern) {
            Objects.requireNonNull(pythonPattern, "pattern");
            String java = pythonPattern.replace("(?P<", "(?<");
            Matcher backRef = Pattern.compile("\\(\\?P=(\\w+)\\)").matcher(java);
            java = backRef.replaceAll("\\\\k<$1>");
            try {
                return Pattern.compile(java);
            } catch (PatternSyntaxException e) {
                throw new GgufFormatException("invalid module regex '" + pythonPattern + "': " + e.getDescription(), e);
            }
        }

        /** Python fnmatch.translate: {@code *}, {@code ?}, {@code [seq]} and {@code [!seq]}. */
        static String globToRegex(String glob) {
            StringBuilder out = new StringBuilder();
            int i = 0;
            int n = glob.length();
            while (i < n) {
                char c = glob.charAt(i++);
                if (c == '*') {
                    out.append(".*");
                } else if (c == '?') {
                    out.append('.');
                } else if (c == '[') {
                    int j = i;
                    if (j < n && glob.charAt(j) == '!') j++;
                    if (j < n && glob.charAt(j) == ']') j++;
                    while (j < n && glob.charAt(j) != ']') j++;
                    if (j >= n) {
                        out.append("\\[");
                    } else {
                        String stuff = glob.substring(i, j).replace("\\", "\\\\").replace("[", "\\[")
                                .replace("&&", "\\&\\&");
                        i = j + 1;
                        if (stuff.startsWith("!")) {
                            stuff = "^" + stuff.substring(1);
                        } else if (stuff.startsWith("^")) {
                            stuff = "\\" + stuff;
                        }
                        out.append('[').append(stuff).append(']');
                    }
                } else {
                    out.append(Pattern.quote(String.valueOf(c)));
                }
            }
            return out.toString();
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof ModuleMatcher && kind == ((ModuleMatcher) o).kind
                    && pattern.equals(((ModuleMatcher) o).pattern);
        }

        @Override
        public int hashCode() {
            return Objects.hash(kind, pattern);
        }

        @Override
        public String toString() {
            return kind.name().toLowerCase(Locale.ROOT) + ":" + pattern;
        }
    }
}
