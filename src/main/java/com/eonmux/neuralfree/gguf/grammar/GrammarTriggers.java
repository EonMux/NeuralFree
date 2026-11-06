package com.eonmux.neuralfree.gguf.grammar;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * The triggers that gate a grammar: generation runs unconstrained until one fires, then the grammar constrains the
 * output from the trigger on. The trigger kinds are llama.cpp's {@code common_grammar_trigger} types
 * (common/sampling.cpp:224-260): a literal {@linkplain Builder#word word}, a {@linkplain Builder#pattern pattern} searched anywhere
 * in the output, a {@linkplain Builder#patternFull full pattern} the whole output must match, and a {@linkplain Builder#token token}
 * id. A word and a pattern start the grammar where the match starts; a pattern with capturing groups starts it at
 * the first non-empty group ({@code llama_grammar_trigger_pattern::find}).
 *
 * <p>The {@link Mode} says what happens once the grammar completes: {@link Mode#LAZY} is llama.cpp's lazy grammar
 * ({@code grammar_lazy}), which stays in force to the end of the generation, and {@link Mode#DISPATCH} is xgrammar's
 * tag dispatch (the vLLM structural tag with {@code triggers}), which returns to free text after each completed
 * structure and waits for the next trigger.
 *
 * <p>Immutable; the lists are copied in and handed out unmodifiable.
 */
public final class GrammarTriggers {

    /** What the grammar does after it completes. */
    public enum Mode {
        /** Constrained from the first trigger to the end of the generation (llama.cpp {@code grammar_lazy}). */
        LAZY,
        /** Back to free text after each completed grammar sentence, waiting for the next trigger (xgrammar). */
        DISPATCH
    }

    /** The llama.cpp trigger types ({@code common_grammar_trigger_type}). */
    public enum Kind { WORD, PATTERN, PATTERN_FULL, TOKEN }

    /** One trigger: the regex it searches with, or the token id it fires on. */
    public static final class Trigger {
        private final Kind kind;
        private final String source;
        private final Pattern regex;
        private final int token;

        private Trigger(Kind kind, String source, Pattern regex, int token) {
            this.kind = kind;
            this.source = source;
            this.regex = regex;
            this.token = token;
        }

        public Kind kind() {
            return kind;
        }

        /** The word or pattern text, or the token id as text. */
        public String source() {
            return source;
        }

        /** The token id of a token trigger, or -1 for a text trigger. */
        public int token() {
            return token;
        }

        /**
         * Where the grammar starts in {@code text} when this text trigger fires, or -1 when it does not: the first
         * non-empty capturing group of the match, else the match start ({@code llama_grammar_trigger_pattern::find}).
         */
        public int find(CharSequence text) {
            if (regex == null) {
                return -1;
            }
            Matcher m = regex.matcher(text);
            boolean found = kind == Kind.PATTERN_FULL ? m.matches() : m.find();
            if (!found) {
                return -1;
            }
            for (int g = 1; g <= m.groupCount(); g++) {
                if (m.start(g) >= 0 && m.end(g) > m.start(g)) {
                    return m.start(g);
                }
            }
            return m.start();
        }

        @Override
        public String toString() {
            return kind.name().toLowerCase() + "(" + source + ")";
        }
    }

    private final List<Trigger> triggers;
    private final Mode mode;

    private GrammarTriggers(List<Trigger> triggers, Mode mode) {
        this.triggers = Collections.unmodifiableList(new ArrayList<>(triggers));
        this.mode = mode;
    }

    /** A new builder in {@code mode}. */
    public static Builder builder(Mode mode) {
        return new Builder(Objects.requireNonNull(mode, "mode"));
    }

    /** {@link Mode#LAZY} over literal words, the trigger set llama.cpp's tool-call grammars carry most often. */
    public static GrammarTriggers lazyWords(List<String> words) {
        Builder b = builder(Mode.LAZY);
        for (String w : words) {
            b.word(w);
        }
        return b.build();
    }

    /** The triggers, text and token alike, in insertion order. */
    public List<Trigger> triggers() {
        return triggers;
    }

    /** The token ids among the triggers. */
    public int[] tokens() {
        return triggers.stream().filter(t -> t.kind == Kind.TOKEN).mapToInt(t -> t.token).toArray();
    }

    /** The literal words among the triggers (the ones a sampler can see coming token by token). */
    public List<String> words() {
        List<String> out = new ArrayList<>();
        for (Trigger t : triggers) {
            if (t.kind == Kind.WORD) {
                out.add(t.source);
            }
        }
        return out;
    }

    public Mode mode() {
        return mode;
    }

    /** The trigger whose match starts earliest in {@code text}, as its start index, or -1 when none fires. */
    public int firstStart(CharSequence text) {
        int best = -1;
        for (Trigger t : triggers) {
            int at = t.find(text);
            if (at >= 0 && (best < 0 || at < best)) {
                best = at;
            }
        }
        return best;
    }

    /** Whether {@code token} is a token trigger. */
    public boolean firesOn(int token) {
        for (Trigger t : triggers) {
            if (t.kind == Kind.TOKEN && t.token == token) {
                return true;
            }
        }
        return false;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof GrammarTriggers)) {
            return false;
        }
        GrammarTriggers that = (GrammarTriggers) o;
        if (mode != that.mode || triggers.size() != that.triggers.size()) {
            return false;
        }
        for (int i = 0; i < triggers.size(); i++) {
            Trigger a = triggers.get(i);
            Trigger b = that.triggers.get(i);
            if (a.kind != b.kind || !a.source.equals(b.source)) {
                return false;
            }
        }
        return true;
    }

    @Override
    public int hashCode() {
        return Objects.hash(mode, triggers.toString());
    }

    @Override
    public String toString() {
        return "GrammarTriggers(" + mode + ", " + triggers + ")";
    }

    /** Builder of an immutable {@link GrammarTriggers}; at least one trigger is required. */
    public static final class Builder {
        private final Mode mode;
        private final List<Trigger> triggers = new ArrayList<>();

        private Builder(Mode mode) {
            this.mode = mode;
        }

        /** A literal word: fires where the word first appears (llama.cpp {@code TRIGGER_TYPE_WORD}). */
        public Builder word(String word) {
            if (word == null || word.isEmpty()) {
                throw new IllegalArgumentException("trigger word is null or empty");
            }
            triggers.add(new Trigger(Kind.WORD, word, Pattern.compile(Pattern.quote(word)), -1));
            return this;
        }

        /** A regex searched anywhere in the output (llama.cpp {@code TRIGGER_TYPE_PATTERN}). */
        public Builder pattern(String regex) {
            triggers.add(new Trigger(Kind.PATTERN, regex, compile(regex), -1));
            return this;
        }

        /** A regex the whole output so far must match (llama.cpp {@code TRIGGER_TYPE_PATTERN_FULL}). */
        public Builder patternFull(String regex) {
            triggers.add(new Trigger(Kind.PATTERN_FULL, regex, compile(regex), -1));
            return this;
        }

        /** A token id that fires when sampled (llama.cpp {@code TRIGGER_TYPE_TOKEN}). */
        public Builder token(int id) {
            if (id < 0) {
                throw new IllegalArgumentException("trigger token id " + id + " is negative");
            }
            triggers.add(new Trigger(Kind.TOKEN, Integer.toString(id), null, id));
            return this;
        }

        /** @throws IllegalArgumentException when no trigger was added */
        public GrammarTriggers build() {
            if (triggers.isEmpty()) {
                throw new IllegalArgumentException("a trigger-gated grammar needs at least one trigger");
            }
            return new GrammarTriggers(triggers, mode);
        }

        private static Pattern compile(String regex) {
            if (regex == null || regex.isEmpty()) {
                throw new IllegalArgumentException("trigger pattern is null or empty");
            }
            try {
                return Pattern.compile(regex);
            } catch (PatternSyntaxException e) {
                throw new IllegalArgumentException("trigger pattern '" + regex + "' does not compile: "
                        + e.getDescription(), e);
            }
        }
    }
}
