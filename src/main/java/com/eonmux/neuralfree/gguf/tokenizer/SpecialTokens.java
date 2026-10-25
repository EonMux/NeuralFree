package com.eonmux.neuralfree.gguf.tokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Special-token ids, effective token attributes, the end-of-generation set and the
 * special-token partition of raw text, following {@code llama_vocab::impl::load}
 * (src/llama-vocab.cpp:2196-3424) and {@code tokenizer_st_partition} (3547-3663).
 *
 * <p>Resolution order, as in llama.cpp:</p>
 * <ol>
 *   <li>Per-vocabulary-type default ids (2206-2408); ChatGLM/GLM pre-tokenizers clear BOS.</li>
 *   <li>Defaults out of range are disabled, then in-range {@code tokenizer.ggml.*_token_id}
 *       overrides apply, the deprecated prefix/suffix/middle keys last (2868-2912).</li>
 *   <li>Unset EOT, EOM and FIM ids are detected from token text; detected tokens become CONTROL
 *       (2972-3132). Tokens are scanned in id order, so the lowest matching id wins.</li>
 *   <li>CONTROL tokens whose text contains "unused" become UNUSED (3136-3157).</li>
 *   <li>The EOG set: FIM pad/repo/sep, the stop texts (made CONTROL), then EOS, EOT and EOM;
 *       the harmony markers become USER_DEFINED; {@code <|end|>} leaves the set when
 *       {@code <|return|>} and {@code <|call|>} (or {@code <|calls|>} and {@code <|flush|>}) are
 *       present, and {@code </s>} leaves it when the vocabulary marks it as a normal token
 *       (3162-3313). The vocabulary's extra end-of-generation ids
 *       ({@link TokenizerVocab#extraEogIds()}) join last.</li>
 *   <li>The partition order: CONTROL, USER_DEFINED and UNKNOWN tokens, longest UTF-8 text first
 *       (3316-3330); then the LSTRIP/RSTRIP attributes by pre-tokenizer, architecture and model
 *       name (3348-3424).</li>
 * </ol>
 * A NONE vocabulary has no special tokens: llama.cpp returns from {@code load} before any of this.
 *
 * <p>The Hugging Face added-token attributes of the vocabulary overlay take part in the
 * partition: {@link TokenizerVocab#ATTR_SINGLE_WORD} tokens match only between non-word
 * characters of the text being split, and {@link TokenizerVocab#ATTR_NORMALIZED} tokens match
 * the normalized text, after the other tokens have been split out. The post-processor ids of the
 * vocabulary replace BOS and EOS around encoded text ({@link #leadingIds()},
 * {@link #trailingIds()}).</p>
 */
public final class SpecialTokens {

    /** {@code LLAMA_TOKEN_NULL}. */
    public static final int NULL = -1;

    private static final int HIDDEN_ATTRS = TokenizerVocab.ATTR_CONTROL | TokenizerVocab.ATTR_UNKNOWN;
    private static final int[] NO_IDS = new int[0];

    private final TokenizerVocab vocab;
    private final int bos;
    private final int eos;
    private final int eot;
    private final int eom;
    private final int unk;
    private final int sep;
    private final int pad;
    private final int mask;
    private final int fimPre;
    private final int fimSuf;
    private final int fimMid;
    private final int fimPad;
    private final int fimRep;
    private final int fimSep;
    private final int[] attrs;
    private final Set<Integer> eogIds;
    private final int[] partitionOrder;
    private final int[] suppressTokens;
    private final int[] prefixIds;
    private final int[] suffixIds;
    /** True when a partition token is {@link TokenizerVocab#ATTR_NORMALIZED}. */
    private final boolean normalizedTokens;
    /**
     * The normalized text of each {@link TokenizerVocab#ATTR_NORMALIZED} partition token, by id;
     * null when the vocabulary normalizer is the identity or no such token exists.
     */
    private final String[] normalizedTexts;

    private SpecialTokens(SpecialTokenRules.Result r) {
        this.vocab = r.vocab;
        this.bos = r.ids[SpecialTokenRules.BOS];
        this.eos = r.ids[SpecialTokenRules.EOS];
        this.eot = r.ids[SpecialTokenRules.EOT];
        this.eom = r.ids[SpecialTokenRules.EOM];
        this.unk = r.ids[SpecialTokenRules.UNK];
        this.sep = r.ids[SpecialTokenRules.SEP];
        this.pad = r.ids[SpecialTokenRules.PAD];
        this.mask = r.ids[SpecialTokenRules.MASK];
        this.fimPre = r.ids[SpecialTokenRules.FIM_PRE];
        this.fimSuf = r.ids[SpecialTokenRules.FIM_SUF];
        this.fimMid = r.ids[SpecialTokenRules.FIM_MID];
        this.fimPad = r.ids[SpecialTokenRules.FIM_PAD];
        this.fimRep = r.ids[SpecialTokenRules.FIM_REP];
        this.fimSep = r.ids[SpecialTokenRules.FIM_SEP];
        this.attrs = r.attrs;
        this.eogIds = Collections.unmodifiableSet(new LinkedHashSet<>(r.eog));
        this.partitionOrder = r.partitionOrder;
        this.suppressTokens = r.suppressTokens;
        this.prefixIds = r.vocab.prefixIds();
        this.suffixIds = r.vocab.suffixIds();
        boolean normalized = false;
        for (int id : partitionOrder) normalized |= (attrs[id] & TokenizerVocab.ATTR_NORMALIZED) != 0;
        this.normalizedTokens = normalized;
        this.normalizedTexts = normalized && !r.vocab.normalizer().isIdentity() ? normalizedTexts(r.vocab) : null;
    }

    private String[] normalizedTexts(TokenizerVocab v) {
        String[] out = new String[attrs.length];
        for (int id : partitionOrder) {
            if ((attrs[id] & TokenizerVocab.ATTR_NORMALIZED) != 0) out[id] = v.normalizer().normalize(v.text(id));
        }
        return out;
    }

    /** Builds the special-token view of a vocabulary. */
    public static SpecialTokens of(TokenizerVocab vocab) {
        if (vocab == null) throw new IllegalArgumentException("vocabulary is null");
        return new SpecialTokens(SpecialTokenRules.resolve(vocab));
    }

    public int bos() { return bos; }
    public int eos() { return eos; }
    public int eot() { return eot; }
    public int eom() { return eom; }
    public int unk() { return unk; }
    public int sep() { return sep; }
    public int pad() { return pad; }
    public int mask() { return mask; }
    /** {@code llama_vocab_cls}: llama.cpp uses the BOS token as CLS. */
    public int cls() { return bos; }
    public int fimPre() { return fimPre; }
    public int fimSuf() { return fimSuf; }
    public int fimMid() { return fimMid; }
    public int fimPad() { return fimPad; }
    public int fimRep() { return fimRep; }
    public int fimSep() { return fimSep; }

    /** The BOS id; fails when the vocabulary asks for BOS but has none. */
    public int requireBos() { return require(bos, "BOS"); }
    /** The EOS id; fails when the vocabulary asks for EOS but has none. */
    public int requireEos() { return require(eos, "EOS"); }
    /** The SEP id; fails when the vocabulary asks for SEP but has none. */
    public int requireSep() { return require(sep, "SEP"); }

    /**
     * The ids encoding puts before the text when special tokens are added: the vocabulary's
     * post-processor prefix ({@link TokenizerVocab#prefixIds()}) when present, else BOS when the
     * vocabulary adds BOS, else none. A copy.
     *
     * @throws IllegalStateException when BOS is required but undefined
     */
    public int[] leadingIds() {
        if (prefixIds != null) return prefixIds.clone();
        return vocab.addBos() ? new int[]{requireBos()} : NO_IDS.clone();
    }

    /**
     * The ids encoding puts after the text when special tokens are added: the vocabulary's
     * post-processor suffix ({@link TokenizerVocab#suffixIds()}) when present, else EOS when the
     * vocabulary adds EOS, else none. A copy.
     *
     * @throws IllegalStateException when EOS is required but undefined
     */
    public int[] trailingIds() {
        if (suffixIds != null) return suffixIds.clone();
        return vocab.addEos() ? new int[]{requireEos()} : NO_IDS.clone();
    }

    /** End-of-generation token ids. */
    public Set<Integer> eogIds() { return eogIds; }
    public boolean isEog(int id) { return id != NULL && eogIds.contains(id); }

    /** {@code tokenizer.ggml.suppress_tokens}, in-range ids only; a copy. */
    public int[] suppressTokens() { return suppressTokens.clone(); }

    /**
     * The effective {@code llama_token_attr} of a token after llama.cpp's load-time overrides
     * (CONTROL for detected EOG/FIM tokens, USER_DEFINED for harmony markers, UNUSED, LSTRIP,
     * RSTRIP); {@link TokenizerVocab#ATTR_UNDEFINED} out of range.
     */
    public int attr(int id) { return id >= 0 && id < attrs.length ? attrs[id] : TokenizerVocab.ATTR_UNDEFINED; }
    public boolean hasAttr(int id, int mask) { return (attr(id) & mask) != 0; }

    /**
     * True when a token contributes no text to user-visible output: out-of-range ids, BOS, PAD,
     * end-of-generation tokens, and CONTROL or UNKNOWN tokens ({@code token_to_piece} with
     * {@code special = false}).
     */
    public boolean isHidden(int id) {
        if (id < 0 || id >= attrs.length) return true;
        return id == bos || id == pad || isEog(id) || (attrs[id] & HIDDEN_ATTRS) != 0;
    }

    /**
     * {@code tokenizer_st_partition}: splits raw text around special tokens, longest first. With
     * {@code parseSpecial} false, CONTROL and UNKNOWN tokens stay plain text while USER_DEFINED
     * tokens are still split out. LSTRIP and RSTRIP drop C-locale whitespace next to a match.
     * Text fragments are not normalized; see {@link #partition(String, boolean, boolean)}.
     */
    public List<Fragment> partition(String text, boolean parseSpecial) {
        return partition(text, parseSpecial, false);
    }

    /**
     * {@link #partition(String, boolean)} in Hugging Face's two phases ({@code AddedVocabulary}
     * {@code extract_and_normalize}): the tokens that are not {@link TokenizerVocab#ATTR_NORMALIZED}
     * split the raw text; with {@code normalize}, each remaining text fragment is then normalized
     * by {@link TokenizerVocab#normalizer()} (and dropped when that leaves it empty); finally the
     * {@link TokenizerVocab#ATTR_NORMALIZED} tokens split what is left, matching their own
     * normalized text when {@code normalize} is set. Without normalized tokens and with an
     * identity normalizer this is the single llama.cpp pass.
     *
     * @throws IllegalStateException when a normalizer regex exceeds its step budget
     */
    public List<Fragment> partition(String text, boolean parseSpecial, boolean normalize) {
        List<Fragment> fragments = new ArrayList<>();
        if (text == null || text.isEmpty()) return fragments;
        fragments.add(Fragment.text(text));
        fragments = splitTokens(fragments, parseSpecial, false, false);
        if (normalize && !vocab.normalizer().isIdentity()) fragments = normalized(fragments);
        if (normalizedTokens) fragments = splitTokens(detached(fragments), parseSpecial, true, normalize);
        return Collections.unmodifiableList(fragments);
    }

    /**
     * One partition phase over the tokens whose {@link TokenizerVocab#ATTR_NORMALIZED} bit equals
     * {@code normalizedPhase}; in that phase {@code normalize} matches their normalized texts.
     */
    private List<Fragment> splitTokens(List<Fragment> fragments, boolean parseSpecial, boolean normalizedPhase,
                                       boolean normalize) {
        boolean normalizedText = normalizedPhase && normalize && normalizedTexts != null;
        List<Fragment> current = fragments;
        for (int id : partitionOrder) {
            if (!parseSpecial && (attrs[id] & HIDDEN_ATTRS) != 0) continue;
            if (((attrs[id] & TokenizerVocab.ATTR_NORMALIZED) != 0) != normalizedPhase) continue;
            String special = normalizedText ? normalizedTexts[id] : vocab.text(id);
            if (special.isEmpty()) continue;
            List<Fragment> next = new ArrayList<>(current.size() + 2);
            for (Fragment f : current) {
                if (f.isToken()) next.add(f);
                else splitFragment(f, special, id, next);
            }
            current = next;
        }
        return current;
    }

    private List<Fragment> normalized(List<Fragment> fragments) {
        List<Fragment> out = new ArrayList<>(fragments.size());
        for (Fragment f : fragments) {
            if (f.isToken()) {
                out.add(f);
                continue;
            }
            String s = vocab.normalizer().normalize(f.text());
            if (!s.isEmpty()) out.add(Fragment.text(s));
        }
        return out;
    }

    /**
     * The fragments with each text fragment as a sentence of its own: Hugging Face splits the
     * normalized tokens out of every split of the first phase separately, so their
     * {@code single_word} check sees only that split.
     */
    private static List<Fragment> detached(List<Fragment> fragments) {
        List<Fragment> out = new ArrayList<>(fragments.size());
        for (Fragment f : fragments) out.add(f.isToken() ? f : Fragment.text(f.text()));
        return out;
    }

    /**
     * Splits the text fragment {@code f} at the matches of {@code special}. The text pieces keep
     * the code points next to them in the phase's sentence, so a later {@code single_word} check
     * sees the same neighbours as Hugging Face's single pass over that sentence.
     */
    private void splitFragment(Fragment f, String special, int id, List<Fragment> out) {
        String s = f.text();
        boolean lstrip = (attrs[id] & TokenizerVocab.ATTR_LSTRIP) != 0;
        boolean rstrip = (attrs[id] & TokenizerVocab.ATTR_RSTRIP) != 0;
        boolean singleWord = (attrs[id] & TokenizerVocab.ATTR_SINGLE_WORD) != 0;
        int pos = 0;
        int from = 0;
        while (true) {
            int match = s.indexOf(special, from);
            if (match < 0) break;
            if (singleWord && !isSingleWord(f, match, match + special.length())) {
                from = match + special.length();
                continue;
            }
            int leftEnd = match;
            if (lstrip) {
                while (leftEnd > pos && isCSpace(s.charAt(leftEnd - 1))) leftEnd--;
            }
            if (leftEnd > pos) out.add(f.slice(pos, leftEnd));
            out.add(Fragment.token(id));
            pos = match + special.length();
            if (rstrip) {
                while (pos < s.length() && isCSpace(s.charAt(pos))) pos++;
            }
            from = pos;
        }
        if (pos < s.length()) out.add(pos == 0 ? f : f.slice(pos, s.length()));
    }

    /**
     * Hugging Face {@code single_word}: the match {@code [start, end)} of fragment {@code f}
     * neither follows nor precedes a word character of the sentence being split.
     */
    private static boolean isSingleWord(Fragment f, int start, int end) {
        return !isWordChar(f.codePointBefore(start)) && !isWordChar(f.codePointAt(end));
    }

    /**
     * Regex {@code \w} with Unicode classes: alphabetic, marks, decimal digits, connector punctuation,
     * joiners; false for {@link Fragment#NONE}.
     */
    private static boolean isWordChar(int cp) {
        if (cp == Fragment.NONE) return false;
        if (Character.isAlphabetic(cp) || cp == 0x200C || cp == 0x200D) return true;
        switch (Character.getType(cp)) {
            case Character.NON_SPACING_MARK:
            case Character.ENCLOSING_MARK:
            case Character.COMBINING_SPACING_MARK:
            case Character.DECIMAL_DIGIT_NUMBER:
            case Character.CONNECTOR_PUNCTUATION:
                return true;
            default:
                return false;
        }
    }

    /** C {@code isspace} in the C locale. */
    private static boolean isCSpace(char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == 0x0B || c == '\f' || c == '\r';
    }

    /** The special-token ids in partition order (longest text first); a copy. */
    public int[] partitionOrder() { return partitionOrder.clone(); }

    private static int require(int id, String what) {
        if (id == NULL) {
            throw new IllegalStateException("The vocabulary requires a " + what
                    + " token for special-token encoding but defines none");
        }
        return id;
    }

    /** Debug summary of the resolved ids. */
    @Override
    public String toString() {
        return "SpecialTokens{bos=" + bos + ", eos=" + eos + ", eot=" + eot + ", eom=" + eom + ", unk=" + unk
                + ", sep=" + sep + ", pad=" + pad + ", mask=" + mask + ", fim=" + Arrays.toString(new int[]{
                fimPre, fimSuf, fimMid, fimPad, fimRep, fimSep}) + ", eog=" + eogIds + "}";
    }

    /**
     * One partition fragment: either raw text or a special token id. Immutable. A text fragment
     * also records the code points just outside it in the sentence it was split from
     * ({@link #NONE} at the sentence's edges).
     */
    public static final class Fragment {
        /** The neighbour of a text fragment at an edge of its sentence. */
        static final int NONE = -1;

        private final String text;
        private final int token;
        private final int before;
        private final int after;

        private Fragment(String text, int token, int before, int after) {
            this.text = text;
            this.token = token;
            this.before = before;
            this.after = after;
        }

        static Fragment text(String text) { return new Fragment(text, NULL, NONE, NONE); }
        static Fragment token(int id) { return new Fragment(null, id, NONE, NONE); }

        /** The text {@code [from, to)} of this text fragment, with its neighbours in the sentence. */
        Fragment slice(int from, int to) {
            return new Fragment(text.substring(from, to), NULL, codePointBefore(from), codePointAt(to));
        }

        /** The sentence's code point before index {@code i} of this text fragment, or {@link #NONE}. */
        int codePointBefore(int i) {
            return i == 0 ? before : text.codePointBefore(i);
        }

        /** The sentence's code point at index {@code i} of this text fragment, or {@link #NONE}. */
        int codePointAt(int i) {
            return i == text.length() ? after : text.codePointAt(i);
        }

        public boolean isToken() { return text == null; }
        /** The special token id; {@link #NULL} for a text fragment. */
        public int token() { return token; }
        /** The raw text; null for a token fragment. */
        public String text() { return text; }
    }
}
