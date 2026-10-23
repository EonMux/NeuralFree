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
 *       (3162-3313).</li>
 *   <li>The partition order: CONTROL, USER_DEFINED and UNKNOWN tokens, longest UTF-8 text first
 *       (3316-3330); then the LSTRIP/RSTRIP attributes by pre-tokenizer, architecture and model
 *       name (3348-3424).</li>
 * </ol>
 * A NONE vocabulary has no special tokens: llama.cpp returns from {@code load} before any of this.
 */
public final class SpecialTokens {

    /** {@code LLAMA_TOKEN_NULL}. */
    public static final int NULL = -1;

    private static final int HIDDEN_ATTRS = TokenizerVocab.ATTR_CONTROL | TokenizerVocab.ATTR_UNKNOWN;

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
     */
    public List<Fragment> partition(String text, boolean parseSpecial) {
        List<Fragment> fragments = new ArrayList<>();
        if (text == null || text.isEmpty()) return fragments;
        fragments.add(Fragment.text(text));
        for (int id : partitionOrder) {
            if (!parseSpecial && (attrs[id] & HIDDEN_ATTRS) != 0) continue;
            String special = vocab.text(id);
            List<Fragment> next = new ArrayList<>(fragments.size() + 2);
            for (Fragment f : fragments) {
                if (f.isToken()) next.add(f);
                else splitFragment(f.text(), special, id, next);
            }
            fragments = next;
        }
        return Collections.unmodifiableList(fragments);
    }

    private void splitFragment(String s, String special, int id, List<Fragment> out) {
        boolean lstrip = (attrs[id] & TokenizerVocab.ATTR_LSTRIP) != 0;
        boolean rstrip = (attrs[id] & TokenizerVocab.ATTR_RSTRIP) != 0;
        int pos = 0;
        while (true) {
            int match = s.indexOf(special, pos);
            if (match < 0) break;
            int leftEnd = match;
            if (lstrip) {
                while (leftEnd > pos && isCSpace(s.charAt(leftEnd - 1))) leftEnd--;
            }
            if (leftEnd > pos) out.add(Fragment.text(s.substring(pos, leftEnd)));
            out.add(Fragment.token(id));
            pos = match + special.length();
            if (rstrip) {
                while (pos < s.length() && isCSpace(s.charAt(pos))) pos++;
            }
        }
        if (pos < s.length()) out.add(Fragment.text(s.substring(pos)));
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

    /** One partition fragment: either raw text or a special token id. Immutable. */
    public static final class Fragment {
        private final String text;
        private final int token;

        private Fragment(String text, int token) {
            this.text = text;
            this.token = token;
        }

        static Fragment text(String text) { return new Fragment(text, NULL); }
        static Fragment token(int id) { return new Fragment(null, id); }

        public boolean isToken() { return text == null; }
        /** The special token id; {@link #NULL} for a text fragment. */
        public int token() { return token; }
        /** The raw text; null for a token fragment. */
        public String text() { return text; }
    }
}
