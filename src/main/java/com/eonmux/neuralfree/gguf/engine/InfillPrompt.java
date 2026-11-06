package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.tokenizer.SpecialTokens;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The fill-in-the-middle prompt of llama.cpp's {@code format_prompt_infill} ({@code tools/server/server-common.cpp}),
 * the repo-level FIM pattern of Qwen2.5-Coder:
 *
 * <pre>
 * [FIM_REP]myproject\n[FIM_SEP]file0\nextra 0 ... [FIM_SEP]filename\n[FIM_PRE]prefix prompt[FIM_SUF]suffix[FIM_MID]
 * </pre>
 *
 * Without {@code FIM_REP}/{@code FIM_SEP} tokens the extra chunks join behind a {@code --- snippet ---} separator. The
 * prefix keeps its last {@code 3/4 nBatch} tokens and the suffix its first {@code nBatch/4 - 2 - prompt} tokens; the
 * extra context takes what the context leaves after a batch and twice the budget, nearest chunks first. The SPM
 * order puts the suffix first. Immutable inputs, no state.
 */
public final class InfillPrompt {

    /** One extra-context chunk: a file's name and text. */
    public static final class Chunk {
        private final String filename;
        private final String text;

        public Chunk(String filename, String text) {
            this.filename = filename == null ? "tmp" : filename;
            this.text = Objects.requireNonNull(text, "text");
        }

        public String filename() { return filename; }

        public String text() { return text; }
    }

    /** The separator llama.cpp writes between chunks without a {@code FIM_SEP} token: {@code \n\n--- snippet ---\n\n}. */
    private static final String SNIPPET = "\n\n--- snippet ---\n\n";

    private InfillPrompt() {
    }

    /**
     * Whether {@code tokenizer} names the FIM prefix, suffix and middle tokens; null when it does, else the missing
     * ones (llama.cpp's refusal text).
     */
    public static String missingTokens(GGUFTokenizer tokenizer) {
        SpecialTokens s = tokenizer.specialTokens();
        StringBuilder err = new StringBuilder();
        if (s.fimPre() < 0) {
            err.append("prefix token is missing. ");
        }
        if (s.fimSuf() < 0) {
            err.append("suffix token is missing. ");
        }
        if (s.fimMid() < 0) {
            err.append("middle token is missing. ");
        }
        return err.length() == 0 ? null : err.toString().trim();
    }

    /**
     * The infill prompt.
     *
     * @param prefix   the text before the cursor, as tokens
     * @param suffix   the text after the cursor, as tokens
     * @param extra    the extra context chunks, farthest first
     * @param prompt   tokens appended to the prefix (llama.cpp's {@code prompt})
     * @param nBatch   the batch the FIM part must fit
     * @param nPredict the generation budget, or a negative value for none
     * @param nCtx     the context length
     * @param spm      whether the model takes the suffix-prefix-middle order
     * @throws IllegalArgumentException when the tokenizer lacks a FIM token ({@link #missingTokens})
     */
    public static int[] build(GGUFTokenizer tokenizer, int[] prefix, int[] suffix, List<Chunk> extra, int[] prompt,
                              int nBatch, int nPredict, int nCtx, boolean spm) {
        String missing = missingTokens(tokenizer);
        if (missing != null) {
            throw new IllegalArgumentException("Infill is not supported by this model: " + missing);
        }
        SpecialTokens s = tokenizer.specialTokens();
        List<Integer> extraTokens = extraContext(tokenizer, s, extra);
        int prefixTake = Math.min(prefix.length, 3 * (nBatch / 4));
        int suffixTake = Math.min(suffix.length, Math.max(0, nBatch / 4 - (2 + prompt.length)));
        int extraTake = Math.min(Math.max(0, nCtx - nBatch - 2 * Math.max(nPredict, 0)), extraTokens.size());

        List<Integer> pre = new ArrayList<>();
        pre.add(s.fimPre());
        addAll(pre, Arrays.copyOfRange(prefix, prefix.length - prefixTake, prefix.length));
        addAll(pre, prompt);
        List<Integer> suf = new ArrayList<>();
        suf.add(s.fimSuf());
        addAll(suf, Arrays.copyOf(suffix, suffixTake));

        List<Integer> out = new ArrayList<>(extraTokens.subList(extraTokens.size() - extraTake, extraTokens.size()));
        if (tokenizer.vocab().addBos() && tokenizer.bosId() >= 0) {
            out.add(tokenizer.bosId());
        }
        out.addAll(spm ? suf : pre);
        out.addAll(spm ? pre : suf);
        out.add(s.fimMid());
        int[] ids = new int[out.size()];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = out.get(i);
        }
        return ids;
    }

    /** The repo header, each chunk under its file separator (or the snippet separator), and the current file. */
    private static List<Integer> extraContext(GGUFTokenizer tokenizer, SpecialTokens s, List<Chunk> extra) {
        List<Integer> out = new ArrayList<>();
        if (s.fimRep() >= 0) {
            out.add(s.fimRep());
            addAll(out, plain(tokenizer, "myproject\n"));
        }
        for (Chunk chunk : extra == null ? Collections.<Chunk>emptyList() : extra) {
            if (s.fimSep() >= 0) {
                out.add(s.fimSep());
                addAll(out, plain(tokenizer, chunk.filename() + "\n"));
            } else {
                addAll(out, plain(tokenizer, SNIPPET));
            }
            addAll(out, plain(tokenizer, chunk.text()));
        }
        if (s.fimSep() >= 0) {
            out.add(s.fimSep());
            addAll(out, plain(tokenizer, "filename\n"));
        }
        return out;
    }

    private static int[] plain(GGUFTokenizer tokenizer, String text) {
        return tokenizer.encode(text, false, false);
    }

    private static void addAll(List<Integer> out, int[] ids) {
        for (int id : ids) {
            out.add(id);
        }
    }
}
