package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;

import java.io.PrintStream;

/** The {@code --debug-prompt} dump: the formatted prompt, its length and its first and last tokens. */
final class PromptDebug {

    private static final int FIRST_TOKENS = 40;
    private static final int LAST_TOKENS = 20;

    private PromptDebug() {
    }

    /** Prints the dump of one formatted prompt to {@code out}. */
    static void print(PrintStream out, String fullPrompt, int[] tokens, GGUFTokenizer tokenizer) {
        out.println();
        out.println("=== DEBUG: Formatted Prompt ===");
        out.println(fullPrompt);
        out.println("=== END Formatted Prompt ===");
        out.println("Prompt length: " + fullPrompt.length() + " chars, " + tokens.length + " tokens");
        int show = Math.min(FIRST_TOKENS, tokens.length);
        out.println("First " + show + " tokens:");
        for (int i = 0; i < show; i++) {
            out.println("  [" + i + "] id=" + tokens[i] + " raw='" + tokenizer.getToken(tokens[i]) + "'");
        }
        if (tokens.length > FIRST_TOKENS) {
            out.println("  ...");
        }
        int lastStart = Math.max(show, tokens.length - LAST_TOKENS);
        if (lastStart > show) {
            out.println("Last " + (tokens.length - lastStart) + " tokens:");
            for (int i = lastStart; i < tokens.length; i++) {
                out.println("  [" + i + "] id=" + tokens[i] + " raw='" + tokenizer.getToken(tokens[i]) + "'");
            }
        }
        out.println("===");
        out.println();
    }
}
