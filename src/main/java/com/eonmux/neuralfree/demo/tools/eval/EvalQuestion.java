package com.eonmux.neuralfree.demo.tools.eval;

/**
 * One evaluation case as the dataset renders it: the question shown in progress lines, the prompt
 * the chat request carries, and the expected answer the grader compares against.
 */
public final class EvalQuestion {

    private final String text;
    private final String prompt;
    private final String expected;

    public EvalQuestion(String text, String prompt, String expected) {
        if (text == null || prompt == null || expected == null) {
            throw new IllegalArgumentException("an evaluation question needs a text, a prompt and an"
                    + " expected answer");
        }
        this.text = text;
        this.prompt = prompt;
        this.expected = expected;
    }

    /** The question's own text, without the prompt template. */
    public String text() {
        return text;
    }

    /** The prompt the chat request carries, template applied. */
    public String prompt() {
        return prompt;
    }

    /** The expected answer. */
    public String expected() {
        return expected;
    }
}
