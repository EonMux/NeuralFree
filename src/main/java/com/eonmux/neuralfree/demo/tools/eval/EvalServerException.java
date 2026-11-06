package com.eonmux.neuralfree.demo.tools.eval;

/**
 * The failure of an evaluation request after every retry, or of one grading request. The message is
 * the last reason the server gave, so a state file records why a case failed.
 */
public final class EvalServerException extends Exception {

    private static final long serialVersionUID = 1L;

    public EvalServerException(String message) {
        super(message);
    }
}
