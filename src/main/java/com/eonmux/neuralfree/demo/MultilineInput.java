package com.eonmux.neuralfree.demo;

import java.io.BufferedReader;
import java.io.IOException;

/**
 * One submitted console input, possibly several lines, read as llama.cpp common/console.cpp
 * {@code readline_simple} reads it: a line ending in {@code /} returns control with that character dropped, a
 * line ending in {@code \} flips the line's default (join the next line, or in multiline mode end the input)
 * with that character dropped, and in multiline mode every other line joins the next. Joined lines keep their
 * newlines; the submitted text drops the final one.
 */
final class MultilineInput {

    private final BufferedReader reader;

    MultilineInput(BufferedReader reader) {
        if (reader == null) {
            throw new IllegalArgumentException("reader is required");
        }
        this.reader = reader;
    }

    /**
     * Reads one submitted input. End of stream with nothing read returns null; Enter on an empty line returns
     * the empty string, which callers treat as "pass control back".
     *
     * @param multiline whether lines keep joining until a {@code /} or {@code \} ends the input
     */
    String read(boolean multiline) throws IOException {
        StringBuilder buffer = new StringBuilder();
        boolean anotherLine = true;
        while (anotherLine) {
            String line = reader.readLine();
            if (line == null) {
                break;
            }
            anotherLine = multiline;
            if (!line.isEmpty()) {
                char last = line.charAt(line.length() - 1);
                if (last == '/') {
                    line = line.substring(0, line.length() - 1);
                    anotherLine = false;
                } else if (last == '\\') {
                    line = line.substring(0, line.length() - 1);
                    anotherLine = !multiline;
                }
            }
            buffer.append(line).append('\n');
        }
        if (buffer.length() == 0) {
            return null;
        }
        buffer.setLength(buffer.length() - 1);
        return buffer.toString();
    }
}
