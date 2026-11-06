package com.eonmux.neuralfree.demo.tools.serve;

import java.util.ArrayList;
import java.util.List;

/**
 * The command-line splitter of the sweep and startup commands: a line of the server or benchmark
 * command into its arguments, POSIX shell style — arguments separate on unquoted whitespace, single
 * and double quotes group, a backslash escapes the next character inside double quotes and outside
 * them. Python's {@code shlex.split} for the shapes these commands carry.
 */
final class ShellSplit {

    private ShellSplit() {
    }

    /**
     * The arguments of {@code line}.
     *
     * @throws IllegalArgumentException for an unterminated quote or a trailing backslash
     */
    static List<String> split(String line) {
        List<String> out = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inWord = false;
        char quote = 0;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (quote != 0) {
                if (c == quote) {
                    quote = 0;
                } else if (c == '\\' && quote == '"' && i + 1 < line.length()
                        && (line.charAt(i + 1) == '"' || line.charAt(i + 1) == '\\')) {
                    current.append(line.charAt(++i));
                } else {
                    current.append(c);
                }
                continue;
            }
            if (c == '\'' || c == '"') {
                quote = c;
                inWord = true;
                continue;
            }
            if (c == '\\' && i + 1 < line.length()) {
                current.append(line.charAt(++i));
                inWord = true;
                continue;
            }
            if (Character.isWhitespace(c)) {
                if (inWord) {
                    out.add(current.toString());
                    current.setLength(0);
                    inWord = false;
                }
                continue;
            }
            current.append(c);
            inWord = true;
        }
        if (quote != 0) {
            throw new IllegalArgumentException("the command line has an unterminated " + quote + " quote: '"
                    + line + "'");
        }
        if (inWord) {
            out.add(current.toString());
        }
        return out;
    }
}
