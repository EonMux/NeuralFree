package com.eonmux.neuralfree.demo.tools;

import java.util.ArrayList;
import java.util.List;

/**
 * The text splitting of the C++ {@code getline} loops perplexity.cpp reads its datafiles with: a line ends
 * at {@code '\n'}, and the WinoGrande loader's {@code if (in.fail() || in.eof()) break} drops a final line
 * without its newline while the HellaSwag loader's {@code while (getline(...))} keeps it.
 */
final class CliText {

    private CliText() {
    }

    /** Every {@code '\n'}-separated line, the unterminated last line included. */
    static List<String> getlineAll(String text) {
        if (text.isEmpty()) {
            return new ArrayList<>();
        }
        List<String> lines = split(text);
        if (text.charAt(text.length() - 1) == '\n') {
            lines.remove(lines.size() - 1);
        }
        return lines;
    }

    /** The complete {@code '\n'}-terminated lines; the reference's CSV loop drops an unterminated last line. */
    static List<String> getlineComplete(String text) {
        List<String> lines = split(text);
        lines.remove(lines.size() - 1);
        return lines;
    }

    private static List<String> split(String text) {
        List<String> lines = new ArrayList<>();
        int start = 0;
        for (int i = 0; i < text.length(); i++) {
            if (text.charAt(i) == '\n') {
                lines.add(text.substring(start, i));
                start = i + 1;
            }
        }
        lines.add(text.substring(start));
        return lines;
    }
}
