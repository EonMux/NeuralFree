package com.eonmux.neuralfree.tensor.gpu;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Source-text helpers shared by the resource-only checks of the {@code train} kernel library
 * ({@link TrainOpIdParityTest}, {@link TrainKernelInventoryTest}): reading a source, mapping a manifest source to
 * its GLSL twin, and removing comments before a textual scan.
 */
final class TrainKernelSources {

    private TrainKernelSources() {
    }

    /** The file's contents as UTF-8 text. */
    static String read(Path p) throws IOException {
        return new String(Files.readAllBytes(p), StandardCharsets.UTF_8);
    }

    /** GLSL twin path of a manifest path: {@code .cl} to {@code .comp}, {@code .h} to {@code .glsl}. */
    static String twinOf(String manifestPath) {
        if (manifestPath.endsWith(".cl")) {
            return manifestPath.substring(0, manifestPath.length() - 3) + ".comp";
        }
        if (manifestPath.endsWith(".h")) {
            return manifestPath.substring(0, manifestPath.length() - 2) + ".glsl";
        }
        throw new IllegalArgumentException("train manifest entry '" + manifestPath + "' is neither .cl nor .h");
    }

    /**
     * Replaces block comments with spaces (keeping their newlines) and drops line comments. Outside comments the
     * train sources quote only {@code #include} paths and {@code #error} messages, none of which contains a
     * comment marker, so literals need no special handling.
     */
    static String stripComments(String s) {
        StringBuilder out = new StringBuilder(s.length());
        int i = 0;
        while (i < s.length()) {
            char c = s.charAt(i);
            if (c == '/' && i + 1 < s.length() && s.charAt(i + 1) == '*') {
                int end = s.indexOf("*/", i + 2);
                int stop = end < 0 ? s.length() : end + 2;
                for (int k = i; k < stop; k++) {
                    out.append(s.charAt(k) == '\n' ? '\n' : ' ');
                }
                i = stop;
            } else if (c == '/' && i + 1 < s.length() && s.charAt(i + 1) == '/') {
                while (i < s.length() && s.charAt(i) != '\n') {
                    i++;
                }
            } else {
                out.append(c);
                i++;
            }
        }
        return out.toString();
    }
}
