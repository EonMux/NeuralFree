package com.eonmux.neuralfree.gguf;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Set;
import java.util.TreeSet;

/**
 * The kernel's view of this process's mappings of one file ({@code /proc/self/smaps}): the {@code VmFlags} of its
 * mappings, where the paging advice shows as {@code rr} ({@code MADV_RANDOM}), {@code sr} ({@code MADV_SEQUENTIAL})
 * and {@code hg} ({@code MADV_HUGEPAGE}). Linux only.
 */
final class MappingFlags {

    static final Path SMAPS = Path.of("/proc/self/smaps");

    private MappingFlags() {
    }

    /** The union of the {@code VmFlags} of every mapping of {@code file}; empty when the file is not mapped. */
    static Set<String> of(Path file) throws IOException {
        String path = file.toRealPath().toString();
        Set<String> flags = new TreeSet<>();
        boolean ours = false;
        for (String line : Files.readAllLines(SMAPS)) {
            if (isHeader(line)) {
                String[] header = line.split("\\s+", 6);
                ours = header.length == 6 && header[5].equals(path);
            } else if (ours && line.startsWith("VmFlags:")) {
                flags.addAll(Arrays.asList(line.substring("VmFlags:".length()).trim().split("\\s+")));
            }
        }
        return flags;
    }

    /** Whether {@code line} is an {@code smaps} mapping header, {@code start-end perms ...}. */
    private static boolean isHeader(String line) {
        int dash = line.indexOf('-');
        int space = line.indexOf(' ');
        if (line.isEmpty() || Character.isWhitespace(line.charAt(0)) || dash <= 0 || space <= dash) {
            return false;
        }
        for (int i = 0; i < dash; i++) {
            if (Character.digit(line.charAt(i), 16) < 0) {
                return false;
            }
        }
        return true;
    }
}
