package com.eonmux.neuralfree.acceleration.kernels;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Builds the OpenCL program text of a kernel library by concatenating its manifest entries in order.
 *
 * <p>Each entry is read from {@link KernelManifest.Entry#resourcePath(String)} (the library root, or
 * {@value KernelManifest#SHARED_ROOT} for {@code [shared]} entries) and preceded by
 * {@code #line 1 "<path>"} with the entry's plain {@link KernelManifest.Entry#path()}, so compiler
 * diagnostics name the real file and line. No {@code #include} processing takes place: the manifest order
 * already places every dependency before its users, and the sources contain no includes.</p>
 */
public final class KernelSourceAssembler {

    private KernelSourceAssembler() {
    }

    /**
     * Concatenates every manifest entry, including the {@code [opencl]} ones, in manifest order.
     *
     * @param m    the library manifest
     * @param cl   class loader whose resources hold the sources
     * @param root classpath root of the library sources (normally {@code m.root()})
     * @throws IllegalArgumentException naming every entry whose resource is missing
     * @throws java.io.UncheckedIOException when a source exists but cannot be read
     */
    public static String assemble(KernelManifest m, ClassLoader cl, String root) {
        Objects.requireNonNull(m, "manifest must not be null");
        Objects.requireNonNull(cl, "class loader must not be null");
        String r = KernelManifest.requireRoot(root);
        List<String> texts = new ArrayList<>(m.entries().size());
        List<String> missing = new ArrayList<>();
        long total = 0;
        for (KernelManifest.Entry e : m.entries()) {
            String text = KernelManifest.readText(cl, e.resourcePath(r));
            if (text == null) {
                missing.add(describe(e));
                texts.add(null);
            } else {
                texts.add(text);
                total += text.length() + e.path().length() + 16;
            }
        }
        if (!missing.isEmpty()) {
            throw new IllegalArgumentException(missingMessage(r, missing));
        }
        StringBuilder sb = new StringBuilder((int) Math.min(Integer.MAX_VALUE - 8, total));
        for (int i = 0; i < texts.size(); i++) {
            String path = m.entries().get(i).path();
            String text = texts.get(i);
            sb.append("#line 1 \"").append(path).append("\"\n");
            sb.append(text);
            if (!text.isEmpty() && text.charAt(text.length() - 1) != '\n') {
                sb.append('\n');
            }
        }
        return sb.toString();
    }

    /**
     * The manifest entries whose source resource is missing, in manifest order; empty when the library
     * is complete. A library entry is named by its path, a shared entry by {@code "[shared] <path>"}.
     */
    public static List<String> missingSources(KernelManifest m, ClassLoader cl, String root) {
        Objects.requireNonNull(m, "manifest must not be null");
        Objects.requireNonNull(cl, "class loader must not be null");
        String r = KernelManifest.requireRoot(root);
        List<String> missing = new ArrayList<>();
        for (KernelManifest.Entry e : m.entries()) {
            if (cl.getResource(e.resourcePath(r)) == null) {
                missing.add(describe(e));
            }
        }
        return missing;
    }

    /**
     * The reason a library cannot be assembled, naming the classpath roots the missing entries resolve
     * against.
     */
    static String missingMessage(String root, List<String> missing) {
        StringBuilder sb = new StringBuilder("kernel sources missing under classpath:").append(root).append('/');
        for (String m : missing) {
            if (m.startsWith(KernelManifest.SHARED_TAG)) {
                sb.append(" (").append(KernelManifest.SHARED_TAG).append(" entries under classpath:")
                        .append(KernelManifest.SHARED_ROOT).append("/)");
                break;
            }
        }
        return sb.append(": ").append(missing).toString();
    }

    private static String describe(KernelManifest.Entry e) {
        return e.shared() ? KernelManifest.SHARED_TAG + " " + e.path() : e.path();
    }
}
