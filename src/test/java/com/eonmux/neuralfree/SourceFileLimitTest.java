package com.eonmux.neuralfree;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Source-tree invariant: every source file of both modules stays under one shared line limit, so a file
 * grows by taking on a second responsibility only until this test fails and names the files to split into
 * cohesive ones. Covers the Java sources of the core module (main, vector and test sets) and of the server
 * module, and every kernel and shader source under {@code src/main/resources/kernels} and
 * {@code src/main/resources/shaders}.
 *
 * <p>Generated listing files are exempt by name in {@link #GENERATED_LISTINGS}; each entry states what
 * generates the file, and a hand-written source must never be added there.
 *
 * <p>Runs from the project directory (the Gradle test working directory), like
 * {@code LlmKernelInventoryTest}, and counts the source files directly.
 */
public class SourceFileLimitTest {

    /** Largest allowed file length in lines; a longer file must be split. */
    static final int MAX_LINES = 1500;

    /** Java source roots of both modules; {@code src/vector/java} is the core module's SIMD source set. */
    private static final List<String> JAVA_ROOTS = Collections.unmodifiableList(Arrays.asList(
            "src/main/java",
            "src/vector/java",
            "src/test/java",
            "server/src/main/java",
            "server/src/test/java"));

    /** Kernel and shader source roots; every file under them counts, of any extension. */
    private static final List<String> KERNEL_SHADER_ROOTS = Collections.unmodifiableList(Arrays.asList(
            "src/main/resources/kernels",
            "src/main/resources/shaders"));

    /**
     * Generated listing files exempt from the limit, by project-relative path. Each is produced by a tool
     * and verified by its own test, so its length is not a hand-maintained size:
     * <ul>
     *   <li>{@code src/main/resources/kernels/llm/inventory.txt} — the LLM kernel inventory, generated from
     *       the {@code kernels/llm} sources by the {@code updateKernelInventory} Gradle task and checked by
     *       {@code LlmKernelInventoryTest};</li>
     *   <li>{@code src/main/resources/kernels/train/inventory.txt} — the train kernel inventory, generated
     *       from the {@code kernels/train} sources by the same task.</li>
     * </ul>
     */
    private static final Set<String> GENERATED_LISTINGS = Collections.unmodifiableSet(new TreeSet<>(Arrays.asList(
            "src/main/resources/kernels/llm/inventory.txt",
            "src/main/resources/kernels/train/inventory.txt")));

    /** The Java sources of both modules: {@code .java} files under {@link #JAVA_ROOTS}, nothing excluded. */
    private final Scope javaScope = new Scope(JAVA_ROOTS, Collections.singleton(".java"), Collections.emptySet());

    /** The kernel and shader sources: every file under {@link #KERNEL_SHADER_ROOTS} except {@link #GENERATED_LISTINGS}. */
    private final Scope kernelShaderScope = new Scope(KERNEL_SHADER_ROOTS, Collections.emptySet(), GENERATED_LISTINGS);

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void javaSourcesStayWithinTheLimit() throws IOException {
        List<String> over = filesOverLimit(javaScope, projectDir());
        assertTrue(violationMessage("Java source", over), over.isEmpty());
    }

    @Test
    public void kernelAndShaderSourcesStayWithinTheLimit() throws IOException {
        List<String> over = filesOverLimit(kernelShaderScope, projectDir());
        assertTrue(violationMessage("kernel/shader source", over), over.isEmpty());
    }

    /**
     * The detector itself: a file at the limit passes, one line over is named with its count, every
     * over-limit file is reported, and only the exact excluded paths are skipped.
     */
    @Test
    public void detectorNamesEveryOverLimitFileAndAppliesTheBoundary() throws IOException {
        Path base = tmp.newFolder("tree").toPath();
        write(base.resolve("AtLimit.java"), MAX_LINES);
        write(base.resolve("OneOver.java"), MAX_LINES + 1);
        write(base.resolve("deep/TwoOver.java"), MAX_LINES + 2);
        write(base.resolve("generated.txt"), MAX_LINES + 3);
        Scope allFiles = new Scope(Collections.singletonList(base.toString()), Collections.emptySet(),
                Collections.singleton("generated.txt"));

        assertEquals(Arrays.asList(
                "OneOver.java: " + (MAX_LINES + 1) + " lines",
                "deep/TwoOver.java: " + (MAX_LINES + 2) + " lines"), filesOverLimit(allFiles, base));

        allFiles = new Scope(Collections.singletonList(base.toString()), Collections.emptySet(),
                Collections.emptySet());
        assertEquals(Arrays.asList(
                "OneOver.java: " + (MAX_LINES + 1) + " lines",
                "deep/TwoOver.java: " + (MAX_LINES + 2) + " lines",
                "generated.txt: " + (MAX_LINES + 3) + " lines"), filesOverLimit(allFiles, base));

        write(base.resolve("OneOver.java"), MAX_LINES);
        assertEquals(Arrays.asList(
                "deep/TwoOver.java: " + (MAX_LINES + 2) + " lines",
                "generated.txt: " + (MAX_LINES + 3) + " lines"), filesOverLimit(allFiles, base));
    }

    /** One scan: the source roots to walk, the extensions those roots hold (empty for every extension), and the excluded generated paths. */
    private static final class Scope {
        final List<String> roots;
        final Set<String> extensions;
        final Set<String> excluded;

        Scope(List<String> roots, Set<String> extensions, Set<String> excluded) {
            this.roots = roots;
            this.extensions = extensions;
            this.excluded = excluded;
        }
    }

    /** Every file of the scope over the limit, as {@code "path: N lines"} relative to {@code baseDir}; missing roots fail. */
    private static List<String> filesOverLimit(Scope scope, Path baseDir) throws IOException {
        List<String> over = new ArrayList<>();
        for (String root : scope.roots) {
            Path dir = Paths.get(root).isAbsolute() ? Paths.get(root) : baseDir.resolve(root);
            assertTrue("configured source root is missing: " + dir, Files.isDirectory(dir));
            List<Path> files;
            try (Stream<Path> walked = Files.walk(dir)) {
                files = walked.filter(Files::isRegularFile).collect(Collectors.toList());
            }
            for (Path file : files) {
                if (!scope.extensions.isEmpty() && !endsWithAny(file, scope.extensions)) {
                    continue;
                }
                if (scope.excluded.contains(baseDir.relativize(file).toString())) {
                    continue;
                }
                int lines = countLines(file);
                if (lines > MAX_LINES) {
                    over.add(baseDir.relativize(file) + ": " + lines + " lines");
                }
            }
        }
        Collections.sort(over);
        return over;
    }

    private static boolean endsWithAny(Path file, Set<String> extensions) {
        String name = file.getFileName().toString();
        for (String extension : extensions) {
            if (name.endsWith(extension)) {
                return true;
            }
        }
        return false;
    }

    private static String violationMessage(String kind, List<String> over) {
        StringBuilder sb = new StringBuilder()
                .append(over.size()).append(' ').append(kind).append(" files exceed the ").append(MAX_LINES)
                .append("-line limit; split them along their cohesive responsibilities:");
        for (String entry : over) {
            sb.append("\n  ").append(entry);
        }
        return sb.toString();
    }

    /** Lines as a reader counts them: newline-terminated lines plus a final line without its newline. */
    private static int countLines(Path file) throws IOException {
        byte[] bytes = Files.readAllBytes(file);
        int lines = 0;
        for (byte b : bytes) {
            if (b == '\n') {
                lines++;
            }
        }
        if (bytes.length > 0 && bytes[bytes.length - 1] != '\n') {
            lines++;
        }
        return lines;
    }

    private static void write(Path file, int lines) throws IOException {
        Files.createDirectories(file.getParent());
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < lines; i++) {
            sb.append("line ").append(i).append('\n');
        }
        Files.write(file, sb.toString().getBytes(StandardCharsets.UTF_8));
    }

    private static Path projectDir() {
        return Paths.get("").toAbsolutePath().normalize();
    }
}
