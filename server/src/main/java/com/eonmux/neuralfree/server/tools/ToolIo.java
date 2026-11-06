package com.eonmux.neuralfree.server.tools;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The filesystem and process access the server tools run through (llama.cpp's {@code tools_io_basic}, the host
 * runtime): every path resolves against a working directory — the tool call's {@code cwd} or the server process
 * one — with a leading {@code ~/} expanded and {@code .} and {@code ..} normalized away, and every command runs as
 * {@code sh -c} in that directory with a timeout and an output cap. The other llama.cpp runtimes (a container or
 * an SSH target) do not run in this build and are refused by name where they are configured.
 *
 * <p>One instance per tool call: {@code cwd} is fixed at construction. Not thread-safe against concurrent calls
 * of the same instance; distinct calls build their own.</p>
 */
public final class ToolIo {

    /** A listed entry: its path relative to the search base, '/' separators, and whether it is a directory. */
    public record Entry(String rel, boolean dir) { }

    /** A recursive listing: the entries found before the time budget ran out, and whether it did. */
    public record Listing(List<Entry> entries, boolean truncated, String error) { }

    /** A finished process: stdout and stderr combined (the truncation marker appended), its exit code and
     *  whether the timeout killed it. */
    public record Run(String output, int exitCode, boolean timedOut) { }

    private static final String[] JUNK_DIRS = {
        ".git", ".svn", ".hg", "node_modules", "__pycache__",
        ".venv", "venv", "dist", "build", "target", ".cache", ".idea", ".vscode",
    };

    /** The budget one recursive listing may spend before it stops and reports {@code truncated}. */
    static final Duration LIST_BUDGET = Duration.ofSeconds(15);

    private final Path cwd;
    private final String home;

    /** An IO rooted at the process working directory. */
    public ToolIo() {
        this(null);
    }

    /** An IO whose relative paths resolve against {@code cwd} (null or blank keeps the process one). */
    public ToolIo(String cwd) {
        this.cwd = cwd == null || cwd.isBlank() ? Path.of("").toAbsolutePath().normalize() : Path.of(cwd);
        this.home = System.getProperty("user.home", "");
    }

    /**
     * {@code path} as an absolute path: a leading {@code ~/} (or bare {@code ~}) expands to the user home, an
     * absolute path normalizes in place, a relative one resolves against the working directory.
     */
    public Path resolve(String path) {
        String p = expandHome(path);
        Path full = Path.of(p);
        if (!full.isAbsolute()) {
            full = cwd.resolve(full);
        }
        full = full.normalize();
        return full.toString().isEmpty() ? Path.of("/") : full;
    }

    /** {@code path} with a leading {@code ~/} or bare {@code ~} replaced by the user home; others unchanged. */
    private String expandHome(String path) {
        if (home.isEmpty() || !path.startsWith("~")) {
            return path;
        }
        if (path.length() > 1 && path.charAt(1) != '/') {
            return path;
        }
        return home + path.substring(1);
    }

    /** Whether {@code path} names a directory. */
    public boolean isDirectory(String path) {
        return Files.isDirectory(resolve(path));
    }

    /** Whether {@code path} names a regular file. */
    public boolean isRegularFile(String path) {
        return Files.isRegularFile(resolve(path));
    }

    /** The size of the file at {@code path}, or -1 when it cannot be stat'd. */
    public long fileSize(String path) {
        try {
            return Files.size(resolve(path));
        } catch (IOException e) {
            return -1;
        }
    }

    /** The whole file at {@code path}, or null when it does not open. */
    public String read(String path) {
        try {
            return Files.readString(resolve(path), StandardCharsets.UTF_8);
        } catch (IOException | UncheckedIOException e) {
            return null;
        }
    }

    /** Writes {@code content} to {@code path}, creating it and its parent directories; false when it fails. */
    public boolean write(String path, String content) {
        Path full = resolve(path);
        try {
            if (full.getParent() != null) {
                Files.createDirectories(full.getParent());
            }
            Files.writeString(full, content, StandardCharsets.UTF_8);
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    /**
     * The entries under {@code base}: directories recursively ({@code maxDepth} 0 for unlimited, 1 the direct
     * children), files only, or both, the junk directories never descended into, stopping at the time budget —
     * the listing then reports {@code truncated}. An unreadable base answers an {@code error}.
     */
    public Listing list(String base, int maxDepth, boolean files, boolean dirs) {
        Path root = resolve(base);
        if (!Files.isDirectory(root)) {
            return new Listing(List.of(), false, "path is not a directory");
        }
        List<Entry> out = new ArrayList<>();
        long deadline = System.nanoTime() + LIST_BUDGET.toNanos();
        boolean truncated;
        try {
            truncated = walk(root, root, maxDepth, files, dirs, out, deadline);
        } catch (IOException e) {
            return new Listing(out, true, null);
        }
        return new Listing(List.copyOf(out), truncated, null);
    }

    /** Depth-first walk adding entries to {@code out}; true when the budget or an unreadable branch cut it. */
    private boolean walk(Path root, Path dir, int maxDepth, boolean files, boolean dirs, List<Entry> out,
                         long deadline) throws IOException {
        boolean truncated = false;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path child : stream) {
                if (out.size() > 100_000 || System.nanoTime() - deadline > 0) {
                    return true;
                }
                String name = child.getFileName().toString();
                boolean directory = Files.isDirectory(child);
                String rel = root.relativize(child).toString().replace('\\', '/');
                if (files && (!directory || dirs)) {
                    out.add(new Entry(directory ? rel + "/" : rel, directory));
                } else if (dirs && directory) {
                    out.add(new Entry(rel + "/", true));
                }
                if (directory && !isJunk(name) && (maxDepth <= 0 || depth(rel) < maxDepth)) {
                    truncated |= walk(root, child, maxDepth, files, dirs, out, deadline);
                }
            }
        } catch (IOException e) {
            return true; // an unreadable branch ends its walk, the rest continues
        }
        return truncated;
    }

    /** Whether the directory name is one the listings never descend into (llama.cpp's junk set). */
    static boolean isJunk(String name) {
        for (String junk : JUNK_DIRS) {
            if (junk.equals(name)) {
                return true;
            }
        }
        return false;
    }

    /** The depth of a '/'-separated relative path: {@code a/b/c} is 3. */
    private static int depth(String rel) {
        int d = 1;
        for (int i = 0; i < rel.length(); i++) {
            if (rel.charAt(i) == '/') {
                d++;
            }
        }
        return d;
    }

    /**
     * Runs {@code command} under {@code sh -c} in the working directory: stdout and stderr combined, killed at
     * {@code timeout} seconds, the output held to {@code maxOutputSize} bytes (the excess dropped, the marker
     * {@code [output truncated]} appended). A spawn failure answers an explanatory output with exit code -1.
     */
    public Run run(String command, int timeout, int maxOutputSize) {
        return run(command, timeout, maxOutputSize, null);
    }

    /**
     * {@link #run} with every output piece reaching {@code onChunk} as it is read (a streaming tool call);
     * null streams nothing.
     */
    public Run run(String command, int timeout, int maxOutputSize, java.util.function.Consumer<String> onChunk) {
        ProcessBuilder builder = new ProcessBuilder("sh", "-c", command);
        builder.directory(cwd.toFile());
        builder.redirectErrorStream(true);
        Process process;
        try {
            process = builder.start();
        } catch (IOException e) {
            return new Run("failed to start sh: " + e.getMessage(), -1, false);
        }
        // the drain reads until EOF, so the deadline destroys the process: a hung child closes the pipe
        AtomicBoolean timedOut = new AtomicBoolean();
        Thread watchdog = Thread.ofPlatform().daemon().start(() -> {
            try {
                if (!process.waitFor(timeout, TimeUnit.SECONDS)) {
                    timedOut.set(true);
                    process.destroyForcibly();
                }
            } catch (InterruptedException ignored) {
                // the drain finished first; the watchdog exits
            }
        });
        String output = drain(process.getInputStream(), maxOutputSize, onChunk);
        int exit = -1;
        try {
            exit = process.waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            process.destroyForcibly();
            return new Run("interrupted", -1, true);
        }
        watchdog.interrupt();
        return new Run(output, exit, timedOut.get());
    }

    /** Reads a stream to its end as UTF-8 text, held to {@code max} bytes (the marker appended on the cut),
     *  every piece reaching {@code onChunk} when it is not null. */
    static String drain(InputStream in, int max, java.util.function.Consumer<String> onChunk) {
        StringBuilder out = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            char[] buffer = new char[4096];
            int total = 0;
            int read;
            while ((read = reader.read(buffer)) >= 0) {
                String piece = new String(buffer, 0, read);
                if (total + read > max) {
                    String kept = piece.substring(0, Math.max(0, max - total));
                    if (!kept.isEmpty()) {
                        out.append(kept);
                        if (onChunk != null) {
                            onChunk.accept(kept);
                        }
                    }
                    String marker = "\n[output truncated]";
                    out.append(marker);
                    if (onChunk != null) {
                        onChunk.accept(marker);
                    }
                    return out.toString();
                }
                out.append(piece);
                if (onChunk != null) {
                    onChunk.accept(piece);
                }
                total += read;
            }
        } catch (IOException e) {
            String marker = "\n[output truncated: " + e.getMessage() + "]";
            out.append(marker);
            if (onChunk != null) {
                onChunk.accept(marker);
            }
        }
        return out.toString();
    }
}
