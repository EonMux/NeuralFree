package com.eonmux.neuralfree.demo;

import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.List;

/**
 * The files one {@code /glob} command names (tools/cli/cli-context.cpp:567-610): the directory part of the
 * pattern before its first glob character ({@code ! [ * ?}), {@code ~} expanded to {@code $HOME}, resolved
 * against the working directory; then every regular file below it whose {@code /}-separated relative path the
 * rest of the pattern matches ({@link GlobMatcher}), sorted by path. Unreadable directories are skipped, as the
 * reference iterates with {@code skip_permission_denied}.
 */
final class GlobFiles {

    private final Path baseDir;
    private final String pattern;

    private GlobFiles(Path baseDir, String pattern) {
        this.baseDir = baseDir;
        this.pattern = pattern;
    }

    /** Splits {@code pattern} into the directory resolved against {@code cwd} and the glob of relative paths. */
    static GlobFiles split(Path cwd, String pattern) {
        if (cwd == null || pattern == null) {
            throw new IllegalArgumentException("cwd and pattern are required");
        }
        int meta = indexOfMeta(pattern);
        if (meta <= 0) {
            return new GlobFiles(cwd, pattern);
        }
        int slash = pattern.lastIndexOf('/', meta - 1);
        if (slash < 0) {
            return new GlobFiles(cwd, pattern);
        }
        String dir = expandHome(pattern.substring(0, slash));
        return new GlobFiles(cwd.resolve(dir.isEmpty() ? "/" : dir).normalize(), pattern.substring(slash + 1));
    }

    /** The directory the walk starts at. */
    Path baseDir() {
        return baseDir;
    }

    /** The glob matched against the paths relative to {@link #baseDir}. */
    String pattern() {
        return pattern;
    }

    /** The matched regular files, sorted by path; empty when the base directory is missing. */
    List<Path> matches() throws IOException {
        List<Path> found = new ArrayList<>();
        if (!Files.isDirectory(baseDir)) {
            return found;
        }
        Files.walkFileTree(baseDir, new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
                if (attrs.isRegularFile() || attrs.isSymbolicLink() && Files.isRegularFile(file)) {
                    String relative = baseDir.relativize(file).toString().replace('\\', '/');
                    if (GlobMatcher.matches(pattern, relative)) {
                        found.add(file);
                    }
                }
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult visitFileFailed(Path file, IOException e) {
                return FileVisitResult.CONTINUE;
            }
        });
        found.sort(Path::compareTo);
        return found;
    }

    private static int indexOfMeta(String pattern) {
        for (int i = 0; i < pattern.length(); i++) {
            char c = pattern.charAt(i);
            if (c == '!' || c == '[' || c == '*' || c == '?') {
                return i;
            }
        }
        return -1;
    }

    private static String expandHome(String path) {
        if (!path.startsWith("~")) {
            return path;
        }
        String home = System.getenv("HOME");
        return home == null || home.isEmpty() ? path : home + path.substring(1);
    }
}
