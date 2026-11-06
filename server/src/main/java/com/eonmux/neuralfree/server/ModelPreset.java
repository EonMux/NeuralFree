package com.eonmux.neuralfree.server;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * One model preset of the router's INI file (llama.cpp's {@code --models-preset}, its {@code preset.ini}): every
 * {@code [section]} declares one servable model. The section name is the model id requests route by; the keys set
 * what serves it:
 *
 * <ul>
 *   <li>{@code path} (required): the GGUF file, a relative one resolved against the INI file's directory.</li>
 *   <li>{@code alias} and {@code tag}: additional names that resolve to the model, and its informational tags;
 *       both repeat, one entry per line.</li>
 *   <li>{@code n_ctx}, {@code parallel}, {@code ngl} and {@code device}: the engine settings that override the
 *       server's for this model alone (the memory context, the parallel sequences, the GPU layers, the device
 *       filter).</li>
 *   <li>{@code load_on_startup}: load this model when the server starts.</li>
 * </ul>
 *
 * <p>{@link #parse} refuses, naming the section and the key or line: an unknown key, a malformed line, a value out
 * of range, a repeated section or single-value key, a section without a {@code path}, and a path that names no
 * file. Immutable.
 */
final class ModelPreset {

    /** The property that names the INI file, as the refusals cite it. */
    static final String PROPERTY = "neuralfree.server.models.preset";

    private final String name;
    private final Path path;
    private final List<String> aliases;
    private final List<String> tags;
    private final Integer nCtx;
    private final Integer parallel;
    private final Integer ngl;
    private final String device;
    private final boolean loadOnStartup;

    private ModelPreset(Section section) {
        this.name = section.name;
        this.path = section.path;
        this.aliases = List.copyOf(section.aliases);
        this.tags = List.copyOf(section.tags);
        this.nCtx = section.nCtx;
        this.parallel = section.parallel;
        this.ngl = section.ngl;
        this.device = section.device;
        this.loadOnStartup = section.loadOnStartup;
    }

    /** The model id the preset's section declares. */
    String name() {
        return name;
    }

    /** The GGUF file, relative paths already resolved against the INI file's directory. */
    Path path() {
        return path;
    }

    /** Additional names that resolve to the model. */
    List<String> aliases() {
        return aliases;
    }

    /** Informational tags. */
    List<String> tags() {
        return tags;
    }

    /** The model's memory context override, or null for the server's. */
    Integer nCtx() {
        return nCtx;
    }

    /** The model's parallel-sequence override, or null for the server's. */
    Integer parallel() {
        return parallel;
    }

    /** The model's GPU-layer override (llama.cpp's {@code -ngl}), or null for the server's. */
    Integer ngl() {
        return ngl;
    }

    /** The model's device filter ({@code neuralfree.device}), or null for the server's. */
    String device() {
        return device;
    }

    /** Whether the model loads when the server starts (llama.cpp's {@code load-on-startup}). */
    boolean loadOnStartup() {
        return loadOnStartup;
    }

    /**
     * Parses the INI file at {@code file} into one preset per section, in file order.
     *
     * @throws IllegalArgumentException when a line is malformed, a key is unknown or repeated, a value is out of
     *                                 range, a section repeats or names no {@code path}, or a path names no file
     * @throws IOException              when the file does not read
     */
    static List<ModelPreset> parse(Path file) throws IOException {
        List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);
        List<Section> sections = new ArrayList<>();
        Section current = null;
        for (int i = 0; i < lines.size(); i++) {
            String line = stripBom(lines.get(i)).trim();
            if (line.isEmpty() || line.startsWith("#") || line.startsWith(";")) {
                continue;
            }
            if (line.startsWith("[") && line.endsWith("]")) {
                String name = line.substring(1, line.length() - 1).trim();
                if (name.isEmpty()) {
                    throw fail(file, i, "an empty section name");
                }
                if (named(sections, name) != null) {
                    throw fail(file, i, "section [" + name + "], which appears more than once");
                }
                current = new Section(name, file.getParent() == null ? Path.of(".") : file.getParent());
                sections.add(current);
                continue;
            }
            if (current == null) {
                throw fail(file, i, "the key '" + line + "', which lies outside a [section]");
            }
            int eq = line.indexOf('=');
            if (eq <= 0) {
                throw fail(file, i, "the line '" + line + "', which is no 'key = value'");
            }
            current.set(line.substring(0, eq).trim(), line.substring(eq + 1).trim(), file, i);
        }
        List<ModelPreset> out = new ArrayList<>(sections.size());
        for (Section section : sections) {
            if (section.path == null) {
                throw new IllegalArgumentException(PROPERTY + " section [" + section.name + "] carries no path key");
            }
            if (!Files.isRegularFile(section.path)) {
                throw new IllegalArgumentException(PROPERTY + " section [" + section.name + "] path '" + section.path
                        + "' names no file");
            }
            out.add(new ModelPreset(section));
        }
        return out;
    }

    /** The section named {@code name}, or null. */
    private static Section named(List<Section> sections, String name) {
        for (Section section : sections) {
            if (section.name.equals(name)) {
                return section;
            }
        }
        return null;
    }

    /** The mutable section under construction. */
    private static final class Section {
        private final String name;
        private final Path base;
        private Path path;
        private final List<String> aliases = new ArrayList<>();
        private final List<String> tags = new ArrayList<>();
        private Integer nCtx;
        private Integer parallel;
        private Integer ngl;
        private String device;
        private boolean loadOnStartup;

        Section(String name, Path base) {
            this.name = name;
            this.base = base;
        }

        void set(String key, String value, Path file, int line) {
            if (value.isEmpty()) {
                throw fail(file, line, "the key '" + key + "', which carries no value");
            }
            switch (key) {
                case "path" -> {
                    requireUnset(path, key, file, line);
                    path = resolve(value);
                }
                case "alias" -> aliases.add(value);
                case "tag" -> tags.add(value);
                case "n_ctx" -> {
                    requireUnset(nCtx, key, file, line);
                    nCtx = whole(key, value, 1, file, line);
                }
                case "parallel" -> {
                    requireUnset(parallel, key, file, line);
                    parallel = whole(key, value, 1, file, line);
                }
                case "ngl" -> {
                    requireUnset(ngl, key, file, line);
                    ngl = whole(key, value, -1, file, line);
                }
                case "device" -> {
                    requireUnset(device, key, file, line);
                    device = value;
                }
                case "load_on_startup" -> {
                    if (!value.equalsIgnoreCase("true") && !value.equalsIgnoreCase("false")) {
                        throw fail(file, line, "the key 'load_on_startup', whose value must be true or false,"
                                + " got '" + value + "'");
                    }
                    loadOnStartup = Boolean.parseBoolean(value);
                }
                default -> throw fail(file, line, "the unknown key '" + key + "': the keys are path, alias, tag,"
                        + " n_ctx, parallel, ngl, device and load_on_startup");
            }
        }

        /** The refusal for a parse problem of this section at {@code line} (0-based). */
        private IllegalArgumentException fail(Path file, int line, String problem) {
            return new IllegalArgumentException(PROPERTY + " section [" + name + "], line " + (line + 1) + " of "
                    + file.getFileName() + " refuses " + problem);
        }

        private Path resolve(String value) {
            Path raw = Path.of(value);
            return raw.isAbsolute() ? raw : base.resolve(raw).normalize();
        }

        private int whole(String key, String value, int min, Path file, int line) {
            try {
                int parsed = Integer.parseInt(value.trim());
                if (parsed < min) {
                    throw fail(file, line, "the key '" + key + "', whose value must be " + min + " or more,"
                            + " got " + parsed);
                }
                return parsed;
            } catch (NumberFormatException e) {
                throw fail(file, line, "the key '" + key + "', whose value must be a whole number, got '"
                        + value + "'");
            }
        }

        private void requireUnset(Object current, String key, Path file, int line) {
            if (current != null) {
                throw fail(file, line, "the key '" + key + "', which appears more than once");
            }
        }
    }

    /** The refusal for a parse problem at {@code line} (0-based). */
    private static IllegalArgumentException fail(Path file, int line, String problem) {
        return new IllegalArgumentException(PROPERTY + ", line " + (line + 1) + " of " + file.getFileName()
                + " refuses " + problem);
    }

    /** The line with a UTF-8 BOM stripped. */
    private static String stripBom(String line) {
        return line.startsWith("\uFEFF") ? line.substring(1) : line;
    }
}
