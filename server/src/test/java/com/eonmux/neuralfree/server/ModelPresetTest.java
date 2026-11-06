package com.eonmux.neuralfree.server;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The router's preset INI parser ({@link ModelPreset}): the sections, the keys and the refusals. */
class ModelPresetTest {

    @TempDir
    Path dir;

    private static final String GOOD = """
            # the router's presets
            [gemma]
            path = gemma-q4.gguf
            alias = gem
            tag = local
            n_ctx = 4096
            parallel = 2
            ngl = 999
            device = cuda0
            load_on_startup = true

            [bert]
            ; a plain preset
            path = bert.gguf
            """;

    /** Writes {@code ini} and touches every file its {@code path} keys name, as a real layout would have them. */
    private List<ModelPreset> write(String ini) throws IOException {
        Path file = dir.resolve("models.ini");
        for (String line : ini.split("\n")) {
            String trimmed = line.trim();
            if (trimmed.startsWith("path = ")) {
                Path named = Path.of(trimmed.substring("path = ".length()).trim());
                Files.writeString(named.isAbsolute() ? named : dir.resolve(named), "unused");
            }
        }
        Files.writeString(file, ini);
        return ModelPreset.parse(file);
    }

    @Test
    void aSectionDeclaresOneModel() throws IOException {
        List<ModelPreset> presets = write(GOOD);
        assertEquals(2, presets.size());
        ModelPreset gemma = presets.get(0);
        assertEquals("gemma", gemma.name());
        assertEquals(dir.resolve("gemma-q4.gguf"), gemma.path(), "a relative path resolves beside the file");
        assertEquals(List.of("gem"), gemma.aliases());
        assertEquals(List.of("local"), gemma.tags());
        assertEquals(4096, gemma.nCtx());
        assertEquals(2, gemma.parallel());
        assertEquals(999, gemma.ngl());
        assertEquals("cuda0", gemma.device());
        assertTrue(gemma.loadOnStartup());
        ModelPreset bert = presets.get(1);
        assertEquals("bert", bert.name());
        assertNull(bert.nCtx());
        assertNull(bert.device());
        assertTrue(bert.aliases().isEmpty());
    }

    @Test
    void repeatedAliasAndTagKeysAccumulate() throws IOException {
        List<ModelPreset> presets = write("""
                [m]
                path = m.gguf
                alias = a
                alias = b
                tag = x
                tag = y
                """);
        assertEquals(List.of("a", "b"), presets.get(0).aliases());
        assertEquals(List.of("x", "y"), presets.get(0).tags());
    }

    @Test
    void anAbsolutePathStands() throws IOException {
        Path outside = dir.resolve("elsewhere.gguf");
        Files.writeString(outside, "unused");
        List<ModelPreset> presets = write("[m]\npath = " + outside + "\n");
        assertEquals(outside, presets.get(0).path());
    }

    @Test
    void unknownKeysAndBadValuesRefuseNamingTheSection() {
        IllegalArgumentException unknown = assertThrows(IllegalArgumentException.class,
                () -> write("[m]\npath = m.gguf\nctx = 4096\n"));
        assertTrue(unknown.getMessage().contains("ctx") && unknown.getMessage().contains("[m]"),
                unknown.getMessage());
        IllegalArgumentException negative = assertThrows(IllegalArgumentException.class,
                () -> write("[m]\npath = m.gguf\nn_ctx = 0\n"));
        assertTrue(negative.getMessage().contains("n_ctx"), negative.getMessage());
        IllegalArgumentException text = assertThrows(IllegalArgumentException.class,
                () -> write("[m]\npath = m.gguf\nngl = many\n"));
        assertTrue(text.getMessage().contains("ngl"), text.getMessage());
        IllegalArgumentException flag = assertThrows(IllegalArgumentException.class,
                () -> write("[m]\npath = m.gguf\nload_on_startup = sometimes\n"));
        assertTrue(flag.getMessage().contains("load_on_startup"), flag.getMessage());
    }

    @Test
    void structuralProblemsRefuseNamingTheLine() throws IOException {
        assertThrows(IllegalArgumentException.class, () -> write("path = m.gguf\n"));
        assertThrows(IllegalArgumentException.class, () -> write("[m]\npath m.gguf\n"));
        assertThrows(IllegalArgumentException.class, () -> write("[m]\n\n[m]\npath = m.gguf\n"));
        assertThrows(IllegalArgumentException.class, () -> write("[m]\n"));
        IllegalArgumentException missing = assertThrows(IllegalArgumentException.class,
                () -> write("[m]\nngl = 4\n"));
        assertTrue(missing.getMessage().contains("path"), missing.getMessage());
        Path file = dir.resolve("missing.ini");
        Files.writeString(file, "[m]\npath = never-written.gguf\n");
        IllegalArgumentException noFile = assertThrows(IllegalArgumentException.class,
                () -> ModelPreset.parse(file));
        assertTrue(noFile.getMessage().contains("never-written.gguf"), noFile.getMessage());
    }

    @Test
    void theRealRefusalCarriesThePropertyAndFile() throws IOException {
        Path file = dir.resolve("broken.ini");
        Files.writeString(file, "[m]\npath = m.gguf\nnope = 1\n");
        IllegalArgumentException e = assertInstanceOf(IllegalArgumentException.class,
                assertThrows(IllegalArgumentException.class, () -> ModelPreset.parse(file)));
        assertTrue(e.getMessage().contains(ModelPreset.PROPERTY) && e.getMessage().contains("broken.ini")
                        && e.getMessage().contains("3"),
                "the refusal names the property, the file and the line: " + e.getMessage());
    }
}
