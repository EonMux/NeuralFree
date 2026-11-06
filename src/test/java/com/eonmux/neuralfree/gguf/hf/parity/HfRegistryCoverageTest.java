package com.eonmux.neuralfree.gguf.hf.parity;

import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.names.HfArchNames;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Static coverage of the delivered tables (the {@link HfArchNames} validation contract).
 *
 * <ul>
 *   <li>{@link HfArchRegistry#unmapped(HfArchEntry.Kind)} is empty for {@code TEXT} — every {@code archs.tsv} TEXT
 *       name that llama.cpp registers a class for is delivered.</li>
 *   <li>Every {@code archs.tsv} TEXT name either {@link HfArchRegistry#resolve resolves} (an entry registered the name)
 *       or is rejected with a {@link GgufFormatException} that names the architecture — the converter-listed names of
 *       classes without a text conversion (the {@code -} class rows) reject; unresolvable-but-listed names reject with
 *       the converter class in the message.</li>
 *   <li>A name with a converter class in the delivered tables is backed by some entry, so the set of registered names
 *       plus the explicitly rejected names equals the full table (the static set-count property; a table row never
 *       falls out of the loop silently).</li>
 * </ul>
 */
public class HfRegistryCoverageTest {

    @Test
    public void unmappedTextNamesAreEmpty() {
        List<String> unmapped = HfArchRegistry.unmapped(HfArchEntry.Kind.TEXT);
        assertEquals("every archs.tsv TEXT name with a class is registered", List.of(), unmapped);
    }

    @Test
    public void everyTableTextNameResolvesOrRejectsWithItsName() throws IOException {
        Path dir = Files.createTempDirectory("hf-registry");
        try {
            for (HfArchNames.Row row : HfArchNames.rows(HfArchEntry.Kind.TEXT)) {
                if (row.hasClass() && HfArchRegistry.isRejected(row.hfName(), HfArchEntry.Kind.TEXT)) {
                    // A registered rejection (a class whose __init__ raises for the name as a text model, as
                    // UltravoxModel's does, or a converter class NeuralFree refuses by name, HfArchTableGgufOnly)
                    // throws naming the architecture before any file is read.
                    GgufFormatException e = assertThrows(GgufFormatException.class, () -> resolve(dir, row.hfName()));
                    assertTrue(row + ": " + e.getMessage(), e.getMessage().contains(row.hfName()));
                } else if (row.hasClass()) {
                    // resolve() only reads the name; a registered entry returns without touching files.
                    HfArchEntry entry = resolve(dir, row.hfName());
                    assertNotNull(row.toString(), entry);
                    assertTrue(entry.hfNames().contains(row.hfName()));
                    if (row.arch() != null && row.dynamicArchs().isEmpty()) {
                        // The static row's resolver result equals the row's gguf_arch; the delivered
                        // tables resolve static rows through HfArchResolver.of, a fixed answer. A class that
                        // reassigns model_arch in __init__ (dynamic_archs.tsv) answers from the configuration, and
                        // HfDynamicArchCoverageTest checks each of its conditions.
                        assertEquals(row.toString(),
                                row.arch(), entry.arch().resolve(
                                        com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader.fromJson(dir,
                                                "{\"architectures\":[\"" + row.hfName() + "\"],"
                                                        + "\"model_type\":\"unknown_stub_type\"}"),
                                        com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex
                                                .of(new java.util.LinkedHashMap<>())));
                    }
                } else {
                    // A class-less row rejects before any file is read (HfArchRegistry.rejectedEntries).
                    GgufFormatException e = assertThrows(GgufFormatException.class, () -> resolve(dir, row.hfName()));
                    assertTrue(e.getMessage(), e.getMessage().contains("'" + row.hfName() + "'"));
                    assertFalse(e.getMessage(), e.getMessage().isEmpty());
                }
            }
        } finally {
            List<Path> files = Files.walk(dir, 1).collect(Collectors.toList());
            files.sort(Comparator.reverseOrder());
            for (Path p : files) {
                //noinspection ResultOfMethodCallIgnored
                p.toFile().delete();
            }
        }
    }

    @Test
    public void aProjectorNameResolvedAsTextRejectsNamed() {
        // The first MMPROJ row of the tables whose name the converter registers for no TEXT class, resolved as
        // TEXT, rejects with the openProjector note this package spans on every projector checkpoint
        // (HfArchRegistry.resolve, the "vision/audio projector" arm). Rows that carry both a text and a
        // projector class (AudioFlamingo3, CogVLM and friends) open as text models, like the converter's.
        List<HfArchNames.Row> mmproj = HfArchNames.rows(HfArchEntry.Kind.MMPROJ);
        assertFalse("the delivered tables register MMPROJ rows", mmproj.isEmpty());
        String name = null;
        for (HfArchNames.Row row : mmproj) {
            if (!HfArchNames.row(row.hfName(), HfArchEntry.Kind.TEXT).isPresent()) {
                name = row.hfName();
                break;
            }
        }
        assertTrue("the tables register a projector-only name", name != null);
        try {
            Path dir = Files.createTempDirectory("hf-registry-mmproj");
            //noinspection resource
            dir.toFile().deleteOnExit();
            String projector = name;
            GgufFormatException e = assertThrows(GgufFormatException.class, () -> resolve(dir, projector));
            assertTrue(e.getMessage(), e.getMessage().contains(projector));
            assertTrue(e.getMessage(), e.getMessage().contains("projector"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void registeredNameCountCoversTheTable() {
        int registered = 0;
        int rejected = 0;
        for (HfArchNames.Row row : HfArchNames.rows(HfArchEntry.Kind.TEXT)) {
            if (row.hasClass()) {
                assertTrue(row.hfName(), HfArchRegistry.has(row.hfName(), HfArchEntry.Kind.TEXT));
                registered++;
            } else {
                rejected++;
            }
        }
        assertFalse("archs.tsv has TEXT rows", registered + rejected == 0);
        // The two kinds of table rows are the only two fates; nothing else can appear.
        assertEquals(HfArchNames.rows(HfArchEntry.Kind.TEXT).size(), registered + rejected);
    }

    private static HfArchEntry resolve(Path dir, String name) {
        com.eonmux.neuralfree.gguf.hf.config.HfConfig cfg =
                com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader.fromJson(dir, "{\"architectures\":[\""
                        + name + "\"],\"model_type\":\"unknown_stub_type\"}");
        return HfArchRegistry.resolve(cfg, HfArchEntry.Kind.TEXT);
    }
}
