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
 * PT-1, HfArchRegistryCoverageTest: static coverage of the delivered tables (PT-1 spec, HF.md §5.6 the updateHfTables
 * invariants, and the {@link HfArchNames} validation contract).
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
                if (row.hasClass()) {
                    // resolve() only reads the name; a registered entry returns without touching files.
                    HfArchEntry entry = resolve(dir, row.hfName());
                    assertNotNull(row.toString(), entry);
                    assertTrue(entry.hfNames().contains(row.hfName()));
                    if (row.arch() != null) {
                        // The static row's resolver result equals the row's gguf_arch (PT-1 spec); the delivered
                        // tables resolve static rows through HfArchResolver.of, a fixed answer.
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
        // The first MMPROJ row of the tables, resolved as TEXT, rejects with the openProjector note this package
        // spans on every projector checkpoint (HfArchRegistry.resolve, the "vision/audio projector" arm).
        List<HfArchNames.Row> mmproj = HfArchNames.rows(HfArchEntry.Kind.MMPROJ);
        assertFalse("the delivered tables register MMPROJ rows", mmproj.isEmpty());
        String name = mmproj.get(0).hfName();
        try {
            Path dir = Files.createTempDirectory("hf-registry-mmproj");
            //noinspection resource
            dir.toFile().deleteOnExit();
            GgufFormatException e = assertThrows(GgufFormatException.class, () -> resolve(dir, name));
            assertTrue(e.getMessage(), e.getMessage().contains(name));
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
