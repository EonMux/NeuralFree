package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.inspect.GgufHasher;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.utils.json.Json;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@code nftool dump} and {@code nftool hash}: the command-line forms and refusals, the printed reports, the hash
 * lines, and the manifest verification exit codes of llama.cpp's {@code gguf-hash}.
 */
public class GgufInspectToolsCommandTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    // ================================================================== dump

    @Test
    public void dumpParsesItsFlags() throws Exception {
        GgufDumpCommand.Parsed p = GgufDumpCommand.parse(new String[] {"--no-tensors", "--json",
                "--json-array", "m.gguf"});
        assertEquals("m.gguf", p.model);
        assertTrue(p.options.isNoTensors());
        assertTrue(p.options.isJson());
        assertTrue(p.options.isJsonArray());
        assertFalse(p.dataOffset);
    }

    @Test
    public void dumpPrintsTheDataOffsetAndAlignmentAlone() throws Exception {
        Path file = fixture().writeTempFile();
        long dataOffset;
        try (GGUFFileParser src = ModelSources.parseGguf(file.toString())) {
            dataOffset = src.getDataOffset();
        }
        StringBuilder offset = new StringBuilder();
        GgufDumpCommand.execute(GgufDumpCommand.parse(new String[] {"--data-offset", file.toString()}), offset);
        assertEquals(dataOffset + "\n", offset.toString());
        StringBuilder alignment = new StringBuilder();
        GgufDumpCommand.execute(GgufDumpCommand.parse(new String[] {"--data-alignment", file.toString()}),
                alignment);
        assertEquals("32\n", alignment.toString());
    }

    @Test
    public void dumpJsonIsAJsonReport() throws Exception {
        Path file = fixture().writeTempFile();
        StringBuilder out = new StringBuilder();
        GgufDumpCommand.execute(GgufDumpCommand.parse(new String[] {"--json", "--json-array", file.toString()}),
                out);
        @SuppressWarnings("unchecked")
        Map<String, Object> report = (Map<String, Object>) Json.parse(out.toString());
        assertEquals(file.toString(), report.get("filename"));
        assertTrue(((Map<?, ?>) report.get("tensors")).containsKey("t1"));
    }

    @Test
    public void dumpRefusals() {
        assertUsage(new String[] {}, "needs a GGUF file");
        assertUsage(new String[] {"a.gguf", "b.gguf"}, "too many file arguments");
        assertUsage(new String[] {"--nope", "a.gguf"}, "unknown option");
        assertUsage(new String[] {"--markdown", "a.gguf"}, "--markdown");
        assertUsage(new String[] {"--json-array", "a.gguf"}, "--json");
    }

    @Test
    public void dumpHelpPrintsTheUsage() throws Exception {
        StringBuilder out = new StringBuilder();
        GgufDumpCommand.execute(GgufDumpCommand.parse(new String[] {"--help"}), out);
        assertTrue(out.toString().contains("nftool dump"));
    }

    // ================================================================== hash

    @Test
    public void hashParsesItsFlags() throws Exception {
        GgufHashCommand.Parsed defaults = GgufHashCommand.parse(new String[] {"m.gguf"});
        assertTrue(defaults.autoSelect);
        assertTrue(defaults.algorithms.isEmpty());

        GgufHashCommand.Parsed all = GgufHashCommand.parse(new String[] {"--all", "--no_layer", "m.gguf"});
        assertEquals(EnumSet.of(GgufHasher.Algorithm.XXH64, GgufHasher.Algorithm.SHA1,
                GgufHasher.Algorithm.SHA256), all.algorithms);
        assertFalse("--all leaves the UUID out", all.algorithms.contains(GgufHasher.Algorithm.UUID));
        assertTrue(all.noLayer);
        assertFalse(all.autoSelect);

        GgufHashCommand.Parsed check = GgufHashCommand.parse(new String[] {"-c", "m.manifest", "m.gguf"});
        assertEquals("m.manifest", check.check.toString());
        assertEquals("m.gguf", check.model);
    }

    @Test
    public void hashRefusals() {
        assertHashUsage(new String[] {}, "needs a GGUF file");
        assertHashUsage(new String[] {"--nope", "m.gguf"}, "unknown option");
        assertHashUsage(new String[] {"-c"}, "needs a value");
        assertHashUsage(new String[] {"a.gguf", "b.gguf"}, "too many file arguments");
    }

    @Test
    public void defaultHashIsXxh64PerTensorAndModel() throws Exception {
        Path file = fixture().writeTempFile();
        StringBuilder out = new StringBuilder();
        assertEquals(0, GgufHashCommand.execute(GgufHashCommand.parse(new String[] {file.toString()}), out));
        String text = out.toString();
        String[] lines = text.split("\n");
        assertEquals(3, lines.length);
        assertTrue(lines[0], lines[0].startsWith("xxh64    "));
        assertTrue(lines[0], lines[0].endsWith(file + ":t1"));
        assertTrue(lines[1], lines[1].endsWith(file + ":t2"));
        assertTrue(lines[2], lines[2].endsWith(file.toString()));
        assertFalse(text.contains(" -  "));
    }

    @Test
    public void noLayerKeepsOnlyTheModelHashes() throws Exception {
        Path file = fixture().writeTempFile();
        StringBuilder out = new StringBuilder();
        assertEquals(0, GgufHashCommand.execute(
                GgufHashCommand.parse(new String[] {"--all", "--no-layer", file.toString()}), out));
        String[] lines = out.toString().split("\n");
        assertEquals(3, lines.length);
        assertEquals("xxh64", lines[0].substring(0, 5));
        assertEquals("sha1", lines[1].substring(0, 4));
        assertEquals("sha256", lines[2].substring(0, 6));
        for (String line : lines) {
            assertTrue(line, line.endsWith(file.toString()));
        }
    }

    @Test
    public void aGeneratedManifestVerifies() throws Exception {
        Path file = fixture().writeTempFile();
        Path manifest = writeManifest(file, "--all");
        StringBuilder out = new StringBuilder();
        int code = GgufHashCommand.execute(
                GgufHashCommand.parse(new String[] {"--check", manifest.toString(), file.toString()}), out);
        assertEquals(0, code);
        String text = out.toString();
        assertTrue(text, text.startsWith("manifest  " + manifest + "  sha256  sha1  xxh64\n"));
        assertTrue(text, text.contains("Verification results for " + manifest + " - Success"));
        assertTrue(text, text.contains("  -  Ok"));
        assertFalse(text, text.contains("Mismatch"));
        // without an algorithm flag, the manifest selects sha256
        for (String line : text.split("\n")) {
            if (line.contains(file.toString()) && !line.startsWith("manifest")) {
                assertTrue(line, line.startsWith("sha256"));
            }
        }
    }

    @Test
    public void aTamperedManifestMismatches() throws Exception {
        Path file = fixture().writeTempFile();
        Path manifest = writeManifest(file, "--all");
        String text = new String(Files.readAllBytes(manifest), StandardCharsets.UTF_8);
        String tampered = text.replaceFirst("sha256    [0-9a-f]", "sha256    0");
        Files.write(manifest, tampered.getBytes(StandardCharsets.UTF_8));
        StringBuilder out = new StringBuilder();
        int code = GgufHashCommand.execute(
                GgufHashCommand.parse(new String[] {"--check", manifest.toString(), file.toString()}), out);
        assertEquals(GgufHashCommand.EXIT_FAILURE, code);
        assertTrue(out.toString(), out.toString().contains("  -  Mismatch"));
        assertTrue(out.toString(), out.toString().contains(" - Failure"));
    }

    @Test
    public void aManifestWithoutMatchingTargetsReportsAMissingEntry() throws Exception {
        Path file = fixture().writeTempFile();
        Path manifest = tmp.newFile("other.manifest").toPath();
        Files.write(manifest, "xxh64    0123456789abcdef  other.gguf\n".getBytes(StandardCharsets.UTF_8));
        StringBuilder out = new StringBuilder();
        int code = GgufHashCommand.execute(
                GgufHashCommand.parse(new String[] {"--check", manifest.toString(), file.toString()}), out);
        assertEquals(GgufHashCommand.EXIT_MANIFEST_MISSING_ENTRY, code);
        assertTrue(out.toString(), out.toString().contains(" - Manifest Missing Entry"));
    }

    @Test
    public void aManifestWithoutKnownHashTypesIsRefused() throws Exception {
        Path file = fixture().writeTempFile();
        Path manifest = tmp.newFile("unknown.manifest").toPath();
        Files.write(manifest, "md5  abc  other.gguf\n".getBytes(StandardCharsets.UTF_8));
        StringBuilder out = new StringBuilder();
        int code = GgufHashCommand.execute(
                GgufHashCommand.parse(new String[] {"--check", manifest.toString(), file.toString()}), out);
        assertEquals(GgufHashCommand.EXIT_MANIFEST_UNKNOWN_HASH, code);
        assertTrue(out.toString(), out.toString().contains("ERROR manifest does not have any known hash format"));
    }

    @Test
    public void aMissingManifestIsARefusedFile() throws Exception {
        Path file = fixture().writeTempFile();
        StringBuilder out = new StringBuilder();
        int code = GgufHashCommand.execute(GgufHashCommand.parse(new String[] {
                "--check", file.resolveSibling("missing.manifest").toString(), file.toString()}), out);
        assertEquals(GgufHashCommand.EXIT_MANIFEST_FILE_ERROR, code);
        assertTrue(out.toString(), out.toString().contains("ERROR cannot open manifest"));
    }

    @Test
    public void anExplicitAlgorithmVerifiesOnlyThatType() throws Exception {
        Path file = fixture().writeTempFile();
        Path manifest = writeManifest(file, "--xxh64");
        StringBuilder out = new StringBuilder();
        int code = GgufHashCommand.execute(GgufHashCommand.parse(new String[] {
                "--check", manifest.toString(), "--xxh64", file.toString()}), out);
        assertEquals(0, code);
        assertTrue(out.toString(), out.toString().contains("manifest  " + manifest + "  xxh64\n"));
        for (String line : out.toString().split("\n")) {
            if (!line.startsWith("manifest") && !line.isEmpty() && !line.startsWith("Verification")) {
                assertTrue(line, line.startsWith("xxh64"));
            }
        }
    }

    @Test
    public void uuidHashesTheModelOnly() throws Exception {
        Path file = fixture().writeTempFile();
        StringBuilder out = new StringBuilder();
        assertEquals(0, GgufHashCommand.execute(
                GgufHashCommand.parse(new String[] {"--uuid", file.toString()}), out));
        String[] lines = out.toString().split("\n");
        assertEquals(1, lines.length);
        assertTrue(lines[0], lines[0].startsWith("uuid      "));
        assertTrue(lines[0], lines[0].endsWith("  " + file));
        assertTrue(lines[0], lines[0].matches("uuid\\s+[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab]"
                + "[0-9a-f]{3}-[0-9a-f]{12}  .*"));
    }

    // ================================================================== helpers

    private Path writeManifest(Path file, String... flags) throws Exception {
        String[] args = new String[flags.length + 1];
        System.arraycopy(flags, 0, args, 0, flags.length);
        args[flags.length] = file.toString();
        StringBuilder out = new StringBuilder();
        assertEquals(0, GgufHashCommand.execute(GgufHashCommand.parse(args), out));
        Path manifest = tmp.newFile("model-" + flags[0].replace("-", "") + ".manifest").toPath();
        Files.write(manifest, out.toString().getBytes(StandardCharsets.UTF_8));
        return manifest;
    }

    private static void assertUsage(String[] args, String message) {
        GgufDumpCommand.UsageException e = assertThrows(GgufDumpCommand.UsageException.class,
                () -> GgufDumpCommand.parse(args));
        assertTrue(e.getMessage(), e.getMessage().contains(message));
    }

    private static void assertHashUsage(String[] args, String message) {
        GgufHashCommand.UsageException e = assertThrows(GgufHashCommand.UsageException.class,
                () -> GgufHashCommand.parse(args));
        assertTrue(e.getMessage(), e.getMessage().contains(message));
    }

    private static GgufTestWriter fixture() {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("synthetic");
        w.u32("general.alignment", 32);
        w.tensorF32("t1", new long[]{4}, 1.0f, 2.0f, 3.0f, 4.0f);
        w.tensorF32("t2", new long[]{2, 2}, 5.0f, 6.0f, 7.0f, 8.0f);
        return w;
    }
}
