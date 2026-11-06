package com.eonmux.neuralfree.gguf.inspect;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Test;

import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * {@link GgufHasher} over a real GGUF fixture: the per-tensor and model digests of every algorithm against the
 * reference algorithms computed on the fixture's own tensor bytes, the reference's line order and targets, and
 * {@code --no-layer}.
 */
public class GgufHasherTest {

    private static final Set<GgufHasher.Algorithm> ALL = EnumSet.of(GgufHasher.Algorithm.XXH64,
            GgufHasher.Algorithm.SHA1, GgufHasher.Algorithm.SHA256, GgufHasher.Algorithm.UUID);

    @Test
    public void everyAlgorithmMatchesTheReferenceOnTheTensorBytes() throws Exception {
        GgufTestWriter writer = fixture();
        Path file = writer.writeTempFile();
        byte[] a = writer.tensorData("t.a");
        byte[] b = writer.tensorData("t.b");
        byte[] all = concat(a, b);
        try (GGUFFileParser src = ModelSources.parseGguf(file.toString())) {
            GgufHasher.Result result = GgufHasher.hash(src, ALL, false);
            List<GgufHasher.Line> lines = result.lines();
            assertEquals(10, lines.size());

            assertEquals(GgufHasher.Algorithm.XXH64, lines.get(0).algorithm());
            assertEquals(Xxh64.hex(Xxh64.hash(a, 0, a.length)), lines.get(0).hash());
            assertEquals(file + ":t.a", lines.get(0).target());

            assertEquals("sha1 a", digest("SHA-1", a), lines.get(1).hash());
            assertEquals(GgufHasher.Algorithm.SHA256, lines.get(2).algorithm());
            assertEquals("sha256 a", digest("SHA-256", a), lines.get(2).hash());

            assertEquals(file + ":t.b", lines.get(3).target());
            assertEquals(digest("SHA-1", b), lines.get(4).hash());

            assertEquals(GgufHasher.Algorithm.XXH64, lines.get(6).algorithm());
            assertEquals(file.toString(), lines.get(6).target());
            assertEquals(Xxh64.hex(Xxh64.hash(all, 0, all.length)), lines.get(6).hash());
            assertEquals(digest("SHA-1", all), lines.get(7).hash());
            assertEquals(digest("SHA-256", all), lines.get(8).hash());
            assertEquals(GgufHasher.Algorithm.UUID, lines.get(9).algorithm());
            assertEquals(UuidV5.of(UuidV5.NAMESPACE_LLAMA_CPP, all).toString(), lines.get(9).hash());
            assertEquals(file.toString(), lines.get(9).target());
        }
    }

    @Test
    public void noLayerKeepsOnlyTheModelLines() throws Exception {
        GgufTestWriter writer = fixture();
        Path file = writer.writeTempFile();
        try (GGUFFileParser src = ModelSources.parseGguf(file.toString())) {
            GgufHasher.Result result = GgufHasher.hash(src, ALL, true);
            assertEquals(Arrays.asList(GgufHasher.Algorithm.XXH64, GgufHasher.Algorithm.SHA1,
                    GgufHasher.Algorithm.SHA256, GgufHasher.Algorithm.UUID), algorithms(result));
            for (GgufHasher.Line line : result.lines()) {
                assertEquals(file.toString(), line.target());
            }
        }
    }

    @Test
    public void hashingUuidAloneProducesOneModelLine() throws Exception {
        Path file = fixture().writeTempFile();
        try (GGUFFileParser src = ModelSources.parseGguf(file.toString())) {
            GgufHasher.Result result = GgufHasher.hash(src, EnumSet.of(GgufHasher.Algorithm.UUID), false);
            assertEquals(1, result.lines().size());
            assertEquals(GgufHasher.Algorithm.UUID, result.lines().get(0).algorithm());
        }
    }

    @Test
    public void anEmptyAlgorithmSetIsRefused() throws Exception {
        Path file = fixture().writeTempFile();
        try (GGUFFileParser src = ModelSources.parseGguf(file.toString())) {
            assertThrows(IllegalArgumentException.class,
                    () -> GgufHasher.hash(src, EnumSet.noneOf(GgufHasher.Algorithm.class), false));
        }
    }

    @Test
    public void algorithmNamesRoundTrip() {
        for (GgufHasher.Algorithm a : GgufHasher.Algorithm.values()) {
            assertEquals(a, GgufHasher.Algorithm.of(a.text()));
        }
        assertEquals(null, GgufHasher.Algorithm.of("md5"));
    }

    private static List<GgufHasher.Algorithm> algorithms(GgufHasher.Result result) {
        List<GgufHasher.Algorithm> out = new java.util.ArrayList<>();
        for (GgufHasher.Line line : result.lines()) {
            out.add(line.algorithm());
        }
        return out;
    }

    private static GgufTestWriter fixture() {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("synthetic");
        w.str("general.name", "hash fixture");
        w.u32("general.alignment", 32);
        w.u32("tokenizer.ggml.bos_token_id", 1);
        w.tensorF32("t.a", new long[]{4}, 1.0f, 2.0f, 3.0f, 4.0f);
        w.tensorF32("t.b", new long[]{2, 2}, 5.0f, 6.0f, 7.0f, 8.0f);
        return w;
    }

    private static String digest(String algorithm, byte[] data) {
        try {
            return UuidV5.hex(MessageDigest.getInstance(algorithm).digest(data));
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    private static byte[] concat(byte[] a, byte[] b) {
        byte[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }
}
