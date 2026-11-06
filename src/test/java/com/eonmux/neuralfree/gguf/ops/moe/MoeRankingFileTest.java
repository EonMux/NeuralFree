package com.eonmux.neuralfree.gguf.ops.moe;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The {@link MoeRankingFile} codec: the version 2 round trip with its header derived from the entries and the
 * pairs hottest first, Avalon's version 2 layout read as written, the version 1 pairs read without heats in file
 * order, the bytes after the declared pairs ignored as {@code moe_cache_ranking_read} does, the named refusals (not
 * an expert ranking, a short header, unsupported version, truncated) and the atomic replacement that leaves no
 * {@code .tmp} sibling.
 */
public class MoeRankingFileTest {

    private Path file;

    @Before
    public void openFile() throws IOException {
        file = Files.createTempFile("moe-cache-ranking", ".bin");
    }

    @After
    public void removeFile() throws IOException {
        Files.deleteIfExists(file);
        Files.deleteIfExists(file.resolveSibling(file.getFileName() + ".tmp"));
    }

    @Test
    public void versionTwoRoundTripsHottestFirstWithItsHeader() throws IOException {
        MoeRankingFile.write(file, Arrays.asList(
                new RankedExpert(3, 5, 7),
                new RankedExpert(1, 2, 41),
                new RankedExpert(1, 9, 41)));
        MoeRankingFile.Parsed parsed = MoeRankingFile.read(file);
        assertEquals(Arrays.asList(
                new RankedExpert(1, 2, 41),
                new RankedExpert(1, 9, 41),
                new RankedExpert(3, 5, 7)), parsed.entries());
        assertEquals("layers = max layer + 1", 4, parsed.nLayers());
        assertEquals("experts = max expert + 1", 10, parsed.nExperts());
    }

    /** Equal heats order by layer, then expert; the written file is read back in its pair order. */
    @Test
    public void tiesOrderLayerThenExpertAndReadKeepsFileOrder() throws IOException {
        MoeRankingFile.write(file, Arrays.asList(
                new RankedExpert(9, 9, 3),
                new RankedExpert(0, 4, 3),
                new RankedExpert(0, 4, 1),
                new RankedExpert(2, 0, 3)));
        MoeRankingFile.Parsed parsed = MoeRankingFile.read(file);
        assertEquals(Arrays.asList(
                new RankedExpert(0, 4, 3),
                new RankedExpert(2, 0, 3),
                new RankedExpert(9, 9, 3),
                new RankedExpert(0, 4, 1)), parsed.entries());
    }

    /** Avalon's version 2 layout as {@code moe_cache_profile_save_heat} writes it: header counts and heats kept. */
    @Test
    public void avalonVersionTwoReadsHottestFirst() throws IOException {
        ByteBuffer out = header(2, 2, 8, 3, 3 * 8);
        out.putShort((short) 0).putShort((short) 3).putInt(900);
        out.putShort((short) 1).putShort((short) 5).putInt(400);
        out.putShort((short) 0).putShort((short) 1).putInt(70);
        Files.write(file, out.array());
        MoeRankingFile.Parsed parsed = MoeRankingFile.read(file);
        assertEquals(2, parsed.nLayers());
        assertEquals(8, parsed.nExperts());
        assertEquals(Arrays.asList(
                new RankedExpert(0, 3, 900),
                new RankedExpert(1, 5, 400),
                new RankedExpert(0, 1, 70)), parsed.entries());
    }

    /** Bytes after the declared pairs are not read, as Avalon reads only the declared pairs. */
    @Test
    public void bytesAfterTheDeclaredPairsAreIgnored() throws IOException {
        ByteBuffer out = header(2, 1, 4, 1, 8 + 6);
        out.putShort((short) 0).putShort((short) 2).putInt(5);
        Files.write(file, out.array());
        assertEquals(Collections.singletonList(new RankedExpert(0, 2, 5)), MoeRankingFile.read(file).entries());
    }

    @Test
    public void aShortHeaderIsRefusedByName() throws IOException {
        Files.write(file, new byte[]{'S', 'T', 'R', 'P'});
        IOException e = expectReadFailure();
        assertTrue(e.getMessage(), e.getMessage().contains("is not an expert ranking: 4 bytes"));
    }

    /** Strata's version 1: (layer, expert) pairs, heat 0, file order kept. */
    @Test
    public void versionOnePairsReadWithoutHeats() throws IOException {
        ByteBuffer out = ByteBuffer.allocate(24 + 8).order(ByteOrder.LITTLE_ENDIAN);
        out.put("STRP".getBytes(StandardCharsets.US_ASCII));
        out.putInt(1);
        out.putInt(2);
        out.putInt(4);
        out.putInt(2);
        out.putInt(2);
        out.putShort((short) 1);
        out.putShort((short) 3);
        out.putShort((short) 0);
        out.putShort((short) 0);
        Files.write(file, out.array());
        MoeRankingFile.Parsed parsed = MoeRankingFile.read(file);
        assertEquals(Arrays.asList(new RankedExpert(1, 3, 0), new RankedExpert(0, 0, 0)), parsed.entries());
        assertEquals(2, parsed.nLayers());
        assertEquals(4, parsed.nExperts());
    }

    @Test
    public void anEmptyRankingWritesTheHeaderAlone() throws IOException {
        MoeRankingFile.write(file, Collections.emptyList());
        MoeRankingFile.Parsed parsed = MoeRankingFile.read(file);
        assertEquals(Collections.emptyList(), parsed.entries());
        assertEquals(0, parsed.nLayers());
        assertEquals(0, parsed.nExperts());
    }

    @Test
    public void foreignBytesAreRefusedByName() throws IOException {
        Files.write(file, new byte[24]);
        IOException e = expectReadFailure();
        assertTrue(e.getMessage(), e.getMessage().contains("is not an expert ranking"));
    }

    @Test
    public void unknownVersionsAreRefusedByName() throws IOException {
        ByteBuffer out = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
        out.put("STRP".getBytes(StandardCharsets.US_ASCII));
        out.putInt(3);
        Files.write(file, out.array());
        IOException e = expectReadFailure();
        assertTrue(e.getMessage(), e.getMessage().contains("unsupported version 3"));
    }

    @Test
    public void truncatedFilesAreRefusedByName() throws IOException {
        ByteBuffer out = ByteBuffer.allocate(24 + 5).order(ByteOrder.LITTLE_ENDIAN);
        out.put("STRP".getBytes(StandardCharsets.US_ASCII));
        out.putInt(2);
        out.putInt(1);
        out.putInt(2);
        out.putInt(1);
        out.putInt(1);
        Files.write(file, out.array());
        IOException e = expectReadFailure();
        assertTrue(e.getMessage(), e.getMessage().contains("truncated"));
    }

    /** A second write replaces the first and leaves no {@code .tmp} sibling. */
    @Test
    public void writingReplacesTheFileWithoutATmpSibling() throws IOException {
        MoeRankingFile.write(file, Collections.singletonList(new RankedExpert(0, 0, 1)));
        MoeRankingFile.write(file, Collections.singletonList(new RankedExpert(0, 1, 2)));
        MoeRankingFile.Parsed parsed = MoeRankingFile.read(file);
        assertEquals(Collections.singletonList(new RankedExpert(0, 1, 2)), parsed.entries());
        assertTrue("no .tmp sibling remains", Files.notExists(file.resolveSibling(file.getFileName() + ".tmp")));
    }

    /** A pair beyond the format's {@code uint16} expert range is refused. */
    @Test(expected = IllegalArgumentException.class)
    public void expertIndexesBeyondTheFormatAreRefused() {
        new RankedExpert(0, 0x10000, 1);
    }

    /** A header of {@code version, layers, experts} declaring {@code pairs}, with room for {@code body} bytes. */
    private static ByteBuffer header(int version, int layers, int experts, int pairs, int body) {
        ByteBuffer out = ByteBuffer.allocate(24 + body).order(ByteOrder.LITTLE_ENDIAN);
        out.put("STRP".getBytes(StandardCharsets.US_ASCII));
        out.putInt(version);
        out.putInt(layers);
        out.putInt(experts);
        out.putInt(pairs);
        out.putInt(pairs);
        return out;
    }

    private IOException expectReadFailure() {
        try {
            MoeRankingFile.read(file);
        } catch (IOException e) {
            return e;
        }
        throw new AssertionError("the read of " + file + " should have failed");
    }
}
