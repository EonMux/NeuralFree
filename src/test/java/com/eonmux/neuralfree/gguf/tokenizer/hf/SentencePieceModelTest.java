package com.eonmux.neuralfree.gguf.tokenizer.hf;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link SentencePieceModel} on protobuf bytes built by the test: proto2 defaults for absent fields, explicit values
 * (including negative varints), skipping of unknown fields of every wire type, message merging, and rejection of
 * malformed input and of pieces SentencePiece itself refuses.
 */
public class SentencePieceModelTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** A minimal protobuf encoder for the test messages. */
    private static final class Proto {
        private final ByteArrayOutputStream out = new ByteArrayOutputStream();

        Proto varint(int field, long value) {
            tag(field, 0);
            raw(value);
            return this;
        }

        Proto fixed32(int field, float value) {
            tag(field, 5);
            int bits = Float.floatToIntBits(value);
            for (int i = 0; i < 4; i++) {
                out.write(bits >>> (8 * i));
            }
            return this;
        }

        Proto fixed64(int field, long value) {
            tag(field, 1);
            for (int i = 0; i < 8; i++) {
                out.write((int) (value >>> (8 * i)));
            }
            return this;
        }

        Proto bytes(int field, byte[] value) {
            tag(field, 2);
            raw(value.length);
            out.write(value, 0, value.length);
            return this;
        }

        Proto string(int field, String value) {
            return bytes(field, value.getBytes(StandardCharsets.UTF_8));
        }

        Proto message(int field, Proto value) {
            return bytes(field, value.toBytes());
        }

        Proto startGroup(int field) {
            tag(field, 3);
            return this;
        }

        Proto endGroup(int field) {
            tag(field, 4);
            return this;
        }

        byte[] toBytes() {
            return out.toByteArray();
        }

        private void tag(int field, int wire) {
            raw(((long) field << 3) | wire);
        }

        private void raw(long v) {
            while ((v & ~0x7FL) != 0) {
                out.write((int) ((v & 0x7F) | 0x80));
                v >>>= 7;
            }
            out.write((int) v);
        }
    }

    private static Proto piece(String text, float score, int type) {
        return new Proto().string(1, text).fixed32(2, score).varint(3, type);
    }

    /** {@code <unk>} UNKNOWN, {@code <s>} CONTROL, {@code ▁a} NORMAL. */
    private static Proto basePieces() {
        return new Proto()
                .message(1, piece("<unk>", 0f, 2))
                .message(1, piece("<s>", 0f, 3))
                .message(1, piece("▁a", -1.5f, 1));
    }

    private static void assertMessageContains(Throwable e, String expected) {
        assertTrue(e.getMessage(), e.getMessage().contains(expected));
    }

    @Test
    public void absentSpecsTakeTheProto2Defaults() throws IOException {
        SentencePieceModel m = SentencePieceModel.parse(basePieces().toBytes());
        assertEquals(3, m.pieceCount());
        assertArrayEquals(new String[]{"<unk>", "<s>", "▁a"}, m.pieces());
        assertArrayEquals(new float[]{0f, 0f, -1.5f}, m.scores(), 0f);
        assertEquals(SentencePieceModel.PieceType.UNKNOWN, m.type(0));
        assertEquals(SentencePieceModel.PieceType.CONTROL, m.type(1));
        assertEquals(SentencePieceModel.PieceType.NORMAL, m.type(2));

        SentencePieceModel.TrainerSpec t = m.trainerSpec();
        assertEquals(SentencePieceModel.ModelType.UNIGRAM, t.modelType());
        assertEquals(8000, t.vocabSize());
        assertEquals(0, t.unkId());
        assertEquals(1, t.bosId());
        assertEquals(2, t.eosId());
        assertEquals(-1, t.padId());
        assertEquals("<unk>", t.unkPiece());
        assertEquals("<s>", t.bosPiece());
        assertEquals("</s>", t.eosPiece());
        assertEquals("<pad>", t.padPiece());
        assertFalse(t.byteFallback());
        assertFalse(t.splitDigits());
        assertFalse(t.treatWhitespaceAsSuffix());

        SentencePieceModel.NormalizerSpec n = m.normalizerSpec();
        assertTrue("add_dummy_prefix absent -> true", n.addDummyPrefix());
        assertTrue(n.removeExtraWhitespaces());
        assertTrue(n.escapeWhitespaces());
        assertNull(n.name());
        assertEquals(0, n.precompiledCharsmap().length);
        assertNull(m.denormalizerSpec());
    }

    @Test
    public void aTrainerSpecWithoutPadIdKeepsMinusOne() throws IOException {
        Proto trainer = new Proto().varint(3, 2).varint(4, 32000).varint(35, 1);
        SentencePieceModel m = SentencePieceModel.parse(basePieces().message(2, trainer).toBytes());
        assertEquals(SentencePieceModel.ModelType.BPE, m.trainerSpec().modelType());
        assertEquals(32000, m.trainerSpec().vocabSize());
        assertTrue(m.trainerSpec().byteFallback());
        assertEquals(-1, m.trainerSpec().padId());
    }

    @Test
    public void explicitFieldsAreRead() throws IOException {
        byte[] charsmap = {1, 2, 3, 4};
        Proto trainer = new Proto().varint(24, 1).varint(25, 1).varint(40, 3).varint(41, 4).varint(42, 5)
                .varint(43, 0).string(45, "[UNK]").string(46, "[BOS]").string(47, "[EOS]").string(48, "[PAD]");
        Proto normalizer = new Proto().string(1, "nmt_nfkc").bytes(2, charsmap).varint(3, 0).varint(4, 0)
                .varint(5, 0);
        Proto denormalizer = new Proto().string(1, "identity");
        SentencePieceModel m = SentencePieceModel.parse(basePieces().message(2, trainer).message(3, normalizer)
                .message(5, denormalizer).toBytes());
        SentencePieceModel.TrainerSpec t = m.trainerSpec();
        assertTrue(t.treatWhitespaceAsSuffix());
        assertTrue(t.splitDigits());
        assertEquals(3, t.unkId());
        assertEquals(4, t.bosId());
        assertEquals(5, t.eosId());
        assertEquals(0, t.padId());
        assertEquals("[UNK]", t.unkPiece());
        assertEquals("[BOS]", t.bosPiece());
        assertEquals("[EOS]", t.eosPiece());
        assertEquals("[PAD]", t.padPiece());
        SentencePieceModel.NormalizerSpec n = m.normalizerSpec();
        assertEquals("nmt_nfkc", n.name());
        assertArrayEquals(charsmap, n.precompiledCharsmap());
        assertFalse(n.addDummyPrefix());
        assertFalse(n.removeExtraWhitespaces());
        assertFalse(n.escapeWhitespaces());
        assertNotNull(m.denormalizerSpec());
        assertEquals("identity", m.denormalizerSpec().name());
        assertTrue(m.denormalizerSpec().addDummyPrefix());
    }

    @Test
    public void negativeInt32ValuesUseTenByteVarints() throws IOException {
        Proto trainer = new Proto().varint(43, 5).varint(43, -1L).varint(41, -1L);
        SentencePieceModel m = SentencePieceModel.parse(basePieces().message(2, trainer).toBytes());
        assertEquals("the last value wins", -1, m.trainerSpec().padId());
        assertEquals(-1, m.trainerSpec().bosId());
    }

    @Test
    public void repeatedSpecMessagesMerge() throws IOException {
        SentencePieceModel m = SentencePieceModel.parse(basePieces()
                .message(2, new Proto().varint(3, 2))
                .message(2, new Proto().varint(4, 100))
                .message(3, new Proto().varint(3, 0))
                .message(3, new Proto().string(1, "nfkc"))
                .toBytes());
        assertEquals(SentencePieceModel.ModelType.BPE, m.trainerSpec().modelType());
        assertEquals(100, m.trainerSpec().vocabSize());
        assertFalse(m.normalizerSpec().addDummyPrefix());
        assertEquals("nfkc", m.normalizerSpec().name());
    }

    @Test
    public void unknownFieldsOfEveryWireTypeAreSkipped() throws IOException {
        Proto selfTest = new Proto().message(1, new Proto().string(1, "input").string(2, "expected"));
        Proto unknownPiece = piece("▁b", -2f, 1).varint(7, 1).string(8, "x");
        Proto trainer = new Proto().string(1, "corpus.txt").fixed32(10, 0.9995f).varint(3, 2)
                .fixed64(77, 42L).varint(43, 1);
        byte[] data = basePieces()
                .message(1, unknownPiece)
                .message(2, trainer)
                .message(4, selfTest)
                .varint(99, 12345)
                .fixed64(100, -1L)
                .fixed32(101, 1f)
                .string(102, "ignored")
                .startGroup(103).varint(1, 7).startGroup(2).string(3, "nested").endGroup(2).endGroup(103)
                .toBytes();
        SentencePieceModel m = SentencePieceModel.parse(data);
        assertEquals(4, m.pieceCount());
        assertEquals("▁b", m.piece(3));
        assertEquals(-2f, m.score(3), 0f);
        assertEquals(SentencePieceModel.ModelType.BPE, m.trainerSpec().modelType());
        assertEquals(1, m.trainerSpec().padId());
    }

    @Test
    public void aKnownFieldWithAnotherWireTypeIsAnUnknownField() throws IOException {
        Proto trainer = new Proto().string(43, "not a varint").varint(3, 2);
        SentencePieceModel m = SentencePieceModel.parse(basePieces().message(2, trainer).toBytes());
        assertEquals(-1, m.trainerSpec().padId());
        assertEquals(SentencePieceModel.ModelType.BPE, m.trainerSpec().modelType());
    }

    @Test
    public void enumValuesOutsideTheEnumKeepTheDefault() throws IOException {
        SentencePieceModel m = SentencePieceModel.parse(basePieces()
                .message(1, piece("▁c", 0f, 42))
                .message(2, new Proto().varint(3, 9))
                .toBytes());
        assertEquals(SentencePieceModel.PieceType.NORMAL, m.type(3));
        assertEquals(SentencePieceModel.ModelType.UNIGRAM, m.trainerSpec().modelType());
    }

    @Test
    public void copiesDoNotChangeTheModel() throws IOException {
        SentencePieceModel m = SentencePieceModel.parse(basePieces()
                .message(3, new Proto().bytes(2, new byte[]{9})).toBytes());
        m.pieces()[0] = "x";
        m.scores()[2] = 5f;
        m.normalizerSpec().precompiledCharsmap()[0] = 0;
        assertEquals("<unk>", m.piece(0));
        assertEquals(-1.5f, m.score(2), 0f);
        assertArrayEquals(new byte[]{9}, m.normalizerSpec().precompiledCharsmap());
    }

    @Test
    public void malformedBytesAreRejectedWithTheirPosition() {
        byte[] good = basePieces().toBytes();
        IOException truncated = assertThrows(IOException.class,
                () -> SentencePieceModel.parse(Arrays.copyOf(good, good.length - 3)));
        assertMessageContains(truncated, "Invalid SentencePiece model: ");
        assertMessageContains(truncated, " at byte ");

        byte[] lengthBomb = new byte[]{(byte) ((1 << 3) | 2), (byte) 0xFF, (byte) 0xFF, (byte) 0xFF, (byte) 0x7F};
        assertMessageContains(assertThrows(IOException.class, () -> SentencePieceModel.parse(lengthBomb)), "exceeds");

        byte[] longVarint = new byte[12];
        Arrays.fill(longVarint, (byte) 0x80);
        longVarint[0] = (byte) 0x08;
        assertThrows(IOException.class, () -> SentencePieceModel.parse(longVarint));

        assertThrows(IOException.class, () -> SentencePieceModel.parse(new byte[]{0}));
        assertThrows(IOException.class, () -> SentencePieceModel.parse(new byte[]{(byte) ((1 << 3) | 7)}));
        assertThrows(IOException.class,
                () -> SentencePieceModel.parse(basePieces().startGroup(9).varint(1, 1).toBytes()));
        assertThrows(IOException.class,
                () -> SentencePieceModel.parse(basePieces().startGroup(9).endGroup(10).toBytes()));
    }

    @Test
    public void invalidUtf8PiecesAreRejected() {
        Proto bad = new Proto().bytes(1, new byte[]{(byte) 0xE2, (byte) 0x96}).fixed32(2, 0f);
        IOException e = assertThrows(IOException.class,
                () -> SentencePieceModel.parse(basePieces().message(1, bad).toBytes()));
        assertMessageContains(e, "UTF-8");
    }

    @Test
    public void piecesSentencePieceRefusesAreRejected() {
        assertMessageContains(assertThrows(IOException.class, () -> SentencePieceModel.parse(basePieces()
                .message(1, piece("▁a", 0f, 4)).toBytes())), "already defined");
        assertMessageContains(assertThrows(IOException.class, () -> SentencePieceModel.parse(basePieces()
                .message(1, piece("", 0f, 1)).toBytes())), "is empty");
        assertMessageContains(assertThrows(IOException.class, () -> SentencePieceModel.parse(basePieces()
                .message(1, piece("<unk2>", 0f, 2)).toBytes())), "both UNKNOWN");
        assertMessageContains(assertThrows(IOException.class, () -> SentencePieceModel.parse(new Proto()
                .message(1, piece("a", 0f, 1)).toBytes())), "no piece is UNKNOWN");
    }

    @Test
    public void aControlPieceMayShareTextWithANormalPiece() throws IOException {
        SentencePieceModel m = SentencePieceModel.parse(basePieces().message(1, piece("<s>", 0f, 1)).toBytes());
        assertEquals(4, m.pieceCount());
    }

    @Test
    public void readNamesTheCheckpointAndFile() throws IOException {
        Path dir = tmp.newFolder().toPath();
        Path file = dir.resolve("tokenizer.model");
        Files.write(file, basePieces().toBytes());
        assertEquals(3, SentencePieceModel.read(file).pieceCount());
        Files.write(file, new byte[]{0x0A, 0x7F});
        IOException e = assertThrows(IOException.class, () -> SentencePieceModel.read(file));
        assertTrue(e.getMessage(), e.getMessage().startsWith("Invalid Hugging Face checkpoint '" + dir + "': "
                + file + ": "));
    }
}
