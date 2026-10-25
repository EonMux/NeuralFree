package com.eonmux.neuralfree.gguf.tokenizer.hf;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable view of a SentencePiece {@code ModelProto} ({@code tokenizer.model}, {@code spiece.model},
 * {@code sentencepiece.bpe.model}), read by a dependency-free protobuf wire reader.
 *
 * <p>Field numbers and proto2 defaults are those of {@code sentencepiece_model.proto}: {@code ModelProto} pieces = 1,
 * trainer_spec = 2, normalizer_spec = 3, denormalizer_spec = 5; {@code SentencePiece} piece = 1, score = 2,
 * type = 3 (default NORMAL); {@code TrainerSpec} model_type = 3 (UNIGRAM), vocab_size = 4 (8000),
 * treat_whitespace_as_suffix = 24, split_digits = 25, byte_fallback = 35 (all false), unk_id = 40 (0), bos_id = 41 (1),
 * eos_id = 42 (2), pad_id = 43 (-1), unk/bos/eos/pad_piece = 45..48 ({@code <unk> <s> </s> <pad>});
 * {@code NormalizerSpec} name = 1, precompiled_charsmap = 2, add_dummy_prefix = 3, remove_extra_whitespaces = 4,
 * escape_whitespaces = 5 (all true). As protobuf does, unknown fields are skipped by wire type, a known field with an
 * unexpected wire type is an unknown field, a repeated message field merges, the last scalar wins, and an enum value
 * outside the enum leaves the field at its default. Every length is bounds-checked and strings must be UTF-8. The
 * pieces must satisfy SentencePiece's own load rules (non-empty, unique, exactly one UNKNOWN piece).</p>
 */
public final class SentencePieceModel {

    /** {@code SentencePiece.Type}. */
    public enum PieceType {
        NORMAL(1), UNKNOWN(2), CONTROL(3), USER_DEFINED(4), UNUSED(5), BYTE(6);

        private final int code;

        PieceType(int code) {
            this.code = code;
        }

        public int code() { return code; }

        static PieceType ofCode(long code) {
            for (PieceType t : values()) {
                if (t.code == code) {
                    return t;
                }
            }
            return null;
        }
    }

    /** {@code TrainerSpec.ModelType}. */
    public enum ModelType {
        UNIGRAM(1), BPE(2), WORD(3), CHAR(4);

        private final int code;

        ModelType(int code) {
            this.code = code;
        }

        public int code() { return code; }

        static ModelType ofCode(long code) {
            for (ModelType t : values()) {
                if (t.code == code) {
                    return t;
                }
            }
            return null;
        }
    }

    /** The {@code TrainerSpec} fields tokenization depends on. */
    public static final class TrainerSpec {
        private final ModelType modelType;
        private final int vocabSize;
        private final boolean treatWhitespaceAsSuffix;
        private final boolean splitDigits;
        private final boolean byteFallback;
        private final int unkId;
        private final int bosId;
        private final int eosId;
        private final int padId;
        private final String unkPiece;
        private final String bosPiece;
        private final String eosPiece;
        private final String padPiece;

        private TrainerSpec(TrainerFields f) {
            this.modelType = f.modelType;
            this.vocabSize = f.vocabSize;
            this.treatWhitespaceAsSuffix = f.treatWhitespaceAsSuffix;
            this.splitDigits = f.splitDigits;
            this.byteFallback = f.byteFallback;
            this.unkId = f.unkId;
            this.bosId = f.bosId;
            this.eosId = f.eosId;
            this.padId = f.padId;
            this.unkPiece = f.unkPiece;
            this.bosPiece = f.bosPiece;
            this.eosPiece = f.eosPiece;
            this.padPiece = f.padPiece;
        }

        public ModelType modelType() { return modelType; }

        /** The trained vocabulary size ({@code vocab_size}); the piece count is {@link #pieceCount()}. */
        public int vocabSize() { return vocabSize; }

        public boolean treatWhitespaceAsSuffix() { return treatWhitespaceAsSuffix; }

        public boolean splitDigits() { return splitDigits; }

        public boolean byteFallback() { return byteFallback; }

        public int unkId() { return unkId; }

        public int bosId() { return bosId; }

        public int eosId() { return eosId; }

        /** {@code pad_id}; -1 when the model has no padding piece. */
        public int padId() { return padId; }

        public String unkPiece() { return unkPiece; }

        public String bosPiece() { return bosPiece; }

        public String eosPiece() { return eosPiece; }

        public String padPiece() { return padPiece; }
    }

    /** A {@code NormalizerSpec} (also the denormalizer). */
    public static final class NormalizerSpec {
        private final String name;
        private final byte[] precompiledCharsmap;
        private final boolean addDummyPrefix;
        private final boolean removeExtraWhitespaces;
        private final boolean escapeWhitespaces;

        private NormalizerSpec(NormalizerFields f) {
            this.name = f.name;
            this.precompiledCharsmap = f.precompiledCharsmap;
            this.addDummyPrefix = f.addDummyPrefix;
            this.removeExtraWhitespaces = f.removeExtraWhitespaces;
            this.escapeWhitespaces = f.escapeWhitespaces;
        }

        /** {@code name}, or null when unset. */
        public String name() { return name; }

        /** {@code precompiled_charsmap} (a copy); empty when unset. */
        public byte[] precompiledCharsmap() { return precompiledCharsmap.clone(); }

        public boolean addDummyPrefix() { return addDummyPrefix; }

        public boolean removeExtraWhitespaces() { return removeExtraWhitespaces; }

        public boolean escapeWhitespaces() { return escapeWhitespaces; }
    }

    /** TrainerSpec fields while parsing; repeated occurrences of the message merge into one. */
    private static final class TrainerFields {
        ModelType modelType = ModelType.UNIGRAM;
        int vocabSize = 8000;
        boolean treatWhitespaceAsSuffix;
        boolean splitDigits;
        boolean byteFallback;
        int unkId = 0;
        int bosId = 1;
        int eosId = 2;
        int padId = -1;
        String unkPiece = "<unk>";
        String bosPiece = "<s>";
        String eosPiece = "</s>";
        String padPiece = "<pad>";
    }

    /** NormalizerSpec fields while parsing; repeated occurrences of the message merge into one. */
    private static final class NormalizerFields {
        String name;
        byte[] precompiledCharsmap = new byte[0];
        boolean addDummyPrefix = true;
        boolean removeExtraWhitespaces = true;
        boolean escapeWhitespaces = true;
    }

    /** Most pieces a model may hold. */
    public static final int MAX_PIECES = HfTokenizerFiles.MAX_TOKENS;

    private static final int WIRE_VARINT = 0;
    private static final int WIRE_FIXED64 = 1;
    private static final int WIRE_LEN = 2;
    private static final int WIRE_START_GROUP = 3;
    private static final int WIRE_END_GROUP = 4;
    private static final int WIRE_FIXED32 = 5;
    private static final int MAX_GROUP_DEPTH = 64;

    private final String[] pieces;
    private final float[] scores;
    private final PieceType[] types;
    private final TrainerSpec trainerSpec;
    private final NormalizerSpec normalizerSpec;
    private final NormalizerSpec denormalizerSpec;

    private SentencePieceModel(String[] pieces, float[] scores, PieceType[] types, TrainerSpec trainerSpec,
                               NormalizerSpec normalizerSpec, NormalizerSpec denormalizerSpec) {
        this.pieces = pieces;
        this.scores = scores;
        this.types = types;
        this.trainerSpec = trainerSpec;
        this.normalizerSpec = normalizerSpec;
        this.denormalizerSpec = denormalizerSpec;
    }

    /**
     * Parses a serialized {@code ModelProto}.
     *
     * @throws IOException {@code "Invalid SentencePiece model: <detail> at byte <n>"} when the bytes are not one
     */
    public static SentencePieceModel parse(byte[] data) throws IOException {
        Objects.requireNonNull(data, "data");
        return parse(data, "Invalid SentencePiece model: ");
    }

    /**
     * Reads {@code file} (at most {@link HfTokenizerFiles#MAX_FILE_BYTES}).
     *
     * @throws IOException {@code "Invalid Hugging Face checkpoint '<dir>': <file>: <detail>"} when it is not a model
     */
    public static SentencePieceModel read(Path file) throws IOException {
        Objects.requireNonNull(file, "file");
        byte[] data = HfTokenizerFiles.readBytes(file);
        String prefix = HfTokenizerFiles.filePrefix(file);
        return parse(data, prefix);
    }

    /** Number of pieces; the ids are {@code 0 .. pieceCount() - 1}. */
    public int pieceCount() { return pieces.length; }

    public String piece(int id) { return pieces[id]; }

    public float score(int id) { return scores[id]; }

    public PieceType type(int id) { return types[id]; }

    /** The pieces by id (a copy). */
    public String[] pieces() { return pieces.clone(); }

    /** The scores by id (a copy). */
    public float[] scores() { return scores.clone(); }

    /** The piece types by id (a copy). */
    public PieceType[] types() { return types.clone(); }

    /** The trainer spec; the proto2 defaults when the model has none. */
    public TrainerSpec trainerSpec() { return trainerSpec; }

    /** The normalizer spec; the proto2 defaults when the model has none. */
    public NormalizerSpec normalizerSpec() { return normalizerSpec; }

    /** The denormalizer spec, or null when the model has none. */
    public NormalizerSpec denormalizerSpec() { return denormalizerSpec; }

    @Override
    public String toString() {
        return "SentencePieceModel{" + trainerSpec.modelType + ", pieces=" + pieces.length + '}';
    }

    // ------------------------------------------------------------------ ModelProto

    private static SentencePieceModel parse(byte[] data, String prefix) throws IOException {
        Wire in = new Wire(data, 0, data.length, prefix);
        List<String> pieces = new ArrayList<>();
        List<Float> scores = new ArrayList<>();
        List<PieceType> types = new ArrayList<>();
        TrainerFields trainer = new TrainerFields();
        NormalizerFields normalizer = new NormalizerFields();
        NormalizerFields denormalizer = null;
        while (in.hasMore()) {
            long tag = in.varint();
            int field = in.fieldOf(tag);
            int wire = (int) (tag & 7);
            if (wire == WIRE_LEN && field == 1) {
                if (pieces.size() == MAX_PIECES) {
                    throw in.error("more than " + MAX_PIECES + " pieces");
                }
                readPiece(in.message(), pieces, scores, types);
            } else if (wire == WIRE_LEN && field == 2) {
                readTrainerSpec(in.message(), trainer);
            } else if (wire == WIRE_LEN && field == 3) {
                readNormalizerSpec(in.message(), normalizer);
            } else if (wire == WIRE_LEN && field == 5) {
                if (denormalizer == null) {
                    denormalizer = new NormalizerFields();
                }
                readNormalizerSpec(in.message(), denormalizer);
            } else {
                in.skip(field, wire);
            }
        }
        validatePieces(pieces, types, prefix);
        int n = pieces.size();
        float[] scoreArray = new float[n];
        for (int i = 0; i < n; i++) {
            scoreArray[i] = scores.get(i);
        }
        return new SentencePieceModel(pieces.toArray(new String[0]), scoreArray, types.toArray(new PieceType[0]),
                new TrainerSpec(trainer), new NormalizerSpec(normalizer),
                denormalizer == null ? null : new NormalizerSpec(denormalizer));
    }

    /**
     * The piece rules of SentencePiece's {@code ModelInterface::InitializePieces}: no empty piece; a piece is defined
     * once among the NORMAL / USER_DEFINED / UNUSED pieces and once among the others; exactly one UNKNOWN piece.
     */
    private static void validatePieces(List<String> pieces, List<PieceType> types, String prefix) throws IOException {
        Map<String, Integer> normal = new HashMap<>();
        Map<String, Integer> reserved = new HashMap<>();
        int unk = -1;
        for (int i = 0; i < pieces.size(); i++) {
            String piece = pieces.get(i);
            PieceType type = types.get(i);
            if (piece.isEmpty()) {
                throw new IOException(prefix + "piece " + i + " is empty");
            }
            boolean isNormal = type == PieceType.NORMAL || type == PieceType.USER_DEFINED || type == PieceType.UNUSED;
            Integer previous = (isNormal ? normal : reserved).putIfAbsent(piece, i);
            if (previous != null) {
                throw new IOException(prefix + "piece " + i + " '" + piece + "' is already defined as piece "
                        + previous);
            }
            if (type == PieceType.UNKNOWN) {
                if (unk >= 0) {
                    throw new IOException(prefix + "pieces " + unk + " and " + i + " are both UNKNOWN");
                }
                unk = i;
            }
        }
        if (unk < 0) {
            throw new IOException(prefix + "no piece is UNKNOWN");
        }
    }

    private static void readPiece(Wire in, List<String> pieces, List<Float> scores, List<PieceType> types)
            throws IOException {
        String piece = null;
        float score = 0f;
        PieceType type = PieceType.NORMAL;
        while (in.hasMore()) {
            long tag = in.varint();
            int field = in.fieldOf(tag);
            int wire = (int) (tag & 7);
            if (field == 1 && wire == WIRE_LEN) {
                piece = in.string();
            } else if (field == 2 && wire == WIRE_FIXED32) {
                score = Float.intBitsToFloat(in.fixed32());
            } else if (field == 3 && wire == WIRE_VARINT) {
                PieceType t = PieceType.ofCode(in.varint());
                if (t != null) {
                    type = t;
                }
            } else {
                in.skip(field, wire);
            }
        }
        pieces.add(piece == null ? "" : piece);
        scores.add(score);
        types.add(type);
    }

    private static void readTrainerSpec(Wire in, TrainerFields t) throws IOException {
        while (in.hasMore()) {
            long tag = in.varint();
            int field = in.fieldOf(tag);
            int wire = (int) (tag & 7);
            if (wire == WIRE_VARINT && readTrainerScalar(in, field, t)) {
                continue;
            }
            if (wire == WIRE_LEN && field >= 45 && field <= 48) {
                String s = in.string();
                switch (field) {
                    case 45: t.unkPiece = s; break;
                    case 46: t.bosPiece = s; break;
                    case 47: t.eosPiece = s; break;
                    default: t.padPiece = s; break;
                }
                continue;
            }
            in.skip(field, wire);
        }
    }

    /** Reads a varint field of TrainerSpec; false (nothing consumed) when {@code field} is not one this view keeps. */
    private static boolean readTrainerScalar(Wire in, int field, TrainerFields t) throws IOException {
        switch (field) {
            case 3: {
                ModelType m = ModelType.ofCode(in.varint());
                if (m != null) {
                    t.modelType = m;
                }
                return true;
            }
            case 4: t.vocabSize = (int) in.varint(); return true;
            case 24: t.treatWhitespaceAsSuffix = in.varint() != 0; return true;
            case 25: t.splitDigits = in.varint() != 0; return true;
            case 35: t.byteFallback = in.varint() != 0; return true;
            case 40: t.unkId = (int) in.varint(); return true;
            case 41: t.bosId = (int) in.varint(); return true;
            case 42: t.eosId = (int) in.varint(); return true;
            case 43: t.padId = (int) in.varint(); return true;
            default: return false;
        }
    }

    private static void readNormalizerSpec(Wire in, NormalizerFields s) throws IOException {
        while (in.hasMore()) {
            long tag = in.varint();
            int field = in.fieldOf(tag);
            int wire = (int) (tag & 7);
            if (field == 1 && wire == WIRE_LEN) {
                s.name = in.string();
            } else if (field == 2 && wire == WIRE_LEN) {
                s.precompiledCharsmap = in.bytes();
            } else if (field == 3 && wire == WIRE_VARINT) {
                s.addDummyPrefix = in.varint() != 0;
            } else if (field == 4 && wire == WIRE_VARINT) {
                s.removeExtraWhitespaces = in.varint() != 0;
            } else if (field == 5 && wire == WIRE_VARINT) {
                s.escapeWhitespaces = in.varint() != 0;
            } else {
                in.skip(field, wire);
            }
        }
    }

    // ------------------------------------------------------------------ wire reader

    /** A bounds-checked cursor over {@code data[pos, limit)}. */
    private static final class Wire {
        private final byte[] data;
        private final int limit;
        private final String prefix;
        private int pos;

        Wire(byte[] data, int pos, int limit, String prefix) {
            this.data = data;
            this.pos = pos;
            this.limit = limit;
            this.prefix = prefix;
        }

        boolean hasMore() {
            return pos < limit;
        }

        IOException error(String detail) {
            return new IOException(prefix + detail + " at byte " + pos);
        }

        /** The field number of {@code tag}; 0 and numbers above 2^29 - 1 are invalid. */
        int fieldOf(long tag) throws IOException {
            long field = tag >>> 3;
            if (field == 0 || field > 0x1FFFFFFFL) {
                throw error("invalid field number " + field);
            }
            return (int) field;
        }

        long varint() throws IOException {
            long result = 0;
            for (int shift = 0; shift < 64; shift += 7) {
                if (pos >= limit) {
                    throw error("truncated varint");
                }
                int b = data[pos++] & 0xFF;
                if (shift == 63 && (b & 0x7E) != 0) {
                    throw error("varint overflows 64 bits");
                }
                result |= (long) (b & 0x7F) << shift;
                if ((b & 0x80) == 0) {
                    return result;
                }
            }
            throw error("varint longer than 10 bytes");
        }

        int fixed32() throws IOException {
            require(4);
            int v = ByteBuffer.wrap(data, pos, 4).order(ByteOrder.LITTLE_ENDIAN).getInt();
            pos += 4;
            return v;
        }

        /** The length of a length-delimited field, checked against the bytes left. */
        int length() throws IOException {
            long len = varint();
            if (len < 0 || len > limit - pos) {
                throw error("length " + Long.toUnsignedString(len) + " exceeds the " + (limit - pos)
                        + " bytes left");
            }
            return (int) len;
        }

        Wire message() throws IOException {
            int len = length();
            Wire sub = new Wire(data, pos, pos + len, prefix);
            pos += len;
            return sub;
        }

        byte[] bytes() throws IOException {
            int len = length();
            byte[] out = Arrays.copyOfRange(data, pos, pos + len);
            pos += len;
            return out;
        }

        String string() throws IOException {
            int start = pos;
            int len = length();
            int at = pos;
            pos += len;
            try {
                return StandardCharsets.UTF_8.newDecoder()
                        .onMalformedInput(CodingErrorAction.REPORT)
                        .onUnmappableCharacter(CodingErrorAction.REPORT)
                        .decode(ByteBuffer.wrap(data, at, len)).toString();
            } catch (CharacterCodingException e) {
                pos = start;
                throw error("string is not valid UTF-8");
            }
        }

        /** Skips one field of {@code wire} type; groups are skipped to their matching end tag. */
        void skip(int field, int wire) throws IOException {
            skip(field, wire, 0);
        }

        private void skip(int field, int wire, int depth) throws IOException {
            switch (wire) {
                case WIRE_VARINT:
                    varint();
                    return;
                case WIRE_FIXED64:
                    require(8);
                    pos += 8;
                    return;
                case WIRE_LEN:
                    pos += length();
                    return;
                case WIRE_FIXED32:
                    require(4);
                    pos += 4;
                    return;
                case WIRE_START_GROUP:
                    if (depth >= MAX_GROUP_DEPTH) {
                        throw error("groups nested deeper than " + MAX_GROUP_DEPTH);
                    }
                    while (true) {
                        if (!hasMore()) {
                            throw error("group " + field + " is not terminated");
                        }
                        long tag = varint();
                        int f = fieldOf(tag);
                        int w = (int) (tag & 7);
                        if (w == WIRE_END_GROUP) {
                            if (f != field) {
                                throw error("end of group " + f + " inside group " + field);
                            }
                            return;
                        }
                        skip(f, w, depth + 1);
                    }
                case WIRE_END_GROUP:
                    throw error("unexpected end of group " + field);
                default:
                    throw error("invalid wire type " + wire + " for field " + field);
            }
        }

        private void require(int n) throws IOException {
            if (limit - pos < n) {
                throw error("truncated field: " + n + " bytes needed, " + (limit - pos) + " left");
            }
        }
    }
}
