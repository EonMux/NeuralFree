package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.OnlineQuant;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.engine.DraftHeadOptions;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.quant.OnlineQuantEncoders;
import com.eonmux.neuralfree.gguf.quant.OnlineQuantizer;
import com.eonmux.neuralfree.gguf.quant.OnlineQuantizers;
import com.eonmux.neuralfree.gguf.quant.ScanningQuantizer;
import com.eonmux.neuralfree.utils.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The selection and encoding steps of a derived draft head ({@link DerivedDraftSource}): the kept target ids of a
 * candidate vocabulary — the mandatory special ids, the ids of the frequency ranking in rank order and an
 * ascending fill of the rest (the BPE merge-rank prior), stored ascending so the device readout's
 * {@code d2t} stays ascending — the ranking file reader, and the row encoders that store the derived rows as
 * {@link DraftHeadOptions.Prec#Q8_0}, {@link DraftHeadOptions.Prec#FP8} or {@link DraftHeadOptions.Prec#NVFP4}
 * (through the nvfp4_per_token encoder, whose per-tensor global scale rides beside the rows as an output
 * multiplier). Stateless.
 */
final class DraftHeadQuantizer {

    private DraftHeadQuantizer() {
    }

    /**
     * The {@code size} kept target ids of a vocabulary of {@code nVocab}, ascending: {@code mandatoryIds} first
     * (the draft's special ids and the user's keep list), then {@code rankedIds} in rank order (skipping the kept
     * ones), then the smallest unused ids ascending.
     *
     * @param sizeKey the property that named {@code size}, for the refusals
     * @throws IllegalArgumentException when {@code size} reduces nothing, exceeds the vocabulary, or the mandatory
     *                                  ids alone do not fit
     */
    static int[] subset(int nVocab, int size, int[] rankedIds, int[] mandatoryIds, String sizeKey) {
        if (size <= 0 || size >= nVocab) {
            throw new IllegalArgumentException(sizeKey + "=" + size + " keeps " + size + " rows of a "
                    + nVocab + "-row head, which reduces nothing; drop the option or set it below " + nVocab);
        }
        boolean[] kept = new boolean[nVocab];
        int n = 0;
        for (int id : Objects.requireNonNull(mandatoryIds, "mandatoryIds")) {
            requireId(id, nVocab, "a mandatory draft-head token id");
            n += kept[id] ? 0 : 1;
            kept[id] = true;
        }
        if (n > size) {
            throw new IllegalArgumentException(sizeKey + "=" + size + " is smaller than the " + n
                    + " mandatory draft-head token ids (the draft's special tokens and the keep list)");
        }
        if (rankedIds != null) {
            for (int id : rankedIds) {
                requireId(id, nVocab, "a ranked draft-head token id");
                if (!kept[id] && n < size) {
                    kept[id] = true;
                    n++;
                }
            }
        }
        for (int v = 0; v < nVocab && n < size; v++) {
            if (!kept[v]) {
                kept[v] = true;
                n++;
            }
        }
        int[] out = new int[n];
        int i = 0;
        for (int v = 0; v < nVocab; v++) {
            if (kept[v]) {
                out[i++] = v;
            }
        }
        return out;
    }

    /**
     * The ranked token ids of a frequency file: a JSON object with a {@code "ranked_ids"} array (SpecPilot's
     * calibration export), a bare JSON array of ids, or text lines of {@code "<id>"} or {@code "<id> <count>"},
     * most frequent first. The counts are not read; the order ranks.
     *
     * @throws IOException              when the file cannot be read
     * @throws IllegalArgumentException naming the file and the offending line or entry when an id is not an
     *                                  integer inside the vocabulary, is repeated, or the file holds neither form
     */
    static int[] readRankedIds(Path file, int nVocab) throws IOException {
        byte[] bytes = Files.readAllBytes(file);
        String text = new String(bytes, StandardCharsets.UTF_8).trim();
        if (text.startsWith("{") || text.startsWith("[")) {
            return jsonRankedIds(file, text, nVocab);
        }
        return lineRankedIds(file, text, nVocab);
    }

    /** The ids of a JSON ranking file. */
    private static int[] jsonRankedIds(Path file, String text, int nVocab) {
        Object parsed;
        try {
            parsed = Json.parse(text, Json.Limits.UNTRUSTED);
        } catch (RuntimeException e) {
            throw new IllegalArgumentException("the draft-head ranking file " + file + " is not valid JSON: "
                    + e.getMessage(), e);
        }
        List<?> ids;
        if (parsed instanceof Map) {
            Object ranked = ((Map<?, ?>) parsed).get("ranked_ids");
            if (!(ranked instanceof List)) {
                throw new IllegalArgumentException("the draft-head ranking file " + file + " has no"
                        + " \"ranked_ids\" array");
            }
            ids = (List<?>) ranked;
        } else if (parsed instanceof List) {
            ids = (List<?>) parsed;
        } else {
            throw new IllegalArgumentException("the draft-head ranking file " + file + " holds neither a"
                    + " \"ranked_ids\" object nor an id array");
        }
        int[] out = new int[ids.size()];
        boolean[] seen = new boolean[nVocab];
        for (int i = 0; i < ids.size(); i++) {
            int id = jsonId(file, ids.get(i), i);
            requireId(id, nVocab, "the draft-head ranking file " + file + " entry " + i);
            if (seen[id]) {
                throw new IllegalArgumentException("the draft-head ranking file " + file + " ranks token " + id
                        + " more than once");
            }
            seen[id] = true;
            out[i] = id;
        }
        return out;
    }

    /** One JSON entry as an integer token id. */
    private static int jsonId(Path file, Object v, int i) {
        if (!(v instanceof Number)) {
            throw new IllegalArgumentException("the draft-head ranking file " + file + " entry " + i + " is "
                    + (v == null ? "null" : v.toString()) + ", not a token id");
        }
        double d = ((Number) v).doubleValue();
        if (d != Math.rint(d) || d < 0 || d > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("the draft-head ranking file " + file + " entry " + i + " is "
                    + v + ", not a token id");
        }
        return (int) d;
    }

    /** The ids of a line ranking file. */
    private static int[] lineRankedIds(Path file, String text, int nVocab) {
        String[] lines = text.split("\n", -1);
        int[] out = new int[lines.length];
        boolean[] seen = new boolean[nVocab];
        int n = 0;
        for (int i = 0; i < lines.length; i++) {
            String line = lines[i].trim();
            if (line.isEmpty()) {
                continue;
            }
            int space = line.indexOf(' ');
            String idPart = space < 0 ? line : line.substring(0, space);
            int id;
            try {
                id = Integer.parseInt(idPart);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("the draft-head ranking file " + file + " line " + (i + 1)
                        + " does not start with a token id: " + line, e);
            }
            requireId(id, nVocab, "the draft-head ranking file " + file + " line " + (i + 1));
            if (seen[id]) {
                throw new IllegalArgumentException("the draft-head ranking file " + file + " ranks token " + id
                        + " more than once");
            }
            seen[id] = true;
            out[n++] = id;
        }
        return Arrays.copyOf(out, n);
    }

    private static void requireId(int id, int nVocab, String what) {
        if (id < 0 || id >= nVocab) {
            throw new IllegalArgumentException(what + " is " + id + ", outside the vocabulary of " + nVocab);
        }
    }

    /**
     * The storage type of one derived row of {@code cols} values at {@code prec}: Q8_0, NVFP4, or FP8's
     * F8_E4M3_B128 for rows of whole 128-value blocks and F8_E4M3_B32 otherwise.
     *
     * @throws IllegalArgumentException when a row of {@code cols} values is no whole number of the type's blocks
     */
    static GGMLType rowType(DraftHeadOptions.Prec prec, int cols) {
        if (prec == DraftHeadOptions.Prec.Q8_0) {
            requireWholeBlocks(GGMLType.Q8_0, cols);
            return GGMLType.Q8_0;
        }
        if (prec == DraftHeadOptions.Prec.NVFP4) {
            requireWholeBlocks(GGMLType.NVFP4, cols);
            return GGMLType.NVFP4;
        }
        GGMLType type = cols % 128 == 0 ? GGMLType.F8_E4M3_B128 : GGMLType.F8_E4M3_B32;
        requireWholeBlocks(type, cols);
        return type;
    }

    private static void requireWholeBlocks(GGMLType type, int cols) {
        if (cols <= 0 || cols % type.getBlockSize() != 0) {
            throw new IllegalArgumentException("a row of " + cols + " values is no whole number of " + type
                    + " blocks of " + type.getBlockSize() + ": the draft head precision cannot store the head");
        }
    }

    /** The rows of a derived tensor: reads row {@code r} of {@code rows} into {@code dst[0..cols)}. */
    interface Rows {
        void read(int r, float[] dst);
    }

    /** The packed rows of a derived tensor and the output multiplier of its precision, when it carries one. */
    static final class Encoded {
        private final byte[] bytes;
        private final float[] outputScale;

        private Encoded(byte[] bytes, float[] outputScale) {
            this.bytes = bytes;
            this.outputScale = outputScale;
        }

        /** The packed rows, one {@link #rowType} row per input row, little-endian. */
        byte[] bytes() {
            return bytes;
        }

        /** The output multiplier of the rows (the NVFP4 global scale), or null when the blocks carry the full scale. */
        float[] outputScale() {
            return outputScale == null ? null : outputScale.clone();
        }
    }

    /**
     * The packed bytes of {@code rows} rows of {@code cols} values at {@code prec}, one {@link #rowType} row per
     * input row, little-endian. NVFP4 scans the whole tensor first: its global scale is per tensor, as vLLM's
     * {@code draft_lm_head_quantization=nvfp4} computes it over the whole head.
     *
     * @throws IllegalArgumentException when a row of {@code cols} values is no whole number of the type's blocks
     */
    static Encoded encode(DraftHeadOptions.Prec prec, int rows, int cols, Rows source) {
        GGMLType type = rowType(prec, cols);
        long rowBytes = BlockLayout.of(type).rowBytes(cols);
        if (rowBytes > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("a " + type + " row of " + cols + " values needs " + rowBytes
                    + " bytes");
        }
        return prec == DraftHeadOptions.Prec.NVFP4
                ? encodeNvfp4(type, rows, cols, source)
                : encodeRows(prec, type, rows, cols, source);
    }

    /** The Q8_0 or FP8 rows: one row-local quantizer call per input row, the blocks carrying the full scale. */
    private static Encoded encodeRows(DraftHeadOptions.Prec prec, GGMLType type, int rows, int cols, Rows source) {
        OnlineQuantizer q = OnlineQuantizers.get(prec == DraftHeadOptions.Prec.Q8_0 ? OnlineQuant.Q8_0
                : OnlineQuant.FP8_PER_CHANNEL);
        if (!q.typeFor(cols).equals(type)) {
            throw new IllegalArgumentException("the " + prec + " row quantizer stores a row of " + cols
                    + " values as " + q.typeFor(cols) + ", not " + type);
        }
        int bytes = Math.toIntExact(BlockLayout.of(type).rowBytes(cols));
        byte[] out = new byte[Math.toIntExact((long) bytes * rows)];
        float[] x = new float[cols];
        for (int r = 0; r < rows; r++) {
            source.read(r, x);
            q.quantizeRow(x, 0, cols, out, r * bytes);
        }
        return new Encoded(out, null);
    }

    /**
     * The NVFP4 rows of the nvfp4_per_token encoder over one tensor: the scan of every row sets the per-tensor
     * global scale, {@code freeze} fixes it and the quantize pass stores the rows under it; the scale is the
     * output multiplier the {@code <base>.scale} sibling carries.
     */
    private static Encoded encodeNvfp4(GGMLType type, int rows, int cols, Rows source) {
        ScanningQuantizer q = OnlineQuantEncoders.nvfp4PerToken().open(rows, cols, 1);
        int bytes = Math.toIntExact(BlockLayout.of(type).rowBytes(cols));
        byte[] out = new byte[Math.toIntExact((long) bytes * rows)];
        float[] x = new float[cols];
        for (int r = 0; r < rows; r++) {
            source.read(r, x);
            q.scanRow(r, x, 0, cols);
        }
        q.freeze();
        for (int r = 0; r < rows; r++) {
            source.read(r, x);
            q.quantizeRow(r, x, 0, cols, out, r * bytes);
        }
        float[] scale = q.outputScale();
        if (scale == null || scale.length != 1 || !Float.isFinite(scale[0]) || scale[0] <= 0.0f) {
            throw new IllegalStateException("the nvfp4 row quantizer of a " + rows + " x " + cols
                    + " tensor carries no single positive global scale");
        }
        return new Encoded(out, scale);
    }

    /**
     * The packed bytes of the {@code keptIds} rows of {@code src}, copied verbatim in the tensor's own storage: a
     * GGUF matrix quantizes each row on its own, so the kept rows are exactly the rows the drafter would have read.
     */
    static byte[] copyRows(CpuTensorRef src, int[] keptIds, int cols) {
        GGMLType type = src.type();
        long rowBytes = BlockLayout.of(type).rowBytes(cols);
        byte[] out = new byte[Math.toIntExact(rowBytes * keptIds.length)];
        for (int r = 0; r < keptIds.length; r++) {
            java.nio.ByteBuffer row = src.rowBuffer(keptIds[r]);
            int base = src.rowOffset(keptIds[r]);
            int dstOff = (int) (r * rowBytes);
            for (int i = 0; i < (int) rowBytes; i++) {
                out[dstOff + i] = row.get(base + i);
            }
        }
        return out;
    }
}
