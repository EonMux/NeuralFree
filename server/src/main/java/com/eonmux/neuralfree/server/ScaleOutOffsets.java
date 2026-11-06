package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.tokenizer.StreamingDecoder;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.List;

/**
 * The character offsets a render endpoint reports for its prompt tokens (vLLM's
 * {@code return_token_offsets}): for every token, the half-open {@code (start, end)} span of the rendered prompt
 * text the token covers. The token pieces are read from the tokenizer's raw bytes, so special tokens the template
 * wrote literally (a {@code <|im_start|>}) cover their text while a token with no source span (an inserted BOS)
 * reports {@code (0, 0)}, as vLLM's fast-tokenizer offsets render special tokens.
 *
 * <p>Offsets are computed over the whole prompt before {@code truncate_prompt_tokens} and cut with it, so a
 * left-truncated prompt keeps the offsets of the tokens it kept.</p>
 */
final class ScaleOutOffsets {

    private ScaleOutOffsets() {
    }

    /**
     * The offsets of {@code tokens} in {@code text}, one entry per token. A token piece that does not appear in the
     * text at the cursor (a tokenizer reading additions the source does not hold) takes the zero-width span at the
     * cursor.
     */
    static List<int[]> of(EngineService service, String text, int[] tokens) {
        List<String> pieces = pieces(service, tokens);
        List<int[]> out = new ArrayList<>(tokens.length);
        int cursor = 0;
        for (String piece : pieces) {
            if (piece.isEmpty()) {
                out.add(new int[] {0, 0});
                continue;
            }
            int start = locate(text, cursor, piece);
            if (start < 0) {
                out.add(new int[] {cursor, cursor});
                continue;
            }
            int length = matchedLength(text, start, piece);
            int end = Math.min(text.length(), start + length);
            // the wire offsets count Unicode code points, as a Python tokenizer's offset mapping does
            out.add(new int[] {text.codePointCount(0, start), text.codePointCount(0, end)});
            cursor = Math.max(cursor, end);
        }
        return out;
    }

    /**
     * {@code offsets} cut like {@code tokens} was: the first {@code truncate_prompt_tokens} entries on the
     * default right side, the last ones on the left; no truncation returns {@code offsets} unchanged.
     */
    static List<int[]> slice(List<int[]> offsets, int[] tokens, JsonView body) {
        if (!body.has("truncate_prompt_tokens") || offsets.size() == tokens.length) {
            return offsets;
        }
        int keep = tokens.length;
        if (keep >= offsets.size()) {
            return offsets;
        }
        boolean left = "left".equals(body.optStr("truncation_side", "right"));
        return new ArrayList<>(offsets.subList(left ? offsets.size() - keep : 0, left ? offsets.size() : keep));
    }

    /** {@code offsets} as JSON arrays, or null. */
    static List<Object> json(List<int[]> offsets) {
        if (offsets == null) {
            return null;
        }
        List<Object> out = new ArrayList<>(offsets.size());
        for (int[] offset : offsets) {
            out.add(List.of(offset[0], offset[1]));
        }
        return out;
    }

    /** The text each token contributes, complete UTF-8 characters only. */
    private static List<String> pieces(EngineService service, int[] tokens) {
        StreamingDecoder decoder = new StreamingDecoder(service::tokenPiece);
        StringBuilder[] pieces = new StringBuilder[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            pieces[i] = new StringBuilder(decoder.decodeToken(tokens[i]));
        }
        String tail = decoder.flush();
        if (!tail.isEmpty() && tokens.length > 0) {
            pieces[tokens.length - 1].append(tail);
        }
        List<String> out = new ArrayList<>(tokens.length);
        for (StringBuilder piece : pieces) {
            out.add(piece.toString());
        }
        return out;
    }

    /** Where {@code piece} starts at or after {@code cursor}, or -1. */
    private static int locate(String text, int cursor, String piece) {
        if (text.startsWith(piece, cursor)) {
            return cursor;
        }
        // the sequence decoder drops the first piece's leading space a SentencePiece or Metaspace tokenizer adds
        if (piece.startsWith(" ") && text.startsWith(piece.substring(1), cursor)) {
            return cursor;
        }
        int at = text.indexOf(piece, cursor);
        if (at >= 0) {
            return at;
        }
        String stripped = piece.stripLeading();
        return stripped.isEmpty() ? -1 : text.indexOf(stripped, cursor);
    }

    /** The characters {@code piece} covers at {@code start}: the piece, or without a dropped leading space. */
    private static int matchedLength(String text, int start, String piece) {
        if (text.startsWith(piece, start)) {
            return piece.length();
        }
        if (piece.startsWith(" ") && text.startsWith(piece.substring(1), start)) {
            return piece.length() - 1;
        }
        String stripped = piece.stripLeading();
        return Math.min(piece.length(), text.length() - start) - (piece.length() - stripped.length());
    }
}
