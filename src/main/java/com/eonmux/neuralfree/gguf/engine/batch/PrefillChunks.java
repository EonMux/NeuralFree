package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;

import java.util.Arrays;
import java.util.List;

/**
 * The prefill pieces of a {@link BatchEngine} step and the cursor they advance: a text-only prompt's pending rows
 * slice into budget-bounded chunks, a media prompt's pending rows schedule chunk by chunk within the step's
 * encoder budget (a media item's rows attend each other, so an item cannot split; a text or prompt-embedding
 * chunk the reused prefix ends inside resumes at its first unreused row), and a recompute replays the prompt and
 * then the fed decode tokens. Each slot's {@code prefillRows} counts the rows already written; a media chunk piece
 * runs as its own forward batch.
 */
final class PrefillChunks {

    private PrefillChunks() {
    }

    /**
     * The step's prefill piece of a text-only {@code slot}: the next {@code rows} rows of the prompt tail, or of
     * the history a recompute replays; only the piece that completes the prefill asks for logits.
     */
    static Piece chunkPiece(Slot slot, int rows) {
        Submission sub = slot.submission;
        int rowFrom = slot.recompute ? slot.prefillRows : sub.reused + slot.prefillRows;
        int[] tokens = Arrays.copyOfRange(slot.recompute ? slot.history() : sub.prompt.samplerTokens(), rowFrom,
                rowFrom + rows);
        // the rows take the prefill's position cursor: a self-extend compression moved it off the row index
        int from = slot.nextPromptPos;
        boolean finalChunk = slot.prefillRows + rows >= slot.pendingTotal();
        Piece piece = new Piece(slot, false, rows, from, from, tokens, finalChunk && !slot.recompute, finalChunk);
        piece.promptRowBase = rowFrom; // a fresh slice's rows are prompt rows; a recompute replays the prompt first
        piece.readRows = slot.submission.wantsPromptLogprobs();
        return piece;
    }

    static int chunkPieces(Slot slot, List<Piece> pieces, int encoderLeft, int encoderBudget) {
        return chunkPieces(slot, pieces, encoderLeft, encoderBudget, -1);
    }

    /**
     * Schedules a media prompt's pending prefill as chunk pieces from its first unwritten row (past a reused
     * prefix and the rows already written), each as its own forward batch, within the step's encoder budget
     * (vLLM's {@code encoder_compute_budget}). A media item ({@link MediaChunk#isMediaItem()}) prefills whole —
     * its rows attend each other (the shared base position, or the non-causal mask), the rule that also keeps a
     * reused prefix from ending inside it ({@link CacheRows#cut}) — and schedules only while the budget
     * {@code encoderLeft} still holds its rows: an unschedulable item ends the walk, so the chunks after it wait
     * for a later step and nothing prefills past an unwritten item. An item that alone exceeds the step's whole
     * budget {@code encoderBudget} schedules while the step's budget is still untouched (vLLM raises the budget to
     * the largest item, so the item always fits) and leaves it spent for every later media item of the step, in
     * this prompt or another. Text and prompt-embedding chunks take no encoder budget (vLLM's budget caps encoder
     * output only) and run from the cursor on when it sits inside them, while the step's token budget stays with
     * the decodes and the text prompts' chunks. The fed decode tokens a recompute replays run after the prompt's
     * chunks as one text piece. Returns the media item rows scheduled.
     *
     * <p>Over a memory with context checkpoints, {@code stop} is the prefill's next checkpoint row (absolute) or
     * -1: the walk ends there, a text or prompt-embedding chunk is cut at it, and a media item it falls inside
     * runs whole with the checkpoint at the item's end row — the checkpoint then holds the state at exactly the
     * row the walk stopped at.</p>
     */
    static int chunkPieces(Slot slot, List<Piece> pieces, int encoderLeft, int encoderBudget, int stop) {
        MediaPrompt prompt = slot.submission.prompt;
        int base = slot.recompute ? 0 : slot.submission.reused;
        int end = base + slot.pendingTotal();
        int row = base + slot.prefillRows;
        int scheduled = 0;
        while (row < end) {
            if (slot.recompute && row >= prompt.nTokens()) {
                Piece tail = fedTailPiece(slot, row, end);
                pieces.add(tail);
                row += tail.rows;
                continue;
            }
            ChunkCursor at = chunkAt(prompt, row);
            int skip = row - at.rowBase;
            MediaChunk c = rest(prompt.chunks().get(at.chunk), skip, at.posBase + skip);
            if (stop > row) {
                if (c.isMediaItem()) {
                    if (!fitsEncoderBudget(c.nTokens(), encoderLeft, encoderBudget)) {
                        break; // the item waits for a later step; nothing past it prefills this step
                    }
                    if (row + c.nTokens() > stop) {
                        // the checkpoint row sits inside the item: the item runs whole, the checkpoint lands
                        // at its end, and nothing past it schedules this step
                        encoderLeft -= c.nTokens();
                        scheduled += c.nTokens();
                        boolean finalChunk = row + c.nTokens() >= end;
                        Piece piece = Piece.mediaChunk(slot, c, row, at.posBase + skip,
                                finalChunk && !slot.recompute, finalChunk);
                        piece.promptRowBase = row;
                        piece.readRows = slot.submission.wantsPromptLogprobs();
                        pieces.add(piece);
                        slot.pendingCheckpointRow = row + c.nTokens();
                        return scheduled;
                    }
                } else if (row + c.nTokens() > stop) {
                    c = cut(c, stop - row, at.posBase + skip);
                }
            }
            if (c.isMediaItem()) {
                if (!fitsEncoderBudget(c.nTokens(), encoderLeft, encoderBudget)) {
                    break; // the item waits for a later step; nothing past it prefills this step
                }
                encoderLeft -= c.nTokens();
                scheduled += c.nTokens();
            }
            boolean finalChunk = row + c.nTokens() >= end;
            Piece piece = Piece.mediaChunk(slot, c, row, at.posBase + skip, finalChunk && !slot.recompute,
                    finalChunk);
            piece.promptRowBase = row; // a chunk's rows sit at their prompt places (fresh or replayed)
            piece.readRows = slot.submission.wantsPromptLogprobs();
            pieces.add(piece);
            row += c.nTokens();
            if (stop > 0 && row == stop && stop < end) {
                slot.pendingCheckpointRow = stop;
                break; // the checkpoint captures at the walk's stop; the rest schedules after it
            }
        }
        return scheduled;
    }

    /**
     * The first {@code rows} rows of {@code chunk} at position {@code pos}: a text chunk's leading tokens or a
     * prompt-embedding run's leading rows. Media items never cut (the caller schedules them whole).
     */
    private static MediaChunk cut(MediaChunk chunk, int rows, int pos) {
        if (chunk.isText()) {
            return MediaChunk.text(Arrays.copyOfRange(chunk.tokens(), 0, rows));
        }
        float[][] embeddings = chunk.embeddings();
        return MediaChunk.embeds(pos, Arrays.copyOfRange(embeddings, 0, rows));
    }

    /**
     * Whether a media item of {@code rows} rows schedules with {@code left} of the step's encoder budget
     * {@code budget} remaining: it fits what is left, or it alone exceeds the whole budget and the step has
     * scheduled no media rows yet (vLLM's budget is at least the largest multimodal item).
     */
    private static boolean fitsEncoderBudget(int rows, int left, int budget) {
        return rows <= left || (rows > budget && left >= budget);
    }

    /**
     * The rows of {@code chunk} from row {@code skip} on, at position {@code pos} and after: the chunk itself at
     * its start, a text chunk's later tokens, or a prompt-embedding run's later rows.
     *
     * @throws IllegalStateException when the cursor sits inside a media item, which prefills whole
     */
    private static MediaChunk rest(MediaChunk chunk, int skip, int pos) {
        if (skip == 0) {
            return chunk;
        }
        if (chunk.isText()) {
            int[] tokens = chunk.tokens();
            return MediaChunk.text(Arrays.copyOfRange(tokens, skip, tokens.length));
        }
        if (!chunk.isMediaItem()) {
            float[][] rows = chunk.embeddings();
            return MediaChunk.embeds(pos, Arrays.copyOfRange(rows, skip, rows.length));
        }
        throw new IllegalStateException("the prefill cursor sits " + skip + " rows inside a media item of "
                + chunk.nTokens() + " rows, which prefills whole");
    }

    /**
     * The preempted run's fed decodes a media recompute replays from replay row {@code rows} up to {@code end}:
     * plain token rows at consecutive positions past the prompt, completing the recompute.
     */
    private static Piece fedTailPiece(Slot slot, int rows, int end) {
        MediaPrompt prompt = slot.submission.prompt;
        int fed = rows - prompt.nTokens();
        int take = end - rows;
        int from = prompt.nPositions() + fed;
        return new Piece(slot, false, take, from, rows, Arrays.copyOfRange(slot.fedTokens(), fed, fed + take),
                false, true);
    }

    /**
     * Walks the prompt's chunk list to the chunk holding prompt row {@code row}: a media prompt's prefill
     * advances by whole chunks after the first, so only a reused prefix leaves {@code row} inside a chunk (a text
     * or prompt-embedding chunk; never a media item, see {@link CacheRows#cut}).
     */
    private static ChunkCursor chunkAt(MediaPrompt prompt, int row) {
        int r = 0;
        int pos = 0;
        List<MediaChunk> chunks = prompt.chunks();
        for (int i = 0; i < chunks.size(); i++) {
            MediaChunk c = chunks.get(i);
            if (r + c.nTokens() > row) {
                return new ChunkCursor(i, r, pos);
            }
            r += c.nTokens();
            pos += c.nPositions();
        }
        throw new IllegalStateException("prompt row " + row + " is past the prompt's " + prompt.nTokens()
                + " rows");
    }

    /** A chunk's place in its prompt: its index and the row and position the chunk starts at. */
    private static final class ChunkCursor {
        final int chunk;
        final int rowBase;
        final int posBase;

        ChunkCursor(int chunk, int rowBase, int posBase) {
            this.chunk = chunk;
            this.rowBase = rowBase;
            this.posBase = posBase;
        }
    }

    /**
     * A chunk without an output row committed its rows: the cursor advances, and the chunk that completes a
     * recompute leaves the armed decode token to feed at {@code nextPos} on the next pump.
     */
    static void chunkWritten(Slot slot, Piece p) {
        slot.prefillRows += p.rows;
        slot.nextPromptPos = p.from + p.rows;
        if (p.finalChunk) {
            slot.needsPrefill = false;
            slot.recompute = false;
        }
    }

    /**
     * A media prompt's chunk piece as its own forward batch: a text chunk's tokens at consecutive positions, a
     * media chunk's embedding rows at their positions (section 0 of the chunk's media positions, the temporal
     * coordinate the memory stores) with the chunk's position sections — for a multi-section model — and its
     * non-causal flag, every row in the sequence's adapter slot. Only the last row is an output row when
     * {@code p.output} is set; a prompt-logprobs piece asks every row's logits.
     */
    static ForwardBatch chunkBatch(Piece p) {
        MediaChunk c = p.chunk;
        int n = c.nTokens();
        ForwardBatch.Builder batch = ForwardBatch.builder(n);
        if (c.isText()) {
            int[] tokens = c.tokens();
            for (int i = 0; i < n; i++) {
                batch.add(tokens[i], p.posBase + i, p.slot.seq, p.readRows || (p.output && i == n - 1));
            }
            return withAdapterSlot(batch, p, n).build();
        }
        MediaRopePositions pos = c.positions();
        float[][] rows = c.embeddings();
        for (int i = 0; i < n; i++) {
            batch.addEmbedding(rows[i], pos.position(0, i), p.slot.seq,
                    p.readRows || (p.output && i == n - 1));
        }
        if (pos.kind() != MediaRopePositions.Kind.NORMAL) {
            batch.ropePositions(pos.sections());
        }
        if (c.nonCausal()) {
            batch.nonCausal();
        }
        return withAdapterSlot(batch, p, n).build();
    }

    /** Sets the {@code n} rows of {@code batch} in their adapter slots, when the submission runs one. */
    private static ForwardBatch.Builder withAdapterSlot(ForwardBatch.Builder batch, Piece p, int n) {
        Submission sub = p.slot.submission;
        if (sub.runsAdapterSlots()) {
            int[] slots = new int[n];
            for (int i = 0; i < n; i++) {
                slots[i] = sub.adapterSlot(p.rowBase + i);
            }
            batch.adapterSlots(slots);
        }
        return batch;
    }
}
