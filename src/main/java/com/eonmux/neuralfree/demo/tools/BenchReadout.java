package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.GreedyStep;

import java.util.Locale;

/**
 * What a timed generation step of {@code nftool bench} reads back ({@code --tg-readout}): the whole logits row, as
 * llama-bench's decode leaves it in the host buffer; the device's greedy id
 * ({@link ForwardBackend#decodeGreedyOnDevice}), the readout of greedy generation; or the device's top candidates
 * ({@link ForwardBackend#decodeTopKOnDevice}), the readout of a sampler chain whose selection fits a cone. In those
 * modes the next token comes from the bench's fixed-seed generator, so they run the same token sequence. The
 * {@code greedy} mode runs greedy generation itself: each step decodes the previous step's greedy id through
 * {@link ForwardBackend#chainGreedy}, as the generation loop does, so a backend that chains on the device enqueues
 * each step before the previous id reaches the host. Immutable.
 */
final class BenchReadout {

    /** The readout kinds. */
    enum Kind {
        LOGITS, ARGMAX, TOPK, GREEDY
    }

    /** The default readout, the reference's. */
    static final BenchReadout LOGITS = new BenchReadout(Kind.LOGITS, 0);

    /** The largest top-k cone accepted. */
    static final int MAX_TOP_K = 1024;

    private final Kind kind;
    private final int topK;

    private BenchReadout(Kind kind, int topK) {
        this.kind = kind;
        this.topK = topK;
    }

    /**
     * A {@code --tg-readout} value: {@code logits}, {@code argmax}, {@code greedy} or {@code topk=<k>} with
     * {@code 1 <= k <=} {@value #MAX_TOP_K}.
     *
     * @throws BenchCommand.UsageException naming {@code flag} for any other value
     */
    static BenchReadout parse(String value, String flag) throws BenchCommand.UsageException {
        String v = value.trim().toLowerCase(Locale.ROOT);
        if (v.equals("logits")) {
            return LOGITS;
        }
        if (v.equals("argmax")) {
            return new BenchReadout(Kind.ARGMAX, 0);
        }
        if (v.equals("greedy")) {
            return new BenchReadout(Kind.GREEDY, 0);
        }
        if (v.startsWith("topk=")) {
            int k;
            try {
                k = Integer.parseInt(v.substring("topk=".length()));
            } catch (NumberFormatException e) {
                throw new BenchCommand.UsageException(flag + " " + value + ": the top-k cone must be an integer");
            }
            if (k < 1 || k > MAX_TOP_K) {
                throw new BenchCommand.UsageException(flag + " " + value + ": the top-k cone must be between 1 and "
                        + MAX_TOP_K);
            }
            return new BenchReadout(Kind.TOPK, k);
        }
        throw new BenchCommand.UsageException(flag + " " + value + " is invalid; use logits, argmax, greedy or"
                + " topk=<k>");
    }

    /** The readout kind. */
    Kind kind() {
        return kind;
    }

    /** The top-k cone of a {@link Kind#TOPK} readout, 0 otherwise. */
    int topK() {
        return topK;
    }

    /** Whether the steps after the first decode the previous step's greedy id ({@link Kind#GREEDY}). */
    boolean feedsBack() {
        return kind == Kind.GREEDY;
    }

    /**
     * Greedy generation of {@code n} steps from {@code token} at {@code pos}: every later step decodes the previous
     * step's greedy id at the next position, enqueued through {@link ForwardBackend#chainGreedy} before that id is
     * read, the generation loop's order.
     */
    static void generateGreedy(ForwardBackend backend, int token, int pos, int n) {
        if (n < 1) {
            return;
        }
        GreedyStep step = backend.startGreedy(token, pos);
        for (int i = 1; i < n; i++) {
            GreedyStep next = backend.chainGreedy(step, pos + i);
            step.id();
            step = next;
        }
        step.id();
    }

    /** One timed decode of {@code token} at {@code pos} with this readout. */
    void decode(ForwardBackend backend, int token, int pos) {
        switch (kind) {
            case ARGMAX:
            case GREEDY:
                backend.decodeGreedyOnDevice(token, pos);
                return;
            case TOPK:
                backend.decodeTopKOnDevice(token, pos, topK);
                return;
            default:
                backend.decode(token, pos);
        }
    }

    @Override
    public String toString() {
        return kind == Kind.TOPK ? "topk=" + topK : kind.name().toLowerCase(Locale.ROOT);
    }
}
