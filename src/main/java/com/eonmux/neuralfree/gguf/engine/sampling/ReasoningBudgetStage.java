package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.Sampler;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The reasoning-budget stage (llama.cpp common/reasoning-budget.cpp, vLLM's {@code thinking_token_budget}): it counts
 * the tokens generated inside a reasoning block and forces the end-tag token sequence when the budget runs out, so a
 * reasoning model cannot think past its budget. A Java-API custom stage ({@code Sampler.create(config, bannedTokens,
 * extra, vocab)}); built per generation by {@link #of} from the request's budget settings and the reasoning parser's
 * tags.
 *
 * <p>The state machine is llama.cpp's: {@code IDLE} watches for a start tag; {@code COUNTING} counts the budget down
 * while watching for a natural end tag; {@code WAITING_UTF8} lets tokens pass until one carries a complete UTF-8
 * sequence (the budget ran out mid character); {@code FORCING} masks every candidate but the next forced token — the
 * optional message, then the first end tag; {@code DONE} passes through and re-arms on a new start tag, so several
 * reasoning blocks each get a fresh budget. Prompt tokens flow through {@code accept} like generated ones, so a
 * generation prompt that opens reasoning starts with its budget armed. A parser whose reasoning opens at the output's
 * first token without a start tag ({@link ReasoningFilter#startsInReasoning}) arms the stage from that initial state
 * instead (llama.cpp's {@code initial_state} COUNTING): the stage arms when it first applies, after the prompt tokens.
 * {@link #force} moves a counting stage into {@code FORCING} from another thread — the mid-stream reasoning-end
 * control.
 */
public final class ReasoningBudgetStage extends SamplerStage {

    /** Where the stage is in its state machine (llama.cpp's common_reasoning_budget_state). */
    public enum State { IDLE, COUNTING, WAITING_UTF8, FORCING, DONE }

    /** The token sequences that open a reasoning block. */
    private final TokenMatcher starts;
    /** The token sequences that close one naturally. */
    private final TokenMatcher ends;
    /** The sequence forced when the budget runs out: the message, then the first end tag. */
    private final int[] forced;
    /** The budget, or {@link Integer#MAX_VALUE} when only a control action can force. */
    private final int budget;
    private final SamplerVocab vocab;

    /** Whether the output starts inside reasoning, so the stage arms before the first generated token. */
    private final boolean armsAtOutput;

    private volatile State state = State.IDLE;
    private volatile int forcePos;
    /** Whether the stage applied to a candidate set since its construction or reset: the output started. */
    private boolean outputStarted;
    private int remaining;
    private int endMatch = -1;
    /** Successful {@link #force} calls, so a restore keeps a control action that arrived after its save. */
    private int forces;

    private ReasoningBudgetStage(List<int[]> startSeqs, List<int[]> endSeqs, int[] forced, int budget,
                                 SamplerVocab vocab, boolean armsAtOutput) {
        this.armsAtOutput = armsAtOutput;
        this.starts = new TokenMatcher(startSeqs);
        this.ends = new TokenMatcher(endSeqs);
        this.forced = forced.clone();
        this.budget = budget;
        this.remaining = budget;
        this.vocab = vocab;
    }

    /**
     * The stage of {@code request} over {@code filter}'s tags, or null when the request arms no budget stage (neither
     * {@link GenerationRequest#reasoningControl()} nor a non-negative {@link GenerationRequest#reasoningBudget()}). The
     * request's explicit tags win over the parser's; a parser without tags and no explicit tags refuse by name. The
     * forcing sequence is the optional message followed by the first end tag.
     *
     * @throws IllegalArgumentException when the armed request resolves to no start or end tag, or a tag tokenizes to
     *                                 no tokens
     */
    public static ReasoningBudgetStage of(ReasoningFilter filter, GenerationRequest request, SamplerVocab vocab) {
        return of(filter, request, vocab, null);
    }

    /**
     * {@link #of(ReasoningFilter, GenerationRequest, SamplerVocab)} for a generation after {@code renderedPrompt}:
     * when {@code filter} starts that generation inside reasoning ({@link ReasoningFilter#startsInReasoning}), the
     * stage arms at the output's start even if no start tag passes through it.
     *
     * @param renderedPrompt the prompt as the reasoning parser sees it, or null when unknown
     * @throws IllegalArgumentException when the armed request resolves to no start or end tag, or a tag tokenizes to
     *                                 no tokens
     */
    public static ReasoningBudgetStage of(ReasoningFilter filter, GenerationRequest request, SamplerVocab vocab,
                                          String renderedPrompt) {
        Objects.requireNonNull(request, "request");
        if (!request.reasoningControl() && request.reasoningBudget() < 0) {
            return null;
        }
        Objects.requireNonNull(vocab, "vocab");
        List<String> startTags = resolveStarts(filter, request.reasoningBudgetStartTags());
        List<String> endTags = resolveEnds(filter, request.reasoningBudgetEndTags());
        if (startTags.isEmpty() || endTags.isEmpty()) {
            throw new IllegalArgumentException("the reasoning budget needs a start and an end tag: the model's"
                    + " reasoning parser names none and the request passes no reasoning_budget_start_tag /"
                    + " reasoning_budget_end_tags");
        }
        List<int[]> startSeqs = tokenizeAll(startTags, vocab, "reasoning_budget_start_tag");
        List<int[]> endSeqs = tokenizeAll(endTags, vocab, "reasoning_budget_end_tags");
        List<Integer> forcedIds = new ArrayList<>();
        String message = request.reasoningBudgetMessage();
        if (message != null && !message.isEmpty()) {
            for (int id : tokenizeOne(message, vocab, "reasoning_budget_message")) {
                forcedIds.add(id);
            }
        }
        for (int id : endSeqs.get(0)) {
            forcedIds.add(id);
        }
        int[] forced = new int[forcedIds.size()];
        for (int i = 0; i < forced.length; i++) {
            forced[i] = forcedIds.get(i);
        }
        int effective = request.reasoningBudget() >= 0 ? request.reasoningBudget() : Integer.MAX_VALUE;
        return new ReasoningBudgetStage(startSeqs, endSeqs, forced, effective, vocab,
                filter.startsInReasoning(renderedPrompt));
    }

    /** The start tags: the request's, else the parser's (a harmony model's analysis channel). */
    private static List<String> resolveStarts(ReasoningFilter filter, List<String> explicit) {
        Objects.requireNonNull(filter, "filter");
        if (!explicit.isEmpty()) {
            return explicit;
        }
        if (filter.mode() == ReasoningFilter.Mode.THINK_TAGS) {
            return filter.startTags();
        }
        if (filter.mode() == ReasoningFilter.Mode.HARMONY_CHANNELS) {
            return Collections.singletonList(ReasoningFilter.H_CHANNEL + "analysis" + ReasoningFilter.H_MESSAGE);
        }
        return Collections.emptyList();
    }

    /** The end tags: the request's, else the parser's (a harmony message's terminators, a new header included). */
    private static List<String> resolveEnds(ReasoningFilter filter, List<String> explicit) {
        Objects.requireNonNull(filter, "filter");
        if (!explicit.isEmpty()) {
            return explicit;
        }
        if (filter.mode() == ReasoningFilter.Mode.THINK_TAGS) {
            List<String> ends = new ArrayList<>(filter.endTags());
            ends.addAll(filter.interruptTags());
            return ends;
        }
        if (filter.mode() == ReasoningFilter.Mode.HARMONY_CHANNELS) {
            return List.of(ReasoningFilter.H_END, ReasoningFilter.H_RETURN, ReasoningFilter.H_CALL,
                    ReasoningFilter.M_EOM, ReasoningFilter.M_EOT, ReasoningFilter.H_START);
        }
        return Collections.emptyList();
    }

    private static List<int[]> tokenizeAll(List<String> tags, SamplerVocab vocab, String what) {
        List<int[]> out = new ArrayList<>(tags.size());
        for (String tag : tags) {
            out.add(tokenizeOne(tag, vocab, what));
        }
        return out;
    }

    private static int[] tokenizeOne(String tag, SamplerVocab vocab, String what) {
        int[] ids = vocab.tokenizeSpecial(Objects.requireNonNull(tag, what));
        if (ids.length == 0) {
            throw new IllegalArgumentException(what + " '" + tag + "' tokenizes to no tokens");
        }
        return ids;
    }

    /** The state machine's current state. */
    public State state() {
        return state;
    }

    /** Whether the stage currently forces the next token of its end sequence. */
    public boolean forcing() {
        return state == State.FORCING && forcePos < forced.length;
    }

    /**
     * Moves a counting stage into {@code FORCING} (llama.cpp's {@code common_reasoning_budget_force}, the mid-stream
     * reasoning-end control); every other state is left as it is.
     *
     * @return whether the transition happened
     */
    public synchronized boolean force() {
        if (state != State.COUNTING) {
            return false;
        }
        enterForcing();
        forces++;
        return true;
    }

    @Override
    public synchronized void accept(int token) {
        switch (state) {
            case IDLE: {
                if (starts.advance(token) >= 0) {
                    arm();
                }
                break;
            }
            case COUNTING:
            case WAITING_UTF8: {
                int hit = ends.advance(token);
                if (hit >= 0) {
                    state = State.DONE;
                    endMatch = hit;
                    break;
                }
                boolean complete = vocab == null || isCompleteUtf8(vocab.piece(token));
                if (state == State.WAITING_UTF8) {
                    if (complete) {
                        enterForcing();
                    }
                } else {
                    remaining--;
                    if (remaining <= 0) {
                        if (complete) {
                            enterForcing();
                        } else {
                            state = State.WAITING_UTF8;
                            ends.reset();
                        }
                    }
                }
                break;
            }
            case FORCING: {
                // the mask makes every sampled token the next forced one, so a token that is not it raced the
                // force() call in (sampled before the transition, accepted after) and the position re-masks
                if (forcePos < forced.length && token == forced[forcePos]) {
                    // an end tag already inside the forced sequence also completes it (llama.cpp tracks both)
                    int hit = ends.advance(token);
                    forcePos++;
                    if (hit >= 0 || forcePos >= forced.length) {
                        state = State.DONE;
                        endMatch = hit;
                    }
                } else if (forcePos >= forced.length) {
                    state = State.DONE;
                }
                break;
            }
            case DONE: {
                if (starts.advance(token) >= 0) {
                    endMatch = -1;
                    arm();
                }
                break;
            }
            default:
                break;
        }
    }

    /** Starts a fresh countdown (a start tag arrived); a budget of zero forces at once. */
    private void arm() {
        state = State.COUNTING;
        remaining = budget;
        if (remaining <= 0) {
            enterForcing();
        }
    }

    /**
     * Marks the output's start: every prompt token has been accepted. A stage whose parser starts the output inside
     * reasoning arms here unless a start tag in the prompt already armed it.
     */
    private synchronized void startOutput() {
        outputStarted = true;
        if (armsAtOutput && (state == State.IDLE || state == State.DONE)) {
            endMatch = -1;
            arm();
        }
    }

    /** Whether the parser starts the output inside reasoning, so the stage arms at the output's start. */
    public boolean armsAtOutput() {
        return armsAtOutput;
    }

    private void enterForcing() {
        state = State.FORCING;
        forcePos = 0;
        ends.reset();
    }

    @Override
    public void apply(Sampler.Candidates candidates) {
        if (!outputStarted) {
            startOutput();
        }
        if (state != State.FORCING) {
            return; // passthrough
        }
        int pos = forcePos;
        if (pos >= forced.length) {
            return;
        }
        int wanted = forced[pos];
        int found = -1;
        for (int i = 0; i < candidates.size(); i++) {
            if (candidates.id(i) == wanted) {
                found = i;
                break;
            }
        }
        if (found < 0) {
            // an earlier stage truncated the forced token away: it takes over the last candidate's slot
            found = candidates.size() - 1;
            candidates.replaceCandidate(found, wanted, 0f);
        }
        for (int i = 0; i < candidates.size(); i++) {
            if (i != found) {
                candidates.setLogit(i, Float.NEGATIVE_INFINITY);
            }
        }
    }

    @Override
    public boolean modifies() {
        return true; // forcing can always begin (a budget, or the mid-stream control)
    }

    @Override
    public synchronized void reset() {
        state = State.IDLE;
        remaining = budget;
        starts.reset();
        ends.reset();
        forcePos = 0;
        endMatch = -1;
        outputStarted = false;
    }

    /** The countdown state, the forced position, both tag matchers' recent tokens and the count of forces. */
    @Override
    public synchronized Object saveState() {
        return new Saved(state, forcePos, outputStarted, remaining, endMatch, forces, starts.recent(),
                ends.recent());
    }

    /**
     * Puts back a saved state; a {@link #force} that succeeded after the state was saved is applied again when the
     * restored state counts, so a control action is never undone.
     */
    @Override
    public synchronized void restoreState(Object saved) {
        if (!(saved instanceof Saved)) {
            throw new IllegalArgumentException("not a reasoning-budget stage state: " + saved);
        }
        Saved s = (Saved) saved;
        state = s.state;
        forcePos = s.forcePos;
        outputStarted = s.outputStarted;
        remaining = s.remaining;
        endMatch = s.endMatch;
        starts.restore(s.startsRecent);
        ends.restore(s.endsRecent);
        if (forces != s.forces && state == State.COUNTING) {
            enterForcing();
        }
    }

    /** A saved stage state ({@link #saveState}). */
    private static final class Saved {
        final State state;
        final int forcePos;
        final boolean outputStarted;
        final int remaining;
        final int endMatch;
        final int forces;
        final int[] startsRecent;
        final int[] endsRecent;

        Saved(State state, int forcePos, boolean outputStarted, int remaining, int endMatch, int forces,
              int[] startsRecent, int[] endsRecent) {
            this.state = state;
            this.forcePos = forcePos;
            this.outputStarted = outputStarted;
            this.remaining = remaining;
            this.endMatch = endMatch;
            this.forces = forces;
            this.startsRecent = startsRecent;
            this.endsRecent = endsRecent;
        }
    }

    /**
     * Whether {@code piece} ends on a UTF-8 character boundary, llama.cpp's {@code common_utf8_is_complete}: the
     * bytes after the last non-continuation byte must complete that character.
     */
    static boolean isCompleteUtf8(byte[] piece) {
        int n = piece.length;
        if (n == 0) {
            return true;
        }
        int limit = Math.min(4, n);
        for (int i = 1; i <= limit; i++) {
            int c = piece[n - i] & 0xFF;
            if ((c & 0xC0) != 0x80) {
                int expected = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
                return i >= expected;
            }
        }
        return false;
    }

    @Override
    public String toString() {
        return "ReasoningBudgetStage{" + state + ", remaining=" + remaining + ", budget="
                + (budget == Integer.MAX_VALUE ? "unlimited" : String.valueOf(budget)) + "}";
    }

    /**
     * The longest sequence of its set that the accepted tokens currently end with (the Aho-Corasick match of
     * llama.cpp's token_matcher, over a small history window — the tag sequences are a few tokens long).
     */
    private static final class TokenMatcher {

        private final int[][] seqs;
        private final int window;
        private final int[] recent;
        private int filled;

        TokenMatcher(List<int[]> sequences) {
            List<int[]> unique = new ArrayList<>(sequences.size());
            for (int[] seq : sequences) {
                boolean duplicate = false;
                for (int[] kept : unique) {
                    duplicate = Arrays.equals(kept, seq);
                    if (duplicate) {
                        break;
                    }
                }
                if (!duplicate) {
                    unique.add(seq.clone());
                }
            }
            this.seqs = unique.toArray(new int[0][]);
            int longest = 0;
            for (int[] seq : this.seqs) {
                longest = Math.max(longest, seq.length);
            }
            this.window = Math.max(0, longest - 1);
            this.recent = new int[window];
        }

        /** The index of the longest sequence ending at {@code token}, or -1; the token joins the history. */
        int advance(int token) {
            int best = -1;
            for (int i = 0; i < seqs.length; i++) {
                int[] seq = seqs[i];
                if (seq[seq.length - 1] != token) {
                    continue;
                }
                int need = seq.length - 1;
                if (filled < need) {
                    continue;
                }
                boolean match = true;
                for (int j = 0; j < need; j++) {
                    if (recent[filled - need + j] != seq[j]) {
                        match = false;
                        break;
                    }
                }
                if (match && (best < 0 || seq.length > seqs[best].length)) {
                    best = i;
                }
            }
            if (window > 0) {
                if (filled < window) {
                    recent[filled++] = token;
                } else {
                    System.arraycopy(recent, 1, recent, 0, window - 1);
                    recent[window - 1] = token;
                }
            }
            return best;
        }

        void reset() {
            filled = 0;
        }

        /** The recent tokens the next match reads, oldest first. */
        int[] recent() {
            return Arrays.copyOf(recent, filled);
        }

        /** Puts back tokens {@link #recent()} returned. */
        void restore(int[] tokens) {
            System.arraycopy(tokens, 0, recent, 0, tokens.length);
            filled = tokens.length;
        }
    }
}
