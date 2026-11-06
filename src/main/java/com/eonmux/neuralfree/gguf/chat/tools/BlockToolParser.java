package com.eonmux.neuralfree.gguf.chat.tools;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * The shared reading of formats whose tool calls sit in blocks inside free text: the text is scanned for the
 * earliest opener, the text before it is content, and the subclass reads the block ({@link #readBlock}). Formats
 * whose call may open the output without an opener (Llama 3's bare JSON, the bare pythonic list) report it through
 * {@link #bareStart}.
 *
 * <p>Content rules, the same for every format: without calls the content is the text verbatim; whitespace between
 * content and a call, between calls and after the last call is dropped, and non-blank text after a call joins the
 * content after a newline. Under {@code partial} the content holds back a trailing proper prefix of an opener and
 * trailing whitespace, so every streamed content text is a prefix of the next.
 *
 * <p>A block the subclass finds malformed stays in the content verbatim (vLLM falls back to content when its regex
 * finds no call). A block cut off by the end of a final text keeps its finished calls; the unfinished rest stays in
 * the content.
 */
public abstract class BlockToolParser implements ToolParser {

    /** {@link #bareStart}: the text may still become a bare call; nothing can be released yet. */
    public static final int UNDECIDED = -2;

    /** How a block read ended. */
    public enum State { COMPLETE, INCOMPLETE, MALFORMED }

    /** One block read: its calls, where it ends and how; {@code rest} is where an unfinished call starts. */
    public static final class Block {
        public final State state;
        public final List<ToolCall> calls;
        public final int end;
        public final int rest;

        private Block(State state, List<ToolCall> calls, int end, int rest) {
            this.state = state;
            this.calls = calls;
            this.end = end;
            this.rest = rest;
        }

        public static Block complete(List<ToolCall> calls, int end) {
            return new Block(State.COMPLETE, calls, end, -1);
        }

        /** Cut off: {@code calls} are what was read (the last may be partial), {@code rest} where it started. */
        public static Block incomplete(List<ToolCall> calls, int rest) {
            return new Block(State.INCOMPLETE, calls, -1, rest);
        }

        public static Block malformed(int end) {
            return new Block(State.MALFORMED, Collections.<ToolCall>emptyList(), end, -1);
        }
    }

    private final List<String> openers;

    protected BlockToolParser(List<String> openers) {
        this.openers = Collections.unmodifiableList(new ArrayList<>(openers));
    }

    @Override
    public List<String> triggers() {
        return openers;
    }

    /**
     * Reads the block whose opener {@code opener} starts at {@code at} ({@code opener} is empty for a bare call).
     * {@link Block#end} of a malformed block is where scanning resumes; a malformed block must end past {@code at}.
     */
    protected abstract Block readBlock(String text, int at, String opener, boolean partial);

    /**
     * Where a call opens the text without an opener: its index, -1 when the text does not open with one, or
     * {@link #UNDECIDED} while the text so far could still become one. The default: no bare calls.
     */
    protected int bareStart(String text, boolean partial) {
        return -1;
    }

    @Override
    public ToolParse parse(String text, boolean partial) {
        Content content = new Content(partial);
        List<ToolCall> calls = new ArrayList<>();
        int pos = 0;
        int bare = bareStart(text, partial);
        if (bare == UNDECIDED) {
            return ToolParse.empty();
        }
        if (bare >= 0) {
            Block b = readBlock(text, bare, "", partial);
            if (b.state != State.MALFORMED) {
                content.beforeCall(text.substring(0, bare));
                if (!take(b, text, bare, calls, content, partial)) {
                    return new ToolParse(content.text(), calls);
                }
                pos = b.end;
            } else if (partial) {
                // a bare call that cannot be one: the whole text is content (bare calls open the output only)
                content.tail(text, Collections.<String>emptyList());
                return new ToolParse(content.text(), calls);
            }
        }
        while (true) {
            int at = -1;
            String opener = null;
            for (String o : openers) {
                int i = text.indexOf(o, pos);
                if (i >= 0 && (at < 0 || i < at)) {
                    at = i;
                    opener = o;
                }
            }
            if (at < 0) {
                content.tail(text.substring(pos), openers);
                break;
            }
            Block b = readBlock(text, at, opener, partial);
            if (b.state == State.MALFORMED) {
                content.raw(text.substring(pos, Math.max(b.end, at + opener.length())));
                pos = Math.max(b.end, at + opener.length());
                continue;
            }
            content.beforeCall(text.substring(pos, at));
            if (!take(b, text, at, calls, content, partial)) {
                break;
            }
            pos = b.end;
        }
        return new ToolParse(content.text(), calls);
    }

    /** Adds the block's calls; false when it was cut off (scanning stops). */
    private static boolean take(Block b, String text, int at, List<ToolCall> calls, Content content,
                                boolean partial) {
        if (b.state == State.COMPLETE) {
            calls.addAll(b.calls);
            content.afterCall();
            return true;
        }
        if (partial) {
            calls.addAll(b.calls);
            return false;
        }
        for (ToolCall c : b.calls) {
            if (c.complete()) {
                calls.add(c);
            }
        }
        content.afterCall();
        content.raw(text.substring(b.rest >= 0 ? b.rest : at));
        return false;
    }

    /** The content text under the rules of the class doc. */
    private static final class Content {
        private final boolean partial;
        private final StringBuilder out = new StringBuilder();
        private boolean afterCall;
        private boolean anyCall;

        Content(boolean partial) {
            this.partial = partial;
        }

        /** Text a call follows: its trailing whitespace drops (and its leading whitespace after a call). */
        void beforeCall(String seg) {
            append(afterCall ? seg.trim() : rtrim(seg));
            anyCall = true;
        }

        void afterCall() {
            afterCall = true;
            anyCall = true;
        }

        /** Text no further call follows (yet): verbatim when final without calls, else held back and trimmed. */
        void tail(String seg, List<String> openers) {
            if (partial) {
                seg = seg.substring(0, safeEnd(seg, openers));
                seg = rtrim(seg);
            } else if (anyCall) {
                seg = rtrim(seg);
            }
            append(afterCall ? ltrim(seg) : seg);
        }

        /** Verbatim text: a malformed block, the unfinished rest of a cut-off block. */
        void raw(String seg) {
            if (afterCall && out.length() > 0 && !seg.isEmpty()) {
                out.append('\n');
            }
            out.append(afterCall ? ltrim(seg) : seg);
            afterCall = false;
        }

        private void append(String seg) {
            if (seg.isEmpty()) {
                return;
            }
            if (afterCall && out.length() > 0) {
                out.append('\n');
            }
            out.append(seg);
            afterCall = false;
        }

        String text() {
            return out.toString();
        }
    }

    /** The end of the part of {@code seg} no opener can still start in: a trailing proper opener prefix is held. */
    static int safeEnd(String seg, List<String> openers) {
        int end = seg.length();
        for (String o : openers) {
            for (int k = Math.min(o.length() - 1, seg.length()); k > 0; k--) {
                if (seg.regionMatches(seg.length() - k, o, 0, k)) {
                    end = Math.min(end, seg.length() - k);
                    break;
                }
            }
        }
        return end;
    }

    static String rtrim(String s) {
        int e = s.length();
        while (e > 0 && Character.isWhitespace(s.charAt(e - 1))) {
            e--;
        }
        return s.substring(0, e);
    }

    static String ltrim(String s) {
        int b = 0;
        while (b < s.length() && Character.isWhitespace(s.charAt(b))) {
            b++;
        }
        return s.substring(b);
    }

    /** Whether {@code text} from {@code i} is a proper prefix of {@code literal} (the literal may still arrive). */
    static boolean prefixOf(String text, int i, String literal) {
        int n = text.length() - i;
        return n < literal.length() && literal.regionMatches(0, text, i, n);
    }
}
