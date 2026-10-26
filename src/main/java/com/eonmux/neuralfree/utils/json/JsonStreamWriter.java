package com.eonmux.neuralfree.utils.json;

import java.io.IOException;
import java.math.BigInteger;
import java.util.ArrayDeque;
import java.util.Objects;

/**
 * Writes one JSON value incrementally to an {@link Appendable} under one {@link Json.Style}: the streaming counterpart
 * of {@link Json#write}, for SSE chunks and large arrays whose host values are built member by member. Escaping and
 * float formatting are {@link Json#write}'s shared internals, so a finished document is byte-identical to the one
 * {@link Json#write} produces for the same value.
 *
 * <p>The nesting discipline is the document structure itself: at most one name may be pending per open object, every
 * opened object and array must be closed, and a value must be written between an object member's name and the next
 * name. Misuse — a value where a name is due, a close of the wrong container, a close with nothing open — throws
 * {@link IllegalStateException}. Consecutive top-level values are written without any separator; the caller frames
 * them, as an SSE {@code data:} line does.</p>
 *
 * <p>Not thread-safe: owned by its writing thread (§3.3).</p>
 */
public final class JsonStreamWriter {

    private final Appendable out;
    private final Json.Style st;
    private final ArrayDeque<Frame> frames = new ArrayDeque<>();
    /** An object member was named; its value is due before the next name, value or close. */
    private boolean awaitValue;

    /** One open object or array, with whether a member was written already. */
    private static final class Frame {
        final boolean object;
        boolean hasMember;

        Frame(boolean object) {
            this.object = object;
        }
    }

    /** @param out receives the document; {@code style} carries the formatting {@link Json#write} would apply */
    public JsonStreamWriter(Appendable out, Json.Style style) {
        this.out = Objects.requireNonNull(out, "out");
        this.st = Objects.requireNonNull(style, "style");
    }

    /** Opens an object; its members are written as {@link #name(String)} / value pairs. */
    public void beginObject() throws IOException {
        begin(true);
    }

    /** Opens an array. */
    public void beginArray() throws IOException {
        begin(false);
    }

    private void begin(boolean object) throws IOException {
        if (awaitValue) throw new IllegalStateException("object member without a value");
        out.append(object ? '{' : '[');
        frames.push(new Frame(object));
    }

    /** Writes a member name of the open object, and the {@link Json.Style#keySep()} after it. */
    public void name(String name) throws IOException {
        Objects.requireNonNull(name, "name");
        Frame f = frames.peek();
        if (f == null || !f.object) throw new IllegalStateException("name outside an object");
        if (awaitValue) throw new IllegalStateException("object member without a value");
        if (f.hasMember) out.append(st.itemSep());
        Json.newline(out, st.indent(), frames.size());
        f.hasMember = true;
        Json.writeString(name, st, out);
        out.append(st.keySep());
        awaitValue = true;
    }

    public void value(String value) throws IOException {
        beforeValue();
        Json.writeString(Objects.requireNonNull(value, "value"), st, out);
    }

    public void value(long value) throws IOException {
        beforeValue();
        out.append(Long.toString(value));
    }

    /** {@code value} as {@link Json#numberText(double)}; a non-finite double throws when {@code Style.rejectNonFinite}. */
    public void value(double value) throws IOException {
        beforeValue();
        Json.writeNumber(value, st, out);
    }

    public void value(boolean value) throws IOException {
        beforeValue();
        out.append(value ? "true" : "false");
    }

    public void value(BigInteger value) throws IOException {
        beforeValue();
        out.append(Objects.requireNonNull(value, "value").toString());
    }

    /** Writes the JSON {@code null} literal. */
    public void nullValue() throws IOException {
        beforeValue();
        out.append("null");
    }

    /**
     * Writes {@code value} by delegating to {@link Json#write} when it is a {@link Map}, {@link List},
     * {@link String}, {@link Number}, {@link Boolean} or null, so the value model of a parsed document needs no
     * manual walk for large members. A delegated container indents from the document root; with an indented style,
     * walk such values member by member with {@link #beginArray} and friends instead.
     */
    public void value(Object value) throws IOException {
        beforeValue();
        Json.write(value, st, out);
    }

    /**
     * Emits exactly the framing {@link Json#write} writes before a member value or array element: the pending
     * object name, or the item separator and indent of the open array; a top-level value takes none.
     */
    private void beforeValue() throws IOException {
        Frame f = frames.peek();
        if (f == null) return;
        if (f.object) {
            if (!awaitValue) throw new IllegalStateException("object member without a name");
            awaitValue = false;
            return;
        }
        if (f.hasMember) out.append(st.itemSep());
        Json.newline(out, st.indent(), frames.size());
        f.hasMember = true;
    }

    public void endObject() throws IOException {
        end(true);
    }

    public void endArray() throws IOException {
        end(false);
    }

    private void end(boolean object) throws IOException {
        if (awaitValue) throw new IllegalStateException("object member without a value");
        Frame f = frames.peek();
        if (f == null) throw new IllegalStateException("no container is open");
        if (f.object != object) {
            throw new IllegalStateException(object ? "endObject with an array open" : "endArray with an object open");
        }
        if (f.hasMember) Json.newline(out, st.indent(), frames.size() - 1);
        out.append(object ? '}' : ']');
        frames.pop();
    }
}
