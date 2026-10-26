package com.eonmux.neuralfree.utils.json;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.math.BigInteger;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * One parse of a JSON document for {@link Json}: recursive descent over an explicit container stack, so nesting
 * depth never consumes the Java stack. Every {@link Json.Limits} check precedes the allocation it bounds.
 */
final class JsonParser {

    private static final Logger LOGGER = LogManager.getLogger(Json.class);
    private static final char[] HEX = "0123456789abcdef".toCharArray();

    /** Marker returned by {@link #begin()} when it opened a non-empty container. */
    private static final Object OPENED = new Object();

    private final String s;
    private final Json.Limits limits;
    private final String source;
    private final ArrayDeque<Frame> stack = new ArrayDeque<>();
    private int pos;
    private long values;

    JsonParser(String s, int start, Json.Limits limits, String source) {
        this.s = s;
        this.pos = start;
        this.limits = limits;
        this.source = source;
    }

    /** One open container: a map with the key whose value is being parsed, or a list. */
    private static final class Frame {
        final Map<String, Object> map;
        final List<Object> list;
        /** Members counted so far, against {@code limits.maxMembers()}. */
        int members;
        String key;

        Frame(Map<String, Object> map, List<Object> list) {
            this.map = map;
            this.list = list;
        }
    }

    private JsonException error(String msg) {
        return new JsonException(msg, pos);
    }

    /** Parses the whole text from the start offset: one value, optionally surrounded by whitespace. */
    Object document() {
        ws();
        Object v = value();
        ws();
        if (pos != s.length()) throw error("extra data");
        return v;
    }

    private void ws() {
        while (pos < s.length()) {
            char c = s.charAt(pos);
            if (c != ' ' && c != '\t' && c != '\r' && c != '\n') return;
            pos++;
        }
    }

    private boolean eat(char c) {
        if (pos < s.length() && s.charAt(pos) == c) {
            pos++;
            return true;
        }
        return false;
    }

    private boolean eat(String lit) {
        if (s.startsWith(lit, pos)) {
            pos += lit.length();
            return true;
        }
        return false;
    }

    /** Parses a value, descending into containers and climbing out as each one closes. */
    private Object value() {
        descend:
        while (true) {
            Object v = begin();
            if (v == OPENED) continue;
            while (true) {
                Frame top = stack.peek();
                if (top == null) return v;
                if (top.map != null) top.map.put(top.key, v);
                else top.list.add(v);
                ws();
                if (top.map != null) {
                    if (eat('}')) {
                        v = stack.pop().map;
                        continue;
                    }
                    if (!eat(',')) throw error("expected ',' or '}'");
                    ws();
                    member(top);
                } else {
                    if (eat(']')) {
                        v = stack.pop().list;
                        continue;
                    }
                    if (!eat(',')) throw error("expected ',' or ']'");
                    ws();
                }
                continue descend;
            }
        }
    }

    /** Parses a scalar, an empty container, or opens a container and returns {@link #OPENED}. */
    private Object begin() {
        if (pos >= s.length()) throw error("unexpected end of input");
        Frame top = stack.peek();
        if (top != null && top.list != null && ++top.members > limits.maxMembers()) {
            throw error("array member exceeds maxMembers " + limits.maxMembers());
        }
        char c = s.charAt(pos);
        count();
        if (c == '{') {
            Frame f = open(new Frame(new LinkedHashMap<>(), null));
            ws();
            if (eat('}')) return stack.pop().map;
            member(f);
            return OPENED;
        }
        if (c == '[') {
            open(new Frame(null, new ArrayList<>()));
            ws();
            if (eat(']')) return stack.pop().list;
            return OPENED;
        }
        if (c == '"') return string();
        if (eat("true")) return Boolean.TRUE;
        if (eat("false")) return Boolean.FALSE;
        if (eat("null")) return null;
        if (eat("NaN")) {
            nonFinite("NaN");
            return Double.NaN;
        }
        if (eat("Infinity")) {
            nonFinite("Infinity");
            return Double.POSITIVE_INFINITY;
        }
        if (eat("-Infinity")) {
            nonFinite("-Infinity");
            return Double.NEGATIVE_INFINITY;
        }
        if (c == '-' || (c >= '0' && c <= '9')) return number();
        throw error("unexpected character '" + c + "'");
    }

    /** Rejects the {@code NaN} or {@code Infinity} literal just eaten, at its start, when {@code limits} forbids it. */
    private void nonFinite(String literal) {
        if (limits.allowNonFinite()) return;
        throw new JsonException("non-finite literal '" + literal + "' is not allowed", pos - literal.length());
    }

    private void count() {
        if (++values > limits.maxValues()) throw error("document exceeds maxValues " + limits.maxValues());
    }

    private Frame open(Frame f) {
        if (stack.size() >= limits.maxDepth()) throw error("nesting depth exceeds maxDepth " + limits.maxDepth());
        pos++;
        stack.push(f);
        return f;
    }

    /** Reads {@code "key" :} into {@code f}, applying the duplicate-key policy. */
    private void member(Frame f) {
        if (++f.members > limits.maxMembers()) {
            throw error("object member exceeds maxMembers " + limits.maxMembers());
        }
        if (pos >= s.length() || s.charAt(pos) != '"') throw error("expected property name");
        int keyPos = pos;
        String k = string();
        Json.DuplicateKeys policy = limits.duplicateKeys();
        if (policy != Json.DuplicateKeys.LAST_WINS && f.map.containsKey(k)) {
            if (policy == Json.DuplicateKeys.REJECT) {
                throw new JsonException("duplicate key '" + k + "' at " + pathTo(k), keyPos);
            }
            if (LOGGER.isDebugEnabled()) {
                LOGGER.debug("{}duplicate JSON key '{}' at {} (position {}); the last value wins",
                        source == null ? "" : source + ": ", k, pathTo(k), keyPos);
            }
        }
        ws();
        if (!eat(':')) throw error("expected ':'");
        ws();
        f.key = k;
    }

    /** {@code a.b[2].key}: the path from the root to member {@code key} of the innermost open object. */
    private String pathTo(String key) {
        StringBuilder sb = new StringBuilder();
        Iterator<Frame> it = stack.descendingIterator();
        while (it.hasNext()) {
            Frame f = it.next();
            if (f == stack.peek()) break;
            if (f.map != null) sb.append(sb.length() == 0 ? "" : ".").append(f.key);
            else sb.append('[').append(f.list.size()).append(']');
        }
        return sb.append(sb.length() == 0 ? "" : ".").append(key).toString();
    }

    private String string() {
        pos++;
        StringBuilder sb = null;
        int run = pos;
        int max = limits.maxStringChars();
        while (true) {
            if (pos >= s.length()) throw error("unterminated string");
            char c = s.charAt(pos);
            if (c == '"') {
                int len = (sb == null ? 0 : sb.length()) + (pos - run);
                if (len > max) throw error("string exceeds maxStringChars " + max);
                String out = sb == null ? s.substring(run, pos) : sb.append(s, run, pos).toString();
                pos++;
                return out;
            }
            if (c == '\\') {
                if (sb == null) sb = new StringBuilder();
                if (sb.length() + (pos - run) > max) throw error("string exceeds maxStringChars " + max);
                sb.append(s, run, pos);
                pos++;
                escape(sb);
                if (sb.length() > max) throw error("string exceeds maxStringChars " + max);
                run = pos;
                continue;
            }
            if (c < 0x20) throw error("invalid control character in string");
            if (Character.isSurrogate(c)) {
                if (Character.isHighSurrogate(c) && pos + 1 < s.length() && Character.isLowSurrogate(s.charAt(pos + 1))) {
                    pos += 2;
                    continue;
                }
                throw error("lone surrogate in string");
            }
            pos++;
        }
    }

    /** Decodes the escape after a backslash; a {@code \\u} high surrogate must be followed by a {@code \\u} low one. */
    private void escape(StringBuilder sb) {
        if (pos >= s.length()) throw error("unterminated escape");
        char e = s.charAt(pos++);
        switch (e) {
            case '"': sb.append('"'); break;
            case '\\': sb.append('\\'); break;
            case '/': sb.append('/'); break;
            case 'b': sb.append('\b'); break;
            case 'f': sb.append('\f'); break;
            case 'n': sb.append('\n'); break;
            case 'r': sb.append('\r'); break;
            case 't': sb.append('\t'); break;
            case 'u': {
                int at = pos - 2;
                char u = hex4();
                if (Character.isLowSurrogate(u)) throw new JsonException("lone surrogate '\\u" + hex(u) + "'", at);
                if (Character.isHighSurrogate(u)) {
                    if (!s.startsWith("\\u", pos)) throw new JsonException("lone surrogate '\\u" + hex(u) + "'", at);
                    pos += 2;
                    char low = hex4();
                    if (!Character.isLowSurrogate(low)) throw new JsonException("lone surrogate '\\u" + hex(u) + "'", at);
                    sb.append(u).append(low);
                } else {
                    sb.append(u);
                }
                break;
            }
            default: throw error("invalid escape '\\" + e + "'");
        }
    }

    private char hex4() {
        if (pos + 4 > s.length()) throw error("truncated \\u escape");
        int v = 0;
        for (int i = 0; i < 4; i++) {
            int d = hexDigit(s.charAt(pos + i));
            if (d < 0) throw error("invalid \\u escape");
            v = (v << 4) | d;
        }
        pos += 4;
        return (char) v;
    }

    private static int hexDigit(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }

    private static String hex(char c) {
        return new String(new char[]{HEX[(c >> 12) & 0xF], HEX[(c >> 8) & 0xF], HEX[(c >> 4) & 0xF], HEX[c & 0xF]});
    }

    /** {@code -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?}, ASCII digits only. */
    private Object number() {
        int start = pos;
        eat('-');
        int intStart = pos;
        if (!digit()) throw invalidNumber(start);
        if (s.charAt(pos) == '0') pos++;
        else skipDigits();
        int intDigits = pos - intStart;
        boolean integral = true;
        if (eat('.')) {
            integral = false;
            if (!digit()) throw invalidNumber(start);
            skipDigits();
        }
        if (pos < s.length() && (s.charAt(pos) == 'e' || s.charAt(pos) == 'E')) {
            integral = false;
            pos++;
            if (pos < s.length() && (s.charAt(pos) == '+' || s.charAt(pos) == '-')) pos++;
            if (!digit()) throw invalidNumber(start);
            skipDigits();
        }
        String t = s.substring(start, pos);
        if (!integral) return Double.parseDouble(t);
        if (intDigits <= 18) return Long.parseLong(t);
        if (intDigits > Json.MAX_INTEGER_DIGITS) {
            pos = start;
            throw error("integer literal exceeds " + Json.MAX_INTEGER_DIGITS + " digits");
        }
        BigInteger big = new BigInteger(t);
        return big.bitLength() < Long.SIZE ? (Object) big.longValue() : big;
    }

    private boolean digit() {
        return pos < s.length() && s.charAt(pos) >= '0' && s.charAt(pos) <= '9';
    }

    private void skipDigits() {
        while (digit()) pos++;
    }

    private JsonException invalidNumber(int start) {
        String t = s.substring(start, Math.min(pos + 1, s.length()));
        pos = start;
        return error("invalid number '" + t + "'");
    }
}
