package com.eonmux.neuralfree.server;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * The live resumable stream sessions keyed by conversation id (llama.cpp's {@code stream_session_manager}),
 * one conversation = at most one live session: a new session for a known id replaces (cancels and ends) the
 * previous one. A sweeper thread evicts sessions finished longer than the TTL, so a replayed stream does not
 * keep its bytes forever.
 */
final class StreamSessions implements AutoCloseable {

    /** How long a finished session keeps its bytes (llama.cpp's 300 s). */
    static final int DEFAULT_TTL_SECONDS = 300;
    /** The per-session byte cap (llama.cpp's 4 MiB). */
    static final int DEFAULT_MAX_BYTES = 4 * 1024 * 1024;
    /** How often the sweeper runs (llama.cpp's 60 s). */
    static final int DEFAULT_SWEEP_SECONDS = 60;

    private final ConcurrentMap<String, StreamSession> sessions = new ConcurrentHashMap<>();
    private final int maxBytes;
    private final long ttlMillis;
    private final long sweepMillis;
    private volatile boolean open = true;
    private final Thread sweeper;

    /** The process registry over the reference defaults, swept by a daemon thread. */
    StreamSessions() {
        this(DEFAULT_MAX_BYTES, DEFAULT_TTL_SECONDS, DEFAULT_SWEEP_SECONDS);
    }

    /** A registry with an explicit byte cap, finished-session TTL and sweep interval (seconds). */
    StreamSessions(int maxBytes, int ttlSeconds, int sweepSeconds) {
        if (maxBytes <= 0) {
            throw new IllegalArgumentException("the stream session byte cap must be positive, got " + maxBytes);
        }
        if (ttlSeconds <= 0 || sweepSeconds <= 0) {
            throw new IllegalArgumentException("the stream session TTL and sweep interval must be positive,"
                    + " got " + ttlSeconds + "s and " + sweepSeconds + "s");
        }
        this.maxBytes = maxBytes;
        this.ttlMillis = ttlSeconds * 1000L;
        this.sweepMillis = sweepSeconds * 1000L;
        this.sweeper = new Thread(this::sweepLoop, "neuralfree-stream-sweeper");
        this.sweeper.setDaemon(true);
        this.sweeper.start();
    }

    /**
     * Installs a fresh session for {@code conversationId}, cancelling and ending the previous one, so its
     * producer stops and its readers see it done.
     */
    StreamSession createOrReplace(String conversationId) {
        requireId(conversationId);
        StreamSession fresh = new StreamSession(conversationId, maxBytes);
        StreamSession previous = sessions.put(conversationId, fresh);
        if (previous != null) {
            previous.cancel();
            previous.finish();
        }
        return fresh;
    }

    /** The live session of {@code conversationId}, or null. */
    StreamSession get(String conversationId) {
        requireId(conversationId);
        return sessions.get(conversationId);
    }

    /** A snapshot of every session, live and finished-but-unexpired. */
    List<StreamSession> all() {
        return new ArrayList<>(sessions.values());
    }

    /** Forgets {@code conversationId}'s session, leaving any producer to finish it in its own time. */
    void evict(String conversationId) {
        requireId(conversationId);
        StreamSession removed = sessions.remove(conversationId);
        if (removed != null) {
            removed.finish();
        }
    }

    /**
     * The explicit stop: cancels {@code conversationId}'s session — its producer stops pulling output and
     * cancels the generation at its next check — removes it and wakes its readers.
     */
    void evictAndCancel(String conversationId) {
        requireId(conversationId);
        StreamSession removed = sessions.remove(conversationId);
        if (removed != null) {
            removed.cancel();
            removed.finish();
        }
    }

    /** Stops the sweeper and ends every session, so no reader ever waits on a dead registry. */
    @Override
    public void close() {
        open = false;
        sweeper.interrupt();
        for (Map.Entry<String, StreamSession> entry : sessions.entrySet()) {
            entry.getValue().finish();
        }
        sessions.clear();
    }

    /** Evicts the sessions finished longer than the TTL, then sleeps until the next sweep or the close. */
    private void sweepLoop() {
        while (open) {
            try {
                Thread.sleep(sweepMillis);
            } catch (InterruptedException e) {
                return;
            }
            sweep();
        }
    }

    private void sweep() {
        long cutoff = System.currentTimeMillis() - ttlMillis;
        for (Map.Entry<String, StreamSession> entry : sessions.entrySet()) {
            StreamSession session = entry.getValue();
            long completedAt = session.completedAt();
            if (completedAt != 0 && completedAt * 1000L <= cutoff && sessions.remove(entry.getKey(), session)) {
                session.finish();
            }
        }
    }

    private static void requireId(String conversationId) {
        if (conversationId == null || conversationId.isEmpty()) {
            throw new IllegalArgumentException("a stream session takes a non-empty conversation id");
        }
    }
}
