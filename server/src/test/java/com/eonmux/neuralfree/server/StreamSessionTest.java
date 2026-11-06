package com.eonmux.neuralfree.server;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * One resumable stream session: the producer's bytes replay from any offset the buffer still holds, readers
 * block for live bytes until the session finishes, the cap drops the front, and the mirror records past a
 * client that went away. The registry replaces, evicts, sweeps and closes.
 */
@Timeout(30)
class StreamSessionTest {

    private static byte[] bytes(String text) {
        return text.getBytes(StandardCharsets.UTF_8);
    }

    /** The full text of a session drained from {@code from}: the session must be finished, or the read keeps
     * waiting for live bytes. */
    private static String drained(StreamSession session, long from) throws Exception {
        StringBuilder out = new StringBuilder();
        session.readFrom(from, chunk -> {
            out.append(new String(chunk, StandardCharsets.UTF_8));
            return true;
        });
        return out.toString();
    }

    @Test
    void recordedBytesReplayFromAnyOffsetTheBufferHolds() throws Exception {
        StreamSession session = new StreamSession("c", 1024);
        assertTrue(session.append(bytes("data: {\"a\": 1}\n\n")));
        assertEquals(StreamSession.ReadStatus.OK, session.readFrom(0, chunk -> false));
        assertEquals(16, session.totalBytes());
        assertEquals(0, session.droppedPrefix());
        assertFalse(session.isDone());
        assertEquals(0, session.completedAt());
        session.finish();
        assertEquals("data: {\"a\": 1}\n\n", drained(session, 0));
        assertEquals("1}\n\n", drained(session, 12));
    }

    @Test
    void theCapDropsTheFrontAndAnOffsetBelowItIsLost() throws Exception {
        StreamSession session = new StreamSession("c", 8);
        assertTrue(session.append(bytes("12345678")));
        assertTrue(session.append(bytes("90")));
        assertEquals(2, session.droppedPrefix(), "the first two bytes left the buffer");
        assertEquals(10, session.totalBytes(), "the total counts the dropped bytes too");
        session.finish();
        assertEquals("7890", drained(session, 6));
        assertEquals("90", drained(session, 8));
        assertEquals("34567890", drained(session, 2));
        assertEquals(StreamSession.ReadStatus.OFFSET_LOST, session.readFrom(1, chunk -> false));
    }

    @Test
    void oneChunkLargerThanTheCapKeepsOnlyItsTail() throws Exception {
        StreamSession session = new StreamSession("c", 8);
        assertTrue(session.append(bytes("abcdefghij")));
        assertEquals(2, session.droppedPrefix());
        assertEquals(10, session.totalBytes());
        session.finish();
        assertEquals("cdefghij", drained(session, 2));
    }

    @Test
    void finishingWakesTheReadersAndEndsTheSession() throws Exception {
        StreamSession session = new StreamSession("c", 1024);
        assertTrue(session.append(bytes("data: x\n\n")));
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            Callable<String> reader = () -> drained(session, 9);
            Future<String> waiting = pool.submit(reader);
            Thread.sleep(100);
            assertFalse(waiting.isDone(), "the reader waits past the recorded bytes");
            session.finish();
            assertEquals("", waiting.get(5, TimeUnit.SECONDS), "the finished session ends the read");
            assertTrue(session.isDone());
            assertNotEquals(0, session.completedAt());
            assertFalse(session.append(bytes("data: y\n\n")), "a finished session records nothing");
            assertEquals("data: x\n\n", drained(session, 0), "the recorded bytes stay replayable");
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    void aReaderDrainsLiveBytesAsTheyArrive() throws Exception {
        StreamSession session = new StreamSession("c", 1024);
        assertTrue(session.append(bytes("data: 1\n\n")));
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            Future<String> reading = pool.submit(() -> drained(session, 0));
            Thread.sleep(100);
            assertTrue(session.append(bytes("data: 2\n\n")));
            session.finish();
            assertEquals("data: 1\n\ndata: 2\n\n", reading.get(5, TimeUnit.SECONDS));
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    void cancellingMarksTheSessionWithoutEndingIt() throws Exception {
        StreamSession session = new StreamSession("c", 1024);
        assertFalse(session.isCancelled());
        session.cancel();
        assertTrue(session.isCancelled());
        assertFalse(session.isDone(), "the cancel alone does not end the session");
        assertTrue(session.append(bytes("data: x\n\n")));
    }

    @Test
    void aSessionTakesAnIdAndAPositiveCap() {
        assertThrows(IllegalArgumentException.class, () -> new StreamSession("", 16));
        assertThrows(IllegalArgumentException.class, () -> new StreamSession(null, 16));
        assertThrows(IllegalArgumentException.class, () -> new StreamSession("c", 0));
    }

    @Test
    void theMirrorWritesTheClientAndTheSession() throws Exception {
        StreamSession session = new StreamSession("c", 1024);
        ByteArrayOutputStream client = new ByteArrayOutputStream();
        OutputStream mirror = session.mirror(client);
        mirror.write(bytes("data: {\"x\": 1}\n\n"));
        mirror.flush();
        assertEquals("data: {\"x\": 1}\n\n", client.toString(StandardCharsets.UTF_8));
        session.finish();
        assertEquals("data: {\"x\": 1}\n\n", drained(session, 0));
    }

    @Test
    void theMirrorRecordsPastAClientThatWentAway() throws Exception {
        StreamSession session = new StreamSession("c", 1024);
        OutputStream mirror = session.mirror(failingClient());
        mirror.write(bytes("data: 1\n\n"));
        mirror.write(bytes("data: 2\n\n"));
        mirror.flush();
        session.finish();
        assertEquals("data: 1\n\ndata: 2\n\n", drained(session, 0), "the session keeps recording");
    }

    @Test
    void theMirrorKeepsStreamingToTheClientOnceTheSessionEnded() throws Exception {
        StreamSession session = new StreamSession("c", 1024);
        ByteArrayOutputStream client = new ByteArrayOutputStream();
        OutputStream mirror = session.mirror(client);
        mirror.write(bytes("data: 1\n\n"));
        session.finish();
        mirror.write(bytes("data: 2\n\n"));
        mirror.flush();
        assertEquals("data: 1\n\ndata: 2\n\n", client.toString(StandardCharsets.UTF_8));
        assertEquals("data: 1\n\n", drained(session, 0), "the ended session records nothing more");
    }

    @Test
    void theMirrorForwardsTheCloseToTheClient() throws Exception {
        StreamSession session = new StreamSession("c", 1024);
        ByteArrayOutputStream client = new ByteArrayOutputStream();
        session.mirror(client).close();
        assertEquals(0, client.size());
    }

    @Test
    void replacingASessionCancelsAndEndsThePreviousOne() throws Exception {
        StreamSessions sessions = new StreamSessions(1024, 300, 60);
        try {
            StreamSession first = sessions.createOrReplace("conv");
            StreamSession second = sessions.createOrReplace("conv");
            assertSame(second, sessions.get("conv"));
            assertTrue(first.isCancelled(), "the replaced session is cancelled");
            assertTrue(first.isDone(), "the replaced session ends, waking its readers");
            assertFalse(second.isCancelled());
            assertTrue(second.append(bytes("data: x\n\n")));
        } finally {
            sessions.close();
        }
    }

    @Test
    void evictionAndTheExplicitStopForgetTheSession() {
        StreamSessions sessions = new StreamSessions(1024, 300, 60);
        try {
            sessions.createOrReplace("a").finish();
            sessions.evict("a");
            assertNull(sessions.get("a"));

            StreamSession stopped = sessions.createOrReplace("b");
            sessions.evictAndCancel("b");
            assertNull(sessions.get("b"));
            assertTrue(stopped.isCancelled());
            assertTrue(stopped.isDone(), "the stop wakes the readers");
        } finally {
            sessions.close();
        }
    }

    @Test
    void theSweeperEvictsFinishedSessionsAfterTheTtl() throws Exception {
        StreamSessions sessions = new StreamSessions(1024, 1, 1);
        try {
            StreamSession finished = sessions.createOrReplace("done");
            finished.finish();
            StreamSession live = sessions.createOrReplace("live");
            long deadline = System.currentTimeMillis() + 8000;
            while (System.currentTimeMillis() < deadline && sessions.get("done") != null) {
                Thread.sleep(50);
            }
            assertNull(sessions.get("done"), "the sweeper evicted the finished session");
            assertSame(live, sessions.get("live"), "a live session is never swept");
            assertTrue(finished.isDone());
        } finally {
            sessions.close();
        }
    }

    @Test
    void closingEndsEverySessionAndStopsTheSweeper() throws Exception {
        StreamSessions sessions = new StreamSessions(1024, 300, 60);
        StreamSession session = sessions.createOrReplace("c");
        assertTrue(session.append(bytes("data: x\n\n")));
        sessions.close();
        assertTrue(session.isDone());
        assertNull(sessions.get("c"));
        List<StreamSession> all = sessions.all();
        assertTrue(all.isEmpty(), "the closed registry holds nothing");
    }

    @Test
    void theRegistryTakesAPositiveCapTtlAndSweepInterval() {
        assertThrows(IllegalArgumentException.class, () -> new StreamSessions(0, 300, 60));
        assertThrows(IllegalArgumentException.class, () -> new StreamSessions(1024, 0, 60));
        assertThrows(IllegalArgumentException.class, () -> new StreamSessions(1024, 300, 0));
    }

    /** A client stream whose every write and flush fails, as a disconnected client's does. */
    private static OutputStream failingClient() {
        return new OutputStream() {
            @Override
            public void write(int b) throws IOException {
                throw new IOException("the client went away");
            }

            @Override
            public void write(byte[] b, int off, int len) throws IOException {
                throw new IOException("the client went away");
            }

            @Override
            public void flush() throws IOException {
                throw new IOException("the client went away");
            }
        };
    }
}
