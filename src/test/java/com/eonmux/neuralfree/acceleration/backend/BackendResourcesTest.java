package com.eonmux.neuralfree.acceleration.backend;

import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/** Per-backend lock and attachments: creation once, reverse-order close, failure reporting and isolation. */
public class BackendResourcesTest {

    private static final Object KEY_A = new Object();
    private static final Object KEY_B = new Object();
    private static final Object KEY_C = new Object();

    private ComputeBackend backend;
    private ComputeBackend other;
    private final List<String> closed = new ArrayList<>();

    @Before
    public void setUp() {
        backend = TrainParity.hostBackend();
        other = TrainParity.hostBackend();
    }

    @After
    public void tearDown() {
        BackendResources.closeAll(backend);
        BackendResources.closeAll(other);
    }

    private final class Recorder implements AutoCloseable {
        private final String name;
        private final RuntimeException failure;

        Recorder(String name) {
            this(name, null);
        }

        Recorder(String name, RuntimeException failure) {
            this.name = name;
            this.failure = failure;
        }

        @Override
        public void close() {
            closed.add(name);
            if (failure != null) {
                throw failure;
            }
        }
    }

    @Test
    public void lockIsStablePerBackendAndDistinctAcrossBackends() {
        ReentrantLock a = BackendResources.lock(backend);
        assertSame(a, BackendResources.lock(backend));
        assertNotSame(a, BackendResources.lock(other));
    }

    @Test
    public void attachmentIsCreatedOnce() {
        int[] calls = new int[1];
        Recorder first = BackendResources.attachment(backend, KEY_A, () -> {
            calls[0]++;
            return new Recorder("a");
        });
        Recorder second = BackendResources.attachment(backend, KEY_A, () -> {
            calls[0]++;
            return new Recorder("a2");
        });
        assertSame(first, second);
        assertEquals(1, calls[0]);
        assertSame(first, BackendResources.existing(backend, KEY_A));
        assertNull(BackendResources.existing(other, KEY_A));
        assertNull(BackendResources.existing(backend, KEY_B));
    }

    @Test
    public void attachmentFactoryRunsUnderTheBackendLock() {
        BackendResources.attachment(backend, KEY_A, () -> {
            assertTrue(BackendResources.lock(backend).isHeldByCurrentThread());
            return new Recorder("a");
        });
    }

    @Test(expected = IllegalStateException.class)
    public void nullFromFactoryIsRejected() {
        BackendResources.attachment(backend, KEY_A, () -> null);
    }

    @Test
    public void failedFactoryStoresNothing() {
        try {
            BackendResources.attachment(backend, KEY_A, () -> {
                throw new IllegalArgumentException("boom");
            });
            fail("factory failure must propagate");
        } catch (IllegalArgumentException expected) {
            assertEquals("boom", expected.getMessage());
        }
        assertNull(BackendResources.existing(backend, KEY_A));
    }

    @Test
    public void closeAllClosesInReverseCreationOrderAndResetsTheLock() {
        ReentrantLock before = BackendResources.lock(backend);
        BackendResources.attachment(backend, KEY_A, () -> new Recorder("a"));
        BackendResources.attachment(backend, KEY_B, () -> new Recorder("b"));
        BackendResources.attachment(backend, KEY_C, () -> new Recorder("c"));

        BackendResources.closeAll(backend);

        assertEquals(List.of("c", "b", "a"), closed);
        assertNull(BackendResources.existing(backend, KEY_A));
        assertNotSame(before, BackendResources.lock(backend));
        BackendResources.closeAll(backend);
        assertEquals(3, closed.size());
    }

    @Test
    public void closeAllClosesEveryAttachmentAndRethrowsTheFirstFailure() {
        IllegalStateException first = new IllegalStateException("first");
        IllegalStateException second = new IllegalStateException("second");
        BackendResources.attachment(backend, KEY_A, () -> new Recorder("a", second));
        BackendResources.attachment(backend, KEY_B, () -> new Recorder("b"));
        BackendResources.attachment(backend, KEY_C, () -> new Recorder("c", first));
        try {
            BackendResources.closeAll(backend);
            fail("closeAll must rethrow the close failure");
        } catch (IllegalStateException e) {
            assertSame(first, e);
            assertEquals(1, e.getSuppressed().length);
            assertSame(second, e.getSuppressed()[0]);
        }
        assertEquals(List.of("c", "b", "a"), closed);
        assertNull(BackendResources.existing(backend, KEY_B));
    }

    @Test
    public void closeAllLeavesOtherBackendsAlone() {
        BackendResources.attachment(backend, KEY_A, () -> new Recorder("a"));
        Recorder kept = BackendResources.attachment(other, KEY_A, () -> new Recorder("other"));

        BackendResources.closeAll(backend);

        assertEquals(List.of("a"), closed);
        assertSame(kept, BackendResources.existing(other, KEY_A));
    }

    @Test
    public void detachClosesOneAttachment() {
        BackendResources.attachment(backend, KEY_A, () -> new Recorder("a"));
        BackendResources.attachment(backend, KEY_B, () -> new Recorder("b"));

        BackendResources.detach(backend, KEY_A);
        BackendResources.detach(backend, KEY_A);

        assertEquals(List.of("a"), closed);
        assertNull(BackendResources.existing(backend, KEY_A));
        assertTrue(BackendResources.existing(backend, KEY_B) instanceof Recorder);
    }

    @Test
    public void detachAllVisitsEveryBackend() {
        BackendResources.attachment(backend, KEY_A, () -> new Recorder("a1"));
        BackendResources.attachment(other, KEY_A, () -> new Recorder("a2"));
        BackendResources.attachment(other, KEY_B, () -> new Recorder("b"));

        BackendResources.detachAll(KEY_A);

        assertEquals(2, closed.size());
        assertTrue(closed.containsAll(List.of("a1", "a2")));
        assertNull(BackendResources.existing(backend, KEY_A));
        assertNull(BackendResources.existing(other, KEY_A));
        assertTrue(BackendResources.existing(other, KEY_B) instanceof Recorder);
    }

    @Test
    public void attachmentAfterCloseAllStartsAFreshEntry() {
        BackendResources.attachment(backend, KEY_A, () -> new Recorder("a"));
        BackendResources.closeAll(backend);
        Recorder fresh = BackendResources.attachment(backend, KEY_A, () -> new Recorder("fresh"));
        assertSame(fresh, BackendResources.existing(backend, KEY_A));
    }
}
