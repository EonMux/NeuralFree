package com.eonmux.neuralfree.gguf.rpc;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeDispatch;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The server's resolution of a dispatch's buffer arguments against a connection that holds thousands of buffers,
 * over two fake devices served in-process: every buffer argument reaches the device as the buffer its id names, and
 * an unknown id or a buffer of another device than the kernel's is refused with its message before anything runs.
 */
public class RpcDispatchArgumentsTest {

    private static final String KEY = "dispatch-arguments-key";
    private static final int BUFFERS = 3000;

    private FakeComputeBackend first;
    private FakeComputeBackend second;
    private RpcComputeServer server;
    private RpcConnection connection;

    @Before
    public void setUp() throws IOException {
        first = FakeComputeBackend.create();
        second = FakeComputeBackend.create();
        server = new RpcComputeServer(Arrays.<ComputeBackend>asList(first, second), KEY, "127.0.0.1", 0, null,
                null);
        server.start();
        connection = RpcConnection.open(RpcServerSpec.parse("127.0.0.1:" + server.port() + "@" + KEY));
    }

    @After
    public void tearDown() {
        if (connection != null) {
            connection.close();
        }
        if (server != null) {
            server.close();
        }
    }

    @Test
    public void resolvesEveryBufferArgumentAmongThousandsOfBuffers() {
        long[] ids = new long[BUFFERS];
        for (int i = 0; i < BUFFERS; i++) {
            ids[i] = alloc(0, sizeOf(i));
        }
        long kernel = compile(0);
        int[] picks = {BUFFERS - 1, 0, BUFFERS / 2, 7};
        Object[] args = {buffer(ids[picks[0]]), 42, buffer(ids[picks[1]]), 1.5f, buffer(ids[picks[2]]), 9L,
                buffer(ids[picks[3]])};
        enqueue(kernel, args);
        connection.call(new RpcWire.Writer(RpcWire.OP_FINISH));

        List<FakeDispatch> dispatches = first.dispatchesOf("probe");
        assertEquals(1, dispatches.size());
        FakeDispatch d = dispatches.get(0);
        assertEquals(args.length, d.argCount());
        assertEquals(sizeOf(picks[0]), ((GpuBuffer) d.arg(0)).sizeInBytes());
        assertEquals(42, d.arg(1));
        assertEquals(sizeOf(picks[1]), ((GpuBuffer) d.arg(2)).sizeInBytes());
        assertEquals(1.5f, d.arg(3));
        assertEquals(sizeOf(picks[2]), ((GpuBuffer) d.arg(4)).sizeInBytes());
        assertEquals(9L, d.arg(5));
        assertEquals(sizeOf(picks[3]), ((GpuBuffer) d.arg(6)).sizeInBytes());
        assertSame("the same id resolves to the same device buffer", d.arg(0), resolvedAgain(kernel, ids[picks[0]]));
    }

    @Test
    public void refusesABufferIdTheConnectionDidNotAllocate() {
        long id = alloc(0, 64);
        long kernel = compile(0);
        long unknown = id + 1000;
        try {
            enqueue(kernel, new Object[]{buffer(id), buffer(unknown)});
            fail("a dispatch with an unknown buffer id ran");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage()
                    .endsWith("buffer id " + unknown + " was not allocated by this connection"));
        }
        assertTrue(first.dispatchesOf("probe").isEmpty());
    }

    @Test
    public void refusesABufferOfAnotherDevice() {
        long onFirst = alloc(0, 64);
        long onSecond = alloc(1, 64);
        long kernel = compile(1);
        try {
            enqueue(kernel, new Object[]{buffer(onSecond), 3, buffer(onFirst)});
            fail("a kernel on device 1 took a buffer of device 0");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage()
                    .endsWith("a kernel on device 1 takes a buffer of device 0"));
        }
        assertTrue(second.dispatchesOf("probe").isEmpty());
        enqueue(kernel, new Object[]{buffer(onSecond), 3});
        connection.call(new RpcWire.Writer(RpcWire.OP_FINISH));
        assertEquals("the connection serves on after the refusal", 1, second.dispatchesOf("probe").size());
    }

    @Test
    public void reportsAnUnknownIdBeforeAForeignDevice() {
        long onFirst = alloc(0, 64);
        long kernel = compile(1);
        long unknown = onFirst + 1000;
        try {
            enqueue(kernel, new Object[]{buffer(onFirst), buffer(unknown)});
            fail("the dispatch ran");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage()
                    .endsWith("buffer id " + unknown + " was not allocated by this connection"));
        }
    }

    private Object resolvedAgain(long kernel, long id) {
        enqueue(kernel, new Object[]{buffer(id)});
        connection.call(new RpcWire.Writer(RpcWire.OP_FINISH));
        List<FakeDispatch> dispatches = first.dispatchesOf("probe");
        return dispatches.get(dispatches.size() - 1).arg(0);
    }

    private static long sizeOf(int index) {
        return 16L + 4L * index;
    }

    private long alloc(int device, long size) {
        return connection.call(new RpcWire.Writer(RpcWire.OP_ALLOC).i32(device).i64(size).bool(false)).i64();
    }

    private long compile(int device) {
        return connection.call(new RpcWire.Writer(RpcWire.OP_COMPILE).i32(device).utf("probe")
                .rawString("__kernel void probe() {}").nullableUtf(null)).i64();
    }

    private void enqueue(long kernel, Object[] args) {
        RpcWire.Writer w = new RpcWire.Writer(RpcWire.OP_ENQUEUE).i64(kernel).longArray(new long[]{64})
                .longArray(new long[]{64});
        RpcWire.writeArgs(w, args);
        connection.call(w);
    }

    private static RemoteBuffer buffer(long id) {
        return new RemoteBuffer(null, id, 16);
    }
}
