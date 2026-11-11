package com.eonmux.neuralfree.gguf.rpc;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.util.Collections;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The read of a pipelined endpoint against a scripted server that speaks the wire protocol by hand: the read frame
 * arrives before the server answered the batch of buffered commands ahead of it (one round trip for both), a failed
 * batch raises its own error ahead of the read's answer, and the connection stays in step afterwards.
 */
public class RemoteEndpointPipelineTest {

    private static final String KEY = "pipeline-test-key";
    private static final byte[] DATA = {1, 2, 3, 4};

    private ServerSocket listening;
    private ExecutorService serverThread;
    private Socket accepted;

    @Before
    public void setUp() throws IOException {
        listening = new ServerSocket(0, 1, InetAddress.getLoopbackAddress());
        serverThread = Executors.newSingleThreadExecutor();
    }

    @After
    public void tearDown() throws IOException {
        serverThread.shutdownNow();
        if (accepted != null) {
            accepted.close();
        }
        listening.close();
    }

    @Test
    public void aReadRidesInTheRoundTripOfTheBufferedCommands() throws Exception {
        Future<?> script = serve(wire -> {
            assertEquals(RpcWire.OP_BATCH, firstByte(wire.readFrame()));
            // the read must arrive while the batch is still unanswered
            assertEquals(RpcWire.OP_READ, firstByte(wire.readFrame()));
            wire.writeFrame(new RpcWire.Writer(RpcWire.STATUS_OK).i32(2).toBytes());
            wire.writeFrame(new RpcWire.Writer(RpcWire.STATUS_OK).rawBytes(DATA).toBytes());
        });
        RemoteEndpoint endpoint = open();
        try {
            endpoint.submit(fill(), true);
            long seq = endpoint.submit(fill(), true);
            assertFalse(endpoint.finishedThrough(seq));
            byte[] got = endpoint.request(read(), true).rawBytes();
            assertArrayEquals(DATA, got);
            assertTrue("the read is an observation point for the batch", endpoint.finishedThrough(seq));
            script.get(10, TimeUnit.SECONDS);
        } finally {
            endpoint.abandon();
        }
    }

    @Test
    public void aFailedBatchRaisesItsErrorAheadOfThePipelinedRead() throws Exception {
        Future<?> script = serve(wire -> {
            assertEquals(RpcWire.OP_BATCH, firstByte(wire.readFrame()));
            assertEquals(RpcWire.OP_READ, firstByte(wire.readFrame()));
            wire.writeFrame(new RpcWire.Writer(RpcWire.STATUS_ERROR).u8(RpcWire.ERR_ILLEGAL_ARGUMENT)
                    .utf("[0 of 1 batch members executed] buffer id 9 was not allocated by this connection")
                    .toBytes());
            wire.writeFrame(new RpcWire.Writer(RpcWire.STATUS_OK).rawBytes(new byte[]{9, 9, 9, 9}).toBytes());
            // the next read goes out alone: nothing is buffered after the failed batch
            assertEquals(RpcWire.OP_READ, firstByte(wire.readFrame()));
            wire.writeFrame(new RpcWire.Writer(RpcWire.STATUS_OK).rawBytes(DATA).toBytes());
        });
        RemoteEndpoint endpoint = open();
        try {
            long seq = endpoint.submit(fill(), true);
            try {
                endpoint.request(read(), true);
                fail("the read hid the failed batch");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage()
                        .contains("[0 of 1 batch members executed] buffer id 9 was not allocated"));
            }
            assertFalse("a failed batch finished nothing", endpoint.finishedThrough(seq));
            assertArrayEquals("the connection stays in step", DATA, endpoint.request(read(), true).rawBytes());
            script.get(10, TimeUnit.SECONDS);
        } finally {
            endpoint.abandon();
        }
    }

    @Test
    public void aReadWithNothingBufferedIsOneFrame() throws Exception {
        Future<?> script = serve(wire -> {
            assertEquals(RpcWire.OP_READ, firstByte(wire.readFrame()));
            wire.writeFrame(new RpcWire.Writer(RpcWire.STATUS_OK).rawBytes(DATA).toBytes());
        });
        RemoteEndpoint endpoint = open();
        try {
            byte[] into = new byte[6];
            endpoint.request(read(), true).rawBytesInto(into, 2, DATA.length);
            assertArrayEquals(new byte[]{0, 0, 1, 2, 3, 4}, into);
            script.get(10, TimeUnit.SECONDS);
        } finally {
            endpoint.abandon();
        }
    }

    /** One scripted exchange after the handshake. */
    @FunctionalInterface
    private interface Script {
        void run(RpcWire wire) throws IOException;
    }

    private Future<?> serve(Script script) {
        return serverThread.submit(() -> {
            accepted = listening.accept();
            // a read that waits for a frame the client never sends fails the script instead of hanging
            accepted.setSoTimeout(5000);
            DataInputStream in = new DataInputStream(new BufferedInputStream(accepted.getInputStream()));
            DataOutputStream out = new DataOutputStream(new BufferedOutputStream(accepted.getOutputStream()));
            RpcWire wire = new RpcWire(in, out);
            out.writeInt(RpcWire.MAGIC);
            out.writeInt(RpcWire.VERSION);
            out.write(new byte[RpcWire.NONCE_BYTES]);
            out.flush();
            in.readFully(new byte[32]);
            RpcWire.Writer info = new RpcWire.Writer(RpcWire.STATUS_OK);
            RpcWire.writeHandshake(info, Collections.singletonList(new RemoteDeviceInfo(
                    ComputeBackend.BackendType.OPENCL, FakeComputeBackend.defaultCapabilities(), false,
                    "scripted")), false);
            wire.writeFrame(info.toBytes());
            try {
                script.run(wire);
            } catch (SocketTimeoutException e) {
                accepted.close();
                throw new AssertionError("the client waited for an answer before sending the next frame", e);
            } catch (IOException | RuntimeException | AssertionError e) {
                // a broken script releases the client blocked on its answer
                accepted.close();
                throw e;
            }
            return null;
        });
    }

    private RemoteEndpoint open() throws IOException {
        return RemoteEndpoint.open(RpcServerSpec.parse("127.0.0.1:" + listening.getLocalPort() + "@" + KEY));
    }

    private static RpcWire.Writer fill() {
        return new RpcWire.Writer(RpcWire.OP_FILL).i64(1).i64(0).i64(4).i32(0);
    }

    private static RpcWire.Writer read() {
        return new RpcWire.Writer(RpcWire.OP_READ).i64(1).i64(0).i32(DATA.length);
    }

    private static byte firstByte(byte[] frame) {
        return frame[0];
    }
}
