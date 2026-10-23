package com.eonmux.neuralfree.distributed;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.Test;
import static org.junit.Assert.*;

import java.io.DataOutputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

public class TCPBackendTest {

    @Test
    public void testConstructionWithValidParams() {
        String[] addresses = {"localhost", "localhost"};
        int[] ports = {12345, 12346};
        
        TCPBackend backend = new TCPBackend(0, 2, addresses, ports);
        
        assertEquals(0, backend.rank());
        assertEquals(2, backend.worldSize());
        assertTrue(backend.isInitialized());
        
        backend.close();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidRankThrowsNegative() {
        String[] addresses = {"localhost", "localhost"};
        int[] ports = {12345, 12346};
        new TCPBackend(-1, 2, addresses, ports);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidRankThrowsTooHigh() {
        String[] addresses = {"localhost", "localhost"};
        int[] ports = {12345, 12346};
        new TCPBackend(2, 2, addresses, ports);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testMismatchedAddressesLengthThrows() {
        String[] addresses = {"localhost"};
        int[] ports = {12345, 12346};
        new TCPBackend(0, 2, addresses, ports);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testMismatchedPortsLengthThrows() {
        String[] addresses = {"localhost", "localhost"};
        int[] ports = {12345};
        new TCPBackend(0, 2, addresses, ports);
    }

    @Test
    public void testOperationsAfterCloseThrow() {
        String[] addresses = {"localhost", "localhost"};
        int[] ports = {12347, 12348};
        
        TCPBackend backend = new TCPBackend(0, 2, addresses, ports);
        backend.close();
        
        try {
            backend.rank();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException expected) {}
        
        try {
            backend.worldSize();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException expected) {}
    }

    @Test
    public void testCloseIsIdempotent() {
        String[] addresses = {"localhost"};
        int[] ports = {12349};
        
        TCPBackend backend = new TCPBackend(0, 1, addresses, ports);
        
        backend.close();
        backend.close();
        backend.close();
    }

    @Test
    public void testSingleProcessTCPBackend() {
        String[] addresses = {"localhost"};
        int[] ports = {12350};
        
        TCPBackend backend = new TCPBackend(0, 1, addresses, ports);
        
        assertEquals(0, backend.rank());
        assertEquals(1, backend.worldSize());
        
        backend.barrier();
        
        backend.close();
    }

    /**
     * Connections that fail the handshake (closed at once, junk bytes, a valid handshake announcing a rank outside
     * the job) are refused one by one, and the real rank 1 is still accepted afterwards.
     */
    @Test
    public void testStrayConnectionsDoNotStopAccepting() throws Exception {
        String[] addresses = {"localhost", "localhost"};
        int[] ports = {12361, 12362};
        TCPBackend rank0 = new TCPBackend(0, 2, addresses, ports);
        try {
            try (Socket closedAtOnce = connect(ports[0])) {
                assertTrue(closedAtOnce.isConnected());
            }
            try (Socket junk = connect(ports[0])) {
                junk.getOutputStream().write(new byte[]{1, 2, 3, 4});
                junk.getOutputStream().flush();
                assertEquals("refused and closed", -1, junk.getInputStream().read());
            }
            try (Socket outOfRange = connect(ports[0])) {
                writeHandshake(outOfRange, 2, 7);
                assertEquals("refused and closed", -1, outOfRange.getInputStream().read());
            }
            try (Socket otherJob = connect(ports[0])) {
                writeHandshake(otherJob, 3, 1);
                assertEquals("refused and closed", -1, otherJob.getInputStream().read());
            }
            TCPBackend rank1 = startRank1(addresses, ports);
            try {
                CompletableFuture<Void> barrier1 = CompletableFuture.runAsync(rank1::barrier);
                rank0.barrier();
                barrier1.get(60, TimeUnit.SECONDS);
            } finally {
                rank1.close();
            }
        } finally {
            rank0.close();
        }
    }

    /** A second connection announcing an already connected rank is refused; the established connection stays. */
    @Test
    public void testDuplicateRankConnectionIsRefused() throws Exception {
        String[] addresses = {"localhost", "localhost"};
        int[] ports = {12363, 12364};
        TCPBackend rank0 = new TCPBackend(0, 2, addresses, ports);
        try {
            TCPBackend rank1 = startRank1(addresses, ports);
            try (Socket impostor = connect(ports[0])) {
                writeHandshake(impostor, 2, 1);
                assertEquals("refused and closed", -1, impostor.getInputStream().read());
                CompletableFuture<Void> barrier1 = CompletableFuture.runAsync(rank1::barrier);
                rank0.barrier();
                barrier1.get(60, TimeUnit.SECONDS);
            } finally {
                rank1.close();
            }
        } finally {
            rank0.close();
        }
    }

    /** A peer that goes away makes the next collective fail at once, and the failed backend closes itself. */
    @Test
    public void testLostPeerFailsTheCollectiveAndClosesTheBackend() throws Exception {
        String[] addresses = {"localhost", "localhost"};
        int[] ports = {12365, 12366};
        ComputeBackend host = TrainParity.hostBackend();
        TCPBackend rank0 = new TCPBackend(0, 2, addresses, ports);
        try (Tensor values = Tensor.fromArray(host, new float[]{1f, 2f})) {
            TCPBackend rank1 = startRank1(addresses, ports);
            CompletableFuture<Void> barrier1 = CompletableFuture.runAsync(rank1::barrier);
            rank0.barrier();
            barrier1.get(60, TimeUnit.SECONDS);
            rank1.close();
            assertThrows(RuntimeException.class, () -> rank0.allReduce(values, ReduceOp.SUM));
            assertFalse("closed after the failure", rank0.isInitialized());
        } finally {
            rank0.close();
            TrainGpu.release(host);
        }
    }

    private static TCPBackend startRank1(String[] addresses, int[] ports) throws Exception {
        return CompletableFuture.supplyAsync(() -> new TCPBackend(1, 2, addresses, ports)).get(60, TimeUnit.SECONDS);
    }

    /** A raw connection to {@code port} whose reads give up after 30 seconds. */
    private static Socket connect(int port) throws IOException {
        Socket socket = new Socket();
        socket.connect(new InetSocketAddress("localhost", port), 5000);
        socket.setSoTimeout(30_000);
        return socket;
    }

    private static void writeHandshake(Socket socket, int worldSize, int rank) throws IOException {
        DataOutputStream out = new DataOutputStream(socket.getOutputStream());
        out.writeInt(TCPBackend.HANDSHAKE_MAGIC);
        out.writeInt(worldSize);
        out.writeInt(rank);
        out.flush();
    }
}
