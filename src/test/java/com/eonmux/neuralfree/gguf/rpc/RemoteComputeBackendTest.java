package com.eonmux.neuralfree.gguf.rpc;

import com.eonmux.neuralfree.Config.ACCELERATION_MODE;
import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.KernelTaskBatch;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.gguf.gpu.pipeline.GpuDevices;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The remote backend against a real OpenCL device served in-process: the handshake reports the server's device, and
 * a kernel compiled, argument-bound and dispatched over the protocol adds one to a device buffer exactly as on the
 * local backend. Also the pipelined batch (a decode step's commands flush together and their events complete
 * lazily), the launch batch, the weight cache probe, the multi-device server, the key requirements and the
 * per-connection resource scope.
 */
public class RemoteComputeBackendTest {

    private static final String KEY = "remote-compute-test-key";
    private static final String ADD_ONE =
            "__kernel void add_one(__global float* x) { x[get_global_id(0)] += 1.0f; }";
    /** Above the write-probe threshold, so a cached write stores and probes. */
    private static final int WEIGHT_BYTES = 2 << 20;

    private HWAcceleration acceleration;
    private ComputeBackend local;
    private RpcComputeServer server;
    private RemoteComputeBackend remote;

    @Before
    public void setUp() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        acceleration = HWAcceleration.forMode(ACCELERATION_MODE.OPENCL);
        GpuDevices.Selected selected = GpuDevices.selectOne(acceleration, "0", "--device");
        local = selected.backend();
        server = new RpcComputeServer(local, KEY, "127.0.0.1", 0);
        server.start();
        remote = RemoteComputeBackend.connect(spec(server.port(), KEY));
    }

    @After
    public void tearDown() {
        if (remote != null) {
            remote.close();
        }
        if (server != null) {
            server.close();
        }
        if (acceleration != null) {
            acceleration.shutdown();
        }
        System.clearProperty(RemoteEndpoint.PROP_PIPELINE);
    }

    private static RpcServerSpec spec(int port, String key) {
        return RpcServerSpec.parse("127.0.0.1:" + port + "@" + key);
    }

    @Test
    public void reportsTheServersDevice() {
        DeviceCapabilities served = local.capabilities();
        DeviceCapabilities seen = remote.capabilities();
        assertEquals(local.backendType(), remote.backendType());
        assertEquals(served.getDeviceName(), seen.getDeviceName());
        assertEquals(served.getTotalVramBytes(), seen.getTotalVramBytes());
        assertEquals(served.getMaxWorkGroupCount()[0], seen.getMaxWorkGroupCount()[0]);
        assertEquals(served.getDeviceClass(), seen.getDeviceClass());
        assertEquals(served.getIdentity(), seen.getIdentity());
        assertFalse(remote.supportsHostMappedBuffers());
        assertTrue(server.port() > 0);
    }

    /** A float read decodes the answer's payload into the destination's first floats, from any source offset. */
    @Test
    public void aFloatReadDecodesItsRangeFromTheAnswer() {
        GpuBuffer buffer = remote.allocateBuffer(4L * Float.BYTES);
        try {
            ComputeQueue queue = remote.computeQueue();
            queue.enqueueWriteBuffer(buffer, 0, 4L * Float.BYTES, new float[]{1f, -2.5f, 3f, Float.MIN_VALUE});
            float[] out = {9f, 9f, 9f};
            queue.enqueueReadBuffer(buffer, 2L * Float.BYTES, 2L * Float.BYTES, out);
            assertArrayEquals(new float[]{3f, Float.MIN_VALUE, 9f}, out, 0f);
            assertThrows(IllegalArgumentException.class,
                    () -> queue.enqueueReadBuffer(buffer, 0, 6, new float[2]));
            assertThrows(IllegalArgumentException.class,
                    () -> queue.enqueueReadBuffer(buffer, 0, 4L * Float.BYTES, new float[3]));
        } finally {
            buffer.close();
        }
    }

    @Test
    public void dispatchesAKernelOverTheProtocol() {
        KernelHandle kernel = remote.compileKernel("add_one", ADD_ONE, "");
        GpuBuffer buffer = remote.allocateBuffer(4L * Float.BYTES);
        try {
            ComputeQueue queue = remote.computeQueue();
            queue.enqueueWriteBuffer(buffer, 0, 4L * Float.BYTES, new float[]{1f, 2f, 3f, 4f});
            GpuEvent event = queue.enqueueKernel(kernel, new long[]{4}, new long[]{1}, new Object[]{buffer});
            assertNotNull(event);
            assertEquals(-1L, event.getElapsedNanos());
            queue.finish();
            assertTrue(event.isComplete());
            float[] out = new float[4];
            queue.enqueueReadBuffer(buffer, 0, 4L * Float.BYTES, out);
            assertArrayEquals(new float[]{2f, 3f, 4f, 5f}, out, 0f);
        } finally {
            buffer.close();
        }
    }

    @Test
    public void pipelinesADecodeStepIntoOneFlush() {
        // the shape of a decode step: a positions write, several dispatches, then the hand-off read; the commands
        // buffer and their events stay pending until an observation flushes the batch and finishes the device
        KernelHandle kernel = remote.compileKernel("add_one", ADD_ONE, "");
        GpuBuffer buffer = remote.allocateBuffer(4L * Float.BYTES);
        try {
            ComputeQueue queue = remote.computeQueue();
            queue.enqueueWriteBuffer(buffer, 0, 4L * Float.BYTES, new float[]{0f, 1f, 2f, 3f});
            GpuEvent write = queue.enqueueMarker();
            GpuEvent first = queue.enqueueKernel(kernel, new long[]{4}, new long[]{1}, new Object[]{buffer});
            GpuEvent second = queue.enqueueKernel(kernel, new long[]{4}, new long[]{1}, new Object[]{buffer});
            assertFalse("the buffered commands complete only when observed", write.isComplete());
            assertFalse(first.isComplete());
            assertFalse(second.isComplete());
            float[] out = new float[4];
            queue.enqueueReadBuffer(buffer, 0, 4L * Float.BYTES, out);
            assertArrayEquals(new float[]{2f, 3f, 4f, 5f}, out, 0f);
            assertTrue(first.isComplete());
            assertTrue(second.isComplete());
        } finally {
            buffer.close();
        }
    }

    @Test
    public void awaitsADeferredEventWithoutARoundTripCommand() throws InterruptedException {
        KernelHandle kernel = remote.compileKernel("add_one", ADD_ONE, "");
        GpuBuffer buffer = remote.allocateBuffer(4L * Float.BYTES);
        try {
            ComputeQueue queue = remote.computeQueue();
            queue.enqueueWriteBuffer(buffer, 0, 4L * Float.BYTES, new float[]{1f, 2f, 3f, 4f});
            GpuEvent dispatched = queue.enqueueKernel(kernel, new long[]{4}, new long[]{1}, new Object[]{buffer});
            dispatched.awaitCompletion();
            assertTrue(dispatched.isComplete());
            dispatched.close();
            float[] out = new float[4];
            queue.enqueueReadBuffer(buffer, 0, 4L * Float.BYTES, out);
            assertArrayEquals(new float[]{2f, 3f, 4f, 5f}, out, 0f);
        } finally {
            buffer.close();
        }
    }

    @Test
    public void keepsTheSynchronousProtocolWhenThePipelineIsOff() throws IOException {
        System.setProperty(RemoteEndpoint.PROP_PIPELINE, "false");
        remote.close();
        server.close();
        server = new RpcComputeServer(local, KEY, "127.0.0.1", 0);
        server.start();
        remote = RemoteComputeBackend.connect(spec(server.port(), KEY));
        assertFalse(remote.batch().pipelined());
        KernelHandle kernel = remote.compileKernel("add_one", ADD_ONE, "");
        GpuBuffer buffer = remote.allocateBuffer(4L * Float.BYTES);
        try {
            ComputeQueue queue = remote.computeQueue();
            queue.enqueueWriteBuffer(buffer, 0, 4L * Float.BYTES, new float[]{1f, 2f, 3f, 4f});
            GpuEvent event = queue.enqueueKernel(kernel, new long[]{4}, new long[]{1}, new Object[]{buffer});
            assertTrue("a synchronous command completes with its answer", event.isComplete());
            float[] out = new float[4];
            queue.enqueueReadBuffer(buffer, 0, 4L * Float.BYTES, out);
            assertArrayEquals(new float[]{2f, 3f, 4f, 5f}, out, 0f);
        } finally {
            buffer.close();
        }
    }

    @Test
    public void refusesAnInvalidPipelineSetting() throws IOException {        System.setProperty(RemoteEndpoint.PROP_PIPELINE, "maybe");
        try {
            RemoteComputeBackend.connect(spec(server.port(), KEY));
            fail("accepted neuralfree.rpc.pipeline=maybe");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains(RemoteEndpoint.PROP_PIPELINE));
        }
    }

    @Test
    public void launchesABatchOfDispatchesAsOneSubmission() {
        KernelHandle kernel = remote.compileKernel("add_one", ADD_ONE, "");
        GpuBuffer left = remote.allocateBuffer(2L * Float.BYTES);
        GpuBuffer right = remote.allocateBuffer(2L * Float.BYTES);
        try {
            ComputeQueue queue = remote.computeQueue();
            assertTrue(queue.supportsKernelBatching());
            KernelTaskBatch batch = queue.beginKernelBatch();
            queue.enqueueWriteBuffer(left, 0, 2L * Float.BYTES, new float[]{0f, 10f});
            queue.enqueueWriteBuffer(right, 0, 2L * Float.BYTES, new float[]{100f, 200f});
            batch.add(kernel, new long[]{2}, new long[]{1}, new Object[]{left});
            batch.add(kernel, new long[]{2}, new long[]{1}, new Object[]{right});
            batch.add(kernel, new long[]{2}, new long[]{1}, new Object[]{right});
            assertEquals(3, batch.taskCount());
            GpuEvent launched = batch.launch(queue);
            assertEquals(1, batch.launches());
            assertEquals(3, batch.tasksLaunched());
            assertEquals(0, batch.graphReuses());
            assertEquals(0, batch.taskCount());
            queue.finish();
            assertTrue(launched.isComplete());
            batch.close();
            float[] gotLeft = new float[2];
            float[] gotRight = new float[2];
            queue.enqueueReadBuffer(left, 0, 2L * Float.BYTES, gotLeft);
            queue.enqueueReadBuffer(right, 0, 2L * Float.BYTES, gotRight);
            assertArrayEquals(new float[]{1f, 11f}, gotLeft, 0f);
            assertArrayEquals(new float[]{102f, 202f}, gotRight, 0f);
        } finally {
            left.close();
            right.close();
        }
    }

    @Test
    public void refusesAnEmptyBatchLaunch() {
        KernelHandle kernel = remote.compileKernel("add_one", ADD_ONE, "");
        ComputeQueue queue = remote.computeQueue();
        KernelTaskBatch batch = queue.beginKernelBatch();
        try {
            batch.launch(queue);
            fail("launched an empty batch");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("empty"));
        } finally {
            batch.close();
            assertTrue(batch.isClosed());
        }
    }

    @Test
    public void copiesFillsAndReadsBack() {
        GpuBuffer source = remote.allocateBuffer(4L * Float.BYTES);
        GpuBuffer target = remote.allocateBuffer(4L * Float.BYTES);
        try {
            ComputeQueue queue = remote.computeQueue();
            queue.enqueueWriteBuffer(source, 0, 4L * Float.BYTES, new float[]{5f, 6f, 7f, 8f});
            queue.enqueueCopyBuffer(source, 0, target, 0, 4L * Float.BYTES);
            queue.finish();
            float[] copied = new float[4];
            queue.enqueueReadBuffer(target, 0, 4L * Float.BYTES, copied);
            assertArrayEquals(new float[]{5f, 6f, 7f, 8f}, copied, 0f);

            queue.enqueueFillBuffer(target, 0, 4L * Float.BYTES, 0);
            queue.finish();
            float[] zeroed = new float[]{9f, 9f, 9f, 9f};
            queue.enqueueReadBuffer(target, 0, 4L * Float.BYTES, zeroed);
            assertArrayEquals(new float[4], zeroed, 0f);
        } finally {
            source.close();
            target.close();
        }
    }

    @Test
    public void tracksAllocatedBytesWithTheOpenBuffers() {
        long before = remote.getAllocatedBytes();
        GpuBuffer buffer = remote.allocateBuffer(1024);
        assertEquals(before + 1024, remote.getAllocatedBytes());
        buffer.close();
        assertEquals(before, remote.getAllocatedBytes());
    }

    @Test
    public void opensOneViewPerServedDevice() throws IOException {
        List<CLDevice> devices = acceleration.getCLDevices();
        org.junit.Assume.assumeTrue("the multi-device server needs two OpenCL devices, found "
                + devices.size(), devices.size() >= 2);
        ComputeBackend second = acceleration.createBackend(devices.get(1));
        try {
            RpcComputeServer multi = new RpcComputeServer(
                    java.util.Arrays.<ComputeBackend>asList(local, second), KEY, "127.0.0.1", 0, null, null);
            try {
                multi.start();
                List<RemoteComputeBackend> views = RemoteComputeBackend.openAll(spec(multi.port(), KEY));
                assertEquals(2, views.size());
                assertEquals(local.capabilities().getDeviceName(), views.get(0).capabilities().getDeviceName());
                assertEquals(second.capabilities().getDeviceName(), views.get(1).capabilities().getDeviceName());
                // device 1 takes a buffer, a kernel and a dispatch of its own
                RemoteComputeBackend onSecond = views.get(1);
                KernelHandle kernel = onSecond.compileKernel("add_one", ADD_ONE, "");
                GpuBuffer buffer = onSecond.allocateBuffer(2L * Float.BYTES);
                try {
                    ComputeQueue queue = onSecond.computeQueue();
                    queue.enqueueWriteBuffer(buffer, 0, 2L * Float.BYTES, new float[]{7f, 8f});
                    queue.enqueueKernel(kernel, new long[]{2}, new long[]{1}, new Object[]{buffer});
                    queue.finish();
                    float[] out = new float[2];
                    queue.enqueueReadBuffer(buffer, 0, 2L * Float.BYTES, out);
                    assertArrayEquals(new float[]{8f, 9f}, out, 0f);
                } finally {
                    buffer.close();
                }
                for (RemoteComputeBackend view : views) {
                    view.close();
                }
            } finally {
                multi.close();
            }
        } finally {
            second.close();
        }
    }

    @Test
    public void probesTheWeightCacheAcrossAReconnect() throws IOException {
        Path dir = Files.createTempDirectory("rpc-weight-cache");
        try {
            server.close();
            RpcWeightCache cache = new RpcWeightCache(dir);
            server = new RpcComputeServer(java.util.Collections.singletonList(local), KEY, "127.0.0.1", 0,
                    cache, null);
            server.start();
            byte[] weights = new byte[WEIGHT_BYTES];
            for (int i = 0; i < weights.length; i++) {
                weights[i] = (byte) (i * 31);
            }
            long hash = RpcWire.fnv1a(weights);
            // first upload stores the payload under its content hash
            try (RpcConnection first = RpcConnection.open(spec(server.port(), KEY))) {
                long id = first.call(new RpcWire.Writer(RpcWire.OP_ALLOC).i32(0).i64(weights.length)
                        .bool(false)).i64();
                first.call(new RpcWire.Writer(RpcWire.OP_WRITE).i64(id).i64(0).rawBytes(weights).bool(true));
            }
            assertEquals(1, cache.entries());
            // a reconnect probes the same hash and the server writes the cached bytes to the new buffer
            try (RpcConnection second = RpcConnection.open(spec(server.port(), KEY))) {
                long id = second.call(new RpcWire.Writer(RpcWire.OP_ALLOC).i32(0).i64(weights.length)
                        .bool(false)).i64();
                boolean hit = second.call(new RpcWire.Writer(RpcWire.OP_WRITE_HASH).i64(id).i64(0)
                        .i64(weights.length).rawBytes(RpcWire.hashBytes(hash))).bool();
                assertTrue("the cached write answered a hit", hit);
                byte[] readBack = second.call(new RpcWire.Writer(RpcWire.OP_READ).i64(id).i64(0)
                        .i32(weights.length)).rawBytes();
                assertArrayEquals("the cached write carried the stored payload", weights, readBack);
            }
            // an unknown hash answers a miss
            try (RpcConnection third = RpcConnection.open(spec(server.port(), KEY))) {
                long id = third.call(new RpcWire.Writer(RpcWire.OP_ALLOC).i32(0).i64(4096).bool(false)).i64();
                boolean miss = third.call(new RpcWire.Writer(RpcWire.OP_WRITE_HASH).i64(id).i64(0).i64(4096)
                        .rawBytes(RpcWire.hashBytes(0x1234567890abcdefL))).bool();
                assertFalse(miss);
            }
            assertTrue(Files.exists(dir.resolve(RpcWire.hashName(hash))));
        } finally {
            deleteTree(dir);
        }
    }

    @Test
    public void servesACachedWriteEndToEnd() throws IOException {
        Path dir = Files.createTempDirectory("rpc-weight-cache-e2e");
        try {
            server.close();
            remote.close();
            server = new RpcComputeServer(java.util.Collections.singletonList(local), KEY, "127.0.0.1", 0,
                    new RpcWeightCache(dir), null);
            server.start();
            remote = RemoteComputeBackend.connect(spec(server.port(), KEY));
            // the same weights reach the buffer over a fresh connection; the payload round-trips through the cache
            byte[] weights = new byte[WEIGHT_BYTES];
            for (int i = 0; i < weights.length; i++) {
                weights[i] = (byte) (i * 7);
            }
            GpuBuffer buffer = remote.allocateBuffer(weights.length);
            try {
                ComputeQueue queue = remote.computeQueue();
                queue.enqueueWriteBuffer(buffer, 0, weights.length, weights);
                queue.finish();
                byte[] out = new byte[weights.length];
                queue.enqueueReadBuffer(buffer, 0, weights.length, out);
                assertArrayEquals(weights, out);
            } finally {
                buffer.close();
            }
        } finally {
            deleteTree(dir);
        }
    }

    @Test
    public void connectsOverTls() throws Exception {
        Path dir = Files.createTempDirectory("rpc-tls");
        try {
            TlsMaterial tls = TlsMaterial.generate(dir, "127.0.0.1");
            server.close();
            server = new RpcComputeServer(java.util.Collections.singletonList(local), KEY, "127.0.0.1", 0, null,
                    RpcTls.server(tls.cert, tls.key).context());
            server.start();
            System.setProperty(RpcTls.PROP_TRUSTSTORE, tls.truststore.toString());
            System.setProperty(RpcTls.PROP_TRUSTSTORE_PASSWORD, TlsMaterial.PASSWORD);
            RemoteComputeBackend secure = null;
            try {
                secure = RemoteComputeBackend.connect(RpcServerSpec.parse(
                        "tls://127.0.0.1:" + server.port() + "@" + KEY));
                assertEquals(local.backendType(), secure.backendType());
                KernelHandle kernel = secure.compileKernel("add_one", ADD_ONE, "");
                GpuBuffer buffer = secure.allocateBuffer(Float.BYTES);
                try {
                    ComputeQueue queue = secure.computeQueue();
                    queue.enqueueWriteBuffer(buffer, 0, Float.BYTES, new float[]{41f});
                    queue.enqueueKernel(kernel, new long[]{1}, new long[]{1}, new Object[]{buffer});
                    queue.finish();
                    float[] out = new float[1];
                    queue.enqueueReadBuffer(buffer, 0, Float.BYTES, out);
                    assertArrayEquals(new float[]{42f}, out, 0f);
                } finally {
                    buffer.close();
                }
            } finally {
                if (secure != null) {
                    secure.close();
                }
                System.clearProperty(RpcTls.PROP_TRUSTSTORE);
                System.clearProperty(RpcTls.PROP_TRUSTSTORE_PASSWORD);
            }
            // without the trust store the JVM's default authorities reject the self-signed certificate
            try {
                RemoteComputeBackend.connect(RpcServerSpec.parse("tls://127.0.0.1:" + server.port() + "@" + KEY));
                fail("connected without the server's certificate in a trust store");
            } catch (IOException expected) {
                // the TLS handshake fails
            }
        } finally {
            deleteTree(dir);
        }
    }

    @Test
    public void refusesAConnectionWithTheWrongKey() {
        try {
            RemoteComputeBackend.connect(spec(server.port(), "not-the-key"));
            fail("connected with the wrong key");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("authentication failed"));
        } catch (IOException e) {
            fail("the refusal arrived as an I/O error: " + e.getMessage());
        }
    }

    @Test
    public void refusesToServeWithoutAKey() {
        try {
            new RpcComputeServer(local, "  ", "127.0.0.1", 0);
            fail("a server without a key was accepted");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("key"));
        }
    }

    @Test
    public void scopesBufferIdsToTheirConnection() throws IOException {
        RpcServerSpec where = spec(server.port(), KEY);
        try (RpcConnection first = RpcConnection.open(where);
             RpcConnection second = RpcConnection.open(where)) {
            long id = first.call(new RpcWire.Writer(RpcWire.OP_ALLOC).i32(0).i64(16).bool(false)).i64();
            assertTrue(id > 0);
            try {
                second.call(new RpcWire.Writer(RpcWire.OP_WRITE).i64(id).i64(0).rawBytes(new byte[4]).bool(false));
                fail("another connection wrote into the first connection's buffer");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage().contains("not allocated by this connection"));
            }
        }
    }

    private static void deleteTree(Path dir) {
        try (java.util.stream.Stream<Path> files = Files.walk(dir)) {
            files.sorted(java.util.Comparator.reverseOrder()).forEach(p -> {
                try {
                    Files.deleteIfExists(p);
                } catch (IOException ignored) {
                    // a leftover temporary file is not a test failure
                }
            });
        } catch (IOException ignored) {
            // the tree is already gone
        }
    }

    /** A self-signed certificate and PKCS#8 key over {@code keytool}, plus the trust store the client opens. */
    private static final class TlsMaterial {
        static final String PASSWORD = "rpc-tls-test";
        final Path cert;
        final Path key;
        final Path truststore;

        TlsMaterial(Path cert, Path key, Path truststore) {
            this.cert = cert;
            this.key = key;
            this.truststore = truststore;
        }

        static TlsMaterial generate(Path dir, String host) throws Exception {
            Path keystore = dir.resolve("identity.p12");
            Path cert = dir.resolve("cert.pem");
            Path key = dir.resolve("key.pem");
            Path truststore = dir.resolve("truststore.p12");
            ProcessBuilder keytool = new ProcessBuilder("keytool", "-genkeypair", "-alias", "rpc",
                    "-keyalg", "EC", "-keysize", "256", "-dname", "CN=" + host, "-ext",
                    "SAN=dns:" + host + ",ip:127.0.0.1", "-storetype", "PKCS12",
                    "-keystore", keystore.toString(), "-storepass", PASSWORD, "-noprompt");
            keytool.redirectErrorStream(true);
            Process process = keytool.start();
            if (process.waitFor() != 0) {
                throw new IllegalStateException("keytool failed: "
                        + new String(process.getInputStream().readAllBytes()));
            }
            // export the certificate as PEM and the private key as unencrypted PKCS#8
            java.security.KeyStore store = java.security.KeyStore.getInstance("PKCS12");
            try (java.io.InputStream in = Files.newInputStream(keystore)) {
                store.load(in, PASSWORD.toCharArray());
            }
            java.security.cert.Certificate chain = store.getCertificate("rpc");
            Files.write(cert, pem("CERTIFICATE", chain.getEncoded()));
            java.security.Key identity = store.getKey("rpc", PASSWORD.toCharArray());
            Files.write(key, pem("PRIVATE KEY", identity.getEncoded()));
            // the client trusts the certificate through a PKCS12 store holding it
            java.security.KeyStore trust = java.security.KeyStore.getInstance("PKCS12");
            trust.load(null, null);
            trust.setCertificateEntry("rpc", chain);
            try (java.io.OutputStream out = Files.newOutputStream(truststore)) {
                trust.store(out, PASSWORD.toCharArray());
            }
            return new TlsMaterial(cert, key, truststore);
        }

        private static byte[] pem(String label, byte[] der) {
            String base64 = java.util.Base64.getMimeEncoder(64, "\n".getBytes()).encodeToString(der);
            return ("-----BEGIN " + label + "-----\n" + base64 + "\n-----END " + label + "-----\n")
                    .getBytes(java.nio.charset.StandardCharsets.UTF_8);
        }
    }
}
