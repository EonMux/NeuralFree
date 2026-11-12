package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.StateTransfer;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeOp;
import com.eonmux.neuralfree.gguf.testutil.FakeOpKind;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * A block leaving a device memory for the host tier ({@link BlockImage#read}) reads every cache layer's K and V rows
 * behind one wait: the reads are enqueued back to back and the queue drains once, instead of once per read, and the
 * image installs the bytes the device held.
 */
public class BlockImageDeviceReadTest {

    private static final int PAGE = 4;
    private static final int CTX = 32;
    private static final int WIDTH = 8;
    private static final int LAYERS = 3;

    private FakeComputeBackend backend;

    @Before
    public void setUp() {
        backend = FakeComputeBackend.create();
    }

    @After
    public void tearDown() {
        backend.close();
    }

    private static MemoryPlan plan() {
        MemoryConfig cfg = MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16)
                .nCtx(CTX).ubatch(8).parallel(1).kvPageSize(PAGE).build();
        MemoryPlan.Builder b = MemoryPlan.builder(MemoryKind.KV, CTX, 8)
                .group(MemoryPlan.KvGroup.of(0, CTX, false, KvWindow.FULL_CAUSAL, PAGE, CTX / PAGE));
        for (int il = 0; il < LAYERS; il++) {
            b.layer(LayerMemoryPlan.builder(il, LayerMemoryPlan.Kind.KV).kv(KvLayout.of(CTX, false, WIDTH, WIDTH,
                    KvDType.F16, KvDType.F16, KvWindow.FULL_CAUSAL), 0).build());
        }
        return b.config(cfg).build();
    }

    private static long count(List<FakeOp> ops, FakeOpKind kind) {
        return ops.stream().filter(op -> op.kind() == kind).count();
    }

    @Test
    public void aBlocksRowsLeaveTheDeviceBehindOneDrain() {
        try (DeviceModelMemory mem = DeviceModelMemory.allocate(backend, backend.computeQueue(), plan(),
                MropeTextPositions.PPP0, null)) {
            BlockLayout layout = BlockLayout.of(mem);
            List<StateTransfer.Cache> caches = layout.caches();
            assertEquals(LAYERS, caches.size());
            int page = 2;
            Random r = new Random(5);
            byte[][][] stored = new byte[LAYERS][2][];
            for (int c = 0; c < LAYERS; c++) {
                for (int side = 0; side < 2; side++) {
                    long rowBytes = side == 0 ? caches.get(c).layout().rowBytesK() : caches.get(c).layout().rowBytesV();
                    stored[c][side] = new byte[Math.toIntExact(PAGE * rowBytes)];
                    r.nextBytes(stored[c][side]);
                    caches.get(c).write(side == 1, page * PAGE, PAGE, stored[c][side]);
                }
            }
            PrefixCache.Block block = new PrefixCache.Block(new byte[32], null,
                    CacheRows.ofTokens(new int[] {11, 12, 13, 14}), true, null, new int[] {page}, null);

            backend.clearLog();
            BlockImage image = BlockImage.read(block, PAGE, true, layout);
            List<FakeOp> ops = backend.ops();
            assertEquals("one read per layer's K and V", 2L * LAYERS, count(ops, FakeOpKind.READ));
            assertEquals("the reads drain the queue once", 1L, count(ops, FakeOpKind.FINISH));

            int dst = 5;
            image.install(layout, new int[] {dst});
            for (int c = 0; c < LAYERS; c++) {
                for (int side = 0; side < 2; side++) {
                    byte[] back = new byte[stored[c][side].length];
                    caches.get(c).read(side == 1, dst * PAGE, PAGE, back);
                    assertArrayEquals("layer " + c + (side == 0 ? " K" : " V"), stored[c][side], back);
                }
            }
        }
    }
}
