package com.eonmux.neuralfree.distributed;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorValues;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.function.BiConsumer;
import java.util.function.IntFunction;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * The collectives of {@link SharedMemoryBackend} and {@link TCPBackend} with three ranks, each on its own thread with
 * its own CPU backend: in-place all-reduce (SUM, MEAN, MAX) and broadcast, all-gather, barriers, and repeated rounds
 * of one collective; and calls that differ between the ranks (size, op, kind, source rank), which fail on every rank
 * and leave the backend usable. Every expected value is exact in float.
 */
public class CollectivesTest {

    private static final int RANKS = 3;

    @Test
    public void sharedMemoryCollectives() throws Exception {
        String prefix = "nf_collectives_test_" + System.nanoTime();
        runRanks(rank -> new SharedMemoryBackend(rank, RANKS, prefix), CollectivesTest::exerciseRank);
    }

    @Test
    public void tcpCollectives() throws Exception {
        String[] addresses = {"localhost", "localhost", "localhost"};
        int[] ports = {12371, 12372, 12373};
        runRanks(rank -> new TCPBackend(rank, RANKS, addresses, ports), CollectivesTest::exerciseRank);
    }

    @Test
    public void sharedMemoryMismatchFailsOnEveryRank() throws Exception {
        String prefix = "nf_collectives_mismatch_" + System.nanoTime();
        runRanks(rank -> new SharedMemoryBackend(rank, RANKS, prefix), CollectivesTest::exerciseMismatches);
    }

    @Test
    public void tcpMismatchFailsOnEveryRank() throws Exception {
        String[] addresses = {"localhost", "localhost", "localhost"};
        int[] ports = {12374, 12375, 12376};
        runRanks(rank -> new TCPBackend(rank, RANKS, addresses, ports), CollectivesTest::exerciseMismatches);
    }

    @Test
    public void inPlaceCollectivesRequireFloat32Tensors() {
        ComputeBackend host = TrainParity.hostBackend();
        try (SharedMemoryBackend single = new SharedMemoryBackend(0, 1, "nf_collectives_dtype_" + System.nanoTime());
             Tensor indices = Tensor.fromArray(host, new int[]{1, 2})) {
            assertThrows(IllegalArgumentException.class, () -> single.allReduce(indices, ReduceOp.SUM));
            assertThrows(IllegalArgumentException.class, () -> single.broadcast(indices, 0));
        } finally {
            TrainGpu.release(host);
        }
    }

    /** All-gather exchanges FLOAT16, BFLOAT16 and FLOAT64 inputs and returns them in the input's shape and dtype. */
    @Test
    public void allGatherKeepsTheInputDType() {
        ComputeBackend host = TrainParity.hostBackend();
        float[] values = {1.5f, -2f, 0f, 3f};
        try (SharedMemoryBackend single = new SharedMemoryBackend(0, 1, "nf_collectives_gather_" + System.nanoTime());
             Tensor floats = Tensor.fromArray(host, new float[][]{{1.5f, -2f}, {0f, 3f}})) {
            for (DType dtype : new DType[]{DType.FLOAT16, DType.BFLOAT16, DType.FLOAT64}) {
                try (Tensor input = floats.to(dtype)) {
                    List<Tensor> output = new ArrayList<>(Arrays.asList((Tensor) null));
                    single.allGather(output, input);
                    try (Tensor gathered = output.get(0)) {
                        assertEquals(dtype, gathered.dtype());
                        assertEquals(input.shape(), gathered.shape());
                        assertArrayEquals(dtype.toString(), values, TensorValues.toFloats(gathered), 0f);
                    }
                }
            }
        } finally {
            TrainGpu.release(host);
        }
    }

    /** Runs {@code body} on every rank, each on its own thread, and rethrows the first rank failure. */
    private static void runRanks(IntFunction<DistributedBackend> factory,
                                 BiConsumer<DistributedBackend, Integer> body) throws Exception {
        ExecutorService pool = Executors.newFixedThreadPool(RANKS);
        try {
            List<Future<?>> ranks = new ArrayList<>();
            for (int r = 0; r < RANKS; r++) {
                int rank = r;
                ranks.add(pool.submit(() -> {
                    try (DistributedBackend backend = factory.apply(rank)) {
                        body.accept(backend, rank);
                    }
                    return null;
                }));
            }
            for (Future<?> rank : ranks) {
                try {
                    rank.get(120, TimeUnit.SECONDS);
                } catch (ExecutionException e) {
                    if (e.getCause() instanceof Exception) {
                        throw (Exception) e.getCause();
                    }
                    throw new AssertionError(e.getCause());
                }
            }
        } finally {
            pool.shutdownNow();
        }
    }

    private static void exerciseRank(DistributedBackend backend, int rank) {
        ComputeBackend host = TrainParity.hostBackend();
        try {
            float[] contribution = {rank + 1, -(rank + 1), 0.5f * rank};
            assertArrayEquals("SUM", new float[]{6f, -6f, 1.5f}, allReduce(backend, host, contribution, ReduceOp.SUM),
                0f);
            assertArrayEquals("MEAN", new float[]{2f, -2f, 0.5f},
                allReduce(backend, host, contribution, ReduceOp.MEAN), 0f);
            assertArrayEquals("MAX", new float[]{3f, -1f, 1f}, allReduce(backend, host, contribution, ReduceOp.MAX),
                0f);

            try (Tensor values = Tensor.fromArray(host, rank == 1 ? new float[]{7f, 8f} : new float[]{0f, 0f})) {
                backend.broadcast(values, 1);
                assertArrayEquals("broadcast from rank 1", new float[]{7f, 8f}, values.toFloatArray(), 0f);
            }

            List<Tensor> gathered = new ArrayList<>(Arrays.asList(new Tensor[RANKS]));
            try (Tensor input = Tensor.fromArray(host, new float[]{rank, 10f * rank})) {
                backend.allGather(gathered, input);
                for (int r = 0; r < RANKS; r++) {
                    assertEquals("gathered shape", input.shape(), gathered.get(r).shape());
                    assertArrayEquals("gathered from rank " + r, new float[]{r, 10f * r},
                        gathered.get(r).toFloatArray(), 0f);
                }
            } finally {
                for (Tensor t : gathered) {
                    if (t != null) {
                        t.close();
                    }
                }
            }

            backend.barrier();
            backend.barrier();
            for (int round = 0; round < 5; round++) {
                float[] one = {round, 1f};
                assertArrayEquals("round " + round, new float[]{RANKS * round, RANKS},
                    allReduce(backend, host, one, ReduceOp.SUM), 0f);
            }
        } finally {
            TrainGpu.release(host);
        }
    }

    /**
     * Calls that differ between the ranks: rank 2 reduces three values where the others reduce two, rank 1 takes the
     * maximum where the others sum, rank 0 enters a barrier where the others all-reduce, rank 2 broadcasts from itself
     * where the others receive from rank 1. Every rank fails each time, and a matching all-reduce then succeeds.
     */
    private static void exerciseMismatches(DistributedBackend backend, int rank) {
        ComputeBackend host = TrainParity.hostBackend();
        try {
            float[] sized = rank == 2 ? new float[]{1f, 2f, 3f} : new float[]{1f, 2f};
            assertThrows("size", IllegalArgumentException.class,
                () -> allReduce(backend, host, sized, ReduceOp.SUM));
            assertThrows("op", IllegalArgumentException.class,
                () -> allReduce(backend, host, new float[]{1f}, rank == 1 ? ReduceOp.MAX : ReduceOp.SUM));
            assertThrows("kind", IllegalArgumentException.class, () -> {
                if (rank == 0) {
                    backend.barrier();
                } else {
                    allReduce(backend, host, new float[]{1f, 2f}, ReduceOp.SUM);
                }
            });
            try (Tensor values = Tensor.fromArray(host, new float[]{rank, rank})) {
                assertThrows("source rank", IllegalArgumentException.class,
                    () -> backend.broadcast(values, rank == 2 ? 2 : 1));
            }
            assertArrayEquals("usable afterwards", new float[]{6f},
                allReduce(backend, host, new float[]{rank + 1}, ReduceOp.SUM), 0f);
        } finally {
            TrainGpu.release(host);
        }
    }

    private static float[] allReduce(DistributedBackend backend, ComputeBackend host, float[] values, ReduceOp op) {
        try (Tensor tensor = Tensor.fromArray(host, values)) {
            backend.allReduce(tensor, op);
            return tensor.toFloatArray();
        }
    }
}
