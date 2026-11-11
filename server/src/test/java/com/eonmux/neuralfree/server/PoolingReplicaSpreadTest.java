package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.backend;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A multi-input pooling call over a two-replica pool: its inputs spread over both replicas (each replica's engine
 * prefills a share of them) and the answers come back in input order, equal to the answers of a single replica.
 * The memoryless encoder pool answers its batch over both replicas the same way.
 */
class PoolingReplicaSpreadTest {

    private static final int INPUTS = 6;

    private static Path decoder;
    private static Path encoder;
    private static EngineService pool;
    private static EngineService single;
    private static EngineService encoderPool;
    private static EngineService encoderSingle;

    private static EngineConfig pagedConfig() {
        MemoryConfig memory = MemoryConfig.builder().nCtx(512).ubatch(8).parallel(2)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        return EngineConfig.builder().backend(backend()).pooling(PoolingKind.LAST).memory(memory).build();
    }

    private static ServerOptions replicas(int n) {
        return new ServerOptions(null, ServerOptions.DEFAULT_TIMEOUT, false, 0, n, null);
    }

    @BeforeAll
    static void open() throws IOException {
        decoder = SyntheticModels.llama().writeTempFile();
        pool = new EngineService(decoder.toString(), pagedConfig(), replicas(2));
        single = new EngineService(decoder.toString(), pagedConfig(), replicas(1));
        encoder = SyntheticModels.bert().writeTempFile();
        EngineConfig encoderConfig = EngineConfig.builder().backend(backend()).build();
        encoderPool = new EngineService(encoder.toString(), encoderConfig, replicas(2));
        encoderSingle = new EngineService(encoder.toString(), encoderConfig, replicas(1));
    }

    @AfterAll
    static void close() {
        pool.close();
        single.close();
        encoderPool.close();
        encoderSingle.close();
    }

    private static int[][] inputs() {
        int[][] out = new int[INPUTS][];
        for (int i = 0; i < INPUTS; i++) {
            out[i] = new int[3 + i];
            for (int t = 0; t < out[i].length; t++) {
                out[i][t] = 4 + (i * 11 + t * 7) % 50;
            }
        }
        return out;
    }

    @Test
    void theInputsSpreadOverBothReplicasAndAnswerInOrder() throws IOException {
        List<EngineReplica> replicas = pool.replicas();
        assertEquals(2, replicas.size());
        long[] before = new long[replicas.size()];
        for (int r = 0; r < before.length; r++) {
            before[r] = replicas.get(r).stats().promptTokens();
        }
        int[][] inputs = inputs();
        float[][] spread = pool.pooling().embedAll(null, inputs);
        float[][] alone = single.pooling().embedAll(null, inputs);
        assertEquals(INPUTS, spread.length);
        for (int i = 0; i < INPUTS; i++) {
            assertArrayEquals(alone[i], spread[i], 1e-5f, "input " + i + " answers as on one replica");
        }
        for (int r = 0; r < before.length; r++) {
            assertTrue(replicas.get(r).stats().promptTokens() > before[r],
                    "replica " + r + " prefilled a share of the inputs");
        }
    }

    @Test
    void aMemorylessEncoderPoolAnswersItsBatchInOrder() throws IOException {
        int[][] inputs = inputs();
        float[][] spread = encoderPool.pooling().embedAll(null, inputs);
        float[][] alone = encoderSingle.pooling().embedAll(null, inputs);
        for (int i = 0; i < INPUTS; i++) {
            assertArrayEquals(alone[i], spread[i], 1e-5f, "input " + i + " answers as on one replica");
            assertArrayEquals(encoderSingle.pooling().embed(null, inputs[i]), spread[i], 1e-5f,
                    "input " + i + " answers as its own request");
        }
    }
}
