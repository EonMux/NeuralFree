package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@code bench-throughput}: the whole workload leaves at once, the dataset flags carry over from
 * {@code bench-serve}, and the traffic flags that shape a run belong to {@code bench-serve}.
 */
public class BenchThroughputCommandTest {

    private FakeOpenAiServer server;

    @Before
    public void start() throws IOException {
        server = new FakeOpenAiServer();
    }

    @After
    public void stop() {
        server.close();
    }

    @Test
    public void theTrafficFlagsAreRefusedByName() {
        for (String flag : new String[]{"--request-rate", "--request_rate", "--burstiness",
                "--max-concurrency", "--ramp-up-strategy", "--num-warmups", "--goodput"}) {
            try {
                BenchThroughputCommand.refuseTrafficFlags(new String[]{flag, "1"});
                fail(flag + " belongs to bench-serve");
            } catch (ServeBenchCommand.UsageException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().startsWith(
                        flag.replace('_', '-')));
                assertTrue(expected.getMessage().contains("bench-serve"));
            }
        }
    }

    @Test
    public void aRunSendsTheWholeWorkloadAtOnce() throws IOException {
        server.tokensPerResponse = 2;
        ServeBenchOptions opts = ServeBenchOptions.builder()
                .baseUrl(server.baseUrl())
                .model("served-model")
                .datasetName("random").numPrompts(3).randomInputLen(3).randomOutputLen(2)
                .percentileMetrics("ttft,e2el").metricPercentiles("50")
                .build();
        Map<String, Object> result = BenchThroughputCommand.executeAndReport(opts);
        assertEquals(3, result.get("completed"));
        assertEquals(0, result.get("failed"));
        assertEquals(6L, result.get("total_output_tokens"));
        assertTrue((Double) result.get("request_throughput") > 0.0);
    }
}
