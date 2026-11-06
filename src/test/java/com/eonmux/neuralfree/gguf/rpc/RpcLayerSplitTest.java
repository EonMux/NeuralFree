package com.eonmux.neuralfree.gguf.rpc;

import com.eonmux.neuralfree.Config.ACCELERATION_MODE;
import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;

/**
 * A layer split whose second stage is a remote device: the model splits over the local OpenCL device 0 and an
 * in-process {@code rpc-server} holding device 1, and the logits match the same split over both devices locally.
 * This exercises the whole path — remote device selection, weight upload over the protocol, per-layer dispatch and
 * the residual-stream hand-off to the remote stage.
 */
public class RpcLayerSplitTest {

    private static final String KEY = "layer-split-rpc-key";
    private static final float TOL = 1e-3f;
    private static final int[] PROMPT = {1, 17, 42, 5, 99, 26};
    private static final int[] DECODE = {77, 3};

    private HWAcceleration acceleration;
    private RpcComputeServer server;
    private Path model;

    @Before
    public void setUp() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        acceleration = HWAcceleration.forMode(ACCELERATION_MODE.OPENCL);
        List<CLDevice> devices = new ArrayList<>(acceleration.getCLDevices());
        Assume.assumeTrue("the split needs two OpenCL devices, found " + devices.size(), devices.size() >= 2);
        ComputeBackend served = acceleration.createBackend(devices.get(1));
        server = new RpcComputeServer(served, KEY, "127.0.0.1", 0);
        server.start();
        model = SyntheticModels.llama().writeTempFile();
    }

    @After
    public void tearDown() {
        if (server != null) {
            server.close();
        }
        if (acceleration != null) {
            acceleration.shutdown();
        }
    }

    @Test
    public void remoteSecondStageMatchesTheLocalSplit() throws IOException {
        EngineConfig local = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .splitMode(EngineConfig.SplitMode.LAYER)
                .addGpuDevice("0")
                .addGpuDevice("1")
                .memory(MemoryConfig.defaults().toBuilder().nCtx(64).ubatch(4).build())
                .build();
        EngineConfig remote = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .splitMode(EngineConfig.SplitMode.LAYER)
                .addGpuDevice("0")
                .addRpcServer(RpcServerSpec.parse("127.0.0.1:" + server.port() + "@" + KEY))
                .memory(MemoryConfig.defaults().toBuilder().nCtx(64).ubatch(4).build())
                .build();

        float[][] reference = logits(local);
        float[][] got = logits(remote);
        for (int i = 0; i < reference.length; i++) {
            assertArrayEquals("step " + i + " of the remote split matches the local split", reference[i], got[i],
                    TOL);
        }
    }

    /** Prefill logits of {@link #PROMPT} and the decode logits of {@link #DECODE}. */
    private float[][] logits(EngineConfig cfg) throws IOException {
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg)) {
            ForwardBackend forward = m.forwardBackend();
            float[][] out = new float[1 + DECODE.length][];
            out[0] = forward.prefill(PROMPT, 0).clone();
            for (int i = 0; i < DECODE.length; i++) {
                out[1 + i] = forward.decode(DECODE[i], PROMPT.length + i).clone();
            }
            forward.reset();
            return out;
        }
    }
}
