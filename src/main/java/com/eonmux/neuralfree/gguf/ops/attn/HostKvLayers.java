package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;
import java.util.function.IntPredicate;

/**
 * The host-KV decoration of a hybrid CPU+GPU load (llama.cpp {@code -nkvo}): the device layers whose KV cache stays in
 * host RAM run their attention through a {@link HostKvMixer} over the host memory. The registry's other mixers, FFN
 * blocks and extras are shared.
 */
public final class HostKvLayers {

    private HostKvLayers() {
    }

    /**
     * {@code base} with every trunk layer {@code hostKvLayers} names bound through its host-KV mixer over
     * {@code hostContext}; {@code base} itself when no layer does.
     */
    public static GpuBlockRegistry decorate(GpuBlockRegistry base, CpuForwardContext hostContext,
                                            IntPredicate hostKvLayers) {
        Objects.requireNonNull(base, "base");
        Objects.requireNonNull(hostContext, "hostContext");
        Objects.requireNonNull(hostKvLayers, "hostKvLayers");
        return base.withMixerDecorator((layer, deviceBlock) -> TensorNames.STACK_BLK.equals(layer.stack())
                && hostKvLayers.test(layer.il())
                ? HostKvMixer.wrap(deviceBlock, hostContext) : deviceBlock);
    }
}
