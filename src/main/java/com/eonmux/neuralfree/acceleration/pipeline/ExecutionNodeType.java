package com.eonmux.neuralfree.acceleration.pipeline;

/** Where an {@link ExecutionNode} runs: a host transfer in either direction, a device kernel, or an ordering point. */
public enum ExecutionNodeType {
    HOST_TO_DEVICE_TRANSFER,
    DEVICE_TO_HOST_TRANSFER,
    KERNEL_DISPATCH,
    BARRIER
}
