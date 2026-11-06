/**
 * KV cache events of the batch engine's prefix cache (vLLM's {@code vllm/distributed/kv_events.py}): the
 * {@link com.eonmux.neuralfree.gguf.engine.kvevents.KvCacheEvent} kinds, the per-step
 * {@link com.eonmux.neuralfree.gguf.engine.kvevents.KvEventBatch}, the pluggable
 * {@link com.eonmux.neuralfree.gguf.engine.kvevents.KvEventPublisher}, the broker-free publishers in
 * {@link com.eonmux.neuralfree.gguf.engine.kvevents.KvEventPublishers}, and vLLM's ZMQ transport
 * ({@link com.eonmux.neuralfree.gguf.engine.kvevents.ZmqEventPublisher}, its
 * {@link com.eonmux.neuralfree.gguf.engine.kvevents.ZmqEventPublisherConfig} and the
 * {@link com.eonmux.neuralfree.gguf.engine.kvevents.KvEventsConfig} JSON it accepts).
 */
package com.eonmux.neuralfree.gguf.engine.kvevents;
