/**
 * Disaggregated prefill and decode (vLLM's {@code vllm/distributed/kv_transfer}): the KV transfer configuration of
 * an instance ({@link KvTransferConfig}), the prefill/decode role it plays ({@link KvRole}), the per-request
 * {@code kv_transfer_params} that route a request between the two instances ({@link KvTransferParams}), and the
 * connectors that carry a sequence's cached rows from the prefill instance to the decode instance
 * ({@link KvConnector}, {@link TcpKvConnector}, {@link FileKvConnector}).
 *
 * <p>A prefill instance runs the prompt and exports the prefix-cache chain its sequence leaves
 * ({@link com.eonmux.neuralfree.gguf.engine.batch.BatchEngine#exportKv}); a decode instance imports that chain into
 * its own prefix cache ({@link com.eonmux.neuralfree.gguf.engine.batch.BatchEngine#importKv}), so its prompt adopts
 * the prefilled rows and only the last prompt token recomputes. The chain bytes are the slot file format, which
 * already carries the block layout identity, so a chain only reaches a memory of the same page size, KV group count,
 * cache dtypes and widths.
 *
 * <p>The NIXL, Mooncake and LMCache connectors of the reference engines need native RDMA/transfer-engine bindings;
 * this package ships the two transports a JVM can run on its own: a TCP pull/push connector and a shared-storage
 * connector. {@link KvConnectors} names the reference connectors when a configuration asks for them instead of
 * silently degrading.
 */
package com.eonmux.neuralfree.gguf.engine.kvtransfer;
