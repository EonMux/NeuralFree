/**
 * Transport-neutral continuous batching over the paged memory (PARITY plan P2, WP BE-2): {@link
 * com.eonmux.neuralfree.gguf.engine.batch.BatchEngine} holds one open model, admits submitted prompts into the
 * paged memory's sequence slots while {@code PagedSequences.prepare} can reserve their pages, and pumps engine
 * steps — one prompt prefill or one decode row per running sequence — through {@code
 * ForwardBackend.forwardBatch}. Every sequence runs its own {@link
 * com.eonmux.neuralfree.gguf.engine.SequenceRun} token pipeline (sample, stop, route), so a batch step computes
 * the same texts as the same requests run one at a time through {@link
 * com.eonmux.neuralfree.gguf.engine.GenerationLoop}.
 *
 * <p>The package carries no transport types: callers submit token prompts and receive routed text pieces, and
 * the HTTP, SSE and WebSocket adapters of {@code com.eonmux.neuralfree.server} drive {@code pump} from their
 * worker threads.
 */
package com.eonmux.neuralfree.gguf.engine.batch;
