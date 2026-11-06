/**
 * The Quarkus serving module of NeuralFree: the HTTP, SSE, WebSocket and
 * gRPC adapters and every protocol mapper (OpenAI, Anthropic, llama.cpp native, Ollama, SageMaker,
 * Vertex, Cohere, SGLang native, vLLM extras) over the transport-neutral core engine API
 * ({@code com.eonmux.neuralfree.gguf.engine.batch}), plus authentication, CORS, TLS, health,
 * metrics export and tracing.
 *
 * <p>Java 21 toolchain over the Java 11 core ({@code implementation project(':')}); Quarkus BOM
 * 3.40.1 pins every extension version. No Quarkus, Jakarta, Vert.x, Mutiny or gRPC type appears in
 * the core: bodies are read as bytes and parsed with the core JSON stack, and responses are written
 * with it too.
 */
package com.eonmux.neuralfree.server;
