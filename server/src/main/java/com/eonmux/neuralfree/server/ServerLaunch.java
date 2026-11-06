package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.SecretProperties;
import com.eonmux.neuralfree.cli.DeviceListing;
import io.quarkus.runtime.Quarkus;
import io.quarkus.runtime.annotations.QuarkusMain;

/**
 * The server launch command (llama-server's and {@code vllm serve}'s flags): {@link ServerLaunchArgs} maps the
 * flags to the {@code neuralfree.*} and {@code quarkus.*} system properties, then the Quarkus application starts
 * under them. The secret settings the {@code neuralfree serve} relaunch forwards in the file
 * {@link SecretProperties#FILE_ENV} names are loaded first and the file deleted. {@code --help} prints the flags,
 * {@code --list-devices} prints the devices the engine enumerates and exits, and a refused argument names the flag
 * on stderr and exits 1.
 */
@QuarkusMain
public final class ServerLaunch {

    private ServerLaunch() {
    }

    public static void main(String[] args) {
        try {
            SecretProperties.loadFromEnvironment();
        } catch (RuntimeException e) {
            System.err.println("neuralfree-server: " + e.getMessage());
            System.exit(1);
            return;
        }
        ServerLaunchArgs parsed;
        try {
            parsed = ServerLaunchArgs.parse(args);
        } catch (IllegalArgumentException e) {
            System.err.println("neuralfree-server: " + e.getMessage());
            System.err.println("Run with --help for the flags.");
            System.exit(1);
            return;
        }
        if (parsed.help()) {
            System.out.println(usage());
            return;
        }
        if (parsed.listDevices()) {
            DeviceListing.print(System.out);
            return;
        }
        parsed.properties().forEach(System::setProperty);
        tracingFromProperties();
        Quarkus.run();
    }

    /**
     * The OpenTelemetry runtime of a pre-set {@code -Dneuralfree.server.otlp.traces.endpoint} (the flag goes
     * through {@link ServerLaunchArgs}): the SDK ships disabled, so the endpoint turns it on and points the
     * exporter at it — unless the launcher configured the SDK itself with an explicit
     * {@code -Dquarkus.otel.sdk.disabled}, which then decides.
     */
    private static void tracingFromProperties() {
        String endpoint = System.getProperty(ServerOptions.OTLP_TRACES_ENDPOINT_PROPERTY);
        if (endpoint == null || endpoint.isBlank() || System.getProperty("quarkus.otel.sdk.disabled") != null) {
            return;
        }
        System.setProperty("quarkus.otel.sdk.disabled", "false");
        System.setProperty("quarkus.otel.exporter.otlp.traces.endpoint", endpoint.trim());
    }

    /** The {@code --help} text: every flag, the property it sets and the default it overrides. */
    static String usage() {
        return """
                neuralfree server - an OpenAI-compatible API server over a GGUF model

                Usage: java -jar neuralfree-server.jar [flags] [model]
                       (the model is the first unnamed argument, as `vllm serve <model>` takes it)

                Launch flags (each sets the property a -D launch would):
                  -m, --model PATH             model to load (neuralfree.server.model); without it the server
                                               boots unloaded and /health reports the miss
                  --host HOST                  interface to bind (quarkus.http.host, default 127.0.0.1)
                  --allowed-hosts HOSTS        trusted Host names for server tools and MCP, comma-separated;
                                               loopback and --host are always allowed; '*' disables the guard
                                               (neuralfree.server.allowed.hosts)
                  --port PORT                  port to bind (quarkus.http.port, default 8080), 1..65535
                  -dev, --device LIST          GPUs the model runs on: 0-based device indices of the backend's
                                               list or name substrings, comma-separated (see --list-devices);
                                               none runs the CPU engine, two or more split by layer
                                               (neuralfree.gpu.devices)
                  --rpc SERVERS                remote compute servers a layer split adds as devices, as
                                               host:port entries comma-separated (llama.cpp --rpc); each
                                               runs nftool rpc-server. The shared key comes from
                                               NEURALFREE_RPC_KEY, preferred over an inline host:port@key,
                                               which is visible in ps and shell history (neuralfree.rpc)
                  -sm, --split-mode MODE       multi-GPU split: none (one GPU) | layer (each GPU holds a
                                               contiguous span of layers and their KV caches, vLLM
                                               pipeline parallelism) | row (feed-forward shards joined
                                               by an all-reduce) | tensor (vLLM tensor parallelism: the
                                               attention heads, their KV caches and the feed-forward
                                               divided over the ranks) (neuralfree.split.mode); a
                                               --device list of two or more GPUs splits by layer
                  -ts, --tensor-split SHARES   positive share of each GPU, comma-separated and one per
                                               device, e.g. 3,1 (layer spans, or feed-forward shards
                                               under row); refused under tensor mode, which divides
                                               evenly, and without a layer or row split
                                               (neuralfree.tensor.split)
                  -mg, --main-gpu INDEX        index into --device of the GPU holding the embedding and
                                               the output head; with --split-mode none, the one GPU the
                                               model runs on (neuralfree.main.gpu)
                  -tp, --tensor-parallel-size N
                                               vLLM tensor parallelism over N GPUs: --split-mode tensor
                                               over the first N device indices, or over the N devices
                                               --device lists (neuralfree.split.mode,
                                               neuralfree.gpu.devices); 1 is vLLM's default
                  -pp, --pipeline-parallel-size N
                                               vLLM pipeline parallelism: a layer split over N GPUs,
                                               the first N device indices or the N devices --device
                                               lists; inexpressible with -tp above 1 or --rpc
                                               (neuralfree.split.mode, neuralfree.gpu.devices)
                  -dcp, --decode-context-parallel-size N
                                               vLLM decode context parallelism: the context split over
                                               N GPUs, the first N device indices or the N devices
                                               --device lists; the ranks shard the paged KV pools by
                                               whole pages; inexpressible with -tp or -pp above 1
                                               (neuralfree.split.mode, neuralfree.gpu.devices)
                  -pcp, --prefill-context-parallel-size N
                                               vLLM prefill context parallelism beside the context split:
                                               every prefill's query rows cut into DualChunkSwap chunk pairs
                                               over the ranks' page-table shards; needs the same rank count
                                               as -dcp (neuralfree.prefill_context_parallel_size)
                  --dcp-kv-cache-interleave-size N
                                               vLLM's dcp KV interleave: the context split interleaves
                                               whole pages, so N must equal the page size
                                               (neuralfree.kv.page.size) of the paged memory -np or
                                               neuralfree.kv.paged selects
                  --enable-expert-parallel, -ep tensor split only: each rank holds whole experts instead
                                               of a slice of every expert (vLLM's flag;
                                               neuralfree.expert.parallel)
                  --enable-eplb, --no-eplb     pack the experts of expert parallelism over the ranks by
                                               their recorded routed-row load and rearrange them from
                                               the live load while the model runs, as vLLM's
                                               --enable-eplb does; needs --enable-expert-parallel and
                                               a profile, a save path or the rearrangement
                                               (neuralfree.eplb, neuralfree.eplb.rebalance)
                  --eplb-profile PATH|off      the expert-load profile read at load, a
                                               moe-cache-ranking.bin (neuralfree.eplb.profile)
                  --eplb-save PATH|off         write the routed rows counted while the model runs to
                                               this file (neuralfree.eplb.save)
                  --eplb-policy POLICY         the expert placement policy; default is vLLM's weighted
                                               packing (neuralfree.eplb.policy)
                  --eplb-config JSON           vLLM's EPLBConfig object: policy, num_redundant_experts,
                                               window_size and step_interval map to their
                                               neuralfree.eplb.* properties; log_balancedness and
                                               log_balancedness_interval are accepted (the balance
                                               figures log at every boundary), as are use_async=false,
                                               enable_migration_batching=false and communicator=null;
                                               the async, batched-migration and communicator settings
                                               refuse by name
                  --numa TYPE                  CPU thread placement on a multi-node host: distribute |
                                               isolate | numactl (neuralfree.numa); no effect with one
                                               NUMA node
                  -lm, --load-mode MODE        tensor reads: auto | none | mmap | mlock | mmap+mlock |
                                               dio (neuralfree.tensor.read, default mmap)
                  --mlock                      pin the model's tensor pages in RAM (mmap+mlock, or
                                               mlock after --no-mmap)
                  --no-mmap                    read tensors into process buffers instead of mapping them
                  -a, --alias NAMES            comma-separated names to serve (neuralfree.server.model.name,
                                               llama.cpp --alias); default: the model path
                  --served-model-name NAMES    vLLM's spelling of --alias
                  --api-key KEY                require this bearer key; repeatable, keys join with commas
                                               (neuralfree.server.api.key)
                  --api-key-file FILE          one key per line, # comments (neuralfree.server.api.key.file)
                  --grpc                       serve the gRPC API (neuralfree.Inference and
                                               neuralfree.Control) on the HTTP host and port, with server
                                               reflection and the gRPC health service; off by default
                                               (neuralfree.server.grpc; vLLM's `vllm serve --grpc`). A
                                               configured API key is required on every gRPC call except
                                               the standard health Check
                  --dev-mode                   serve vLLM's development and RL control routes (the cache
                                               resets, /server_info, sleep, pause, abort and the weight
                                               transfer); off they answer 404
                                               (neuralfree.server.dev.mode; VLLM_SERVER_DEV_MODE=1 is
                                               vLLM's environment switch); only for trusted deployments
                  --enable-sleep-mode          vLLM's sleep-mode switch: serves the checkpoint window and
                                               lets /sleep, /wake_up and /release_kv_cache_memory release
                                               memory; off the window answers 404 and the sleep routes the
                                               sleep-mode refusal (neuralfree.server.sleep.mode)
                  --enable-elastic-ep          serve /scale_elastic_ep and /is_scaling_elastic_ep, which
                                               resize the replica pool at runtime; off both answer 404
                                               (neuralfree.server.elastic.ep)
                  --enable-fault-tolerance     serve /v1/fault_tolerance/status and /apply (retry and
                                               scale_down over the replica pool); off both answer 404
                                               (neuralfree.server.fault.tolerance)
                  --profiler-config JSON       vLLM's ProfilerConfig object naming the profiler kind and its
                                               trace directory, e.g. '{"profiler": "torch",
                                               "torch_profiler_dir": "/tmp/profile"}'; 'torch', the JFR
                                               window's counterpart, is accepted and vLLM's 'cuda' and
                                               'proton' refuse by name; without a profiler
                                               /start_profile and /stop_profile answer 404
                                               (neuralfree.server.profiler.config)
                  -c, --ctx-size N             context length; 0 lets the fitter pick (neuralfree.ctx)
                  --max-model-len N            vLLM's spelling of --ctx-size
                  -np, --parallel N            sequence slots the engine holds, 1..256 (neuralfree.parallel)
                  --pooling TYPE               pooling override: none | mean | cls | last | rank (llama.cpp
                                               --pooling; neuralfree.pooling); default: the GGUF's pooling_type;
                                               rank refuses a model without a classifier head
                  --override-pooler-config JSON
                                               vLLM's spelling: a JSON object whose pooling_type (or
                                               seq_pooling_type / tok_pooling_type) names the same override
                                               ({"pooling_type": "MEAN"}; ALL pools per token); any other pooler
                                               key refuses. --pooler-config is the same
                  --chat-template JINJA        template source, built-in name (chatml, llama3, gemma, ...) or
                                               template file (neuralfree.chat.template)
                  --chat-template-file FILE    template file, UTF-8, one of the built-in names also accepted
                                               (neuralfree.chat.template.file); refuses --chat-template
                  --chat-template-kwargs JSON  default template variables, a JSON object such as
                                               '{"enable_thinking": false}'; a request's own variables win;
                                               repeatable, merged (neuralfree.chat.template.kwargs)
                  --default-chat-template-kwargs JSON
                                               vLLM's spelling of --chat-template-kwargs
                  --trust-request-chat-template
                                               let requests supply their own chat_template (vLLM's flag;
                                               neuralfree.server.trust.request.chat.template); without it
                                               such a request is refused with 400
                  --chat-template-content-format auto|string|openai
                                               message content as the template reads it: detected (auto),
                                               always a string, or always typed parts (vLLM's flag;
                                               neuralfree.chat.template.content.format)
                  --reasoning-parser NAME      reasoning delimiters by vLLM parser name: auto, none or a
                                               registered parser (neuralfree.reasoning.parser)
                  --tool-call-parser NAME      tool-call syntax by vLLM parser name: auto, none or a
                                               registered parser, including autoparser for the template's
                                               own syntax (neuralfree.tool.parser)
                  --enable-auto-tool-choice    vLLM's flag; requires --tool-call-parser
                  --logits-processors CLASS... custom logits processor factory classes to load beside the
                                               plugin providers; requests configure them with vllm_xargs
                                               (neuralfree.logits.processors)
                  --reasoning-preserve         keep the reasoning of earlier assistant turns in the prompt
                                               for templates that support it: preserve_reasoning defaults
                                               to true, as without either flag
                                               (neuralfree.chat.reasoning.preserve)
                  --no-reasoning-preserve      preserve_reasoning defaults to false: earlier reasoning is
                                               dropped from the prompt
                  --logprobs-mode MODE         what OpenAI logprobs score: raw_logprobs (default),
                                               raw_logits, processed_logprobs or processed_logits (vLLM's
                                               flag; neuralfree.server.logprobs.mode)
                  --watermark-config JSON      vLLM's text watermark for generated tokens: a JSON object with
                                               algorithm (gumbel | dual_key_gumbel), alpha, context_width,
                                               deduplicate_contexts (none | single_turn | all),
                                               deduplicate_contexts_max_history, prf and
                                               allow_target_only_watermarking; the key (unsigned 64-bit) comes
                                               from NEURALFREE_WATERMARK_KEY or the file
                                               NEURALFREE_WATERMARK_KEY_FILE names, or from the JSON "key"
                                               (neuralfree.watermark.config); requests switch it off with
                                               "watermarking": false, and /v1/watermark/detect scores text
                                               under the same key
                  --otlp-traces-endpoint ENDPOINT
                                               export one OpenTelemetry span per generation request, with
                                               its queue, time-to-first-token and token-count attributes,
                                               to this OTLP endpoint (vLLM's flag;
                                               neuralfree.server.otlp.traces.endpoint); turning it on also
                                               enables the OpenTelemetry SDK, and
                                               OTEL_EXPORTER_OTLP_TRACES_PROTOCOL picks the protocol,
                                               grpc (default) or http/protobuf; requests carrying W3C
                                               traceparent headers join the caller's trace
                  --lora PATH                 LoRA adapter to load; repeatable (neuralfree.lora)
                  --lora-scaled PATH SCALE     LoRA adapter at a scale; repeatable (neuralfree.lora path:scale)
                  --lora-modules NAME=PATH...  per-request LoRA adapters loaded at startup, selected by a
                                               request's model or lora list; also vLLM's JSON
                                               {"name", "path"} (neuralfree.server.lora.modules)
                  --max-loras N                per-request adapters resident and running at once
                                               (neuralfree.lora.max)
                  --max-lora-rank N            highest adapter rank a load accepts (neuralfree.lora.max.rank)
                  --max-cpu-loras N            unloaded adapters the host cache keeps (neuralfree.lora.cache)
                  --enable-lora                per-request adapters on a GPU, reserving the device memory
                                               of --max-loras slots of --max-lora-rank at load
                                               (neuralfree.lora.enable; implied by --lora, --lora-modules
                                               and runtime adapter updating); the CPU serves them either way
                  --ssl-key-file FILE          TLS private key, PEM (quarkus.http.ssl.certificate.key-files)
                  --ssl-keyfile FILE           vLLM's spelling of --ssl-key-file
                  --ssl-cert-file FILE         TLS certificate chain, PEM; needs --ssl-key-file
                                               (quarkus.http.ssl.certificate.files)
                  --ssl-certfile FILE          vLLM's spelling of --ssl-cert-file
                  --slot-save-path DIR         turn the slot save, restore and erase endpoints on, files
                                               under DIR (neuralfree.server.slot.save.path)
                  -nkvo, --no-kv-offload       Keep every layer's KV cache in host RAM: its projections and FFN
                                               run on the device, its attention on the host over that cache
                                               (neuralfree.kv.offload=false)
                  -kvo, --kv-offload           Keep the KV caches on the device (default; neuralfree.kv.offload)
                  -cram, --cache-ram N         MiB of host RAM for prefix-cache blocks evicted from the KV
                                               pools; 0 off (default), -1 no limit (neuralfree.cache.ram)
                  -cdisk, --cache-disk N       MiB of disk for prefix-cache blocks the host tier drops, one file
                                               per block under --cache-disk-dir (vLLM's tiering filesystem
                                               tier); 0 off (default), -1 no limit; needs --cache-ram
                                               (neuralfree.cache.disk.mib)
                  --cache-disk-dir DIR         directory of the disk tier's block files, required with
                                               --cache-disk (neuralfree.cache.disk.dir)
                  -mm, --mmproj FILE           projector GGUF that encodes image, video and audio content
                                               parts (neuralfree.server.mmproj); --no-mmproj drops it
                  -mmu, --mmproj-url URL       projector downloaded once into the hub cache at startup,
                                               http or https (neuralfree.server.mmproj.url)
                  --mmproj-auto (default)      when no projector was given, take the mmproj GGUF beside
                                               the model file (neuralfree.server.mmproj.auto);
                                               --no-mmproj-auto turns the search off
                  --mmproj-offload, --no-mmproj-offload
                                               run the vision encoder on a GPU, or on the CPU; default on,
                                               following the engine's backend (neuralfree.mmproj.offload)
                  -mmdev, --mmproj-device DEV  GPU device of the vision encoder (index or name substring);
                                               none keeps it on the CPU (neuralfree.mmproj.device)
                  --mmproj-devices DEVS        comma-separated devices of the encoder's rank group, one rank
                                               each (neuralfree.mmproj.devices)
                  --mm-encoder-tp-mode MODE    how the encoder spreads over the rank group: weights (the
                                               encoder whole on one device, the default) or data (the full
                                               weights on every rank, the images spread over them)
                                               (neuralfree.mmproj.tp.mode)
                  --ec-transfer-config JSON    share the encoder output of every media item with the other
                                               servers of the deployment, vLLM's form, e.g. {"ec_connector":
                                               "ECExampleConnector", "ec_role": "ec_producer",
                                               "ec_connector_extra_config": {"shared_storage_path": "/shared"}}
                  --media-path DIR             directory file:// media URLs may read from; without it they
                                               are refused (neuralfree.server.media.path)
                  --allowed-local-media-path DIR  vLLM's spelling of --media-path
                  --allowed-media-domains HOSTS   comma-separated hosts http(s) media URLs may name; default
                                               any host (neuralfree.server.media.domains)
                  --limit-mm-per-prompt JSON   most items per prompt, e.g. {"image": 2, "audio": 1}; 999 each
                                               by default, 0 turns a modality off (neuralfree.server.media.limits)
                  --image-min-tokens N         smallest image token count of dynamic-resolution projectors
                                               (neuralfree.server.image.min.tokens)
                  --image-max-tokens N         largest image token count of dynamic-resolution projectors
                                               (neuralfree.server.image.max.tokens)
                  --mm-processor-cache-gb GIB  budget of each media cache layer (encodings, decoded and
                                               preprocessed media); a media part's uuid keys its entries;
                                               default 4, at most an eighth of the heap; 0 turns it off
                                               (neuralfree.server.media.cache.gb)
                  --media-io-kwargs JSON       media loader options by modality, vLLM's form, e.g.
                                               {"image": {"image_mode": "L", "rgba_background_color": [0, 0, 0]}};
                                               the video loader takes num_frames, fps and max_duration; the
                                               options key the cached items too
                                               (neuralfree.server.media.io.kwargs)
                  --video-fps N                frames per second video frames sample at when neither the model
                                               nor the media loader options name a rate; 0 keeps the source
                                               rate (neuralfree.server.video.fps)
                  --video-timestamp-interval MS  interval between the default video template's text timestamps;
                                               0 stamps none (neuralfree.server.video.timestamp.interval)
                  --video-ffmpeg-dir DIR       directory of the ffmpeg and ffprobe binaries; the PATH by
                                               default (neuralfree.server.video.ffmpeg.dir)
                  --video-pruning-rate Q       share of each video's embedding rows video token pruning
                                               drops, in [0, 1); 0 prunes nothing
                                               (neuralfree.server.video.pruning.rate)
                  --video-pruning-method M     pruning method of a positive rate: evs (default) or vidcom2
                                               (neuralfree.server.video.pruning.method)
                  -dp, --data-parallel-size N  data-parallel size: N replicas of the model behind this server
                                               (neuralfree.server.replicas); with a rank or a local size, the
                                               deployment's global size (neuralfree.server.dp.size)
                  --dp-size N                  SGLang's spelling of --data-parallel-size
                  -dpn, --data-parallel-rank R serve rank R alone behind an external load balancer (vLLM's
                                               external-LB mode; neuralfree.server.dp.rank), R < size
                  -dpe, --data-parallel-external-lb
                                               external-LB mode; needs --data-parallel-rank
                  -dpl, --data-parallel-size-local N
                                               serve N consecutive ranks here (hybrid mode with a larger size)
                  -dpr, --data-parallel-start-rank S
                                               the first rank of --data-parallel-size-local, default 0
                  -dph, --data-parallel-hybrid-lb
                                               hybrid mode; needs --data-parallel-size-local
                  --data-parallel-devices LIST comma-separated device names, one replica each
                                               (neuralfree.server.devices); --dp-devices is the same
                  --load-balance-method NAME   shortest_queue (default), round_robin or minimum_tokens
                                               (SGLang's; also auto, total_requests, total_tokens;
                                               neuralfree.server.load.balance)
                  --kv-events-config JSON      vLLM's KV cache event config (neuralfree.kv.events):
                                               {"enable_kv_cache_events": true} publishes BlockStored,
                                               BlockRemoved and AllBlocksCleared on the ZMQ transport
                                               (tcp://*:5557), as vLLM does; "publisher" takes zmq, log,
                                               null or a KvEventPublisher class name, and "endpoint",
                                               "replay_endpoint", "buffer_steps", "hwm", "max_queue_size"
                                               and "topic" shape the ZMQ transport
                  --kv-transfer-config JSON    disaggregated prefill/decode (vLLM's flag;
                                               neuralfree.server.kv.transfer.config): a KVTransferConfig
                                               object naming kv_connector (TcpKvConnector, a socket
                                               pull, or FileKvConnector, shared storage), kv_role
                                               (kv_producer, kv_consumer, kv_both), engine_id, kv_ip,
                                               kv_port, kv_rank, kv_parallel_size, kv_buffer_size,
                                               kv_load_failure_policy (fail | recompute) and the
                                               connector's kv_connector_extra_config (auth_token, also
                                               from NEURALFREE_KV_TRANSFER_TOKEN[_FILE]; a producer on a
                                               non-loopback kv_ip refuses to start without one, and
                                               allowed_remote_hosts bounds the peers a decode side pulls
                                               from, loopback plus this instance's kv_ip by default);
                                               requests then carry
                                               kv_transfer_params on /v1/completions and /v1/chat/
                                               completions. NIXL, Mooncake, LMCache and MultiConnector
                                               refuse by name (native bindings this build does not ship)
                  --prefix-caching-hash-algo NAME
                                               the block-hash algorithm of the prefix cache (vLLM's
                                               --prefix-caching-hash-algo; neuralfree.prefix.hash.algo):
                                               sha256 (default, over vLLM's pickle serialization),
                                               sha256_cbor (over canonical CBOR, the cross-language form),
                                               xxhash and xxhash_cbor; the published BlockStored hashes
                                               equal vLLM's per algorithm
                  --scheduling-policy NAME     fcfs (default): requests in arrival order, priority 0 only;
                                               priority: the lowest request "priority" first, ties by
                                               arrival, and memory pressure preempts the highest value
                                               (neuralfree.scheduling.policy)
                  --ctx-checkpoints N, --swa-checkpoints N, -ctxcp N
                                               context checkpoints the batch engine's prefix cache keeps per
                                               sequence slot for a memory whose prefix cannot be reused from
                                               KV pages alone (recurrent state, sliding-window groups),
                                               default 32 (neuralfree.ctx.checkpoints); 0 turns prefix reuse
                                               off for such a memory
                  --checkpoint-min-step N, -cms N
                                               token spacing of the checkpoints a long prompt's prefill takes
                                               on its way, default 8192 (neuralfree.checkpoint.min.step);
                                               0 sets no minimum
                  --spec-type T[,T..]          speculative decoding drafters, repeatable, appends
                                               (neuralfree.draft.type): ngram-simple, ngram-map-k,
                                               ngram-map-k4v, ngram-mod, ngram-cache, suffix (per-sequence
                                               state each), draft-simple / auto (the -md draft model); several
                                               chain, the lookups first; draft-mtp is refused
                  -md, --model-draft FILE      draft model GGUF drafting every running sequence in one batch,
                                               its KV in the paged memory (neuralfree.draft); MTP and EAGLE-3
                                               draft files are refused at load
                  -ctkd / -ctvd TYPE           draft KV type of K / V (neuralfree.draft.kv.typeK / typeV)
                  -ubd, --spec-draft-n-ubatch N  draft forward ubatch, 0 = the target's (neuralfree.draft.ubatch)
                  -devd, --spec-draft-device LIST  GPUs the draft model loads on: 0-based device indices or
                                               name substrings (see --list-devices); none keeps it on the host
                                               (neuralfree.draft.gpu.devices)
                  -ngld, --spec-draft-ngl N    draft layers on the device: an exact count, auto or all
                                               (neuralfree.draft.n.gpu.layers)
                  -otd, --spec-draft-override-tensor RULES  draft tensor placement, pattern=cpu|gpu,...
                                               (neuralfree.draft.override.tensor)
                  --spec-draft-p-min P         stop drafting below this draft probability (neuralfree.draft.p.min)
                  --spec-draft-sample-method M greedy (default) or probabilistic: the draft model samples its
                                               tokens at the request's temperature (vLLM draft_sample_method;
                                               neuralfree.draft.sample.method)
                  --[no-]spec-draft-heterogeneous-vocab
                                               draft across a draft model of a different vocabulary by the
                                               token-level intersection of the two tokenizers (vLLM
                                               use_heterogeneous_vocab; greedy drafting only;
                                               neuralfree.draft.heterogeneous.vocab)
                  --spec-draft-n-max N         most drafted tokens per round, default 3 (neuralfree.draft.max);
                                               --draft-max is the same
                  --spec-draft-n-min N         shortest draft verified, default 0 (neuralfree.draft.min);
                                               --draft-min is the same
                  --spec-ngram-simple-size-n N lookup n-gram of ngram-simple, default 12, 2..64
                                               (neuralfree.draft.ngram.simple.size.n)
                  --spec-ngram-simple-size-n-min N
                                               smallest lookup n-gram, 1..64; below -size-n the drafter takes
                                               the longest match in the range, as vLLM's prompt_lookup_min /
                                               prompt_lookup_max (neuralfree.draft.ngram.simple.size.n.min);
                                               default -size-n. --ngram-prompt-lookup-min / -max are the same
                  --spec-ngram-simple-size-m N draft m-gram of ngram-simple, default 48, 1..128
                                               (neuralfree.draft.ngram.simple.size.m)
                  --spec-ngram-map-k-size-n/-size-m/-min-hits N   ngram-map-k key n-gram, value m-gram, key hits
                                               (12 / 48 / 1); the same for --spec-ngram-map-k4v-*
                  --spec-ngram-mod-n-match/-n-max/-n-min N   ngram-mod hashed n-gram, longest and shortest
                                               draft (24 / 64 / 48)
                  -lcs / -lcd FILE             ngram-cache static (read only) / dynamic (written on shutdown
                                               with every sequence's n-grams) cache files
                  --spec-suffix-max-tree-depth N, --spec-suffix-max-spec-factor F,
                  --spec-suffix-max-spec-offset F, --spec-suffix-min-token-prob P,
                  --spec-suffix-max-cached-requests N   suffix decoding limits
                  --spec-verify M              match (default) | rejection | block | fly | synthetic
                                               (neuralfree.draft.verify); --spec-draft-sampling
                                               greedy|probabilistic = match|rejection; per-request sampling
                                               parameters apply; a request whose chain ends in adaptive_p is
                                               refused beside a verify mode other than match (mirostat runs
                                               in every mode)
                  --spec-fly-window N, --spec-fly-entropy-threshold F, --spec-fly-entropy-top-k N   fly
                                               verification settings
                  --spec-synth-len L | --spec-synth-rates P0,P1,..   synthetic acceptance for benchmarking
                                               (needs -Dneuralfree.spec.synthetic=true)
                  --speculative-config JSON    vLLM's speculative config, each key applied as its flag:
                                               {"method": "ngram", "num_speculative_tokens": 5,
                                               "prompt_lookup_max": 4, "prompt_lookup_min": 2}; keys of what
                                               the server does not run (the MTP route draft-mtp) and unknown
                                               keys refuse by name
                  --jinja                      render prompts through the model's Jinja template (the
                                               default, llama.cpp's --jinja)
                  --no-jinja                   render llama.cpp's non-Jinja route instead: the model's
                                               template is replaced by the built-in template its text
                                               detects (src/llama-chat.cpp), a template naming none is
                                               refused (neuralfree.chat.jinja=false)
                  --metrics                    accepted: the Prometheus endpoint at /metrics is always served
                  --tools TOOL1,TOOL2,...      enable the built-in server tools the model and POST /tools can
                                               run: read_file, file_glob_search, grep_search,
                                               exec_shell_command, write_file, edit_file, get_info, or
                                               "all" (llama.cpp's flag; neuralfree.server.tools); only in
                                               trusted environments
                  --mcp-servers-config FILE    Cursor-compatible {"mcpServers": {...}} JSON of stdio MCP tool
                                               servers, their tools exposed as <server>_<tool> (llama.cpp's
                                               flag; neuralfree.server.mcp.config); only in trusted
                                               environments
                  --mcp-servers-json JSON      the same document inline (llama.cpp's flag;
                                               neuralfree.server.mcp.json)
                  --tools-runtime SPEC         refused: this build runs the tools in the server process, not
                                               in a container or over ssh
                  -ag, --agent                 enable every built-in server tool (llama.cpp's --agent)
                  --ui, --webui                serve the bundled chat UI at / (default on)
                  --no-ui, --no-webui          turn the UI off: the UI routes answer 404 and /, /index.html and
                                                /ui/* need the API key like every endpoint
                                                (neuralfree.server.webui)
                  --path DIR                   serve the UI assets from DIR instead of the bundle (llama.cpp
                                                --path; neuralfree.server.ui.path); files outside DIR are
                                                refused, the caching and ETags are the bundle's
                  --ui-config JSON             the UI's default settings as JSON; the UI applies them under the
                                                settings its users store in the browser, and /props reports
                                                them as ui_settings (--webui-config is the same;
                                                neuralfree.server.ui.config)
                  --ui-config-file FILE        read --ui-config from FILE; bad JSON fails the launch by name
                  --api-prefix PREFIX          serve every route (API, UI, probes, /metrics) under PREFIX as
                                                llama.cpp does (neuralfree.server.api.prefix); bare paths
                                                answer 404, and a prefix is simple path segments such as /api
                  --cors-origins ORIGINS       comma-separated origins browsers may call from, or * (the
                                                default, which echoes the Origin back so credentials stay
                                                usable); the special value localhost allows localhost origins
                                                only (quarkus.http.cors.*)
                  --allowed-origins JSON       vLLM's spelling: a JSON list of origins, e.g.
                                                '["http://ui.example"]'
                  --cors-methods METHODS       preflight-allowed methods, default GET, POST, DELETE, OPTIONS
                  --allowed-methods JSON       vLLM's spelling of --cors-methods: a JSON list of methods
                  --cors-headers HEADERS       preflight-allowed headers, default *
                  --allowed-headers JSON       vLLM's spelling of --cors-headers: a JSON list of headers
                  --cors-credentials           allow credentialed browser calls, on by default as in llama.cpp;
                                                --no-cors-credentials turns it off
                  --allow-credentials          vLLM's spelling of --cors-credentials
                  --reuse-port                 allow several sockets to bind the same port
                                                (quarkus.http.so-reuse-port)
                  --list-devices               print the devices OpenCL, CUDA, Vulkan, HIP and Metal
                                               enumerate, plus those of every installed accelerator
                                               backend, each with its 0-based device index, and exit
                  -h, --help                   this text

                Engine tuning beyond these flags travels through its neuralfree.* properties, as in any
                launch: -Dneuralfree.kv.typeK=q8_0 and friends (see the EngineConfig documentation).

                Examples:
                  java -jar neuralfree-server.jar -m model.gguf --port 8080 -np 4 -c 32768
                  NEURALFREE_API_KEY="$(cat /run/secrets/nf-api-key)" \\
                    java -jar neuralfree-server.jar model.gguf --alias qwen3
                  java -jar neuralfree-server.jar -m model.gguf --ssl-cert-file cert.pem --ssl-key-file key.pem
                  java -jar neuralfree-server.jar -m model.gguf --data-parallel-devices "RTX 5080,Radeon"
                  java -jar neuralfree-server.jar -m model.gguf -dp 4 --data-parallel-rank 2 --port 8002
                """;
    }
}
