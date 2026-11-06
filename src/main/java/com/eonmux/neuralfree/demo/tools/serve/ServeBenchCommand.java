package com.eonmux.neuralfree.demo.tools.serve;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import com.eonmux.neuralfree.gguf.chat.ChatTemplateKwargs;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

/**
 * {@code nftool bench-serve}: the online serving benchmark of vLLM's {@code vllm bench serve}
 * ({@code benchmarks/serve.py}), pointed at a running OpenAI-compatible server — NeuralFree's own
 * or any other. A dataset ({@code random}, {@code random-mm}, {@code random-rerank},
 * {@code prefix_repetition}, {@code sharegpt}, {@code custom}, {@code custom_image},
 * {@code custom_audio}, {@code burstgpt}, {@code spec_bench}, {@code speed_bench},
 * {@code timed_trace}, {@code blazedit}, {@code asr} or {@code hf}) samples the workload, the
 * arrival process spaces the requests by the request rate, burstiness and ramp-up (or the trace's
 * own timestamps), the client streams every request and stamps the token events, and the report
 * gives the throughputs and the TTFT, TPOT, ITL and end-to-end latency percentiles. The server's
 * Prometheus metrics are snapshotted before and after the run, so speculative decoding's acceptance
 * and an audio run's RTFx report in the result. Results save as JSON ({@code --save-result}) under
 * the reference's name scheme, {@code --profile} brackets the measured run with the server's JFR
 * profile window, and the timeline and dataset-statistics plots are written as self-contained
 * HTML/SVG files. A non-OpenAI backend and remote code refuse by name, never silently.
 */
public final class ServeBenchCommand {

    /** The command surface: {@code --help} and every error print it. */
    static final String USAGE =
            "nftool bench-serve [options]\n"
            + "options:\n"
            + "  -h, --help                            this text\n"
            + "server:\n"
            + "  --host <h>                    server host, default 127.0.0.1\n"
            + "  --port <p>                    server port, default 8000\n"
            + "  --base-url <url>              server base url instead of --host/--port\n"
            + "  --endpoint <path>             /v1/completions (default) or /v1/chat/completions\n"
            + "  --model <id>                  the served model; default: the server's first model\n"
            + "  --served-model-name <id>      the model name the requests carry\n"
            + "  --header KEY=VALUE            a header of every request, repeatable\n"
            + "  --insecure                    trust self-signed certificates\n"
            + "dataset:\n"
            + "  --dataset-name <n>            random (default), random-mm, random-rerank, prefix_repetition,\n"
            + "                                sharegpt, custom, custom_image, custom_audio, burstgpt,\n"
            + "                                spec_bench, speed_bench, timed_trace, blazedit, asr, hf\n"
            + "  --dataset-path <path>         the dataset file or local dataset directory\n"
            + "  --num-prompts <n>             workload size, default 1000\n"
            + "  --seed <n>                    sampling seed, default 0\n"
            + "  --no-oversample               keep a file dataset's own sample count\n"
            + "  --disable-shuffle             keep the dataset file's order\n"
            + "  --input-len <n>               general input length (maps to the dataset's input flag)\n"
            + "  --output-len <n>              general output length (maps to the dataset's output flags)\n"
            + "  --random-input-len <n>        random dataset mean input, default 1024\n"
            + "  --random-output-len <n>       random dataset mean output, default 128\n"
            + "  --random-range-ratio <r|json> length spread in [0,1), a float or {\"input\":..,\"output\":..}, default 0.0\n"
            + "  --random-prefix-len <n>       shared prefix tokens, default 0\n"
            + "  --random-batch-size <n>       batch size of the synthetic datasets, default 1\n"
            + "  --no-reranker                 the rerank model scores pairs natively\n"
            + "  --random-mm-base-items-per-request <n>    random-mm items per request, default 1\n"
            + "  --random-mm-num-mm-items-range-ratio <r>  random-mm item count spread, default 0.0\n"
            + "  --random-mm-limit-mm-per-prompt <json>    per-modality item caps\n"
            + "  --random-mm-bucket-config <config>        \'{\"(h, w, frames)\": probability, ...}\'\n"
            + "  --prefix-repetition-prefix-len <n>    shared prefix tokens, default 256\n"
            + "  --prefix-repetition-suffix-len <n>    per-request suffix tokens, default 256\n"
            + "  --prefix-repetition-num-prefixes <n>  distinct prefixes, default 10\n"
            + "  --prefix-repetition-output-len <n>    output tokens, default 128\n"
            + "  --sharegpt-output-len <n>     fixed output length (default: the second turn's length)\n"
            + "  --custom-output-len <n>       fixed output length, default 256 (-1: the record's output_tokens)\n"
            + "  --spec-bench-output-len <n>   default 256;  --spec-bench-category <c>\n"
            + "  --speed-bench-dataset-subset <s>  the SPEED-Bench subset file, default qualitative\n"
            + "  --speed-bench-output-len <n>  default 4096;  --speed-bench-category <c>\n"
            + "  --timed-trace-chunk-hash-size <n>      tokens per hash, default 16\n"
            + "  --timed-trace-sec-multiplier <f>       timestamp to seconds, default 1\n"
            + "  --timed-trace-label-timestamp <name>   default timestamp\n"
            + "  --timed-trace-label-input-length <name>  default input_length\n"
            + "  --timed-trace-label-output-length <name> default output_length\n"
            + "  --timed-trace-label-hash-ids <name>    default hash_ids\n"
            + "  --self-timed                  the trace timestamps schedule the requests\n"
            + "  --blazedit-min-distance <f>   default 0.0;  --blazedit-max-distance <f>  default 1.0\n"
            + "  --asr-min-audio-len-sec <f>   default 0.0;  --asr-max-audio-len-sec <f>  default inf\n"
            + "  --hf-subset <s>  --hf-split <s>  --hf-name <id>  --hf-output-len <n>\n"
            + "  --no-stream                   load the Hub dataset completely instead of streaming\n"
            + "  --enable-multimodal-chat      wrap a multimodal dataset's media in a chat message\n"
            + "  --skip-chat-template          send raw prompts where the reference renders a template\n"
            + "  --chat-template-kwargs <json> variables bound into every client-side template render\n"
            + "traffic:\n"
            + "  --request-rate <r>            requests per second, 'inf' sends all at once (default inf)\n"
            + "  --burstiness <b>              gamma shape of the arrivals; 1 is Poisson, inf constant gaps\n"
            + "  --max-concurrency <n>         requests in flight at once\n"
            + "  --ramp-up-strategy <s>        linear or exponential, from --ramp-up-start-rps to --ramp-up-end-rps\n"
            + "  --ramp-up-start-rps <n>       the ramp-up's first rate\n"
            + "  --ramp-up-end-rps <n>         the ramp-up's last rate\n"
            + "  --ready-check-timeout-sec <n> wait this long for the endpoint to answer the first request\n"
            + "  --num-warmups <n>             warmup requests before the measured run, default 0\n"
            + "  --probe-request-rate <r>      single-token probe requests per second, default 0 (off)\n"
            + "metrics and results:\n"
            + "  --percentile-metrics <list>   of ttft,tpot,itl,e2el,client_queue_time,\n"
            + "                                e2el_including_client_queue; default ttft,tpot,itl\n"
            + "  --metric-percentiles <list>   of numbers in [0,100], default 99\n"
            + "  --goodput KEY:VALUE           service level objectives in ms, keys ttft,tpot,e2el, repeatable\n"
            + "  --save-result                 save the result JSON (the file name prints)\n"
            + "  --append-result               append to the result JSON instead of overwriting\n"
            + "  --save-detailed               include the per-request arrays in the result JSON\n"
            + "  --result-dir <dir>            directory of the result file, default the working directory\n"
            + "  --result-filename <name>      exact result file name\n"
            + "  --metadata KEY=VALUE          a key of the result JSON, repeatable\n"
            + "  --label <l>                   the result's label, default openai\n"
            + "  --request-id-prefix <p>       the X-Request-Id prefix, default bench-<random>-\n"
            + "  --disable-tqdm                no progress lines on stderr\n"
            + "  --profile                     bracket the run with POST /start_profile and /stop_profile\n"
            + "  --plot-timeline               write <result>.timeline.html beside the result\n"
            + "  --timeline-itl-thresholds <ms,ms>  timeline colours, default 25,50\n"
            + "  --plot-dataset-stats          write <result>.dataset_stats.svg beside the result\n"
            + "sampling (of every request):\n"
            + "  --temperature <t>  --top-p <p>  --top-k <k>  --min-p <p>\n"
            + "  --frequency-penalty <f>  --presence-penalty <f>  --repetition-penalty <f>\n"
            + "  --logprobs <n>                n logprobs per token of the completions endpoint\n"
            + "  --extra-body <json>           extra fields of every request body\n"
            + "  --use-beam-search             the completions/chat requests carry use_beam_search\n"
            + "  --lora-modules <m...>         adapter names the requests address at random\n"
            + "  --lora-assignment <s>         random (default) or round-robin\n"
            + "tokenizer:\n"
            + "  --tokenizer <path>            a local GGUF file or Hugging Face directory to count with\n"
            + "  --tokenizer-mode <mode>       auto (default) or hf\n"
            + "  --skip-tokenizer-init         initialize no tokenizer (custom datasets count one token)\n"
            + "Not supported (refused by name): --backend other than openai, --trust-remote-code,\n"
            + "video buckets of --random-mm-bucket-config. Prompt lengths count through --tokenizer\n"
            + "when given, else through the server's /tokenize; token counts come from the stream's\n"
            + "usage chunk.\n";

    /** An invalid command line or an unsupported option; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    private ServeBenchCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code bench-serve}. */
    public static void run(String[] args) {
        try {
            ServeBenchOptions opts = parse(args);
            if (opts == null) {
                return;
            }
            execute(opts, System.out, System.err);
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException | IllegalStateException | IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /**
     * Parses the flags into the options; null when {@code --help} printed the usage. Unknown options
     * and unsupported reference options refuse by name; long flags accept underscores like the
     * reference replaces them.
     *
     * @throws UsageException naming the offending flag
     */
    static ServeBenchOptions parse(String[] args) throws UsageException {
        ServeBenchOptions.Builder b = ServeBenchOptions.builder();
        Map<String, String> metadata = new LinkedHashMap<>();
        Map<String, Double> goodput = new LinkedHashMap<>();
        Integer inputLen = null;
        Integer outputLen = null;
        String rampStrategy = null;
        Integer rampStart = null;
        Integer rampEnd = null;
        boolean rateGiven = false;
        String timedTimestamp = "timestamp";
        String timedInputLength = "input_length";
        String timedOutputLength = "output_length";
        String timedHashIds = "hash_ids";
        double blazeditMin = 0.0;
        double blazeditMax = 1.0;
        double asrMin = 0.0;
        double asrMax = Double.POSITIVE_INFINITY;
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (a.equals("--help") || a.equals("-h")) {
                System.out.print(USAGE);
                return null;
            }
            if (a.startsWith("--")) {
                a = a.replace('_', '-');
            }
            switch (a) {
                case "--host":
                    b.host(value(a, args, i++));
                    break;
                case "--port":
                    b.port(port(value(a, args, i++), a));
                    break;
                case "--base-url":
                    b.baseUrl(value(a, args, i++));
                    break;
                case "--endpoint":
                    b.endpoint(value(a, args, i++));
                    break;
                case "--model":
                    b.model(value(a, args, i++));
                    break;
                case "--served-model-name":
                    b.servedModelName(value(a, args, i++));
                    break;
                case "--label":
                    b.label(value(a, args, i++));
                    break;
                case "--header":
                    i = repeatable(a, args, i, pair -> header(b, pair));
                    break;
                case "--dataset-name":
                    b.datasetName(datasetName(value(a, args, i++)));
                    break;
                case "--dataset-path":
                    b.datasetPath(value(a, args, i++));
                    break;
                case "--num-prompts":
                    b.numPrompts(nonNegativeInt(value(a, args, i++), a));
                    break;
                case "--seed":
                    b.seed(longValue(value(a, args, i++), a));
                    break;
                case "--no-oversample":
                    b.noOversample(true);
                    break;
                case "--disable-shuffle":
                    b.disableShuffle(true);
                    break;
                case "--input-len":
                    inputLen = positiveInt(value(a, args, i++), a);
                    break;
                case "--output-len":
                    outputLen = positiveInt(value(a, args, i++), a);
                    break;
                case "--random-input-len":
                    b.randomInputLen(positiveInt(value(a, args, i++), a));
                    break;
                case "--random-output-len":
                    b.randomOutputLen(positiveInt(value(a, args, i++), a));
                    break;
                case "--random-range-ratio":
                    b.randomRangeRatio(ServeBenchOptions.parseRangeRatio(value(a, args, i++)));
                    break;
                case "--random-prefix-len":
                    b.randomPrefixLen(nonNegativeInt(value(a, args, i++), a));
                    break;
                case "--prefix-repetition-prefix-len":
                    b.prefixRepetitionPrefixLen(positiveInt(value(a, args, i++), a));
                    break;
                case "--prefix-repetition-suffix-len":
                    b.prefixRepetitionSuffixLen(positiveInt(value(a, args, i++), a));
                    break;
                case "--prefix-repetition-num-prefixes":
                    b.prefixRepetitionNumPrefixes(positiveInt(value(a, args, i++), a));
                    break;
                case "--prefix-repetition-output-len":
                    b.prefixRepetitionOutputLen(positiveInt(value(a, args, i++), a));
                    break;
                case "--sharegpt-output-len":
                    b.sharegptOutputLen(positiveInt(value(a, args, i++), a));
                    break;
                case "--custom-output-len":
                    b.customOutputLen(intAtLeast(value(a, args, i++), -1, a));
                    break;
                case "--random-batch-size":
                    b.randomBatchSize(positiveInt(value(a, args, i++), a));
                    break;
                case "--no-reranker":
                    b.noReranker(true);
                    break;
                case "--random-mm-base-items-per-request":
                    b.randomMmBaseItemsPerRequest(nonNegativeInt(value(a, args, i++), a));
                    break;
                case "--random-mm-num-mm-items-range-ratio":
                    b.randomMmNumMmItemsRangeRatio(ratioInclusive(value(a, args, i++), a));
                    break;
                case "--random-mm-limit-mm-per-prompt":
                    b.randomMmLimitMmPerPrompt(limitPerPrompt(value(a, args, i++), a));
                    break;
                case "--random-mm-bucket-config":
                    b.randomMmBucketConfig(bucketConfig(value(a, args, i++), a));
                    break;
                case "--spec-bench-output-len":
                    b.specBenchOutputLen(positiveInt(value(a, args, i++), a));
                    break;
                case "--spec-bench-category":
                    b.specBenchCategory(value(a, args, i++));
                    break;
                case "--speed-bench-dataset-subset":
                    b.speedBenchDatasetSubset(speedBenchSubset(value(a, args, i++)));
                    break;
                case "--speed-bench-output-len":
                    b.speedBenchOutputLen(positiveInt(value(a, args, i++), a));
                    break;
                case "--speed-bench-category":
                    b.speedBenchCategory(value(a, args, i++));
                    break;
                case "--timed-trace-chunk-hash-size":
                    b.timedTraceChunkHashSize(positiveInt(value(a, args, i++), a));
                    break;
                case "--timed-trace-sec-multiplier":
                    b.timedTraceSecMultiplier(positiveDouble(value(a, args, i++), a));
                    break;
                case "--timed-trace-label-timestamp":
                    timedTimestamp = value(a, args, i++);
                    break;
                case "--timed-trace-label-input-length":
                    timedInputLength = value(a, args, i++);
                    break;
                case "--timed-trace-label-output-length":
                    timedOutputLength = value(a, args, i++);
                    break;
                case "--timed-trace-label-hash-ids":
                    timedHashIds = value(a, args, i++);
                    break;
                case "--self-timed":
                    b.selfTimed(true);
                    break;
                case "--no-self-timed":
                    b.selfTimed(false);
                    break;
                case "--blazedit-min-distance":
                    blazeditMin = doubleValue(value(a, args, i++), a);
                    break;
                case "--blazedit-max-distance":
                    blazeditMax = doubleValue(value(a, args, i++), a);
                    break;
                case "--asr-min-audio-len-sec":
                    asrMin = nonNegativeDouble(value(a, args, i++), a);
                    break;
                case "--asr-max-audio-len-sec":
                    asrMax = nonNegativeDouble(value(a, args, i++), a);
                    break;
                case "--hf-subset":
                    b.hfSubset(value(a, args, i++));
                    break;
                case "--hf-split":
                    b.hfSplit(value(a, args, i++));
                    break;
                case "--hf-name":
                    b.hfName(value(a, args, i++));
                    break;
                case "--hf-output-len":
                    b.hfOutputLen(positiveInt(value(a, args, i++), a));
                    break;
                case "--enable-multimodal-chat":
                    b.enableMultimodalChat(true);
                    break;
                case "--probe-request-rate":
                    b.probeRequestRate(positiveDouble(value(a, args, i++), a));
                    break;
                case "--use-beam-search":
                    b.useBeamSearch(true);
                    break;
                case "--tokenizer":
                    b.tokenizerPath(value(a, args, i++));
                    break;
                case "--tokenizer-mode":
                    b.tokenizerMode(tokenizerMode(value(a, args, i++)));
                    break;
                case "--skip-tokenizer-init":
                    b.skipTokenizerInit(true);
                    break;
                case "--request-rate":
                    b.requestRate(ServeBenchOptions.parseRequestRate(value(a, args, i++)));
                    rateGiven = true;
                    break;
                case "--burstiness":
                    b.burstiness(burstiness(value(a, args, i++), a));
                    break;
                case "--max-concurrency":
                    b.maxConcurrency(positiveInt(value(a, args, i++), a));
                    break;
                case "--ramp-up-strategy":
                    rampStrategy = rampUpStrategy(value(a, args, i++));
                    break;
                case "--ramp-up-start-rps":
                    rampStart = nonNegativeInt(value(a, args, i++), a);
                    break;
                case "--ramp-up-end-rps":
                    rampEnd = nonNegativeInt(value(a, args, i++), a);
                    break;
                case "--percentile-metrics":
                    b.percentileMetrics(value(a, args, i++));
                    break;
                case "--metric-percentiles":
                    b.metricPercentiles(value(a, args, i++));
                    break;
                case "--goodput":
                    i = repeatable(a, args, i, pair -> goodput(b, pair, goodput));
                    break;
                case "--ignore-eos":
                    b.ignoreEos(true);
                    break;
                case "--num-warmups":
                    b.numWarmups(nonNegativeInt(value(a, args, i++), a));
                    break;
                case "--ready-check-timeout-sec":
                    b.readyCheckTimeoutSec(nonNegativeInt(value(a, args, i++), a));
                    break;
                case "--save-result":
                    b.saveResult(true);
                    break;
                case "--append-result":
                    b.appendResult(true);
                    break;
                case "--save-detailed":
                    b.saveDetailed(true);
                    break;
                case "--result-dir":
                    b.resultDir(value(a, args, i++));
                    break;
                case "--result-filename":
                    b.resultFilename(value(a, args, i++));
                    break;
                case "--metadata":
                    i = repeatable(a, args, i, pair -> metadata(metadata, pair));
                    break;
                case "--request-id-prefix":
                    b.requestIdPrefix(value(a, args, i++));
                    break;
                case "--disable-tqdm":
                    b.disableTqdm(true);
                    break;
                case "--temperature":
                    b.samplingParam("temperature", doubleValue(value(a, args, i++), a));
                    break;
                case "--top-p":
                    b.samplingParam("top_p", doubleValue(value(a, args, i++), a));
                    break;
                case "--top-k":
                    b.samplingParam("top_k", intAtLeast(value(a, args, i++), 1, a));
                    break;
                case "--min-p":
                    b.samplingParam("min_p", doubleValue(value(a, args, i++), a));
                    break;
                case "--frequency-penalty":
                    b.samplingParam("frequency_penalty", doubleValue(value(a, args, i++), a));
                    break;
                case "--presence-penalty":
                    b.samplingParam("presence_penalty", doubleValue(value(a, args, i++), a));
                    break;
                case "--repetition-penalty":
                    b.samplingParam("repetition_penalty", doubleValue(value(a, args, i++), a));
                    break;
                case "--logprobs":
                    b.logprobs(intAtLeast(value(a, args, i++), 0, a));
                    break;
                case "--extra-body":
                    extraBody(b, value(a, args, i++));
                    break;
                case "--lora-modules":
                    b.loraModules(values(args, i));
                    i += valueCount(args, i);
                    break;
                case "--lora-assignment":
                    b.loraAssignment(assignment(value(a, args, i++)));
                    break;
                case "--insecure":
                    b.insecure(true);
                    break;
                case "--backend":
                    String backend = value(a, args, i++);
                    if (!"openai".equals(backend)) {
                        throw new UsageException(a + " '" + backend + "' is not supported: the"
                                + " benchmark speaks the OpenAI-compatible endpoints only");
                    }
                    break;
                case "--chat-template-kwargs":
                    b.chatTemplateKwargs(chatTemplateKwargs(value(a, args, i++), a));
                    break;
                case "--skip-chat-template":
                    b.skipChatTemplate(true);
                    break;
                case "--no-stream":
                    b.noStream(true);
                    break;
                case "--profile":
                    b.profile(true);
                    break;
                case "--plot-timeline":
                    b.plotTimeline(true);
                    break;
                case "--timeline-itl-thresholds":
                    b.timelineItlThresholds(timelineThresholds(value(a, args, i++), a));
                    break;
                case "--plot-dataset-stats":
                    b.plotDatasetStats(true);
                    break;
                case "--trust-remote-code":
                    throw new UsageException(a + " is not supported: no remote code runs; the harness"
                            + " downloads the repository's data files and reads them locally");
                default:
                    if (a.startsWith("-")) {
                        throw new UsageException("unknown option " + a + " (run with --help for the options)");
                    }
                    throw new UsageException("bench-serve takes no positional argument, got '" + a + "'");
            }
        }
        b.rampUp(rampStrategy, rampStart, rampEnd);
        b.timedTraceLabels(timedTimestamp, timedInputLength, timedOutputLength, timedHashIds);
        b.blazeditDistance(blazeditMin, blazeditMax);
        b.asrAudioLen(asrMin, asrMax);
        applyGeneralLengths(b, inputLen, outputLen);
        b.metadata(metadata);
        b.goodput(goodput);
        ServeBenchOptions opts = b.build();
        validate(opts, rateGiven);
        return opts;
    }

    /**
     * The run's validation over builder-built options too (the sweep's): the ramp-up's rules, the
     * concurrency cap, the dataset-path rules.
     *
     * @param rateGiven whether {@code --request-rate} appeared, which a ramp-up forbids
     * @throws UsageException naming the offending flag
     */
    static void validate(ServeBenchOptions opts, boolean rateGiven) throws UsageException {
        if (opts.maxConcurrency() != null && opts.maxConcurrency() <= 0) {
            throw new UsageException("--max-concurrency must be greater than 0");
        }
        if (opts.numPrompts() < 0) {
            throw new UsageException("--num-prompts must be at least 0");
        }
        if (ServeBenchDataset.SYNTHETIC.contains(opts.datasetName()) && opts.datasetPath() != null) {
            throw new UsageException("cannot use the '" + opts.datasetName() + "' dataset with --dataset-path;"
                    + " the dataset file datasets are " + ServeBenchDataset.NAMES);
        }
        validateTokenizer(opts);
        if (opts.skipTokenizerInit() && !opts.chatTemplateKwargs().isEmpty()) {
            throw new UsageException("--chat-template-kwargs needs a tokenizer to render the template;"
                    + " it cannot be combined with --skip-tokenizer-init");
        }
        validateEndpoint(opts);
        String strategy = opts.rampUpStrategy();
        if (strategy != null) {
            if (rateGiven) {
                throw new UsageException("when using a ramp-up, do not specify --request-rate: the ramp-up"
                        + " controls the request rate");
            }
            if (opts.rampUpStartRps() == null || opts.rampUpEndRps() == null) {
                throw new UsageException("when using --ramp-up-strategy, both --ramp-up-start-rps and"
                        + " --ramp-up-end-rps must be specified");
            }
            if (opts.rampUpStartRps() < 0 || opts.rampUpEndRps() < 0) {
                throw new UsageException("the ramp-up start and end RPS must be non-negative");
            }
            if (opts.rampUpStartRps() > opts.rampUpEndRps()) {
                throw new UsageException("the ramp-up start RPS must be less than or equal to the end RPS");
            }
            if ("exponential".equals(strategy) && opts.rampUpStartRps() == 0) {
                throw new UsageException("for an exponential ramp-up, the start RPS cannot be 0");
            }
        }
        if (opts.selfTimed() && !"timed_trace".equals(opts.datasetName())) {
            throw new UsageException("--self-timed needs the timed_trace dataset");
        }
        if (opts.asrMinAudioLenSec() > opts.asrMaxAudioLenSec()) {
            throw new UsageException("--asr-min-audio-len-sec cannot exceed --asr-max-audio-len-sec");
        }
        BlazeditDataset.validateDistance(opts.blazeditMinDistance(), opts.blazeditMaxDistance());
        opts.percentiles();
        opts.percentileMetricNames();
    }

    /** The tokenizer flags: a local path for a real tokenizer, or the reference's tokenizer-free set. */
    private static void validateTokenizer(ServeBenchOptions opts) throws UsageException {
        if (!"auto".equals(opts.tokenizerMode()) && !"hf".equals(opts.tokenizerMode())) {
            throw new UsageException("--tokenizer-mode is auto or hf, not '" + opts.tokenizerMode() + "'");
        }
        if (opts.tokenizerPath() != null) {
            java.nio.file.Path path;
            try {
                path = java.nio.file.Paths.get(opts.tokenizerPath());
            } catch (java.nio.file.InvalidPathException e) {
                throw new UsageException("--tokenizer is not a path: " + opts.tokenizerPath());
            }
            if (!java.nio.file.Files.exists(path)) {
                throw new UsageException("--tokenizer path does not exist: " + path);
            }
        }
        if (opts.skipTokenizerInit() && opts.tokenizerPath() != null) {
            throw new UsageException("--skip-tokenizer-init and --tokenizer are mutually exclusive");
        }
        if (opts.skipTokenizerInit()
                && !ServeBenchDataset.TOKENIZER_FREE.contains(opts.datasetName())) {
            throw new UsageException("--skip-tokenizer-init leaves no tokenizer for the '"
                    + opts.datasetName() + "' dataset: it needs one, so use one of "
                    + ServeBenchDataset.TOKENIZER_FREE);
        }
    }

    /** The dataset's endpoint, and the flags each endpoint can and cannot carry. */
    private static void validateEndpoint(ServeBenchOptions opts) throws UsageException {
        String endpoint = ServeBenchRunner.resolvedEndpoint(opts);
        String dataset = opts.datasetName();
        if (ServeBenchDataset.RERANK.contains(dataset) && !ServeBenchClient.RERANK.equals(endpoint)) {
            throw new UsageException("the '" + dataset + "' dataset requires --endpoint "
                    + ServeBenchClient.RERANK + ", not '" + endpoint + "'");
        }
        if (ServeBenchDataset.AUDIO.contains(dataset) && !ServeBenchClient.AUDIO.equals(endpoint)) {
            throw new UsageException("the '" + dataset + "' dataset requires --endpoint "
                    + ServeBenchClient.AUDIO + ", not '" + endpoint + "'");
        }
        if (ServeBenchDataset.CHAT.contains(dataset) && !ServeBenchClient.CHAT.equals(endpoint)) {
            throw new UsageException("the '" + dataset + "' dataset requires --endpoint "
                    + ServeBenchClient.CHAT + ", not '" + endpoint + "'");
        }
        if (ServeBenchDataset.TOKEN_PROMPTS.contains(dataset)
                && !ServeBenchClient.COMPLETIONS.equals(endpoint)) {
            throw new UsageException("the '" + dataset + "' dataset sends token ids, which only "
                    + ServeBenchClient.COMPLETIONS + " carries, not '" + endpoint + "'");
        }
        boolean generation = ServeBenchClient.COMPLETIONS.equals(endpoint)
                || ServeBenchClient.CHAT.equals(endpoint);
        if (opts.useBeamSearch() && !generation) {
            throw new UsageException("--use-beam-search needs a generation endpoint, not '" + endpoint + "'");
        }
        if (opts.probeRequestRate() > 0.0 && !generation) {
            throw new UsageException("--probe-request-rate sends generation requests, so it needs "
                    + ServeBenchClient.COMPLETIONS + " or " + ServeBenchClient.CHAT + ", not '"
                    + endpoint + "'");
        }
        if (opts.enableMultimodalChat() && ServeBenchDataset.AUDIO.contains(dataset)) {
            throw new UsageException("--enable-multimodal-chat is not supported for the '" + dataset
                    + "' dataset: its audio uploads to " + ServeBenchClient.AUDIO);
        }
    }

    /** The general lengths map onto the datasets' own flags, as the reference's {@code --input-len} does. */
    private static void applyGeneralLengths(ServeBenchOptions.Builder b, Integer inputLen, Integer outputLen) {
        if (inputLen != null) {
            b.inputLen(inputLen);
            b.randomInputLen(inputLen);
        }
        if (outputLen != null) {
            b.outputLen(outputLen);
            b.randomOutputLen(outputLen);
            b.sharegptOutputLen(outputLen);
            b.customOutputLen(outputLen);
            b.prefixRepetitionOutputLen(outputLen);
            b.hfOutputLen(outputLen);
            b.specBenchOutputLen(outputLen);
            b.speedBenchOutputLen(outputLen);
        }
    }

    /** The next argument, or the flag's own error: a flag with no value has no default to invent. */
    private static String value(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }

    /** A {@code KEY=VALUE} pair of a repeatable flag. */
    private interface PairConsumer {

        void accept(String pair) throws UsageException;
    }

    /** Whether an argument is a value (the next of {@code --goodput}'s repeatable pairs). */
    private static boolean isValue(String arg) {
        return !arg.startsWith("-");
    }

    /**
     * Consumes every value of a repeatable flag, as the reference's {@code nargs="*"} does.
     *
     * @return the index of the flag's last value, the for loop advancing past it
     * @throws UsageException when the flag carries no value
     */
    private static int repeatable(String flag, String[] args, int index, PairConsumer consumer)
            throws UsageException {
        int i = index;
        while (i + 1 < args.length && isValue(args[i + 1])) {
            consumer.accept(args[++i]);
        }
        if (i == index) {
            throw new UsageException(flag + " needs at least one KEY=VALUE");
        }
        return i;
    }

    /** The values of a repeatable flag: every following non-flag argument. */
    private static List<String> values(String[] args, int index) throws UsageException {
        List<String> out = new ArrayList<>();
        int i = index + 1;
        while (i < args.length && isValue(args[i])) {
            out.add(args[i]);
            i++;
        }
        if (out.isEmpty()) {
            throw new UsageException("--lora-modules needs at least one adapter name");
        }
        return out;
    }

    /** The count of values a repeatable flag consumed. */
    private static int valueCount(String[] args, int index) {
        int n = 0;
        int i = index + 1;
        while (i < args.length && isValue(args[i])) {
            n++;
            i++;
        }
        return n;
    }

    /** {@code --header KEY=VALUE}: exactly one {@code =}, a non-empty key. */
    private static void header(ServeBenchOptions.Builder b, String pair) throws UsageException {
        int eq = pair.indexOf('=');
        if (eq <= 0) {
            throw new UsageException("--header needs the KEY=VALUE format, not '" + pair + "'");
        }
        b.header(pair.substring(0, eq).trim(), pair.substring(eq + 1).trim());
    }

    /** {@code --metadata KEY=VALUE}: exactly one {@code =}. */
    private static void metadata(Map<String, String> metadata, String pair) throws UsageException {
        int eq = pair.indexOf('=');
        if (eq <= 0) {
            throw new UsageException("--metadata needs the KEY=VALUE format, not '" + pair + "'");
        }
        metadata.put(pair.substring(0, eq).trim(), pair.substring(eq + 1).trim());
    }

    /** {@code --goodput KEY:VALUE}: a metric of ttft, tpot, e2el and a non-negative millisecond value. */
    private static void goodput(ServeBenchOptions.Builder b, String pair, Map<String, Double> goodput)
            throws UsageException {
        int colon = pair.indexOf(':');
        if (colon <= 0) {
            throw new UsageException("--goodput needs the KEY:VALUE format in milliseconds, not '" + pair + "'");
        }
        String key = pair.substring(0, colon).trim();
        if (!List.of("ttft", "tpot", "e2el").contains(key)) {
            throw new UsageException("the service level objective name is ttft, tpot or e2el, not '" + key + "'");
        }
        double value;
        try {
            value = Double.parseDouble(pair.substring(colon + 1).trim());
        } catch (NumberFormatException e) {
            throw new UsageException("the service level objective value of '" + key + "' is a number"
                    + " in milliseconds, not '" + pair.substring(colon + 1) + "'");
        }
        if (value < 0) {
            throw new UsageException("the service level objective value of '" + key + "' must be"
                    + " non-negative, got " + value);
        }
        goodput.put(key, value);
    }

    /** {@code --extra-body JSON}: an object merged into every request body. */
    private static void extraBody(ServeBenchOptions.Builder b, String json) throws UsageException {
        Object parsed;
        try {
            parsed = Json.parse(json, Json.Limits.UNTRUSTED);
        } catch (JsonException e) {
            throw new UsageException("--extra-body needs a JSON object, not '" + json + "'");
        }
        if (!(parsed instanceof Map)) {
            throw new UsageException("--extra-body needs a JSON object, not '" + json + "'");
        }
        for (Map.Entry<?, ?> e : ((Map<?, ?>) parsed).entrySet()) {
            b.samplingParam(String.valueOf(e.getKey()), e.getValue());
        }
    }

    /** {@code --chat-template-kwargs JSON}: the variables bound into every client-side render. */
    private static Map<String, Object> chatTemplateKwargs(String json, String flag) throws UsageException {
        try {
            return ChatTemplateKwargs.parse(json, flag);
        } catch (IllegalArgumentException e) {
            throw new UsageException(flag + " " + e.getMessage());
        }
    }

    /** {@code --timeline-itl-thresholds ms,ms}: exactly two non-negative millisecond values. */
    private static List<Double> timelineThresholds(String value, String flag) throws UsageException {
        String[] parts = value.split(",");
        if (parts.length != 2) {
            throw new UsageException(flag + " needs exactly two comma-separated millisecond values,"
                    + " not '" + value + "'");
        }
        List<Double> out = new ArrayList<>(2);
        for (String part : parts) {
            double threshold;
            try {
                threshold = Double.parseDouble(part.trim());
            } catch (NumberFormatException e) {
                throw new UsageException(flag + " needs two numbers, not '" + value + "'");
            }
            if (!(threshold >= 0.0) || Double.isInfinite(threshold)) {
                throw new UsageException(flag + " thresholds must be non-negative finite numbers, got "
                        + threshold);
            }
            out.add(threshold);
        }
        if (out.get(0) > out.get(1)) {
            throw new UsageException(flag + " needs the lower threshold first, got '" + value + "'");
        }
        return out;
    }

    /** The dataset name, one of {@link ServeBenchDataset#NAMES}. */
    private static String datasetName(String v) throws UsageException {
        if (!ServeBenchDataset.NAMES.contains(v)) {
            throw new UsageException("the dataset is one of " + ServeBenchDataset.NAMES + ", not '" + v + "'");
        }
        return v;
    }

    private static int port(String v, String flag) throws UsageException {
        int p = intAtLeast(v, 0, flag);
        if (p > 65535) {
            throw new UsageException(flag + " must be in [0, 65535], got " + p);
        }
        return p;
    }

    private static double burstiness(String v, String flag) throws UsageException {
        String lower = v.trim().toLowerCase(Locale.ROOT);
        if (lower.equals("inf") || lower.equals("infinity")) {
            return Double.POSITIVE_INFINITY;
        }
        double b;
        try {
            b = Double.parseDouble(v.trim());
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs a number or 'inf', not '" + v + "'");
        }
        if (!(b > 0.0) || Double.isNaN(b)) {
            throw new UsageException(flag + " must be positive, got " + b);
        }
        return b;
    }

    private static String rampUpStrategy(String v) throws UsageException {
        if (!List.of("linear", "exponential").contains(v)) {
            throw new UsageException("--ramp-up-strategy is linear or exponential, not '" + v + "'");
        }
        return v;
    }

    private static String assignment(String v) throws UsageException {
        if (!List.of("random", "round-robin").contains(v)) {
            throw new UsageException("--lora-assignment is random or round-robin, not '" + v + "'");
        }
        return v;
    }

    private static String tokenizerMode(String v) throws UsageException {
        if (!"auto".equals(v) && !"hf".equals(v)) {
            throw new UsageException("--tokenizer-mode is auto or hf, not '" + v + "'");
        }
        return v;
    }

    private static String speedBenchSubset(String v) throws UsageException {
        if (!List.of("qualitative", "throughput_1k", "throughput_2k", "throughput_8k",
                "throughput_16k", "throughput_32k").contains(v)) {
            throw new UsageException("--speed-bench-dataset-subset is qualitative, throughput_1k,"
                    + " throughput_2k, throughput_8k, throughput_16k or throughput_32k, not '" + v + "'");
        }
        return v;
    }

    private static double positiveDouble(String v, String flag) throws UsageException {
        double value = finiteDouble(v, flag);
        if (!(value > 0.0)) {
            throw new UsageException(flag + " must be positive, got " + value);
        }
        return value;
    }

    private static double nonNegativeDouble(String v, String flag) throws UsageException {
        String lower = v.trim().toLowerCase(Locale.ROOT);
        if (lower.equals("inf") || lower.equals("infinity")) {
            return Double.POSITIVE_INFINITY;
        }
        double value = finiteDouble(v, flag);
        if (value < 0.0) {
            throw new UsageException(flag + " must be non-negative, got " + value);
        }
        return value;
    }

    private static double finiteDouble(String v, String flag) throws UsageException {
        double value;
        try {
            value = Double.parseDouble(v.trim());
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs a number, not '" + v + "'");
        }
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            throw new UsageException(flag + " needs a finite number, not '" + v + "'");
        }
        return value;
    }

    private static double ratioInclusive(String v, String flag) throws UsageException {
        double value = finiteDouble(v, flag);
        if (!(value >= 0.0 && value <= 1.0)) {
            throw new UsageException(flag + " must be in [0, 1], got " + value);
        }
        return value;
    }

    /** The per-modality item caps of {@code --random-mm-limit-mm-per-prompt}. */
    private static Map<String, Integer> limitPerPrompt(String json, String flag) throws UsageException {
        Object parsed;
        try {
            parsed = Json.parse(json, Json.Limits.UNTRUSTED);
        } catch (JsonException e) {
            throw new UsageException(flag + " needs a JSON object of modality caps, not '" + json + "'");
        }
        if (!(parsed instanceof Map)) {
            throw new UsageException(flag + " needs a JSON object of modality caps, not '" + json + "'");
        }
        Map<String, Integer> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : ((Map<?, ?>) parsed).entrySet()) {
            if (!(entry.getValue() instanceof Number)) {
                throw new UsageException(flag + " needs integer caps, but '" + entry.getKey() + "' is "
                        + entry.getValue());
            }
            int limit = ((Number) entry.getValue()).intValue();
            if (limit < 0) {
                throw new UsageException(flag + " caps cannot be negative, got " + limit);
            }
            out.put(String.valueOf(entry.getKey()), limit);
        }
        if (out.isEmpty()) {
            throw new UsageException(flag + " needs at least one modality cap");
        }
        return out;
    }

    /**
     * The random-mm bucket configuration: a Python literal of {@code (height, width, frames)} tuple
     * keys and probabilities, or the same with quoted {@code "height,width,frames"} keys.
     */
    private static Map<String, Double> bucketConfig(String value, String flag) throws UsageException {
        String text = value.trim();
        if (!text.startsWith("{") || !text.endsWith("}")) {
            throw new UsageException(flag + " needs an object like {(256, 256, 1): 0.5}, not '" + value + "'");
        }
        text = text.substring(1, text.length() - 1).trim();
        Map<String, Double> out = new LinkedHashMap<>();
        if (text.isEmpty()) {
            throw new UsageException(flag + " needs at least one bucket");
        }
        for (String entry : splitTopLevel(text)) {
            int colon = entry.lastIndexOf(':');
            if (colon <= 0) {
                throw new UsageException(flag + " entry '" + entry + "' is not '(height, width, frames): p'");
            }
            String key = entry.substring(0, colon).trim();
            String probabilityText = entry.substring(colon + 1).trim();
            if (key.startsWith("(") && key.endsWith(")")) {
                key = key.substring(1, key.length() - 1);
            } else if (key.startsWith("\"") && key.endsWith("\"")) {
                key = key.substring(1, key.length() - 1);
            } else {
                throw new UsageException(flag + " entry '" + entry + "' is not '(height, width, frames): p'");
            }
            String[] parts = key.split(",");
            if (parts.length != 3) {
                throw new UsageException(flag + " key '" + key + "' is not 'height,width,frames'");
            }
            try {
                String normalized = Integer.parseInt(parts[0].trim()) + ","
                        + Integer.parseInt(parts[1].trim()) + ","
                        + Integer.parseInt(parts[2].trim());
                double probability = Double.parseDouble(probabilityText);
                if (probability < 0.0) {
                    throw new UsageException(flag + " probabilities cannot be negative");
                }
                out.put(normalized, probability);
            } catch (NumberFormatException e) {
                throw new UsageException(flag + " entry '" + entry + "' is not '(height, width, frames): p'");
            }
        }
        return out;
    }

    /** Splits a bucket list on the commas outside its parentheses. */
    private static List<String> splitTopLevel(String text) {
        List<String> out = new ArrayList<>();
        StringBuilder entry = new StringBuilder();
        int depth = 0;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c == '(' || c == '[') {
                depth++;
            } else if (c == ')' || c == ']') {
                depth--;
            }
            if (c == ',' && depth == 0) {
                out.add(entry.toString().trim());
                entry.setLength(0);
            } else {
                entry.append(c);
            }
        }
        if (entry.toString().trim().length() > 0) {
            out.add(entry.toString().trim());
        }
        return out;
    }

    private static int positiveInt(String v, String flag) throws UsageException {
        int n = intAtLeast(v, 0, flag);
        if (n < 1) {
            throw new UsageException(flag + " needs a positive integer, not '" + v + "'");
        }
        return n;
    }

    private static int nonNegativeInt(String v, String flag) throws UsageException {
        return intAtLeast(v, 0, flag);
    }

    private static int intAtLeast(String v, int floor, String flag) throws UsageException {
        try {
            int n = Integer.parseInt(v.trim());
            if (n < floor) {
                throw new NumberFormatException();
            }
            return n;
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs an integer of at least " + floor + ", not '" + v + "'");
        }
    }

    private static long longValue(String v, String flag) throws UsageException {
        try {
            return Long.parseLong(v.trim());
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs an integer, not '" + v + "'");
        }
    }

    private static double doubleValue(String v, String flag) throws UsageException {
        try {
            return Double.parseDouble(v.trim());
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs a number, not '" + v + "'");
        }
    }

    /**
     * Parses nothing and runs one benchmark over built options: the entry the sweep applies its
     * parameter records to.
     *
     * @throws IOException for an unreachable server, a failed request or an unwritable result file
     */
    static Map<String, Object> execute(ServeBenchOptions opts, Appendable out, Appendable err) throws IOException {
        return ServeBenchRunner.execute(opts, out, err);
    }
}
