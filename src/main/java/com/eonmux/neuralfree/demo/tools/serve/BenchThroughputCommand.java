package com.eonmux.neuralfree.demo.tools.serve;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * {@code nftool bench-throughput}: the offline throughput benchmark of vLLM's {@code vllm bench
 * throughput} in its server form: the whole workload goes out at once ({@code --num-prompts}
 * requests, no rate, no cap) over a running OpenAI-compatible server, and the report gives the
 * request, output-token and total-token throughputs over the run's duration, with the end-to-end
 * latency statistics. The engine-level run of the reference has no HTTP equivalent: the traffic
 * flags of {@code bench-serve} refuse here by name, the dataset flags carry over unchanged.
 */
public final class BenchThroughputCommand {

    /** The command surface: {@code --help} and every error print it. */
    static final String USAGE =
            "nftool bench-throughput [options]\n"
            + "options:\n"
            + "  -h, --help                            this text\n"
            + "  --host <h> / --port <p> / --base-url <url> / --endpoint <path>\n"
            + "  --model <id> / --served-model-name <id> / --header KEY=VALUE / --insecure\n"
            + "  --dataset-name <n>            random (default), random-mm, random-rerank, prefix_repetition,\n"
            + "                                sharegpt, custom, custom_image, custom_audio, burstgpt,\n"
            + "                                spec_bench, speed_bench, timed_trace, blazedit, asr, hf\n"
            + "  --dataset-path <path>         the dataset file or local dataset directory\n"
            + "  --num-prompts <n>             workload size, default 1000\n"
            + "  --seed <n>  --no-oversample  --disable-shuffle\n"
            + "  --input-len <n>  --output-len <n>\n"
            + "  --random-input-len <n>  --random-output-len <n>  --random-range-ratio <r|json>  --random-prefix-len <n>\n"
            + "  --prefix-repetition-prefix-len <n>  --prefix-repetition-suffix-len <n>  --prefix-repetition-num-prefixes <n>  --prefix-repetition-output-len <n>\n"
            + "  --sharegpt-output-len <n>  --custom-output-len <n>\n"
            + "  --percentile-metrics <list>   of ttft,tpot,itl,e2el,client_queue_time,\n"
            + "                                e2el_including_client_queue, default ttft,tpot,itl\n"
            + "  --metric-percentiles <list>   of numbers in [0,100], default 99\n"
            + "  --save-result  --append-result  --save-detailed  --result-dir <dir>  --result-filename <name>\n"
            + "  --metadata KEY=VALUE  --label <l>  --request-id-prefix <p>  --disable-tqdm\n"
            + "  --temperature <t>  --top-p <p>  --top-k <k>  --min-p <p>\n"
            + "  --frequency-penalty <f>  --presence-penalty <f>  --repetition-penalty <f>\n"
            + "  --logprobs <n>  --extra-body <json>\n"
            + "  --lora-modules <m...>  --lora-assignment <s>\n"
            + "The workload leaves all at once; traffic shaping belongs to bench-serve (--request-rate,\n"
            + "--burstiness, --max-concurrency and the ramp-up flags are refused).\n";

    private BenchThroughputCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code bench-throughput}. */
    public static void run(String[] args) {
        try {
            refuseTrafficFlags(args);
            ServeBenchOptions opts = ServeBenchCommand.parse(args);
            if (opts == null) {
                return;
            }
            if (!Double.isInfinite(opts.requestRate()) || opts.maxConcurrency() != null
                    || opts.rampUpStrategy() != null) {
                throw new IllegalArgumentException("bench-throughput sends the whole workload at once:"
                        + " bench-serve carries the traffic flags");
            }
            executeAndReport(opts);
        } catch (ServeBenchCommand.UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException | IllegalStateException | IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /** The traffic flags the command does not carry, refused by name before the shared parse. */
    static void refuseTrafficFlags(String[] args) throws ServeBenchCommand.UsageException {
        List<String> traffic = List.of("--request-rate", "--burstiness", "--max-concurrency",
                "--ramp-up-strategy", "--ramp-up-start-rps", "--ramp-up-end-rps", "--num-warmups",
                "--ready-check-timeout-sec", "--goodput");
        for (String arg : args) {
            String normalized = arg.startsWith("--") ? arg.replace('_', '-') : arg;
            if (traffic.contains(normalized)) {
                throw new ServeBenchCommand.UsageException(normalized + " belongs to bench-serve:"
                        + " bench-throughput sends the whole workload at once");
            }
        }
    }

    /** The run and its report; the result JSON returns for the tests' reuse. */
    static Map<String, Object> executeAndReport(ServeBenchOptions opts) throws IOException {
        return ServeBenchCommand.execute(opts, System.out, System.err);
    }
}
