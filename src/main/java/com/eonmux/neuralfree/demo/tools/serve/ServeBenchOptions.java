package com.eonmux.neuralfree.demo.tools.serve;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

/**
 * The flags of one serving benchmark run, as the commands parse them: the server address and
 * endpoint, the dataset and its parameters, the traffic (request rate, burstiness, ramp-up, maximum
 * concurrency), the reported percentiles, the goodput objectives and the result file. Immutable;
 * the builder carries the defaults the references set. Field semantics match
 * {@code vllm bench serve}'s flags of the same name.
 */
public final class ServeBenchOptions {

    /** The metric names {@code --percentile-metrics} accepts, the reference's. */
    public static final List<String> METRIC_NAMES = List.of("ttft", "tpot", "itl", "e2el",
            "client_queue_time", "e2el_including_client_queue");

    /** The per-metric range ratio of the random dataset: one float, or input and output separately. */
    public static final class RangeRatio {

        private final double input;
        private final double output;

        RangeRatio(double input, double output) {
            this.input = input;
            this.output = output;
        }

        /** The input length's ratio. */
        public double input() {
            return input;
        }

        /** The output length's ratio. */
        public double output() {
            return output;
        }
    }

    private final String host;
    private final int port;
    private final String baseUrl;
    private final String endpoint;
    private final boolean endpointExplicit;
    private final String model;
    private final String servedModelName;
    private final String label;
    private final Map<String, String> headers;
    private final String datasetName;
    private final String datasetPath;
    private final int numPrompts;
    private final long seed;
    private final boolean noOversample;
    private final boolean disableShuffle;
    private final Integer inputLen;
    private final Integer outputLen;
    private final int randomInputLen;
    private final int randomOutputLen;
    private final RangeRatio randomRangeRatio;
    private final int randomPrefixLen;
    private final int prefixRepetitionPrefixLen;
    private final int prefixRepetitionSuffixLen;
    private final int prefixRepetitionNumPrefixes;
    private final int prefixRepetitionOutputLen;
    private final Integer sharegptOutputLen;
    private final int customOutputLen;
    private final double requestRate;
    private final double burstiness;
    private final Integer maxConcurrency;
    private final String rampUpStrategy;
    private final Integer rampUpStartRps;
    private final Integer rampUpEndRps;
    private final String percentileMetrics;
    private final String metricPercentiles;
    private final Map<String, Double> goodput;
    private final boolean ignoreEos;
    private final int numWarmups;
    private final boolean disableTqdm;
    private final boolean saveResult;
    private final boolean saveDetailed;
    private final boolean appendResult;
    private final Map<String, String> metadata;
    private final String resultDir;
    private final String resultFilename;
    private final String requestIdPrefix;
    private final Map<String, Object> samplingParams;
    private final Integer logprobs;
    private final List<String> loraModules;
    private final String loraAssignment;
    private final int readyCheckTimeoutSec;
    private final boolean insecure;
    private final String hfSubset;
    private final String hfSplit;
    private final String hfName;
    private final Integer hfOutputLen;
    private final int specBenchOutputLen;
    private final String specBenchCategory;
    private final int timedTraceChunkHashSize;
    private final double timedTraceSecMultiplier;
    private final String timedTraceLabelTimestamp;
    private final String timedTraceLabelInputLength;
    private final String timedTraceLabelOutputLength;
    private final String timedTraceLabelHashIds;
    private final boolean selfTimed;
    private final double blazeditMinDistance;
    private final double blazeditMaxDistance;
    private final double asrMinAudioLenSec;
    private final double asrMaxAudioLenSec;
    private final int randomMmBaseItemsPerRequest;
    private final double randomMmNumMmItemsRangeRatio;
    private final Map<String, Integer> randomMmLimitMmPerPrompt;
    private final Map<String, Double> randomMmBucketConfig;
    private final boolean noReranker;
    private final int randomBatchSize;
    private final String speedBenchDatasetSubset;
    private final int speedBenchOutputLen;
    private final String speedBenchCategory;
    private final boolean enableMultimodalChat;
    private final boolean useBeamSearch;
    private final double probeRequestRate;
    private final String tokenizerPath;
    private final String tokenizerMode;
    private final boolean skipTokenizerInit;
    private final boolean profile;
    private final boolean plotTimeline;
    private final List<Double> timelineItlThresholds;
    private final boolean plotDatasetStats;
    private final Map<String, Object> chatTemplateKwargs;
    private final boolean skipChatTemplate;
    private final boolean noStream;

    private ServeBenchOptions(Builder b) {
        this.host = b.host;
        this.port = b.port;
        this.baseUrl = b.baseUrl;
        this.endpoint = b.endpoint;
        this.endpointExplicit = b.endpointExplicit;
        this.model = b.model;
        this.servedModelName = b.servedModelName;
        this.label = b.label;
        this.headers = copyOf(b.headers);
        this.datasetName = b.datasetName;
        this.datasetPath = b.datasetPath;
        this.numPrompts = b.numPrompts;
        this.seed = b.seed;
        this.noOversample = b.noOversample;
        this.disableShuffle = b.disableShuffle;
        this.inputLen = b.inputLen;
        this.outputLen = b.outputLen;
        this.randomInputLen = b.randomInputLen;
        this.randomOutputLen = b.randomOutputLen;
        this.randomRangeRatio = b.randomRangeRatio;
        this.randomPrefixLen = b.randomPrefixLen;
        this.prefixRepetitionPrefixLen = b.prefixRepetitionPrefixLen;
        this.prefixRepetitionSuffixLen = b.prefixRepetitionSuffixLen;
        this.prefixRepetitionNumPrefixes = b.prefixRepetitionNumPrefixes;
        this.prefixRepetitionOutputLen = b.prefixRepetitionOutputLen;
        this.sharegptOutputLen = b.sharegptOutputLen;
        this.customOutputLen = b.customOutputLen;
        this.requestRate = b.requestRate;
        this.burstiness = b.burstiness;
        this.maxConcurrency = b.maxConcurrency;
        this.rampUpStrategy = b.rampUpStrategy;
        this.rampUpStartRps = b.rampUpStartRps;
        this.rampUpEndRps = b.rampUpEndRps;
        this.percentileMetrics = b.percentileMetrics;
        this.metricPercentiles = b.metricPercentiles;
        this.goodput = copyOf(b.goodput);
        this.ignoreEos = b.ignoreEos;
        this.numWarmups = b.numWarmups;
        this.disableTqdm = b.disableTqdm;
        this.saveResult = b.saveResult;
        this.saveDetailed = b.saveDetailed;
        this.appendResult = b.appendResult;
        this.metadata = copyOf(b.metadata);
        this.resultDir = b.resultDir;
        this.resultFilename = b.resultFilename;
        this.requestIdPrefix = b.requestIdPrefix;
        this.samplingParams = copyOf(b.samplingParams);
        this.logprobs = b.logprobs;
        this.loraModules = b.loraModules == null ? null : new java.util.ArrayList<>(b.loraModules);
        this.loraAssignment = b.loraAssignment;
        this.readyCheckTimeoutSec = b.readyCheckTimeoutSec;
        this.insecure = b.insecure;
        this.hfSubset = b.hfSubset;
        this.hfSplit = b.hfSplit;
        this.hfName = b.hfName;
        this.hfOutputLen = b.hfOutputLen;
        this.specBenchOutputLen = b.specBenchOutputLen;
        this.specBenchCategory = b.specBenchCategory;
        this.timedTraceChunkHashSize = b.timedTraceChunkHashSize;
        this.timedTraceSecMultiplier = b.timedTraceSecMultiplier;
        this.timedTraceLabelTimestamp = b.timedTraceLabelTimestamp;
        this.timedTraceLabelInputLength = b.timedTraceLabelInputLength;
        this.timedTraceLabelOutputLength = b.timedTraceLabelOutputLength;
        this.timedTraceLabelHashIds = b.timedTraceLabelHashIds;
        this.selfTimed = b.selfTimed;
        this.blazeditMinDistance = b.blazeditMinDistance;
        this.blazeditMaxDistance = b.blazeditMaxDistance;
        this.asrMinAudioLenSec = b.asrMinAudioLenSec;
        this.asrMaxAudioLenSec = b.asrMaxAudioLenSec;
        this.randomMmBaseItemsPerRequest = b.randomMmBaseItemsPerRequest;
        this.randomMmNumMmItemsRangeRatio = b.randomMmNumMmItemsRangeRatio;
        this.randomMmLimitMmPerPrompt = copyOf(b.randomMmLimitMmPerPrompt);
        this.randomMmBucketConfig = copyOf(b.randomMmBucketConfig);
        this.noReranker = b.noReranker;
        this.randomBatchSize = b.randomBatchSize;
        this.speedBenchDatasetSubset = b.speedBenchDatasetSubset;
        this.speedBenchOutputLen = b.speedBenchOutputLen;
        this.speedBenchCategory = b.speedBenchCategory;
        this.enableMultimodalChat = b.enableMultimodalChat;
        this.useBeamSearch = b.useBeamSearch;
        this.probeRequestRate = b.probeRequestRate;
        this.tokenizerPath = b.tokenizerPath;
        this.tokenizerMode = b.tokenizerMode;
        this.skipTokenizerInit = b.skipTokenizerInit;
        this.profile = b.profile;
        this.plotTimeline = b.plotTimeline;
        this.timelineItlThresholds = List.copyOf(b.timelineItlThresholds);
        this.plotDatasetStats = b.plotDatasetStats;
        this.chatTemplateKwargs = copyOf(b.chatTemplateKwargs);
        this.skipChatTemplate = b.skipChatTemplate;
        this.noStream = b.noStream;
    }

    private static <K, V> Map<K, V> copyOf(Map<K, V> map) {
        return map == null ? new LinkedHashMap<>() : new LinkedHashMap<>(map);
    }

    /** A builder carrying the reference defaults. */
    public static Builder builder() {
        return new Builder();
    }

    public String host() {
        return host;
    }

    public int port() {
        return port;
    }

    public String baseUrl() {
        return baseUrl;
    }

    public String endpoint() {
        return endpoint;
    }

    /** Whether {@code --endpoint} was given; otherwise the dataset selects the endpoint. */
    public boolean endpointExplicit() {
        return endpointExplicit;
    }

    public String model() {
        return model;
    }

    public String servedModelName() {
        return servedModelName;
    }

    public String label() {
        return label;
    }

    public Map<String, String> headers() {
        return new LinkedHashMap<>(headers);
    }

    public String datasetName() {
        return datasetName;
    }

    public String datasetPath() {
        return datasetPath;
    }

    public int numPrompts() {
        return numPrompts;
    }

    public long seed() {
        return seed;
    }

    public boolean noOversample() {
        return noOversample;
    }

    public boolean disableShuffle() {
        return disableShuffle;
    }

    /** The general input length, mapped onto the dataset's own input flag; null for the defaults. */
    public Integer inputLen() {
        return inputLen;
    }

    /** The general output length, mapped onto the dataset's own output flag; null for the defaults. */
    public Integer outputLen() {
        return outputLen;
    }

    public int randomInputLen() {
        return randomInputLen;
    }

    public int randomOutputLen() {
        return randomOutputLen;
    }

    public RangeRatio randomRangeRatio() {
        return randomRangeRatio;
    }

    public int randomPrefixLen() {
        return randomPrefixLen;
    }

    public int prefixRepetitionPrefixLen() {
        return prefixRepetitionPrefixLen;
    }

    public int prefixRepetitionSuffixLen() {
        return prefixRepetitionSuffixLen;
    }

    public int prefixRepetitionNumPrefixes() {
        return prefixRepetitionNumPrefixes;
    }

    public int prefixRepetitionOutputLen() {
        return prefixRepetitionOutputLen;
    }

    public Integer sharegptOutputLen() {
        return sharegptOutputLen;
    }

    /** The custom dataset's output length; -1 reads every record's {@code output_tokens}. */
    public int customOutputLen() {
        return customOutputLen;
    }

    public double requestRate() {
        return requestRate;
    }

    public double burstiness() {
        return burstiness;
    }

    public Integer maxConcurrency() {
        return maxConcurrency;
    }

    public String rampUpStrategy() {
        return rampUpStrategy;
    }

    public Integer rampUpStartRps() {
        return rampUpStartRps;
    }

    public Integer rampUpEndRps() {
        return rampUpEndRps;
    }

    /** The metrics percentiles report ({@code ttft,tpot,itl,e2el}); null for the dataset's default. */
    public String percentileMetrics() {
        return percentileMetrics;
    }

    public String metricPercentiles() {
        return metricPercentiles;
    }

    public Map<String, Double> goodput() {
        return new LinkedHashMap<>(goodput);
    }

    public boolean ignoreEos() {
        return ignoreEos;
    }

    public int numWarmups() {
        return numWarmups;
    }

    public boolean disableTqdm() {
        return disableTqdm;
    }

    public boolean saveResult() {
        return saveResult;
    }

    public boolean saveDetailed() {
        return saveDetailed;
    }

    public boolean appendResult() {
        return appendResult;
    }

    public Map<String, String> metadata() {
        return new LinkedHashMap<>(metadata);
    }

    public String resultDir() {
        return resultDir;
    }

    public String resultFilename() {
        return resultFilename;
    }

    public String requestIdPrefix() {
        return requestIdPrefix;
    }

    /** The sampling parameters of every request ({@code temperature}, {@code top_p}, ...). */
    public Map<String, Object> samplingParams() {
        return new LinkedHashMap<>(samplingParams);
    }

    public Integer logprobs() {
        return logprobs;
    }

    public List<String> loraModules() {
        return loraModules == null ? null : new ArrayList<>(loraModules);
    }


    public String loraAssignment() {
        return loraAssignment;
    }

    public int readyCheckTimeoutSec() {
        return readyCheckTimeoutSec;
    }

    public boolean insecure() {
        return insecure;
    }

    /** The Hugging Face dataset subset ({@code --hf-subset}), or null. */
    public String hfSubset() {
        return hfSubset;
    }

    /** The Hugging Face split ({@code --hf-split}), or null for the dataset's default. */
    public String hfSplit() {
        return hfSplit;
    }

    /** The Hugging Face dataset name a local {@code --dataset-path} stands for, or null. */
    public String hfName() {
        return hfName;
    }

    /** The fixed output length of the {@code hf} datasets, or null for the dataset's own. */
    public Integer hfOutputLen() {
        return hfOutputLen;
    }

    public int specBenchOutputLen() {
        return specBenchOutputLen;
    }

    public String specBenchCategory() {
        return specBenchCategory;
    }

    public int timedTraceChunkHashSize() {
        return timedTraceChunkHashSize;
    }

    public double timedTraceSecMultiplier() {
        return timedTraceSecMultiplier;
    }

    public String timedTraceLabelTimestamp() {
        return timedTraceLabelTimestamp;
    }

    public String timedTraceLabelInputLength() {
        return timedTraceLabelInputLength;
    }

    public String timedTraceLabelOutputLength() {
        return timedTraceLabelOutputLength;
    }

    public String timedTraceLabelHashIds() {
        return timedTraceLabelHashIds;
    }

    /** Whether a trace supplies the request arrivals instead of the traffic flags. */
    public boolean selfTimed() {
        return selfTimed;
    }

    public double blazeditMinDistance() {
        return blazeditMinDistance;
    }

    public double blazeditMaxDistance() {
        return blazeditMaxDistance;
    }

    public double asrMinAudioLenSec() {
        return asrMinAudioLenSec;
    }

    public double asrMaxAudioLenSec() {
        return asrMaxAudioLenSec;
    }

    public int randomMmBaseItemsPerRequest() {
        return randomMmBaseItemsPerRequest;
    }

    public double randomMmNumMmItemsRangeRatio() {
        return randomMmNumMmItemsRangeRatio;
    }

    public Map<String, Integer> randomMmLimitMmPerPrompt() {
        return new LinkedHashMap<>(randomMmLimitMmPerPrompt);
    }

    /** The bucket probabilities keyed by {@code "height,width,frames"}. */
    public Map<String, Double> randomMmBucketConfig() {
        return new LinkedHashMap<>(randomMmBucketConfig);
    }

    public boolean noReranker() {
        return noReranker;
    }

    /** The batch size of the synthetic datasets ({@code --random-batch-size}). */
    public int randomBatchSize() {
        return randomBatchSize;
    }

    public String speedBenchDatasetSubset() {
        return speedBenchDatasetSubset;
    }

    public int speedBenchOutputLen() {
        return speedBenchOutputLen;
    }

    public String speedBenchCategory() {
        return speedBenchCategory;
    }

    /** Whether the datasets that support it render their media as chat content parts. */
    public boolean enableMultimodalChat() {
        return enableMultimodalChat;
    }

    /** Whether the generation requests carry {@code use_beam_search}. */
    public boolean useBeamSearch() {
        return useBeamSearch;
    }

    /** The rate of the single-token probe requests; 0 disables the probes. */
    public double probeRequestRate() {
        return probeRequestRate;
    }

    /** The local tokenizer file or Hugging Face directory, or null to count through the server. */
    public String tokenizerPath() {
        return tokenizerPath;
    }

    public String tokenizerMode() {
        return tokenizerMode;
    }

    /** Whether no tokenizer is initialized and the datasets count one token per prompt. */
    public boolean skipTokenizerInit() {
        return skipTokenizerInit;
    }

    /** Whether the run brackets the measured workload with the server's profile window. */
    public boolean profile() {
        return profile;
    }

    /** Whether a timeline HTML plot is written beside the result. */
    public boolean plotTimeline() {
        return plotTimeline;
    }

    /** The two ITL thresholds in milliseconds the timeline colours by. */
    public List<Double> timelineItlThresholds() {
        return new ArrayList<>(timelineItlThresholds);
    }

    /** Whether a dataset-statistics SVG plot is written beside the result. */
    public boolean plotDatasetStats() {
        return plotDatasetStats;
    }

    /** The {@code chat_template_kwargs} bound into every client-side chat template render. */
    public Map<String, Object> chatTemplateKwargs() {
        return new LinkedHashMap<>(chatTemplateKwargs);
    }

    /** Whether the datasets that render client-side send their raw prompts. */
    public boolean skipChatTemplate() {
        return skipChatTemplate;
    }

    /** Whether the Hub dataset is loaded completely rather than streamed (the harness always loads fully). */
    public boolean noStream() {
        return noStream;
    }

    /** The base URL the requests target: {@code --base-url}, else {@code http://host:port}. */
    public String resolvedBaseUrl() {
        return baseUrl != null ? baseUrl : "http://" + host + ":" + port;
    }

    /** The parsed {@code --metric-percentiles}, each in {@code [0, 100]}. */
    public List<Double> percentiles() {
        List<Double> out = new ArrayList<>();
        for (String part : metricPercentiles.split(",")) {
            String name = part.trim();
            if (name.isEmpty()) {
                throw new IllegalArgumentException("--metric-percentiles carries an empty percentile");
            }
            double p;
            try {
                p = Double.parseDouble(name);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("--metric-percentiles needs a comma-separated list of"
                        + " numbers, not '" + metricPercentiles + "'");
            }
            if (p < 0.0 || p > 100.0) {
                throw new IllegalArgumentException("the metric percentiles must be in [0, 100], got " + p);
            }
            out.add(p);
        }
        if (out.isEmpty()) {
            throw new IllegalArgumentException("--metric-percentiles needs at least one percentile");
        }
        return out;
    }

    /** The parsed {@code --percentile-metrics}, each one of the reference's metric names. */
    public List<String> percentileMetricNames() {
        String list = percentileMetrics != null ? percentileMetrics : "ttft,tpot,itl";
        if (list.trim().isEmpty()) {
            throw new IllegalArgumentException("--percentile-metrics needs at least one metric");
        }
        List<String> out = new ArrayList<>();
        for (String part : list.split(",")) {
            String name = part.trim();
            if (!METRIC_NAMES.contains(name)) {
                throw new IllegalArgumentException("the percentile metrics are " + METRIC_NAMES + ", not '"
                        + name + "'");
            }
            if (!out.contains(name)) {
                out.add(name);
            }
        }
        if (out.isEmpty()) {
            throw new IllegalArgumentException("--percentile-metrics needs at least one metric");
        }
        return out;
    }

    /** The builder of one immutable {@link ServeBenchOptions}, with the reference defaults. */
    public static final class Builder {

        private String host = "127.0.0.1";
        private int port = 8000;
        private String baseUrl;
        private String endpoint = "/v1/completions";
        private boolean endpointExplicit;
        private String model;
        private String servedModelName;
        private String label;
        private Map<String, String> headers = new LinkedHashMap<>();
        private String datasetName = "random";
        private String datasetPath;
        private int numPrompts = 1000;
        private long seed;
        private boolean noOversample;
        private boolean disableShuffle;
        private Integer inputLen;
        private Integer outputLen;
        private int randomInputLen = RandomDataset.DEFAULT_INPUT_LEN;
        private int randomOutputLen = RandomDataset.DEFAULT_OUTPUT_LEN;
        private RangeRatio randomRangeRatio = new RangeRatio(RandomDataset.DEFAULT_RANGE_RATIO,
                RandomDataset.DEFAULT_RANGE_RATIO);
        private int randomPrefixLen = RandomDataset.DEFAULT_PREFIX_LEN;
        private int prefixRepetitionPrefixLen = PrefixRepetitionDataset.DEFAULT_PREFIX_LEN;
        private int prefixRepetitionSuffixLen = PrefixRepetitionDataset.DEFAULT_SUFFIX_LEN;
        private int prefixRepetitionNumPrefixes = PrefixRepetitionDataset.DEFAULT_NUM_PREFIXES;
        private int prefixRepetitionOutputLen = PrefixRepetitionDataset.DEFAULT_OUTPUT_LEN;
        private Integer sharegptOutputLen;
        private int customOutputLen = 256;
        private double requestRate = Double.POSITIVE_INFINITY;
        private double burstiness = 1.0;
        private Integer maxConcurrency;
        private String rampUpStrategy;
        private Integer rampUpStartRps;
        private Integer rampUpEndRps;
        private String percentileMetrics;
        private String metricPercentiles = "99";
        private Map<String, Double> goodput = new LinkedHashMap<>();
        private boolean ignoreEos;
        private int numWarmups;
        private boolean disableTqdm;
        private boolean saveResult;
        private boolean saveDetailed;
        private boolean appendResult;
        private Map<String, String> metadata = new LinkedHashMap<>();
        private String resultDir;
        private String resultFilename;
        private String requestIdPrefix = "bench-" + UUID.randomUUID().toString().substring(0, 8) + "-";
        private Map<String, Object> samplingParams = new LinkedHashMap<>();
        private Integer logprobs;
        private List<String> loraModules;
        private String loraAssignment = "random";
        private int readyCheckTimeoutSec;
        private boolean insecure;
        private String hfSubset;
        private String hfSplit;
        private String hfName;
        private Integer hfOutputLen;
        private int specBenchOutputLen = 256;
        private String specBenchCategory;
        private int timedTraceChunkHashSize = 16;
        private double timedTraceSecMultiplier = 1.0;
        private String timedTraceLabelTimestamp = "timestamp";
        private String timedTraceLabelInputLength = "input_length";
        private String timedTraceLabelOutputLength = "output_length";
        private String timedTraceLabelHashIds = "hash_ids";
        private boolean selfTimed;
        private double blazeditMinDistance = 0.0;
        private double blazeditMaxDistance = 1.0;
        private double asrMinAudioLenSec = 0.0;
        private double asrMaxAudioLenSec = Double.POSITIVE_INFINITY;
        private int randomMmBaseItemsPerRequest = 1;
        private double randomMmNumMmItemsRangeRatio = 0.0;
        private Map<String, Integer> randomMmLimitMmPerPrompt = new LinkedHashMap<>();
        private Map<String, Double> randomMmBucketConfig = new LinkedHashMap<>();
        private boolean noReranker;
        private int randomBatchSize = 1;
        private String speedBenchDatasetSubset = "qualitative";
        private int speedBenchOutputLen = 4096;
        private String speedBenchCategory;
        private boolean enableMultimodalChat;
        private boolean useBeamSearch;
        private double probeRequestRate;
        private String tokenizerPath;
        private String tokenizerMode = "auto";
        private boolean skipTokenizerInit;
        private boolean profile;
        private boolean plotTimeline;
        private List<Double> timelineItlThresholds = List.of(25.0, 50.0);
        private boolean plotDatasetStats;
        private Map<String, Object> chatTemplateKwargs = new LinkedHashMap<>();
        private boolean skipChatTemplate;
        private boolean noStream;

        {
            randomMmLimitMmPerPrompt.put("image", 255);
            randomMmLimitMmPerPrompt.put("video", 1);
            randomMmBucketConfig.put("256,256,1", 0.5);
            randomMmBucketConfig.put("720,1280,1", 0.5);
            randomMmBucketConfig.put("720,1280,16", 0.0);
        }

        public Builder host(String v) {
            this.host = v;
            return this;
        }

        public Builder port(int v) {
            this.port = v;
            return this;
        }

        public Builder baseUrl(String v) {
            this.baseUrl = v;
            return this;
        }

        public Builder endpoint(String v) {
            this.endpoint = v;
            this.endpointExplicit = true;
            return this;
        }

        public Builder model(String v) {
            this.model = v;
            return this;
        }

        public Builder servedModelName(String v) {
            this.servedModelName = v;
            return this;
        }

        public Builder label(String v) {
            this.label = v;
            return this;
        }

        public Builder header(String key, String value) {
            this.headers.put(key, value);
            return this;
        }

        public Builder datasetName(String v) {
            this.datasetName = v;
            return this;
        }

        public Builder datasetPath(String v) {
            this.datasetPath = v;
            return this;
        }

        public Builder numPrompts(int v) {
            this.numPrompts = v;
            return this;
        }

        public Builder seed(long v) {
            this.seed = v;
            return this;
        }

        public Builder noOversample(boolean v) {
            this.noOversample = v;
            return this;
        }

        public Builder disableShuffle(boolean v) {
            this.disableShuffle = v;
            return this;
        }

        public Builder inputLen(Integer v) {
            this.inputLen = v;
            return this;
        }

        public Builder outputLen(Integer v) {
            this.outputLen = v;
            return this;
        }

        public Builder randomInputLen(int v) {
            this.randomInputLen = v;
            return this;
        }

        public Builder randomOutputLen(int v) {
            this.randomOutputLen = v;
            return this;
        }

        public Builder randomRangeRatio(RangeRatio v) {
            this.randomRangeRatio = v;
            return this;
        }

        public Builder randomPrefixLen(int v) {
            this.randomPrefixLen = v;
            return this;
        }

        public Builder prefixRepetitionPrefixLen(int v) {
            this.prefixRepetitionPrefixLen = v;
            return this;
        }

        public Builder prefixRepetitionSuffixLen(int v) {
            this.prefixRepetitionSuffixLen = v;
            return this;
        }

        public Builder prefixRepetitionNumPrefixes(int v) {
            this.prefixRepetitionNumPrefixes = v;
            return this;
        }

        public Builder prefixRepetitionOutputLen(int v) {
            this.prefixRepetitionOutputLen = v;
            return this;
        }

        public Builder sharegptOutputLen(Integer v) {
            this.sharegptOutputLen = v;
            return this;
        }

        public Builder customOutputLen(int v) {
            this.customOutputLen = v;
            return this;
        }

        public Builder requestRate(double v) {
            this.requestRate = v;
            return this;
        }

        public Builder burstiness(double v) {
            this.burstiness = v;
            return this;
        }

        public Builder maxConcurrency(Integer v) {
            this.maxConcurrency = v;
            return this;
        }

        public Builder rampUp(String strategy, Integer startRps, Integer endRps) {
            this.rampUpStrategy = strategy;
            this.rampUpStartRps = startRps;
            this.rampUpEndRps = endRps;
            return this;
        }

        public Builder percentileMetrics(String v) {
            this.percentileMetrics = v;
            return this;
        }

        public Builder metricPercentiles(String v) {
            this.metricPercentiles = v;
            return this;
        }

        public Builder goodput(Map<String, Double> v) {
            this.goodput = v == null ? new LinkedHashMap<>() : v;
            return this;
        }

        public Builder ignoreEos(boolean v) {
            this.ignoreEos = v;
            return this;
        }

        public Builder numWarmups(int v) {
            this.numWarmups = v;
            return this;
        }

        public Builder disableTqdm(boolean v) {
            this.disableTqdm = v;
            return this;
        }

        public Builder saveResult(boolean v) {
            this.saveResult = v;
            return this;
        }

        public Builder saveDetailed(boolean v) {
            this.saveDetailed = v;
            return this;
        }

        public Builder appendResult(boolean v) {
            this.appendResult = v;
            return this;
        }

        public Builder metadata(Map<String, String> v) {
            this.metadata = v == null ? new LinkedHashMap<>() : v;
            return this;
        }

        public Builder resultDir(String v) {
            this.resultDir = v;
            return this;
        }

        public Builder resultFilename(String v) {
            this.resultFilename = v;
            return this;
        }

        public Builder requestIdPrefix(String v) {
            this.requestIdPrefix = v;
            return this;
        }

        public Builder samplingParam(String name, Object value) {
            this.samplingParams.put(name, value);
            return this;
        }

        public Builder logprobs(Integer v) {
            this.logprobs = v;
            return this;
        }

        public Builder loraModules(List<String> v) {
            this.loraModules = v;
            return this;
        }

        public Builder loraAssignment(String v) {
            this.loraAssignment = v;
            return this;
        }

        public Builder readyCheckTimeoutSec(int v) {
            this.readyCheckTimeoutSec = v;
            return this;
        }

        public Builder insecure(boolean v) {
            this.insecure = v;
            return this;
        }

        public Builder hfSubset(String v) {
            this.hfSubset = v;
            return this;
        }

        public Builder hfSplit(String v) {
            this.hfSplit = v;
            return this;
        }

        public Builder hfName(String v) {
            this.hfName = v;
            return this;
        }

        public Builder hfOutputLen(Integer v) {
            this.hfOutputLen = v;
            return this;
        }

        public Builder specBenchOutputLen(int v) {
            this.specBenchOutputLen = v;
            return this;
        }

        public Builder specBenchCategory(String v) {
            this.specBenchCategory = v;
            return this;
        }

        public Builder timedTraceChunkHashSize(int v) {
            this.timedTraceChunkHashSize = v;
            return this;
        }

        public Builder timedTraceSecMultiplier(double v) {
            this.timedTraceSecMultiplier = v;
            return this;
        }

        public Builder timedTraceLabels(String timestamp, String inputLength, String outputLength,
                                        String hashIds) {
            this.timedTraceLabelTimestamp = timestamp;
            this.timedTraceLabelInputLength = inputLength;
            this.timedTraceLabelOutputLength = outputLength;
            this.timedTraceLabelHashIds = hashIds;
            return this;
        }

        public Builder selfTimed(boolean v) {
            this.selfTimed = v;
            return this;
        }

        public Builder blazeditDistance(double min, double max) {
            this.blazeditMinDistance = min;
            this.blazeditMaxDistance = max;
            return this;
        }

        public Builder asrAudioLen(double min, double max) {
            this.asrMinAudioLenSec = min;
            this.asrMaxAudioLenSec = max;
            return this;
        }

        public Builder randomMmBaseItemsPerRequest(int v) {
            this.randomMmBaseItemsPerRequest = v;
            return this;
        }

        public Builder randomMmNumMmItemsRangeRatio(double v) {
            this.randomMmNumMmItemsRangeRatio = v;
            return this;
        }

        public Builder randomMmLimitMmPerPrompt(Map<String, Integer> v) {
            this.randomMmLimitMmPerPrompt = v == null ? new LinkedHashMap<>() : new LinkedHashMap<>(v);
            return this;
        }

        public Builder randomMmBucketConfig(Map<String, Double> v) {
            this.randomMmBucketConfig = v == null ? new LinkedHashMap<>() : new LinkedHashMap<>(v);
            return this;
        }

        public Builder noReranker(boolean v) {
            this.noReranker = v;
            return this;
        }

        public Builder randomBatchSize(int v) {
            this.randomBatchSize = v;
            return this;
        }

        public Builder speedBenchDatasetSubset(String v) {
            this.speedBenchDatasetSubset = v;
            return this;
        }

        public Builder speedBenchOutputLen(int v) {
            this.speedBenchOutputLen = v;
            return this;
        }

        public Builder speedBenchCategory(String v) {
            this.speedBenchCategory = v;
            return this;
        }

        public Builder enableMultimodalChat(boolean v) {
            this.enableMultimodalChat = v;
            return this;
        }

        public Builder useBeamSearch(boolean v) {
            this.useBeamSearch = v;
            return this;
        }

        public Builder probeRequestRate(double v) {
            this.probeRequestRate = v;
            return this;
        }

        public Builder tokenizerPath(String v) {
            this.tokenizerPath = v;
            return this;
        }

        public Builder tokenizerMode(String v) {
            this.tokenizerMode = v;
            return this;
        }

        public Builder skipTokenizerInit(boolean v) {
            this.skipTokenizerInit = v;
            return this;
        }

        public Builder profile(boolean v) {
            this.profile = v;
            return this;
        }

        public Builder plotTimeline(boolean v) {
            this.plotTimeline = v;
            return this;
        }

        public Builder timelineItlThresholds(List<Double> v) {
            this.timelineItlThresholds = v == null ? List.of(25.0, 50.0) : new ArrayList<>(v);
            return this;
        }

        public Builder plotDatasetStats(boolean v) {
            this.plotDatasetStats = v;
            return this;
        }

        public Builder chatTemplateKwargs(Map<String, Object> v) {
            this.chatTemplateKwargs = v == null ? new LinkedHashMap<>() : new LinkedHashMap<>(v);
            return this;
        }

        public Builder skipChatTemplate(boolean v) {
            this.skipChatTemplate = v;
            return this;
        }

        public Builder noStream(boolean v) {
            this.noStream = v;
            return this;
        }

        public ServeBenchOptions build() {
            return new ServeBenchOptions(this);
        }
    }

    /**
     * The {@code --random-range-ratio} value: a plain float applied to both lengths, or a JSON object
     * of {@code input} and {@code output} ratios.
     *
     * @throws IllegalArgumentException naming the value when it parses as neither
     */
    public static RangeRatio parseRangeRatio(String value) {
        String trimmed = value.trim();
        try {
            double both = Double.parseDouble(trimmed);
            if (!(both >= 0.0 && both < 1.0)) {
                throw new IllegalArgumentException("the random range ratio must be in [0, 1), got " + both);
            }
            return new RangeRatio(both, both);
        } catch (NumberFormatException ignored) {
            // not a float: the JSON object form follows
        }
        Object parsed;
        try {
            parsed = Json.parse(trimmed, Json.Limits.UNTRUSTED);
        } catch (JsonException e) {
            throw new IllegalArgumentException("the random range ratio is a float or a JSON object of"
                    + " 'input' and 'output' ratios, not '" + value + "'");
        }
        if (!(parsed instanceof Map)) {
            throw new IllegalArgumentException("the random range ratio is a float or a JSON object of"
                    + " 'input' and 'output' ratios, not '" + value + "'");
        }
        Map<?, ?> map = (Map<?, ?>) parsed;
        return new RangeRatio(ratioOf(map, "input", value), ratioOf(map, "output", value));
    }

    /** One ratio of the JSON object form, required and range-checked. */
    private static double ratioOf(Map<?, ?> map, String key, String original) {
        Object v = map.get(key);
        if (!(v instanceof Number)) {
            throw new IllegalArgumentException("the random range ratio object carries a '" + key
                    + "' ratio, not '" + original + "'");
        }
        double ratio = ((Number) v).doubleValue();
        if (!(ratio >= 0.0 && ratio < 1.0)) {
            throw new IllegalArgumentException("the '" + key + "' range ratio must be in [0, 1), got " + ratio);
        }
        return ratio;
    }

    /** The request rate: a positive float, with Python's {@code inf} spellings accepted. */
    public static double parseRequestRate(String value) {
        String lower = value.trim().toLowerCase(Locale.ROOT);
        if (lower.equals("inf") || lower.equals("infinity")) {
            return Double.POSITIVE_INFINITY;
        }
        double rate;
        try {
            rate = Double.parseDouble(value.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("--request-rate needs a number or 'inf', not '" + value + "'");
        }
        if (!(rate > 0.0) || Double.isNaN(rate)) {
            throw new IllegalArgumentException("--request-rate must be positive, got " + rate);
        }
        return rate;
    }
}
