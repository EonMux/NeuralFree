package com.eonmux.neuralfree.demo.tools.eval;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * The settings of one task-accuracy run: the endpoints and their concurrency, the dataset, the
 * sampling, the retry policy, the grader (and the endpoint it uses), the state file and the resume
 * flag. Immutable; the builder carries the reference's defaults.
 */
public final class EvalOptions {

    /** The state file a run writes when {@code --output} is not given. */
    public static final String DEFAULT_OUTPUT = "neuralfree-eval-state.json";

    private final List<EvalServer> servers;
    private final String dataset;
    private final String datasetSource;
    private final Integer nCases;
    private final long seed;
    private final EvalSampling sampling;
    private final int retries;
    private final double retryDelay;
    private final String model;
    private final boolean verbose;
    private final Path output;
    private final String graderType;
    private final String graderScript;
    private final String graderServer;
    private final boolean graderServerExplicit;
    private final String graderModel;
    private final String graderApiKey;
    private final boolean resume;

    private EvalOptions(Builder b) {
        this.servers = new ArrayList<>(b.servers);
        this.dataset = b.dataset;
        this.datasetSource = b.datasetSource;
        this.nCases = b.nCases;
        this.seed = b.seed;
        this.sampling = b.sampling;
        this.retries = b.retries;
        this.retryDelay = b.retryDelay;
        this.model = b.model;
        this.verbose = b.verbose;
        this.output = b.output;
        this.graderType = b.graderType;
        this.graderScript = b.graderScript;
        this.graderServer = b.graderServer;
        this.graderServerExplicit = b.graderServerExplicit;
        this.graderModel = b.graderModel;
        this.graderApiKey = b.graderApiKey;
        this.resume = b.resume;
    }

    public static Builder builder() {
        return new Builder();
    }

    public List<EvalServer> servers() {
        return new ArrayList<>(servers);
    }

    public String dataset() {
        return dataset;
    }

    /** The {@code --dataset-source} override, null for the canonical repository. */
    public String datasetSource() {
        return datasetSource;
    }

    /** The requested case count, null for the whole suite once. */
    public Integer nCases() {
        return nCases;
    }

    public long seed() {
        return seed;
    }

    public EvalSampling sampling() {
        return sampling;
    }

    public int retries() {
        return retries;
    }

    public double retryDelay() {
        return retryDelay;
    }

    /** The model name the requests carry, null for the reference's {@code llama}. */
    public String model() {
        return model;
    }

    public boolean verbose() {
        return verbose;
    }

    public Path output() {
        return output;
    }

    public String graderType() {
        return graderType;
    }

    /** The CLI grader script, null for the other graders. */
    public String graderScript() {
        return graderScript;
    }

    /** The LLM grader's endpoint URL. */
    public String graderServer() {
        return graderServer;
    }

    /** Whether {@code --grader-server} was given rather than defaulted. */
    public boolean graderServerExplicit() {
        return graderServerExplicit;
    }

    /** The LLM grader's model name. */
    public String graderModel() {
        return graderModel;
    }

    /** The LLM grader's bearer token, null when the grader endpoint needs none. */
    public String graderApiKey() {
        return graderApiKey;
    }

    public boolean resume() {
        return resume;
    }

    /** The builder of the options, the reference's defaults already set. */
    public static final class Builder {

        private List<EvalServer> servers = new ArrayList<>();
        private String dataset = "aime";
        private String datasetSource;
        private Integer nCases;
        private long seed = 1234L;
        private EvalSampling sampling = new EvalSampling(null, null, null, null, null, -1);
        private int retries = 5;
        private double retryDelay = 10.0;
        private String model;
        private boolean verbose;
        private Path output = Paths.get(DEFAULT_OUTPUT);
        private String graderType = "llm";
        private String graderScript;
        private String graderServer = "";
        private boolean graderServerExplicit;
        private String graderModel = "";
        private String graderApiKey;
        private boolean resume;

        public Builder servers(List<EvalServer> value) {
            this.servers = new ArrayList<>(value);
            return this;
        }

        public Builder dataset(String value) {
            this.dataset = value;
            return this;
        }

        public Builder datasetSource(String value) {
            this.datasetSource = value;
            return this;
        }

        public Builder nCases(Integer value) {
            this.nCases = value;
            return this;
        }

        public Builder seed(long value) {
            this.seed = value;
            return this;
        }

        public Builder sampling(EvalSampling value) {
            this.sampling = value;
            return this;
        }

        public Builder retries(int value) {
            this.retries = value;
            return this;
        }

        public Builder retryDelay(double value) {
            this.retryDelay = value;
            return this;
        }

        public Builder model(String value) {
            this.model = value;
            return this;
        }

        public Builder verbose(boolean value) {
            this.verbose = value;
            return this;
        }

        public Builder output(Path value) {
            this.output = value;
            return this;
        }

        public Builder graderType(String value) {
            this.graderType = value;
            return this;
        }

        public Builder graderScript(String value) {
            this.graderScript = value;
            return this;
        }

        public Builder graderServer(String value) {
            this.graderServer = value;
            return this;
        }

        public Builder graderServerExplicit(boolean value) {
            this.graderServerExplicit = value;
            return this;
        }

        public Builder graderModel(String value) {
            this.graderModel = value;
            return this;
        }

        public Builder graderApiKey(String value) {
            this.graderApiKey = value;
            return this;
        }

        public Builder resume(boolean value) {
            this.resume = value;
            return this;
        }

        /**
         * @throws IllegalArgumentException naming an unknown dataset or grader, a missing server,
         *                                  script or model, or an out-of-range count
         */
        public EvalOptions build() {
            if (servers.isEmpty()) {
                throw new IllegalArgumentException("--server must name at least one server");
            }
            if (!EvalDatasets.NAMES.contains(dataset)) {
                throw new IllegalArgumentException("--dataset is one of " + EvalDatasets.NAMES
                        + ", not '" + dataset + "'");
            }
            if (!EvalGrader.TYPES.contains(graderType)) {
                throw new IllegalArgumentException("--grader-type is one of " + EvalGrader.TYPES
                        + ", not '" + graderType + "'");
            }
            if ("cli".equals(graderType) && (graderScript == null || graderScript.isEmpty())) {
                throw new IllegalArgumentException("--grader-type cli requires --grader-script");
            }
            if ("llm".equals(graderType) && (graderModel == null || graderModel.isEmpty())) {
                throw new IllegalArgumentException(
                        "--grader-type llm requires --grader-model or --model");
            }
            if (nCases != null && nCases < 1) {
                throw new IllegalArgumentException("--n_cases must be at least 1, got " + nCases);
            }
            if (retries < 0) {
                throw new IllegalArgumentException("--retries must be at least 0, got " + retries);
            }
            if (retryDelay < 0) {
                throw new IllegalArgumentException("--retry-delay must be at least 0, got "
                        + retryDelay);
            }
            if (output == null) {
                throw new IllegalArgumentException("--output must name a state file");
            }
            if (graderServer == null || graderServer.isEmpty()) {
                graderServer = servers.get(0).url();
            }
            return new EvalOptions(this);
        }
    }
}
