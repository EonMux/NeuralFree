package com.eonmux.neuralfree.gguf.hub;

import java.util.Objects;

/**
 * A Hugging Face Hub model reference, parsed from the {@code -hf} expression
 * {@code <user>/<model>[:quant][@revision]} the way llama.cpp's downloader splits
 * {@code hf_repo_with_tag} ({@code common/download.cpp:42-95}): the repository id before the first
 * {@code :}, the quant tag after it (never empty), and the revision after the {@code @} (a branch
 * name or a commit hash). {@link #parseDataset} reads the same expression for a dataset repository,
 * which carries no quant tag. The mirror of llama.cpp's {@code @revision} is the {@code hft}
 * tool surface. The repository id rules are llama.cpp's {@code is_valid_repo_id}:
 * up to 256 chars, exactly one {@code /}, and its special chars ({@code / . -}) only between the
 * base chars ({@code A-Za-z0-9_}). Immutable.
 */
public final class HubRef {

    /**
     * What a repository holds, selecting the Hub API namespace, the resolve URL prefix and the cache
     * folder prefix: a model repository ({@code api/models}, {@code /user/model}) or a dataset
     * repository ({@code api/datasets}, {@code /datasets/user/model}).
     */
    public enum RepoType {
        MODEL("models", "", "models--"),
        DATASET("datasets", "datasets/", "datasets--");

        private final String apiSegment;
        private final String resolvePrefix;
        private final String folderPrefix;

        RepoType(String apiSegment, String resolvePrefix, String folderPrefix) {
            this.apiSegment = apiSegment;
            this.resolvePrefix = resolvePrefix;
            this.folderPrefix = folderPrefix;
        }

        /** The Hub API segment: {@code models} or {@code datasets}. */
        public String apiSegment() {
            return apiSegment;
        }

        /** The path prefix of the resolve URL: empty for a model, {@code datasets/} for a dataset. */
        public String resolvePrefix() {
            return resolvePrefix;
        }

        /** The cache folder prefix: {@code models--} or {@code datasets--}. */
        public String folderPrefix() {
            return folderPrefix;
        }
    }

    private final String repository;
    private final String quant;
    private final String revision;
    private final RepoType type;

    private HubRef(String repository, String quant, String revision, RepoType type) {
        this.repository = repository;
        this.quant = quant;
        this.revision = revision;
        this.type = type;
    }

    /**
     * Parses {@code user/repo[:quant][@revision]} into a model {@link HubRef}; anything else is
     * rejected with a message that names the expected format.
     */
    public static HubRef parse(String spec) {
        return parse(spec, RepoType.MODEL);
    }

    /**
     * Parses {@code user/repo[@revision]} into a dataset {@link HubRef}: a dataset repository has no
     * quant tag, so the {@code :quant} form is rejected.
     */
    public static HubRef parseDataset(String spec) {
        return parse(spec, RepoType.DATASET);
    }

    private static HubRef parse(String spec, RepoType type) {
        String s = spec == null ? "" : spec.trim();
        if (s.isEmpty()) {
            throw new IllegalArgumentException("invalid HF repo format, expected <user>/<model>[:quant][@revision]");
        }
        String revision = null;
        int at = s.lastIndexOf('@');
        if (at >= 0) {
            revision = s.substring(at + 1);
            s = s.substring(0, at);
            if (revision.isEmpty()) {
                throw new IllegalArgumentException("invalid HF repo format, expected <user>/<model>[:quant][@revision]:"
                        + " the revision after '@' is empty: " + spec);
            }
            if (!isRevision(revision)) {
                throw new IllegalArgumentException("invalid HF repo revision '" + revision
                        + "': use A-Za-z0-9._- only");
            }
        }
        String repo = s;
        String quant = null;
        int colon = s.indexOf(':');
        if (colon >= 0) {
            if (type == RepoType.DATASET) {
                throw new IllegalArgumentException("invalid HF dataset format, expected <user>/<name>[@revision]:"
                        + " a dataset repository has no quant tag: " + spec);
            }
            repo = s.substring(0, colon);
            quant = s.substring(colon + 1);
            if (quant.isEmpty()) {
                throw new IllegalArgumentException("invalid HF repo format, expected <user>/<model>[:quant][@revision]:"
                        + " the quant after ':' is empty: " + spec);
            }
            if (!isQuant(quant)) {
                throw new IllegalArgumentException("invalid HF repo quant '" + quant
                        + "': use A-Za-z0-9._- only");
            }
        }
        if (!isValidRepoId(repo)) {
            throw new IllegalArgumentException("invalid HF repo format, expected <user>/<model>[:quant][@revision]: "
                    + spec);
        }
        return new HubRef(repo, quant, revision, type);
    }

    /** The repository id, {@code user/model}. */
    public String repository() {
        return repository;
    }

    /** What the repository holds: a model or a dataset. */
    public RepoType type() {
        return type;
    }

    /** The user part of the repository id. */
    public String user() {
        return repository.substring(0, repository.indexOf('/'));
    }

    /** The model part of the repository id. */
    public String model() {
        return repository.substring(repository.indexOf('/') + 1);
    }

    /** The quant tag ({@code Q4_K_M}), or null when the expression carries none. */
    public String quant() {
        return quant;
    }

    /** The revision (a branch name or a commit), or null for the Hub's default revision. */
    public String revision() {
        return revision;
    }

    /** The cache folder name llama.cpp derives from a repository id: {@code models--user--model}. */
    public String folderName() {
        return type.folderPrefix() + repository.replace("/", "--");
    }

    /** The normalized expression: {@code [datasets/]user/repo[:quant][@revision]}. */
    @Override
    public String toString() {
        StringBuilder out = new StringBuilder();
        if (type == RepoType.DATASET) {
            out.append("datasets/");
        }
        out.append(repository);
        if (quant != null) {
            out.append(':').append(quant);
        }
        if (revision != null) {
            out.append('@').append(revision);
        }
        return out.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof HubRef)) {
            return false;
        }
        HubRef other = (HubRef) o;
        return type == other.type
                && repository.equals(other.repository)
                && Objects.equals(quant, other.quant)
                && Objects.equals(revision, other.revision);
    }

    @Override
    public int hashCode() {
        return Objects.hash(repository, quant, revision, type);
    }

    /**
     * llama.cpp's {@code is_valid_repo_id}: non-empty, at most 256 chars, exactly one {@code /}, and
     * the chars {@code / . -} only between base chars ({@code A-Za-z0-9_}), so no leading, trailing
     * or doubled special char.
     */
    private static boolean isValidRepoId(String repo) {
        if (repo.isEmpty() || repo.length() > 256) {
            return false;
        }
        int slashes = 0;
        boolean lastWasSpecial = true;
        for (int i = 0; i < repo.length(); i++) {
            char c = repo.charAt(i);
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_') {
                lastWasSpecial = false;
            } else if (c == '/' || c == '.' || c == '-') {
                if (lastWasSpecial) {
                    return false;
                }
                if (c == '/') {
                    slashes++;
                }
                lastWasSpecial = true;
            } else {
                return false;
            }
        }
        return !lastWasSpecial && slashes == 1;
    }

    /** A revision name: static path chars only, no {@code ..} segment, at most 256 chars. */
    private static boolean isRevision(String revision) {
        if (revision.isEmpty() || revision.length() > 256) {
            return false;
        }
        for (int i = 0; i < revision.length(); i++) {
            char c = revision.charAt(i);
            boolean ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
                    || c == '.' || c == '-' || c == '_';
            if (!ok) {
                return false;
            }
        }
        return !revision.equals(".") && !revision.equals("..");
    }

    /** A quant tag: static file-name chars only, at most 128 chars. */
    private static boolean isQuant(String quant) {
        if (quant.length() > 128) {
            return false;
        }
        for (int i = 0; i < quant.length(); i++) {
            char c = quant.charAt(i);
            boolean ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
                    || c == '.' || c == '-' || c == '_';
            if (!ok) {
                return false;
            }
        }
        return true;
    }
}
