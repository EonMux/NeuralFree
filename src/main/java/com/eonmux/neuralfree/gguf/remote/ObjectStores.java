package com.eonmux.neuralfree.gguf.remote;

/**
 * Builds the {@link ObjectStore} an {@link ObjectStoreUri} names, reading its endpoint and credentials from system
 * properties first and the environment second. The property and variable names follow vLLM's and the SDKs': S3 uses
 * {@code AWS_ENDPOINT_URL} / {@code S3_ENDPOINT_URL}, {@code AWS_REGION} and the {@code AWS_ACCESS_KEY_ID} /
 * {@code AWS_SECRET_ACCESS_KEY} pair (or the tensorizer names {@code S3_ACCESS_KEY_ID} /
 * {@code S3_SECRET_ACCESS_KEY}); GCS uses a {@code GCS_ACCESS_TOKEN} or {@code GOOGLE_OAUTH_ACCESS_TOKEN} bearer
 * token; Azure uses its endpoint, account, SAS token or account key. Without credentials every store is accessed
 * anonymously, which public buckets accept.
 */
public final class ObjectStores {

    /** The S3 endpoint ({@code AWS_ENDPOINT_URL} and {@code S3_ENDPOINT_URL} follow). */
    public static final String PROP_S3_ENDPOINT = "neuralfree.remote.s3.endpoint";
    /** The S3 signing region ({@code AWS_REGION} and {@code AWS_DEFAULT_REGION} follow). */
    public static final String PROP_S3_REGION = "neuralfree.remote.s3.region";
    /** The GCS JSON API root ({@code https://storage.googleapis.com}). */
    public static final String PROP_GCS_ENDPOINT = "neuralfree.remote.gcs.endpoint";
    /** A GCS bearer token ({@code GCS_ACCESS_TOKEN} and {@code GOOGLE_OAUTH_ACCESS_TOKEN} follow). */
    public static final String PROP_GCS_TOKEN = "neuralfree.remote.gcs.token";
    /** The Azure Blob service endpoint ({@code AZURE_STORAGE_ENDPOINT} follows). */
    public static final String PROP_AZURE_ENDPOINT = "neuralfree.remote.azure.endpoint";
    /** The Azure storage account ({@code AZURE_STORAGE_ACCOUNT} follows). */
    public static final String PROP_AZURE_ACCOUNT = "neuralfree.remote.azure.account";
    /** An Azure shared access signature ({@code AZURE_STORAGE_SAS_TOKEN} follows). */
    public static final String PROP_AZURE_SAS = "neuralfree.remote.azure.sas";

    private ObjectStores() {
    }

    /**
     * The store {@code uri} names, with the credentials of the environment.
     *
     * @throws IllegalArgumentException when the store's endpoint or credentials are incomplete or malformed
     */
    public static ObjectStore open(ObjectStoreUri uri) {
        switch (uri.scheme()) {
            case S3:
                return s3(uri.bucket());
            case GCS:
                return gcs(uri.bucket());
            case AZURE:
                return azure(uri.bucket());
            default:
                throw new IllegalArgumentException("the object store " + uri.scheme() + " is not supported");
        }
    }

    private static ObjectStore s3(String bucket) {
        String region = first(value(PROP_S3_REGION), env("AWS_REGION"), env("AWS_DEFAULT_REGION"), "us-east-1");
        String endpoint = first(value(PROP_S3_ENDPOINT), env("AWS_ENDPOINT_URL"), env("S3_ENDPOINT_URL"),
                "https://s3." + region + ".amazonaws.com");
        String access = first(env("AWS_ACCESS_KEY_ID"), env("S3_ACCESS_KEY_ID"), null);
        String secret = first(env("AWS_SECRET_ACCESS_KEY"), env("S3_SECRET_ACCESS_KEY"), null);
        String session = first(env("AWS_SESSION_TOKEN"), null);
        return new S3ObjectStore(endpoint, bucket, region, access, secret, session);
    }

    private static ObjectStore gcs(String bucket) {
        String endpoint = first(value(PROP_GCS_ENDPOINT), "https://storage.googleapis.com");
        String token = first(value(PROP_GCS_TOKEN), env("GCS_ACCESS_TOKEN"), env("GOOGLE_OAUTH_ACCESS_TOKEN"), null);
        return new GcsObjectStore(endpoint, bucket, token);
    }

    private static ObjectStore azure(String container) {
        String account = first(value(PROP_AZURE_ACCOUNT), env("AZURE_STORAGE_ACCOUNT"), null);
        String configured = first(value(PROP_AZURE_ENDPOINT), env("AZURE_STORAGE_ENDPOINT"), null);
        String endpoint;
        if (configured != null) {
            endpoint = configured;
        } else if (account != null) {
            endpoint = "https://" + account + ".blob.core.windows.net";
        } else {
            throw new IllegalArgumentException("Azure Blob Storage needs " + PROP_AZURE_ACCOUNT
                    + " or AZURE_STORAGE_ACCOUNT (or " + PROP_AZURE_ENDPOINT + ") to name its endpoint");
        }
        String key = first(env("AZURE_STORAGE_KEY"), env("AZURE_STORAGE_ACCESS_KEY"), null);
        String sas = first(value(PROP_AZURE_SAS), env("AZURE_STORAGE_SAS_TOKEN"), null);
        return new AzureBlobObjectStore(endpoint, container, account, key, sas);
    }

    private static String first(String... values) {
        for (String v : values) {
            if (v != null && !v.trim().isEmpty()) {
                return v.trim();
            }
        }
        return null;
    }

    /** A system property, trimmed, or null. */
    private static String value(String property) {
        return System.getProperty(property);
    }

    /** An environment value, trimmed, or null. */
    private static String env(String name) {
        return System.getenv(name);
    }
}
