package com.eonmux.neuralfree.acceleration.cache;

import java.util.Objects;

public final class CacheKey {
    private final String kernelName;
    private final String sourceHash;
    private final String buildOptions;
    private final int cachedHashCode;

    public CacheKey(String kernelName, String sourceHash, String buildOptions) {
        this.kernelName = Objects.requireNonNull(kernelName, "kernelName must not be null");
        this.sourceHash = Objects.requireNonNull(sourceHash, "sourceHash must not be null");
        this.buildOptions = buildOptions != null ? buildOptions : "";
        this.cachedHashCode = computeHashCode();
    }

    public String kernelName() {
        return kernelName;
    }

    public String sourceHash() {
        return sourceHash;
    }

    public String buildOptions() {
        return buildOptions;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        CacheKey cacheKey = (CacheKey) o;
        return kernelName.equals(cacheKey.kernelName) &&
               sourceHash.equals(cacheKey.sourceHash) &&
               buildOptions.equals(cacheKey.buildOptions);
    }

    @Override
    public int hashCode() {
        return cachedHashCode;
    }

    private int computeHashCode() {
        return Objects.hash(kernelName, sourceHash, buildOptions);
    }

    @Override
    public String toString() {
        return "CacheKey{" +
               "kernelName='" + kernelName + '\'' +
               ", sourceHash='" + sourceHash + '\'' +
               ", buildOptions='" + buildOptions + '\'' +
               '}';
    }
}
