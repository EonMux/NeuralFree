package com.eonmux.neuralfree.gguf.video;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * The video input settings of a projector context, llama.cpp's {@code mtmd_helper_video_init_params}
 * (tools/mtmd/mtmd-helper.h) and vLLM's video pruning configuration (vllm/config/multimodal.py
 * {@code video_pruning_rate} / {@code video_pruning_method}): the directory of the ffmpeg and ffprobe binaries,
 * the sampling rate a model without its own recommendation samples at, the interval of the default template's
 * per-frame timestamps, and the video token pruning. A pruning rate of 0 (the default) prunes nothing; a rate in
 * {@code (0, 1)} keeps {@code 1 - rate} of every video's embedding rows. Immutable; built with {@link #builder()}.
 */
public final class VideoOptions {

    /** llama.cpp's helper default: frames a model without a recommendation samples at. */
    public static final double DEFAULT_FPS_TARGET = 4.0;
    /** llama.cpp's helper default: an {@code [XmY.YYs]} stamp every 5000 ms. */
    public static final int DEFAULT_TIMESTAMP_INTERVAL_MS = 5000;
    /** vLLM's default pruning method when a rate is set. */
    public static final VideoPruner.Method DEFAULT_PRUNING_METHOD = VideoPruner.Method.EVS;

    private final Path ffmpegDir;
    private final double fpsTarget;
    private final int timestampIntervalMs;
    private final double pruningRate;
    private final VideoPruner.Method pruningMethod;

    private VideoOptions(Builder b) {
        this.ffmpegDir = b.ffmpegDir;
        this.fpsTarget = b.fpsTarget;
        this.timestampIntervalMs = b.timestampIntervalMs;
        this.pruningRate = b.pruningRate;
        this.pruningMethod = b.pruningMethod;
    }

    /** llama.cpp's helper defaults and no pruning. */
    public static VideoOptions defaults() {
        return builder().build();
    }

    /** A builder at the defaults. */
    public static Builder builder() {
        return new Builder();
    }

    /** The directory the ffmpeg and ffprobe binaries are taken from, or null for the PATH. */
    public Path ffmpegDir() {
        return ffmpegDir;
    }

    /** The frames-per-second rate a video samples at when neither the model nor the request names one. */
    public double fpsTarget() {
        return fpsTarget;
    }

    /** The interval in milliseconds of the default template's {@code [XmY.YYs]} stamps; 0 stamps none. */
    public int timestampIntervalMs() {
        return timestampIntervalMs;
    }

    /** The fraction of every video's embedding rows pruning drops; 0 prunes nothing. */
    public double pruningRate() {
        return pruningRate;
    }

    /** The pruning algorithm a positive rate applies. */
    public VideoPruner.Method pruningMethod() {
        return pruningMethod;
    }

    /** Whether video token pruning runs. */
    public boolean prunes() {
        return pruningRate > 0;
    }

    /** Builder of {@link VideoOptions}. */
    public static final class Builder {
        private Path ffmpegDir;
        private double fpsTarget = DEFAULT_FPS_TARGET;
        private int timestampIntervalMs = DEFAULT_TIMESTAMP_INTERVAL_MS;
        private double pruningRate = 0;
        private VideoPruner.Method pruningMethod = DEFAULT_PRUNING_METHOD;

        private Builder() {
        }

        /**
         * The directory the ffmpeg and ffprobe binaries are taken from, or null for the PATH (llama.cpp's
         * {@code --video-ffmpeg-dir}).
         *
         * @throws IllegalArgumentException when the path is not an existing directory
         */
        public Builder ffmpegDir(Path dir) {
            if (dir == null) {
                this.ffmpegDir = null;
                return this;
            }
            if (!Files.isDirectory(dir)) {
                throw new IllegalArgumentException("the video ffmpeg directory " + dir + " must be an existing"
                        + " directory");
            }
            this.ffmpegDir = dir;
            return this;
        }

        /**
         * The sampling rate a video falls back to when neither the model nor the request names one.
         *
         * @throws IllegalArgumentException when the rate is negative or not finite
         */
        public Builder fpsTarget(double fps) {
            if (!(fps >= 0) || Double.isNaN(fps) || Double.isInfinite(fps)) {
                throw new IllegalArgumentException("the video sampling rate must not be negative, got " + fps);
            }
            this.fpsTarget = fps;
            return this;
        }

        /**
         * The interval in milliseconds of the default template's {@code [XmY.YYs]} stamps.
         *
         * @throws IllegalArgumentException when the interval is negative
         */
        public Builder timestampIntervalMs(int ms) {
            if (ms < 0) {
                throw new IllegalArgumentException("the video timestamp interval must not be negative, got " + ms);
            }
            this.timestampIntervalMs = ms;
            return this;
        }

        /**
         * The fraction of every video's embedding rows pruning drops; 0 (the default) prunes nothing.
         *
         * @throws IllegalArgumentException when the rate is outside {@code [0, 1)}
         */
        public Builder pruningRate(double rate) {
            if (!(rate >= 0) || rate >= 1 || Double.isNaN(rate) || Double.isInfinite(rate)) {
                throw new IllegalArgumentException("the video pruning rate must be in [0, 1), got " + rate);
            }
            this.pruningRate = rate;
            return this;
        }

        /** The pruning algorithm a positive rate applies. */
        public Builder pruningMethod(VideoPruner.Method method) {
            this.pruningMethod = Objects.requireNonNull(method, "method");
            return this;
        }

        /** The options. */
        public VideoOptions build() {
            return new VideoOptions(this);
        }
    }
}
