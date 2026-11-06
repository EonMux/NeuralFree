package com.eonmux.neuralfree.gguf.media;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The input arguments of an {@code ffmpeg} or {@code ffprobe} run over client-supplied media: the protocol
 * whitelist, the demuxer whitelist of the media kind, and the {@code -i} input. A staged file is read through the
 * {@code file} protocol by the containers of its kind; bytes on standard input are read through the {@code pipe}
 * protocol. The whitelists stop a crafted upload, such as an HLS playlist or an ffconcat script posing as media,
 * from making the process open other files or URLs. An {@code ffmpeg} run over a staged file also gets
 * {@code -nostdin}, since nothing may flow through its standard input; {@code ffprobe} has no such option.
 */
public final class FfmpegInput {

    /** The media kind of a run, selecting the demuxer whitelist. */
    public enum Media {
        VIDEO, AUDIO, IMAGE
    }

    /** The input location of a run over the client's bytes on standard input. */
    public static final String STDIN = "pipe:0";

    /** The containers the video decoder accepts, the set llama.cpp's {@code mtmd} helper reads. */
    private static final String VIDEO_DEMUXERS =
            "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi,flv,mpegts";
    /** The containers the audio decoder forwards to ffmpeg; WAV, AIFF and AU decode in-process. */
    private static final String AUDIO_DEMUXERS =
            "wav,aiff,au,mp3,flac,ogg,mov,mp4,m4a,3gp,3g2,mj2,matroska,webm";
    /** The still-image demuxers the image decoder forwards to ffmpeg: WebP, still and animated. */
    private static final String IMAGE_DEMUXERS = "webp_pipe,webp_anim,image2,image2pipe";

    private FfmpegInput() {
    }

    /**
     * The input arguments of an {@code ffmpeg} run over a staged file: {@code -nostdin}, the {@code file} protocol
     * only, the demuxers of {@code media}, and {@code -i file}.
     */
    public static List<String> ffmpegFile(Media media, Path file) {
        return arguments(media, "file", Objects.requireNonNull(file, "file").toString(), true);
    }

    /**
     * The input arguments of an {@code ffprobe} run over a staged file: the {@code file} protocol only, the
     * demuxers of {@code media}, and {@code -i file}. ffprobe has no {@code -nostdin} option.
     */
    public static List<String> ffprobeFile(Media media, Path file) {
        return arguments(media, "file", Objects.requireNonNull(file, "file").toString(), false);
    }

    /**
     * The input arguments of an {@code ffmpeg} run over the client's bytes on standard input: the {@code pipe}
     * protocol only, the demuxers of {@code media}, and {@code -i pipe:0}.
     */
    public static List<String> ffmpegStdin(Media media) {
        return arguments(media, "pipe", STDIN, false);
    }

    private static List<String> arguments(Media media, String protocol, String location, boolean noStdin) {
        Objects.requireNonNull(media, "media");
        List<String> arguments = new ArrayList<>();
        if (noStdin) {
            arguments.add("-nostdin");
        }
        arguments.add("-protocol_whitelist");
        arguments.add(protocol);
        arguments.add("-format_whitelist");
        arguments.add(demuxers(media));
        arguments.add("-i");
        arguments.add(location);
        return Collections.unmodifiableList(arguments);
    }

    private static String demuxers(Media media) {
        switch (media) {
            case VIDEO:
                return VIDEO_DEMUXERS;
            case AUDIO:
                return AUDIO_DEMUXERS;
            case IMAGE:
                return IMAGE_DEMUXERS;
            default:
                throw new IllegalArgumentException("unknown media kind " + media);
        }
    }
}
