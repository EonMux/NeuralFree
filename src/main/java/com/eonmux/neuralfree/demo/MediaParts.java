package com.eonmux.neuralfree.demo;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * OpenAI multimodal content parts built from local files (tools/cli/cli-context.cpp
 * {@code stage_media_file}): an image as a {@code image_url} data URI, audio as {@code input_audio} with its
 * format, video as {@code input_video}. The command of a file follows its extension
 * ({@code media_type_from_ext}); an unreadable file is refused naming it.
 */
final class MediaParts {

    private MediaParts() {
    }

    /**
     * The content part of {@code fileName} for the media command {@code kind}.
     *
     * @throws IllegalArgumentException for a kind other than {@code image}, {@code audio} or {@code video}
     * @throws IOException naming the file when it cannot be read
     */
    static Object part(String kind, String fileName) throws IOException {
        switch (kind) {
            case "image": return imagePart(fileName);
            case "audio": return audioPart(fileName);
            case "video": return videoPart(fileName);
            default: throw new IllegalArgumentException("media kind '" + kind + "' is not image, audio or video");
        }
    }

    /** The media command a file belongs to: {@code audio}, {@code video} or {@code image}. */
    static String kindFromName(String fileName) {
        String ext = extension(fileName);
        if (ext.equals(".wav") || ext.equals(".mp3")) {
            return "audio";
        }
        if (ext.equals(".mp4") || ext.equals(".avi") || ext.equals(".mkv") || ext.equals(".mov")
                || ext.equals(".webm")) {
            return "video";
        }
        return "image";
    }

    /**
     * An {@code image_url} part with the file's bytes base64-encoded into a data URI. The MIME type comes from
     * the image's magic bytes; like llama.cpp's {@code data:image/unknown;base64}, an unrecognized image is sent
     * as {@code image/unknown} and the server detects the type from the data.
     */
    static Object imagePart(String fileName) throws IOException {
        byte[] data = read(fileName);
        String uri = "data:" + imageMime(data) + ";base64," + Base64.getEncoder().encodeToString(data);
        return part("image_url", "url", uri);
    }

    /** An {@code input_audio} part: the file's bytes base64-encoded, with the {@code wav} or {@code mp3} format. */
    static Object audioPart(String fileName) throws IOException {
        byte[] data = read(fileName);
        String format = extension(fileName).equals(".mp3") ? "mp3" : "wav";
        Map<String, Object> audio = new LinkedHashMap<>();
        audio.put("data", Base64.getEncoder().encodeToString(data));
        audio.put("format", format);
        Map<String, Object> part = new LinkedHashMap<>();
        part.put("type", "input_audio");
        part.put("input_audio", audio);
        return part;
    }

    /** An {@code input_video} part with the file's bytes base64-encoded. */
    static Object videoPart(String fileName) throws IOException {
        byte[] data = read(fileName);
        return part("input_video", "data", Base64.getEncoder().encodeToString(data));
    }

    private static Object part(String type, String field, String value) {
        Map<String, Object> inner = new LinkedHashMap<>();
        inner.put(field, value);
        Map<String, Object> part = new LinkedHashMap<>();
        part.put("type", type);
        part.put(type, inner);
        return part;
    }

    private static byte[] read(String fileName) throws IOException {
        Path path = Paths.get(fileName);
        if (!Files.isRegularFile(path)) {
            throw new IOException("file does not exist or cannot be opened: '" + fileName + "'");
        }
        return Files.readAllBytes(path);
    }

    private static String imageMime(byte[] data) {
        if (data.length >= 8 && (data[0] & 0xFF) == 0x89 && data[1] == 'P' && data[2] == 'N' && data[3] == 'G') {
            return "image/png";
        }
        if (data.length >= 3 && (data[0] & 0xFF) == 0xFF && (data[1] & 0xFF) == 0xD8 && (data[2] & 0xFF) == 0xFF) {
            return "image/jpeg";
        }
        if (data.length >= 6 && data[0] == 'G' && data[1] == 'I' && data[2] == 'F') {
            return "image/gif";
        }
        if (data.length >= 12 && data[0] == 'R' && data[1] == 'I' && data[2] == 'F' && data[3] == 'F'
                && data[8] == 'W' && data[9] == 'E' && data[10] == 'B' && data[11] == 'P') {
            return "image/webp";
        }
        if (data.length >= 2 && data[0] == 'B' && data[1] == 'M') {
            return "image/bmp";
        }
        return "image/unknown";
    }

    private static String extension(String fileName) {
        String name = Paths.get(fileName).getFileName().toString();
        int dot = name.lastIndexOf('.');
        return dot < 0 ? "" : name.substring(dot).toLowerCase(Locale.ROOT);
    }
}
