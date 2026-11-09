package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;

import java.io.IOException;
import java.util.List;

/**
 * The projector file and the encoders bound over it: what a release frees and a reopen loads again.
 */
final class MmprojEncoders {
    final MmprojModel mmproj;
    final VisionEncoder vision;
    final CpuAudioEncoder audio;

    private MmprojEncoders(MmprojModel mmproj, VisionEncoder vision, CpuAudioEncoder audio) {
        this.mmproj = mmproj;
        this.vision = vision;
        this.audio = audio;
    }

    /**
     * Opens the projector at {@code path} and its encoders, each checked against the text width.
     *
     * @throws IllegalArgumentException when the file holds no encoder or a width differs from the text model's
     */
    static MmprojEncoders open(String path, MmprojOptions options, int textEmbd,
                               List<LoraAttachment.Binding> loras) throws IOException {
        MmprojModel mm = MmprojModel.open(path);
        VisionEncoder vision = null;
        CpuAudioEncoder audio = null;
        try {
            if (!mm.supportsVision() && !mm.supportsAudio() && !mm.supportsGenAudio()) {
                throw new IllegalArgumentException("the projector " + path
                        + " has no vision, audio or generation encoder");
            }
            if (mm.supportsVision()) {
                vision = mm.openVisionEncoder(options);
                requireWidth("vision", vision.model().outputDim(), textEmbd, path);
                vision.attachLoras(loras);
            }
            if (mm.supportsAudio()) {
                audio = mm.openAudioEncoder();
                requireWidth("audio", audio.model().outputWidth(), textEmbd, path);
            }
            return new MmprojEncoders(mm, vision, audio);
        } catch (IOException | RuntimeException e) {
            new MmprojEncoders(mm, vision, audio).close();
            throw e;
        }
    }

    private static void requireWidth(String what, int projector, int text, String path) {
        if (projector != text) {
            throw new IllegalArgumentException("mismatch between the text model (n_embd = " + text + ") and the "
                    + what + " projector of " + path + " (n_embd = " + projector + "); the mmproj file does not"
                    + " belong to this model");
        }
    }

    /** Closes the encoders, then the projector file. */
    void close() throws IOException {
        if (vision != null) {
            vision.close();
        }
        if (audio != null) {
            audio.close();
        }
        mmproj.close();
    }
}
