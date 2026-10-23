package com.eonmux.neuralfree.nn.selfsupervised;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.Sequential;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.activations.ReLU;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.Test;
import static org.junit.Assert.*;

public class SelfSupervisedTest {

    @Test
    public void testContrastiveLearnerCreation() {
        Module encoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        ContrastiveLearner learner = new ContrastiveLearner(encoder, 128, 64, 0.07f);
        
        assertEquals(128, learner.embeddingDim());
        assertEquals(64, learner.projectionDim());
        assertEquals(0.07f, learner.temperature(), 0.001f);
        assertNotNull(learner.encoder());
        assertNotNull(learner.projectionHead());
        
        learner.close();
    }

    @Test
    public void testContrastiveLearnerInvalidParams() {
        Module encoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        
        try {
            new ContrastiveLearner(null, 128, 64);
            fail("Should throw for null encoder");
        } catch (NullPointerException e) {
        }
        
        try {
            new ContrastiveLearner(encoder, 0, 64);
            fail("Should throw for non-positive embedding dim");
        } catch (IllegalArgumentException e) {
        }
        
        try {
            new ContrastiveLearner(encoder, 128, 0);
            fail("Should throw for non-positive projection dim");
        } catch (IllegalArgumentException e) {
        }
        
        try {
            new ContrastiveLearner(encoder, 128, 64, 0);
            fail("Should throw for non-positive temperature");
        } catch (IllegalArgumentException e) {
        }
        
        encoder.close();
    }

    @Test
    public void testContrastiveLearnerAccessors() {
        Module encoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        ContrastiveLearner learner = new ContrastiveLearner(encoder, 256, 128, 0.1f);
        
        assertEquals(256, learner.embeddingDim());
        assertEquals(128, learner.projectionDim());
        assertEquals(0.1f, learner.temperature(), 0.001f);
        
        learner.close();
    }

    @Test
    public void testContrastiveLearnerToString() {
        Module encoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        ContrastiveLearner learner = new ContrastiveLearner(encoder, 128, 64, 0.07f);
        
        String str = learner.toString();
        assertTrue(str.contains("ContrastiveLearner"));
        assertTrue(str.contains("128"));
        assertTrue(str.contains("64"));
        assertTrue(str.contains("0.07"));
        
        learner.close();
        
        str = learner.toString();
        assertTrue(str.contains("closed"));
    }

    @Test
    public void testMaskedAutoencoderCreation() {
        Module encoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        Module decoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        
        MaskedAutoencoder mae = new MaskedAutoencoder(encoder, decoder, 4, 8, 8, 3, 0.75f);
        
        assertEquals(4, mae.patchSize());
        assertEquals(8, mae.imageHeight());
        assertEquals(8, mae.imageWidth());
        assertEquals(3, mae.channels());
        assertEquals(0.75f, mae.maskRatio(), 0.001f);
        assertEquals(4, mae.numPatches());
        
        mae.close();
    }

    @Test
    public void testMAEMaskCreation() {
        Module encoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        Module decoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        
        MaskedAutoencoder mae = new MaskedAutoencoder(encoder, decoder, 4, 8, 8, 3, 0.75f);
        mae.setRandom(new Random(42));
        
        MaskedAutoencoder.MAEMask mask = mae.createMask();
        
        int expectedMasked = (int) (mae.numPatches() * 0.75f);
        assertEquals(expectedMasked, mask.numMasked());
        assertEquals(mae.numPatches() - expectedMasked, mask.numVisible());
        assertEquals(0.75f, mask.maskRatio(), 0.001f);
        
        mae.close();
    }

    @Test
    public void testMAENumPatches() {
        Module encoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        Module decoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        
        MaskedAutoencoder mae1 = new MaskedAutoencoder(encoder, decoder, 4, 8, 8, 3);
        assertEquals(4, mae1.numPatches());
        
        MaskedAutoencoder mae2 = new MaskedAutoencoder(encoder, decoder, 16, 32, 32, 3);
        assertEquals(4, mae2.numPatches());
        
        MaskedAutoencoder mae3 = new MaskedAutoencoder(encoder, decoder, 8, 224, 224, 3);
        assertEquals(784, mae3.numPatches());
        
        mae1.close();
        mae2.close();
        mae3.close();
    }

    @Test
    public void testMAEInvalidParams() {
        Module encoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        Module decoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        
        try {
            new MaskedAutoencoder(null, decoder, 4, 8, 8, 3, 0.75f);
            fail("Should throw for null encoder");
        } catch (NullPointerException e) {
        }
        
        try {
            new MaskedAutoencoder(encoder, null, 4, 8, 8, 3, 0.75f);
            fail("Should throw for null decoder");
        } catch (NullPointerException e) {
        }
        
        try {
            new MaskedAutoencoder(encoder, decoder, 0, 8, 8, 3, 0.75f);
            fail("Should throw for non-positive patch size");
        } catch (IllegalArgumentException e) {
        }
        
        try {
            new MaskedAutoencoder(encoder, decoder, 4, 0, 8, 3, 0.75f);
            fail("Should throw for non-positive image height");
        } catch (IllegalArgumentException e) {
        }
        
        try {
            new MaskedAutoencoder(encoder, decoder, 4, 8, 8, 0, 0.75f);
            fail("Should throw for non-positive channels");
        } catch (IllegalArgumentException e) {
        }
        
        try {
            new MaskedAutoencoder(encoder, decoder, 4, 8, 8, 3, 0);
            fail("Should throw for zero mask ratio");
        } catch (IllegalArgumentException e) {
        }
        
        try {
            new MaskedAutoencoder(encoder, decoder, 4, 8, 8, 3, 1.0f);
            fail("Should throw for mask ratio >= 1");
        } catch (IllegalArgumentException e) {
        }
        
        encoder.close();
        decoder.close();
    }

    @Test
    public void testMAEToString() {
        Module encoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        Module decoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        MaskedAutoencoder mae = new MaskedAutoencoder(encoder, decoder, 4, 8, 8, 3, 0.75f);
        
        String str = mae.toString();
        assertTrue(str.contains("MaskedAutoencoder"));
        assertTrue(str.contains("4"));
        assertTrue(str.contains("8x8"));
        assertTrue(str.contains("0.75"));
        
        mae.close();
        
        str = mae.toString();
        assertTrue(str.contains("closed"));
    }

    @Test
    public void testMaskedLanguageModelCreation() {
        Module encoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        MaskedLanguageModel mlm = new MaskedLanguageModel(encoder, 1000, 256, 0.15f, 103);
        
        assertEquals(1000, mlm.vocabSize());
        assertEquals(256, mlm.hiddenSize());
        assertEquals(0.15f, mlm.maskProbability(), 0.001f);
        assertEquals(103, mlm.maskTokenId());
        
        mlm.close();
    }

    @Test
    public void testMLMMaskCreation() {
        Module encoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        MaskedLanguageModel mlm = new MaskedLanguageModel(encoder, 1000, 256, 0.15f, 103);
        
        int[] tokens = new int[32];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = (i % 100) + 1;
        }
        
        MaskedLanguageModel.MLMMask mask = mlm.createMask(tokens, 16);
        
        assertTrue("Some tokens should be masked", mask.numMasked() > 0);
        assertTrue("Not all tokens should be masked", mask.numMasked() < 16);
        assertEquals(16, mask.sequenceLength());
        
        mlm.close();
    }

    @Test
    public void testMLMApplyMask() {
        Module encoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        MaskedLanguageModel mlm = new MaskedLanguageModel(encoder, 1000, 256, 0.15f, 103);
        
        int[] tokens = new int[16];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = i + 1;
        }
        
        MaskedLanguageModel.MLMMask mask = mlm.createMask(tokens, 16);
        int[] maskedTokens = mlm.applyMask(tokens, mask);
        
        assertNotNull(maskedTokens);
        assertEquals(16, maskedTokens.length);
        
        mlm.close();
    }

    @Test
    public void testMLMInvalidParams() {
        Module encoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        
        try {
            new MaskedLanguageModel(null, 1000, 256);
            fail("Should throw for null encoder");
        } catch (NullPointerException e) {
        }
        
        try {
            new MaskedLanguageModel(encoder, 0, 256);
            fail("Should throw for non-positive vocab size");
        } catch (IllegalArgumentException e) {
        }
        
        try {
            new MaskedLanguageModel(encoder, 1000, 0);
            fail("Should throw for non-positive hidden size");
        } catch (IllegalArgumentException e) {
        }
        
        try {
            new MaskedLanguageModel(encoder, 1000, 256, 0, 103);
            fail("Should throw for zero mask probability");
        } catch (IllegalArgumentException e) {
        }
        
        try {
            new MaskedLanguageModel(encoder, 1000, 256, 1.0f, 103);
            fail("Should throw for mask probability >= 1");
        } catch (IllegalArgumentException e) {
        }
        
        encoder.close();
    }

    @Test
    public void testMLMToString() {
        Module encoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        MaskedLanguageModel mlm = new MaskedLanguageModel(encoder, 1000, 256, 0.15f, 103);
        
        String str = mlm.toString();
        assertTrue(str.contains("MaskedLanguageModel"));
        assertTrue(str.contains("1000"));
        assertTrue(str.contains("256"));
        
        mlm.close();
    }

    @Test
    public void testDINOTrainerCreation() {
        Module student = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        
        DINOTrainer dino = new DINOTrainer(student, 64, 0.996f, 0.9f, 0.04f, 0.1f);
        
        assertEquals(64, dino.outputDim());
        assertEquals(0.996f, dino.emaMomentum(), 0.001f);
        assertEquals(0.04f, dino.teacherTemp(), 0.001f);
        assertEquals(0.1f, dino.studentTemp(), 0.001f);
        assertNotNull(dino.student());
        
        dino.close();
    }

    @Test
    public void testDINOInvalidParams() {
        Module student = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        
        try {
            new DINOTrainer(null, 64);
            fail("Should throw for null student");
        } catch (NullPointerException e) {
        }
        
        try {
            new DINOTrainer(student, 0);
            fail("Should throw for non-positive output dim");
        } catch (IllegalArgumentException e) {
        }
        
        try {
            new DINOTrainer(student, 64, 0, 0.9f, 0.04f, 0.1f);
            fail("Should throw for zero EMA momentum");
        } catch (IllegalArgumentException e) {
        }
        
        try {
            new DINOTrainer(student, 64, 1.0f, 0.9f, 0.04f, 0.1f);
            fail("Should throw for EMA momentum >= 1");
        } catch (IllegalArgumentException e) {
        }
        
        student.close();
    }

    @Test
    public void testDINOTeacherIsNullBeforeInit() {
        Module student = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        
        DINOTrainer dino = new DINOTrainer(student, 64);
        assertNull(dino.teacher());
        
        dino.close();
    }

    @Test
    public void testDINOToString() {
        Module student = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        DINOTrainer dino = new DINOTrainer(student, 64, 0.996f, 0.9f, 0.04f, 0.1f);
        
        String str = dino.toString();
        assertTrue(str.contains("DINOTrainer"));
        assertTrue(str.contains("64"));
        
        dino.close();
    }

    @Test
    public void testCLIPTrainerCreation() {
        Module imageEncoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        Module textEncoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        
        CLIPTrainer clip = new CLIPTrainer(imageEncoder, textEncoder, 256, 256, 128);
        
        assertEquals(256, clip.imageEmbedDim());
        assertEquals(256, clip.textEmbedDim());
        assertEquals(128, clip.sharedEmbedDim());
        assertNotNull(clip.imageEncoder());
        assertNotNull(clip.textEncoder());
        assertNotNull(clip.imageProjection());
        assertNotNull(clip.textProjection());
        
        clip.close();
    }

    @Test
    public void testCLIPSetTemperature() {
        Module imageEncoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        Module textEncoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        
        CLIPTrainer clip = new CLIPTrainer(imageEncoder, textEncoder, 256, 256, 128, true);
        
        float defaultTemp = clip.temperature();
        clip.setTemperature(0.1f);
        assertEquals(0.1f, clip.temperature(), 0.001f);
        
        clip.close();
    }

    @Test
    public void testCLIPInvalidParams() {
        Module imageEncoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        Module textEncoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        
        try {
            new CLIPTrainer(null, textEncoder, 256, 256, 128);
            fail("Should throw for null image encoder");
        } catch (NullPointerException e) {
        }
        
        try {
            new CLIPTrainer(imageEncoder, null, 256, 256, 128);
            fail("Should throw for null text encoder");
        } catch (NullPointerException e) {
        }
        
        try {
            new CLIPTrainer(imageEncoder, textEncoder, 0, 256, 128);
            fail("Should throw for non-positive image embed dim");
        } catch (IllegalArgumentException e) {
        }
        
        try {
            new CLIPTrainer(imageEncoder, textEncoder, 256, 0, 128);
            fail("Should throw for non-positive text embed dim");
        } catch (IllegalArgumentException e) {
        }
        
        try {
            new CLIPTrainer(imageEncoder, textEncoder, 256, 256, 0);
            fail("Should throw for non-positive shared embed dim");
        } catch (IllegalArgumentException e) {
        }
        
        imageEncoder.close();
        textEncoder.close();
    }

    @Test
    public void testCLIPSetInvalidTemperature() {
        Module imageEncoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        Module textEncoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        
        CLIPTrainer clip = new CLIPTrainer(imageEncoder, textEncoder, 256, 256, 128);
        
        try {
            clip.setTemperature(0);
            fail("Should throw for non-positive temperature");
        } catch (IllegalArgumentException e) {
        }
        
        clip.close();
    }

    @Test
    public void testCLIPToString() {
        Module imageEncoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        Module textEncoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        CLIPTrainer clip = new CLIPTrainer(imageEncoder, textEncoder, 256, 256, 128);
        
        String str = clip.toString();
        assertTrue(str.contains("CLIPTrainer"));
        assertTrue(str.contains("256"));
        assertTrue(str.contains("128"));
        
        clip.close();
    }

    @Test
    public void testContrastiveLearnerClosedState() {
        Module encoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        ContrastiveLearner learner = new ContrastiveLearner(encoder, 128, 64);
        
        learner.close();
        
        try {
            learner.forward(null);
            fail("Should throw for closed module");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("closed"));
        }
    }

    @Test
    public void testMAEClosedState() {
        Module encoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        Module decoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        MaskedAutoencoder mae = new MaskedAutoencoder(encoder, decoder, 4, 8, 8, 3);
        
        mae.close();
        
        try {
            mae.patchify(null);
            fail("Should throw for closed module");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("closed"));
        }
    }

    @Test
    public void testMLMClosedState() {
        Module encoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        MaskedLanguageModel mlm = new MaskedLanguageModel(encoder, 1000, 256);
        
        mlm.close();
        
        try {
            mlm.forward(null);
            fail("Should throw for closed module");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("closed"));
        }
    }

    @Test
    public void testDINOClosedState() {
        Module student = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        DINOTrainer dino = new DINOTrainer(student, 64);
        
        dino.close();
        
        try {
            dino.forward(null);
            fail("Should throw for closed module");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("closed"));
        }
    }

    @Test
    public void testCLIPClosedState() {
        Module imageEncoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        Module textEncoder = new Module() {
            @Override
            public Tensor forward(Tensor input) { return input; }
        };
        CLIPTrainer clip = new CLIPTrainer(imageEncoder, textEncoder, 256, 256, 128);
        
        clip.close();
        
        try {
            clip.forward(null);
            fail("Should throw for closed module");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("closed"));
        }
    }
}
