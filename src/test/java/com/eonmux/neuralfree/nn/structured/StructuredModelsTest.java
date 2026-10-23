package com.eonmux.neuralfree.nn.structured;

import com.eonmux.neuralfree.nn.graph.GraphData;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import org.junit.Test;

import static org.junit.Assert.*;

public class StructuredModelsTest {

    @Test
    public void testNeuralODECreation() {
        NeuralODE ode = new NeuralODE(8, 16);
        
        assertNotNull(ode);
        assertEquals(8, ode.inputDim());
        assertEquals(16, ode.hiddenDim());
        assertEquals(1, ode.numLayers());
        assertEquals(NeuralODE.Solver.RK4, ode.solver());
    }

    @Test
    public void testNeuralODECustomConfig() {
        NeuralODE ode = new NeuralODE(8, 16, 3, NeuralODE.Solver.EULER, 1e-4f, 1e-6f);
        
        assertEquals(3, ode.numLayers());
        assertEquals(NeuralODE.Solver.EULER, ode.solver());
        assertEquals(1e-4f, ode.rtol(), 1e-10f);
        assertEquals(1e-6f, ode.atol(), 1e-10f);
    }

    @Test
    public void testNeuralODEInvalidInputDim() {
        try {
            new NeuralODE(0, 16);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("inputDim"));
        }
    }

    @Test
    public void testNeuralODEInvalidHiddenDim() {
        try {
            new NeuralODE(8, 0);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("hiddenDim"));
        }
    }

    @Test
    public void testNeuralODENotInitialized() {
        NeuralODE ode = new NeuralODE(8, 16);
        
        try {
            ode.forward((Tensor) null);
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }

    @Test
    public void testNeuralODEToString() {
        NeuralODE ode = new NeuralODE(8, 16, 2, NeuralODE.Solver.DOPRI5, 1e-4f, 1e-6f);
        String str = ode.toString();
        
        assertTrue(str.contains("NeuralODE"));
        assertTrue(str.contains("inputDim=8"));
        assertTrue(str.contains("hiddenDim=16"));
        assertTrue(str.contains("numLayers=2"));
        assertTrue(str.contains("solver=DOPRI5"));
    }

    @Test
    public void testNeuralRadianceFieldCreation() {
        NeuralRadianceField nerf = new NeuralRadianceField(64, 4);
        
        assertNotNull(nerf);
        assertEquals(64, nerf.hiddenDim());
        assertEquals(4, nerf.numLayers());
    }

    @Test
    public void testNeuralRadianceFieldCustomConfig() {
        NeuralRadianceField nerf = new NeuralRadianceField(
            10, 4, 128, 8, new int[]{4, 6}, 0.1f, 10.0f, 128
        );
        
        assertEquals(10, nerf.posEncodingDim());
        assertEquals(4, nerf.dirEncodingDim());
        assertEquals(128, nerf.hiddenDim());
        assertEquals(8, nerf.numLayers());
        assertEquals(0.1f, nerf.near(), 1e-6f);
        assertEquals(10.0f, nerf.far(), 1e-6f);
        assertEquals(128, nerf.numSamples());
    }

    @Test
    public void testNeuralRadianceFieldInvalidHiddenDim() {
        try {
            new NeuralRadianceField(0, 4);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("hiddenDim"));
        }
    }

    @Test
    public void testNeuralRadianceFieldInvalidNearFar() {
        try {
            new NeuralRadianceField(10, 4, 64, 4, new int[]{4}, 1.0f, 0.1f, 64);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("far must be greater than near"));
        }
    }

    @Test
    public void testNeuralRadianceFieldNotInitialized() {
        NeuralRadianceField nerf = new NeuralRadianceField(64, 4);
        
        try {
            nerf.forward((Tensor) null);
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }

    @Test
    public void testNeuralRadianceFieldToString() {
        NeuralRadianceField nerf = new NeuralRadianceField(64, 4);
        String str = nerf.toString();
        
        assertTrue(str.contains("NeuralRadianceField"));
        assertTrue(str.contains("hiddenDim=64"));
        assertTrue(str.contains("numLayers=4"));
    }

    @Test
    public void testGaussianSplattingCreation() {
        GaussianSplatting gs = new GaussianSplatting(1000, 256, 256);
        
        assertNotNull(gs);
        assertEquals(1000, gs.numGaussians());
        assertEquals(256, gs.imageWidth());
        assertEquals(256, gs.imageHeight());
    }

    @Test
    public void testGaussianSplattingInvalidNumGaussians() {
        try {
            new GaussianSplatting(0, 256, 256);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("numGaussians"));
        }
    }

    @Test
    public void testGaussianSplattingInvalidImageWidth() {
        try {
            new GaussianSplatting(1000, 0, 256);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("imageWidth"));
        }
    }

    @Test
    public void testGaussianSplattingInvalidImageHeight() {
        try {
            new GaussianSplatting(1000, 256, 0);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("imageHeight"));
        }
    }

    @Test
    public void testGaussianSplattingNotInitialized() {
        GaussianSplatting gs = new GaussianSplatting(100, 64, 64);
        
        try {
            gs.getMeans();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }

    @Test
    public void testGaussianSplattingToString() {
        GaussianSplatting gs = new GaussianSplatting(500, 128, 128);
        String str = gs.toString();
        
        assertTrue(str.contains("GaussianSplatting"));
        assertTrue(str.contains("numGaussians=500"));
        assertTrue(str.contains("imageWidth=128"));
        assertTrue(str.contains("imageHeight=128"));
    }

    @Test
    public void testGraphTransformerCreation() {
        GraphTransformer gt = new GraphTransformer(16, 64, 32, 4, 2);
        
        assertNotNull(gt);
        assertEquals(16, gt.inFeatures());
        assertEquals(64, gt.hiddenDim());
        assertEquals(32, gt.outFeatures());
        assertEquals(4, gt.numHeads());
        assertEquals(2, gt.numLayers());
    }

    @Test
    public void testGraphTransformerCustomConfig() {
        GraphTransformer gt = new GraphTransformer(16, 64, 32, 4, 2, 0.2f, true);
        
        assertEquals(0.2f, gt.dropoutRate(), 1e-6f);
        assertTrue(gt.useEdgeFeatures());
    }

    @Test
    public void testGraphTransformerInvalidInFeatures() {
        try {
            new GraphTransformer(0, 64, 32, 4, 2);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("inFeatures"));
        }
    }

    @Test
    public void testGraphTransformerInvalidHiddenDim() {
        try {
            new GraphTransformer(16, 0, 32, 4, 2);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("hiddenDim"));
        }
    }

    @Test
    public void testGraphTransformerInvalidNumHeads() {
        try {
            new GraphTransformer(16, 64, 32, 0, 2);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("numHeads"));
        }
    }

    @Test
    public void testGraphTransformerHiddenDimNotDivisibleByHeads() {
        try {
            new GraphTransformer(16, 65, 32, 4, 2);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("divisible by numHeads"));
        }
    }

    @Test
    public void testGraphTransformerInvalidDropoutRate() {
        try {
            new GraphTransformer(16, 64, 32, 4, 2, 1.5f, false);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("dropoutRate"));
        }
    }

    @Test
    public void testGraphTransformerNotInitialized() {
        GraphTransformer gt = new GraphTransformer(16, 64, 32, 4, 2);
        
        try {
            gt.forward((GraphData) null);
            fail("Expected IllegalStateException or NullPointerException");
        } catch (IllegalStateException | NullPointerException expected) {
        }
    }

    @Test
    public void testGraphTransformerToString() {
        GraphTransformer gt = new GraphTransformer(16, 64, 32, 4, 2);
        String str = gt.toString();
        
        assertTrue(str.contains("GraphTransformer"));
        assertTrue(str.contains("inFeatures=16"));
        assertTrue(str.contains("hiddenDim=64"));
        assertTrue(str.contains("outFeatures=32"));
        assertTrue(str.contains("numHeads=4"));
        assertTrue(str.contains("numLayers=2"));
    }

    @Test
    public void testNeuralODEForwardRequiresTensorInput() {
        NeuralODE ode = new NeuralODE(8, 16);
        
        try {
            ode.forward((Tensor) null);
            fail("Expected NullPointerException or IllegalStateException");
        } catch (NullPointerException | IllegalStateException expected) {
        }
    }

    @Test
    public void testNeuralRadianceFieldForwardRequiresTensorInput() {
        NeuralRadianceField nerf = new NeuralRadianceField(64, 4);
        
        try {
            nerf.forward((Tensor) null);
            fail("Expected NullPointerException or IllegalStateException");
        } catch (NullPointerException | IllegalStateException expected) {
        }
    }

    @Test
    public void testGraphTransformerForwardRequiresGraphData() {
        GraphTransformer gt = new GraphTransformer(16, 64, 32, 4, 2);
        
        try {
            gt.forward((Tensor) null);
            fail("Expected UnsupportedOperationException");
        } catch (UnsupportedOperationException e) {
            assertTrue(e.getMessage().contains("GraphTransformer requires GraphData"));
        }
    }
}
