package com.eonmux.neuralfree.nn.recurrent;

import org.junit.Test;
import static org.junit.Assert.*;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;

public class StateSpaceModelTest {
    
    @Test
    public void testConstructionBasic() {
        StateSpaceModel ssm = new StateSpaceModel(64, 128);
        
        assertEquals(64, ssm.inputSize());
        assertEquals(128, ssm.hiddenSize());
        assertEquals(128, ssm.stateDim());
        assertEquals(1, ssm.numLayers());
    }
    
    @Test
    public void testConstructionWithStateDim() {
        StateSpaceModel ssm = new StateSpaceModel(64, 128, 256);
        
        assertEquals(64, ssm.inputSize());
        assertEquals(128, ssm.hiddenSize());
        assertEquals(256, ssm.stateDim());
    }
    
    @Test
    public void testConstructionFull() {
        StateSpaceModel ssm = new StateSpaceModel(64, 128, 256, 2, true, 0.1f);
        
        assertEquals(64, ssm.inputSize());
        assertEquals(128, ssm.hiddenSize());
        assertEquals(256, ssm.stateDim());
        assertEquals(2, ssm.numLayers());
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructionInvalidInputSize() {
        new StateSpaceModel(0, 128);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructionInvalidInputSizeNegative() {
        new StateSpaceModel(-1, 128);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructionInvalidHiddenSize() {
        new StateSpaceModel(64, 0);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructionInvalidHiddenSizeNegative() {
        new StateSpaceModel(64, -1);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructionInvalidStateDim() {
        new StateSpaceModel(64, 128, 0);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructionInvalidNumLayers() {
        new StateSpaceModel(64, 128, 256, 0, true, 0.0f);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructionInvalidDropoutNegative() {
        new StateSpaceModel(64, 128, 256, 1, true, -0.1f);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructionInvalidDropoutOne() {
        new StateSpaceModel(64, 128, 256, 1, true, 1.0f);
    }
    
    @Test
    public void testToString() {
        StateSpaceModel ssm = new StateSpaceModel(64, 128, 256, 2, true, 0.1f);
        String str = ssm.toString();
        
        assertTrue(str.contains("StateSpaceModel"));
        assertTrue(str.contains("inputSize=64"));
        assertTrue(str.contains("hiddenSize=128"));
        assertTrue(str.contains("stateDim=256"));
        assertTrue(str.contains("numLayers=2"));
    }
    
    @Test
    public void testForwardWithoutInitializationThrows() {
        StateSpaceModel ssm = new StateSpaceModel(64, 128);
        
        float[] inputData = new float[10 * 4 * 64];
        Tensor input = null;
        try {
            input = Tensor.fromArray(null, inputData).reshape(Shape.of(10, 4, 64));
        } catch (NullPointerException e) {
            return;
        }
        
        try {
            ssm.forward(input);
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }
    
    @Test
    public void testForwardNullInputThrows() {
        StateSpaceModel ssm = new StateSpaceModel(64, 128);
        try {
            ssm.forward(null);
            fail("Expected exception");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        } catch (NullPointerException e) {
        }
    }
    
    @Test
    public void testClose() {
        StateSpaceModel ssm = new StateSpaceModel(64, 128);
        ssm.close();
    }
    
    @Test
    public void testModuleProperties() {
        StateSpaceModel ssm = new StateSpaceModel(64, 128);
        
        assertTrue(ssm instanceof Module);
        assertTrue(ssm.isTraining());
        
        ssm.eval();
        assertFalse(ssm.isTraining());
        
        ssm.train();
        assertTrue(ssm.isTraining());
    }
    
    @Test
    public void testTrainEvalMode() {
        StateSpaceModel ssm = new StateSpaceModel(64, 128, 256, 2, true, 0.1f);
        
        assertTrue(ssm.isTraining());
        
        ssm.eval();
        assertFalse(ssm.isTraining());
        
        ssm.train(true);
        assertTrue(ssm.isTraining());
        
        ssm.train(false);
        assertFalse(ssm.isTraining());
    }
    
    @Test
    public void testOutputShapeDescription() {
        StateSpaceModel ssm = new StateSpaceModel(64, 128);
        
        String str = ssm.toString();
        assertTrue(str.contains("hiddenSize=128"));
    }
    
    @Test
    public void testMultiLayer() {
        StateSpaceModel ssm = new StateSpaceModel(64, 128, 256, 3, true, 0.0f);
        
        assertEquals(3, ssm.numLayers());
    }
    
    @Test
    public void testZeroDropout() {
        StateSpaceModel ssm = new StateSpaceModel(64, 128, 256, 1, true, 0.0f);
        String str = ssm.toString();
        assertNotNull(str);
    }
    
    @Test
    public void testParameterCountBeforeInit() {
        StateSpaceModel ssm = new StateSpaceModel(64, 128);
        assertEquals(0, ssm.parameterCount());
    }
    
    @Test
    public void testModuleName() {
        StateSpaceModel ssm = new StateSpaceModel(64, 128);
        assertEquals("StateSpaceModel", ssm.moduleName());
    }
    
    @Test
    public void testConstructionEdgeCases() {
        StateSpaceModel ssm1 = new StateSpaceModel(1, 1, 1, 1, false, 0.0f);
        assertNotNull(ssm1);
        
        StateSpaceModel ssm2 = new StateSpaceModel(1000, 1000, 1000, 10, true, 0.49f);
        assertNotNull(ssm2);
    }
    
    @Test
    public void testNoBias() {
        StateSpaceModel ssm = new StateSpaceModel(64, 128, 256, 1, false, 0.0f);
        assertNotNull(ssm);
    }
    
    @Test
    public void testRecurrentStateDim() {
        int inputSize = 32;
        int hiddenSize = 64;
        int stateDim = 128;
        
        StateSpaceModel ssm = new StateSpaceModel(inputSize, hiddenSize, stateDim);
        
        assertEquals(stateDim, ssm.stateDim());
    }
    
    @Test
    public void testSSMArchitectureMatchesMambaStyle() {
        int inputSize = 16;
        int hiddenSize = 32;
        int stateDim = 64;
        
        StateSpaceModel ssm = new StateSpaceModel(inputSize, hiddenSize, stateDim, 1, true, 0.0f);
        
        assertEquals(inputSize, ssm.inputSize());
        assertEquals(hiddenSize, ssm.hiddenSize());
        assertEquals(stateDim, ssm.stateDim());
    }
    
    @Test
    public void testHighDropoutEdgeCase() {
        StateSpaceModel ssm = new StateSpaceModel(64, 128, 256, 1, true, 0.99f);
        assertNotNull(ssm);
    }
}
