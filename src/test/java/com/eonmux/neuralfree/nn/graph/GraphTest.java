package com.eonmux.neuralfree.nn.graph;

import org.junit.Test;

import static org.junit.Assert.*;

public class GraphTest {

    @Test
    public void testGCNConvCreation() {
        GCNConv gcn = new GCNConv(16, 32);
        
        assertNotNull(gcn);
        assertEquals(16, gcn.inFeatures());
        assertEquals(32, gcn.outFeatures());
        assertTrue(gcn.hasBias());
        assertTrue(gcn.isNormalized());
    }

    @Test
    public void testGCNConvCreationNoBias() {
        GCNConv gcn = new GCNConv(16, 32, false, true);
        
        assertFalse(gcn.hasBias());
    }

    @Test
    public void testGCNConvCreationNoNormalize() {
        GCNConv gcn = new GCNConv(16, 32, true, false);
        
        assertFalse(gcn.isNormalized());
    }

    @Test
    public void testGCNConvNotInitialized() {
        GCNConv gcn = new GCNConv(16, 32);
        
        try {
            gcn.weight();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }

    @Test
    public void testGCNConvNoBiasException() {
        GCNConv gcn = new GCNConv(16, 32, false, true);
        
        try {
            gcn.bias();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("no bias"));
        }
    }

    @Test
    public void testGCNConvInvalidFeatures() {
        try {
            new GCNConv(0, 32);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("inFeatures"));
        }
        
        try {
            new GCNConv(16, 0);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("outFeatures"));
        }
    }

    @Test
    public void testGCNConvToString() {
        GCNConv gcn = new GCNConv(16, 32);
        String str = gcn.toString();
        
        assertTrue(str.contains("GCNConv"));
        assertTrue(str.contains("inFeatures=16"));
        assertTrue(str.contains("outFeatures=32"));
        assertTrue(str.contains("useBias=true"));
        assertTrue(str.contains("normalize=true"));
    }

    @Test
    public void testGATConvCreation() {
        GATConv gat = new GATConv(16, 32, 4, true);
        
        assertNotNull(gat);
        assertEquals(16, gat.inFeatures());
        assertEquals(32, gat.outFeatures());
        assertEquals(4, gat.heads());
        assertTrue(gat.isConcat());
        assertTrue(gat.hasBias());
    }

    @Test
    public void testGATConvCreationNoConcat() {
        GATConv gat = new GATConv(16, 32, 4, false);
        
        assertFalse(gat.isConcat());
    }

    @Test
    public void testGATConvDefaultConstructor() {
        GATConv gat = new GATConv(16, 32);
        
        assertEquals(1, gat.heads());
        assertTrue(gat.isConcat());
    }

    @Test
    public void testGATConvNotInitialized() {
        GATConv gat = new GATConv(16, 32);
        
        try {
            gat.weight();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }

    @Test
    public void testGATConvInvalidFeatures() {
        try {
            new GATConv(0, 32);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("inFeatures"));
        }
        
        try {
            new GATConv(16, 0);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("outFeatures"));
        }
    }

    @Test
    public void testGATConvInvalidHeads() {
        try {
            new GATConv(16, 32, 0, true);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("heads"));
        }
    }

    @Test
    public void testGATConvProperties() {
        GATConv gat = new GATConv(16, 32, 4, true, false, 0.1f, 0.5f);
        
        assertEquals(0.1f, gat.negativeSlope(), 1e-6f);
        assertEquals(0.5f, gat.dropout(), 1e-6f);
        assertFalse(gat.hasBias());
    }

    @Test
    public void testGATConvToString() {
        GATConv gat = new GATConv(16, 32, 4, true);
        String str = gat.toString();
        
        assertTrue(str.contains("GATConv"));
        assertTrue(str.contains("inFeatures=16"));
        assertTrue(str.contains("outFeatures=32"));
        assertTrue(str.contains("heads=4"));
    }

    @Test
    public void testGraphSAGEConvCreation() {
        GraphSAGEConv sage = new GraphSAGEConv(16, 32);
        
        assertNotNull(sage);
        assertEquals(16, sage.inFeatures());
        assertEquals(32, sage.outFeatures());
        assertEquals(GraphSAGEConv.Aggregator.MEAN, sage.aggregator());
        assertTrue(sage.hasBias());
        assertTrue(sage.isNormalized());
    }

    @Test
    public void testGraphSAGEConvAggregators() {
        GraphSAGEConv mean = new GraphSAGEConv(16, 32, GraphSAGEConv.Aggregator.MEAN, true, true);
        GraphSAGEConv maxPool = new GraphSAGEConv(16, 32, GraphSAGEConv.Aggregator.MAX_POOL, true, false);
        GraphSAGEConv lstm = new GraphSAGEConv(16, 32, GraphSAGEConv.Aggregator.LSTM, false, true);
        
        assertEquals(GraphSAGEConv.Aggregator.MEAN, mean.aggregator());
        assertEquals(GraphSAGEConv.Aggregator.MAX_POOL, maxPool.aggregator());
        assertEquals(GraphSAGEConv.Aggregator.LSTM, lstm.aggregator());
        assertFalse(maxPool.isNormalized());
        assertFalse(lstm.hasBias());
    }

    @Test
    public void testGraphSAGEConvNotInitialized() {
        GraphSAGEConv sage = new GraphSAGEConv(16, 32);
        
        try {
            sage.weightSelf();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }

    @Test
    public void testGraphSAGEConvNoBiasException() {
        GraphSAGEConv sage = new GraphSAGEConv(16, 32, GraphSAGEConv.Aggregator.MEAN, false, true);
        
        try {
            sage.bias();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("no bias"));
        }
    }

    @Test
    public void testGraphSAGEConvInvalidFeatures() {
        try {
            new GraphSAGEConv(0, 32);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("inFeatures"));
        }
        
        try {
            new GraphSAGEConv(16, 0);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("outFeatures"));
        }
    }

    @Test
    public void testGraphSAGEConvToString() {
        GraphSAGEConv sage = new GraphSAGEConv(16, 32);
        String str = sage.toString();
        
        assertTrue(str.contains("GraphSAGEConv"));
        assertTrue(str.contains("inFeatures=16"));
        assertTrue(str.contains("outFeatures=32"));
        assertTrue(str.contains("aggregator=MEAN"));
    }

    @Test
    public void testGINConvCreation() {
        GINConv gin = new GINConv(16, 32);
        
        assertNotNull(gin);
        assertEquals(16, gin.inFeatures());
        assertEquals(32, gin.outFeatures());
        assertEquals(0.0f, gin.eps(), 1e-6f);
        assertFalse(gin.isEpsTrainable());
    }

    @Test
    public void testGINConvWithEps() {
        GINConv gin = new GINConv(16, 32, 0.5f, false);
        
        assertEquals(0.5f, gin.eps(), 1e-6f);
    }

    @Test
    public void testGINConvWithTrainableEps() {
        GINConv gin = new GINConv(16, 32, 0.5f, true);
        
        assertTrue(gin.isEpsTrainable());
    }

    @Test
    public void testGINConvNotInitialized() {
        GINConv gin = new GINConv(16, 32);
        
        try {
            gin.mlp();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }

    @Test
    public void testGINConvInvalidFeatures() {
        try {
            new GINConv(0, 32);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("inFeatures"));
        }
        
        try {
            new GINConv(16, 0);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("outFeatures"));
        }
    }

    @Test
    public void testGINConvToString() {
        GINConv gin = new GINConv(16, 32);
        String str = gin.toString();
        
        assertTrue(str.contains("GINConv"));
        assertTrue(str.contains("inFeatures=16"));
        assertTrue(str.contains("outFeatures=32"));
        assertTrue(str.contains("eps=0.0"));
    }

    @Test
    public void testGlobalMeanPoolCreation() {
        GlobalMeanPool pool = new GlobalMeanPool();
        assertNotNull(pool);
    }

    @Test
    public void testGlobalMeanPoolToString() {
        GlobalMeanPool pool = new GlobalMeanPool();
        String str = pool.toString();
        assertTrue(str.contains("GlobalMeanPool"));
    }

    @Test
    public void testGlobalMeanPoolForwardRequiresTensorInput() {
        GlobalMeanPool pool = new GlobalMeanPool();
        
        try {
            pool.forward((com.eonmux.neuralfree.tensor.Tensor) null);
            fail("Expected UnsupportedOperationException");
        } catch (UnsupportedOperationException e) {
            assertTrue(e.getMessage().contains("GraphData"));
        }
    }

    @Test
    public void testGlobalMaxPoolCreation() {
        GlobalMaxPool pool = new GlobalMaxPool();
        assertNotNull(pool);
    }

    @Test
    public void testGlobalMaxPoolToString() {
        GlobalMaxPool pool = new GlobalMaxPool();
        String str = pool.toString();
        assertTrue(str.contains("GlobalMaxPool"));
    }

    @Test
    public void testGlobalMaxPoolForwardRequiresTensorInput() {
        GlobalMaxPool pool = new GlobalMaxPool();
        
        try {
            pool.forward((com.eonmux.neuralfree.tensor.Tensor) null);
            fail("Expected UnsupportedOperationException");
        } catch (UnsupportedOperationException e) {
            assertTrue(e.getMessage().contains("GraphData"));
        }
    }

    @Test
    public void testTopKPoolCreation() {
        TopKPool pool = new TopKPool(16, 0.5f);
        
        assertNotNull(pool);
        assertEquals(16, pool.inFeatures());
        assertEquals(0.5f, pool.ratio(), 1e-6f);
    }

    @Test
    public void testTopKPoolWithMinScore() {
        TopKPool pool = new TopKPool(16, 0.5f, 0.1f);
        
        assertEquals(0.1f, pool.minScore(), 1e-6f);
    }

    @Test
    public void testTopKPoolInvalidInFeatures() {
        try {
            new TopKPool(0, 0.5f);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("inFeatures"));
        }
    }

    @Test
    public void testTopKPoolInvalidRatioZero() {
        try {
            new TopKPool(16, 0.0f);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("ratio"));
        }
    }

    @Test
    public void testTopKPoolInvalidRatioTooHigh() {
        try {
            new TopKPool(16, 1.5f);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("ratio"));
        }
    }

    @Test
    public void testTopKPoolNotInitialized() {
        TopKPool pool = new TopKPool(16, 0.5f);
        
        try {
            pool.weight();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }

    @Test
    public void testTopKPoolToString() {
        TopKPool pool = new TopKPool(16, 0.5f);
        String str = pool.toString();
        
        assertTrue(str.contains("TopKPool"));
        assertTrue(str.contains("inFeatures=16"));
        assertTrue(str.contains("ratio=0.5"));
    }

    @Test
    public void testAggregatorEnum() {
        assertEquals(3, GraphSAGEConv.Aggregator.values().length);
        assertNotNull(GraphSAGEConv.Aggregator.valueOf("MEAN"));
        assertNotNull(GraphSAGEConv.Aggregator.valueOf("MAX_POOL"));
        assertNotNull(GraphSAGEConv.Aggregator.valueOf("LSTM"));
    }
}
