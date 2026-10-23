package com.eonmux.neuralfree.nn.recurrent;

import org.junit.Test;

import static org.junit.Assert.*;

public class RecurrentModulesTest {

    @Test
    public void testRNNCellConstruction() {
        RNNCell cell = new RNNCell(10, 20);
        assertEquals(10, cell.inputSize());
        assertEquals(20, cell.hiddenSize());
        assertTrue(cell.hasBias());
        assertEquals(RNNCell.TANH, cell.activation());
    }

    @Test
    public void testRNNCellConstructionWithRelu() {
        RNNCell cell = new RNNCell(10, 20, true, RNNCell.RELU);
        assertEquals(RNNCell.RELU, cell.activation());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testRNNCellInvalidInputSize() {
        new RNNCell(0, 20);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testRNNCellInvalidHiddenSize() {
        new RNNCell(10, 0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testRNNCellInvalidActivation() {
        new RNNCell(10, 20, true, "invalid");
    }

    @Test
    public void testRNNCellNoBiasConstruction() {
        RNNCell cell = new RNNCell(10, 20, false);
        assertFalse(cell.hasBias());
    }

    @Test
    public void testLSTMCellConstruction() {
        LSTMCell cell = new LSTMCell(10, 20);
        assertEquals(10, cell.inputSize());
        assertEquals(20, cell.hiddenSize());
        assertTrue(cell.hasBias());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLSTMCellInvalidInputSize() {
        new LSTMCell(0, 20);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLSTMCellInvalidHiddenSize() {
        new LSTMCell(10, 0);
    }

    @Test
    public void testLSTMCellNoBiasConstruction() {
        LSTMCell cell = new LSTMCell(10, 20, false);
        assertFalse(cell.hasBias());
    }

    @Test
    public void testGRUCellConstruction() {
        GRUCell cell = new GRUCell(10, 20);
        assertEquals(10, cell.inputSize());
        assertEquals(20, cell.hiddenSize());
        assertTrue(cell.hasBias());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testGRUCellInvalidInputSize() {
        new GRUCell(0, 20);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testGRUCellInvalidHiddenSize() {
        new GRUCell(10, 0);
    }

    @Test
    public void testGRUCellNoBiasConstruction() {
        GRUCell cell = new GRUCell(10, 20, false);
        assertFalse(cell.hasBias());
    }

    @Test
    public void testRNNConstruction() {
        RNN rnn = new RNN(10, 20, 2);
        assertEquals(10, rnn.inputSize());
        assertEquals(20, rnn.hiddenSize());
        assertEquals(2, rnn.numLayers());
        assertFalse(rnn.isBidirectional());
        assertEquals(1, rnn.numDirections());
    }

    @Test
    public void testRNNBidirectionalConstruction() {
        RNN rnn = new RNN(10, 20, 2, true);
        assertTrue(rnn.isBidirectional());
        assertEquals(2, rnn.numDirections());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testRNNInvalidInputSize() {
        new RNN(0, 20, 2);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testRNNInvalidHiddenSize() {
        new RNN(10, 0, 2);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testRNNInvalidNumLayers() {
        new RNN(10, 20, 0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testRNNInvalidDropout() {
        new RNN(10, 20, 2, false, 1.0f, RNNCell.TANH, true);
    }

    @Test
    public void testLSTMConstruction() {
        LSTM lstm = new LSTM(10, 20, 2);
        assertEquals(10, lstm.inputSize());
        assertEquals(20, lstm.hiddenSize());
        assertEquals(2, lstm.numLayers());
        assertFalse(lstm.isBidirectional());
    }

    @Test
    public void testLSTMBidirectionalConstruction() {
        LSTM lstm = new LSTM(10, 20, 2, true);
        assertTrue(lstm.isBidirectional());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLSTMInvalidInputSize() {
        new LSTM(0, 20, 2);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLSTMInvalidHiddenSize() {
        new LSTM(10, 0, 2);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLSTMInvalidNumLayers() {
        new LSTM(10, 20, 0);
    }

    @Test
    public void testGRUConstruction() {
        GRU gru = new GRU(10, 20, 2);
        assertEquals(10, gru.inputSize());
        assertEquals(20, gru.hiddenSize());
        assertEquals(2, gru.numLayers());
        assertFalse(gru.isBidirectional());
    }

    @Test
    public void testGRUBidirectionalConstruction() {
        GRU gru = new GRU(10, 20, 2, true);
        assertTrue(gru.isBidirectional());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testGRUInvalidInputSize() {
        new GRU(0, 20, 2);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testGRUInvalidHiddenSize() {
        new GRU(10, 0, 2);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testGRUInvalidNumLayers() {
        new GRU(10, 20, 0);
    }

    @Test
    public void testRNNCellToString() {
        RNNCell cell = new RNNCell(10, 20, true, RNNCell.RELU);
        String str = cell.toString();
        assertTrue(str.contains("RNNCell"));
        assertTrue(str.contains("inputSize=10"));
        assertTrue(str.contains("hiddenSize=20"));
        assertTrue(str.contains("activation='relu'"));
    }

    @Test
    public void testLSTMCellToString() {
        LSTMCell cell = new LSTMCell(10, 20);
        String str = cell.toString();
        assertTrue(str.contains("LSTMCell"));
        assertTrue(str.contains("inputSize=10"));
        assertTrue(str.contains("hiddenSize=20"));
    }

    @Test
    public void testGRUCellToString() {
        GRUCell cell = new GRUCell(10, 20);
        String str = cell.toString();
        assertTrue(str.contains("GRUCell"));
        assertTrue(str.contains("inputSize=10"));
        assertTrue(str.contains("hiddenSize=20"));
    }

    @Test
    public void testRNNToString() {
        RNN rnn = new RNN(10, 20, 2, true, 0.5f, RNNCell.TANH, true);
        String str = rnn.toString();
        assertTrue(str.contains("RNN"));
        assertTrue(str.contains("inputSize=10"));
        assertTrue(str.contains("hiddenSize=20"));
        assertTrue(str.contains("numLayers=2"));
        assertTrue(str.contains("bidirectional=true"));
    }

    @Test
    public void testLSTMToString() {
        LSTM lstm = new LSTM(10, 20, 2, true, 0.5f, true);
        String str = lstm.toString();
        assertTrue(str.contains("LSTM"));
        assertTrue(str.contains("inputSize=10"));
        assertTrue(str.contains("hiddenSize=20"));
        assertTrue(str.contains("bidirectional=true"));
    }

    @Test
    public void testGRUToString() {
        GRU gru = new GRU(10, 20, 2, true, 0.5f, true);
        String str = gru.toString();
        assertTrue(str.contains("GRU"));
        assertTrue(str.contains("inputSize=10"));
        assertTrue(str.contains("hiddenSize=20"));
        assertTrue(str.contains("bidirectional=true"));
    }
}
