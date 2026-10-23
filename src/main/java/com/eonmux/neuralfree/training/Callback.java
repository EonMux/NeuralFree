package com.eonmux.neuralfree.training;

public interface Callback {
    
    /** Called once before the first epoch of a run, with the trainer running it. */
    default void onTrainBegin(TrainingContext trainer) {
    }
    
    default void onTrainEnd(TrainingResult result) {
    }
    
    default void onEpochBegin(int epoch, int totalEpochs) {
    }
    
    default void onEpochEnd(int epoch, float trainLoss, Float valLoss) {
    }
    
    default void onBatchBegin(int batch, int totalBatches) {
    }
    
    default void onBatchEnd(int batch, float loss) {
    }
    
    default void onValidationBegin(int epoch) {
    }
    
    default void onValidationEnd(int epoch, EvalResult result) {
    }
    
    default String name() {
        return getClass().getSimpleName();
    }
}
