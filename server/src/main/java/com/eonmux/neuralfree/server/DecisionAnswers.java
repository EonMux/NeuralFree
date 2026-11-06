package com.eonmux.neuralfree.server;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The answer of one evaluated decision question: the head's option scores softmaxed with the model's
 * per-type temperature and the score-question offsets, and the answer objects of the two served APIs. The
 * softmax, the temperature buckets and the confidence formulas follow llama.cpp's server
 * ({@code format_answer}, {@code get_temperature}, the TypeSafe confidence formulas); the vLLM shapes follow
 * {@code vllm.entrypoints.openai.decisions.adapters.make_answer}. Immutable.
 */
final class DecisionAnswers {

    private DecisionAnswers() {
    }

    /**
     * The model's temperature for a question: the metadata names it per type and per option-count bucket
     * ({@code choice.3_5} for three to five options), the bare type name is the fallback, and one is the
     * default (llama.cpp's {@code get_temperature}).
     */
    static float temperature(Map<String, Float> temperatures, String typeName, int nOptions) {
        String bucket = nOptions <= 2 ? "2" : nOptions <= 5 ? "3_5" : nOptions <= 10 ? "6_10" : "11";
        return temperature(temperatures, typeName, bucket);
    }

    /**
     * {@link #temperature(Map, String, int)} with the family's own bucket name ({@code small}/{@code mid}/
     * {@code large} for lev, the option-count buckets otherwise).
     */
    static float temperature(Map<String, Float> temperatures, String typeName, String bucket) {
        for (String name : List.of(typeName + "." + bucket, typeName)) {
            Float temp = temperatures.get(name);
            if (temp != null) {
                return temp;
            }
        }
        return 1.0f;
    }

    /**
     * The option probabilities of one question's scores: a NaN or infinite score means the head could not use
     * the decision order; score questions add the model's offsets ({@code decision.score_bias.<n>}) before the
     * softmax, which runs over the logits shifted by their maximum, scaled by the temperature (llama.cpp's
     * {@code format_answer}).
     *
     * @throws IllegalStateException when the head could not evaluate the question
     */
    static double[] probabilities(float[] scores, boolean score, float[] scoreBias, float temperature) {
        double[] logits = new double[scores.length];
        for (int i = 0; i < scores.length; i++) {
            if (Float.isNaN(scores[i]) || scores[i] == Float.POSITIVE_INFINITY) {
                throw new IllegalStateException("the model could not evaluate the decision");
            }
            logits[i] = scores[i] + (score && scoreBias != null ? scoreBias[i] : 0.0);
        }
        double max = Double.NEGATIVE_INFINITY;
        for (double logit : logits) {
            max = Math.max(max, logit);
        }
        double[] probs = new double[logits.length];
        double sum = 0.0;
        for (int i = 0; i < probs.length; i++) {
            probs[i] = Math.exp((logits[i] - max) / temperature);
            sum += probs[i];
        }
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }
        return probs;
    }

    /** The choice confidence: the top probability's edge over uniform, rescaled (TypeSafe's formula). */
    static double confidenceChoice(double[] probs) {
        if (probs.length < 2) {
            return 1.0;
        }
        double uniform = 1.0 / probs.length;
        double pMax = max(probs);
        return Math.max(0.0, (pMax - uniform) / (1.0 - uniform));
    }

    /** The score confidence: one minus the mean distance to the mode, relative to uniform's (TypeSafe's). */
    static double confidenceScore(double[] probs) {
        if (probs.length < 2) {
            return 1.0;
        }
        int mode = maxIndex(probs);
        double dist = 0.0;
        double distUniform = 0.0;
        for (int i = 0; i < probs.length; i++) {
            dist += probs[i] * Math.abs((double) i - (double) mode);
            distUniform += Math.abs(i - (probs.length - 1) / 2.0) / probs.length;
        }
        return Math.max(0.0, 1.0 - dist / distUniform);
    }

    /** The vllm-decision2 confidence: one minus the normalized entropy of the distribution, clamped. */
    static double confidenceEntropy(double[] probs) {
        if (probs.length < 2) {
            return 1.0;
        }
        double entropy = 0.0;
        for (double p : probs) {
            if (p > 0.0) {
                entropy -= p * Math.log(p);
            }
        }
        return Math.max(0.0, Math.min(1.0, 1.0 - entropy / Math.log(probs.length)));
    }

    /**
     * The {@code /v1/systemone} answer of one question under the family's readout: a lev noul question answers the
     * expected rating, a score question the expected level (the median level for lfm3-d1 and lfm3-d1-omni, the first
     * whose cumulative probability reaches one half), and the confidence follows the family (the normalized entropy,
     * the answer's probability for the lfm3 d1 families, or TypeSafe's per-type formula).
     */
    static Map<String, Object> systemOne(DecisionFamily family, DecisionQuestions.Question question, double[] probs) {
        Map<String, Object> answer = new LinkedHashMap<>();
        answer.put("type", DecisionPrompt.typeName(question.type()));
        if (question.type() == DecisionQuestions.NOUL) {
            if (family.ratingNoul) {
                double expected = 0.0;
                for (int i = 0; i < probs.length; i++) {
                    expected += probs[i] * i / (probs.length - 1);
                }
                answer.put("noul", expected);
                return answer;
            }
            for (int i = 0; i < probs.length; i++) {
                if ("true".equals(question.options().get(i).key())) {
                    answer.put("noul", probs[i]);
                }
            }
            return answer;
        }
        Map<String, Object> probabilities = new LinkedHashMap<>();
        for (int i = 0; i < probs.length; i++) {
            probabilities.put(question.options().get(i).key(), probs[i]);
        }
        if (question.type() == DecisionQuestions.CHOICE) {
            answer.put("choice", question.options().get(maxIndex(probs)).key());
            answer.put("probabilities", probabilities);
            answer.put("confidence", confidence(family.confidence, DecisionQuestions.CHOICE, probs));
            return answer;
        }
        double expected = family.medianScore() ? median(probs) : expected(probs);
        Map<String, Object> legend = new LinkedHashMap<>();
        for (int i = 0; i < probs.length; i++) {
            Object description = question.options().get(i).description();
            legend.put(question.options().get(i).key(),
                    family.canonicalLegend && !(description instanceof String) ? jsonDump(description) : description);
        }
        answer.put("score", expected);
        answer.put("legend", legend);
        answer.put("probabilities", probabilities);
        answer.put("confidence", confidence(family.confidence, DecisionQuestions.SCORE, probs));
        return answer;
    }

    /** The expected level of a distribution over ordered levels. */
    private static double expected(double[] probs) {
        double expected = 0.0;
        for (int i = 0; i < probs.length; i++) {
            expected += i * probs[i];
        }
        return expected;
    }

    /** The median level of a distribution over ordered levels, the first whose cumulative mass reaches one half. */
    private static double median(double[] probs) {
        double mass = 0.0;
        int level = probs.length - 1;
        for (int i = 0; i < probs.length && mass < 0.5; i++) {
            mass += probs[i];
            level = i;
        }
        return level;
    }

    /** The confidence of a question type under the family's formula. */
    private static double confidence(DecisionFamily.Confidence formula, int type, double[] probs) {
        return switch (formula) {
            case ENTROPY -> confidenceEntropy(probs);
            case MAX -> probs[maxIndex(probs)];
            case TYPESAFE -> type == DecisionQuestions.SCORE ? confidenceScore(probs) : confidenceChoice(probs);
        };
    }

    /**
     * The {@code /v1/decisions} answer of one question, over its probabilities in request order: the predicate
     * read as the probability of true, the choice read as the best choice value with per-value probabilities, the
     * score read as the expected level with per-level probabilities; choice and score carry the top probability
     * as confidence, the predicate carries none (vLLM's {@code make_answer}). The answer types are vLLM's own
     * literals: predicate, choice and score.
     */
    static Map<String, Object> vllm(DecisionQuestions.Question question, double[] probs) {
        Map<String, Object> answer = new LinkedHashMap<>();
        answer.put("type", question.type() == DecisionQuestions.NOUL ? "predicate"
                : DecisionPrompt.typeName(question.type()));
        answer.put("name", question.name());
        if (question.type() == DecisionQuestions.NOUL) {
            answer.put("probability", probs[probs.length - 1]);
            return answer;
        }
        if (question.type() == DecisionQuestions.CHOICE) {
            answer.put("choice", question.choiceValues().get(maxIndex(probs)));
            List<Map<String, Object>> probabilities = new ArrayList<>();
            for (int i = 0; i < probs.length; i++) {
                Map<String, Object> row = new LinkedHashMap<>();
                row.put("value", question.choiceValues().get(i));
                row.put("probability", probs[i]);
                probabilities.add(row);
            }
            answer.put("probabilities", probabilities);
            answer.put("confidence", probs[maxIndex(probs)]);
            return answer;
        }
        double expected = 0.0;
        List<Map<String, Object>> probabilities = new ArrayList<>();
        for (int i = 0; i < probs.length; i++) {
            expected += i * probs[i];
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("value", i);
            row.put("label", question.levelLabels().get(i));
            row.put("probability", probs[i]);
            probabilities.add(row);
        }
        answer.put("score", expected);
        answer.put("probabilities", probabilities);
        answer.put("confidence", probs[maxIndex(probs)]);
        return answer;
    }

    /** A JSON value as its compact canonical dump: the legend's form for non-text descriptions. */
    private static String jsonDump(Object value) {
        return com.eonmux.neuralfree.utils.json.Json.toString(value, com.eonmux.neuralfree.utils.json.Json.Style.CANONICAL);
    }

    private static double max(double[] probs) {
        double out = Double.NEGATIVE_INFINITY;
        for (double p : probs) {
            out = Math.max(out, p);
        }
        return out;
    }

    private static int maxIndex(double[] probs) {
        int out = 0;
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > probs[out]) {
                out = i;
            }
        }
        return out;
    }
}
