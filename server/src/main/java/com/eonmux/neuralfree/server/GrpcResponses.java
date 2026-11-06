package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.grpc.FinishInfo;
import com.eonmux.neuralfree.grpc.GenerateResponse;
import com.eonmux.neuralfree.grpc.PromptInfo;
import com.eonmux.neuralfree.grpc.PromptTokenLogprob;
import com.eonmux.neuralfree.grpc.SequenceOutput;
import com.eonmux.neuralfree.grpc.TokenInfo;
import com.eonmux.neuralfree.grpc.TokenLogprob;

import java.util.List;

/**
 * The gRPC response bodies of one generation: the prompt info of the first response, the sequence outputs of the
 * generated pieces and the finish info of the final one. A token's capture holds the sampled token first and its
 * alternatives after it, as the engine captures them; the alternatives are the {@code candidates} the request asked
 * for. An impossible token's log probability reports the OpenAI floor, as the HTTP surfaces do.
 */
final class GrpcResponses {

    private GrpcResponses() {
    }

    /** A response carrying the prompt info, the first response of a request. */
    static GenerateResponse promptResponse(PromptInfo prompt) {
        return GenerateResponse.newBuilder().setPromptInfo(prompt).build();
    }

    /** The prompt info of {@code batch}, with the token ids and log probabilities the request asked for. */
    static PromptInfo promptInfo(GenerationBatch batch, GrpcRequestPlan plan) {
        int[] tokens = batch.promptTokenIds(0);
        PromptInfo.Builder info = PromptInfo.newBuilder()
                .setNumPromptTokens(tokens.length)
                .setNumCachedTokens((int) Math.min(Integer.MAX_VALUE, batch.cachedTokens(0)));
        if (plan.promptTokenIds()) {
            for (int token : tokens) {
                info.addTokenIds(token);
            }
        }
        List<TokenScore[]> captures = null;
        for (int choice = 0; choice < batch.size() && captures == null; choice++) {
            captures = batch.promptLogprobs(choice);
        }
        if (captures != null) {
            for (int i = 0; i < captures.size(); i++) {
                TokenScore[] capture = captures.get(i);
                if (capture != null && capture.length > 0) {
                    info.addLogprobs(PromptTokenLogprob.newBuilder().setTokenIndex(i).setToken(logprob(capture, true)));
                }
            }
        }
        return info.build();
    }

    /** The output of one streamed piece; its finish info comes with the sequence's final output. */
    static SequenceOutput chunk(int choice, GenerationBatch.Text piece, GrpcRequestPlan plan) {
        SequenceOutput.Builder out = SequenceOutput.newBuilder().setIndex(choice);
        if (plan.outputText()) {
            out.setText(piece.text());
        }
        if (plan.outputTokenIds() && piece.tokenId() >= 0) {
            out.addTokenIds(piece.tokenId());
        }
        out.setNumTokens(plan.outputTokenIds() && piece.tokenId() >= 0 ? 1 : 0);
        if ((plan.outputLogprobs() || plan.outputCandidates()) && piece.logprobs() != null) {
            out.addLogprobs(logprob(piece.logprobs(), plan.outputCandidates()));
        }
        return out.build();
    }

    /** The final output of a finished sequence: the whole text and tokens for a single response, its finish info. */
    static SequenceOutput finalOutput(EngineService service, int choice, GenerationResult result,
                                      GenerationBatch batch, GrpcRequestPlan plan, boolean streamed) {
        SequenceOutput.Builder out = SequenceOutput.newBuilder().setIndex(choice);
        int[] tokens = result.tokens();
        out.setNumTokens(tokens.length);
        if (!streamed) {
            if (plan.outputText()) {
                out.setText(result.text());
            }
            if (plan.outputTokenIds()) {
                for (int token : tokens) {
                    out.addTokenIds(token);
                }
            }
            if (plan.outputLogprobs() || plan.outputCandidates()) {
                for (TokenScore[] step : result.logprobs()) {
                    out.addLogprobs(logprob(step, plan.outputCandidates()));
                }
            }
        }
        out.setFinishInfo(finishInfo(service, choice, result, batch));
        return out.build();
    }

    /** A sequence's finish info: the reason, the stop that named it, the counts. */
    static FinishInfo finishInfo(EngineService service, int choice, GenerationResult result, GenerationBatch batch) {
        FinishInfo.Builder finish = FinishInfo.newBuilder()
                .setNumOutputTokens(result.tokens().length)
                .setNumCachedTokens((int) Math.min(Integer.MAX_VALUE, batch.cachedTokens(choice)));
        StopReason reason = result.stopReason();
        if (reason == null) {
            return finish.setFinishReason(FinishInfo.FinishReason.NOT_FINISHED).build();
        }
        switch (reason) {
            case EOS:
                finish.setFinishReason(FinishInfo.FinishReason.STOP);
                int stop = result.stopTokenId();
                if (stop >= 0) {
                    finish.setStopTokenId(stop);
                } else {
                    int eos = service.tokenizer().eosId();
                    if (eos >= 0) {
                        finish.setEosTokenId(eos);
                    }
                }
                break;
            case STOP_STRING:
                finish.setFinishReason(FinishInfo.FinishReason.STOP);
                if (result.stopString() != null) {
                    finish.setStopString(result.stopString());
                } else if (result.stopTokenId() >= 0) {
                    finish.setStopTokenId(result.stopTokenId());
                }
                break;
            case CANCELLED:
                finish.setFinishReason(FinishInfo.FinishReason.ABORTED);
                break;
            default:
                finish.setFinishReason(FinishInfo.FinishReason.LENGTH);
                break;
        }
        return finish.build();
    }

    /** One token's log probability with the alternatives after the sampled token. */
    static TokenLogprob logprob(TokenScore[] step, boolean candidates) {
        TokenScore sampled = step[0];
        TokenLogprob.Builder out = TokenLogprob.newBuilder()
                .setTokenId(sampled.tokenId())
                .setLogprob(value(sampled))
                .setRank(Math.max(1, sampled.rank()));
        if (candidates) {
            for (int i = 1; i < step.length; i++) {
                out.addCandidates(TokenInfo.newBuilder()
                        .setId(step[i].tokenId())
                        .setLogprob(value(step[i]))
                        .setRank(Math.max(1, step[i].rank())));
            }
        }
        return out.build();
    }

    /** The log probability reported for a token; an impossible token reports the OpenAI floor. */
    private static float value(TokenScore score) {
        float logprob = score.logprob();
        return Float.isFinite(logprob) ? logprob : (float) OpenAiBodies.FLOOR_LOGPROB;
    }
}
