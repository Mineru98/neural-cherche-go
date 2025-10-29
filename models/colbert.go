package models

import (
	"fmt"
	"math"
)

// ColBERT represents a ColBERT model with ONNX backend
type ColBERT struct {
	model             *ONNXModel
	maxLengthQuery    int
	maxLengthDocument int
	embeddingSize     int
	queryPrefix       string
	documentPrefix    string
	linearWeights     [][]float32 // [embeddingSize][hiddenSize]
	hasLinearLayer    bool
}

// ColBERTConfig holds configuration for ColBERT model
type ColBERTConfig struct {
	ModelPath         string
	MaxLengthQuery    int
	MaxLengthDocument int
	EmbeddingSize     int
	QueryPrefix       string
	DocumentPrefix    string
	LinearWeights     [][]float32 // Optional linear projection weights
}

// NewColBERT creates a new ColBERT model
func NewColBERT(config ColBERTConfig) (*ColBERT, error) {
	model, err := NewONNXModel(config.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load ONNX model: %w", err)
	}

	colbert := &ColBERT{
		model:             model,
		maxLengthQuery:    config.MaxLengthQuery,
		maxLengthDocument: config.MaxLengthDocument,
		embeddingSize:     config.EmbeddingSize,
		queryPrefix:       config.QueryPrefix,
		documentPrefix:    config.DocumentPrefix,
		linearWeights:     config.LinearWeights,
		hasLinearLayer:    len(config.LinearWeights) > 0,
	}

	if colbert.queryPrefix == "" {
		colbert.queryPrefix = "[Q] "
	}
	if colbert.documentPrefix == "" {
		colbert.documentPrefix = "[D] "
	}

	return colbert, nil
}

// Encode encodes texts into embeddings
// Returns: [batch_size, seq_len, embedding_size]
func (c *ColBERT) Encode(texts []string, queryMode bool) ([][][]float32, error) {
	// Add prefix
	prefix := c.documentPrefix
	if queryMode {
		prefix = c.queryPrefix
	}

	prefixedTexts := make([]string, len(texts))
	for i, text := range texts {
		prefixedTexts[i] = prefix + text
	}

	// Note: Actual tokenization should be done with a proper tokenizer
	// For now, this is a placeholder
	// In real implementation, you'd use HuggingFace tokenizer
	// This method assumes inputs are already tokenized
	return nil, fmt.Errorf("tokenization not implemented yet - use EncodeTokenized instead")
}

// EncodeTokenized encodes pre-tokenized input
func (c *ColBERT) EncodeTokenized(
	inputIDs [][]int64,
	attentionMask [][]int64,
	queryMode bool,
) ([][][]float32, error) {
	// Prepare inputs for ONNX model
	inputs := map[string]interface{}{
		"input_ids":      inputIDs,
		"attention_mask": attentionMask,
	}

	// Run inference
	outputs, err := c.model.Run(inputs)
	if err != nil {
		return nil, fmt.Errorf("model inference failed: %w", err)
	}

	// Extract hidden states (last layer)
	// Output format depends on the model, typically "last_hidden_state"
	hiddenStatesOutput, exists := outputs["last_hidden_state"]
	if !exists {
		// Try alternative output names
		if hs, ok := outputs["hidden_states"]; ok {
			hiddenStatesOutput = hs
		} else {
			return nil, fmt.Errorf("could not find hidden states in model output")
		}
	}

	hiddenStatesMap := hiddenStatesOutput.(map[string]interface{})
	hiddenStatesFlat := hiddenStatesMap["data"].([]float32)
	shape := hiddenStatesMap["shape"].([]int64)

	// Reshape to [batch_size, seq_len, hidden_size]
	batchSize := int(shape[0])
	seqLen := int(shape[1])
	hiddenSize := int(shape[2])

	embeddings := make([][][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		embeddings[i] = make([][]float32, seqLen)
		for j := 0; j < seqLen; j++ {
			embeddings[i][j] = make([]float32, hiddenSize)
			startIdx := (i*seqLen + j) * hiddenSize
			copy(embeddings[i][j], hiddenStatesFlat[startIdx:startIdx+hiddenSize])
		}
	}

	// Apply linear projection if available
	if c.hasLinearLayer {
		embeddings = c.applyLinearProjection(embeddings, attentionMask)
	}

	// Apply attention mask
	embeddings = c.applyAttentionMask(embeddings, attentionMask)

	// Normalize embeddings (L2 normalization)
	embeddings = c.normalizeEmbeddings(embeddings)

	return embeddings, nil
}

// applyLinearProjection applies linear projection to reduce embedding dimension
func (c *ColBERT) applyLinearProjection(
	embeddings [][][]float32,
	attentionMask [][]int64,
) [][][]float32 {
	batchSize := len(embeddings)
	seqLen := len(embeddings[0])
	hiddenSize := len(embeddings[0][0])
	embSize := c.embeddingSize

	projected := make([][][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		projected[i] = make([][]float32, seqLen)
		for j := 0; j < seqLen; j++ {
			projected[i][j] = make([]float32, embSize)

			// Matrix multiplication: embeddings[i][j] @ linearWeights^T
			for k := 0; k < embSize; k++ {
				var sum float32
				for l := 0; l < hiddenSize; l++ {
					sum += embeddings[i][j][l] * c.linearWeights[k][l]
				}
				projected[i][j][k] = sum
			}
		}
	}

	return projected
}

// applyAttentionMask applies attention mask to embeddings
func (c *ColBERT) applyAttentionMask(
	embeddings [][][]float32,
	attentionMask [][]int64,
) [][][]float32 {
	batchSize := len(embeddings)
	seqLen := len(embeddings[0])
	embSize := len(embeddings[0][0])

	masked := make([][][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		masked[i] = make([][]float32, seqLen)
		for j := 0; j < seqLen; j++ {
			masked[i][j] = make([]float32, embSize)
			if j < len(attentionMask[i]) && attentionMask[i][j] == 1 {
				copy(masked[i][j], embeddings[i][j])
			}
			// Else leave as zeros
		}
	}

	return masked
}

// normalizeEmbeddings applies L2 normalization to embeddings
func (c *ColBERT) normalizeEmbeddings(embeddings [][][]float32) [][][]float32 {
	batchSize := len(embeddings)
	seqLen := len(embeddings[0])
	embSize := len(embeddings[0][0])

	normalized := make([][][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		normalized[i] = make([][]float32, seqLen)
		for j := 0; j < seqLen; j++ {
			// Compute L2 norm
			var norm float32
			for k := 0; k < embSize; k++ {
				norm += embeddings[i][j][k] * embeddings[i][j][k]
			}
			norm = float32(math.Sqrt(float64(norm)))

			// Normalize
			normalized[i][j] = make([]float32, embSize)
			if norm > 0 {
				for k := 0; k < embSize; k++ {
					normalized[i][j][k] = embeddings[i][j][k] / norm
				}
			}
		}
	}

	return normalized
}

// Scores computes ColBERT scores between queries and documents
func (c *ColBERT) Scores(
	queryEmbeddings [][][]float32,
	documentEmbeddings [][][]float32,
) []float64 {
	batchSize := len(queryEmbeddings)
	scores := make([]float64, batchSize)

	for i := 0; i < batchSize; i++ {
		scores[i] = MaxSim(queryEmbeddings[i], documentEmbeddings[i])
	}

	return scores
}

// MaxSim computes the MaxSim score between query and document
// query: [query_len, embedding_size]
// document: [doc_len, embedding_size]
func MaxSim(query [][]float32, document [][]float32) float64 {
	queryLen := len(query)
	docLen := len(document)

	if queryLen == 0 || docLen == 0 {
		return 0
	}

	embSize := len(query[0])

	var totalScore float64

	// For each query token, find max similarity with any document token
	for i := 0; i < queryLen; i++ {
		var maxSim float32

		for j := 0; j < docLen; j++ {
			// Compute dot product (cosine similarity since vectors are normalized)
			var sim float32
			for k := 0; k < embSize; k++ {
				sim += query[i][k] * document[j][k]
			}

			if sim > maxSim {
				maxSim = sim
			}
		}

		totalScore += float64(maxSim)
	}

	return totalScore
}

// Close releases model resources
func (c *ColBERT) Close() error {
	if c.model != nil {
		return c.model.Close()
	}
	return nil
}
