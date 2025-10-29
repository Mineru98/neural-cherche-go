package rank

import (
	"fmt"

	"github.com/Mineru98/neural-cherche-go"
	"github.com/Mineru98/neural-cherche-go/models"
	"github.com/Mineru98/neural-cherche-go/tokenizer"
	"github.com/Mineru98/neural-cherche-go/utils"
)

// ColBERTRanker implements ColBERT-based re-ranking
type ColBERTRanker struct {
	Key       string
	On        []string
	Model     *models.ColBERT
	Tokenizer *tokenizer.BERTTokenizer
}

// NewColBERTRanker creates a new ColBERT ranker
func NewColBERTRanker(
	key string,
	on []string,
	model *models.ColBERT,
	tokenizer *tokenizer.BERTTokenizer,
) *ColBERTRanker {
	return &ColBERTRanker{
		Key:       key,
		On:        on,
		Model:     model,
		Tokenizer: tokenizer,
	}
}

// EncodeDocuments encodes documents into ColBERT embeddings
func (r *ColBERTRanker) EncodeDocuments(
	documents []neuralcherche.Document,
	batchSize int,
) (map[string]neuralcherche.DocumentEmbedding, error) {
	embeddings := make(map[string]neuralcherche.DocumentEmbedding)

	// Extract document contents
	contents := make([]string, len(documents))
	for i, doc := range documents {
		contents[i] = r.joinFields(doc)
	}

	// Process in batches
	batches := utils.Batchify(contents, batchSize)
	currentIdx := 0

	for _, batch := range batches {
		// Tokenize batch
		inputIDs, attentionMask, err := r.Tokenizer.EncodeBatch(batch)
		if err != nil {
			return nil, fmt.Errorf("tokenization failed: %w", err)
		}

		// Encode with ColBERT
		batchEmbeddings, err := r.Model.EncodeTokenized(inputIDs, attentionMask, false)
		if err != nil {
			return nil, fmt.Errorf("encoding failed: %w", err)
		}

		// Store embeddings
		for i, emb := range batchEmbeddings {
			docIdx := currentIdx + i
			key := documents[docIdx][r.Key]
			embeddings[key] = emb
		}

		currentIdx += len(batch)
	}

	return embeddings, nil
}

// EncodeQueries encodes queries into ColBERT embeddings
func (r *ColBERTRanker) EncodeQueries(
	queries []string,
	batchSize int,
) (map[string]neuralcherche.QueryEmbedding, error) {
	embeddings := make(map[string]neuralcherche.QueryEmbedding)

	// Process in batches
	batches := utils.Batchify(queries, batchSize)
	currentIdx := 0

	for _, batch := range batches {
		// Tokenize batch
		inputIDs, attentionMask, err := r.Tokenizer.EncodeBatch(batch)
		if err != nil {
			return nil, fmt.Errorf("tokenization failed: %w", err)
		}

		// Encode with ColBERT in query mode
		batchEmbeddings, err := r.Model.EncodeTokenized(inputIDs, attentionMask, true)
		if err != nil {
			return nil, fmt.Errorf("encoding failed: %w", err)
		}

		// Store embeddings
		for i, emb := range batchEmbeddings {
			queryIdx := currentIdx + i
			embeddings[queries[queryIdx]] = emb
		}

		currentIdx += len(batch)
	}

	return embeddings, nil
}

// Rank re-ranks candidate documents for each query
func (r *ColBERTRanker) Rank(
	documents [][]neuralcherche.Document,
	queryEmbeddings map[string]neuralcherche.QueryEmbedding,
	documentEmbeddings map[string]neuralcherche.DocumentEmbedding,
	k int,
) ([][]neuralcherche.SearchResult, error) {
	results := make([][]neuralcherche.SearchResult, len(documents))

	queryIdx := 0
	for _, queryEmb := range queryEmbeddings {
		if queryIdx >= len(documents) {
			break
		}

		queryEmbedding := queryEmb.([][][]float32)[0] // Extract first item from batch

		candidateDocs := documents[queryIdx]
		scores := make([]float64, len(candidateDocs))

		// Compute scores for each candidate
		for i, doc := range candidateDocs {
			docKey := doc[r.Key]
			docEmb, exists := documentEmbeddings[docKey]
			if !exists {
				scores[i] = 0
				continue
			}

			docEmbedding := docEmb.([][][]float32)[0] // Extract first item from batch
			scores[i] = models.MaxSim(queryEmbedding, docEmbedding)
		}

		// Get top-k
		topK := k
		if topK > len(candidateDocs) {
			topK = len(candidateDocs)
		}

		topIndices, topScores := utils.TopK(scores, topK)

		// Build results
		queryResults := make([]neuralcherche.SearchResult, len(topIndices))
		for i, idx := range topIndices {
			queryResults[i] = neuralcherche.SearchResult{
				Document:   candidateDocs[idx],
				Similarity: topScores[i],
			}
		}

		results[queryIdx] = queryResults
		queryIdx++
	}

	// Handle remaining queries (if any)
	for i := queryIdx; i < len(documents); i++ {
		results[i] = []neuralcherche.SearchResult{}
	}

	return results, nil
}

// EncodeCandidateDocuments encodes only the candidate documents
func (r *ColBERTRanker) EncodeCandidateDocuments(
	documents []neuralcherche.Document,
	candidates [][]neuralcherche.Document,
	batchSize int,
) (map[string]neuralcherche.DocumentEmbedding, error) {
	// Create a map of all documents
	docMap := make(map[string]neuralcherche.Document)
	for _, doc := range documents {
		docMap[doc[r.Key]] = doc
	}

	// Collect unique candidate documents
	uniqueCandidates := make(map[string]neuralcherche.Document)
	for _, queryCandidates := range candidates {
		for _, candidate := range queryCandidates {
			key := candidate[r.Key]
			if _, exists := uniqueCandidates[key]; !exists {
				if doc, found := docMap[key]; found {
					uniqueCandidates[key] = doc
				}
			}
		}
	}

	// Convert to slice
	candidateList := make([]neuralcherche.Document, 0, len(uniqueCandidates))
	for _, doc := range uniqueCandidates {
		candidateList = append(candidateList, doc)
	}

	// Encode candidates
	return r.EncodeDocuments(candidateList, batchSize)
}

// joinFields joins multiple document fields into one string
func (r *ColBERTRanker) joinFields(doc neuralcherche.Document) string {
	result := ""
	for i, field := range r.On {
		if value, exists := doc[field]; exists {
			if i > 0 {
				result += " "
			}
			result += value
		}
	}
	return result
}

// Close releases ranker resources
func (r *ColBERTRanker) Close() error {
	if r.Model != nil {
		return r.Model.Close()
	}
	return nil
}
