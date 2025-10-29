package retrieve

import (
	"fmt"
	"math"

	"github.com/Mineru98/neural-cherche-go"
	"github.com/Mineru98/neural-cherche-go/tokenizer"
	"github.com/Mineru98/neural-cherche-go/utils"
)

// BM25 implements BM25 retriever
type BM25 struct {
	Key        string
	On         []string
	Tokenizer  *tokenizer.CharNGramTokenizer
	Documents  []neuralcherche.Document
	Matrix     []SparseVector // Transposed: [feature][doc] -> BM25 score component
	K1         float64        // Term frequency saturation parameter
	B          float64        // Length normalization parameter
	Epsilon    float64        // Smoothing term
	NDocuments int
	AvgDocLen  float64
	DocLengths []float64
	TF         []float64 // Total term frequencies across all documents
	fitted     bool
}

// NewBM25 creates a new BM25 retriever
func NewBM25(key string, on []string, minN, maxN int, analyzer string, k1, b, epsilon float64) *BM25 {
	return &BM25{
		Key:        key,
		On:         on,
		Tokenizer:  tokenizer.NewCharNGramTokenizer(minN, maxN, analyzer),
		Documents:  make([]neuralcherche.Document, 0),
		Matrix:     make([]SparseVector, 0),
		K1:         k1,
		B:          b,
		Epsilon:    epsilon,
		DocLengths: make([]float64, 0),
		fitted:     false,
	}
}

// EncodeDocuments encodes documents into BM25 term frequency vectors
func (bm *BM25) EncodeDocuments(documents []neuralcherche.Document) (map[string]neuralcherche.DocumentEmbedding, error) {
	contents := make([]string, len(documents))
	for i, doc := range documents {
		contents[i] = bm.joinFields(doc)
	}

	// Fit vocabulary on first call
	if !bm.fitted {
		bm.Tokenizer.FitVocabulary(contents)
		bm.fitted = true
	}

	// Compute raw term frequencies
	embeddings := make(map[string]neuralcherche.DocumentEmbedding)
	for i, doc := range documents {
		tfVector := bm.Tokenizer.Transform(contents[i])
		key := doc[bm.Key]
		embeddings[key] = SparseVector(tfVector)
	}

	return embeddings, nil
}

// EncodeQueries encodes queries into term frequency vectors
func (bm *BM25) EncodeQueries(queries []string) (map[string]neuralcherche.QueryEmbedding, error) {
	if !bm.fitted {
		return nil, fmt.Errorf("must call EncodeDocuments first to fit vocabulary")
	}

	embeddings := make(map[string]neuralcherche.QueryEmbedding)
	for _, query := range queries {
		tfVector := bm.Tokenizer.Transform(query)
		embeddings[query] = SparseVector(tfVector)
	}

	return embeddings, nil
}

// Add adds documents to the index and computes BM25 weights
func (bm *BM25) Add(documentEmbeddings map[string]neuralcherche.DocumentEmbedding) error {
	vocabSize := bm.Tokenizer.VocabularySize()

	// Initialize matrix and TF if needed
	if len(bm.Matrix) == 0 {
		bm.Matrix = make([]SparseVector, vocabSize)
		bm.TF = make([]float64, vocabSize)
		for i := range bm.Matrix {
			bm.Matrix[i] = make(SparseVector)
		}
	}

	// Process documents
	startIdx := len(bm.Documents)
	for key, emb := range documentEmbeddings {
		docIdx := len(bm.Documents)
		bm.Documents = append(bm.Documents, neuralcherche.Document{bm.Key: key})

		vec := emb.(SparseVector)

		// Compute document length
		var docLen float64
		for _, count := range vec {
			docLen += count
		}
		bm.DocLengths = append(bm.DocLengths, docLen)

		// Add to matrix and update TF
		for featureIdx, count := range vec {
			if featureIdx < vocabSize {
				bm.Matrix[featureIdx][docIdx] = count
				bm.TF[featureIdx] += count
			}
		}
	}

	bm.NDocuments = len(bm.Documents)

	// Compute average document length
	var totalLen float64
	for _, length := range bm.DocLengths {
		totalLen += length
	}
	bm.AvgDocLen = totalLen / float64(bm.NDocuments)

	// Apply BM25 transformation to newly added documents
	bm.applyBM25Transform(startIdx)

	return nil
}

// applyBM25Transform applies BM25 formula to documents starting from startIdx
func (bm *BM25) applyBM25Transform(startIdx int) {
	vocabSize := bm.Tokenizer.VocabularySize()

	// For each feature, apply BM25 to new documents
	for featureIdx := 0; featureIdx < vocabSize; featureIdx++ {
		// Compute IDF
		df := len(bm.Matrix[featureIdx]) // document frequency
		idf := 0.0
		if df > 0 {
			// IDF = log((N - df + 0.5) / (df + 0.5) + 1)
			idf = math.Log((float64(bm.NDocuments)-float64(df)+0.5)/(float64(df)+0.5) + 1)
		}

		// Apply BM25 formula to each document
		for docIdx := startIdx; docIdx < bm.NDocuments; docIdx++ {
			if tf, exists := bm.Matrix[featureIdx][docIdx]; exists {
				// Length normalization
				docLen := bm.DocLengths[docIdx]
				normFactor := bm.K1 * (1 - bm.B + bm.B*(docLen/bm.AvgDocLen))

				// BM25 score component
				// score = (tf * (k1 + 1)) / (tf + normFactor) + epsilon
				bm25Score := (tf*(bm.K1+1))/(tf+normFactor) + bm.Epsilon

				// Multiply by IDF
				bm.Matrix[featureIdx][docIdx] = bm25Score * idf
			}
		}
	}

	// Normalize document vectors
	bm.normalizeDocuments()
}

// normalizeDocuments normalizes document vectors to unit length
func (bm *BM25) normalizeDocuments() {
	docNorms := make([]float64, bm.NDocuments)

	// Compute norms
	for featureIdx := range bm.Matrix {
		for docIdx, value := range bm.Matrix[featureIdx] {
			docNorms[docIdx] += value * value
		}
	}

	for i := range docNorms {
		if docNorms[i] > 0 {
			docNorms[i] = math.Sqrt(docNorms[i])
		} else {
			docNorms[i] = 1
		}
	}

	// Normalize
	for featureIdx := range bm.Matrix {
		for docIdx := range bm.Matrix[featureIdx] {
			if docNorms[docIdx] > 0 {
				bm.Matrix[featureIdx][docIdx] /= docNorms[docIdx]
			}
		}
	}
}

// Search retrieves top-k documents for queries
func (bm *BM25) Search(queryEmbeddings map[string]neuralcherche.QueryEmbedding, k int) ([][]neuralcherche.SearchResult, error) {
	if k > bm.NDocuments {
		k = bm.NDocuments
	}

	results := make([][]neuralcherche.SearchResult, 0, len(queryEmbeddings))

	for _, emb := range queryEmbeddings {
		queryVec := emb.(SparseVector)

		// Compute similarities with all documents
		scores := make([]float64, bm.NDocuments)
		for featureIdx, queryValue := range queryVec {
			if featureIdx < len(bm.Matrix) {
				for docIdx, docValue := range bm.Matrix[featureIdx] {
					// Query term frequency is typically 1 for BM25
					// but we use the actual count here
					scores[docIdx] += queryValue * docValue
				}
			}
		}

		// Get top-k
		topIndices, topScores := utils.TopK(scores, k)

		queryResults := make([]neuralcherche.SearchResult, 0, k)
		for i, idx := range topIndices {
			if topScores[i] > 0 {
				queryResults = append(queryResults, neuralcherche.SearchResult{
					Document:   bm.Documents[idx],
					Similarity: topScores[i],
				})
			}
		}

		results = append(results, queryResults)
	}

	return results, nil
}

// joinFields joins multiple document fields into one string
func (bm *BM25) joinFields(doc neuralcherche.Document) string {
	parts := make([]string, 0, len(bm.On))
	for _, field := range bm.On {
		if value, exists := doc[field]; exists {
			parts = append(parts, value)
		}
	}
	return fmt.Sprintf("%s", parts)
}
