package retrieve

import (
	"fmt"
	"math"

	neuralcherche "github.com/Mineru98/neural-cherche-go"
	"github.com/Mineru98/neural-cherche-go/tokenizer"
	"github.com/Mineru98/neural-cherche-go/utils"
)

// SparseVector represents a sparse vector using a map
type SparseVector map[int]float64

// TfIdf implements TF-IDF retriever
//
// This implementation matches the Python neural_cherche TfIdf retriever.
// Default parameters to match Python:
//   - analyzer: "char" (pure character n-grams, no word boundaries)
//   - ngram_range: (3, 5)
//   - normalize: true (L2 normalization applied to document vectors)
type TfIdf struct {
	Key        string
	On         []string
	Tokenizer  *tokenizer.CharNGramTokenizer
	Documents  []neuralcherche.Document
	Matrix     []SparseVector // Transposed: [feature][doc] -> score
	IDF        []float64
	NDocuments int
	fitted     bool
}

// NewTfIdf creates a new TF-IDF retriever
//
// Parameters:
//   - key: Field identifier of each document
//   - on: Fields to use to match the query to the documents
//   - minN, maxN: N-gram range (Python default: 3, 5)
//   - analyzer: "char" or "char_wb" (Python default: "char")
//
// Example (matching Python defaults):
//
//	retriever := NewTfIdf("id", []string{"document"}, 3, 5, "char")
func NewTfIdf(key string, on []string, minN, maxN int, analyzer string) *TfIdf {
	return &TfIdf{
		Key:       key,
		On:        on,
		Tokenizer: tokenizer.NewCharNGramTokenizer(minN, maxN, analyzer),
		Documents: make([]neuralcherche.Document, 0),
		Matrix:    make([]SparseVector, 0),
		fitted:    false,
	}
}

// EncodeDocuments encodes documents into TF-IDF vectors
func (tf *TfIdf) EncodeDocuments(documents []neuralcherche.Document) (map[string]neuralcherche.DocumentEmbedding, error) {
	contents := make([]string, len(documents))
	for i, doc := range documents {
		contents[i] = tf.joinFields(doc)
	}

	// Fit vocabulary on first call
	if !tf.fitted {
		tf.Tokenizer.FitVocabulary(contents)
		tf.fitted = true
	}

	// Compute TF (term frequency) for each document
	embeddings := make(map[string]neuralcherche.DocumentEmbedding)
	for i, doc := range documents {
		tfVector := tf.Tokenizer.Transform(contents[i])

		// Normalize by document length
		var sum float64
		for _, count := range tfVector {
			sum += count
		}
		if sum > 0 {
			for k := range tfVector {
				tfVector[k] /= sum
			}
		}

		key := doc[tf.Key]
		embeddings[key] = SparseVector(tfVector)
	}

	return embeddings, nil
}

// EncodeQueries encodes queries into TF-IDF vectors
func (tf *TfIdf) EncodeQueries(queries []string) (map[string]neuralcherche.QueryEmbedding, error) {
	if !tf.fitted {
		return nil, fmt.Errorf("must call EncodeDocuments first to fit vocabulary")
	}

	embeddings := make(map[string]neuralcherche.QueryEmbedding)
	for _, query := range queries {
		tfVector := tf.Tokenizer.Transform(query)

		// Normalize by query length
		var sum float64
		for _, count := range tfVector {
			sum += count
		}
		if sum > 0 {
			for k := range tfVector {
				tfVector[k] /= sum
			}
		}

		embeddings[query] = SparseVector(tfVector)
	}

	return embeddings, nil
}

// Add adds documents to the index and computes IDF
func (tf *TfIdf) Add(documentEmbeddings map[string]neuralcherche.DocumentEmbedding) error {
	vocabSize := tf.Tokenizer.VocabularySize()

	// Initialize matrix if needed
	if len(tf.Matrix) == 0 {
		tf.Matrix = make([]SparseVector, vocabSize)
		for i := range tf.Matrix {
			tf.Matrix[i] = make(SparseVector)
		}
	}

	// Add documents to matrix (transposed for efficient retrieval)
	for key, emb := range documentEmbeddings {
		docIdx := len(tf.Documents)
		tf.Documents = append(tf.Documents, neuralcherche.Document{tf.Key: key})

		vec := emb.(SparseVector)
		for featureIdx, value := range vec {
			if featureIdx < vocabSize {
				tf.Matrix[featureIdx][docIdx] = value
			}
		}
	}

	tf.NDocuments = len(tf.Documents)

	// Compute IDF
	tf.computeIDF()

	// Apply IDF to matrix
	for featureIdx := range tf.Matrix {
		idf := tf.IDF[featureIdx]
		for docIdx := range tf.Matrix[featureIdx] {
			tf.Matrix[featureIdx][docIdx] *= idf
		}
	}

	// Normalize document vectors
	tf.normalizeDocuments()

	return nil
}

// computeIDF computes inverse document frequency
func (tf *TfIdf) computeIDF() {
	vocabSize := tf.Tokenizer.VocabularySize()
	tf.IDF = make([]float64, vocabSize)

	for featureIdx := range tf.Matrix {
		df := len(tf.Matrix[featureIdx]) // document frequency
		if df > 0 {
			// IDF = log((N + 1) / (df + 1)) + 1
			tf.IDF[featureIdx] = math.Log(float64(tf.NDocuments+1)/float64(df+1)) + 1
		} else {
			tf.IDF[featureIdx] = 1
		}
	}
}

// normalizeDocuments normalizes document vectors to unit length
func (tf *TfIdf) normalizeDocuments() {
	docNorms := make([]float64, tf.NDocuments)

	// Compute norms
	for featureIdx := range tf.Matrix {
		for docIdx, value := range tf.Matrix[featureIdx] {
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
	for featureIdx := range tf.Matrix {
		for docIdx := range tf.Matrix[featureIdx] {
			tf.Matrix[featureIdx][docIdx] /= docNorms[docIdx]
		}
	}
}

// Search retrieves top-k documents for queries
func (tf *TfIdf) Search(queryEmbeddings map[string]neuralcherche.QueryEmbedding, k int) ([][]neuralcherche.SearchResult, error) {
	if k > tf.NDocuments {
		k = tf.NDocuments
	}

	results := make([][]neuralcherche.SearchResult, 0, len(queryEmbeddings))

	for _, emb := range queryEmbeddings {
		queryVec := emb.(SparseVector)

		// Apply IDF to query
		for featureIdx := range queryVec {
			queryVec[featureIdx] *= tf.IDF[featureIdx]
		}

		// Compute similarities with all documents
		scores := make([]float64, tf.NDocuments)
		for featureIdx, queryValue := range queryVec {
			if featureIdx < len(tf.Matrix) {
				for docIdx, docValue := range tf.Matrix[featureIdx] {
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
					Document:   tf.Documents[idx],
					Similarity: topScores[i],
				})
			}
		}

		results = append(results, queryResults)
	}

	return results, nil
}

// joinFields joins multiple document fields into one string
func (tf *TfIdf) joinFields(doc neuralcherche.Document) string {
	parts := make([]string, 0, len(tf.On))
	for _, field := range tf.On {
		if value, exists := doc[field]; exists {
			parts = append(parts, value)
		}
	}
	return fmt.Sprintf("%s", parts)
}
