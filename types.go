package neuralcherche

// Document represents a searchable document with key-value fields
type Document map[string]string

// QueryEmbedding represents the embedding of a query
type QueryEmbedding interface{}

// DocumentEmbedding represents the embedding of a document
type DocumentEmbedding interface{}

// SearchResult represents a single search result with similarity score
type SearchResult struct {
	Document   Document
	Similarity float64
}

// Retriever is the interface for all retriever implementations
type Retriever interface {
	// EncodeDocuments encodes a list of documents into embeddings
	EncodeDocuments(documents []Document) (map[string]DocumentEmbedding, error)

	// EncodeQueries encodes a list of queries into embeddings
	EncodeQueries(queries []string) (map[string]QueryEmbedding, error)

	// Add adds document embeddings to the retriever's index
	Add(documentEmbeddings map[string]DocumentEmbedding) error

	// Search retrieves top-k documents for given query embeddings
	Search(queryEmbeddings map[string]QueryEmbedding, k int) ([][]SearchResult, error)
}

// Ranker is the interface for all ranker implementations
type Ranker interface {
	// EncodeDocuments encodes documents into embeddings
	EncodeDocuments(documents []Document) (map[string]DocumentEmbedding, error)

	// EncodeQueries encodes queries into embeddings
	EncodeQueries(queries []string) (map[string]QueryEmbedding, error)

	// Rank re-ranks candidate documents for each query
	Rank(
		documents [][]Document,
		queryEmbeddings map[string]QueryEmbedding,
		documentEmbeddings map[string]DocumentEmbedding,
		k int,
	) ([][]SearchResult, error)
}

// Model is the interface for neural network models
type Model interface {
	// Encode encodes text into embeddings
	Encode(texts []string, queryMode bool) ([][]float32, error)

	// Close releases model resources
	Close() error
}
