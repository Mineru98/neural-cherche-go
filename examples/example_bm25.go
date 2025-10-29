package main

import (
	"fmt"

	neuralcherche "github.com/Mineru98/neural-cherche-go"
	"github.com/Mineru98/neural-cherche-go/retrieve"
)

func main() {
	// Create documents
	documents := []neuralcherche.Document{
		{"id": "0", "document": "Food"},
		{"id": "1", "document": "Sports"},
		{"id": "2", "document": "Cinema"},
	}

	// Create BM25 retriever
	// Parameters: key, on, minN, maxN, analyzer, k1, b, epsilon
	retriever := retrieve.NewBM25("id", []string{"document"}, 3, 5, "char_wb", 1.5, 0.75, 0.0)

	// Encode and add documents
	documentEmbeddings, err := retriever.EncodeDocuments(documents)
	if err != nil {
		panic(err)
	}

	err = retriever.Add(documentEmbeddings)
	if err != nil {
		panic(err)
	}

	// Create queries
	queries := []string{"Food", "Sports", "Cinema", "Cinema food sports"}

	// Encode queries
	queryEmbeddings, err := retriever.EncodeQueries(queries)
	if err != nil {
		panic(err)
	}

	// Search
	results, err := retriever.Search(queryEmbeddings, 3)
	if err != nil {
		panic(err)
	}

	// Print results
	for i, queryResults := range results {
		fmt.Printf("\nQuery: %s\n", queries[i])
		for _, result := range queryResults {
			fmt.Printf("  ID: %s, Similarity: %.4f\n", result.Document["id"], result.Similarity)
		}
	}
}
