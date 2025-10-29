package tokenizer

import (
	"strings"
)

// CharNGramTokenizer tokenizes text into character n-grams
//
// This implementation matches sklearn's CountVectorizer/TfidfVectorizer with:
//   - analyzer="char": Character-level n-grams, preserving whitespace
//   - analyzer="char_wb": Character n-grams within word boundaries
//
// Key differences from traditional tokenizers:
//   - "char" analyzer includes whitespace in n-grams (matching sklearn)
//   - "char_wb" analyzer adds space markers around words before n-gramming
type CharNGramTokenizer struct {
	MinN       int
	MaxN       int
	Analyzer   string // "char" or "char_wb" (word boundary)
	Lowercase  bool
	StopWords  map[string]bool
	Vocabulary map[string]int
}

// NewCharNGramTokenizer creates a new character n-gram tokenizer
func NewCharNGramTokenizer(minN, maxN int, analyzer string) *CharNGramTokenizer {
	return &CharNGramTokenizer{
		MinN:       minN,
		MaxN:       maxN,
		Analyzer:   analyzer,
		Lowercase:  true,
		StopWords:  make(map[string]bool),
		Vocabulary: make(map[string]int),
	}
}

// Tokenize converts text into tokens
func (t *CharNGramTokenizer) Tokenize(text string) []string {
	if t.Lowercase {
		text = strings.ToLower(text)
	}

	var tokens []string

	if t.Analyzer == "char_wb" {
		// Word boundary: add spaces around words
		// This matches sklearn's char_wb analyzer
		words := strings.Fields(text)
		for _, word := range words {
			if t.StopWords[word] {
				continue
			}
			// Add word boundary markers
			word = " " + word + " "
			tokens = append(tokens, t.extractNGrams(word)...)
		}
	} else {
		// Pure character n-grams
		// This matches sklearn's char analyzer
		// Note: Unlike some implementations, sklearn's "char" analyzer
		// PRESERVES whitespace characters in the n-grams
		tokens = t.extractNGrams(text)
	}

	return tokens
}

// extractNGrams extracts n-grams from a string
func (t *CharNGramTokenizer) extractNGrams(text string) []string {
	runes := []rune(text)
	var ngrams []string

	for n := t.MinN; n <= t.MaxN; n++ {
		for i := 0; i <= len(runes)-n; i++ {
			ngram := string(runes[i : i+n])
			ngrams = append(ngrams, ngram)
		}
	}

	return ngrams
}

// FitVocabulary builds vocabulary from a list of texts
func (t *CharNGramTokenizer) FitVocabulary(texts []string) {
	t.Vocabulary = make(map[string]int)
	idx := 0

	for _, text := range texts {
		tokens := t.Tokenize(text)
		for _, token := range tokens {
			if _, exists := t.Vocabulary[token]; !exists {
				t.Vocabulary[token] = idx
				idx++
			}
		}
	}
}

// Transform converts tokens to indices
func (t *CharNGramTokenizer) Transform(text string) map[int]float64 {
	tokens := t.Tokenize(text)
	counts := make(map[int]float64)

	for _, token := range tokens {
		if idx, exists := t.Vocabulary[token]; exists {
			counts[idx]++
		}
	}

	return counts
}

// VocabularySize returns the size of the vocabulary
func (t *CharNGramTokenizer) VocabularySize() int {
	return len(t.Vocabulary)
}
