package tokenizer

import (
	"fmt"

	"github.com/daulet/tokenizers"
)

// BERTTokenizer wraps HuggingFace BERT tokenizer
type BERTTokenizer struct {
	tokenizer  *tokenizers.Tokenizer
	maxLength  int
	padding    bool
	truncation bool
}

// NewBERTTokenizer creates a new BERT tokenizer from a tokenizer.json file
func NewBERTTokenizer(tokenizerPath string, maxLength int) (*BERTTokenizer, error) {
	tk, err := tokenizers.FromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	return &BERTTokenizer{
		tokenizer:  tk,
		maxLength:  maxLength,
		padding:    true,
		truncation: true,
	}, nil
}

// Encode encodes a single text
func (bt *BERTTokenizer) Encode(text string) ([]int64, []int64, error) {
	encoding := bt.tokenizer.EncodeWithOptions(text, false)

	ids := encoding.IDs
	attentionMask := make([]int64, len(ids))
	for i := range attentionMask {
		attentionMask[i] = 1
	}

	// Truncate if needed
	if bt.truncation && len(ids) > bt.maxLength {
		ids = ids[:bt.maxLength]
		attentionMask = attentionMask[:bt.maxLength]
	}

	// Pad if needed
	if bt.padding && len(ids) < bt.maxLength {
		padding := make([]uint32, bt.maxLength-len(ids))
		ids = append(ids, padding...)
		paddingMask := make([]int64, bt.maxLength-len(attentionMask))
		attentionMask = append(attentionMask, paddingMask...)
	}

	// Convert to int64
	inputIDs := make([]int64, len(ids))
	for i, id := range ids {
		inputIDs[i] = int64(id)
	}

	return inputIDs, attentionMask, nil
}

// EncodeBatch encodes multiple texts
func (bt *BERTTokenizer) EncodeBatch(texts []string) ([][]int64, [][]int64, error) {
	inputIDs := make([][]int64, len(texts))
	attentionMasks := make([][]int64, len(texts))

	for i, text := range texts {
		ids, mask, err := bt.Encode(text)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to encode text %d: %w", i, err)
		}
		inputIDs[i] = ids
		attentionMasks[i] = mask
	}

	return inputIDs, attentionMasks, nil
}

// Close releases tokenizer resources
func (bt *BERTTokenizer) Close() error {
	if bt.tokenizer != nil {
		bt.tokenizer.Close()
		bt.tokenizer = nil
	}
	return nil
}

// VocabularySize returns the vocabulary size
func (bt *BERTTokenizer) VocabularySize() int {
	return int(bt.tokenizer.VocabSize())
}

// Decode decodes token IDs back to text
func (bt *BERTTokenizer) Decode(ids []uint32, skipSpecialTokens bool) (string, error) {
	text := bt.tokenizer.Decode(ids, skipSpecialTokens)
	return text, nil
}
