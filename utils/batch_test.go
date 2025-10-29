package utils

import (
	"testing"
)

func TestBatchify(t *testing.T) {
	items := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	batchSize := 3

	batches := Batchify(items, batchSize)

	expectedBatches := 4
	if len(batches) != expectedBatches {
		t.Errorf("Expected %d batches, got %d", expectedBatches, len(batches))
	}

	expectedLengths := []int{3, 3, 3, 1}
	for i, batch := range batches {
		if len(batch) != expectedLengths[i] {
			t.Errorf("Batch %d: expected length %d, got %d", i, expectedLengths[i], len(batch))
		}
	}
}

func TestBatchifyEmpty(t *testing.T) {
	items := []int{}
	batchSize := 3

	batches := Batchify(items, batchSize)

	if len(batches) != 0 {
		t.Errorf("Expected 0 batches for empty input, got %d", len(batches))
	}
}

func TestBatchifyPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for batch size <= 0")
		}
	}()

	items := []int{1, 2, 3}
	Batchify(items, 0)
}

func TestTopK(t *testing.T) {
	scores := []float64{0.1, 0.5, 0.3, 0.9, 0.2}
	k := 3

	indices, values := TopK(scores, k)

	if len(indices) != k {
		t.Errorf("Expected %d indices, got %d", k, len(indices))
	}

	if len(values) != k {
		t.Errorf("Expected %d values, got %d", k, len(values))
	}

	// Check if values are in descending order
	for i := 0; i < len(values)-1; i++ {
		if values[i] < values[i+1] {
			t.Errorf("Values not in descending order: %v", values)
		}
	}

	// Check if first value is maximum
	if values[0] != 0.9 {
		t.Errorf("Expected maximum value 0.9, got %f", values[0])
	}
}

func TestDotProduct(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}

	expected := 1*4 + 2*5 + 3*6
	result := DotProduct(a, b)

	if result != float64(expected) {
		t.Errorf("Expected %d, got %f", expected, result)
	}
}

func TestNorm(t *testing.T) {
	v := []float64{3, 4}
	expected := 5.0

	result := Norm(v)

	if result != expected {
		t.Errorf("Expected %f, got %f", expected, result)
	}
}

func TestCosineSimilarity(t *testing.T) {
	a := []float64{1, 0}
	b := []float64{1, 0}

	result := CosineSimilarity(a, b)

	if result != 1.0 {
		t.Errorf("Expected 1.0 for identical vectors, got %f", result)
	}

	c := []float64{1, 0}
	d := []float64{0, 1}

	result = CosineSimilarity(c, d)

	if result != 0.0 {
		t.Errorf("Expected 0.0 for orthogonal vectors, got %f", result)
	}
}

func TestNormalize(t *testing.T) {
	v := []float64{3, 4}
	result := Normalize(v)

	norm := Norm(result)

	if norm < 0.999 || norm > 1.001 {
		t.Errorf("Expected normalized vector to have norm ~1.0, got %f", norm)
	}
}
