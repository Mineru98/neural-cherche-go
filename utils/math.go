package utils

import "math"

// TopK returns the indices and values of top k elements
func TopK(scores []float64, k int) ([]int, []float64) {
	if k > len(scores) {
		k = len(scores)
	}

	type pair struct {
		index int
		value float64
	}

	pairs := make([]pair, len(scores))
	for i, v := range scores {
		pairs[i] = pair{i, v}
	}

	// Partial sort to get top k
	for i := 0; i < k; i++ {
		maxIdx := i
		for j := i + 1; j < len(pairs); j++ {
			if pairs[j].value > pairs[maxIdx].value {
				maxIdx = j
			}
		}
		pairs[i], pairs[maxIdx] = pairs[maxIdx], pairs[i]
	}

	indices := make([]int, k)
	values := make([]float64, k)
	for i := 0; i < k; i++ {
		indices[i] = pairs[i].index
		values[i] = pairs[i].value
	}

	return indices, values
}

// ArgSort returns the indices that would sort the slice
func ArgSort(scores []float64, descending bool) []int {
	indices := make([]int, len(scores))
	for i := range indices {
		indices[i] = i
	}

	// Simple bubble sort (can be optimized)
	for i := 0; i < len(indices); i++ {
		for j := i + 1; j < len(indices); j++ {
			swap := false
			if descending {
				swap = scores[indices[j]] > scores[indices[i]]
			} else {
				swap = scores[indices[j]] < scores[indices[i]]
			}
			if swap {
				indices[i], indices[j] = indices[j], indices[i]
			}
		}
	}

	return indices
}

// DotProduct computes the dot product of two vectors
func DotProduct(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("vectors must have same length")
	}

	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// Norm computes the L2 norm of a vector
func Norm(v []float64) float64 {
	var sum float64
	for _, x := range v {
		sum += x * x
	}
	return math.Sqrt(sum)
}

// CosineSimilarity computes cosine similarity between two vectors
func CosineSimilarity(a, b []float64) float64 {
	dot := DotProduct(a, b)
	normA := Norm(a)
	normB := Norm(b)

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (normA * normB)
}

// Normalize normalizes a vector to unit length
func Normalize(v []float64) []float64 {
	norm := Norm(v)
	if norm == 0 {
		return v
	}

	result := make([]float64, len(v))
	for i, x := range v {
		result[i] = x / norm
	}
	return result
}
