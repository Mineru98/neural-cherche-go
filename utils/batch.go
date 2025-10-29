package utils

import (
	"fmt"
	"sync"
)

// Batchify splits a slice into batches of specified size
func Batchify[T any](items []T, batchSize int) [][]T {
	if batchSize <= 0 {
		panic("batch size must be positive")
	}

	batches := make([][]T, 0, (len(items)+batchSize-1)/batchSize)
	for i := 0; i < len(items); i += batchSize {
		end := i + batchSize
		if end > len(items) {
			end = len(items)
		}
		batches = append(batches, items[i:end])
	}
	return batches
}

// BatchProcess processes items in batches with a worker function
func BatchProcess[T any, R any](
	items []T,
	batchSize int,
	worker func(batch []T) ([]R, error),
) ([]R, error) {
	batches := Batchify(items, batchSize)
	results := make([]R, 0, len(items))

	for i, batch := range batches {
		batchResults, err := worker(batch)
		if err != nil {
			return nil, fmt.Errorf("batch %d failed: %w", i, err)
		}
		results = append(results, batchResults...)
	}

	return results, nil
}

// BatchProcessParallel processes items in batches concurrently
func BatchProcessParallel[T any, R any](
	items []T,
	batchSize int,
	workers int,
	worker func(batch []T) ([]R, error),
) ([]R, error) {
	batches := Batchify(items, batchSize)

	type result struct {
		index   int
		results []R
		err     error
	}

	resultChan := make(chan result, len(batches))
	batchChan := make(chan struct {
		index int
		batch []T
	}, len(batches))

	// Start workers
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range batchChan {
				batchResults, err := worker(job.batch)
				resultChan <- result{
					index:   job.index,
					results: batchResults,
					err:     err,
				}
			}
		}()
	}

	// Send batches to workers
	go func() {
		for i, batch := range batches {
			batchChan <- struct {
				index int
				batch []T
			}{i, batch}
		}
		close(batchChan)
	}()

	// Wait for workers to finish
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect results
	resultMap := make(map[int][]R)
	for res := range resultChan {
		if res.err != nil {
			return nil, fmt.Errorf("batch %d failed: %w", res.index, res.err)
		}
		resultMap[res.index] = res.results
	}

	// Reconstruct ordered results
	results := make([]R, 0, len(items))
	for i := 0; i < len(batches); i++ {
		results = append(results, resultMap[i]...)
	}

	return results, nil
}

// ProgressBar is a simple progress indicator
type ProgressBar struct {
	total   int
	current int
	desc    string
	mu      sync.Mutex
}

// NewProgressBar creates a new progress bar
func NewProgressBar(total int, desc string) *ProgressBar {
	return &ProgressBar{
		total: total,
		desc:  desc,
	}
}

// Increment increments the progress bar
func (pb *ProgressBar) Increment() {
	pb.mu.Lock()
	defer pb.mu.Unlock()
	pb.current++
	if pb.current%10 == 0 || pb.current == pb.total {
		fmt.Printf("\r%s: %d/%d (%.1f%%)", pb.desc, pb.current, pb.total,
			float64(pb.current)/float64(pb.total)*100)
	}
	if pb.current == pb.total {
		fmt.Println()
	}
}

// Finish completes the progress bar
func (pb *ProgressBar) Finish() {
	pb.mu.Lock()
	defer pb.mu.Unlock()
	pb.current = pb.total
	fmt.Printf("\r%s: %d/%d (100.0%%)\n", pb.desc, pb.total, pb.total)
}
