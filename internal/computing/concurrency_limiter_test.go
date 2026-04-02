package computing

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// --- Semaphore tests ---

func TestSemaphore_AcquireRelease(t *testing.T) {
	s := NewSemaphore(3)

	if !s.Acquire(100 * time.Millisecond) {
		t.Fatal("expected acquire to succeed")
	}
	s.Release(10 * time.Millisecond)

	_, _, acquired, released, _, _, _ := s.GetStats()
	if acquired != 1 {
		t.Errorf("expected 1 acquired, got %d", acquired)
	}
	if released != 1 {
		t.Errorf("expected 1 released, got %d", released)
	}
}

func TestSemaphore_MaxConcurrent(t *testing.T) {
	s := NewSemaphore(2)

	if !s.Acquire(50 * time.Millisecond) {
		t.Fatal("expected first acquire to succeed")
	}
	if !s.Acquire(50 * time.Millisecond) {
		t.Fatal("expected second acquire to succeed")
	}

	// Third should block and timeout
	ok := s.Acquire(20 * time.Millisecond)
	if ok {
		t.Error("expected third acquire to fail (max reached)")
	}
}

func TestSemaphore_AcquireTimeout(t *testing.T) {
	s := NewSemaphore(1)
	s.Acquire(100 * time.Millisecond) // fill it

	start := time.Now()
	ok := s.Acquire(20 * time.Millisecond)
	elapsed := time.Since(start)

	if ok {
		t.Error("expected acquire to fail on full semaphore")
	}
	if elapsed < 20*time.Millisecond {
		t.Errorf("expected to wait at least 20ms, waited %v", elapsed)
	}

	_, _, _, _, _, timeouts, _ := s.GetStats()
	if timeouts < 1 {
		t.Error("expected at least 1 timeout recorded")
	}
}

func TestSemaphore_TryAcquire_NonBlocking(t *testing.T) {
	s := NewSemaphore(1)
	s.Acquire(100 * time.Millisecond)

	if s.TryAcquire() {
		t.Error("expected TryAcquire to fail on full semaphore")
	}

	_, _, _, _, rejected, _, _ := s.GetStats()
	if rejected < 1 {
		t.Error("expected rejected count > 0")
	}
}

func TestSemaphore_SetMax_Increase(t *testing.T) {
	s := NewSemaphore(1)
	s.Acquire(100 * time.Millisecond) // fill at max=1

	// Increase max, then waiter should be able to acquire
	done := make(chan bool, 1)
	go func() {
		done <- s.Acquire(200 * time.Millisecond)
	}()

	time.Sleep(10 * time.Millisecond)
	s.SetMax(2) // now there's room

	select {
	case ok := <-done:
		if !ok {
			t.Error("expected acquire to succeed after SetMax increase")
		}
	case <-time.After(300 * time.Millisecond):
		t.Error("timed out waiting for acquire after SetMax")
	}
}

func TestSemaphore_HoldTimeTracking(t *testing.T) {
	s := NewSemaphore(5)
	s.Acquire(100 * time.Millisecond)
	s.Release(50 * time.Millisecond)

	_, _, _, _, _, _, avgHold := s.GetStats()
	if avgHold != 50 {
		t.Errorf("expected avg hold time 50ms, got %v", avgHold)
	}
}

// --- ConcurrencyLimiter tests ---

func TestConcurrencyLimiter_AcquireRelease(t *testing.T) {
	cfg := DefaultConcurrencyConfig()
	cfg.AcquireTimeout = 100 * time.Millisecond
	cfg.EnableGPUAwareness = false

	cl := NewConcurrencyLimiter(cfg, nil)
	ctx := context.Background()

	token, err := cl.Acquire(ctx, "test-model")
	if err != nil {
		t.Fatalf("expected acquire to succeed, got: %v", err)
	}
	token.Release()
	token.Release() // double release should be safe

	m := cl.GetMetrics()
	if m.TotalAcquired != 1 {
		t.Errorf("expected 1 acquired, got %d", m.TotalAcquired)
	}
}

func TestConcurrencyLimiter_GlobalLimit(t *testing.T) {
	cfg := ConcurrencyConfig{
		GlobalMaxConcurrent: 2,
		DefaultModelMax:     10,
		AcquireTimeout:      20 * time.Millisecond,
		EnableGPUAwareness:  false,
	}
	cl := NewConcurrencyLimiter(cfg, nil)
	ctx := context.Background()

	t1, err := cl.Acquire(ctx, "model-a")
	if err != nil {
		t.Fatal(err)
	}
	t2, err := cl.Acquire(ctx, "model-b")
	if err != nil {
		t.Fatal(err)
	}

	// Third should fail (global limit = 2)
	_, err = cl.Acquire(ctx, "model-c")
	if err == nil {
		t.Error("expected error when global limit reached")
	}

	t1.Release()
	t2.Release()
}

func TestConcurrencyLimiter_ModelLimit(t *testing.T) {
	cfg := ConcurrencyConfig{
		GlobalMaxConcurrent: 50,
		DefaultModelMax:     2,
		AcquireTimeout:      20 * time.Millisecond,
		EnableGPUAwareness:  false,
	}
	cl := NewConcurrencyLimiter(cfg, nil)
	cl.RegisterModel("small-model", 2, 4000)
	ctx := context.Background()

	t1, _ := cl.Acquire(ctx, "small-model")
	t2, _ := cl.Acquire(ctx, "small-model")

	// Third should fail at model level
	_, err := cl.Acquire(ctx, "small-model")
	if err == nil {
		t.Error("expected error when model limit reached")
	}

	t1.Release()
	t2.Release()
}

func TestConcurrencyLimiter_TryAcquire_NonBlocking(t *testing.T) {
	cfg := ConcurrencyConfig{
		GlobalMaxConcurrent: 1,
		DefaultModelMax:     1,
		AcquireTimeout:      100 * time.Millisecond,
		EnableGPUAwareness:  false,
	}
	cl := NewConcurrencyLimiter(cfg, nil)
	token, err := cl.TryAcquire("model-x")
	if err != nil {
		t.Fatal(err)
	}

	// Now full — TryAcquire should fail immediately
	_, err = cl.TryAcquire("model-x")
	if err == nil {
		t.Error("expected immediate failure when full")
	}
	token.Release()
}

func TestConcurrencyLimiter_SetGlobalMax(t *testing.T) {
	cfg := ConcurrencyConfig{
		GlobalMaxConcurrent: 1,
		DefaultModelMax:     10,
		AcquireTimeout:      50 * time.Millisecond,
		EnableGPUAwareness:  false,
	}
	cl := NewConcurrencyLimiter(cfg, nil)
	ctx := context.Background()

	t1, _ := cl.Acquire(ctx, "model")
	_, err := cl.Acquire(ctx, "model")
	if err == nil {
		t.Error("expected error at global limit 1")
	}

	cl.SetGlobalMax(2)
	t2, err := cl.Acquire(ctx, "model")
	if err != nil {
		t.Errorf("expected success after SetGlobalMax(2), got: %v", err)
	}
	t1.Release()
	t2.Release()
}

func TestConcurrencyLimiter_SetModelMax(t *testing.T) {
	cfg := ConcurrencyConfig{
		GlobalMaxConcurrent: 10,
		DefaultModelMax:     1,
		AcquireTimeout:      50 * time.Millisecond,
		EnableGPUAwareness:  false,
	}
	cl := NewConcurrencyLimiter(cfg, nil)
	cl.RegisterModel("model-y", 1, 4000)
	ctx := context.Background()

	t1, _ := cl.Acquire(ctx, "model-y")
	_, err := cl.Acquire(ctx, "model-y")
	if err == nil {
		t.Error("expected error at model limit 1")
	}
	t1.Release()

	cl.SetModelMax("model-y", 3)
	tokens := make([]*ConcurrencyToken, 3)
	for i := 0; i < 3; i++ {
		tokens[i], err = cl.Acquire(ctx, "model-y")
		if err != nil {
			t.Errorf("expected acquire %d to succeed after SetModelMax(3): %v", i+1, err)
		}
	}
	for _, tok := range tokens {
		if tok != nil {
			tok.Release()
		}
	}
}

func TestConcurrencyLimiter_UnregisterModel(t *testing.T) {
	cfg := ConcurrencyConfig{
		GlobalMaxConcurrent: 10,
		DefaultModelMax:     5,
		AcquireTimeout:      100 * time.Millisecond,
		EnableGPUAwareness:  false,
	}
	cl := NewConcurrencyLimiter(cfg, nil)
	cl.RegisterModel("model-z", 1, 4000)

	cl.UnregisterModel("model-z")

	ctx := context.Background()
	// After unregister, should use DefaultModelMax (5)
	token, err := cl.Acquire(ctx, "model-z")
	if err != nil {
		t.Errorf("expected success after unregister (uses default): %v", err)
	}
	if token != nil {
		token.Release()
	}
}

func TestConcurrencyLimiter_GetModelConcurrency(t *testing.T) {
	cfg := ConcurrencyConfig{
		GlobalMaxConcurrent: 10,
		DefaultModelMax:     5,
		AcquireTimeout:      100 * time.Millisecond,
		EnableGPUAwareness:  false,
	}
	cl := NewConcurrencyLimiter(cfg, nil)
	cl.RegisterModel("model-q", 3, 4000)

	ctx := context.Background()
	t1, _ := cl.Acquire(ctx, "model-q")
	t2, _ := cl.Acquire(ctx, "model-q")

	current, max := cl.GetModelConcurrency("model-q")
	if current != 2 {
		t.Errorf("expected current=2, got %d", current)
	}
	if max != 3 {
		t.Errorf("expected max=3, got %d", max)
	}

	t1.Release()
	t2.Release()
}

func TestConcurrencyLimiter_Concurrent(t *testing.T) {
	cfg := ConcurrencyConfig{
		GlobalMaxConcurrent: 20,
		DefaultModelMax:     10,
		AcquireTimeout:      100 * time.Millisecond,
		EnableGPUAwareness:  false,
	}
	cl := NewConcurrencyLimiter(cfg, nil)
	ctx := context.Background()

	var wg sync.WaitGroup
	var success, failed int64

	for i := 0; i < 30; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			token, err := cl.Acquire(ctx, "concurrent-model")
			if err != nil {
				atomic.AddInt64(&failed, 1)
				return
			}
			atomic.AddInt64(&success, 1)
			time.Sleep(2 * time.Millisecond)
			token.Release()
		}()
	}
	wg.Wait()

	if success+failed != 30 {
		t.Errorf("expected 30 total, got %d", success+failed)
	}
}

func TestConcurrencyLimiter_StartStop(t *testing.T) {
	cfg := DefaultConcurrencyConfig()
	cfg.EnableGPUAwareness = false
	cl := NewConcurrencyLimiter(cfg, nil)

	cl.Start()
	cl.Start() // double start should not panic

	cl.Stop()
	cl.Stop() // double stop should not panic
}

func TestConcurrencyLimiter_ContextDeadline(t *testing.T) {
	cfg := ConcurrencyConfig{
		GlobalMaxConcurrent: 1,
		DefaultModelMax:     1,
		AcquireTimeout:      10 * time.Second,
		EnableGPUAwareness:  false,
	}
	cl := NewConcurrencyLimiter(cfg, nil)

	// Fill the slot
	bgCtx := context.Background()
	t1, _ := cl.Acquire(bgCtx, "model")

	// Short deadline context
	ctx, cancel := context.WithTimeout(bgCtx, 20*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err := cl.Acquire(ctx, "model")
	elapsed := time.Since(start)

	if err == nil {
		t.Error("expected error due to context deadline")
	}
	if elapsed > 200*time.Millisecond {
		t.Errorf("waited too long: %v", elapsed)
	}

	t1.Release()
}
