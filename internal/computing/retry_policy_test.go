package computing

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"
)

// --- RetryPolicy tests ---

func TestRetryPolicy_NoRetryOnSuccess(t *testing.T) {
	rp := NewRetryPolicy(DefaultRetryConfig())
	calls := 0
	err := rp.Execute(context.Background(), func() error {
		calls++
		return nil
	})
	if err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}
	if calls != 1 {
		t.Fatalf("expected 1 call, got %d", calls)
	}
	metrics := rp.GetMetrics()
	if metrics.TotalAttempts != 1 {
		t.Errorf("expected 1 attempt, got %d", metrics.TotalAttempts)
	}
}

func TestRetryPolicy_RetryOnRetryableError(t *testing.T) {
	cfg := DefaultRetryConfig()
	cfg.MaxRetries = 3
	cfg.InitialDelay = 1 * time.Millisecond
	rp := NewRetryPolicy(cfg)

	calls := 0
	err := rp.Execute(context.Background(), func() error {
		calls++
		if calls < 3 {
			return errors.New("service unavailable")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("expected success after retries, got: %v", err)
	}
	if calls != 3 {
		t.Errorf("expected 3 calls, got %d", calls)
	}
}

func TestRetryPolicy_NoRetryOnNonRetryable(t *testing.T) {
	cfg := DefaultRetryConfig()
	cfg.MaxRetries = 3
	cfg.InitialDelay = 1 * time.Millisecond
	rp := NewRetryPolicy(cfg)

	calls := 0
	err := rp.Execute(context.Background(), func() error {
		calls++
		return errors.New("authentication failed")
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if calls != 1 {
		t.Errorf("expected 1 call (no retry), got %d", calls)
	}
	metrics := rp.GetMetrics()
	if metrics.TotalNonRetryable != 1 {
		t.Errorf("expected 1 non-retryable, got %d", metrics.TotalNonRetryable)
	}
}

func TestRetryPolicy_MaxRetriesExceeded(t *testing.T) {
	cfg := DefaultRetryConfig()
	cfg.MaxRetries = 2
	cfg.InitialDelay = 1 * time.Millisecond
	rp := NewRetryPolicy(cfg)

	calls := 0
	err := rp.Execute(context.Background(), func() error {
		calls++
		return errors.New("503 service unavailable")
	})
	if err == nil {
		t.Fatal("expected error after max retries, got nil")
	}
	// attempt 0, retry 1, retry 2 = 3 calls total
	if calls != 3 {
		t.Errorf("expected 3 calls (initial + 2 retries), got %d", calls)
	}
	metrics := rp.GetMetrics()
	if metrics.TotalFailures != 1 {
		t.Errorf("expected 1 total failure, got %d", metrics.TotalFailures)
	}
}

func TestRetryPolicy_ContextCancellation(t *testing.T) {
	cfg := DefaultRetryConfig()
	cfg.MaxRetries = 5
	cfg.InitialDelay = 100 * time.Millisecond
	rp := NewRetryPolicy(cfg)

	ctx, cancel := context.WithCancel(context.Background())

	calls := 0
	err := rp.Execute(ctx, func() error {
		calls++
		if calls == 2 {
			cancel()
		}
		return errors.New("connection refused")
	})

	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got: %v", err)
	}
	if calls > 3 {
		t.Errorf("too many calls after cancel: %d", calls)
	}
}

func TestRetryPolicy_ExponentialBackoff(t *testing.T) {
	cfg := DefaultRetryConfig()
	cfg.MaxRetries = 3
	cfg.InitialDelay = 10 * time.Millisecond
	cfg.Multiplier = 2.0
	cfg.JitterFactor = 0 // disable jitter for deterministic test
	rp := NewRetryPolicy(cfg)

	delays := []time.Duration{}
	lastTime := time.Now()

	calls := 0
	rp.Execute(context.Background(), func() error {
		now := time.Now()
		if calls > 0 {
			delays = append(delays, now.Sub(lastTime))
		}
		lastTime = now
		calls++
		return errors.New("timeout")
	})

	// delay[0] ≈ 10ms (attempt 0), delay[1] ≈ 20ms (attempt 1)
	if len(delays) >= 2 {
		if delays[1] < delays[0] {
			t.Errorf("expected delay to increase: delay[0]=%v delay[1]=%v", delays[0], delays[1])
		}
	}
}

func TestRetryPolicy_JitterRange(t *testing.T) {
	cfg := DefaultRetryConfig()
	cfg.MaxRetries = 0
	cfg.JitterFactor = 0.2
	cfg.InitialDelay = 100 * time.Millisecond
	rp := NewRetryPolicy(cfg)

	// Calculate delay 10 times, check jitter is within ±20%
	base := float64(cfg.InitialDelay)
	for i := 0; i < 10; i++ {
		d := rp.CalculateDelay(0)
		df := float64(d)
		if df < base*(1-0.2)-1 || df > base*(1+0.2)+1 {
			t.Errorf("jitter out of range: got %v, base %v", d, cfg.InitialDelay)
		}
	}
}

func TestRetryPolicy_IsRetryable_NetworkTimeout(t *testing.T) {
	rp := NewRetryPolicy(DefaultRetryConfig())

	// timeout error should be retryable
	if !rp.IsRetryable(errors.New("connection timeout")) {
		t.Error("expected timeout to be retryable")
	}
	// 502 should be retryable
	if !rp.IsRetryable(errors.New("502 bad gateway")) {
		t.Error("expected 502 to be retryable")
	}
	// 404 should NOT be retryable
	if rp.IsRetryable(errors.New("404 not found")) {
		t.Error("expected 404 to not be retryable")
	}
	// nil should not be retryable
	if rp.IsRetryable(nil) {
		t.Error("nil should not be retryable")
	}
}

func TestRetryPolicy_IsRetryable_AuthErrors(t *testing.T) {
	rp := NewRetryPolicy(DefaultRetryConfig())

	cases := []struct {
		err      string
		expected bool
	}{
		{"unauthorized access", false},
		{"forbidden resource", false},
		{"400 bad request", false},
		{"401 unauthorized", false},
		{"403 forbidden", false},
		{"connection refused", true},
		{"EOF", true},
		{"broken pipe", true},
	}
	for _, c := range cases {
		got := rp.IsRetryable(errors.New(c.err))
		if got != c.expected {
			t.Errorf("IsRetryable(%q) = %v, want %v", c.err, got, c.expected)
		}
	}
}

func TestRetryPolicy_MetricsAccumulation(t *testing.T) {
	cfg := DefaultRetryConfig()
	cfg.MaxRetries = 1
	cfg.InitialDelay = 1 * time.Millisecond
	rp := NewRetryPolicy(cfg)

	// Success
	rp.Execute(context.Background(), func() error { return nil })
	// Retried then success
	count := 0
	rp.Execute(context.Background(), func() error {
		count++
		if count < 2 {
			return errors.New("503")
		}
		return nil
	})
	// Non-retryable
	rp.Execute(context.Background(), func() error { return errors.New("404 not found") })
	// Total failure
	rp.Execute(context.Background(), func() error { return errors.New("504 gateway timeout") })

	m := rp.GetMetrics()
	if m.TotalAttempts != 4 {
		t.Errorf("expected 4 attempts, got %d", m.TotalAttempts)
	}
	if m.TotalNonRetryable != 1 {
		t.Errorf("expected 1 non-retryable, got %d", m.TotalNonRetryable)
	}
}

// --- CircuitBreaker tests ---

func TestCircuitBreaker_InitialClosed(t *testing.T) {
	cb := NewCircuitBreaker(3, 2, 60*time.Second)
	if cb.GetState() != CircuitClosed {
		t.Errorf("expected closed state, got %s", cb.GetState())
	}
	if !cb.Allow() {
		t.Error("expected allow in closed state")
	}
}

func TestCircuitBreaker_OpensAfterThreshold(t *testing.T) {
	cb := NewCircuitBreaker(3, 2, 60*time.Second)
	for i := 0; i < 3; i++ {
		cb.RecordFailure()
	}
	if cb.GetState() != CircuitOpen {
		t.Errorf("expected open state after 3 failures, got %s", cb.GetState())
	}
	if cb.Allow() {
		t.Error("expected deny in open state")
	}
}

func TestCircuitBreaker_HalfOpenAfterTimeout(t *testing.T) {
	cb := NewCircuitBreaker(2, 1, 10*time.Millisecond)
	cb.RecordFailure()
	cb.RecordFailure()

	time.Sleep(20 * time.Millisecond)
	if !cb.Allow() {
		t.Error("expected allow in half-open state after timeout")
	}
	if cb.GetState() != CircuitHalfOpen {
		t.Errorf("expected half-open state, got %s", cb.GetState())
	}
}

func TestCircuitBreaker_RecoveryAfterSuccesses(t *testing.T) {
	cb := NewCircuitBreaker(2, 2, 10*time.Millisecond)
	cb.RecordFailure()
	cb.RecordFailure()

	time.Sleep(20 * time.Millisecond)
	cb.Allow() // transitions to half-open
	cb.RecordSuccess()
	cb.RecordSuccess()

	if cb.GetState() != CircuitClosed {
		t.Errorf("expected closed after 2 successes in half-open, got %s", cb.GetState())
	}
}

func TestCircuitBreaker_Reset(t *testing.T) {
	cb := NewCircuitBreaker(2, 1, time.Second)
	cb.RecordFailure()
	cb.RecordFailure()

	cb.Reset()
	if cb.GetState() != CircuitClosed {
		t.Errorf("expected closed after reset, got %s", cb.GetState())
	}
}

func TestCircuitBreaker_StateChangeCallback(t *testing.T) {
	cb := NewCircuitBreaker(2, 1, 60*time.Second)
	changes := make(chan string, 10)
	cb.SetStateChangeCallback(func(from, to CircuitState) {
		changes <- fmt.Sprintf("%s->%s", from, to)
	})

	cb.RecordFailure()
	cb.RecordFailure()

	select {
	case change := <-changes:
		if change != "closed->open" {
			t.Errorf("unexpected state change: %s", change)
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("expected state change callback to fire")
	}
}

// --- BackoffCalculator tests ---

func TestBackoffCalculator_Exponential(t *testing.T) {
	bc := NewBackoffCalculator(BackoffExponential, 10*time.Millisecond, time.Minute)
	bc.SetJitter(0) // disable jitter for deterministic test

	d0 := bc.Calculate(0)
	d1 := bc.Calculate(1)
	d2 := bc.Calculate(2)

	if d1 <= d0 {
		t.Errorf("expected increasing delays: d0=%v d1=%v", d0, d1)
	}
	if d2 <= d1 {
		t.Errorf("expected increasing delays: d1=%v d2=%v", d1, d2)
	}
}

func TestBackoffCalculator_Linear(t *testing.T) {
	bc := NewBackoffCalculator(BackoffLinear, 10*time.Millisecond, time.Minute)
	bc.SetJitter(0)

	d0 := bc.Calculate(0)
	d1 := bc.Calculate(1)

	// Linear: d1 = 2 * d0
	if d1 != 2*d0 {
		t.Errorf("expected d1=2*d0, got d0=%v d1=%v", d0, d1)
	}
}

func TestBackoffCalculator_Constant(t *testing.T) {
	bc := NewBackoffCalculator(BackoffConstant, 50*time.Millisecond, time.Minute)
	bc.SetJitter(0)

	for i := 0; i < 5; i++ {
		d := bc.Calculate(i)
		if d != 50*time.Millisecond {
			t.Errorf("expected constant 50ms, got %v at attempt %d", d, i)
		}
	}
}

func TestBackoffCalculator_MaxDelayCap(t *testing.T) {
	bc := NewBackoffCalculator(BackoffExponential, 100*time.Millisecond, 200*time.Millisecond)
	bc.SetJitter(0)

	// At high attempt numbers delay should be capped
	d := bc.Calculate(20)
	if d > 200*time.Millisecond {
		t.Errorf("delay exceeded max cap: %v", d)
	}
}
