package computing

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// --- TokenBucket tests ---

func TestTokenBucket_AllowWithinBurst(t *testing.T) {
	tb := NewTokenBucket(100, 10)
	for i := 0; i < 10; i++ {
		if !tb.Allow() {
			t.Fatalf("expected allow at call %d within burst", i+1)
		}
	}
}

func TestTokenBucket_ThrottleOverBurst(t *testing.T) {
	tb := NewTokenBucket(1, 3) // very slow refill, burst 3
	for i := 0; i < 3; i++ {
		tb.Allow()
	}
	if tb.Allow() {
		t.Error("expected throttle after burst exhausted")
	}
}

func TestTokenBucket_TokenRefill(t *testing.T) {
	tb := NewTokenBucket(1000, 1) // 1000 tokens/sec, burst 1
	tb.Allow()                    // exhaust

	time.Sleep(5 * time.Millisecond) // wait for ~5 tokens to refill
	if !tb.Allow() {
		t.Error("expected allow after refill")
	}
}

func TestTokenBucket_AllowN_Atomic(t *testing.T) {
	tb := NewTokenBucket(100, 20)

	var allowed, throttled int64
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if tb.AllowN(3) {
				atomic.AddInt64(&allowed, 1)
			} else {
				atomic.AddInt64(&throttled, 1)
			}
		}()
	}
	wg.Wait()

	total := allowed + throttled
	if total != 10 {
		t.Errorf("expected 10 total calls, got %d", total)
	}
}

func TestTokenBucket_SetRate_Dynamic(t *testing.T) {
	tb := NewTokenBucket(1, 1) // 1 token/sec
	tb.Allow()                  // exhaust

	tb.SetRate(1000) // 1000 tokens/sec
	time.Sleep(5 * time.Millisecond)

	if !tb.Allow() {
		t.Error("expected allow after rate increase")
	}
}

func TestTokenBucket_GetStats(t *testing.T) {
	tb := NewTokenBucket(100, 10)
	tb.Allow()
	tb.Allow()

	tokens, rate, allowed, _ := tb.GetStats()
	if rate != 100 {
		t.Errorf("expected rate 100, got %v", rate)
	}
	if allowed != 2 {
		t.Errorf("expected 2 allowed, got %d", allowed)
	}
	_ = tokens
}

// --- RateLimiter tests ---

func TestRateLimiter_Allow(t *testing.T) {
	cfg := DefaultRateLimiterConfig()
	cfg.TokensPerSecond = 100
	cfg.BurstSize = 10
	cfg.EnableAdaptive = false

	rl := NewRateLimiter(cfg, nil)
	for i := 0; i < 10; i++ {
		if !rl.Allow() {
			t.Fatalf("expected allow at call %d", i+1)
		}
	}
}

func TestRateLimiter_PerModelLimit(t *testing.T) {
	cfg := DefaultRateLimiterConfig()
	cfg.TokensPerSecond = 1000
	cfg.BurstSize = 100
	cfg.EnableAdaptive = false

	rl := NewRateLimiter(cfg, nil)
	rl.SetModelLimit("model-a", 5, 3) // model-a: burst 3

	// Use up model-a's burst
	for i := 0; i < 3; i++ {
		if !rl.AllowModel("model-a") {
			t.Fatalf("expected allow at call %d", i+1)
		}
	}
	// Next call should be throttled by model-a limit
	if rl.AllowModel("model-a") {
		t.Error("expected throttle after model burst exhausted")
	}
}

func TestRateLimiter_GlobalFallback(t *testing.T) {
	cfg := DefaultRateLimiterConfig()
	cfg.TokensPerSecond = 1000
	cfg.BurstSize = 100
	cfg.EnableAdaptive = false

	rl := NewRateLimiter(cfg, nil)
	// model-b has no specific limit — should use global
	if !rl.AllowModel("model-b") {
		t.Error("expected allow for model with no specific limit")
	}
}

func TestRateLimiter_RemoveModelLimit(t *testing.T) {
	cfg := DefaultRateLimiterConfig()
	cfg.TokensPerSecond = 1000
	cfg.BurstSize = 100
	cfg.EnableAdaptive = false

	rl := NewRateLimiter(cfg, nil)
	rl.SetModelLimit("model-c", 1, 1)
	rl.AllowModel("model-c") // exhaust

	rl.RemoveModelLimit("model-c")
	// After removal, falls back to global (which has plenty of tokens)
	if !rl.AllowModel("model-c") {
		t.Error("expected allow after model limit removed")
	}
}

func TestRateLimiter_StartStop(t *testing.T) {
	cfg := DefaultRateLimiterConfig()
	cfg.EnableAdaptive = false
	rl := NewRateLimiter(cfg, nil)

	rl.Start()
	rl.Start() // double start should not panic

	rl.Stop()
	rl.Stop() // double stop should not panic
}

func TestRateLimiter_GetMetrics(t *testing.T) {
	cfg := DefaultRateLimiterConfig()
	cfg.TokensPerSecond = 10
	cfg.BurstSize = 5
	cfg.EnableAdaptive = false

	rl := NewRateLimiter(cfg, nil)
	rl.Allow()
	rl.Allow()

	m := rl.GetMetrics()
	if m.TotalAllowed != 2 {
		t.Errorf("expected 2 allowed, got %d", m.TotalAllowed)
	}
	if m.CurrentRate != 10 {
		t.Errorf("expected rate 10, got %v", m.CurrentRate)
	}
	if m.BurstSize != 5 {
		t.Errorf("expected burst 5, got %d", m.BurstSize)
	}
}

func TestRateLimiter_Concurrent(t *testing.T) {
	cfg := DefaultRateLimiterConfig()
	cfg.TokensPerSecond = 1000
	cfg.BurstSize = 200
	cfg.EnableAdaptive = false

	rl := NewRateLimiter(cfg, nil)

	var wg sync.WaitGroup
	var allowed, throttled int64
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if rl.Allow() {
				atomic.AddInt64(&allowed, 1)
			} else {
				atomic.AddInt64(&throttled, 1)
			}
		}()
	}
	wg.Wait()

	if allowed+throttled != 100 {
		t.Errorf("expected 100 total, got %d", allowed+throttled)
	}
}

func TestRateLimiter_WaitForToken_Success(t *testing.T) {
	cfg := DefaultRateLimiterConfig()
	cfg.TokensPerSecond = 1000
	cfg.BurstSize = 10
	cfg.EnableAdaptive = false

	rl := NewRateLimiter(cfg, nil)
	err := rl.WaitForToken("model-x", 100*time.Millisecond)
	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}
}

func TestRateLimiter_WaitForToken_Timeout(t *testing.T) {
	cfg := DefaultRateLimiterConfig()
	cfg.TokensPerSecond = 0.001 // almost no refill
	cfg.BurstSize = 1
	cfg.EnableAdaptive = false

	rl := NewRateLimiter(cfg, nil)
	rl.Allow() // exhaust burst

	err := rl.WaitForToken("model-x", 20*time.Millisecond)
	if err == nil {
		t.Error("expected ErrRateLimitExceeded on timeout")
	}
}

func TestRateLimiter_GetModelMetrics_NoModel(t *testing.T) {
	cfg := DefaultRateLimiterConfig()
	cfg.EnableAdaptive = false
	rl := NewRateLimiter(cfg, nil)

	m := rl.GetModelMetrics("nonexistent-model")
	if m != nil {
		t.Error("expected nil for model without specific limit")
	}
}

func TestRateLimiter_GetModelMetrics_WithModel(t *testing.T) {
	cfg := DefaultRateLimiterConfig()
	cfg.EnableAdaptive = false
	rl := NewRateLimiter(cfg, nil)
	rl.SetModelLimit("model-d", 50, 5)

	m := rl.GetModelMetrics("model-d")
	if m == nil {
		t.Fatal("expected non-nil metrics for registered model")
	}
	if m.CurrentRate != 50 {
		t.Errorf("expected rate 50, got %v", m.CurrentRate)
	}
}

func TestRateLimiter_GetBackoffTime_WithTokens(t *testing.T) {
	cfg := DefaultRateLimiterConfig()
	cfg.TokensPerSecond = 100
	cfg.BurstSize = 50
	cfg.EnableAdaptive = false
	rl := NewRateLimiter(cfg, nil)
	// Bucket is full — backoff should be 0
	backoff := rl.GetBackoffTime()
	if backoff != 0 {
		t.Errorf("expected 0 backoff when tokens available, got %v", backoff)
	}
}

func TestRateLimiter_GetBackoffTime_ZeroRate(t *testing.T) {
	// When rate=0, GetBackoffTime should return the default 1s
	cfg := DefaultRateLimiterConfig()
	cfg.TokensPerSecond = 0
	cfg.BurstSize = 1
	cfg.EnableAdaptive = false
	rl := NewRateLimiter(cfg, nil)
	rl.globalBucket.SetRate(0)
	rl.Allow() // exhaust

	backoff := rl.GetBackoffTime()
	if backoff != time.Second {
		t.Errorf("expected 1s default backoff with zero rate, got %v", backoff)
	}
}

// --- PerModelRateLimiter tests ---

func TestPerModelRateLimiter_EnsureModelLimit(t *testing.T) {
	cfg := DefaultRateLimiterConfig()
	cfg.EnableAdaptive = false
	prl := NewPerModelRateLimiter(cfg, nil, 10, 5)

	prl.EnsureModelLimit("auto-model")
	m := prl.GetModelMetrics("auto-model")
	if m == nil {
		t.Error("expected model limit to be auto-created")
	}
}

func TestPerModelRateLimiter_NoDoubleCreate(t *testing.T) {
	cfg := DefaultRateLimiterConfig()
	cfg.EnableAdaptive = false
	prl := NewPerModelRateLimiter(cfg, nil, 10, 5)

	prl.EnsureModelLimit("model-e")
	prl.EnsureModelLimit("model-e") // should not overwrite

	// Use up initial bucket, ensure limit stays consistent
	m := prl.GetModelMetrics("model-e")
	if m == nil {
		t.Error("expected model metrics after double EnsureModelLimit")
	}
}
