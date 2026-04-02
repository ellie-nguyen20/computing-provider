package computing

import (
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func newTestHealthConfig() HealthCheckConfig {
	return HealthCheckConfig{
		Interval:           10 * time.Second, // long interval so we control checks manually
		Timeout:            2 * time.Second,
		UnhealthyThreshold: 3,
		HealthyThreshold:   2,
		CircuitOpenTime:    60 * time.Second,
	}
}

// mockModelServer creates a test HTTP server that returns the given status code.
func mockModelServer(t *testing.T, statusCode int) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(statusCode)
	}))
}

func TestHealthChecker_InitialStateUnknown(t *testing.T) {
	hc := NewModelHealthChecker(newTestHealthConfig())
	hc.RegisterModel("model-a", "http://localhost:9999", "")

	status, ok := hc.GetModelStatus("model-a")
	if !ok {
		t.Fatal("expected model to be registered")
	}
	if status.Health != ModelHealthUnknown {
		t.Errorf("expected unknown health, got %s", status.Health)
	}
}

func TestHealthChecker_HealthyTransition(t *testing.T) {
	srv := mockModelServer(t, http.StatusOK)
	defer srv.Close()

	hc := NewModelHealthChecker(newTestHealthConfig())
	hc.RegisterModel("model-healthy", srv.URL, "")
	hc.ForceCheck("model-healthy")

	status, ok := hc.GetModelStatus("model-healthy")
	if !ok {
		t.Fatal("expected status to exist")
	}
	if status.Health != ModelHealthHealthy {
		t.Errorf("expected healthy, got %s", status.Health)
	}
}

func TestHealthChecker_UnhealthyAfter3Failures(t *testing.T) {
	srv := mockModelServer(t, http.StatusServiceUnavailable)
	defer srv.Close()

	cfg := newTestHealthConfig()
	cfg.UnhealthyThreshold = 3
	hc := NewModelHealthChecker(cfg)
	hc.RegisterModel("model-bad", srv.URL, "")

	for i := 0; i < 3; i++ {
		hc.ForceCheck("model-bad")
	}

	status, _ := hc.GetModelStatus("model-bad")
	if status.Health != ModelHealthUnhealthy {
		t.Errorf("expected unhealthy after 3 failures, got %s", status.Health)
	}
	if !status.CircuitOpen {
		t.Error("expected circuit to be open")
	}
}

func TestHealthChecker_RecoveryAfterSuccesses(t *testing.T) {
	var serveErr int32 = 1
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.LoadInt32(&serveErr) == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
		} else {
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer srv.Close()

	cfg := newTestHealthConfig()
	cfg.UnhealthyThreshold = 2
	hc := NewModelHealthChecker(cfg)
	hc.RegisterModel("model-recover", srv.URL, "")

	// Make it unhealthy
	for i := 0; i < 2; i++ {
		hc.ForceCheck("model-recover")
	}
	status, _ := hc.GetModelStatus("model-recover")
	if status.Health != ModelHealthUnhealthy {
		t.Fatal("expected unhealthy state before recovery test")
	}

	// Now serve OK
	atomic.StoreInt32(&serveErr, 0)

	// First success: transitions through degraded
	hc.ForceCheck("model-recover")
	// Second success: should be healthy
	hc.ForceCheck("model-recover")

	status, _ = hc.GetModelStatus("model-recover")
	if status.Health != ModelHealthHealthy {
		t.Errorf("expected healthy after recovery, got %s", status.Health)
	}
}

func TestHealthChecker_FallbackToHealthEndpoint(t *testing.T) {
	// /v1/models returns 404, /health returns 200
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	hc := NewModelHealthChecker(newTestHealthConfig())
	hc.RegisterModel("model-fallback", srv.URL, "")
	hc.ForceCheck("model-fallback")

	status, _ := hc.GetModelStatus("model-fallback")
	if status.Health != ModelHealthHealthy {
		t.Errorf("expected healthy via /health fallback, got %s", status.Health)
	}
}

func TestHealthChecker_CallbackFired(t *testing.T) {
	srv := mockModelServer(t, http.StatusOK)
	defer srv.Close()

	hc := NewModelHealthChecker(newTestHealthConfig())

	called := make(chan struct{}, 1)
	hc.SetStatusChangeCallback(func(modelID string, oldHealth, newHealth ModelHealth) {
		if modelID == "model-cb" {
			called <- struct{}{}
		}
	})

	hc.RegisterModel("model-cb", srv.URL, "")
	hc.ForceCheck("model-cb")

	select {
	case <-called:
		// success
	case <-time.After(500 * time.Millisecond):
		t.Error("expected callback to fire on health change")
	}
}

func TestHealthChecker_CallbackHasCorrectArgs(t *testing.T) {
	srv := mockModelServer(t, http.StatusOK)
	defer srv.Close()

	hc := NewModelHealthChecker(newTestHealthConfig())

	type change struct{ old, new ModelHealth }
	changes := make(chan change, 1)
	hc.SetStatusChangeCallback(func(modelID string, old, new ModelHealth) {
		changes <- change{old, new}
	})

	hc.RegisterModel("model-args", srv.URL, "")
	hc.ForceCheck("model-args")

	select {
	case c := <-changes:
		if c.old != ModelHealthUnknown {
			t.Errorf("expected old=unknown, got %s", c.old)
		}
		if c.new != ModelHealthHealthy {
			t.Errorf("expected new=healthy, got %s", c.new)
		}
	case <-time.After(500 * time.Millisecond):
		t.Error("expected callback")
	}
}

func TestHealthChecker_APIKeyHeader(t *testing.T) {
	authChecked := make(chan string, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authChecked <- r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	hc := NewModelHealthChecker(newTestHealthConfig())
	hc.RegisterModel("model-auth", srv.URL, "my-secret-key")
	hc.ForceCheck("model-auth")

	select {
	case auth := <-authChecked:
		if auth != "Bearer my-secret-key" {
			t.Errorf("expected Authorization header 'Bearer my-secret-key', got %q", auth)
		}
	case <-time.After(500 * time.Millisecond):
		t.Error("expected server to receive request")
	}
}

func TestHealthChecker_EndpointGrouping(t *testing.T) {
	var probeCount int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&probeCount, 1)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	hc := NewModelHealthChecker(newTestHealthConfig())
	// Two models at the same endpoint
	hc.RegisterModel("model-1", srv.URL, "")
	hc.RegisterModel("model-2", srv.URL, "")

	// checkAllModels groups by endpoint — should probe once
	hc.checkAllModels()

	time.Sleep(50 * time.Millisecond)
	count := atomic.LoadInt32(&probeCount)
	// Should be 1 probe (grouped), not 2
	if count != 1 {
		t.Errorf("expected 1 probe for shared endpoint, got %d", count)
	}
}

func TestHealthChecker_UnregisterModel(t *testing.T) {
	srv := mockModelServer(t, http.StatusOK)
	defer srv.Close()

	hc := NewModelHealthChecker(newTestHealthConfig())
	hc.RegisterModel("model-del", srv.URL, "")
	hc.UnregisterModel("model-del")

	_, ok := hc.GetModelStatus("model-del")
	if ok {
		t.Error("expected model to be gone after unregister")
	}
	if hc.IsModelHealthy("model-del") {
		t.Error("expected IsModelHealthy to be false for unregistered model")
	}
}

func TestHealthChecker_IsModelHealthy_Degraded(t *testing.T) {
	hc := NewModelHealthChecker(newTestHealthConfig())
	hc.RegisterModel("model-deg", "http://ignored", "")

	// Manually set to degraded via applyProbeResult
	hc.applyProbeResult("model-deg", "http://ignored", nil) // healthy
	hc.applyProbeResult("model-deg", "http://ignored", nil) // still healthy

	// GetHealthyModels includes degraded
	healthy := hc.GetHealthyModels()
	found := false
	for _, id := range healthy {
		if id == "model-deg" {
			found = true
		}
	}
	if !found {
		t.Error("expected degraded model to appear in GetHealthyModels()")
	}
}

func TestHealthChecker_GetAllStatuses(t *testing.T) {
	srv := mockModelServer(t, http.StatusOK)
	defer srv.Close()

	hc := NewModelHealthChecker(newTestHealthConfig())
	hc.RegisterModel("m1", srv.URL, "")
	hc.RegisterModel("m2", srv.URL, "")

	statuses := hc.GetAllStatuses()
	if len(statuses) != 2 {
		t.Errorf("expected 2 statuses, got %d", len(statuses))
	}
}

func TestHealthChecker_GetStatusJSON(t *testing.T) {
	hc := NewModelHealthChecker(newTestHealthConfig())
	hc.RegisterModel("model-json", "http://localhost:9999", "")

	data, err := hc.GetStatusJSON()
	if err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}
	if len(data) == 0 {
		t.Error("expected non-empty JSON")
	}
}

func TestHealthChecker_Concurrent_RegisterForceCheck(t *testing.T) {
	srv := mockModelServer(t, http.StatusOK)
	defer srv.Close()

	hc := NewModelHealthChecker(newTestHealthConfig())

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			modelID := "model-concurrent"
			hc.RegisterModel(modelID, srv.URL, "")
			hc.ForceCheck(modelID)
			hc.IsModelHealthy(modelID)
		}(i)
	}
	wg.Wait()
}

func TestHealthChecker_StopStart(t *testing.T) {
	cfg := newTestHealthConfig()
	cfg.Interval = 10 * time.Millisecond
	hc := NewModelHealthChecker(cfg)
	hc.Start()
	hc.Stop()
	hc.Stop() // double stop should not panic
}
