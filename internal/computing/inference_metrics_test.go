package computing

import (
	"strings"
	"sync"
	"testing"
	"time"
)

func TestInferenceMetrics_InitialState(t *testing.T) {
	m := NewInferenceMetrics()
	if m.ConnectionState != "disconnected" {
		t.Errorf("expected disconnected, got %s", m.ConnectionState)
	}
	if m.TotalRequests != 0 {
		t.Errorf("expected 0 total requests, got %d", m.TotalRequests)
	}
}

func TestInferenceMetrics_RecordConnectionState(t *testing.T) {
	m := NewInferenceMetrics()

	m.RecordConnectionState("connected")
	if m.ConnectionState != "connected" {
		t.Errorf("expected connected, got %s", m.ConnectionState)
	}
	if m.LastConnectedAt.IsZero() {
		t.Error("expected LastConnectedAt to be set")
	}

	m.RecordConnectionState("disconnected")
	if m.LastDisconnectedAt.IsZero() {
		t.Error("expected LastDisconnectedAt to be set")
	}
}

func TestInferenceMetrics_RecordReconnect(t *testing.T) {
	m := NewInferenceMetrics()
	m.RecordReconnect()
	m.RecordReconnect()
	if m.ReconnectCount != 2 {
		t.Errorf("expected 2 reconnects, got %d", m.ReconnectCount)
	}
}

func TestInferenceMetrics_RecordRequestStart(t *testing.T) {
	m := NewInferenceMetrics()
	m.RecordRequestStart("llama3", false)
	m.RecordRequestStart("llama3", true)

	if m.TotalRequests != 2 {
		t.Errorf("expected 2 total requests, got %d", m.TotalRequests)
	}
	if m.ActiveRequests != 2 {
		t.Errorf("expected 2 active, got %d", m.ActiveRequests)
	}
	if m.StreamingReqs != 1 {
		t.Errorf("expected 1 streaming, got %d", m.StreamingReqs)
	}
	mm, ok := m.ModelMetrics["llama3"]
	if !ok {
		t.Fatal("expected model metrics for llama3")
	}
	if mm.TotalRequests != 2 {
		t.Errorf("expected model total=2, got %d", mm.TotalRequests)
	}
}

func TestInferenceMetrics_RecordRequestEnd_Success(t *testing.T) {
	m := NewInferenceMetrics()
	m.RecordRequestStart("gpt4", false)
	m.RecordRequestEnd("gpt4", 150.0, 100, 200, true, "")

	if m.SuccessfulReqs != 1 {
		t.Errorf("expected 1 success, got %d", m.SuccessfulReqs)
	}
	if m.FailedReqs != 0 {
		t.Errorf("expected 0 failures, got %d", m.FailedReqs)
	}
	if m.TotalTokensIn != 100 {
		t.Errorf("expected 100 tokens in, got %d", m.TotalTokensIn)
	}
	if m.TotalTokensOut != 200 {
		t.Errorf("expected 200 tokens out, got %d", m.TotalTokensOut)
	}
	if m.ActiveRequests != 0 {
		t.Errorf("expected 0 active after end, got %d", m.ActiveRequests)
	}
}

func TestInferenceMetrics_RecordRequestEnd_Failure(t *testing.T) {
	m := NewInferenceMetrics()
	m.RecordRequestStart("gpt4", false)
	m.RecordRequestEnd("gpt4", 50.0, 10, 0, false, "model error")

	if m.FailedReqs != 1 {
		t.Errorf("expected 1 failure, got %d", m.FailedReqs)
	}
	if m.SuccessfulReqs != 0 {
		t.Errorf("expected 0 successes, got %d", m.SuccessfulReqs)
	}
}

func TestInferenceMetrics_ActiveRequests_NoNegative(t *testing.T) {
	m := NewInferenceMetrics()
	// RecordRequestEnd without matching Start should not go negative
	m.RecordRequestEnd("model", 100, 10, 10, true, "")
	if m.ActiveRequests < 0 {
		t.Errorf("active requests went negative: %d", m.ActiveRequests)
	}
}

func TestInferenceMetrics_PerModelBreakdown(t *testing.T) {
	m := NewInferenceMetrics()
	m.RecordRequestStart("model-a", false)
	m.RecordRequestStart("model-b", false)
	m.RecordRequestEnd("model-a", 100, 50, 100, true, "")
	m.RecordRequestEnd("model-b", 200, 30, 60, false, "err")

	ma := m.ModelMetrics["model-a"]
	mb := m.ModelMetrics["model-b"]

	if ma.SuccessfulReqs != 1 || ma.FailedReqs != 0 {
		t.Errorf("model-a: expected 1 success, 0 fail")
	}
	if mb.SuccessfulReqs != 0 || mb.FailedReqs != 1 {
		t.Errorf("model-b: expected 0 success, 1 fail")
	}
}

func TestInferenceMetrics_LatencyStats(t *testing.T) {
	m := NewInferenceMetrics()
	latencies := []float64{100, 200, 150, 120, 180}
	for _, l := range latencies {
		m.RecordRequestStart("model", false)
		m.RecordRequestEnd("model", l, 10, 20, true, "")
	}

	if m.AvgLatencyMs <= 0 {
		t.Error("expected positive average latency")
	}
	if m.P50LatencyMs <= 0 {
		t.Error("expected positive P50")
	}
	if m.P95LatencyMs <= 0 {
		t.Error("expected positive P95")
	}
	if m.P99LatencyMs <= 0 {
		t.Error("expected positive P99")
	}
	if m.P95LatencyMs < m.P50LatencyMs {
		t.Errorf("P95 (%v) should be >= P50 (%v)", m.P95LatencyMs, m.P50LatencyMs)
	}
}

func TestInferenceMetrics_LatencyPercentiles_Accuracy(t *testing.T) {
	m := NewInferenceMetrics()
	// Insert 100 values: 1, 2, ..., 100
	for i := 1; i <= 100; i++ {
		m.RecordRequestStart("m", false)
		m.RecordRequestEnd("m", float64(i), 0, 0, true, "")
	}

	// P50 should be around 50
	if m.P50LatencyMs < 49 || m.P50LatencyMs > 51 {
		t.Errorf("P50 expected ~50, got %v", m.P50LatencyMs)
	}
	// P95 should be around 95
	if m.P95LatencyMs < 93 || m.P95LatencyMs > 97 {
		t.Errorf("P95 expected ~95, got %v", m.P95LatencyMs)
	}
}

func TestInferenceMetrics_CircularBuffer_NoOverflow(t *testing.T) {
	m := NewInferenceMetrics()
	// Fill beyond maxLatencySample (1000)
	for i := 0; i < 1200; i++ {
		m.RecordRequestStart("m", false)
		m.RecordRequestEnd("m", float64(i), 0, 0, true, "")
	}

	// latencies slice should not exceed maxLatencySample
	m.mu.RLock()
	size := len(m.latencies)
	m.mu.RUnlock()
	if size > m.maxLatencySample {
		t.Errorf("latencies buffer exceeded max: %d > %d", size, m.maxLatencySample)
	}
}

func TestInferenceMetrics_RequestHistory(t *testing.T) {
	m := NewInferenceMetrics()
	m.RecordRequest(RequestMetric{RequestID: "req-1", Model: "llama3", Success: true})
	m.RecordRequest(RequestMetric{RequestID: "req-2", Model: "gpt4", Success: false})

	history := m.GetRequestHistory(10, "")
	if len(history) != 2 {
		t.Errorf("expected 2 history entries, got %d", len(history))
	}
}

func TestInferenceMetrics_RequestHistory_ModelFilter(t *testing.T) {
	m := NewInferenceMetrics()
	m.RecordRequest(RequestMetric{RequestID: "r1", Model: "llama3"})
	m.RecordRequest(RequestMetric{RequestID: "r2", Model: "gpt4"})
	m.RecordRequest(RequestMetric{RequestID: "r3", Model: "llama3"})

	filtered := m.GetRequestHistory(10, "llama3")
	if len(filtered) != 2 {
		t.Errorf("expected 2 llama3 entries, got %d", len(filtered))
	}
}

func TestInferenceMetrics_RequestHistory_CircularBuffer(t *testing.T) {
	m := NewInferenceMetrics()
	// overfill history (max 1000)
	for i := 0; i < 1100; i++ {
		m.RecordRequest(RequestMetric{RequestID: "x", Model: "m"})
	}

	m.historyMu.RLock()
	size := len(m.requestHistory)
	m.historyMu.RUnlock()
	if size > m.maxHistorySize {
		t.Errorf("request history exceeded max: %d", size)
	}
}

func TestInferenceMetrics_GetSnapshot_DeepCopy(t *testing.T) {
	m := NewInferenceMetrics()
	m.RecordRequestStart("snap-model", false)
	m.RecordRequestEnd("snap-model", 100, 10, 20, true, "")

	snap := m.GetSnapshot()
	// Mutate snapshot
	snap.TotalRequests = 9999
	snap.ModelMetrics["snap-model"].TotalRequests = 9999

	// Original should be untouched
	if m.TotalRequests == 9999 {
		t.Error("GetSnapshot should return deep copy")
	}
}

func TestInferenceMetrics_UpdateGPUMetrics(t *testing.T) {
	m := NewInferenceMetrics()
	gpus := []GPUMetrics{
		{Index: 0, Name: "H100", UtilizationPct: 75, MemoryUsedMB: 40000, MemoryTotalMB: 80000},
	}
	m.UpdateGPUMetrics(gpus)

	snap := m.GetSnapshot()
	if len(snap.GPUMetrics) != 1 {
		t.Errorf("expected 1 GPU metric, got %d", len(snap.GPUMetrics))
	}
	if snap.GPUMetrics[0].Name != "H100" {
		t.Errorf("expected H100, got %s", snap.GPUMetrics[0].Name)
	}
}

func TestInferenceMetrics_UpdateSystemMetrics(t *testing.T) {
	m := NewInferenceMetrics()
	m.UpdateSystemMetrics(45.5, 60.0, 12.0, 32.0)

	snap := m.GetSnapshot()
	if snap.CPUUsagePercent != 45.5 {
		t.Errorf("expected CPU 45.5, got %v", snap.CPUUsagePercent)
	}
	if snap.MemoryUsedGB != 12.0 {
		t.Errorf("expected mem used 12GB, got %v", snap.MemoryUsedGB)
	}
}

func TestInferenceMetrics_GetPrometheusMetrics(t *testing.T) {
	m := NewInferenceMetrics()
	m.RecordConnectionState("connected")
	m.RecordRequestStart("llama3", false)
	m.RecordRequestEnd("llama3", 100, 50, 100, true, "")

	prom := m.GetPrometheusMetrics()
	if !strings.Contains(prom, "inference_requests_total") {
		t.Error("expected inference_requests_total in Prometheus output")
	}
	if !strings.Contains(prom, "inference_connection_state 1") {
		t.Error("expected connection_state=1 in Prometheus output")
	}
	if !strings.Contains(prom, "inference_tokens_in_total") {
		t.Error("expected tokens_in_total in Prometheus output")
	}
}

func TestInferenceMetrics_Reset(t *testing.T) {
	m := NewInferenceMetrics()
	m.RecordRequestStart("m", false)
	m.RecordRequestEnd("m", 100, 50, 100, true, "")
	m.RecordReconnect()

	m.Reset()

	if m.TotalRequests != 0 {
		t.Errorf("expected 0 after reset, got %d", m.TotalRequests)
	}
	if m.ReconnectCount != 0 {
		t.Errorf("expected 0 reconnects after reset, got %d", m.ReconnectCount)
	}
	if len(m.ModelMetrics) != 0 {
		t.Errorf("expected empty model metrics after reset, got %d", len(m.ModelMetrics))
	}
}

func TestInferenceMetrics_ConcurrentUpdate(t *testing.T) {
	m := NewInferenceMetrics()
	var wg sync.WaitGroup

	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			model := "concurrent-model"
			m.RecordRequestStart(model, false)
			time.Sleep(time.Millisecond)
			m.RecordRequestEnd(model, float64(idx)*10, 10, 20, true, "")
		}(i)
	}
	wg.Wait()

	if m.TotalRequests != 50 {
		t.Errorf("expected 50 total requests, got %d", m.TotalRequests)
	}
	if m.SuccessfulReqs != 50 {
		t.Errorf("expected 50 successes, got %d", m.SuccessfulReqs)
	}
}

// --- Helper function tests ---

func TestSortFloat64s(t *testing.T) {
	input := []float64{5, 2, 8, 1, 9, 3}
	sortFloat64s(input)
	for i := 1; i < len(input); i++ {
		if input[i] < input[i-1] {
			t.Errorf("not sorted at index %d: %v", i, input)
		}
	}
}

func TestPercentile_Empty(t *testing.T) {
	result := percentile([]float64{}, 50)
	if result != 0 {
		t.Errorf("expected 0 for empty slice, got %v", result)
	}
}

func TestPercentile_Single(t *testing.T) {
	result := percentile([]float64{42.5}, 50)
	if result != 42.5 {
		t.Errorf("expected 42.5, got %v", result)
	}
}

func TestPercentile_P0_P100(t *testing.T) {
	sorted := []float64{10, 20, 30, 40, 50}
	p0 := percentile(sorted, 0)
	p100 := percentile(sorted, 100)
	if p0 != 10 {
		t.Errorf("P0 expected 10, got %v", p0)
	}
	if p100 != 50 {
		t.Errorf("P100 expected 50, got %v", p100)
	}
}
