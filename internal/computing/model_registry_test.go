package computing

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// writeModelsJSON writes a models.json file to the given directory
func writeModelsJSON(t *testing.T, dir string, mappings map[string]ModelMapping) {
	t.Helper()
	data, err := json.Marshal(mappings)
	if err != nil {
		t.Fatalf("failed to marshal models.json: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "models.json"), data, 0644); err != nil {
		t.Fatalf("failed to write models.json: %v", err)
	}
}

func newTestRegistry(t *testing.T, dir string) *ModelRegistry {
	t.Helper()
	return NewModelRegistry(dir, nil)
}

func TestModelRegistry_LoadModelsJSON(t *testing.T) {
	dir := t.TempDir()
	writeModelsJSON(t, dir, map[string]ModelMapping{
		"model-a": {Endpoint: "http://localhost:30000", GPUMemory: 8000, Category: "text-generation"},
		"model-b": {Endpoint: "http://localhost:30001", GPUMemory: 4000, Category: "text-generation"},
	})

	r := newTestRegistry(t, dir)
	if err := r.Start(); err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	defer r.Stop()

	models := r.GetAllModels()
	if len(models) != 2 {
		t.Errorf("expected 2 models, got %d", len(models))
	}
}

func TestModelRegistry_InvalidJSON(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "models.json"), []byte("not json {{{"), 0644); err != nil {
		t.Fatal(err)
	}

	r := newTestRegistry(t, dir)
	// Start should not panic on invalid JSON (just logs warning)
	if err := r.Start(); err != nil {
		t.Fatalf("Start should not return error on invalid JSON: %v", err)
	}
	defer r.Stop()

	// No models should be loaded
	if len(r.GetAllModels()) != 0 {
		t.Error("expected 0 models from invalid JSON")
	}
}

func TestModelRegistry_NoModelsJSON(t *testing.T) {
	dir := t.TempDir()
	// No models.json exists

	r := newTestRegistry(t, dir)
	if err := r.Start(); err != nil {
		t.Fatalf("Start should not fail when models.json is missing: %v", err)
	}
	defer r.Stop()

	if len(r.GetAllModels()) != 0 {
		t.Error("expected 0 models when no file")
	}
}

func TestModelRegistry_GetModel_ReturnsCopy(t *testing.T) {
	dir := t.TempDir()
	writeModelsJSON(t, dir, map[string]ModelMapping{
		"model-copy": {Endpoint: "http://localhost:30000"},
	})

	r := newTestRegistry(t, dir)
	r.Start()
	defer r.Stop()

	m, ok := r.GetModel("model-copy")
	if !ok {
		t.Fatal("expected model to exist")
	}

	// Mutate the copy — should not affect registry
	m.Endpoint = "http://mutated:9999"

	m2, _ := r.GetModel("model-copy")
	if m2.Endpoint == "http://mutated:9999" {
		t.Error("GetModel should return a copy, not a reference")
	}
}

func TestModelRegistry_GetReadyModels(t *testing.T) {
	dir := t.TempDir()
	writeModelsJSON(t, dir, map[string]ModelMapping{
		"ready-model": {Endpoint: "http://localhost:30000"},
	})

	r := newTestRegistry(t, dir)
	r.Start()
	defer r.Stop()

	// Simulate health update → model becomes ready
	r.onHealthStatusChange("ready-model", ModelHealthUnknown, ModelHealthHealthy)

	ready := r.GetReadyModels()
	found := false
	for _, m := range ready {
		if m.ID == "ready-model" {
			found = true
		}
	}
	if !found {
		t.Error("expected ready-model in GetReadyModels()")
	}
}

func TestModelRegistry_EnableDisable(t *testing.T) {
	dir := t.TempDir()
	writeModelsJSON(t, dir, map[string]ModelMapping{
		"toggle-model": {Endpoint: "http://localhost:30000"},
	})

	r := newTestRegistry(t, dir)
	r.Start()
	defer r.Stop()

	// Make it ready first
	r.onHealthStatusChange("toggle-model", ModelHealthUnknown, ModelHealthHealthy)

	if err := r.DisableModel("toggle-model"); err != nil {
		t.Fatalf("DisableModel failed: %v", err)
	}

	m, _ := r.GetModel("toggle-model")
	if m.Enabled {
		t.Error("expected model to be disabled")
	}
	if m.State != ModelStateDisabled {
		t.Errorf("expected state disabled, got %s", m.State)
	}

	if err := r.EnableModel("toggle-model"); err != nil {
		t.Fatalf("EnableModel failed: %v", err)
	}

	m, _ = r.GetModel("toggle-model")
	if !m.Enabled {
		t.Error("expected model to be enabled")
	}
}

func TestModelRegistry_EnableDisable_NotFound(t *testing.T) {
	dir := t.TempDir()
	r := newTestRegistry(t, dir)
	r.Start()
	defer r.Stop()

	if err := r.EnableModel("ghost"); err == nil {
		t.Error("expected error enabling non-existent model")
	}
	if err := r.DisableModel("ghost"); err == nil {
		t.Error("expected error disabling non-existent model")
	}
}

func TestModelRegistry_DisabledModel_NotInReady(t *testing.T) {
	dir := t.TempDir()
	writeModelsJSON(t, dir, map[string]ModelMapping{
		"dis-model": {Endpoint: "http://localhost:30000"},
	})

	r := newTestRegistry(t, dir)
	r.Start()
	defer r.Stop()

	r.onHealthStatusChange("dis-model", ModelHealthUnknown, ModelHealthHealthy)
	r.DisableModel("dis-model")

	for _, m := range r.GetReadyModels() {
		if m.ID == "dis-model" {
			t.Error("disabled model should not appear in GetReadyModels()")
		}
	}
}

func TestModelRegistry_GetModelEndpoint(t *testing.T) {
	dir := t.TempDir()
	writeModelsJSON(t, dir, map[string]ModelMapping{
		"ep-model": {Endpoint: "http://localhost:30000"},
	})

	r := newTestRegistry(t, dir)
	r.Start()
	defer r.Stop()

	// Before health update — should still return endpoint for non-unhealthy model
	ep, ok := r.GetModelEndpoint("ep-model")
	if !ok {
		t.Fatal("expected endpoint to be available")
	}
	if ep != "http://localhost:30000" {
		t.Errorf("unexpected endpoint: %s", ep)
	}
}

func TestModelRegistry_GetLocalModelName(t *testing.T) {
	dir := t.TempDir()
	writeModelsJSON(t, dir, map[string]ModelMapping{
		"ollama-model": {Endpoint: "http://localhost:11434", LocalModel: "llama3:8b"},
	})

	r := newTestRegistry(t, dir)
	r.Start()
	defer r.Stop()

	local := r.GetLocalModelName("ollama-model")
	if local != "llama3:8b" {
		t.Errorf("expected local model 'llama3:8b', got %q", local)
	}

	empty := r.GetLocalModelName("nonexistent")
	if empty != "" {
		t.Error("expected empty string for unknown model")
	}
}

func TestModelRegistry_GetModelAPIKey(t *testing.T) {
	dir := t.TempDir()
	writeModelsJSON(t, dir, map[string]ModelMapping{
		"auth-model": {Endpoint: "http://localhost:30000", APIKey: "sk-secret"},
	})

	r := newTestRegistry(t, dir)
	r.Start()
	defer r.Stop()

	key := r.GetModelAPIKey("auth-model")
	if key != "sk-secret" {
		t.Errorf("expected 'sk-secret', got %q", key)
	}
}

func TestModelRegistry_AddCallback(t *testing.T) {
	dir := t.TempDir()
	r := newTestRegistry(t, dir)

	added := make(chan string, 5)
	r.SetCallbacks(
		func(m *RegisteredModel) { added <- m.ID },
		nil,
		nil,
	)
	r.Start()
	defer r.Stop()

	// Write models.json after Start (hot-reload)
	writeModelsJSON(t, dir, map[string]ModelMapping{
		"new-model": {Endpoint: "http://localhost:30000"},
	})
	r.ReloadConfig()

	select {
	case id := <-added:
		if id != "new-model" {
			t.Errorf("expected 'new-model', got %q", id)
		}
	case <-time.After(500 * time.Millisecond):
		t.Error("expected onModelAdded callback to fire")
	}
}

func TestModelRegistry_RemoveCallback(t *testing.T) {
	dir := t.TempDir()
	writeModelsJSON(t, dir, map[string]ModelMapping{
		"to-remove": {Endpoint: "http://localhost:30000"},
	})

	r := newTestRegistry(t, dir)
	removed := make(chan string, 5)
	r.SetCallbacks(nil, func(id string) { removed <- id }, nil)
	r.Start()
	defer r.Stop()

	// Overwrite with empty models
	writeModelsJSON(t, dir, map[string]ModelMapping{})
	r.ReloadConfig()

	select {
	case id := <-removed:
		if id != "to-remove" {
			t.Errorf("expected 'to-remove', got %q", id)
		}
	case <-time.After(500 * time.Millisecond):
		t.Error("expected onModelRemoved callback to fire")
	}
}

func TestModelRegistry_CallbackPanicRecovery(t *testing.T) {
	dir := t.TempDir()
	r := newTestRegistry(t, dir)

	// Callback that panics — registry should not crash
	r.SetCallbacks(
		func(m *RegisteredModel) { panic("intentional panic in test") },
		nil,
		nil,
	)
	r.Start()
	defer r.Stop()

	writeModelsJSON(t, dir, map[string]ModelMapping{
		"panic-model": {Endpoint: "http://localhost:30000"},
	})

	// Should not panic
	r.ReloadConfig()
	time.Sleep(100 * time.Millisecond)

	// Registry should still be functional
	if len(r.GetAllModels()) == 0 {
		t.Error("expected models to be loaded despite callback panic")
	}
}

func TestModelRegistry_HotReload(t *testing.T) {
	dir := t.TempDir()
	writeModelsJSON(t, dir, map[string]ModelMapping{
		"initial": {Endpoint: "http://localhost:30000"},
	})

	r := NewModelRegistry(dir, nil)
	if err := r.Start(); err != nil {
		t.Fatal(err)
	}
	defer r.Stop()

	// Verify initial load
	if len(r.GetAllModels()) != 1 {
		t.Fatal("expected 1 initial model")
	}

	// Overwrite with 2 models — watcher should pick it up
	writeModelsJSON(t, dir, map[string]ModelMapping{
		"initial":  {Endpoint: "http://localhost:30000"},
		"model-v2": {Endpoint: "http://localhost:30001"},
	})

	// Wait for debounce (500ms) + processing
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if len(r.GetAllModels()) == 2 {
			return // success
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Errorf("expected 2 models after hot-reload, got %d", len(r.GetAllModels()))
}

func TestModelRegistry_ReloadConfig(t *testing.T) {
	dir := t.TempDir()
	r := newTestRegistry(t, dir)
	r.Start()
	defer r.Stop()

	writeModelsJSON(t, dir, map[string]ModelMapping{
		"reload-model": {Endpoint: "http://localhost:30000"},
	})

	if err := r.ReloadConfig(); err != nil {
		t.Fatalf("ReloadConfig failed: %v", err)
	}

	if len(r.GetAllModels()) != 1 {
		t.Errorf("expected 1 model after reload, got %d", len(r.GetAllModels()))
	}
}

func TestModelRegistry_GetStatusSummary(t *testing.T) {
	dir := t.TempDir()
	writeModelsJSON(t, dir, map[string]ModelMapping{
		"m1": {Endpoint: "http://localhost:30000"},
		"m2": {Endpoint: "http://localhost:30001"},
	})

	r := newTestRegistry(t, dir)
	r.Start()
	defer r.Stop()

	r.onHealthStatusChange("m1", ModelHealthUnknown, ModelHealthHealthy)
	r.DisableModel("m2")

	summary := r.GetStatusSummary()
	if summary["total"].(int) != 2 {
		t.Errorf("expected total=2, got %v", summary["total"])
	}
	if summary["disabled"].(int) != 1 {
		t.Errorf("expected disabled=1, got %v", summary["disabled"])
	}
}

func TestModelRegistry_GetReadyModelIDs(t *testing.T) {
	dir := t.TempDir()
	writeModelsJSON(t, dir, map[string]ModelMapping{
		"id-model": {Endpoint: "http://localhost:30000"},
	})

	r := newTestRegistry(t, dir)
	r.Start()
	defer r.Stop()

	r.onHealthStatusChange("id-model", ModelHealthUnknown, ModelHealthHealthy)

	ids := r.GetReadyModelIDs()
	if len(ids) == 0 {
		t.Error("expected at least 1 ready model ID")
	}
	found := false
	for _, id := range ids {
		if id == "id-model" {
			found = true
		}
	}
	if !found {
		t.Error("expected 'id-model' in GetReadyModelIDs()")
	}
}

func TestModelRegistry_ConcurrentRead(t *testing.T) {
	dir := t.TempDir()
	writeModelsJSON(t, dir, map[string]ModelMapping{
		"concurrent-1": {Endpoint: "http://localhost:30000"},
		"concurrent-2": {Endpoint: "http://localhost:30001"},
	})

	r := newTestRegistry(t, dir)
	r.Start()
	defer r.Stop()

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r.GetAllModels()
			r.GetReadyModels()
			r.GetModel("concurrent-1")
			r.GetStatusSummary()
		}()
	}
	wg.Wait()
}

func TestModelRegistry_GetModelMappings(t *testing.T) {
	dir := t.TempDir()
	writeModelsJSON(t, dir, map[string]ModelMapping{
		"map-model": {Endpoint: "http://localhost:30000", GPUMemory: 8000, Category: "text-generation"},
	})

	r := newTestRegistry(t, dir)
	r.Start()
	defer r.Stop()

	mappings := r.GetModelMappings()
	if len(mappings) != 1 {
		t.Errorf("expected 1 mapping, got %d", len(mappings))
	}
	m, ok := mappings["map-model"]
	if !ok {
		t.Error("expected 'map-model' in mappings")
	}
	if m.Endpoint != "http://localhost:30000" {
		t.Errorf("unexpected endpoint: %s", m.Endpoint)
	}
}

func TestModelRegistry_GetAllModelHealthMap(t *testing.T) {
	dir := t.TempDir()
	writeModelsJSON(t, dir, map[string]ModelMapping{
		"health-model": {Endpoint: "http://localhost:30000"},
	})

	r := newTestRegistry(t, dir)
	r.Start()
	defer r.Stop()

	r.onHealthStatusChange("health-model", ModelHealthUnknown, ModelHealthHealthy)

	healthMap := r.GetAllModelHealthMap()
	h, ok := healthMap["health-model"]
	if !ok {
		t.Error("expected health-model in health map")
	}
	if h != "healthy" {
		t.Errorf("expected 'healthy', got %q", h)
	}
}

func TestModelRegistry_HealthUpdateCallback(t *testing.T) {
	dir := t.TempDir()
	writeModelsJSON(t, dir, map[string]ModelMapping{
		"cb-model": {Endpoint: "http://localhost:30000"},
	})

	r := newTestRegistry(t, dir)

	healthUpdates := make(chan map[string]string, 5)
	r.SetHealthUpdateCallback(func(m map[string]string) {
		healthUpdates <- m
	})

	r.Start()
	defer r.Stop()

	r.onHealthStatusChange("cb-model", ModelHealthUnknown, ModelHealthHealthy)

	select {
	case update := <-healthUpdates:
		if update["cb-model"] != "healthy" {
			t.Errorf("expected health 'healthy', got %q", update["cb-model"])
		}
	case <-time.After(500 * time.Millisecond):
		t.Error("expected health update callback")
	}
}
