package computing

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

// --- ModelServerError tests ---

func TestModelServerError_Error(t *testing.T) {
	err := &ModelServerError{StatusCode: 503, Message: "service unavailable"}
	got := err.Error()
	if got != "model server returned HTTP 503: service unavailable" {
		t.Errorf("unexpected error message: %s", got)
	}
}

// --- parseModelServerError tests ---

func TestParseModelServerError_OpenAIFormat(t *testing.T) {
	body := []byte(`{"error":{"message":"model not found","type":"invalid_request_error"}}`)
	err := parseModelServerError(404, body)
	if err.StatusCode != 404 {
		t.Errorf("expected status 404, got %d", err.StatusCode)
	}
	if err.Message != "model not found" {
		t.Errorf("expected 'model not found', got %q", err.Message)
	}
}

func TestParseModelServerError_RawBody(t *testing.T) {
	body := []byte("Internal server error")
	err := parseModelServerError(500, body)
	if err.StatusCode != 500 {
		t.Errorf("expected 500, got %d", err.StatusCode)
	}
	if err.Message != "Internal server error" {
		t.Errorf("expected raw body as message, got %q", err.Message)
	}
}

func TestParseModelServerError_EmptyBody(t *testing.T) {
	err := parseModelServerError(503, []byte(""))
	if err.Message == "" {
		t.Error("expected fallback message for empty body")
	}
}

func TestParseModelServerError_LongBodyTruncated(t *testing.T) {
	longBody := make([]byte, 300)
	for i := range longBody {
		longBody[i] = 'x'
	}
	err := parseModelServerError(500, longBody)
	if len(err.Message) > 210 { // 200 chars + "..."
		t.Errorf("expected message to be truncated, got length %d", len(err.Message))
	}
}

func TestParseModelServerError_MalformedJSON(t *testing.T) {
	body := []byte(`not json at all`)
	err := parseModelServerError(400, body)
	if err.Message != "not json at all" {
		t.Errorf("expected raw body, got %q", err.Message)
	}
}

// --- HttpClient tests (using httptest.Server) ---

func TestHttpClient_PostJSON_Success(t *testing.T) {
	type Response struct {
		OK bool `json:"ok"`
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("expected application/json content type")
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(Response{OK: true})
	}))
	defer srv.Close()

	client := NewHttpClient(srv.URL, nil)
	var resp Response
	if err := client.PostJSON("api/test", map[string]string{"key": "value"}, &resp); err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}
	if !resp.OK {
		t.Error("expected OK=true in response")
	}
}

func TestHttpClient_Get_Success(t *testing.T) {
	type Response struct {
		Name string `json:"name"`
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(Response{Name: "test"})
	}))
	defer srv.Close()

	client := NewHttpClient(srv.URL, nil)
	var resp Response
	if err := client.Get("api/info", nil, &resp); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Name != "test" {
		t.Errorf("expected name 'test', got %q", resp.Name)
	}
}

func TestHttpClient_NonOKStatus_ReturnsModelServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write([]byte(`{"error":{"message":"overloaded","type":"server_error"}}`))
	}))
	defer srv.Close()

	client := NewHttpClient(srv.URL, nil)
	var dummy interface{}
	err := client.Get("api/chat", nil, &dummy)
	if err == nil {
		t.Fatal("expected error for 503 response")
	}
	mse, ok := err.(*ModelServerError)
	if !ok {
		t.Fatalf("expected *ModelServerError, got %T", err)
	}
	if mse.StatusCode != 503 {
		t.Errorf("expected status 503, got %d", mse.StatusCode)
	}
	if mse.Message != "overloaded" {
		t.Errorf("expected message 'overloaded', got %q", mse.Message)
	}
}

func TestHttpClient_NonPointerDest(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	client := NewHttpClient(srv.URL, nil)
	var notAPointer map[string]string
	err := client.Get("api/test", nil, notAPointer)
	if err == nil {
		t.Error("expected error when dest is not a pointer")
	}
}

func TestHttpClient_CustomHeaders(t *testing.T) {
	gotHeader := make(chan string, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader <- r.Header.Get("X-Custom")
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	headers := http.Header{}
	headers.Set("X-Custom", "my-value")
	client := NewHttpClient(srv.URL, headers)

	var resp interface{}
	client.Get("api/test", nil, &resp)

	select {
	case h := <-gotHeader:
		if h != "my-value" {
			t.Errorf("expected 'my-value', got %q", h)
		}
	default:
		t.Error("expected server to receive request with custom header")
	}
}

func TestHttpClient_ConnectionRefused(t *testing.T) {
	client := NewHttpClient("http://localhost:1", nil)
	var resp interface{}
	err := client.Get("api/test", nil, &resp)
	if err == nil {
		t.Error("expected error for connection refused")
	}
}

func TestHttpClient_PathWithSlash(t *testing.T) {
	gotPath := make(chan string, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath <- r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	client := NewHttpClient(srv.URL, nil)
	var resp interface{}
	// Test path starting with /
	client.Get("/api/test", nil, &resp)
	select {
	case p := <-gotPath:
		if p != "/api/test" {
			t.Errorf("expected path '/api/test', got %q", p)
		}
	default:
	}
}

func TestHttpClient_PathWithoutSlash(t *testing.T) {
	gotPath := make(chan string, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath <- r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	client := NewHttpClient(srv.URL, nil)
	var resp interface{}
	// Test path without leading /
	client.Get("api/test", nil, &resp)
	select {
	case p := <-gotPath:
		if p != "/api/test" {
			t.Errorf("expected path '/api/test', got %q", p)
		}
	default:
	}
}

func TestHttpClient_PostForm(t *testing.T) {
	gotForm := make(chan string, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r.ParseForm()
		gotForm <- r.FormValue("username")
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	client := NewHttpClient(srv.URL, nil)
	var resp interface{}
	client.PostForm("api/login", url.Values{"username": {"alice"}}, &resp)

	select {
	case val := <-gotForm:
		if val != "alice" {
			t.Errorf("expected username 'alice', got %q", val)
		}
	default:
		t.Error("expected form value to be received")
	}
}
