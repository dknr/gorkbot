package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCompleteHappyPath(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/chat/completions" {
			t.Errorf("expected path /chat/completions, got %s", r.URL.Path)
		}
		if ct := r.Header.Get("Content-Type"); ct != "application/json" {
			t.Errorf("expected Content-Type application/json, got %s", ct)
		}

		var req ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("failed to decode request: %v", err)
		}

		if req.Model != "test-model" {
			t.Errorf("expected model test-model, got %s", req.Model)
		}

		if len(req.Messages) < 1 || req.Messages[0].Role != "system" {
			t.Fatalf("expected first message to be system, got %v", req.Messages[0])
		}
		if req.Messages[0].Content != "You are a test." {
			t.Errorf("expected system prompt 'You are a test.', got %q", req.Messages[0].Content)
		}

		if len(req.Messages) < 2 || req.Messages[1].Role != "user" {
			t.Fatalf("expected second message to be user, got %v", req.Messages[1])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ChatResponse{
			Choices: []Choice{{Message: Message{Content: "\nHello!"}}},
			Usage:   UsageInfo{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30},
		})
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "test-model", "", "You are a test.")
	history := []ChatMessage{{Role: "user", Content: "Hi"}}

	content, pt, ct, tt, err := c.Complete(context.Background(), history)
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}
	if content != "Hello!" {
		t.Errorf("expected content 'Hello!' (trimmed), got %q", content)
	}
	if pt != 10 || ct != 20 || tt != 30 {
		t.Errorf("expected usage (10, 20, 30), got (%d, %d, %d)", pt, ct, tt)
	}
}

func TestCompleteAuthHeader(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Authorization")
		if auth != "Bearer secret-key" {
			t.Errorf("expected Authorization 'Bearer secret-key', got %q", auth)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ChatResponse{
			Choices: []Choice{{Message: Message{Content: "ok"}}},
		})
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "test-model", "secret-key", "")
	_, _, _, _, err := c.Complete(context.Background(), nil)
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}
}

func TestCompleteNoAuthHeader(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if auth := r.Header.Get("Authorization"); auth != "" {
			t.Errorf("expected no Authorization header, got %q", auth)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ChatResponse{
			Choices: []Choice{{Message: Message{Content: "ok"}}},
		})
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "test-model", "", "")
	_, _, _, _, err := c.Complete(context.Background(), nil)
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}
}

func TestCompleteNon200(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "bad gateway", http.StatusBadGateway)
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "test-model", "", "")
	_, _, _, _, err := c.Complete(context.Background(), nil)
	if err == nil {
		t.Fatal("expected error for non-200 status, got nil")
	}
	if !strings.Contains(err.Error(), "502") {
		t.Errorf("expected error to mention status code 502, got: %v", err)
	}
	if !strings.Contains(err.Error(), "bad gateway") {
		t.Errorf("expected error to include the response body, got: %v", err)
	}
}

func TestCompleteNoChoices(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ChatResponse{Choices: []Choice{}})
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "test-model", "", "")
	_, _, _, _, err := c.Complete(context.Background(), nil)
	if err == nil {
		t.Fatal("expected error for empty choices, got nil")
	}
	if !strings.Contains(err.Error(), "no choices") {
		t.Errorf("expected 'no choices' error, got: %v", err)
	}
}

func TestCompleteInvalidJSON(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, "not json")
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "test-model", "", "")
	_, _, _, _, err := c.Complete(context.Background(), nil)
	if err == nil {
		t.Fatal("expected error for invalid JSON, got nil")
	}
	if !strings.Contains(err.Error(), "decode response") {
		t.Errorf("expected 'decode response' error, got: %v", err)
	}
}

func TestCompleteContextCancelled(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ChatResponse{
			Choices: []Choice{{Message: Message{Content: "ok"}}},
		})
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "test-model", "", "")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, _, _, _, err := c.Complete(ctx, nil)
	if err == nil {
		t.Fatal("expected error for cancelled context, got nil")
	}
}

func TestCompleteTrimsWhitespace(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ChatResponse{
			Choices: []Choice{{Message: Message{Content: "\r\n  \t leading"}}},
		})
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "test-model", "", "")
	content, _, _, _, err := c.Complete(context.Background(), nil)
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}
	if content != "leading" {
		t.Errorf("expected 'leading' (all leading whitespace trimmed), got %q", content)
	}
}

func TestNewClient(t *testing.T) {
	c := NewClient("http://example.com", "model-x", "key", "sys")
	if c.BaseURL != "http://example.com" {
		t.Errorf("expected BaseURL 'http://example.com', got %q", c.BaseURL)
	}
	if c.Model != "model-x" {
		t.Errorf("expected Model 'model-x', got %q", c.Model)
	}
	if c.APIKey != "key" {
		t.Errorf("expected APIKey 'key', got %q", c.APIKey)
	}
	if c.SystemPrompt != "sys" {
		t.Errorf("expected SystemPrompt 'sys', got %q", c.SystemPrompt)
	}
}

func TestCompleteMultimodalHappyPath(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/chat/completions" {
			t.Errorf("expected path /chat/completions, got %s", r.URL.Path)
		}
		var req ChatRequestMultimodal
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("failed to decode request: %v", err)
		}
		if req.Model != "vision-model" {
			t.Errorf("expected model vision-model, got %s", req.Model)
		}
		if len(req.Messages) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(req.Messages))
		}
		if req.Messages[0].Role != "system" {
			t.Errorf("expected first message system, got %s", req.Messages[0].Role)
		}
		if len(req.Messages[0].Content) != 1 || req.Messages[0].Content[0].Type != "text" {
			t.Errorf("expected system text part, got %+v", req.Messages[0].Content)
		}
		if req.Messages[0].Content[0].Text != "You are a vision test." {
			t.Errorf("unexpected system prompt %q", req.Messages[0].Content[0].Text)
		}
		if req.Messages[1].Role != "user" {
			t.Errorf("expected second message user, got %s", req.Messages[1].Role)
		}
		parts := req.Messages[1].Content
		if len(parts) != 2 {
			t.Fatalf("expected 2 user content parts, got %d", len(parts))
		}
		if parts[0].Type != "image_url" || parts[0].ImageURL == nil {
			t.Fatalf("expected image_url part, got %+v", parts[0])
		}
		wantURL := "data:image/png;base64,QUJD"
		if parts[0].ImageURL.URL != wantURL {
			t.Errorf("expected image URL %q, got %q", wantURL, parts[0].ImageURL.URL)
		}
		if parts[1].Type != "text" || parts[1].Text != "describe it" {
			t.Errorf("expected text part 'describe it', got %+v", parts[1])
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ChatResponse{
			Choices: []Choice{{Message: Message{Content: " a rock"}}},
			Usage:   UsageInfo{PromptTokens: 5, CompletionTokens: 7, TotalTokens: 12},
		})
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "vision-model", "", "You are a vision test.")
	content, pt, ct, tt, err := c.CompleteMultimodal(context.Background(), "QUJD", "image/png", "describe it")
	if err != nil {
		t.Fatalf("CompleteMultimodal() error: %v", err)
	}
	if content != "a rock" {
		t.Errorf("expected trimmed 'a rock', got %q", content)
	}
	if pt != 5 || ct != 7 || tt != 12 {
		t.Errorf("expected usage (5, 7, 12), got (%d, %d, %d)", pt, ct, tt)
	}
}

func TestCompleteMultimodalOmitsSystemPromptWhenEmpty(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ChatRequestMultimodal
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("failed to decode request: %v", err)
		}
		if len(req.Messages) != 1 {
			t.Fatalf("expected only the user message, got %d", len(req.Messages))
		}
		if req.Messages[0].Role != "user" {
			t.Errorf("expected the user message first, got %s", req.Messages[0].Role)
		}
		if req.Think == nil || *req.Think {
			t.Errorf("expected think=false, got %v", req.Think)
		}
		if req.MaxTokens != 200 {
			t.Errorf("expected max_tokens 200, got %d", req.MaxTokens)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ChatResponse{Choices: []Choice{{Message: Message{Content: "a frog"}}}})
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "qwen3.8-flash-next", "", "")
	think := false
	c.Think = &think
	c.MaxTokens = 200
	if _, _, _, _, err := c.CompleteMultimodal(context.Background(), "QUJD", "image/png", "describe it"); err != nil {
		t.Fatalf("CompleteMultimodal() error: %v", err)
	}
}

func TestCompleteMultimodalOmitsUnsetControls(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := new(bytes.Buffer)
		if _, err := body.ReadFrom(r.Body); err != nil {
			t.Fatalf("failed to read request: %v", err)
		}
		for _, want := range []string{"\"think\"", "\"max_tokens\""} {
			if strings.Contains(body.String(), want) {
				t.Errorf("expected %s to be omitted, got %s", want, body.String())
			}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ChatResponse{Choices: []Choice{{Message: Message{Content: "ok"}}}})
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "vision-model", "", "sys")
	if _, _, _, _, err := c.CompleteMultimodal(context.Background(), "QUJD", "image/png", "describe it"); err != nil {
		t.Fatalf("CompleteMultimodal() error: %v", err)
	}
}

func TestCompleteMultimodalErrorIncludesBody(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":{"message":"image input requires starting ds4-server with --vision FILE"}}`)
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "vision-model", "", "")
	_, _, _, _, err := c.CompleteMultimodal(context.Background(), "QUJD", "image/png", "describe it")
	if err == nil {
		t.Fatal("expected error for 400 status, got nil")
	}
	if !strings.Contains(err.Error(), "400") || !strings.Contains(err.Error(), "--vision") {
		t.Errorf("expected status code and the server reason, got: %v", err)
	}
}

func TestCompleteMultimodalNonImageMimeFallsBackToJPEG(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ChatRequestMultimodal
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("failed to decode request: %v", err)
		}
		parts := req.Messages[1].Content
		if parts[0].ImageURL.URL != "data:image/jpeg;base64,QQ==" {
			t.Errorf("expected jpeg fallback data URL, got %q", parts[0].ImageURL.URL)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ChatResponse{Choices: []Choice{{Message: Message{Content: "ok"}}}})
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "vision-model", "", "sys")
	if _, _, _, _, err := c.CompleteMultimodal(context.Background(), "QQ==", "text/plain", "p"); err != nil {
		t.Fatalf("CompleteMultimodal() error: %v", err)
	}
}

func TestCompleteMultimodalAuthHeader(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if auth := r.Header.Get("Authorization"); auth != "Bearer vision-key" {
			t.Errorf("expected Authorization 'Bearer vision-key', got %q", auth)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ChatResponse{Choices: []Choice{{Message: Message{Content: "ok"}}}})
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "vision-model", "vision-key", "")
	if _, _, _, _, err := c.CompleteMultimodal(context.Background(), "QQ==", "image/png", "p"); err != nil {
		t.Fatalf("CompleteMultimodal() error: %v", err)
	}
}

func TestCompleteMultimodalNon200(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "nope", http.StatusNotFound)
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "vision-model", "", "")
	if _, _, _, _, err := c.CompleteMultimodal(context.Background(), "QQ==", "image/png", "p"); err == nil {
		t.Fatal("expected error for non-200 status, got nil")
	}
}

func TestCompleteMultimodalNoChoices(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ChatResponse{})
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "vision-model", "", "")
	if _, _, _, _, err := c.CompleteMultimodal(context.Background(), "QQ==", "image/png", "p"); err == nil {
		t.Fatal("expected error for empty choices, got nil")
	}
}

func TestCompleteMultimodalInvalidJSON(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "not json")
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "vision-model", "", "")
	if _, _, _, _, err := c.CompleteMultimodal(context.Background(), "QQ==", "image/png", "p"); err == nil {
		t.Fatal("expected error for invalid JSON, got nil")
	}
}

func TestCompleteMultimodalContextCancelled(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ChatResponse{Choices: []Choice{{Message: Message{Content: "ok"}}}})
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "vision-model", "", "")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, _, _, err := c.CompleteMultimodal(ctx, "QQ==", "image/png", "p"); err == nil {
		t.Fatal("expected error for cancelled context, got nil")
	}
}
