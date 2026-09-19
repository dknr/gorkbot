package llm

import (
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
