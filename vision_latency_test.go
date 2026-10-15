//go:build integration

package main

import (
	"context"
	"encoding/base64"
	"os"
	"strings"
	"testing"
	"time"

	"gorkbot/llm"
)

// TestVisionLatency is the eyes regression trip. It is the only test that
// measures the caption path against the live endpoint, and it exists because
// the two knobs there interact: reasoning is charged against max_tokens, so a
// request that leaves thinking enabled at a 100-token budget returns an empty
// caption rather than a short one. That is invisible in a unit test.
//
// The ceiling is generous on purpose. Measured on the same drawing: 15.2s with
// the old 1600-token prompt, 2.4s now, against a shared endpoint.
func TestVisionLatency(t *testing.T) {
	cfg, err := LoadConfig("config.json", false)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.LLMVision.BaseURL == "" {
		t.Skip("no llm_vision configured")
	}
	png, err := os.ReadFile("integration-draws/attempt-193208.png")
	if err != nil {
		t.Skip("no sample image:", err)
	}
	b64 := base64.StdEncoding.EncodeToString(png)

	c := llm.NewClient(cfg.LLMVision.BaseURL, cfg.LLMVision.Model, cfg.LLMVision.APIKey, "")
	c.Think, c.MaxTokens = ptrFalse(), cfg.LLMVision.MaxTokens

	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	start := time.Now()
	text, _, completion, _, err := c.CompleteMultimodal(ctx, b64, "image/png", cfg.LLMVision.ImagePrompt)
	elapsed := time.Since(start)
	t.Logf("caption in %.1fs (%d tokens): %.100s", elapsed.Seconds(), completion, text)
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(text) == "" {
		t.Fatal("empty caption: the token budget left no room for the answer. " +
			"Thinking must stay off on this path.")
	}
	if elapsed > 15*time.Second {
		t.Fatalf("caption took %v; the terse prompt and max_tokens=%d are what keep the "+
			"draw cycle under a minute", elapsed, cfg.LLMVision.MaxTokens)
	}
}

func ptrFalse() *bool {
	no := false
	return &no
}
