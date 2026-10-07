// Package testutil provides in-memory fakes for the LLM client so tests in
// other packages can exercise code paths that depend on llm.Client without
// hitting a network endpoint.
package testutil

import (
	"context"
	"errors"
	"sync"

	"gorkbot/llm"
)

// Client is a programmable fake llm.Client. Its behavior can be changed
// (atomically, via the mutex-guarded fields) between calls.
type Client struct {
	mu           sync.Mutex
	Response     string
	Err          error
	Calls        int
	VisionCalls  int
	VisionPrompt string
	VisionImage  string
	VisionMime   string
}

// Complete records the call and returns the configured response or error.
func (c *Client) Complete(ctx context.Context, history []llm.ChatMessage) (string, int, int, int, error) {
	c.mu.Lock()
	c.Calls++
	resp, err := c.Response, c.Err
	c.mu.Unlock()
	if err != nil {
		return "", 0, 0, 0, err
	}
	return resp, 0, 0, 0, nil
}

// CompleteMultimodal records the call and returns the configured response or
// error.
func (c *Client) CompleteMultimodal(ctx context.Context, imageBase64, mimeType, textPrompt string) (string, int, int, int, error) {
	c.mu.Lock()
	c.VisionCalls++
	c.VisionImage = imageBase64
	c.VisionMime = mimeType
	c.VisionPrompt = textPrompt
	resp, err := c.Response, c.Err
	c.mu.Unlock()
	if err != nil {
		return "", 0, 0, 0, err
	}
	return resp, 0, 0, 0, nil
}

// ErrFake is a sentinel error for tests.
var ErrFake = errors.New("fake llm error")
