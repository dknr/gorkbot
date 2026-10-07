// Package testutil provides an in-memory fake for the LLM client so tests in
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
// (atomically, via the exported fields) between calls.
type Client struct {
	mu      sync.Mutex
	Response string
	Err     error
	Calls   int
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

// ErrFake is a sentinel error for tests.
var ErrFake = errors.New("fake llm error")
