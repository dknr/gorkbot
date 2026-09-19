package main

import (
	"sync"
	"time"

	"gorkbot/llm"
)

// HistoryManager maintains conversation context.
type HistoryManager struct {
	mu           sync.Mutex
	messages     []llm.ChatMessage
	maxMessages  int
	lastActivity time.Time
	timeout      time.Duration
}

// NewHistoryManager creates a new HistoryManager.
func NewHistoryManager(maxMessages int, timeout time.Duration) *HistoryManager {
	return &HistoryManager{
		messages:     make([]llm.ChatMessage, 0, maxMessages),
		maxMessages:  maxMessages,
		lastActivity: time.Now(),
		timeout:      timeout,
	}
}

// AddMessage adds a message to history, clearing if timeout exceeded.
func (h *HistoryManager) AddMessage(msg llm.ChatMessage) {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Check for timeout
	if time.Since(h.lastActivity) > h.timeout {
		h.messages = h.messages[:0] // clear history
	}
	h.lastActivity = time.Now()
	h.messages = append(h.messages, msg)

	// Trim to max
	if len(h.messages) > h.maxMessages {
		h.messages = h.messages[len(h.messages)-h.maxMessages:]
	}
}

// Messages returns a copy of the current history.
func (h *HistoryManager) Messages() []llm.ChatMessage {
	h.mu.Lock()
	defer h.mu.Unlock()

	out := make([]llm.ChatMessage, len(h.messages))
	copy(out, h.messages)
	return out
}
