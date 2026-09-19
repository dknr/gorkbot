package main

import (
	"testing"
	"time"

	"gorkbot/llm"
)

func TestAddMessage(t *testing.T) {
	h := NewHistoryManager(10, time.Hour)

	h.AddMessage(llm.ChatMessage{Role: "user", Content: "hello"})
	h.AddMessage(llm.ChatMessage{Role: "assistant", Content: "hi"})

	msgs := h.Messages()
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}
	if msgs[0].Content != "hello" {
		t.Errorf("expected first message 'hello', got %q", msgs[0].Content)
	}
	if msgs[1].Content != "hi" {
		t.Errorf("expected second message 'hi', got %q", msgs[1].Content)
	}
}

func TestMaxMessagesTrim(t *testing.T) {
	h := NewHistoryManager(3, time.Hour)

	for i := 0; i < 5; i++ {
		h.AddMessage(llm.ChatMessage{Role: "user", Content: string(rune('a' + i))})
	}

	msgs := h.Messages()
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages (max), got %d", len(msgs))
	}
	if msgs[0].Content != "c" {
		t.Errorf("expected first message 'c', got %q", msgs[0].Content)
	}
	if msgs[2].Content != "e" {
		t.Errorf("expected last message 'e', got %q", msgs[2].Content)
	}
}

func TestTimeoutClearsHistory(t *testing.T) {
	h := NewHistoryManager(10, 1*time.Millisecond)

	h.AddMessage(llm.ChatMessage{Role: "user", Content: "old"})

	time.Sleep(5 * time.Millisecond)

	h.AddMessage(llm.ChatMessage{Role: "user", Content: "new"})

	msgs := h.Messages()
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message after timeout clear, got %d", len(msgs))
	}
	if msgs[0].Content != "new" {
		t.Errorf("expected 'new', got %q", msgs[0].Content)
	}
}

func TestNoClearWithinTimeout(t *testing.T) {
	h := NewHistoryManager(10, time.Hour)

	h.AddMessage(llm.ChatMessage{Role: "user", Content: "first"})
	h.AddMessage(llm.ChatMessage{Role: "user", Content: "second"})

	msgs := h.Messages()
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages (within timeout), got %d", len(msgs))
	}
}

func TestMessagesEmpty(t *testing.T) {
	h := NewHistoryManager(10, time.Hour)
	msgs := h.Messages()
	if len(msgs) != 0 {
		t.Fatalf("expected 0 messages, got %d", len(msgs))
	}
}

func TestMessagesReturnsCopy(t *testing.T) {
	h := NewHistoryManager(10, time.Hour)
	h.AddMessage(llm.ChatMessage{Role: "user", Content: "hello"})

	msgs := h.Messages()
	msgs[0].Content = "modified"

	original := h.Messages()
	if original[0].Content != "hello" {
		t.Errorf("expected internal state 'hello', got %q (Messages should return a copy)", original[0].Content)
	}
}

func TestMaxMessagesOne(t *testing.T) {
	h := NewHistoryManager(1, time.Hour)

	h.AddMessage(llm.ChatMessage{Role: "user", Content: "a"})
	h.AddMessage(llm.ChatMessage{Role: "user", Content: "b"})

	msgs := h.Messages()
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	if msgs[0].Content != "b" {
		t.Errorf("expected 'b', got %q", msgs[0].Content)
	}
}
