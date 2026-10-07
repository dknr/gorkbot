package main

import (
	"context"
	"testing"
	"time"

	"github.com/rs/zerolog"
	"maunium.net/go/mautrix/id"

	"gorkbot/llm/testutil"
)

// newTestBot builds a Bot wired to a fake LLM client and in-memory
// reply/typing callbacks.
func newTestBot(t *testing.T, fake *testutil.Client, replies *[]string, typing *[]bool) *Bot {
	t.Helper()
	log := zerolog.Nop()

	b := &Bot{
		log:            log,
		mention:        "@gork:example.org",
		imagePrompt:    "Describe what you see in this image",
		maxHistory:     10,
		historyTimeout: time.Minute,
		historyMgr:     NewRoomHistoryManager(),
		replyFn: func(_ context.Context, _ id.RoomID, _ id.EventID, response string) {
			if replies != nil {
				*replies = append(*replies, response)
			}
		},
		completeFn: fake.Complete,
		stopCh:     make(chan struct{}),
	}
	if typing != nil {
		b.typingFn = func(_ context.Context, _ id.RoomID, isTyping bool) {
			*typing = append(*typing, isTyping)
		}
	}
	return b
}

func TestHandleLLMError(t *testing.T) {
	fake := &testutil.Client{Err: testutil.ErrFake}
	var replies []string
	var typing []bool
	b := newTestBot(t, fake, &replies, &typing)

	b.handle(message{
		ctx:       context.Background(),
		roomID:    id.RoomID("!room:example.org"),
		sender:    "@user:example.org",
		body:      "hello",
		mentioned: true,
		prompt:    "hello",
	})

	if fake.Calls != 1 {
		t.Fatalf("expected 1 LLM call, got %d", fake.Calls)
	}
	if len(replies) != 1 {
		t.Fatalf("expected 1 reply, got %d: %v", len(replies), replies)
	}
	if replies[0] == "" {
		t.Errorf("expected a non-empty failure reply, got empty string")
	}
	// Typing indicator must be set then cleared around the failed call.
	if len(typing) != 2 || typing[0] != true || typing[1] != false {
		t.Errorf("expected typing [true false], got %v", typing)
	}
	// History must contain the user message but no assistant turn.
	hm := b.historyMgr.GetOrCreate(id.RoomID("!room:example.org"), b.maxHistory, b.historyTimeout)
	msgs := hm.Messages()
	if len(msgs) != 1 || msgs[0].Role != "user" {
		t.Errorf("expected 1 user message in history, got %+v", msgs)
	}
}

func TestHandleLLMSuccess(t *testing.T) {
	fake := &testutil.Client{Response: "a reply"}
	var replies []string
	b := newTestBot(t, fake, &replies, nil)

	b.handle(message{
		ctx:       context.Background(),
		roomID:    id.RoomID("!room:example.org"),
		sender:    "@user:example.org",
		body:      "hello",
		mentioned: true,
		prompt:    "hello",
	})

	if len(replies) != 1 || replies[0] != "a reply" {
		t.Fatalf("expected reply \"a reply\", got %v", replies)
	}
	hm := b.historyMgr.GetOrCreate(id.RoomID("!room:example.org"), b.maxHistory, b.historyTimeout)
	msgs := hm.Messages()
	if len(msgs) != 2 || msgs[1].Role != "assistant" || msgs[1].Content != "a reply" {
		t.Errorf("expected assistant turn in history, got %+v", msgs)
	}
}
