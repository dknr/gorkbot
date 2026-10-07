package main

import (
	"context"
	"strings"
	"sync"
	"testing"

	"maunium.net/go/mautrix/id"

	"gorkbot/llm/testutil"
)

// consoleReplies collects replies from a test bot's replyFn.
type consoleReplies struct {
	mu      sync.Mutex
	replies []string
}

func (c *consoleReplies) add(response string) {
	c.mu.Lock()
	c.replies = append(c.replies, response)
	c.mu.Unlock()
}

func (c *consoleReplies) all() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.replies...)
}

// newConsoleTestBot builds a Bot wired to a fake LLM whose replies are
// collected by out.
func newConsoleTestBot(t *testing.T, fake *testutil.Client, out *consoleReplies) *Bot {
	t.Helper()
	b := newTestBot(t, fake, nil, nil)
	b.replyFn = func(_ context.Context, _ id.RoomID, _ id.EventID, response string) {
		out.add(response)
	}
	return b
}

func TestRunConsoleBasics(t *testing.T) {
	fake := &testutil.Client{Response: "yo"}
	out := &consoleReplies{}
	b := newConsoleTestBot(t, fake, out)

	// Blank lines are ignored; "exit" stops the loop before "more".
	b.RunConsole(context.Background(), strings.NewReader("hello\n\nexit\nmore\n"))

	if fake.Calls != 1 {
		t.Fatalf("expected 1 LLM call (hello; blank and exit are skipped), got %d", fake.Calls)
	}
	if got := out.all(); len(got) != 1 || got[0] != "yo" {
		t.Fatalf("expected 1 reply 'yo', got %v", got)
	}
	// History: user message + assistant reply in the console room.
	hm := b.historyMgr.GetOrCreate(id.RoomID(ConsoleRoom), b.maxHistory, b.historyTimeout)
	msgs := hm.Messages()
	if len(msgs) != 2 || msgs[0].Role != "user" || msgs[1].Role != "assistant" {
		t.Fatalf("expected user+assistant history, got %+v", msgs)
	}
	if !strings.HasPrefix(msgs[0].Content, "user: hello") {
		t.Errorf("expected user history starting 'user: hello', got %q", msgs[0].Content)
	}
}

func TestRunConsoleEveryLineIsAPrompt(t *testing.T) {
	// No mention gate: lines without @gork still get answered.
	fake := &testutil.Client{Response: "hey"}
	out := &consoleReplies{}
	b := newConsoleTestBot(t, fake, out)

	b.RunConsole(context.Background(), strings.NewReader("plain line, no mention\n"))

	if fake.Calls != 1 {
		t.Fatalf("expected 1 LLM call for un-mentioned line, got %d", fake.Calls)
	}
	if got := out.all(); len(got) != 1 || got[0] != "hey" {
		t.Fatalf("expected reply 'hey', got %v", got)
	}
}

func TestRunConsoleQuit(t *testing.T) {
	fake := &testutil.Client{Response: "yo"}
	out := &consoleReplies{}
	b := newConsoleTestBot(t, fake, out)

	b.RunConsole(context.Background(), strings.NewReader("first\nQUIT\nsecond\n"))

	if fake.Calls != 1 {
		t.Fatalf("expected 1 LLM call (QUIT exits the loop), got %d", fake.Calls)
	}
}

func TestRunConsoleMultiline(t *testing.T) {
	fake := &testutil.Client{Response: "long answer"}
	out := &consoleReplies{}
	b := newConsoleTestBot(t, fake, out)

	b.RunConsole(context.Background(), strings.NewReader("first part...\nsecond part\n"))

	if fake.Calls != 1 {
		t.Fatalf("expected 1 LLM call for joined multiline prompt, got %d", fake.Calls)
	}
	hm := b.historyMgr.GetOrCreate(id.RoomID(ConsoleRoom), b.maxHistory, b.historyTimeout)
	msgs := hm.Messages()
	if len(msgs) == 0 || !strings.Contains(msgs[0].Content, "first part\nsecond part") {
		t.Fatalf("expected joined prompt in history, got %+v", msgs)
	}
}

func TestRunConsoleHistoryPersistsAcrossTurns(t *testing.T) {
	fake := &testutil.Client{Response: "yo"}
	out := &consoleReplies{}
	b := newConsoleTestBot(t, fake, out)

	b.RunConsole(context.Background(), strings.NewReader("one\ntwo\nthree\n"))

	if fake.Calls != 3 {
		t.Fatalf("expected 3 LLM calls, got %d", fake.Calls)
	}
	if got := out.all(); len(got) != 3 {
		t.Fatalf("expected 3 replies, got %v", got)
	}
	hm := b.historyMgr.GetOrCreate(id.RoomID(ConsoleRoom), b.maxHistory, b.historyTimeout)
	msgs := hm.Messages()
	// 3 user turns + 3 assistant replies.
	if len(msgs) != 6 {
		t.Fatalf("expected 6 history messages, got %d: %+v", len(msgs), msgs)
	}
}

func TestRunConsoleEOF(t *testing.T) {
	fake := &testutil.Client{Response: "yo"}
	out := &consoleReplies{}
	b := newConsoleTestBot(t, fake, out)

	// No explicit exit; the loop must end on reader EOF.
	b.RunConsole(context.Background(), strings.NewReader("last line\n"))

	if fake.Calls != 1 {
		t.Fatalf("expected 1 LLM call before EOF, got %d", fake.Calls)
	}
	if got := out.all(); len(got) != 1 || got[0] != "yo" {
		t.Fatalf("expected reply 'yo', got %v", got)
	}
}
