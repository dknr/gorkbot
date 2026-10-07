package main

import (
	"context"
	"testing"
	"time"

	"maunium.net/go/mautrix/event"
	"maunium.net/go/mautrix/id"

	"github.com/rs/zerolog"

	"gorkbot/llm"
	"gorkbot/llm/testutil"
)

const mentionHTML = `<a href="https://matrix.to/#/@gork:example.org">gork</a>: what is up`

func testLogger() zerolog.Logger { return zerolog.Nop() }

// runWorker starts the worker goroutine on the bot's current workerCh (which
// must be initialized) and registers cleanup that stops it when the test ends.
func runWorker(b *Bot, t *testing.T) {
	t.Helper()
	ch := b.workerCh
	done := make(chan struct{})
	go func() {
		b.worker(ch)
		close(done)
	}()
	t.Cleanup(func() {
		b.Stop()
		<-done
	})
}

// captureLLM is a completeFn that records the history it is given and returns
// a fixed reply. Lets tests assert on what the worker fed to the LLM.
type captureLLM struct {
	histories [][]llm.ChatMessage
}

func (c *captureLLM) complete(ctx context.Context, history []llm.ChatMessage) (string, int, int, int, error) {
	c.histories = append(c.histories, history)
	return "grug reply", 0, 0, 0, nil
}

// newHandlerBot builds a Bot whose worker (started by Handler) uses a
// captureLLM, so tests can assert on what got fed to the LLM.
func newHandlerBot(t *testing.T) (*Bot, *captureLLM) {
	t.Helper()
	cap := &captureLLM{}
	b := &Bot{
		log:            testLogger(),
		mention:        "@gork:example.org",
		maxHistory:     10,
		historyTimeout: time.Minute,
		historyMgr:     NewRoomHistoryManager(),
		completeFn:     cap.complete,
		replyFn:        func(context.Context, id.RoomID, id.EventID, string) {},
		stopCh:         make(chan struct{}),
	}
	t.Cleanup(b.Stop)
	return b, cap
}

// waitForHistory polls until the room has at least n history entries.
func waitForHistory(t *testing.T, b *Bot, room id.RoomID, n int) []llm.ChatMessage {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		msgs := b.historyMgr.GetOrCreate(room, b.maxHistory, b.historyTimeout).Messages()
		if len(msgs) >= n {
			return msgs
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %d history entries, have %d: %+v", n, len(msgs), msgs)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestHandlerPlainMentionTriggersLLM(t *testing.T) {
	b, cap := newHandlerBot(t)
	h := b.Handler()

	h(context.Background(), textEvent(t, "@user:example.org", "@gork:example.org hello grug", agoMs(time.Second)))

	// Mention -> worker calls the LLM with a single user turn.
	waitForHistory(t, b, testRoom, 2)
	if len(cap.histories) != 1 {
		t.Fatalf("expected 1 LLM call, got %d", len(cap.histories))
	}
	turns := cap.histories[0]
	if len(turns) != 1 || turns[0].Role != "user" {
		t.Fatalf("expected 1 user turn, got %+v", turns)
	}
	if turns[0].Content != "@user:example.org: @gork:example.org hello grug" {
		t.Errorf("unexpected user turn %q", turns[0].Content)
	}
}

func TestHandlerNoMentionNoLLM(t *testing.T) {
	b, cap := newHandlerBot(t)
	h := b.Handler()

	h(context.Background(), textEvent(t, "@user:example.org", "no mention here", agoMs(time.Second)))

	// Non-mention -> added to history but no LLM call.
	waitForHistory(t, b, testRoom, 1)
	// Give any (incorrect) LLM call a moment to land.
	time.Sleep(50 * time.Millisecond)
	if len(cap.histories) != 0 {
		t.Fatalf("expected no LLM call for non-mention, got %d", len(cap.histories))
	}
}

func TestHandlerEmptyBodyIgnored(t *testing.T) {
	b, _ := newHandlerBot(t)
	h := b.Handler()

	// Empty body and no formatted body: the event is ignored entirely.
	empty := msgEvent(t, "@user:example.org", event.EventMessage, map[string]any{"msgtype": "m.text"}, agoMs(time.Second))
	h(context.Background(), empty)

	time.Sleep(50 * time.Millisecond)
	if msgs := b.historyMgr.GetOrCreate(testRoom, b.maxHistory, b.historyTimeout).Messages(); len(msgs) != 0 {
		t.Fatalf("expected no history for empty body, got %+v", msgs)
	}
}

func TestHandlerHTMLMentionTriggersLLM(t *testing.T) {
	b, cap := newHandlerBot(t)
	h := b.Handler()

	// The plain body has no mention; only the HTML does.
	h(context.Background(), htmlEvent(t, "@user:example.org", "what is up", mentionHTML, agoMs(time.Second)))

	waitForHistory(t, b, testRoom, 2)
	if len(cap.histories) != 1 {
		t.Fatalf("expected 1 LLM call for HTML mention, got %d", len(cap.histories))
	}
}

func TestHandlerQueueFullDrops(t *testing.T) {
	// Fill the worker queue to capacity with no consumer; the handler's
	// select/send must take the drop (default) branch rather than blocking.
	b := newTestBot(t, &testutil.Client{}, nil, nil)
	h := b.Handler()
	// Stop the worker so it stops draining, then fill the queue.
	b.Stop()
	for i := 0; i < workerQueueSize; i++ {
		b.workerCh <- message{}
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		h(context.Background(), textEvent(t, "@user:example.org", "@gork:example.org dropped", agoMs(time.Second)))
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("handler blocked on a full queue instead of dropping")
	}
}

func TestHandlerImageRoutedToHistory(t *testing.T) {
	encFile, ciphertext := encryptedFilePair(t, fakeImageBytes)
	ib := newImageBot(t, map[string]downloadResult{
		fakeImg: {body: ciphertext, contentType: "image/png"},
	})
	b := ib.bot
	h := b.Handler()

	// Image without a mention in the description: goes to history only.
	evt := imageEvent(t, "@user:example.org", fakeImg, encFile, agoMs(time.Second))
	h(context.Background(), evt)

	waitForHistory(t, b, testRoom, 1)
}

func TestWorkerProcessesUntilStopped(t *testing.T) {
	b := newTestBot(t, &testutil.Client{Response: "reply one"}, nil, nil)
	b.workerCh = make(chan message, workerQueueSize)
	runWorker(b, t)

	b.workerCh <- message{ctx: context.Background(), roomID: testRoom, sender: "@u:x", body: "one", mentioned: true, prompt: "one"}
	waitForHistory(t, b, testRoom, 2)
}

func TestStopClosesWorker(t *testing.T) {
	b := newTestBot(t, &testutil.Client{}, nil, nil)
	b.workerCh = make(chan message, 1)
	done := make(chan struct{})
	go func() {
		b.worker(b.workerCh)
		close(done)
	}()
	b.Stop()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("worker did not exit after Stop")
	}
}

func TestNewBot(t *testing.T) {
	cfg := &Config{}
	cfg.Gork.Mention = "@gork:example.org"
	cfg.LLM.MaxHistory = 7
	cfg.LLM.historyTimeout = 12 * time.Minute
	cfg.Backfill.MaxMessages = 3
	cfg.Backfill.timeout = 24 * time.Hour
	cfg.LLMVision.ImagePrompt = "describe"

	b := NewBot(
		testLogger(),
		cfg,
		nil, // llm client unused by this test
		nil, // vision client unused
		nil, // mautrix client unused
		func(context.Context, id.RoomID, id.EventID, string) {},
		func(context.Context, id.RoomID, bool) {},
	)

	if b.mention != "@gork:example.org" {
		t.Errorf("unexpected mention %q", b.mention)
	}
	if b.maxHistory != 7 {
		t.Errorf("unexpected maxHistory %d", b.maxHistory)
	}
	if b.historyTimeout != 12*time.Minute {
		t.Errorf("unexpected historyTimeout %v", b.historyTimeout)
	}
	if b.backfillMax != 3 || b.backfillWindow != 24*time.Hour {
		t.Errorf("unexpected backfill config %d/%v", b.backfillMax, b.backfillWindow)
	}
	if b.imagePrompt != "describe" {
		t.Errorf("unexpected imagePrompt %q", b.imagePrompt)
	}
	if b.replyFn == nil || b.typingFn == nil || b.historyMgr == nil || b.stopCh == nil {
		t.Error("expected non-nil callbacks/fields from NewBot")
	}
}
