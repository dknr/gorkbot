package main

import (
	"context"
	"errors"
	"testing"
	"time"

	"maunium.net/go/mautrix"
	"maunium.net/go/mautrix/event"
	"maunium.net/go/mautrix/id"

	"gorkbot/llm"
	"gorkbot/llm/testutil"
)

func nowMs() int64 { return time.Now().UnixMilli() }

// agoMs returns the millisecond timestamp for now minus d.
func agoMs(d time.Duration) int64 { return time.Now().Add(-d).UnixMilli() }

// newBackfillBot builds a Bot wired to a fake Matrix client and fake LLM.
func newBackfillBot(t *testing.T, fake *fakeMatrixClient) *Bot {
	t.Helper()
	b := newTestBot(t, &testutil.Client{}, nil, nil)
	b.backfillMax = 10
	b.backfillWindow = 24 * time.Hour
	b.joinedRoomsFn = fake.joinedRoomsFn()
	b.messagesFn = fake.messagesFn()
	b.downloadFn = fake.downloadFn(t)
	return b
}

func historyMessages(b *Bot, roomID id.RoomID) []llm.ChatMessage {
	return b.historyMgr.GetOrCreate(roomID, b.maxHistory, b.historyTimeout).Messages()
}

func TestBackfillSinglePage(t *testing.T) {
	// Two messages, newest first; both within the 24h window.
	pages := map[id.RoomID][]*mautrix.RespMessages{
		testRoom: {{
			Start: "t1",
			End:   "t0",
			Chunk: []*event.Event{
				textEvent(t, "@user:example.org", "second", agoMs(5*time.Minute)),
				textEvent(t, "@user:example.org", "first", agoMs(10*time.Minute)),
			},
		}},
	}
	fake := newFakeMatrixClient([]id.RoomID{testRoom}, pages)
	b := newBackfillBot(t, fake)

	b.Backfill(context.Background())

	msgs := historyMessages(b, testRoom)
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d: %+v", len(msgs), msgs)
	}
	// Chronological order after reversal.
	if msgs[0].Content != "@user:example.org: first" {
		t.Errorf("expected first message 'first', got %q", msgs[0].Content)
	}
	if msgs[1].Content != "@user:example.org: second" {
		t.Errorf("expected second message 'second', got %q", msgs[1].Content)
	}
	if msgs[0].Role != "user" {
		t.Errorf("expected role user, got %q", msgs[0].Role)
	}
}

func TestBackfillSelfIDAssistant(t *testing.T) {
	botID := id.UserID("@gork:example.org")
	evt := textEvent(t, botID, "my old reply", agoMs(time.Minute))
	m := b_chatMessage(t, evt, botID)
	if m.Role != "assistant" {
		t.Errorf("expected assistant role for bot's own message, got %q", m.Role)
	}
	if m.Content != "my old reply" {
		t.Errorf("expected content 'my old reply', got %q", m.Content)
	}
}

// b_chatMessage calls Bot.chatMessage directly for tests.
func b_chatMessage(t *testing.T, evt *event.Event, selfID id.UserID) llm.ChatMessage {
	t.Helper()
	b := newBackfillBot(t, newFakeMatrixClient(nil, nil))
	return b.chatMessage(context.Background(), evt, selfID)
}

func TestBackfillCutoffStopsPagination(t *testing.T) {
	old := agoMs(48 * time.Hour) // outside 24h window
	pages := map[id.RoomID][]*mautrix.RespMessages{
		testRoom: {
			{ // page 1 (newest first): one fresh, one stale
				Chunk: []*event.Event{
					textEvent(t, "@user:example.org", "fresh", agoMs(time.Minute)),
					textEvent(t, "@user:example.org", "stale", old),
				},
			},
			{ // page 2: should never be fetched
				Chunk: []*event.Event{
					textEvent(t, "@user:example.org", "older", agoMs(49*time.Hour)),
				},
			},
		},
	}
	fake := newFakeMatrixClient([]id.RoomID{testRoom}, pages)
	b := newBackfillBot(t, fake)

	b.Backfill(context.Background())

	msgs := historyMessages(b, testRoom)
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message (cutoff after stale event), got %d: %+v", len(msgs), msgs)
	}
	if msgs[0].Content != "@user:example.org: fresh" {
		t.Errorf("expected 'fresh', got %q", msgs[0].Content)
	}
	if fake.pageIdx[testRoom] != 1 {
		t.Errorf("expected only page 1 fetched, got %d pages", fake.pageIdx[testRoom])
	}
}

func TestBackfillMultiplePages(t *testing.T) {
	pages := map[id.RoomID][]*mautrix.RespMessages{
		testRoom: {
			{
				Start: "mid",
				Chunk: []*event.Event{
					textEvent(t, "@user:example.org", "newest", agoMs(time.Minute)),
					textEvent(t, "@user:example.org", "second", agoMs(2*time.Minute)),
				},
			},
			{
				Chunk: []*event.Event{
					textEvent(t, "@user:example.org", "oldest", agoMs(3*time.Minute)),
				},
			},
		},
	}
	fake := newFakeMatrixClient([]id.RoomID{testRoom}, pages)
	b := newBackfillBot(t, fake)

	b.Backfill(context.Background())

	msgs := historyMessages(b, testRoom)
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages, got %d: %+v", len(msgs), msgs)
	}
	if msgs[0].Content != "@user:example.org: oldest" ||
		msgs[1].Content != "@user:example.org: second" ||
		msgs[2].Content != "@user:example.org: newest" {
		t.Errorf("wrong chronological order: %+v", msgs)
	}
}

func TestBackfillMaxMessages(t *testing.T) {
	pages := map[id.RoomID][]*mautrix.RespMessages{
		testRoom: {{
			Chunk: []*event.Event{
				textEvent(t, "@user:example.org", "m3", nowMs()),
				textEvent(t, "@user:example.org", "m2", nowMs()-1),
				textEvent(t, "@user:example.org", "m1", nowMs()-2),
			},
		}},
	}
	fake := newFakeMatrixClient([]id.RoomID{testRoom}, pages)
	b := newBackfillBot(t, fake)
	b.backfillMax = 2

	b.Backfill(context.Background())

	msgs := historyMessages(b, testRoom)
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages (backfillMax), got %d: %+v", len(msgs), msgs)
	}
	if msgs[1].Content != "@user:example.org: m3" {
		t.Errorf("expected newest message 'm3' last, got %+v", msgs)
	}
}

func TestBackfillNonMessageEventsSkipped(t *testing.T) {
	nonMsg := memberEvent(t, "@user:example.org", agoMs(time.Minute))
	pages := map[id.RoomID][]*mautrix.RespMessages{
		testRoom: {{
			Chunk: []*event.Event{
				nonMsg,
				textEvent(t, "@user:example.org", "hello", agoMs(2*time.Minute)),
			},
		}},
	}
	fake := newFakeMatrixClient([]id.RoomID{testRoom}, pages)
	b := newBackfillBot(t, fake)

	b.Backfill(context.Background())

	msgs := historyMessages(b, testRoom)
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message (non-message skipped), got %d: %+v", len(msgs), msgs)
	}
	if msgs[0].Content != "@user:example.org: hello" {
		t.Errorf("expected 'hello', got %q", msgs[0].Content)
	}
}

func TestBackfillMessagesError(t *testing.T) {
	fake := newFakeMatrixClient([]id.RoomID{testRoom}, map[id.RoomID][]*mautrix.RespMessages{})
	b := newBackfillBot(t, fake)

	// Messages() returns an error for a room with no configured pages.
	b.Backfill(context.Background())

	msgs := historyMessages(b, testRoom)
	if len(msgs) != 0 {
		t.Errorf("expected no messages after fetch error, got %+v", msgs)
	}
}

func TestBackfillJoinedRoomsError(t *testing.T) {
	b := newTestBot(t, &testutil.Client{}, nil, nil)
	b.backfillMax = 10
	b.backfillWindow = 24 * time.Hour
	b.joinedRoomsFn = func(ctx context.Context) (*mautrix.RespJoinedRooms, error) {
		return nil, errors.New("rooms boom")
	}

	b.Backfill(context.Background())
	// Nothing to assert beyond not panicking; no rooms were backfilled.
}

func TestFeedBackfillReverses(t *testing.T) {
	b := newBackfillBot(t, newFakeMatrixClient(nil, nil))
	hm := b.historyMgr.GetOrCreate(testRoom, b.maxHistory, b.historyTimeout)
	collected := []llm.ChatMessage{
		{Role: "user", Content: "new"},
		{Role: "user", Content: "old"},
	}
	b.feedBackfill(testRoom, hm, collected)

	msgs := hm.Messages()
	if len(msgs) != 2 || msgs[0].Content != "old" || msgs[1].Content != "new" {
		t.Errorf("expected reversed order [old new], got %+v", msgs)
	}
}

func TestBackfillRoomEndStopsPagination(t *testing.T) {
	// End is set (end of history reached) with fewer messages than max.
	pages := map[id.RoomID][]*mautrix.RespMessages{
		testRoom: {{
			End:   "oldest-token",
			Chunk: []*event.Event{
				textEvent(t, "@user:example.org", "only", agoMs(time.Minute)),
			},
		}},
	}
	fake := newFakeMatrixClient([]id.RoomID{testRoom}, pages)
	b := newBackfillBot(t, fake)

	b.Backfill(context.Background())

	msgs := historyMessages(b, testRoom)
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d: %+v", len(msgs), msgs)
	}
	if fake.pageIdx[testRoom] != 1 {
		t.Errorf("expected pagination to stop after End, got %d pages fetched", fake.pageIdx[testRoom])
	}
}

func TestBackfillZeroTimestampNotCutOff(t *testing.T) {
	// Events with Timestamp == 0 must not trigger the cutoff (regression for
	// the `evt.Timestamp > 0` guard).
	pages := map[id.RoomID][]*mautrix.RespMessages{
		testRoom: {
			{
				Chunk: []*event.Event{
					textEvent(t, "@user:example.org", "zero-ts", 0),
					textEvent(t, "@user:example.org", "fresh", agoMs(time.Minute)),
				},
			},
		},
	}
	fake := newFakeMatrixClient([]id.RoomID{testRoom}, pages)
	b := newBackfillBot(t, fake)

	b.Backfill(context.Background())

	msgs := historyMessages(b, testRoom)
	if len(msgs) != 2 {
		t.Fatalf("expected both messages (zero-ts not cut off), got %d: %+v", len(msgs), msgs)
	}
}
