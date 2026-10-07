package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"testing"
	"time"

	"maunium.net/go/mautrix"
	"maunium.net/go/mautrix/crypto/attachment"
	"maunium.net/go/mautrix/event"
	"maunium.net/go/mautrix/id"
)

const testRoom = "!room:example.org"

// msgEvent builds a Matrix event with the given type and raw content map.
// Content is parsed via ParseRaw, the same way mautrix parses synced events,
// so typed accessors like AsMessage() work.
func msgEvent(t *testing.T, sender id.UserID, eventType event.Type, content map[string]any, ts int64) *event.Event {
	t.Helper()
	b, err := json.Marshal(content)
	if err != nil {
		t.Fatal(err)
	}
	evt := &event.Event{
		Sender:    sender,
		Type:      eventType,
		Timestamp: ts,
		ID:        id.EventID("$evt" + fmt.Sprint(ts) + sender.String()),
		RoomID:    testRoom,
		Content:   event.Content{VeryRaw: b},
	}
	if err := evt.Content.ParseRaw(eventType); err != nil {
		t.Fatalf("failed to parse event content for %s: %v", eventType.Type, err)
	}
	return evt
}

// textEvent builds a plain m.text message.
func textEvent(t *testing.T, sender id.UserID, body string, ts int64) *event.Event {
	return msgEvent(t, sender, event.EventMessage, map[string]any{"msgtype": "m.text", "body": body}, ts)
}

// htmlEvent builds a message with a plain body and HTML formatted body.
func htmlEvent(t *testing.T, sender id.UserID, body, formatted string, ts int64) *event.Event {
	return msgEvent(t, sender, event.EventMessage, map[string]any{
		"msgtype":        "m.text",
		"body":           body,
		"format":         "org.matrix.custom.html",
		"formatted_body": formatted,
	}, ts)
}

// memberEvent builds an m.room.member state event (a non-message event).
func memberEvent(t *testing.T, sender id.UserID, ts int64) *event.Event {
	return msgEvent(t, sender, event.NewEventType("m.room.member"), map[string]any{"membership": "join"}, ts)
}

// imageEvent builds an m.image message pointing at mxcURL, optionally with an
// encrypted file attachment.
func imageEvent(t *testing.T, sender id.UserID, mxcURL string, file *event.EncryptedFileInfo, ts int64) *event.Event {
	content := map[string]any{"msgtype": "m.image", "body": "photo.jpg", "url": mxcURL}
	if file != nil {
		content["file"] = file
	}
	return msgEvent(t, sender, event.EventMessage, content, ts)
}

// encryptedFilePair encrypts plaintext with a fresh A256CTR key and returns
// the EncryptedFileInfo (for the event content) and the ciphertext (for the
// media download response).
func encryptedFilePair(t *testing.T, plaintext []byte) (*event.EncryptedFileInfo, []byte) {
	t.Helper()
	ef := attachment.NewEncryptedFile()
	data := make([]byte, len(plaintext))
	copy(data, plaintext)
	ef.EncryptInPlace(data)
	return &event.EncryptedFileInfo{EncryptedFile: *ef}, data
}

// downloadResponse returns an *http.Response carrying the given body.
func downloadResponse(t *testing.T, body []byte, contentType string) *http.Response {
	t.Helper()
	return &http.Response{
		Body:          io.NopCloser(bytes.NewReader(body)),
		Header:        http.Header{"Content-Type": []string{contentType}},
		ContentLength: int64(len(body)),
	}
}

// downloadResult is the canned outcome of a media download in tests.
type downloadResult struct {
	body        []byte
	contentType string
	err         error
}

// fakeMatrixClient is a test double with canned rooms, message pages
// (newest-first, per room) and download results keyed by mxc URL string.
type fakeMatrixClient struct {
	rooms     []id.RoomID
	pages     map[id.RoomID][]*mautrix.RespMessages
	pageIdx   map[id.RoomID]int
	downloads map[string]downloadResult
}

func newFakeMatrixClient(rooms []id.RoomID, pages map[id.RoomID][]*mautrix.RespMessages) *fakeMatrixClient {
	return &fakeMatrixClient{
		rooms:   rooms,
		pages:   pages,
		pageIdx: map[id.RoomID]int{},
	}
}

// joinedRoomsFn returns a seam function for Bot.joinedRoomsFn.
func (f *fakeMatrixClient) joinedRoomsFn() func(ctx context.Context) (*mautrix.RespJoinedRooms, error) {
	return func(_ context.Context) (*mautrix.RespJoinedRooms, error) {
		return &mautrix.RespJoinedRooms{JoinedRooms: f.rooms}, nil
	}
}

// messagesFn returns a seam function for Bot.messagesFn.
func (f *fakeMatrixClient) messagesFn() func(ctx context.Context, roomID id.RoomID, from string, limit int) (*mautrix.RespMessages, error) {
	return func(_ context.Context, roomID id.RoomID, _ string, _ int) (*mautrix.RespMessages, error) {
		pages := f.pages[roomID]
		if len(pages) == 0 {
			return nil, fmt.Errorf("no pages configured for %s", roomID)
		}
		idx := f.pageIdx[roomID]
		if idx >= len(pages) {
			return nil, fmt.Errorf("no more pages configured for %s", roomID)
		}
		f.pageIdx[roomID] = idx + 1
		return pages[idx], nil
	}
}

// downloadFn returns a seam function for Bot.downloadFn.
func (f *fakeMatrixClient) downloadFn(t *testing.T) func(ctx context.Context, mxcURL id.ContentURI) (*http.Response, error) {
	return func(_ context.Context, mxcURL id.ContentURI) (*http.Response, error) {
		d, ok := f.downloads[mxcURL.String()]
		if d.err != nil {
			return nil, d.err
		}
		if !ok {
			return nil, fmt.Errorf("no download configured for %s", mxcURL)
		}
		return downloadResponse(t, d.body, d.contentType), nil
	}
}

var _ = time.Now
