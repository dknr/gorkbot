package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"image"
	"image/color"
	"image/gif"
	"image/png"
	"testing"
	"time"

	"maunium.net/go/mautrix/event"
	"maunium.net/go/mautrix/id"

	"github.com/rs/zerolog"

	"gorkbot/llm"
	"gorkbot/llm/testutil"
)

// imageBot bundles a Bot wired to a fake vision client and fake downloads.
type imageBot struct {
	bot     *Bot
	vision  *testutil.Client
	replies chan string
}

// newImageBot builds an imageBot. downloads maps mxc URL string -> result.
func newImageBot(t *testing.T, downloads map[string]downloadResult) *imageBot {
	t.Helper()
	vision := &testutil.Client{Response: "a picture of a cave"}
	llmFake := &testutil.Client{Response: "llm reply"}
	repliesCh := make(chan string, 16)
	b := &Bot{
		log:            zerolog.Nop(),
		mention:        "@gork:example.org",
		maxHistory:     10,
		historyTimeout: time.Minute,
		historyMgr:     NewRoomHistoryManager(),
		completeFn:     llmFake.Complete,
		replyFn: func(_ context.Context, _ id.RoomID, _ id.EventID, response string) {
			select {
			case repliesCh <- response:
			default:
			}
		},
		stopCh: make(chan struct{}),
	}
	b.imagePrompt = "Describe what you see in this image"
	b.visionComplete = vision.CompleteMultimodal
	f := newFakeMatrixClient(nil, nil)
	f.downloads = downloads
	b.downloadFn = f.downloadFn(t)
	return &imageBot{bot: b, vision: vision, replies: repliesCh}
}

const fakeImg = "mxc://example.org/img123"

var fakeImageBytes = []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3}

// fakeGIFBytes encodes a 2x1 GIF: opaque blue, then a fully transparent
// palette entry. That transparency is what the GIF-to-PNG path exists to
// resolve, so the bytes are real rather than a magic-byte stub.
func fakeGIFBytes(t *testing.T) []byte {
	t.Helper()
	img := image.NewPaletted(image.Rect(0, 0, 2, 1), color.Palette{
		color.NRGBA{R: 0, G: 0, B: 255, A: 255},
		color.NRGBA{R: 255, G: 0, B: 0, A: 0},
	})
	img.SetColorIndex(0, 0, 0)
	img.SetColorIndex(1, 0, 1)
	var buf bytes.Buffer
	if err := gif.Encode(&buf, img, nil); err != nil {
		t.Fatalf("encode GIF: %v", err)
	}
	return buf.Bytes()
}

func TestPrepareImageGIFBecomesPNGOverWhite(t *testing.T) {
	out, ct, err := prepareImage(fakeGIFBytes(t))
	if err != nil {
		t.Fatalf("prepareImage(GIF) error: %v", err)
	}
	if ct != "image/png" {
		t.Errorf("expected GIF to be declared image/png, got %q", ct)
	}
	img, err := png.Decode(bytes.NewReader(out))
	if err != nil {
		t.Fatalf("decode re-encoded PNG: %v", err)
	}
	if b := img.Bounds(); b.Dx() != 2 || b.Dy() != 1 {
		t.Fatalf("expected a 2x1 image, got %dx%d", b.Dx(), b.Dy())
	}
	for _, tc := range []struct {
		x    int
		want [4]uint8
	}{
		{0, [4]uint8{0, 0, 255, 255}},
		// The transparent index must land on white, not on whatever RGB the
		// GIF happened to store under it.
		{1, [4]uint8{255, 255, 255, 255}},
	} {
		// Compared as components because the PNG decoder hands back whichever
		// concrete image type the header implies.
		r, g, b, a := img.At(tc.x, 0).RGBA()
		got := [4]uint8{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8), uint8(a >> 8)}
		if got != tc.want {
			t.Errorf("pixel %d: expected %+v, got %+v", tc.x, tc.want, got)
		}
	}
}

func TestHandleImageNoVision(t *testing.T) {
	b := newTestBot(t, &testutil.Client{}, nil, nil)
	// No vision model configured: image must be silently skipped.
	evt := imageEvent(t, "@user:example.org", fakeImg, nil, agoMs(time.Minute))

	b.handleImage(context.Background(), evt)

	msgs := b.historyMgr.GetOrCreate(testRoom, b.maxHistory, b.historyTimeout).Messages()
	if len(msgs) != 0 {
		t.Errorf("expected no history entries without vision, got %+v", msgs)
	}
}

func TestHandleImageHappyPath(t *testing.T) {
	encFile, ciphertext := encryptedFilePair(t, fakeImageBytes)
	ib := newImageBot(t, map[string]downloadResult{
		fakeImg: {body: ciphertext, contentType: "image/png"},
	})
	evt := imageEvent(t, "@user:example.org", fakeImg, encFile, agoMs(time.Minute))

	ib.bot.handleImage(context.Background(), evt)

	msgs := ib.bot.historyMgr.GetOrCreate(testRoom, ib.bot.maxHistory, ib.bot.historyTimeout).Messages()
	if len(msgs) != 1 {
		t.Fatalf("expected 1 history entry, got %d: %+v", len(msgs), msgs)
	}
	want := "@user:example.org: [📷 image] a picture of a cave"
	if msgs[0].Content != want {
		t.Errorf("expected %q, got %q", want, msgs[0].Content)
	}
	if ib.vision.VisionCalls != 1 {
		t.Errorf("expected 1 vision call, got %d", ib.vision.VisionCalls)
	}
	// The image bytes must have been base64-encoded and the content type used.
	if ib.vision.VisionImage != base64.StdEncoding.EncodeToString(fakeImageBytes) {
		t.Errorf("expected base64 image payload, got %q", ib.vision.VisionImage)
	}
	if ib.vision.VisionMime != "image/png" {
		t.Errorf("expected mime image/png, got %q", ib.vision.VisionMime)
	}
	if ib.vision.VisionPrompt != "Describe what you see in this image" {
		t.Errorf("unexpected vision prompt %q", ib.vision.VisionPrompt)
	}
}

func TestHandleImageMentionInDescription(t *testing.T) {
	encFile, ciphertext := encryptedFilePair(t, fakeImageBytes)
	ib := newImageBot(t, map[string]downloadResult{
		fakeImg: {body: ciphertext, contentType: "image/jpeg"},
	})
	// Description contains the mention, so the image should be queued to the
	// worker as a mention.
	ib.vision.Response = "a picture of @gork:example.org painting"
	evt := imageEvent(t, "@user:example.org", fakeImg, encFile, agoMs(time.Minute))

	// The image path queues a mention message; drain it and handle it
	// synchronously for a deterministic reply assertion. A dedicated fake
	// client backs the LLM call so the canned reply is unambiguous.
	llmFake := &testutil.Client{Response: "a picture of @gork:example.org painting"}
	ib.bot.completeFn = llmFake.Complete
	ib.bot.workerCh = make(chan message, workerQueueSize)
	ib.bot.handleImage(context.Background(), evt)

	var msg message
	select {
	case msg = <-ib.bot.workerCh:
	default:
		t.Fatal("expected a queued image mention message")
	}
	ib.bot.handle(msg)

	select {
	case reply := <-ib.replies:
		want := "a picture of @gork:example.org painting"
		if reply != want {
			t.Errorf("expected reply %q, got %q", want, reply)
		}
	default:
		t.Fatal("no reply sent after handling queued message")
	}
}

func TestHandleImageMissingURL(t *testing.T) {
	ib := newImageBot(t, nil)
	evt := msgEvent(t, "@user:example.org", event.EventMessage, map[string]any{"msgtype": "m.image", "body": "photo.jpg"}, agoMs(time.Minute))

	ib.bot.handleImage(context.Background(), evt)

	if ib.vision.VisionCalls != 0 {
		t.Errorf("expected no vision call without media URL, got %d", ib.vision.VisionCalls)
	}
}

func TestHandleImageDownloadError(t *testing.T) {
	encFile, _ := encryptedFilePair(t, fakeImageBytes)
	ib := newImageBot(t, map[string]downloadResult{
		fakeImg: {err: errors.New("download boom")},
	})
	evt := imageEvent(t, "@user:example.org", fakeImg, encFile, agoMs(time.Minute))

	ib.bot.handleImage(context.Background(), evt)

	if ib.vision.VisionCalls != 0 {
		t.Errorf("expected no vision call after download error, got %d", ib.vision.VisionCalls)
	}
	msgs := ib.bot.historyMgr.GetOrCreate(testRoom, ib.bot.maxHistory, ib.bot.historyTimeout).Messages()
	if len(msgs) != 0 {
		t.Errorf("expected no history entry after download error, got %+v", msgs)
	}
}

func TestHandleImageDecryptError(t *testing.T) {
	// Ciphertext does not match the file's hash -> Decrypt fails.
	encFile, _ := encryptedFilePair(t, fakeImageBytes)
	ib := newImageBot(t, map[string]downloadResult{
		fakeImg: {body: []byte("not the ciphertext"), contentType: "image/png"},
	})
	evt := imageEvent(t, "@user:example.org", fakeImg, encFile, agoMs(time.Minute))

	ib.bot.handleImage(context.Background(), evt)

	if ib.vision.VisionCalls != 0 {
		t.Errorf("expected no vision call after decrypt error, got %d", ib.vision.VisionCalls)
	}
}

func TestHandleImageVisionError(t *testing.T) {
	encFile, ciphertext := encryptedFilePair(t, fakeImageBytes)
	ib := newImageBot(t, map[string]downloadResult{
		fakeImg: {body: ciphertext, contentType: "image/png"},
	})
	ib.vision.Err = testutil.ErrFake
	evt := imageEvent(t, "@user:example.org", fakeImg, encFile, agoMs(time.Minute))

	ib.bot.handleImage(context.Background(), evt)

	msgs := ib.bot.historyMgr.GetOrCreate(testRoom, ib.bot.maxHistory, ib.bot.historyTimeout).Messages()
	if len(msgs) != 0 {
		t.Errorf("expected no history entry after vision error, got %+v", msgs)
	}
}

func TestDescribeImage(t *testing.T) {
	ib := newImageBot(t, nil)

	// The media type is sniffed from the bytes rather than trusted from the
	// HTTP header.
	desc, err := ib.bot.describeImage(context.Background(), fakeImageBytes)
	if err != nil {
		t.Fatalf("describeImage error: %v", err)
	}
	if desc != "a picture of a cave" {
		t.Errorf("unexpected description %q", desc)
	}
	if ib.vision.VisionMime != "image/png" {
		t.Errorf("expected mime image/png, got %q", ib.vision.VisionMime)
	}

	// A GIF reaches the model as PNG.
	if _, err := ib.bot.describeImage(context.Background(), fakeGIFBytes(t)); err != nil {
		t.Fatalf("describeImage(GIF) error: %v", err)
	}
	if ib.vision.VisionMime != "image/png" {
		t.Errorf("expected GIF to arrive as image/png, got %q", ib.vision.VisionMime)
	}

	// Error propagation.
	ib.vision.Err = testutil.ErrFake
	if _, err := ib.bot.describeImage(context.Background(), fakeImageBytes); err != testutil.ErrFake {
		t.Errorf("expected ErrFake, got: %v", err)
	}
}

func TestChatMessageImageFallbackToBody(t *testing.T) {
	ib := newImageBot(t, map[string]downloadResult{
		fakeImg: {err: errors.New("download boom")},
	})
	// Image download fails: chatMessage falls back to the body text.
	evt := imageEvent(t, "@user:example.org", fakeImg, nil, agoMs(time.Minute))
	m := ib.bot.chatMessage(context.Background(), evt, "@gork:example.org")
	want := "@user:example.org: photo.jpg"
	if m.Content != want {
		t.Errorf("expected fallback to body %q, got %q", want, m.Content)
	}
	if m.Role != "user" {
		t.Errorf("expected user role, got %q", m.Role)
	}
}

func TestChatMessageImageWithDescription(t *testing.T) {
	encFile, ciphertext := encryptedFilePair(t, fakeImageBytes)
	ib := newImageBot(t, map[string]downloadResult{
		fakeImg: {body: ciphertext, contentType: "image/png"},
	})
	evt := imageEvent(t, "@user:example.org", fakeImg, encFile, agoMs(time.Minute))
	m := ib.bot.chatMessage(context.Background(), evt, "@gork:example.org")
	want := "@user:example.org: [📷 image] a picture of a cave"
	if m.Content != want {
		t.Errorf("expected %q, got %q", want, m.Content)
	}
}

func TestChatMessageSelfID(t *testing.T) {
	b := newBackfillBot(t, newFakeMatrixClient(nil, nil))
	evt := textEvent(t, "@gork:example.org", "my reply", agoMs(time.Minute))
	m := b.chatMessage(context.Background(), evt, "@gork:example.org")
	if m.Role != "assistant" || m.Content != "my reply" {
		t.Errorf("expected assistant/'my reply', got %+v", m)
	}
}

func TestChatMessagePlainText(t *testing.T) {
	b := newBackfillBot(t, newFakeMatrixClient(nil, nil))
	evt := textEvent(t, "@user:example.org", "hi there", agoMs(time.Minute))
	m := b.chatMessage(context.Background(), evt, "@gork:example.org")
	if m.Role != "user" || m.Content != "@user:example.org: hi there" {
		t.Errorf("unexpected chat message: %+v", m)
	}
}

var _ llm.ChatMessage
