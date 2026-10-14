package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"maunium.net/go/mautrix"
	"maunium.net/go/mautrix/crypto/attachment"
	"maunium.net/go/mautrix/event"
	"maunium.net/go/mautrix/id"

	"gorkbot/llm"
)

// testPNG is a valid 3x2 PNG so pngSize() can read real dimensions out of it.
var testPNG = func() []byte {
	img := image.NewRGBA(image.Rect(0, 0, 3, 2))
	for y := 0; y < 2; y++ {
		for x := 0; x < 3; x++ {
			img.Set(x, y, color.RGBA{R: uint8(x * 80), G: 10, B: 200, A: 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		panic(err)
	}
	return buf.Bytes()
}()

// toolCallMsg builds an assistant turn containing one tool call.
func toolCallMsg(id, name, args string) llm.ToolMessage {
	return llm.ToolMessage{
		FinishReason: "tool_calls",
		ToolCalls: []llm.ToolCall{{
			ID:       id,
			Type:     "function",
			Function: llm.FunctionCall{Name: name, Arguments: args},
		}},
	}
}

// textMsg builds an assistant turn with no tool calls.
func textMsg(content string) llm.ToolMessage {
	return llm.ToolMessage{FinishReason: "stop", Content: content}
}

// drawBot is a Bot wired with fakes for the LLM tool path, the diffusion
// server, the vision model, the media upload and the image send.
type drawBot struct {
	bot       *Bot
	hm        *HistoryManager
	send      *[]string // captions of images that reached the room
	drawn     *[]string // prompts sent to the diffusion server
	uploads   *int
	turns     *[]toolTurn
	encrypted *[]bool
}

// toolTurn records what the model was offered on each call, so tests can
// assert the tools get withdrawn once the draw budget is spent.
type toolTurn struct {
	toolsOffered int
	toolChoice   any
}

func newDrawBot(t *testing.T, replies []llm.ToolMessage, captions []string, drawErr error) *drawBot {
	t.Helper()
	var mu sync.Mutex
	var turnList []toolTurn
	var drawList []string
	var sendList []string
	uploads := 0
	var uploadEncrypted []bool
	// visionErr is nil; captions are consumed one per generated image.
	captionIdx := 0

	b := &Bot{
		log:            testLogger(),
		mention:        "@gork:example.org",
		maxHistory:     50,
		historyTimeout: time.Minute,
		historyMgr:     NewRoomHistoryManager(),
		image:          ImageConfig{maxRetries: 2, timeout: time.Minute},
		sd:             &sdClient{BaseURL: "http://sd.invalid"},
	}

	b.completeTools = func(_ context.Context, history []llm.ChatMessageTool, tools []llm.Tool, choice any, _ float64, _ int) (llm.ToolMessage, llm.UsageInfo, error) {
		mu.Lock()
		turnList = append(turnList, toolTurn{toolsOffered: len(tools), toolChoice: choice})
		var reply llm.ToolMessage
		if len(replies) > 0 {
			reply, replies = replies[0], replies[1:]
		} else {
			reply = textMsg("(script exhausted)")
		}
		mu.Unlock()
		// A withdrawn-tool turn must also carry tool_choice "none"; that pairing
		// is what makes the model stop calling.
		if len(tools) == 0 && choice != "none" {
			return llm.ToolMessage{}, llm.UsageInfo{}, errors.New("tools withdrawn without tool_choice=none")
		}
		return reply, llm.UsageInfo{}, nil
	}

	b.drawImage = func(_ context.Context, prompt string) ([]byte, error) {
		mu.Lock()
		defer mu.Unlock()
		drawList = append(drawList, prompt)
		if drawErr != nil {
			return nil, drawErr
		}
		return testPNG, nil
	}

	b.visionComplete = func(_ context.Context, _, _, _ string) (string, int, int, int, error) {
		mu.Lock()
		defer mu.Unlock()
		caption := "a red cube on a hill"
		if captionIdx < len(captions) {
			caption = captions[captionIdx]
		}
		captionIdx++
		return caption, 0, 0, 0, nil
	}
	b.imagePrompt = "Describe what you see in this image"

	b.uploadFn = func(_ context.Context, data []byte, _ string, encrypt bool) (id.ContentURI, *event.EncryptedFileInfo, error) {
		mu.Lock()
		defer mu.Unlock()
		uploadEncrypted = append(uploadEncrypted, encrypt)
		if !bytes.Equal(data, testPNG) {
			return id.ContentURI{}, nil, errors.New("uploaded bytes are not the generated image")
		}
		uploads++
		uri := mustContentURI(t, "mxc://example.org/abc")
		if encrypt {
			// Mirrors the real path: the mxc moves into the file object and the
			// plain url comes back empty.
			return id.ContentURI{}, &event.EncryptedFileInfo{
				EncryptedFile: *attachment.NewEncryptedFile(),
				URL:           id.ContentURIString(uri.String()),
			}, nil
		}
		return uri, nil, nil
	}

	b.sendImageFn = func(_ context.Context, room id.RoomID, uri id.ContentURI, file *event.EncryptedFileInfo, caption string, pngBytes []byte) error {
		mu.Lock()
		defer mu.Unlock()
		if room != testRoom {
			return errors.New("image sent to wrong room")
		}
		if uri.String() == "" && (file == nil || file.URL == "") {
			return errors.New("image sent without a media URI")
		}
		if uri.String() != "" && file != nil {
			return errors.New("image sent with both a plain url and an encrypted file")
		}
		if len(pngBytes) == 0 {
			return errors.New("image sent without image bytes")
		}
		sendList = append(sendList, caption)
		return nil
	}

	hm := b.historyMgr.GetOrCreate(testRoom, 50, time.Minute)
	d := &drawBot{bot: b, hm: hm, send: &sendList, drawn: &drawList, uploads: &uploads, turns: &turnList}
	d.encrypted = &uploadEncrypted
	return d
}

func drawMessage() message {
	return message{
		ctx:       context.Background(),
		roomID:    testRoom,
		sender:    "@ada:example.org",
		body:      "@gork:example.org draw me a cube",
		mentioned: true,
		prompt:    "draw me a cube",
		eventID:   "$evt1",
	}
}

func (d *drawBot) run(t *testing.T) imageOutcome {
	t.Helper()
	msg := drawMessage()
	// handle() seeds history with the user turn before the loop runs, and the
	// model template refuses tools without one, so the harness does the same.
	d.hm.AddMessage(llm.ChatMessage{Role: "user", Content: msg.sender + ": " + msg.body})
	out, ok := d.bot.runToolLoop(context.Background(), msg, d.hm, testLogger())
	if !ok {
		t.Fatal("tool loop reported failure")
	}
	return out
}

// historyText joins all history content so tests can grep for captions.
func (d *drawBot) historyText() string {
	var out string
	for _, m := range d.hm.Messages() {
		out += m.Role + "|" + m.Content + "\n"
	}
	return out
}

func TestToolLoopDrawThenAccept(t *testing.T) {
	d := newDrawBot(t, []llm.ToolMessage{
		toolCallMsg("c1", toolDrawImage, `{"prompt":"a red cube"}`),
		textMsg("a cube, as commanded"),
	}, []string{"a red cube on a hill"}, nil)

	out := d.run(t)

	if out.Drawn != 1 || !out.Sent {
		t.Fatalf("drawn=%d sent=%v, want 1/true", out.Drawn, out.Sent)
	}
	if out.ReplyText != "a cube, as commanded" {
		t.Fatalf("reply = %q", out.ReplyText)
	}
	if len(*d.drawn) != 1 || (*d.drawn)[0] != "a red cube" {
		t.Fatalf("draw prompts = %v", *d.drawn)
	}
	if len(*d.send) != 1 {
		t.Fatalf("images sent = %d, want 1", len(*d.send))
	}
	if want := toolHistoryPrefix + "a red cube on a hill"; !contains(d.historyText(), want) {
		t.Fatalf("history missing vision caption %q:\n%s", want, d.historyText())
	}
}

func TestToolLoopAbstainsWithAnswerDirectly(t *testing.T) {
	d := newDrawBot(t, []llm.ToolMessage{
		toolCallMsg("c1", toolAnswerDirect, `{"answer":"why did the bicycle fall"}`),
	}, nil, nil)

	out := d.run(t)

	if out.Drawn != 0 || out.Sent {
		t.Fatalf("drawn=%d sent=%v, want 0/false", out.Drawn, out.Sent)
	}
	if out.ReplyText != "why did the bicycle fall" {
		t.Fatalf("reply = %q", out.ReplyText)
	}
}

// The retry budget is the whole point: two redraws, then the result must go
// out whether or not the model is happy with it.
func TestToolLoopSendsAfterSecondRetryEvenIfModelNeverAccepts(t *testing.T) {
	draw := toolCallMsg("c1", toolDrawImage, `{"prompt":"cube"}`)
	d := newDrawBot(t, []llm.ToolMessage{
		draw, draw, draw, draw, draw, draw, // never accepts
	}, []string{"attempt one", "attempt two", "attempt three"}, nil)

	out := d.run(t)

	if out.Drawn != 3 {
		t.Fatalf("drawn=%d, want 3 (1 draw + 2 retries)", out.Drawn)
	}
	if !out.Sent {
		t.Fatal("image must be sent once the retry budget is spent")
	}
	if len(*d.send) != 1 {
		t.Fatalf("images sent = %d, want exactly 1", len(*d.send))
	}
	// Every attempt's description stays in history, sent or not.
	hist := d.historyText()
	for _, want := range []string{"attempt one", "attempt two", "attempt three"} {
		if !contains(hist, toolHistoryPrefix+want) {
			t.Fatalf("history missing caption %q:\n%s", want, hist)
		}
	}
	// The turn after the budget is spent must have tools withdrawn.
	turns := *d.turns
	if len(turns) != 4 {
		t.Fatalf("model turns = %d, want 4", len(turns))
	}
	for i := 0; i < 3; i++ {
		if turns[i].toolsOffered != 2 {
			t.Fatalf("turn %d offered %d tools, want 2", i, turns[i].toolsOffered)
		}
	}
	if turns[3].toolsOffered != 0 || turns[3].toolChoice != "none" {
		t.Fatalf("final turn offered %d tools with choice %v, want 0 and none",
			turns[3].toolsOffered, turns[3].toolChoice)
	}
}

func TestToolLoopStopsDrawingWhenModelRedrawsLessThanBudget(t *testing.T) {
	d := newDrawBot(t, []llm.ToolMessage{
		toolCallMsg("c1", toolDrawImage, `{"prompt":"cube"}`),
		toolCallMsg("c2", toolDrawImage, `{"prompt":"better cube"}`),
		textMsg("two cubes is enough"),
	}, []string{"bad cube", "good cube"}, nil)

	out := d.run(t)

	if out.Drawn != 2 || !out.Sent {
		t.Fatalf("drawn=%d sent=%v, want 2/true", out.Drawn, out.Sent)
	}
	if len(*d.send) != 1 {
		t.Fatalf("images sent = %d, want 1 (only the accepted one)", len(*d.send))
	}
	if got := (*d.drawn)[1]; got != "better cube" {
		t.Fatalf("retry prompt = %q", got)
	}
}

func TestToolLoopDrawFailureDoesNotSend(t *testing.T) {
	d := newDrawBot(t, []llm.ToolMessage{
		toolCallMsg("c1", toolDrawImage, `{"prompt":"cube"}`),
		textMsg("the forge was cold"),
	}, nil, errors.New("sd exploded"))

	out := d.run(t)

	if out.Sent || *d.uploads != 0 {
		t.Fatalf("sent=%v uploads=%d, want nothing sent", out.Sent, *d.uploads)
	}
	if out.ReplyText != "the forge was cold" {
		t.Fatalf("reply = %q", out.ReplyText)
	}
}

func TestToolLoopUnknownToolIsIgnored(t *testing.T) {
	d := newDrawBot(t, []llm.ToolMessage{
		toolCallMsg("c1", "launch_missiles", `{}`),
		textMsg("no such tool"),
	}, nil, nil)

	out := d.run(t)
	if out.Drawn != 0 || out.Sent {
		t.Fatalf("drawn=%d sent=%v, want 0/false", out.Drawn, out.Sent)
	}
}

func TestSDClientGenerateDecodesPNG(t *testing.T) {
	srv := newSDTestServer(t, testPNG)
	defer srv.Close()

	sd := &sdClient{BaseURL: srv.URL, Width: 512, Height: 512, Steps: 12, CFGScale: 7, Timeout: time.Minute}
	data, err := sd.generate(context.Background(), "a red cube", "")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, testPNG) {
		t.Fatal("returned bytes do not match the served image")
	}
	if w, h, err := pngSize(data); err != nil || w != 3 || h != 2 {
		t.Fatalf("pngSize = %d,%d,%v", w, h, err)
	}
}

func TestSDClientGenerateRejectsEmptyPrompt(t *testing.T) {
	sd := &sdClient{BaseURL: "http://127.0.0.1:1"}
	if _, err := sd.generate(context.Background(), "   ", ""); err == nil {
		t.Fatal("expected an error for an empty prompt")
	}
}

func TestImageConfigDefaults(t *testing.T) {
	cfg := loadTestConfig(t, `{}`)
	if cfg.Image.enabled() {
		t.Fatal("image generation must stay off when base_url is empty")
	}
	if cfg.Image.maxRetries != 2 {
		t.Fatalf("max_retries default = %d, want 2", cfg.Image.maxRetries)
	}
	if cfg.Image.Width != 512 || cfg.Image.Height != 512 || cfg.Image.Steps != 20 || cfg.Image.CFGScale != 7 {
		t.Fatalf("unexpected defaults: %+v", cfg.Image)
	}
	if cfg.Image.timeout != 2*time.Minute {
		t.Fatalf("timeout default = %v", cfg.Image.timeout)
	}
}

func TestImageConfigZeroRetriesIsRespected(t *testing.T) {
	cfg := loadTestConfig(t, `{"image":{"base_url":"http://localhost:8001","max_retries":0}}`)
	if !cfg.Image.enabled() || cfg.Image.maxRetries != 0 {
		t.Fatalf("explicit 0 retries not preserved: %+v", cfg.Image)
	}
}

func TestToolsDisabledWithoutVision(t *testing.T) {
	cfg := loadTestConfig(t, `{"image":{"base_url":"http://localhost:8001"}}`)
	b := NewBot(testLogger(), cfg, nil, nil, nil, func(context.Context, id.RoomID, id.EventID, string) {}, nil)
	if b.sd == nil {
		t.Fatal("expected a diffusion client from config")
	}
	if b.toolsEnabled() {
		t.Fatal("tools must stay off when the vision model is missing: it cannot judge what it drew")
	}
}

func TestMediaEncryptedDefaultsToEncrypted(t *testing.T) {
	// No state store to ask: nothing can say "not encrypted", so it is.
	b := &Bot{mautrixClient: &mautrix.Client{}}
	if !b.mediaEncrypted(context.Background(), testRoom) {
		t.Fatal("a room with no state store must be treated as encrypted")
	}
}

func TestUploadImageEncryptsByDefault(t *testing.T) {
	var gotBody []byte
	var gotQuery string
	var gotType string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotQuery = r.URL.RawQuery
		gotType = r.Header.Get("Content-Type")
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"content_uri":"mxc://example.org/enc1"}`)
	}))
	defer srv.Close()

	cli, err := mautrix.NewClient(srv.URL, "@gork:example.org", "token")
	if err != nil {
		t.Fatal(err)
	}
	b := &Bot{mautrixClient: cli}

	uri, file, err := b.uploadImage(context.Background(), testPNG, "gork.png", true)
	if err != nil {
		t.Fatal(err)
	}
	if file == nil {
		t.Fatal("encrypted upload must return file info")
	}
	if uri.String() != "" {
		t.Fatalf("encrypted upload must not return a plain url, got %s", uri)
	}
	if string(file.URL) != "mxc://example.org/enc1" {
		t.Fatalf("file.url = %q", file.URL)
	}
	if bytes.Equal(gotBody, testPNG) {
		t.Fatal("plaintext bytes were uploaded to an encrypted room")
	}
	if len(gotBody) != len(testPNG) {
		t.Fatalf("ciphertext length %d != plaintext %d", len(gotBody), len(testPNG))
	}
	if gotType != "application/octet-stream" {
		t.Fatalf("encrypted upload content type = %q", gotType)
	}
	if strings.Contains(gotQuery, "filename") {
		t.Fatalf("encrypted upload leaked a filename: %q", gotQuery)
	}
	if file.Version != "v2" || file.Key.Algorithm != "A256CTR" || file.InitVector == "" || file.Hashes.SHA256 == "" {
		t.Fatalf("incomplete encrypted file info: %+v", file)
	}
	// Decrypt from a JSON round-trip, the way a real client does: mautrix caches
	// decoded key material on the struct that encrypted, and that cache does not
	// carry the ciphertext hash, so reusing the same value to decrypt always
	// fails with a hash mismatch.
	wire, err := json.Marshal(file)
	if err != nil {
		t.Fatal(err)
	}
	var fresh event.EncryptedFileInfo
	if err := json.Unmarshal(wire, &fresh); err != nil {
		t.Fatal(err)
	}
	clear, err := fresh.Decrypt(gotBody)
	if err != nil {
		t.Fatalf("decrypting what was uploaded: %v (%s)", err, wire)
	}
	if !bytes.Equal(clear, testPNG) {
		t.Fatal("uploaded ciphertext does not decrypt back to the generated image")
	}
}

func TestUploadImagePlaintextWhenRoomIsNotEncrypted(t *testing.T) {
	var gotBody []byte
	var gotType string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotType = r.Header.Get("Content-Type")
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"content_uri":"mxc://example.org/plain1"}`)
	}))
	defer srv.Close()

	cli, err := mautrix.NewClient(srv.URL, "@gork:example.org", "token")
	if err != nil {
		t.Fatal(err)
	}
	uri, file, err := (&Bot{mautrixClient: cli}).uploadImage(context.Background(), testPNG, "gork.png", false)
	if err != nil {
		t.Fatal(err)
	}
	if file != nil {
		t.Fatal("plaintext upload must not return file info")
	}
	if uri.String() != "mxc://example.org/plain1" {
		t.Fatalf("uri = %s", uri)
	}
	if !bytes.Equal(gotBody, testPNG) {
		t.Fatal("plaintext upload altered the bytes")
	}
	if gotType != "image/png" {
		t.Fatalf("content type = %q", gotType)
	}
}

func TestSendImagePutsEncryptedMediaInFileField(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"event_id":"$1"}`)
	}))
	defer srv.Close()
	cli, err := mautrix.NewClient(srv.URL, "@gork:example.org", "token")
	if err != nil {
		t.Fatal(err)
	}
	b := &Bot{mautrixClient: cli}
	room := id.RoomID("!room:example.org")

	file := &event.EncryptedFileInfo{
		EncryptedFile: *attachment.NewEncryptedFile(),
		URL:           "mxc://example.org/enc1",
	}
	if err := b.sendImage(context.Background(), room, id.ContentURI{}, file, "a cube", testPNG); err != nil {
		t.Fatal(err)
	}
	if _, ok := body["url"]; ok {
		t.Fatalf("encrypted image must not carry a plain url: %v", body["url"])
	}
	f, ok := body["file"].(map[string]any)
	if !ok {
		t.Fatalf("encrypted image has no file object: %v", body)
	}
	if f["url"] != "mxc://example.org/enc1" || f["v"] != "v2" {
		t.Fatalf("bad file object: %v", f)
	}
	if body["msgtype"] != "m.image" {
		t.Fatalf("msgtype = %v", body["msgtype"])
	}
	// info still describes the plaintext image, not the ciphertext.
	info, ok := body["info"].(map[string]any)
	if !ok || info["mimetype"] != "image/png" || info["w"] != float64(3) || info["h"] != float64(2) {
		t.Fatalf("bad info: %v", body["info"])
	}

	// And the plaintext path puts it in url instead.
	body = nil
	if err := b.sendImage(context.Background(), room, mustContentURI(t, "mxc://example.org/plain1"), nil, "a cube", testPNG); err != nil {
		t.Fatal(err)
	}
	if body["url"] != "mxc://example.org/plain1" {
		t.Fatalf("plaintext url = %v", body["url"])
	}
	if _, ok := body["file"]; ok {
		t.Fatal("plaintext image must not carry a file object")
	}
}

func contains(haystack, needle string) bool {
	return len(needle) == 0 || (len(haystack) >= len(needle) && indexOf(haystack, needle) >= 0)
}

func indexOf(haystack, needle string) int {
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return i
		}
	}
	return -1
}

// loadTestConfig writes a minimal config JSON and loads it in console mode,
// which skips the Matrix credential requirements.
func loadTestConfig(t *testing.T, extra string) *Config {
	t.Helper()
	path := t.TempDir() + "/config.json"
	extra = strings.TrimSpace(extra)
	extra = strings.TrimSuffix(strings.TrimPrefix(extra, "{"), "}")
	lead := ""
	if extra != "" {
		lead = ","
	}
	body := `{"homeserver":"https://matrix.example.org","user":"@gork:example.org","password":"x"` + lead + extra + `}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfig(path, true)
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

// newSDTestServer stands in for stable-diffusion.cpp's txt2img endpoint.
func newSDTestServer(t *testing.T, pngBytes []byte) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/sdapi/v1/txt2img" {
			http.NotFound(w, r)
			return
		}
		var req sdRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("bad sd request: %v", err)
		}
		if req.Prompt == "" {
			http.Error(w, "prompt is required", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"images": []string{base64.StdEncoding.EncodeToString(pngBytes)},
			"info":   `{"seed":123}`,
		})
	}))
}

func mustContentURI(t *testing.T, raw string) id.ContentURI {
	t.Helper()
	uri, err := id.ParseContentURI(raw)
	if err != nil {
		t.Fatal(err)
	}
	return uri
}
