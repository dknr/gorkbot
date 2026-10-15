//go:build integration

// Real-endpoint check for the draw_image tool. Requires llama.cpp with the
// Llama 3.2 model, the vision (image-to-text) server, and stable-diffusion.cpp
// all running. Run with:
//
//	go test -tags goolm,integration -run TestIntegrationDrawLoop -v -timeout 15m .
//
// Images land in $GORK_DRAW_DIR (default ./integration-draws) so a human can
// compare what the model asked for against what the eyes reported.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/rs/zerolog"
	"maunium.net/go/mautrix/crypto/attachment"
	"maunium.net/go/mautrix/event"
	"maunium.net/go/mautrix/id"

	"gorkbot/llm"
)

func intEnv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func TestIntegrationDrawLoop(t *testing.T) {
	llmURL := intEnv("GORK_LLM_URL", "http://192.168.1.2:8002/v1")
	visionURL := intEnv("GORK_VISION_URL", "http://192.168.1.2:8000/v1")
	sdURL := intEnv("GORK_SD_URL", "http://localhost:8001")
	outDir := intEnv("GORK_DRAW_DIR", "integration-draws")
	encryptUpload := true
	promptBaked := intEnv("GORK_PROMPT", "draw me a purple cat wearing a top hat and reading a newspaper")
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		t.Fatal(err)
	}

	retries := 2
	cfg := &Config{
		LLM:       LLMConfig{BaseURL: llmURL, Model: "default"},
		LLMVision: LLMVisionConfig{BaseURL: visionURL, Model: "qwen3.8-flash-next", ImagePrompt: visionPromptDefault, MaxTokens: visionMaxTokensDefault},
		Image:     ImageConfig{BaseURL: sdURL, Width: 512, Height: 512, Steps: 12, CFGScale: 7, Timeout: "3m", MaxRetries: &retries},
	}
	cfg.Image.timeout = 3 * time.Minute
	cfg.Image.maxRetries = retries // normally resolved by LoadConfig

	llmClient := llm.NewClient(cfg.LLM.BaseURL, cfg.LLM.Model, "", "You are gork, a troglodyte who occasionally draws pictures. Be brief.")
	visionClient := llm.NewClient(cfg.LLMVision.BaseURL, cfg.LLMVision.Model, "", "")
	visionClient.MaxTokens = cfg.LLMVision.MaxTokens

	log := zerolog.New(zerolog.NewConsoleWriter(func(w *zerolog.ConsoleWriter) { w.Out = os.Stdout }))
	bot := NewBot(log, cfg, llmClient, visionClient, nil,
		func(context.Context, id.RoomID, id.EventID, string) {}, nil)
	if !bot.toolsEnabled() {
		t.Fatal("tools should be enabled with both SD and vision configured")
	}

	var sent []string
	n := 0
	bot.uploadFn = func(_ context.Context, data []byte, _ string, encrypt bool) (id.ContentURI, *event.EncryptedFileInfo, error) {
		n++
		// Save the plaintext so a human can compare it with the caption: what
		// leaves the process is ciphertext when the room is encrypted.
		plaintext := data
		var file *event.EncryptedFileInfo
		if encrypt {
			file = &event.EncryptedFileInfo{EncryptedFile: *attachment.NewEncryptedFile()}
			data = file.Encrypt(data)
			wire, err := json.Marshal(file)
			if err != nil {
				return id.ContentURI{}, nil, err
			}
			var fresh event.EncryptedFileInfo
			if err := json.Unmarshal(wire, &fresh); err != nil {
				return id.ContentURI{}, nil, err
			}
			back, err := fresh.Decrypt(data)
			if err != nil {
				return id.ContentURI{}, nil, fmt.Errorf("uploaded ciphertext does not decrypt: %w", err)
			}
			if !bytes.Equal(back, plaintext) {
				return id.ContentURI{}, nil, errors.New("encrypted round-trip changed the image")
			}
		}
		path := filepath.Join(outDir, "attempt-"+time.Now().Format("150405")+".png")
		if err := os.WriteFile(path, plaintext, 0o644); err != nil {
			return id.ContentURI{}, nil, err
		}
		sent = append(sent, fmt.Sprintf("%s (encrypted=%t, %d bytes on the wire)", path, encrypt, len(data)))
		if file != nil {
			return id.ContentURI{}, file, nil
		}
		return id.MustParseContentURI("mxc://local/" + path), nil, nil
	}
	// The real bot asks the state store; here there is no homeserver client, so
	// force the encrypted path, which is what the reported bug was about.
	bot.mediaEncryptedFn = func(context.Context, id.RoomID) bool { return encryptUpload }

	var delivered []string
	bot.sendImageFn = func(_ context.Context, _ id.RoomID, uri id.ContentURI, file *event.EncryptedFileInfo, caption string, pngBytes []byte) error {
		delivered = append(delivered, caption)
		if file != nil {
			t.Logf("IMAGE DELIVERED encrypted file=%s key=%s... bytes=%d caption=%q", file.URL, file.Key.Key[:8], len(pngBytes), caption)
		} else {
			t.Logf("IMAGE DELIVERED plaintext uri=%s bytes=%d caption=%q", uri, len(pngBytes), caption)
		}
		return nil
	}

	hm := bot.historyMgr.GetOrCreate(testRoom, 50, time.Hour)
	// handle() adds the user turn before the loop runs; the model template
	// rejects a tool request that has no user message at all.
	hm.AddMessage(llm.ChatMessage{Role: "user", Content: "@ada:example.org " + promptBaked})
	msg := message{
		ctx:       context.Background(),
		roomID:    testRoom,
		sender:    "@ada:example.org",
		body:      "@ada:example.org " + promptBaked,
		mentioned: true,
		prompt:    promptBaked,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 12*time.Minute)
	defer cancel()

	out, ok := bot.runToolLoop(ctx, msg, hm, log)
	if !ok {
		t.Fatal("tool loop failed against real endpoints")
	}

	t.Logf("drawn=%d sent=%v reply=%q", out.Drawn, out.Sent, out.ReplyText)
	t.Logf("files=%v", sent)
	t.Logf("--- history ---")
	for _, m := range hm.Messages() {
		t.Logf("%s: %s", m.Role, m.Content)
	}

	if out.Drawn < 1 {
		t.Fatal("model never called draw_image")
	}
	if !out.Sent {
		t.Fatal("nothing was sent to the room")
	}
	if out.Drawn > 3 {
		t.Fatalf("drew %d times, budget is 3", out.Drawn)
	}
	captions := 0
	for _, m := range hm.Messages() {
		if len(m.Content) > len(toolResultPrefix) && contains(m.Content, toolResultPrefix) {
			captions++
		}
	}
	if captions != out.Drawn {
		t.Fatalf("history kept %d captions for %d images", captions, out.Drawn)
	}
}

// TestIntegrationTurboSettingsRoundTrip loads the real config.json and checks
// that the steps, CFG and LoRA it names are the ones the running
// stable-diffusion.cpp server actually applies, as reported back in the txt2img
// "info" echo. This is the check that a config edit landed: the server has no
// startup LoRA option, so an adapter that is not applied is invisible except
// through this echo.
func TestIntegrationTurboSettingsRoundTrip(t *testing.T) {
	cfg := loadLiveImageConfig(t)
	if !cfg.Image.enabled() {
		t.Skip("image.base_url empty in config.json; nothing to check")
	}

	sd := &sdClient{
		BaseURL:        cfg.Image.BaseURL,
		Width:          cfg.Image.Width,
		Height:         cfg.Image.Height,
		Steps:          cfg.Image.Steps,
		CFGScale:       cfg.Image.CFGScale,
		NegativePrompt: cfg.Image.NegativePrompt,
		Lora:           cfg.Image.lora,
		Timeout:        cfg.Image.timeout,
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	_, info, err := sd.generate(ctx, "a red cube on a white background")
	if err != nil {
		t.Fatal(err)
	}

	// The server's own echo is the record of what it applied.
	t.Logf("server applied: %s", sdInfoSummary(info))
	t.Logf("config says:    %d steps, CFG %v, %d lora(s), negative=%q, %dx%d",
		cfg.Image.Steps, cfg.Image.CFGScale, len(cfg.Image.lora), cfg.Image.NegativePrompt,
		cfg.Image.Width, cfg.Image.Height)
	// Steps and CFG are echoed as fields, so they can be asserted. This build
	// does not echo which adapters it applied, so the request side is covered by
	// TestGenerateSendsConfiguredStepsCFGAndLora and the echo is only reported.
	var echo struct {
		Steps    *int     `json:"steps"`
		CFGScale *float64 `json:"cfg_scale"`
	}
	if err := json.Unmarshal([]byte(info), &echo); err != nil {
		t.Fatalf("server echo is not JSON: %v (%s)", err, info[:min(len(info), 120)])
	}
	if echo.Steps == nil || *echo.Steps != cfg.Image.Steps {
		t.Fatalf("server ran %v steps, config says %d", echo.Steps, cfg.Image.Steps)
	}
	if echo.CFGScale == nil || *echo.CFGScale != cfg.Image.CFGScale {
		t.Fatalf("server ran CFG %v, config says %v", echo.CFGScale, cfg.Image.CFGScale)
	}
	for _, l := range cfg.Image.lora {
		if !strings.Contains(info, filepath.Base(l.Path)) {
			t.Logf("warning: adapter %s does not appear in the server echo; this build does not report applied LoRAs",
				filepath.Base(l.Path))
		}
	}
}

// loadLiveImageConfig builds a config around the "image" block of the real
// config.json next to the binary, so the live settings are what get tested.
func loadLiveImageConfig(t *testing.T) *Config {
	t.Helper()
	raw, err := os.ReadFile("config.json")
	if err != nil {
		t.Skipf("no config.json to read: %v", err)
	}
	var whole map[string]json.RawMessage
	if err := json.Unmarshal(raw, &whole); err != nil {
		t.Fatal(err)
	}
	image, ok := whole["image"]
	if !ok {
		t.Skip("config.json has no image block")
	}
	return loadTestConfig(t, `"image":`+string(image))
}
