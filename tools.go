package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/rs/zerolog"
	"maunium.net/go/mautrix"
	"maunium.net/go/mautrix/crypto/attachment"
	"maunium.net/go/mautrix/event"
	"maunium.net/go/mautrix/id"

	"gorkbot/llm"
)

// Tool calling on llama.cpp's Llama 3.2 build is fragile in specific,
// reproducible ways (see RESEARCH notes in the tool loop below). The whole
// design here is driven by four measured facts:
//
//  1. With tools offered on a turn that already contains a tool result, the
//     model re-issues the same call forever and never reads the result.
//  2. Dropping Tools and sending tool_choice:"none" on that same turn makes it
//     answer from the result reliably (measured 0/5 -> 5/5).
//  3. A model with no way to say "no tool needed" always calls the tool
//     anyway. Giving it an answer_directly tool fixes abstention (0/8 -> 3/3).
//  4. Two tool calls emitted in one turn hard-fails the server with HTTP 500
//     ("peg-native format") and the parsed first call is lost, so only the
//     first call in a response is ever acted on.

// toolSystemHint is appended to the configured system prompt on tool-calling
// turns. The Llama 3.2 template ships no tool section at all, so the model has
// to be told the rules of the loop: one call per turn (a second call in one
// turn errors the server), and answer_directly is how it declines to draw.
const toolSystemHint = "You can draw pictures with the draw_image tool. Call it once per turn, " +
	"never twice in the same turn. After you draw, you are told what the picture actually looks like. " +
	"Check that description against what was asked. If anything the user asked for is missing, you must " +
	"call draw_image again with a corrected prompt. Only reply once the picture matches or you have no " +
	"draws left. When you reply, say what you made in one short sentence of your own words; never copy " +
	"the description text. When no picture is needed, reply with the answer_directly tool."

// toolTemperature and toolMaxTokens bound the tool-calling turns. Temperature
// is kept low because tool-call formatting degrades sharply above ~0.3.
const (
	toolTemperature = 0.2
	toolMaxTokens   = 512
	sdBodyLimit     = 1 << 20
)

// toolNames for the tools gork exposes.
const (
	toolDrawImage     = "draw_image"
	toolAnswerDirect  = "answer_directly"
	toolHistoryPrefix = "[🎨 image] "
)

// drawTools returns the tool set offered to the model. answer_directly exists
// purely as the abstention path; without it the model draws a picture for any
// mention, including "tell me a joke about bicycles".
func drawTools() []llm.Tool {
	return []llm.Tool{
		{
			Type: "function",
			Function: llm.ToolFunction{
				Name: toolDrawImage,
				Description: "Draw a new image from a text description and show it to the user. " +
					"You will be told what the drawing actually looks like afterwards.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"prompt": map[string]any{
							"type":        "string",
							"description": "Detailed visual description of the image to draw",
						},
					},
					"required": []string{"prompt"},
				},
			},
		},
		{
			Type: "function",
			Function: llm.ToolFunction{
				Name: toolAnswerDirect,
				Description: "Use this when no drawing is needed and you can answer from your own knowledge. " +
					"Put your whole reply in the answer field.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"answer": map[string]any{
							"type":        "string",
							"description": "The complete reply to send to the user",
						},
					},
					"required": []string{"answer"},
				},
			},
		},
	}
}

// sdClient talks to a stable-diffusion.cpp server over its A1111-compatible
// /sdapi/v1/txt2img endpoint.
type sdClient struct {
	BaseURL    string
	HTTPClient *http.Client
	Width      int
	Height     int
	Steps      int
	CFGScale   float64
	Timeout    time.Duration
}

type sdRequest struct {
	Prompt         string  `json:"prompt"`
	NegativePrompt string  `json:"negative_prompt,omitempty"`
	Width          int     `json:"width"`
	Height         int     `json:"height"`
	Steps          int     `json:"steps"`
	CFGScale       float64 `json:"cfg_scale"`
	Seed           int64   `json:"seed"`
}

type sdResponse struct {
	Images []string `json:"images"`
	Info   string   `json:"info"`
}

// generate renders prompt and returns PNG bytes. Seed is -1 so every retry
// produces a different image; retrying with the same seed would redraw the
// same thing and defeat the purpose.
func (s *sdClient) generate(ctx context.Context, prompt, negative string) ([]byte, error) {
	if strings.TrimSpace(prompt) == "" {
		return nil, errors.New("empty draw prompt")
	}
	client := s.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	if s.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, s.Timeout)
		defer cancel()
	}

	reqBody := sdRequest{
		Prompt:         prompt,
		NegativePrompt: negative,
		Width:          s.Width,
		Height:         s.Height,
		Steps:          s.Steps,
		CFGScale:       s.CFGScale,
		Seed:           -1,
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal sd request: %w", err)
	}

	url := strings.TrimRight(s.BaseURL, "/") + "/sdapi/v1/txt2img"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create sd request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send sd request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, sdBodyLimit>>4))
		return nil, fmt.Errorf("sd status %d: %s", resp.StatusCode, strings.TrimSpace(string(snippet)))
	}

	var parsed sdResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, fmt.Errorf("decode sd response: %w", err)
	}
	if len(parsed.Images) == 0 || parsed.Images[0] == "" {
		return nil, errors.New("sd returned no image")
	}
	png, err := base64.StdEncoding.DecodeString(parsed.Images[0])
	if err != nil {
		return nil, fmt.Errorf("decode sd image: %w", err)
	}
	return png, nil
}

// drawRequest is the model-supplied argument payload for draw_image.
type drawRequest struct {
	Prompt string `json:"prompt"`
}

// answerRequest is the model-supplied argument payload for answer_directly.
type answerRequest struct {
	Answer string `json:"answer"`
}

// decodeArgs unmarshals a tool call's JSON arguments. llama.cpp sometimes emits
// a call with empty arguments, which is reported as an empty object rather than
// a hard error so the caller can send a corrective observation.
func decodeArgs(raw string, target any) error {
	if strings.TrimSpace(raw) == "" {
		return json.Unmarshal([]byte("{}"), target)
	}
	return json.Unmarshal([]byte(raw), target)
}

// hasUserTurn reports whether the history contains at least one user message.
func hasUserTurn(history []llm.ChatMessageTool) bool {
	for _, m := range history {
		if m.Role == "user" {
			return true
		}
	}
	return false
}

// toToolHistory converts plain string history into the tool-path message shape.
func toToolHistory(history []llm.ChatMessage) []llm.ChatMessageTool {
	out := make([]llm.ChatMessageTool, 0, len(history))
	for _, m := range history {
		out = append(out, llm.ChatMessageTool{Role: m.Role, Content: m.Content})
	}
	return out
}

// imageOutcome reports what a tool-loop run did, for logging and tests.
type imageOutcome struct {
	Drawn     int  // images generated
	Sent      bool // an image reached the room
	ReplyText string
}

// runToolLoop answers one mentioned message with the tool set available.
//
// Budget: the model may draw once, then retry up to MaxRetries times. Once the
// budget is spent the tools are withdrawn (fact 2) so the run terminates, and
// whatever the last image was gets sent regardless of how it looks.
//
// Every generated image, including discarded retries, is described through the
// vision model and appended to room history as a user-role message. The
// caption goes in as a user message rather than only as a tool result because
// the model demonstrably ignores tool-result content while tools are offered
// (fact 1); the user-role copy is the channel it actually reads.
func (b *Bot) runToolLoop(ctx context.Context, msg message, hm *HistoryManager, log zerolog.Logger) (imageOutcome, bool) {
	out := imageOutcome{}
	complete := b.completeTools
	if complete == nil {
		if b.llmClient == nil {
			return out, false
		}
		// Copy the client so the tool hint does not leak into the plain path.
		hinted := *b.llmClient
		if hinted.SystemPrompt == "" {
			hinted.SystemPrompt = toolSystemHint
		} else {
			hinted.SystemPrompt += "\n\n" + toolSystemHint
		}
		complete = hinted.CompleteTools
	}
	draw := b.drawImage
	if draw == nil {
		draw = func(ctx context.Context, prompt string) ([]byte, error) {
			return b.sd.generate(ctx, prompt, b.image.NegativePrompt)
		}
	}
	tools := drawTools()
	history := toToolHistory(hm.Messages())

	// The model's chat template injects the tool definitions into the first
	// user message and hard-errors when there is none, so a history that has
	// not been seeded with the current user turn cannot use tools at all.
	if !hasUserTurn(history) {
		log.Warn().Msg("Tool loop skipped: no user turn in history")
		return out, false
	}

	maxDraws := b.image.maxRetries + 1
	var lastPNG []byte
	var lastPrompt string

	for {
		budgetLeft := out.Drawn < maxDraws

		var offered []llm.Tool
		var choice any
		if budgetLeft {
			offered = tools
		} else {
			// Tools withdrawn: force a plain-text answer.
			choice = "none"
		}

		toolMsg, _, err := complete(ctx, history, offered, choice, toolTemperature, toolMaxTokens)
		if err != nil {
			log.Err(err).Msg("Tool-calling LLM request failed")
			return out, false
		}

		// The run ends when the model stops calling tools, and also when the draw
		// budget is spent: at that point tools are withdrawn, so whatever comes
		// back is the closing turn. A tool call in that response is never acted
		// on, otherwise a model that keeps calling (which Llama 3.2 does) would
		// draw forever.
		if len(toolMsg.ToolCalls) == 0 || !budgetLeft {
			if lastPNG != nil && !out.Sent {
				if err := b.publishImage(ctx, msg, lastPNG, lastPrompt, toolMsg.Content); err != nil {
					log.Err(err).Msg("Failed to send drawn image")
				} else {
					out.Sent = true
				}
			}
			out.ReplyText = toolMsg.Content
			return out, true
		}

		// Act on the first call only: a response carrying two calls is a server
		// error (fact 4), and a second call would be lost anyway.
		call := toolMsg.ToolCalls[0]
		history = append(history, llm.ChatMessageTool{
			Role:      "assistant",
			Content:   toolMsg.Content,
			ToolCalls: []llm.ToolCall{call},
		})

		switch call.Function.Name {
		case toolAnswerDirect:
			var args answerRequest
			if err := decodeArgs(call.Function.Arguments, &args); err != nil {
				log.Err(err).Msg("Bad answer_directly arguments")
			}
			reply := strings.TrimSpace(args.Answer)
			if reply == "" {
				reply = strings.TrimSpace(toolMsg.Content)
			}
			if lastPNG != nil && !out.Sent {
				if err := b.publishImage(ctx, msg, lastPNG, lastPrompt, reply); err != nil {
					log.Err(err).Msg("Failed to send drawn image")
				} else {
					out.Sent = true
				}
			}
			out.ReplyText = reply
			return out, true

		case toolDrawImage:
			var args drawRequest
			if err := decodeArgs(call.Function.Arguments, &args); err != nil {
				log.Err(err).Msg("Bad draw_image arguments")
			}
			out.Drawn++
			png, err := draw(ctx, args.Prompt)
			if err != nil {
				log.Err(err).Str("draw_prompt", args.Prompt).Msg("Image generation failed")
				history = append(history,
					toolResult(call, "the drawing failed: "+err.Error()),
					userTurn(msg.sender, "The drawing failed. Reply to the user and say you could not draw it."))
				continue
			}

			caption, captionErr := b.describeImage(ctx, png)
			if captionErr != nil {
				log.Err(captionErr).Msg("Failed to describe generated image")
			}
			// A blank description is a vision-model miss, not an empty picture.
			// Left blank it also breaks the request: the server needs content on
			// every tool turn.
			inspectable := captionErr == nil && strings.TrimSpace(caption) != ""
			if !inspectable {
				caption = "(the drawing came out and could not be inspected)"
			}

			// Persist the description for every attempt, retry or not.
			captionLine := msg.sender + ": " + toolHistoryPrefix + caption
			hm.AddMessage(llm.ChatMessage{Role: "user", Content: captionLine})
			history = append(history, toolResult(call, caption))

			lastPNG, lastPrompt = png, args.Prompt
			log.Info().Int("attempt", out.Drawn).Int("max_draws", maxDraws).
				Str("caption", caption).Msg("Drew image")

			if out.Drawn >= maxDraws {
				// Retry budget spent: this result must be sent.
				if err := b.publishImage(ctx, msg, lastPNG, lastPrompt, ""); err != nil {
					log.Err(err).Msg("Failed to send final image")
				} else {
					out.Sent = true
				}
				history = append(history, userTurn(msg.sender,
					fmt.Sprintf("%s%s The picture has already been sent to the user. That was your last "+
						"draw, you cannot draw again. Tell the user in one short sentence of your own words "+
						"what the picture shows. Do not repeat the description above.",
						toolHistoryPrefix, caption)))
				// Loop again with tools withdrawn so the model closes the turn in text.
				continue
			}

			if !inspectable {
				// Redrawing on an unreadable description throws a generation
				// away at a vision-model hiccup, so keep what was drawn.
				history = append(history, userTurn(msg.sender,
					toolHistoryPrefix+caption+" You could not see this picture, so do not judge it and "+
						"do not redraw it. Reply to the user in one short sentence saying what you drew."))
				continue
			}

			history = append(history, userTurn(msg.sender,
				fmt.Sprintf("%s%s Compare that with what the user asked for. If anything they asked for is "+
					"missing or wrong, call draw_image again right now with a corrected prompt. Only reply to "+
					"the user if the picture matches. You have drawn %d time(s) and may redraw %d more time(s).",
					toolHistoryPrefix, caption, out.Drawn, maxDraws-out.Drawn)))

		default:
			log.Warn().Str("tool", call.Function.Name).Msg("Model called an unknown tool")
			history = append(history, toolResult(call, "no such tool"))
		}
	}
}

// toolResult builds the tool-role observation for a completed call.
func toolResult(call llm.ToolCall, content string) llm.ChatMessageTool {
	return llm.ChatMessageTool{Role: "tool", Content: content, ToolCallID: call.ID}
}

// userTurn builds a user-role message on the tool path.
func userTurn(sender, content string) llm.ChatMessageTool {
	return llm.ChatMessageTool{Role: "user", Content: sender + ": " + content}
}

// publishImage encrypts when the room needs it, uploads, and sends the
// m.image event with whichever field the room's encryption calls for.
func (b *Bot) publishImage(ctx context.Context, msg message, png []byte, drawPrompt, replyText string) error {
	encrypt := b.mediaEncrypted(ctx, msg.roomID)
	uri, file, err := b.uploadImage(ctx, png, "gork.png", encrypt)
	if err != nil {
		return fmt.Errorf("upload image: %w", err)
	}
	caption := replyText
	if caption == "" {
		caption = drawPrompt
	}
	if err := b.sendImage(ctx, msg.roomID, uri, file, caption, png); err != nil {
		return fmt.Errorf("send image: %w", err)
	}
	return nil
}

// mediaEncrypted reports whether media for this room must be uploaded
// encrypted. The answer defaults to yes: a plaintext upload in an encrypted
// room is invisible to everyone who is not on the homeserver that holds the
// media, while an encrypted upload in a plaintext room still renders for
// clients that read the file field. Only a state store that positively answers
// "not encrypted" downgrades to a plain url.
func (b *Bot) mediaEncrypted(ctx context.Context, roomID id.RoomID) bool {
	if b.mediaEncryptedFn != nil {
		return b.mediaEncryptedFn(ctx, roomID)
	}
	if b.mautrixClient == nil {
		return false
	}
	store, ok := b.mautrixClient.StateStore.(interface {
		IsEncrypted(context.Context, id.RoomID) (bool, error)
	})
	if !ok {
		return true
	}
	encrypted, err := store.IsEncrypted(ctx, roomID)
	if err != nil {
		return true
	}
	return encrypted
}

// uploadImage puts the PNG in the homeserver media repo.
//
// In an encrypted room the bytes are encrypted client-side first and uploaded
// as application/octet-stream with no filename, exactly as the spec and every
// other Matrix client do; the returned EncryptedFileInfo carries the key, IV
// and ciphertext hash that go into the event's "file" field, and the plain URL
// comes back empty so it cannot be put in the event by mistake.
func (b *Bot) uploadImage(ctx context.Context, data []byte, name string, encrypt bool) (id.ContentURI, *event.EncryptedFileInfo, error) {
	if b.uploadFn != nil {
		return b.uploadFn(ctx, data, name, encrypt)
	}
	if b.mautrixClient == nil {
		return id.ContentURI{}, nil, errors.New("no homeserver client configured")
	}
	body := data
	contentType := "image/png"
	fileName := name
	var file *event.EncryptedFileInfo
	if encrypt {
		file = &event.EncryptedFileInfo{EncryptedFile: *attachment.NewEncryptedFile()}
		body = file.Encrypt(data)
		contentType = "application/octet-stream"
		fileName = ""
	}
	resp, err := b.mautrixClient.UploadMedia(ctx, mautrix.ReqUploadMedia{
		ContentBytes: body,
		FileName:     fileName,
		ContentType:  contentType,
	})
	if err != nil {
		return id.ContentURI{}, nil, err
	}
	if file != nil {
		// The mxc belongs inside the file object; the event must not also carry
		// a plain url.
		file.URL = id.ContentURIString(resp.ContentURI.String())
		return id.ContentURI{}, file, nil
	}
	return resp.ContentURI, nil, nil
}

// sendImage sends an m.image event to a room. Encrypted media goes in "file"
// with the plaintext mime type and size still described in "info"; plaintext
// media goes in "url". Width and height are read from the PNG so clients can
// reserve layout space before the media loads.
func (b *Bot) sendImage(ctx context.Context, roomID id.RoomID, uri id.ContentURI, file *event.EncryptedFileInfo, caption string, png []byte) error {
	if b.sendImageFn != nil {
		return b.sendImageFn(ctx, roomID, uri, file, caption, png)
	}
	if b.mautrixClient == nil {
		return errors.New("no homeserver client configured")
	}
	info := &event.FileInfo{MimeType: "image/png", Size: len(png)}
	if w, h, err := pngSize(png); err == nil {
		info.Width, info.Height = w, h
	}
	content := &event.MessageEventContent{
		MsgType:  event.MsgImage,
		Body:     caption,
		FileName: "gork.png",
		Info:     info,
	}
	if file != nil {
		content.File = file
	} else {
		content.URL = id.ContentURIString(uri.String())
	}
	_, err := b.mautrixClient.SendMessageEvent(ctx, roomID, event.EventMessage, content)
	return err
}

// pngSize reads the intrinsic dimensions so clients can reserve layout space.
func pngSize(data []byte) (int, int, error) {
	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return 0, 0, err
	}
	return cfg.Width, cfg.Height, nil
}
