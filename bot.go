package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/draw"
	"image/gif"
	"image/png"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"maunium.net/go/mautrix"
	"maunium.net/go/mautrix/event"
	"maunium.net/go/mautrix/id"

	"github.com/rs/zerolog"
	"gorkbot/llm"

	"golang.org/x/net/html"
)

// workerQueueSize bounds how many messages can wait for the worker while an
// LLM call is in flight.
const workerQueueSize = 100

// Bot represents the gork chatbot.
type Bot struct {
	log             zerolog.Logger
	mention         string
	llmClient       *llm.Client
	llmVision       *llm.Client
	visionComplete  func(ctx context.Context, imageBase64, mimeType, textPrompt string) (string, int, int, int, error)
	imagePrompt     string
	maxHistory      int
	historyTimeout  time.Duration
	historyMgr      *RoomHistoryManager
	mautrixClient   *mautrix.Client
	joinedRoomsFn   func(ctx context.Context) (*mautrix.RespJoinedRooms, error)
	joinRoomFn      func(ctx context.Context, roomID id.RoomID) error
	allowedInviters map[id.UserID]struct{}
	messagesFn      func(ctx context.Context, roomID id.RoomID, from string, limit int) (*mautrix.RespMessages, error)
	downloadFn      func(ctx context.Context, mxcURL id.ContentURI) (*http.Response, error)
	selfID          id.UserID
	workerCh        chan message
	workerDone      chan struct{}
	replyFn         func(ctx context.Context, roomID id.RoomID, eventID id.EventID, response string)
	typingFn        func(ctx context.Context, roomID id.RoomID, typing bool)
	backfillMax     int
	backfillWindow  time.Duration
	stopCh          chan struct{}
	completeFn      func(ctx context.Context, history []llm.ChatMessage) (string, int, int, int, error)

	// Image generation (the draw_image tool). sd is nil when image generation is
	// disabled, which also keeps the tool away from the model.
	image         ImageConfig
	sd            *sdClient
	completeTools func(ctx context.Context, history []llm.ChatMessageTool, tools []llm.Tool, toolChoice any,
		temperature float64, maxTokens int) (llm.ToolMessage, llm.UsageInfo, error)
	drawImage   func(ctx context.Context, prompt string) ([]byte, error)
	uploadFn    func(ctx context.Context, data []byte, name string, encrypt bool) (id.ContentURI, *event.EncryptedFileInfo, error)
	sendImageFn func(ctx context.Context, roomID id.RoomID, uri id.ContentURI, file *event.EncryptedFileInfo, caption string, png []byte) error
	// mediaEncryptedFn overrides the room encryption check (test seam).
	mediaEncryptedFn func(ctx context.Context, roomID id.RoomID) bool
}

// NewBot creates a Bot from a config and LLM client. The sendReply function
// is used to send the bot's reply to Matrix; it must not be nil. The
// setTyping function, if non-nil, is used to show a typing indicator while
// waiting on the LLM provider. If llmVision is nil, image events are silently
// ignored. The mautrixClient is used for downloading images from Matrix and
// for joining rooms when a whitelisted user invites the bot.
func NewBot(log zerolog.Logger, config *Config, llmClient *llm.Client, llmVision *llm.Client, mautrixClient *mautrix.Client, sendReply func(ctx context.Context, roomID id.RoomID, eventID id.EventID, response string), setTyping func(ctx context.Context, roomID id.RoomID, typing bool)) *Bot {
	bot := &Bot{
		log:             log,
		mention:         config.Gork.Mention,
		llmClient:       llmClient,
		llmVision:       llmVision,
		imagePrompt:     config.LLMVision.ImagePrompt,
		mautrixClient:   mautrixClient,
		maxHistory:      config.LLM.MaxHistory,
		historyTimeout:  config.LLM.historyTimeout,
		historyMgr:      NewRoomHistoryManager(),
		replyFn:         sendReply,
		typingFn:        setTyping,
		backfillMax:     config.Backfill.MaxMessages,
		backfillWindow:  config.Backfill.timeout,
		allowedInviters: config.Invite.allowedInviters,
		stopCh:          make(chan struct{}),
	}
	if config.Image.enabled() {
		bot.image = config.Image
		bot.sd = &sdClient{
			BaseURL:  config.Image.BaseURL,
			Width:    config.Image.Width,
			Height:   config.Image.Height,
			Steps:    config.Image.Steps,
			CFGScale: config.Image.CFGScale,
			Timeout:  config.Image.timeout,
		}
		bot.drawImage = func(ctx context.Context, prompt string) ([]byte, error) {
			return bot.sd.generate(ctx, prompt, config.Image.NegativePrompt)
		}
	}
	return bot
}

// toolsEnabled reports whether the draw_image tool should be offered. Drawing
// needs both the diffusion server and the vision model, because the model
// cannot judge a picture it cannot see.
func (b *Bot) toolsEnabled() bool {
	return b.sd != nil && !b.visionUnavailable()
}

// RoomHistoryManager manages per-room conversation histories.
type RoomHistoryManager struct {
	mu    sync.RWMutex
	rooms map[roomID]*HistoryManager
}

// roomID is the map key type for per-room histories.
type roomID string

// NewRoomHistoryManager creates a new RoomHistoryManager.
func NewRoomHistoryManager() *RoomHistoryManager {
	return &RoomHistoryManager{
		rooms: make(map[roomID]*HistoryManager),
	}
}

// GetOrCreate returns the HistoryManager for a room, creating one if it does
// not exist yet.
func (r *RoomHistoryManager) GetOrCreate(room id.RoomID, maxMessages int, timeout time.Duration) *HistoryManager {
	rid := roomID(room)
	r.mu.RLock()
	hm, ok := r.rooms[rid]
	r.mu.RUnlock()
	if ok {
		return hm
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	// Double-check after acquiring write lock
	if hm, ok = r.rooms[rid]; ok {
		return hm
	}
	hm = NewHistoryManager(maxMessages, timeout)
	r.rooms[rid] = hm
	return hm
}

// Handler returns an OnMessage handler for the bot. Callers should register it
// with bot.OnMessage before calling bot.Run. This also starts the worker goroutine.
func (b *Bot) Handler() func(ctx context.Context, evt *event.Event) {
	if b.workerCh == nil {
		b.workerCh = make(chan message, workerQueueSize)
	}
	go b.worker(b.workerCh)
	return func(ctx context.Context, evt *event.Event) {
		evtLog := b.log.With().Str("room_id", evt.RoomID.String()).Str("event_id", evt.ID.String()).Str("event_type", evt.Type.Type).Logger()
		evtLog.Debug().Msg("Handler received event")
		content := evt.Content.AsMessage()

		// Handle image/file messages: download, describe, add to history.
		if content.MsgType == event.MsgImage || content.MsgType == event.MsgFile {
			b.handleImage(ctx, evt)
			return
		}

		body := content.Body
		formattedBody := content.FormattedBody
		if body == "" && formattedBody == "" {
			return
		}
		log := b.log.With().
			Str("room_id", evt.RoomID.String()).
			Str("sender", evt.Sender.String()).
			Str("body", body).
			Str("formatted_body", formattedBody).
			Logger()
		log.Info().Msg("Received message")

		// Detect mention in both the HTML formatted body and the plain text body.
		// The config mention is something like "@gork:caos.one".
		mentioned := hasHTMLMention(formattedBody, b.mention) || hasMention(body, b.mention)
		var prompt string
		if mentioned {
			// Prefer HTML prompt if available, fall back to plain text.
			if prompt = extractHTMLPrompt(formattedBody, b.mention); prompt == "" {
				prompt = extractPrompt(body, b.mention)
			}
		}

		select {
		case b.workerCh <- message{ctx: ctx, roomID: evt.RoomID, sender: evt.Sender.String(), body: body, formattedBody: formattedBody, mentioned: mentioned, prompt: prompt}:
		default:
			// Worker queue full; drop the message.
		}
	}
}

// HandleInvite reacts to room invites: gork joins the room when the invite was
// sent by a whitelisted user, and ignores every other invite. Register it with
// OnEvent, not OnMessage: invites arrive as m.room.member state events, so
// non-member events and non-invite membership changes fall through untouched.
func (b *Bot) HandleInvite(ctx context.Context, evt *event.Event) {
	if evt.Type.Type != event.StateMember.Type || evt.StateKey == nil {
		return
	}
	if evt.Content.AsMember().Membership != event.MembershipInvite {
		return
	}
	// The state key names the invited user; the inviter is the event sender.
	invitee := id.UserID(*evt.StateKey)
	if invitee != b.selfID {
		return
	}
	log := b.log.With().
		Str("room_id", evt.RoomID.String()).
		Str("inviter", evt.Sender.String()).
		Str("invitee", invitee.String()).
		Logger()
	if !b.inviterAllowed(evt.Sender) {
		log.Info().Msg("Ignoring invite from sender not on the invite whitelist")
		return
	}
	if err := b.joinRoom(ctx, evt.RoomID); err != nil {
		log.Error().Err(err).Msg("Failed to join room after whitelisted invite")
		return
	}
	log.Info().Msg("Joined room on invite from whitelisted sender")
}

// inviterAllowed reports whether a user is on the invite whitelist. An empty
// whitelist means gork never auto-joins.
func (b *Bot) inviterAllowed(user id.UserID) bool {
	_, ok := b.allowedInviters[user]
	return ok
}

func (b *Bot) joinRoom(ctx context.Context, roomID id.RoomID) error {
	if b.joinRoomFn != nil {
		return b.joinRoomFn(ctx, roomID)
	}
	_, err := b.mautrixClient.JoinRoomByID(ctx, roomID)
	return err
}

func (b *Bot) joinedRooms(ctx context.Context) (*mautrix.RespJoinedRooms, error) {
	if b.joinedRoomsFn != nil {
		return b.joinedRoomsFn(ctx)
	}
	return b.mautrixClient.JoinedRooms(ctx)
}

func (b *Bot) roomMessages(ctx context.Context, roomID id.RoomID, from string, limit int) (*mautrix.RespMessages, error) {
	if b.messagesFn != nil {
		return b.messagesFn(ctx, roomID, from, limit)
	}
	return b.mautrixClient.Messages(ctx, roomID, from, "", mautrix.DirectionBackward, nil, limit)
}

func (b *Bot) downloadMedia(ctx context.Context, mxcURL id.ContentURI) (*http.Response, error) {
	if b.downloadFn != nil {
		return b.downloadFn(ctx, mxcURL)
	}
	return b.mautrixClient.Download(ctx, mxcURL)
}

// handleImage processes an m.image or m.file message: downloads the media,
// sends it through the vision model, adds the description to room history,
// and feeds it into the worker queue if it triggers a mention.
func (b *Bot) handleImage(ctx context.Context, evt *event.Event) {
	if b.visionUnavailable() {
		b.log.Debug().
			Str("room_id", evt.RoomID.String()).
			Str("sender", evt.Sender.String()).
			Msg("Image received but vision model not configured, skipping")
		return
	}

	content := evt.Content.AsMessage()
	log := b.log.With().
		Str("room_id", evt.RoomID.String()).
		Str("sender", evt.Sender.String()).
		Str("msgtype", string(content.MsgType)).
		Str("body", content.Body).
		Logger()

	log.Info().Msg("Processing image message")

	// Download the media
	mediaURL := content.URL
	if mediaURL == "" && content.File != nil {
		mediaURL = content.File.URL
	}
	if mediaURL == "" {
		log.Warn().Str("has_file", fmt.Sprintf("%t", content.File != nil)).Msg("No media URL in image event")
		return
	}
	log.Debug().Str("media_url", string(mediaURL)).Msg("Got media URL")

	resolvedURL, err := mediaURL.Parse()
	if err != nil {
		log.Error().Err(err).Msg("Failed to parse media URL")
		return
	}

	log.Debug().Msg("Downloading image")
	resp, err := b.downloadMedia(ctx, resolvedURL)
	if err != nil {
		log.Error().Err(err).Msg("Failed to download image")
		return
	}
	defer resp.Body.Close()

	log.Debug().
		Str("content_type", resp.Header.Get("Content-Type")).
		Int64("content_length", resp.ContentLength).
		Msg("Downloaded image headers")

	encryptedBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Error().Err(err).Msg("Failed to read image data")
		return
	}

	// Decrypt the image data (Matrix encrypts media in encrypted rooms)
	imageBytes, err := content.File.Decrypt(encryptedBytes)
	if err != nil {
		log.Error().Err(err).Msg("Failed to decrypt image data")
		return
	}

	log.Debug().
		Int("image_bytes", len(imageBytes)).
		Msg("Read image data")

	description, err := b.describeImage(ctx, imageBytes)
	if err != nil {
		log.Error().Err(err).Msg("Failed to describe image")
		return
	}

	log.Info().
		Int("image_bytes", len(imageBytes)).
		Str("description", description).
		Msg("Image described")

	hm := b.historyMgr.GetOrCreate(evt.RoomID, b.maxHistory, b.historyTimeout)
	hm.AddMessage(llm.ChatMessage{Role: "user", Content: evt.Sender.String() + ": [📷 image] " + description})
	log.Debug().Msg("Added image description to history")

	// Check if description triggers mention
	if hasMention(description, b.mention) {
		prompt := extractPrompt(description, b.mention)
		log.Debug().
			Str("description", description).
			Str("mention", b.mention).
			Msg("Description contains mention, feeding to worker")
		select {
		case b.workerCh <- message{ctx: ctx, roomID: evt.RoomID, sender: evt.Sender.String(), body: description, mentioned: true, prompt: prompt}:
		default:
			log.Warn().Msg("Image mention: worker queue full, dropping")
		}
	} else {
		log.Debug().Msg("Description does not contain mention")
	}
}

// visionUnavailable reports whether no vision model is configured.
func (b *Bot) visionUnavailable() bool {
	return b.visionComplete == nil && b.llmVision == nil
}

// prepareImage converts downloaded media into a format the vision server
// accepts, returning the bytes and the media type to declare. Matrix hands
// back whatever the sender uploaded, while ds4 decodes only JPEG and PNG and
// matches the data-URI media type literally.
//
// The type is sniffed from the bytes rather than trusted from the HTTP header:
// homeservers have been known to send "image/jpeg; charset=binary" or a type
// that disagrees with the payload.
//
// GIF becomes PNG, not JPEG. PNG keeps every opaque pixel exact, and the
// transparent index is resolved onto white here instead of leaving the server
// to read the RGB of pixels that carry no color. The frame is flattened to
// NRGBA before encoding so the server never has to handle a paletted PNG with
// a tRNS chunk. Animated GIFs contribute their first frame; motion is not
// something a still description can carry.
func prepareImage(data []byte) ([]byte, string, error) {
	switch ct := http.DetectContentType(data); ct {
	case "image/jpeg", "image/png":
		return data, ct, nil
	case "image/gif":
		frame, err := gif.Decode(bytes.NewReader(data))
		if err != nil {
			return nil, "", fmt.Errorf("decode GIF: %w", err)
		}
		bounds := frame.Bounds()
		flat := image.NewNRGBA(bounds)
		draw.Draw(flat, bounds, image.White, image.Point{}, draw.Src)
		draw.Draw(flat, bounds, frame, bounds.Min, draw.Over)
		var out bytes.Buffer
		if err := png.Encode(&out, flat); err != nil {
			return nil, "", fmt.Errorf("re-encode GIF as PNG: %w", err)
		}
		return out.Bytes(), "image/png", nil
	default:
		return nil, "", fmt.Errorf("unsupported image type %q", ct)
	}
}

func (b *Bot) describeImage(ctx context.Context, imageBytes []byte) (string, error) {
	complete := b.visionComplete
	if complete == nil && b.llmVision != nil {
		complete = b.llmVision.CompleteMultimodal
	}
	if complete == nil {
		return "", errors.New("no vision model configured")
	}
	imageData, mimeType, err := prepareImage(imageBytes)
	if err != nil {
		return "", err
	}
	imageBase64 := base64.StdEncoding.EncodeToString(imageData)
	desc, _, _, _, err := complete(ctx, imageBase64, mimeType, b.imagePrompt)
	return desc, err
}

// worker processes messages one at a time so LLM calls never block the read
// loop and history stays consistent. It exits when workerCh is closed or
// b.stopCh is closed.
func (b *Bot) worker(workerCh <-chan message) {
	for {
		select {
		case msg, ok := <-workerCh:
			if !ok {
				return
			}
			b.handle(msg)
		case <-b.stopCh:
			return
		}
	}
}

type message struct {
	ctx           context.Context
	roomID        id.RoomID
	sender        string
	body          string
	formattedBody string
	mentioned     bool
	prompt        string
	eventID       id.EventID
}

// handle processes a single message from the worker queue.
func (b *Bot) handle(msg message) {
	hm := b.historyMgr.GetOrCreate(msg.roomID, b.maxHistory, b.historyTimeout)

	// Always add to history for context (even if not mentioned).
	hm.AddMessage(llm.ChatMessage{
		Role:    "user",
		Content: msg.sender + ": " + msg.body,
	})

	if !msg.mentioned {
		return
	}

	prompt := msg.prompt

	b.log.Info().
		Str("room_id", msg.roomID.String()).
		Str("sender", msg.sender).
		Str("prompt", prompt).
		Msg("Mention detected")

	// Create a timeout context for the LLM call. When drawing is enabled the
	// budget has to cover the diffusion calls too: worst case is one image plus
	// MaxRetries redraws, each capped by the image server timeout, on top of the
	// model turns around them.
	timeout := 2 * time.Minute
	if b.toolsEnabled() {
		timeout = time.Duration(b.image.maxRetries+1)*b.image.timeout + 2*time.Minute
	}
	ctx, cancel := context.WithTimeout(msg.ctx, timeout)
	defer cancel()

	// Show a typing indicator while waiting on the LLM provider.
	if b.typingFn != nil {
		b.typingFn(msg.ctx, msg.roomID, true)
		defer b.typingFn(msg.ctx, msg.roomID, false)
	}

	if b.toolsEnabled() {
		outcome, ok := b.runToolLoop(ctx, msg, hm, b.log)
		if ok {
			if outcome.ReplyText != "" {
				hm.AddMessage(llm.ChatMessage{Role: "assistant", Content: outcome.ReplyText})
				b.replyFn(msg.ctx, msg.roomID, msg.eventID, outcome.ReplyText)
			}
			b.log.Info().
				Str("room_id", msg.roomID.String()).
				Int("drawn", outcome.Drawn).
				Bool("sent", outcome.Sent).
				Msg("Tool loop finished")
			return
		}
		b.log.Warn().Str("room_id", msg.roomID.String()).Msg("Tool loop failed, falling back to plain reply")
	}

	var response string
	var err error
	if b.completeFn != nil {
		// Test seam: bypass the real client entirely.
		response, _, _, _, err = b.completeFn(ctx, hm.Messages())
	} else {
		response, _, _, _, err = b.llmClient.Complete(ctx, hm.Messages())
	}
	if err != nil {
		b.log.Error().
			Err(err).
			Str("room_id", msg.roomID.String()).
			Str("sender", msg.sender).
			Msg("LLM call failed")
		b.replyFn(msg.ctx, msg.roomID, msg.eventID, "gork fail. try again later.")
		return
	}

	// Add assistant response to history.
	hm.AddMessage(llm.ChatMessage{
		Role:    "assistant",
		Content: response,
	})

	// Send the reply to Matrix.
	b.log.Info().
		Str("room_id", msg.roomID.String()).
		Str("response", response).
		Msg("Sending reply")
	b.replyFn(msg.ctx, msg.roomID, msg.eventID, response)
}

// ConsoleRoom is the pseudo-room used for stdin/stdout mode. There is no
// Matrix room involved; the whole conversation is one session.
const ConsoleRoom = "!console:local"

// RunConsole reads user input line-by-line from in and replies via the bot's
// replyFn, using the single console room's history. It blocks until in
// reaches EOF or ctx is cancelled. Multi-line input is supported: a line
// ending in "..." continues until a line that does not. The mention gate is
// bypassed; every complete line is treated as a prompt to the LLM. "exit" or
// "quit" exits the loop. The worker is drained before returning, so the last
// reply is delivered. If completeFn is non-nil it is used instead of the real
// LLM client (test seam).
func (b *Bot) RunConsole(ctx context.Context, in io.Reader) {
	workerCh := make(chan message, workerQueueSize)
	b.workerDone = make(chan struct{})
	go func() {
		b.worker(workerCh)
		close(b.workerDone)
	}()
	defer func() {
		// Drain the queue so the last reply is delivered before returning.
		close(workerCh)
		<-b.workerDone
	}()

	sc := bufio.NewScanner(in)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var pending strings.Builder
	for sc.Scan() {
		line := strings.TrimRight(sc.Text(), " \t\r")
		if pending.Len() == 0 {
			switch strings.ToLower(line) {
			case "exit", "quit":
				return
			case "":
				continue
			}
		}
		if strings.HasSuffix(line, "...") {
			pending.WriteString(line[:len(line)-3])
			pending.WriteByte('\n')
			continue
		}
		body := line
		if pending.Len() > 0 {
			body = pending.String() + line
			pending.Reset()
		}
		body = strings.TrimSpace(body)
		if body == "" {
			continue
		}
		b.log.Debug().Str("body", body).Msg("Console message received")
		msg := message{
			ctx:       ctx,
			roomID:    id.RoomID(ConsoleRoom),
			sender:    "user",
			body:      body,
			mentioned: true,
			prompt:    body,
		}
		select {
		case workerCh <- msg:
		case <-ctx.Done():
			return
		default:
			b.log.Warn().Msg("Console worker queue full, dropping message")
		}
	}
	if err := sc.Err(); err != nil {
		b.log.Error().Err(err).Msg("Console input read failed")
	}
}

// Stop closes the bot's stop channel to signal the worker to exit.
func (b *Bot) Stop() {
	close(b.stopCh)
}

// Backfill loads recent room history into each room's context before the bot
// starts syncing. It only keeps messages within the configured backfill window
// and count, mirroring the LLM history limits so old messages are not
// resurrected past the normal dropoff.
func (b *Bot) Backfill(ctx context.Context) {
	rooms, err := b.joinedRooms(ctx)
	if err != nil {
		b.log.Error().Err(err).Msg("Failed to list joined rooms for backfill")
		return
	}
	cutoff := time.Now().Add(-b.backfillWindow)
	for _, roomID := range rooms.JoinedRooms {
		b.backfillRoom(ctx, roomID, cutoff)
	}
}

// backfillRoom fetches up to backfillMax message events from a room that are
// newer than cutoff and feeds them into the room's history manager.
func (b *Bot) backfillRoom(ctx context.Context, roomID id.RoomID, cutoff time.Time) {
	hm := b.historyMgr.GetOrCreate(roomID, b.maxHistory, b.historyTimeout)
	var collected []llm.ChatMessage
	from := ""
	b.log.Debug().
		Str("room_id", roomID.String()).
		Int("max_messages", b.backfillMax).
		Time("cutoff", cutoff).
		Msg("Starting backfill")
	for len(collected) < b.backfillMax {
		limit := b.backfillMax - len(collected)
		resp, err := b.roomMessages(ctx, roomID, from, limit)
		if err != nil {
			b.log.Error().Err(err).Str("room_id", roomID.String()).Msg("Failed to fetch room history for backfill")
			return
		}
		for _, evt := range resp.Chunk {
			// Encrypted rooms return m.room.encrypted events; decrypt them so
			// the message content is readable. REST-fetched events don't have
			// their content parsed, so parse it first for the decryptor.
			if evt.Type == event.EventEncrypted && b.mautrixClient != nil && b.mautrixClient.Crypto != nil {
				if err := evt.Content.ParseRaw(evt.Type); err != nil {
					b.log.Warn().Err(err).Str("room_id", roomID.String()).Msg("Failed to parse encrypted event content during backfill")
					continue
				}
				dec, err := b.mautrixClient.Crypto.Decrypt(ctx, evt)
				if err != nil {
					b.log.Warn().Err(err).Str("room_id", roomID.String()).Msg("Failed to decrypt event during backfill")
					continue
				}
				evt = dec
			}
			if evt.Type != event.EventMessage {
				continue
			}
			// Make the message content readable: REST-fetched events have their
			// content parsed into a map during JSON decoding (and decrypted
			// events get re-marshalled above, populating it too), but parse it
			// defensively if only the raw bytes are present.
			if evt.Content.Raw == nil && evt.Content.VeryRaw != nil {
				var msg event.MessageEventContent
				if err := json.Unmarshal(evt.Content.VeryRaw, &msg); err != nil {
					b.log.Warn().Err(err).Str("room_id", roomID.String()).Msg("Failed to parse message content during backfill")
					continue
				}
				evt.Content.Raw = map[string]interface{}{
					"msgtype": msg.MsgType,
					"body":    msg.Body,
				}
			}
			if evt.Timestamp > 0 && cutoff.UnixMilli() > evt.Timestamp {
				// Older than the backfill window. Pages are newest-first, so
				// no further pages will be newer; stop paginating.
				b.feedBackfill(roomID, hm, collected)
				return
			}
			collected = append(collected, b.chatMessage(ctx, evt, b.selfID))
			if len(collected) >= b.backfillMax {
				b.feedBackfill(roomID, hm, collected)
				return
			}
		}
		if resp.End != "" || resp.Start == "" {
			b.feedBackfill(roomID, hm, collected)
			return
		}
		from = resp.Start
	}
	b.feedBackfill(roomID, hm, collected)
}

// feedBackfill adds collected messages to a history manager in chronological
// order. Pages are newest-first, so the slice is reversed first.
func (b *Bot) feedBackfill(roomID id.RoomID, hm *HistoryManager, collected []llm.ChatMessage) {
	for i, j := 0, len(collected)-1; i < j; i, j = i+1, j-1 {
		collected[i], collected[j] = collected[j], collected[i]
	}
	for _, msg := range collected {
		hm.AddMessage(msg)
	}
	b.log.Info().
		Str("room_id", roomID.String()).
		Int("messages", len(collected)).
		Msg("Backfilled room history")
}

// chatMessage converts a message event into an LLM chat message. The bot's own
// messages become assistant turns so the LLM sees both sides of the room.
// For image messages, it calls the vision model to get a description of the
// image content; if the vision model is unavailable or fails, it falls back
// to the body text (alt text / filename).
func (b *Bot) chatMessage(ctx context.Context, evt *event.Event, selfID id.UserID) llm.ChatMessage {
	content := evt.Content.AsMessage()
	if evt.Sender == selfID {
		return llm.ChatMessage{Role: "assistant", Content: content.Body}
	}

	// For image messages, try to get a vision description.
	if content.MsgType == event.MsgImage || content.MsgType == event.MsgFile {
		description := b.backfillDescribeImage(ctx, evt)
		if description != "" {
			return llm.ChatMessage{Role: "user", Content: evt.Sender.String() + ": [📷 image] " + description}
		}
	}

	return llm.ChatMessage{Role: "user", Content: evt.Sender.String() + ": " + content.Body}
}

// backfillDescribeImage downloads an image from a Matrix event and runs it
// through the vision model. Returns the description or "" on any failure.
func (b *Bot) backfillDescribeImage(ctx context.Context, evt *event.Event) string {
	if b.visionUnavailable() {
		return ""
	}

	content := evt.Content.AsMessage()
	mediaURL := content.URL
	if mediaURL == "" && content.File != nil {
		mediaURL = content.File.URL
	}
	if mediaURL == "" {
		return ""
	}

	resolvedURL, err := mediaURL.Parse()
	if err != nil {
		return ""
	}

	resp, err := b.downloadMedia(ctx, resolvedURL)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()

	encryptedBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return ""
	}

	// Decrypt the image data (Matrix encrypts media in encrypted rooms)
	imageBytes, err := content.File.Decrypt(encryptedBytes)
	if err != nil {
		return ""
	}

	description, err := b.describeImage(ctx, imageBytes)
	if err != nil {
		return ""
	}

	return description
}

// hasMention reports whether content contains mention at a word boundary.
// It tries the full mention string first (e.g. "@gork:caos.one"), then the
// short form without the server suffix (e.g. "@gork").
func hasMention(content, mention string) bool {
	if mention == "" || content == "" {
		return false
	}
	// Try exact mention match.
	if hasMentionAtBoundary(content, mention) {
		return true
	}
	// Try short form (just the user part before the colon).
	if i := strings.Index(mention, ":"); i > 0 {
		return hasMentionAtBoundary(content, mention[:i])
	}
	return false
}

// hasMentionAtBoundary reports whether content contains mention at a word boundary.
func hasMentionAtBoundary(content, mention string) bool {
	offset := 0
	for {
		idx := strings.Index(content[offset:], mention)
		if idx == -1 {
			return false
		}
		absIdx := offset + idx
		after := absIdx + len(mention)
		if after >= len(content) || !isWordChar(content[after]) {
			return true
		}
		offset = absIdx + 1
	}
}

// isWordChar reports whether b is a word character (letter, digit, or underscore).
func isWordChar(b byte) bool {
	return b == '_' ||
		(b >= '0' && b <= '9') ||
		(b >= 'a' && b <= 'z') ||
		(b >= 'A' && b <= 'Z')
}

// extractPrompt removes the mention from the message content, returning the
// text that follows it. Returns "" if the mention is not present at a word
// boundary. It tries the full mention string first, then the short form.
func extractPrompt(content, mention string) string {
	if mention == "" {
		return content
	}
	if prompt := extractPromptAtBoundary(content, mention); prompt != "" {
		return prompt
	}
	// Only fall back to short form if the full mention wasn't found at all.
	if i := strings.Index(mention, ":"); i > 0 {
		// Verify the full mention isn't present before trying short form.
		if !containsAtBoundary(content, mention) {
			return extractPromptAtBoundary(content, mention[:i])
		}
	}
	return ""
}

// containsAtBoundary reports whether content contains mention at a word boundary.
func containsAtBoundary(content, mention string) bool {
	offset := 0
	for {
		idx := strings.Index(content[offset:], mention)
		if idx == -1 {
			return false
		}
		absIdx := offset + idx
		after := absIdx + len(mention)
		if after >= len(content) || !isWordChar(content[after]) {
			return true
		}
		offset = absIdx + 1
	}
}

// extractPromptAtBoundary removes mention from content at a word boundary.
func extractPromptAtBoundary(content, mention string) string {
	offset := 0
	for {
		idx := strings.Index(content[offset:], mention)
		if idx == -1 {
			return ""
		}
		absIdx := offset + idx
		after := absIdx + len(mention)
		if after < len(content) && isWordChar(content[after]) {
			offset = absIdx + 1
			continue
		}
		return strings.TrimSpace(content[after:])
	}
}

// hasHTMLMention reports whether the HTML formatted body contains a Matrix
// mention for the given mention string. It looks for <a href="https://matrix.to/#/@user:server">
// tags where the user:server part matches.
func hasHTMLMention(htmlBody, mention string) bool {
	if htmlBody == "" || mention == "" {
		return false
	}
	// The mention in the config is something like "@gork:caos.one" or just "@gork".
	// We need to find an anchor whose href contains the full user ID.
	// The href format is: https://matrix.to/#/@user:server
	for _, attr := range parseHTMLAttr(htmlBody, "a", "href") {
		if strings.Contains(attr, "matrix.to/#/"+mention) {
			return true
		}
	}
	return false
}

// extractHTMLPrompt extracts the text that follows a Matrix mention in the
// HTML formatted body. It walks the HTML tree, finds the mention anchor, and
// collects the text after its closing tag.
func extractHTMLPrompt(htmlBody, mention string) string {
	if htmlBody == "" || mention == "" {
		return ""
	}
	doc, err := html.Parse(strings.NewReader(htmlBody))
	if err != nil {
		return ""
	}

	var mentionAnchor *html.Node
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if mentionAnchor != nil {
			return
		}
		if n.Type == html.ElementNode && n.Data == "a" {
			for _, a := range n.Attr {
				if a.Key == "href" && strings.Contains(a.Val, "matrix.to/#/"+mention) {
					mentionAnchor = n
					return
				}
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(doc)
	if mentionAnchor == nil {
		return ""
	}

	// Collect text after the mention anchor's siblings.
	var result strings.Builder
	for c := mentionAnchor.NextSibling; c != nil; c = c.NextSibling {
		if c.Type == html.TextNode {
			result.WriteString(c.Data)
		} else if c.Type == html.ElementNode {
			// Walk child text nodes
			var walkText func(*html.Node)
			walkText = func(n *html.Node) {
				if n.Type == html.TextNode {
					result.WriteString(n.Data)
				}
				for cc := n.FirstChild; cc != nil; cc = cc.NextSibling {
					walkText(cc)
				}
			}
			walkText(c)
		}
	}
	return strings.TrimSpace(strings.TrimLeft(result.String(), ": "))
}

// parseHTMLAttr returns the values of a specific attribute on elements of a
// specific tag name from the HTML body.
func parseHTMLAttr(htmlBody, tag, attr string) []string {
	var result []string
	doc, err := html.Parse(strings.NewReader(htmlBody))
	if err != nil {
		return result
	}
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode && n.Data == tag {
			for _, a := range n.Attr {
				if a.Key == attr {
					result = append(result, a.Val)
				}
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(doc)
	return result
}
