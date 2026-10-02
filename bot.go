package main

import (
	"context"
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
	log            zerolog.Logger
	mention        string
	llmClient      *llm.Client
	maxHistory     int
	historyTimeout time.Duration
	historyMgr     *RoomHistoryManager
	replyFn        func(ctx context.Context, roomID id.RoomID, eventID id.EventID, response string)
	typingFn       func(ctx context.Context, roomID id.RoomID, typing bool)
	backfillMax    int
	backfillWindow time.Duration
	stopCh         chan struct{}
}

// NewBot creates a Bot from a config and LLM client. The sendReply function
// is used to send the bot's reply to Matrix; it must not be nil. The
// setTyping function, if non-nil, is used to show a typing indicator while
// waiting on the LLM provider.
func NewBot(log zerolog.Logger, config *Config, llmClient *llm.Client, sendReply func(ctx context.Context, roomID id.RoomID, eventID id.EventID, response string), setTyping func(ctx context.Context, roomID id.RoomID, typing bool)) *Bot {
	return &Bot{
		log:            log,
		mention:        config.Gork.Mention,
		llmClient:      llmClient,
		maxHistory:     config.LLM.MaxHistory,
		historyTimeout: config.LLM.historyTimeout,
		historyMgr:     NewRoomHistoryManager(),
		replyFn:        sendReply,
		typingFn:       setTyping,
		backfillMax:    config.Backfill.MaxMessages,
		backfillWindow: config.Backfill.timeout,
		stopCh:         make(chan struct{}),
	}
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
// with bot.OnMessage before calling bot.Run.
func (b *Bot) Handler() func(ctx context.Context, evt *event.Event) {
	workerCh := make(chan message, workerQueueSize)
	go b.worker(workerCh)
	return func(ctx context.Context, evt *event.Event) {
		body := evt.Content.AsMessage().Body
		formattedBody := evt.Content.AsMessage().FormattedBody
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
		case workerCh <- message{ctx: ctx, roomID: evt.RoomID, sender: evt.Sender.String(), body: body, formattedBody: formattedBody, mentioned: mentioned, prompt: prompt}:
		default:
			// Worker queue full; drop the message.
		}
	}
}

// worker processes messages one at a time so LLM calls never block the read
// loop and history stays consistent.
func (b *Bot) worker(workerCh <-chan message) {
	for {
		select {
		case msg := <-workerCh:
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

	// Create a 2-minute timeout context for the LLM call.
	ctx, cancel := context.WithTimeout(msg.ctx, 2*time.Minute)
	defer cancel()

	// Show a typing indicator while waiting on the LLM provider.
	if b.typingFn != nil {
		b.typingFn(msg.ctx, msg.roomID, true)
		defer b.typingFn(msg.ctx, msg.roomID, false)
	}

	response, _, _, _, err := b.llmClient.Complete(ctx, hm.Messages())
	if err != nil {
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

// Stop closes the bot's stop channel to signal the worker to exit.
func (b *Bot) Stop() {
	close(b.stopCh)
}

// Backfill loads recent room history into each room's context before the bot
// starts syncing. It only keeps messages within the configured backfill window
// and count, mirroring the LLM history limits so old messages are not
// resurrected past the normal dropoff.
func (b *Bot) Backfill(ctx context.Context, client *mautrix.Client) {
	rooms, err := client.JoinedRooms(ctx)
	if err != nil {
		b.log.Error().Err(err).Msg("Failed to list joined rooms for backfill")
		return
	}
	cutoff := time.Now().Add(-b.backfillWindow)
	for _, roomID := range rooms.JoinedRooms {
		b.backfillRoom(ctx, client, roomID, cutoff)
	}
}

// backfillRoom fetches up to backfillMax message events from a room that are
// newer than cutoff and feeds them into the room's history manager.
func (b *Bot) backfillRoom(ctx context.Context, client *mautrix.Client, roomID id.RoomID, cutoff time.Time) {
	hm := b.historyMgr.GetOrCreate(roomID, b.maxHistory, b.historyTimeout)
	var collected []llm.ChatMessage
	from := ""
	for len(collected) < b.backfillMax {
		limit := b.backfillMax - len(collected)
		resp, err := client.Messages(ctx, roomID, from, "", mautrix.DirectionBackward, nil, limit)
		if err != nil {
			b.log.Error().Err(err).Str("room_id", roomID.String()).Msg("Failed to fetch room history for backfill")
			return
		}
		for _, evt := range resp.Chunk {
			// Encrypted rooms return m.room.encrypted events; decrypt them so
			// the message content is readable. REST-fetched events don't have
			// their content parsed, so parse it first for the decryptor.
			if evt.Type == event.EventEncrypted && client.Crypto != nil {
				if err := evt.Content.ParseRaw(evt.Type); err != nil {
					b.log.Warn().Err(err).Str("room_id", roomID.String()).Msg("Failed to parse encrypted event content during backfill")
					continue
				}
				dec, err := client.Crypto.Decrypt(ctx, evt)
				if err != nil {
					b.log.Warn().Err(err).Str("room_id", roomID.String()).Msg("Failed to decrypt event during backfill")
					continue
				}
				evt = dec
			}
			if evt.Type != event.EventMessage {
				continue
			}
			if evt.Timestamp > 0 && cutoff.UnixMilli() > evt.Timestamp {
				// Older than the backfill window. Pages are newest-first, so
				// no further pages will be newer; stop paginating.
				b.feedBackfill(roomID, hm, collected)
				return
			}
			collected = append(collected, b.chatMessage(evt, client.UserID))
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
func (b *Bot) chatMessage(evt *event.Event, selfID id.UserID) llm.ChatMessage {
	body := evt.Content.AsMessage().Body
	if evt.Sender == selfID {
		return llm.ChatMessage{Role: "assistant", Content: body}
	}
	return llm.ChatMessage{Role: "user", Content: evt.Sender.String() + ": " + body}
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
