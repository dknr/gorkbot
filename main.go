package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/rs/zerolog"
	"maunium.net/go/mautrix/event"
	"maunium.net/go/mautrix/id"

	"github.com/dknr/ragecore"
	"gorkbot/llm"
)

func main() {
	configPath := flag.String("config", "config.json", "path to the configuration file")
	console := flag.Bool("console", false, "run without Matrix: read prompts from stdin, write replies to stdout")
	flag.Parse()

	logFile, err := os.OpenFile("gorkbot.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to open log file: %v\n", err)
		os.Exit(1)
	}

	// Console mode keeps stdout for gork's replies, so logs go to stderr and
	// the log file; Matrix mode logs to the console and the log file.
	out := os.Stdout
	if *console {
		out = os.Stderr
	}
	consoleWriter := zerolog.NewConsoleWriter(func(w *zerolog.ConsoleWriter) {
		w.TimeFormat = time.StampMilli
		w.Out = out
	})
	multiWriter := io.MultiWriter(consoleWriter, logFile)

	log := zerolog.New(multiWriter).With().Timestamp().Logger().Level(zerolog.DebugLevel)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	cfg, err := LoadConfig(*configPath, *console)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to load config")
	}

	llmClient := llm.NewClient(
		cfg.LLM.BaseURL,
		cfg.LLM.Model,
		cfg.LLM.APIKey,
		cfg.Gork.SystemPrompt,
	)

	if *console {
		runConsole(ctx, log, cfg, llmClient)
		return
	}

	var llmVision *llm.Client
	if cfg.LLMVision.BaseURL != "" {
		llmVision = llm.NewClient(
			cfg.LLMVision.BaseURL,
			cfg.LLMVision.Model,
			cfg.LLMVision.APIKey,
			"Describe what you see in this image in detail",
		)
	}

	rageBot, err := ragecore.New(ctx, &ragecore.Config{
		Homeserver:  cfg.Homeserver,
		User:        cfg.User,
		Password:    cfg.Password,
		RecoveryKey: cfg.RecoveryKey,
		Database:    cfg.Database,
		PickleKey:   cfg.PickleKey,
		DeviceName:  cfg.DeviceName,
	}, log)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to start ragecore bot")
	}

	gorkBot := NewBot(log, cfg, llmClient, llmVision, rageBot.Client(), func(ctx context.Context, roomID id.RoomID, eventID id.EventID, response string) {
		content := &event.MessageEventContent{
			MsgType: event.MsgText,
			Body:    response,
		}
		// Render the LLM's markdown reply to Matrix's custom HTML format so
		// bold, code blocks, headings, etc. display nicely in room.
		if cfg.Gork.Markdown != nil && *cfg.Gork.Markdown {
			content.Format = event.FormatHTML
			content.FormattedBody = renderMarkdown(response)
		}
		// We don't have the original event here, so reply is sent without
		// threading. The full conversation history is maintained in the LLM
		// context, so this is acceptable.
		if _, err := rageBot.Client().SendMessageEvent(ctx, roomID, event.EventMessage, content); err != nil {
			log.Error().Err(err).Str("room_id", roomID.String()).Msg("Failed to send reply")
		}
	}, func(ctx context.Context, roomID id.RoomID, typing bool) {
		// Single typing event with a 30s timeout; expires before long LLM
		// generations finish, but covers short waits.
		if _, err := rageBot.Client().UserTyping(ctx, roomID, typing, 30*time.Second); err != nil {
			log.Error().Err(err).Str("room_id", roomID.String()).Msg("Failed to set typing indicator")
		}
	})

	rageBot.OnMessage(gorkBot.Handler())
	// Invites are m.room.member state events, so they need the catch-all hook;
	// gork only joins when the inviter is on the configured whitelist.
	rageBot.OnEvent(gorkBot.HandleInvite)

	log.Info().Msg("Gork is running")
	gorkBot.selfID = rageBot.Client().UserID
	gorkBot.Backfill(ctx)
	if err := rageBot.Run(ctx); err != nil {
		log.Fatal().Err(err).Msg("Bot failed")
	}
}

// runConsole runs gork without Matrix: each stdin line is a prompt and each
// LLM reply is written to stdout, one response per block.
func runConsole(ctx context.Context, log zerolog.Logger, cfg *Config, llmClient *llm.Client) {
	gorkBot := NewBot(log, cfg, llmClient, nil, nil,
		func(_ context.Context, _ id.RoomID, _ id.EventID, response string) {
			fmt.Fprintln(os.Stdout, response)
		},
		nil,
	)

	// Banner to stderr: stdout is reserved for gork's replies so it can be
	// piped cleanly (e.g. over a serial console with socat).
	fmt.Fprintln(os.Stderr, "gork console: type a prompt per line; '...' continues a line; exit/quit or Ctrl-D to stop")
	gorkBot.RunConsole(ctx, os.Stdin)
}
