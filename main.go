package main

import (
	"context"
	"flag"
	"fmt"
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
	flag.Parse()

	cfg, err := LoadConfig(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load config: %v\n", err)
		os.Exit(1)
	}

	log := zerolog.New(zerolog.NewConsoleWriter(func(w *zerolog.ConsoleWriter) {
		w.TimeFormat = time.StampMilli
	})).With().Timestamp().Logger().Level(zerolog.InfoLevel)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

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

	llmClient := llm.NewClient(
		cfg.LLM.BaseURL,
		cfg.LLM.Model,
		cfg.LLM.APIKey,
		cfg.Gork.SystemPrompt,
	)

	gorkBot := NewBot(log, cfg, llmClient, func(ctx context.Context, roomID id.RoomID, eventID id.EventID, response string) {
		content := &event.MessageEventContent{
			MsgType: event.MsgText,
			Body:    response,
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

	log.Info().Msg("Gork is running")
	gorkBot.Backfill(ctx, rageBot.Client())
	if err := rageBot.Run(ctx); err != nil {
		log.Fatal().Err(err).Msg("Bot failed")
	}
}
