package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"maunium.net/go/mautrix/id"
)

// LLMConfig holds LLM provider configuration.
type LLMConfig struct {
	BaseURL        string `json:"base_url"`
	APIKey         string `json:"api_key"`
	Model          string `json:"model"`
	MaxHistory     int    `json:"max_history"`
	HistoryTimeout string `json:"history_timeout"`
	historyTimeout time.Duration
}

// LLMVisionConfig holds vision model (image-to-text) configuration.
type LLMVisionConfig struct {
	BaseURL     string `json:"base_url"`
	APIKey      string `json:"api_key"`
	Model       string `json:"model"`
	ImagePrompt string `json:"image_prompt"`
}

// GorkConfig holds gork-specific configuration.
type GorkConfig struct {
	SystemPrompt string `json:"system_prompt"`
	Mention      string `json:"mention"`
	Markdown     *bool  `json:"markdown"`
}

// BackfillConfig controls how much room history is loaded into context on
// startup. It mirrors the LLM history limits (max count + timeout) so backfill
// cannot cross the history dropoff window.
type BackfillConfig struct {
	MaxMessages int    `json:"max_messages"`
	Timeout     string `json:"timeout"`
	timeout     time.Duration
}

// InviteConfig controls how the bot reacts to room invites. Gork only joins a
// room automatically when the invite was sent by a user listed in
// AllowedInviters; an empty whitelist means invites are never acted on.
// Matching is exact on full Matrix user IDs.
type InviteConfig struct {
	AllowedInviters []string `json:"allowed_inviters"`
	allowedInviters map[id.UserID]struct{}
}

// Config holds the application configuration.
type Config struct {
	Homeserver  string          `json:"homeserver"`
	User        string          `json:"user"`
	Password    string          `json:"password"`
	RecoveryKey string          `json:"recovery_key"`
	Database    string          `json:"database"`
	PickleKey   string          `json:"pickle_key"`
	DeviceName  string          `json:"device_name"`
	LLM         LLMConfig       `json:"llm"`
	LLMVision   LLMVisionConfig `json:"llm_vision"`
	Gork        GorkConfig      `json:"gork"`
	Backfill    BackfillConfig  `json:"backfill"`
	Invite      InviteConfig    `json:"invite"`
}

// LoadConfig reads the bot config from a JSON file, applying defaults for any
// field that is omitted.
// LoadConfig reads the bot config from a JSON file, applying defaults for any
// field that is omitted. When console is true, Matrix credentials are not
// required; the bot runs against stdin/stdout instead of a homeserver.
func LoadConfig(path string, console bool) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	if !console && (cfg.Homeserver == "" || cfg.User == "" || cfg.Password == "") {
		return nil, fmt.Errorf("homeserver, user and password are required in the config")
	}
	if cfg.Database == "" {
		cfg.Database = "gorkbot.db"
	}
	if cfg.PickleKey == "" {
		cfg.PickleKey = "gorkbot"
	}
	if cfg.DeviceName == "" {
		cfg.DeviceName = "gorkbot"
	}
	if cfg.LLM.BaseURL == "" {
		cfg.LLM.BaseURL = "http://localhost:8080"
	}
	if cfg.LLM.Model == "" {
		cfg.LLM.Model = "llama-3.2-3b-instruct"
	}
	if cfg.LLM.MaxHistory == 0 {
		cfg.LLM.MaxHistory = 100
	}
	if cfg.LLM.HistoryTimeout == "" {
		cfg.LLM.HistoryTimeout = "15m"
	}
	duration, err := time.ParseDuration(cfg.LLM.HistoryTimeout)
	if err != nil {
		return nil, fmt.Errorf("invalid history_timeout %q: %w", cfg.LLM.HistoryTimeout, err)
	}
	cfg.LLM.historyTimeout = duration
	if cfg.Gork.Mention == "" {
		cfg.Gork.Mention = cfg.User
	}
	// Vision model defaults: disabled when base_url is empty.
	if cfg.LLMVision.BaseURL == "" {
		cfg.LLMVision.ImagePrompt = "Describe what you see in this image"
	} else {
		if cfg.LLMVision.Model == "" {
			cfg.LLMVision.Model = "ggml-org/Qwen2-VL-2B-Instruct-GGUF"
		}
		if cfg.LLMVision.ImagePrompt == "" {
			cfg.LLMVision.ImagePrompt = "Describe what you see in this image"
		}
	}
	// Markdown formatting defaults to enabled. A bool pointer lets an explicit
	// "markdown": false opt out, since an omitted field is otherwise
	// indistinguishable from false.
	if cfg.Gork.Markdown == nil {
		enabled := true
		cfg.Gork.Markdown = &enabled
	}
	if cfg.Backfill.MaxMessages == 0 {
		cfg.Backfill.MaxMessages = 50
	}
	if cfg.Backfill.Timeout == "" {
		cfg.Backfill.Timeout = "24h"
	}
	backfillDuration, err := time.ParseDuration(cfg.Backfill.Timeout)
	if err != nil {
		return nil, fmt.Errorf("invalid backfill timeout %q: %w", cfg.Backfill.Timeout, err)
	}
	cfg.Backfill.timeout = backfillDuration
	// Invite whitelist: entries are validated up front so a typo is a startup
	// error instead of an entry that silently never matches.
	cfg.Invite.allowedInviters = make(map[id.UserID]struct{}, len(cfg.Invite.AllowedInviters))
	for _, raw := range cfg.Invite.AllowedInviters {
		inviter := id.UserID(strings.TrimSpace(raw))
		if _, _, err := inviter.Parse(); err != nil {
			return nil, fmt.Errorf("invalid invite.allowed_inviters entry %q: %w", raw, err)
		}
		cfg.Invite.allowedInviters[inviter] = struct{}{}
	}
	return &cfg, nil
}
