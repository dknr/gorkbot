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
	// MaxTokens caps the description. Reasoning counts against this budget on
	// ds4, which is why the vision client disables thinking; raise MaxTokens
	// before re-enabling it.
	MaxTokens int `json:"max_tokens"`
}

// ImageConfig holds the image generation ("hands") configuration: a
// stable-diffusion.cpp server exposing the A1111-compatible txt2img API. When
// BaseURL is empty the draw_image tool is not offered to the model at all and
// the bot behaves as it did before this feature existed.
type ImageConfig struct {
	BaseURL        string  `json:"base_url"`
	Width          int     `json:"width"`
	Height         int     `json:"height"`
	Steps          int     `json:"steps"`
	CFGScale       float64 `json:"cfg_scale"`
	NegativePrompt string  `json:"negative_prompt"`
	Timeout        string  `json:"timeout"`
	// MaxRetries caps redraws after the first image. A pointer distinguishes an
	// explicit 0 (never redraw, always send the first result) from an omitted
	// field, which defaults to 2.
	MaxRetries *int `json:"max_retries"`

	maxRetries int
	timeout    time.Duration
}

// enabled reports whether image generation is configured.
func (i ImageConfig) enabled() bool { return strings.TrimSpace(i.BaseURL) != "" }

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
	Image       ImageConfig     `json:"image"`
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
		if cfg.LLMVision.MaxTokens == 0 {
			cfg.LLMVision.MaxTokens = 1600
		}
	}
	// Image generation defaults: disabled when base_url is empty.
	if cfg.Image.MaxRetries == nil {
		retries := 2
		cfg.Image.MaxRetries = &retries
	}
	if *cfg.Image.MaxRetries < 0 {
		return nil, fmt.Errorf("image.max_retries must be 0 or greater, got %d", *cfg.Image.MaxRetries)
	}
	cfg.Image.maxRetries = *cfg.Image.MaxRetries
	if cfg.Image.Width <= 0 {
		cfg.Image.Width = 512
	}
	if cfg.Image.Height <= 0 {
		cfg.Image.Height = 512
	}
	if cfg.Image.Steps <= 0 {
		cfg.Image.Steps = 20
	}
	if cfg.Image.CFGScale <= 0 {
		cfg.Image.CFGScale = 7
	}
	if cfg.Image.Timeout == "" {
		cfg.Image.Timeout = "2m"
	}
	imageTimeout, err := time.ParseDuration(cfg.Image.Timeout)
	if err != nil {
		return nil, fmt.Errorf("invalid image.timeout %q: %w", cfg.Image.Timeout, err)
	}
	cfg.Image.timeout = imageTimeout
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
