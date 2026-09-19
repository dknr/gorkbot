package main

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
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

// GorkConfig holds gork-specific configuration.
type GorkConfig struct {
	SystemPrompt string `json:"system_prompt"`
	Mention      string `json:"mention"`
}

// Config holds the application configuration.
type Config struct {
	Homeserver  string        `json:"homeserver"`
	User        string        `json:"user"`
	Password    string        `json:"password"`
	RecoveryKey string        `json:"recovery_key"`
	Database    string        `json:"database"`
	PickleKey   string        `json:"pickle_key"`
	DeviceName  string        `json:"device_name"`
	LLM         LLMConfig     `json:"llm"`
	Gork        GorkConfig    `json:"gork"`
}

// LoadConfig reads the bot config from a JSON file, applying defaults for any
// field that is omitted.
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	if cfg.Homeserver == "" || cfg.User == "" || cfg.Password == "" {
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
	return &cfg, nil
}
