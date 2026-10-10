package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestLoadConfigDefaults(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.json")

	configData := `{
		"homeserver": "https://matrix.example.org",
		"user": "@gork:example.org",
		"password": "secret"
	}`
	if err := os.WriteFile(configPath, []byte(configData), 0644); err != nil {
		t.Fatal(err)
	}

	cfg, err := LoadConfig(configPath, false)
	if err != nil {
		t.Fatalf("LoadConfig() error: %v", err)
	}

	if cfg.Homeserver != "https://matrix.example.org" {
		t.Errorf("expected homeserver 'https://matrix.example.org', got %q", cfg.Homeserver)
	}
	if cfg.User != "@gork:example.org" {
		t.Errorf("expected user '@gork:example.org', got %q", cfg.User)
	}
	if cfg.Password != "secret" {
		t.Errorf("expected password 'secret', got %q", cfg.Password)
	}

	// Check defaults
	if cfg.Database != "gorkbot.db" {
		t.Errorf("expected default database 'gorkbot.db', got %q", cfg.Database)
	}
	if cfg.PickleKey != "gorkbot" {
		t.Errorf("expected default pickle_key 'gorkbot', got %q", cfg.PickleKey)
	}
	if cfg.DeviceName != "gorkbot" {
		t.Errorf("expected default device_name 'gorkbot', got %q", cfg.DeviceName)
	}
	if cfg.LLM.BaseURL != "http://localhost:8080" {
		t.Errorf("expected default base_url 'http://localhost:8080', got %q", cfg.LLM.BaseURL)
	}
	if cfg.LLM.Model != "llama-3.2-3b-instruct" {
		t.Errorf("expected default model 'llama-3.2-3b-instruct', got %q", cfg.LLM.Model)
	}
	if cfg.LLM.MaxHistory != 100 {
		t.Errorf("expected default max_history 100, got %d", cfg.LLM.MaxHistory)
	}
	if cfg.LLM.historyTimeout != 15*time.Minute {
		t.Errorf("expected default history_timeout 15m, got %v", cfg.LLM.historyTimeout)
	}
	if cfg.Gork.Mention != cfg.User {
		t.Errorf("expected mention to default to user '@gork:example.org', got %q", cfg.Gork.Mention)
	}
	if cfg.Backfill.MaxMessages != 50 {
		t.Errorf("expected default backfill max_messages 50, got %d", cfg.Backfill.MaxMessages)
	}
	if cfg.Backfill.timeout != 24*time.Hour {
		t.Errorf("expected default backfill timeout 24h, got %v", cfg.Backfill.timeout)
	}
	if cfg.Gork.Markdown == nil || !*cfg.Gork.Markdown {
		t.Errorf("expected default markdown to be enabled, got %v", cfg.Gork.Markdown)
	}
}

func TestLoadConfigFull(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.json")

	configData := `{
		"homeserver": "https://matrix.example.org",
		"user": "@gork:example.org",
		"password": "secret",
		"recovery_key": "recovery123",
		"database": "gork.db",
		"pickle_key": "mykey",
		"device_name": "mydevice",
		"llm": {
			"base_url": "http://localhost:3000",
			"api_key": "llmkey",
			"model": "custom-model",
			"max_history": 50,
			"history_timeout": "5m"
		},
		"gork": {
			"mention": "@gork:other.org"
		},
		"backfill": {
			"max_messages": 25,
			"timeout": "3h"
		}
	}`
	if err := os.WriteFile(configPath, []byte(configData), 0644); err != nil {
		t.Fatal(err)
	}

	cfg, err := LoadConfig(configPath, false)
	if err != nil {
		t.Fatalf("LoadConfig() error: %v", err)
	}

	if cfg.RecoveryKey != "recovery123" {
		t.Errorf("expected recovery_key 'recovery123', got %q", cfg.RecoveryKey)
	}
	if cfg.Database != "gork.db" {
		t.Errorf("expected database 'gork.db', got %q", cfg.Database)
	}
	if cfg.LLM.APIKey != "llmkey" {
		t.Errorf("expected LLM api_key 'llmkey', got %q", cfg.LLM.APIKey)
	}
	if cfg.Gork.Mention != "@gork:other.org" {
		t.Errorf("expected mention '@gork:other.org', got %q", cfg.Gork.Mention)
	}
	if cfg.Backfill.MaxMessages != 25 {
		t.Errorf("expected backfill max_messages 25, got %d", cfg.Backfill.MaxMessages)
	}
	if cfg.Backfill.timeout != 3*time.Hour {
		t.Errorf("expected backfill timeout 3h, got %v", cfg.Backfill.timeout)
	}
	if cfg.Gork.Markdown == nil || !*cfg.Gork.Markdown {
		t.Errorf("expected markdown to be enabled by default, got %v", cfg.Gork.Markdown)
	}
}

func TestLoadConfigMarkdownDisabled(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.json")

	configData := `{
		"homeserver": "https://matrix.example.org",
		"user": "@gork:example.org",
		"password": "secret",
		"gork": {
			"markdown": false
		}
	}`
	if err := os.WriteFile(configPath, []byte(configData), 0644); err != nil {
		t.Fatal(err)
	}

	cfg, err := LoadConfig(configPath, false)
	if err != nil {
		t.Fatalf("LoadConfig() error: %v", err)
	}
	if cfg.Gork.Markdown == nil || *cfg.Gork.Markdown {
		t.Errorf("expected markdown to be disabled, got %v", cfg.Gork.Markdown)
	}
}

func TestLoadConfigMissingRequired(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.json")

	configData := `{
		"homeserver": "https://matrix.example.org"
	}`
	if err := os.WriteFile(configPath, []byte(configData), 0644); err != nil {
		t.Fatal(err)
	}

	_, err := LoadConfig(configPath, false)
	if err == nil {
		t.Fatal("expected error for missing user and password, got nil")
	}
}

func TestLoadConfigInvalidHistoryTimeout(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.json")
	configData := `{
		"homeserver": "https://matrix.example.org",
		"user": "@gork:example.org",
		"password": "secret",
		"llm": { "history_timeout": "not-a-duration" }
	}`
	if err := os.WriteFile(configPath, []byte(configData), 0644); err != nil {
		t.Fatal(err)
	}
	_, err := LoadConfig(configPath, false)
	if err == nil {
		t.Fatal("expected error for invalid history_timeout, got nil")
	}
	if !strings.Contains(err.Error(), "history_timeout") {
		t.Errorf("expected error to mention history_timeout, got: %v", err)
	}
}

func TestLoadConfigInvalidBackfillTimeout(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.json")
	configData := `{
		"homeserver": "https://matrix.example.org",
		"user": "@gork:example.org",
		"password": "secret",
		"backfill": { "timeout": "soon" }
	}`
	if err := os.WriteFile(configPath, []byte(configData), 0644); err != nil {
		t.Fatal(err)
	}
	_, err := LoadConfig(configPath, false)
	if err == nil {
		t.Fatal("expected error for invalid backfill timeout, got nil")
	}
	if !strings.Contains(err.Error(), "backfill") {
		t.Errorf("expected error to mention backfill, got: %v", err)
	}
}

func TestLoadConfigInvalidJSON(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.json")
	if err := os.WriteFile(configPath, []byte("{ not json"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadConfig(configPath, false); err == nil {
		t.Fatal("expected error for invalid JSON, got nil")
	}
}

func TestLoadConfigMissingFile(t *testing.T) {
	if _, err := LoadConfig(filepath.Join(t.TempDir(), "does-not-exist.json"), false); err == nil {
		t.Fatal("expected error for missing config file, got nil")
	}
}

func TestLoadConfigConsoleMode(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.json")

	// No Matrix credentials: invalid for Matrix mode, valid for console mode.
	configData := `{
		"llm": {
			"base_url": "http://localhost:8080",
			"model": "test-model"
		},
		"gork": {
			"system_prompt": "You are gork."
		}
	}`
	if err := os.WriteFile(configPath, []byte(configData), 0644); err != nil {
		t.Fatal(err)
	}

	if _, err := LoadConfig(configPath, false); err == nil {
		t.Fatal("expected error for missing matrix credentials in matrix mode, got nil")
	}

	cfg, err := LoadConfig(configPath, true)
	if err != nil {
		t.Fatalf("LoadConfig(console) error: %v", err)
	}
	if cfg.Gork.SystemPrompt != "You are gork." {
		t.Errorf("expected system prompt preserved, got %q", cfg.Gork.SystemPrompt)
	}
	if cfg.Gork.Mention != "" {
		t.Errorf("expected empty mention in console mode, got %q", cfg.Gork.Mention)
	}
}

// writeConfig writes config JSON to a temp file and returns its path.
func writeConfig(t *testing.T, configData string) string {
	t.Helper()
	configPath := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(configPath, []byte(configData), 0644); err != nil {
		t.Fatal(err)
	}
	return configPath
}

func TestLoadConfigInviteWhitelistDefaultsEmpty(t *testing.T) {
	configPath := writeConfig(t, `{
		"homeserver": "https://matrix.example.org",
		"user": "@gork:example.org",
		"password": "secret"
	}`)

	cfg, err := LoadConfig(configPath, false)
	if err != nil {
		t.Fatalf("LoadConfig() error: %v", err)
	}
	if len(cfg.Invite.allowedInviters) != 0 {
		t.Errorf("expected empty invite whitelist by default, got %v", cfg.Invite.allowedInviters)
	}
}

func TestLoadConfigInviteWhitelist(t *testing.T) {
	configPath := writeConfig(t, `{
		"homeserver": "https://matrix.example.org",
		"user": "@gork:example.org",
		"password": "secret",
		"invite": {
			"allowed_inviters": ["@alice:example.org", " @bob:example.org "]
		}
	}`)

	cfg, err := LoadConfig(configPath, false)
	if err != nil {
		t.Fatalf("LoadConfig() error: %v", err)
	}
	b := &Bot{allowedInviters: cfg.Invite.allowedInviters}
	if !b.inviterAllowed("@alice:example.org") {
		t.Errorf("expected @alice:example.org to be whitelisted")
	}
	// Surrounding whitespace in the config must not break matching.
	if !b.inviterAllowed("@bob:example.org") {
		t.Errorf("expected @bob:example.org to be whitelisted")
	}
	if b.inviterAllowed("@mallory:example.org") {
		t.Errorf("expected @mallory:example.org not to be whitelisted")
	}
}

func TestLoadConfigInviteWhitelistRejectsInvalidID(t *testing.T) {
	configPath := writeConfig(t, `{
		"homeserver": "https://matrix.example.org",
		"user": "@gork:example.org",
		"password": "secret",
		"invite": {
			"allowed_inviters": ["alice:example.org"]
		}
	}`)

	if _, err := LoadConfig(configPath, false); err == nil {
		t.Fatal("expected error for malformed allowed_inviters entry, got nil")
	}
}
