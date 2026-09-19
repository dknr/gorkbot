package main

import (
	"testing"
)

func TestHasMention(t *testing.T) {
	tests := []struct {
		name    string
		content string
		mention string
		want    bool
	}{
		{"exact at start", "@gork:caos.one hello", "@gork:caos.one", true},
		{"exact at end", "hello @gork:caos.one", "@gork:caos.one", true},
		{"short form", "@gork hello", "@gork:caos.one", true},
		{"short form in middle", "hey @gork what's up", "@gork:caos.one", true},
		{"followed by space", "@gork:caos.one hi", "@gork:caos.one", true},
		{"followed by punctuation", "@gork:caos.one!", "@gork:caos.one", true},
		{"followed by period", "@gork:caos.one.", "@gork:caos.one", true},
		{"no mention", "hello world", "@gork:caos.one", false},
		{"partial match no match", "@gorkfreak:caos.one hello", "@gork:caos.one", false},
		{"partial match short", "@gorkfreak hello", "@gork:caos.one", false},
		{"mention in middle", "hey @gork:caos.one what's up", "@gork:caos.one", true},
		{"empty content", "", "@gork:caos.one", false},
		{"mention only", "@gork:caos.one", "@gork:caos.one", true},
		{"short mention only", "@gork", "@gork:caos.one", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := hasMention(tt.content, tt.mention)
			if got != tt.want {
				t.Errorf("hasMention(%q, %q) = %v, want %v", tt.content, tt.mention, got, tt.want)
			}
		})
	}
}

func TestExtractPrompt(t *testing.T) {
	tests := []struct {
		name    string
		content string
		mention string
		want    string
	}{
		{
			name:    "full mention with text after",
			content: "@gork:caos.one hello world",
			mention: "@gork:caos.one",
			want:    "hello world",
		},
		{
			name:    "short mention with text after",
			content: "@gork hello world",
			mention: "@gork:caos.one",
			want:    "hello world",
		},
		{
			name:    "mention at start",
			content: "@gork:caos.one",
			mention: "@gork:caos.one",
			want:    "",
		},
		{
			name:    "no mention",
			content: "hello world",
			mention: "@gork:caos.one",
			want:    "",
		},
		{
			name:    "mention in middle",
			content: "hey @gork what's up",
			mention: "@gork:caos.one",
			want:    "what's up",
		},
		{
			name:    "mention with extra whitespace",
			content: "@gork:caos.one   hello   ",
			mention: "@gork:caos.one",
			want:    "hello",
		},
		{
			name:    "empty content",
			content: "",
			mention: "@gork:caos.one",
			want:    "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractPrompt(tt.content, tt.mention)
			if got != tt.want {
				t.Errorf("extractPrompt(%q, %q) = %q, want %q", tt.content, tt.mention, got, tt.want)
			}
		})
	}
}

func TestHasHTMLMention(t *testing.T) {
	tests := []struct {
		name    string
		html    string
		mention string
		want    bool
	}{
		{
			name:    "full mention with server",
			html:    `<a href="https://matrix.to/#/@gork:caos.one">gork</a>: hello`,
			mention: "@gork:caos.one",
			want:    true,
		},
		{
			name:    "mention without text after",
			html:    `<a href="https://matrix.to/#/@gork:caos.one">gork</a>`,
			mention: "@gork:caos.one",
			want:    true,
		},
		{
			name:    "mention in middle",
			html:    `hey <a href="https://matrix.to/#/@gork:caos.one">gork</a> what's up`,
			mention: "@gork:caos.one",
			want:    true,
		},
		{
			name:    "no mention",
			html:    `hello world`,
			mention: "@gork:caos.one",
			want:    false,
		},
		{
			name:    "partial match no match",
			html:    `<a href="https://matrix.to/#/@gorkfreak:caos.one">gorkfreak</a> hello`,
			mention: "@gork:caos.one",
			want:    false,
		},
		{
			name:    "empty html",
			html:    ``,
			mention: "@gork:caos.one",
			want:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := hasHTMLMention(tt.html, tt.mention)
			if got != tt.want {
				t.Errorf("hasHTMLMention(%q, %q) = %v, want %v", tt.html, tt.mention, got, tt.want)
			}
		})
	}
}

func TestExtractHTMLPrompt(t *testing.T) {
	tests := []struct {
		name    string
		html    string
		mention string
		want    string
	}{
		{
			name:    "mention with text after",
			html:    `<a href="https://matrix.to/#/@gork:caos.one">gork</a>: this is a full mention`,
			mention: "@gork:caos.one",
			want:    "this is a full mention",
		},
		{
			name:    "mention at start",
			html:    `<a href="https://matrix.to/#/@gork:caos.one">gork</a>`,
			mention: "@gork:caos.one",
			want:    "",
		},
		{
			name:    "mention in middle",
			html:    `hey <a href="https://matrix.to/#/@gork:caos.one">gork</a> what's up`,
			mention: "@gork:caos.one",
			want:    "what's up",
		},
		{
			name:    "mention with extra whitespace",
			html:    `<a href="https://matrix.to/#/@gork:caos.one">gork</a>:   hello   `,
			mention: "@gork:caos.one",
			want:    "hello",
		},
		{
			name:    "no mention returns empty",
			html:    `hello world`,
			mention: "@gork:caos.one",
			want:    "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractHTMLPrompt(tt.html, tt.mention)
			if got != tt.want {
				t.Errorf("extractHTMLPrompt(%q, %q) = %q, want %q", tt.html, tt.mention, got, tt.want)
			}
		})
	}
}

func TestParseHTMLAttr(t *testing.T) {
	tests := []struct {
		name  string
		html  string
		tag   string
		attr  string
		want  []string
	}{
		{
			name:  "single anchor href",
			html:  `<a href="https://matrix.to/#/@gork:caos.one">gork</a>`,
			tag:   "a",
			attr:  "href",
			want:  []string{"https://matrix.to/#/@gork:caos.one"},
		},
		{
			name:  "multiple links",
			html:  `<p>see <a href="https://x.com/a">x</a> and <a href="https://github.com/b">github</a></p>`,
			tag:   "a",
			attr:  "href",
			want:  []string{"https://x.com/a", "https://github.com/b"},
		},
		{
			name:  "no matches",
			html:  `<p>hello</p>`,
			tag:   "a",
			attr:  "href",
			want:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseHTMLAttr(tt.html, tt.tag, tt.attr)
			if len(got) != len(tt.want) {
				t.Errorf("parseHTMLAttr(%q, %q, %q) = %v, want %v", tt.html, tt.tag, tt.attr, got, tt.want)
			}
		})
	}
}
