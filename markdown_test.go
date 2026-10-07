package main

import (
	"strings"
	"testing"
)

func TestRenderMarkdown(t *testing.T) {
	tests := []struct {
		name string
		md   string
		want string
	}{
		{
			name: "bold",
			md:   "**bold**",
			want: "<strong>bold</strong>",
		},
		{
			name: "heading",
			md:   "# Title",
			want: "<h1>Title</h1>",
		},
		{
			name: "code block",
			md:   "```go\nfmt.Println(\"hi\")\n```",
			want: "<pre><code class=\"language-go\">fmt.Println(&quot;hi&quot;)\n</code></pre>",
		},
		{
			name: "inline code",
			md:   "`x := 1`",
			want: "<code>x := 1</code>",
		},
		{
			name: "link",
			md:   "[a](https://example.org)",
			want: "<a href=\"https://example.org\">a</a>",
		},
		{
			name: "plain text",
			md:   "hello",
			want: "<p>hello</p>",
		},
		{
			name: "empty",
			md:   "",
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := renderMarkdown(tt.md)
			if !strings.Contains(got, tt.want) {
				t.Errorf("renderMarkdown(%q) = %q, want it to contain %q", tt.md, got, tt.want)
			}
		})
	}
}
