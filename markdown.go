package main

import (
	"bytes"

	"github.com/yuin/goldmark"
	"github.com/yuin/goldmark/extension"
)

// renderer is a shared goldmark instance with a set of common extensions so
// LLM replies get rich HTML formatting in Matrix (headings, strikethrough,
// tables, task lists, etc.).
var renderer = goldmark.New(
	goldmark.WithExtensions(
		extension.GFM,
		extension.Strikethrough,
		extension.Table,
		extension.TaskList,
	),
)

// renderMarkdown converts a markdown document to Matrix's custom HTML format.
// It returns the HTML string, or an empty string if conversion fails.
func renderMarkdown(md string) string {
	if md == "" {
		return ""
	}
	var buf bytes.Buffer
	if err := renderer.Convert([]byte(md), &buf); err != nil {
		return ""
	}
	return buf.String()
}
