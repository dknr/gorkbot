package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// errorBodyLimit caps how much of a failed response body is kept, so a
// server-side HTML error page cannot flood the log.
const errorBodyLimit = 512

// responseError reports a non-200 response. Providers put the actual reason
// (unknown endpoint, missing vision encoder, unsupported image) in the body
// rather than the status code, so a snippet of it goes into the error.
func responseError(resp *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, errorBodyLimit))
	snippet := strings.TrimSpace(string(body))
	if snippet == "" {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}
	return fmt.Errorf("unexpected status code: %d: %s", resp.StatusCode, snippet)
}

// Client is an LLM client for the llama.cpp API.
type Client struct {
	BaseURL      string
	Model        string
	APIKey       string
	SystemPrompt string
	HTTPClient   *http.Client

	// Think turns model-side reasoning on or off on the multimodal path.
	// nil omits the field and leaves the server default in place. Reasoning
	// counts against MaxTokens, so enabling it without raising that budget
	// can leave no room for the visible answer.
	Think *bool

	// MaxTokens caps generated tokens. It is only sent on the multimodal
	// path; zero omits the field and uses the server default.
	MaxTokens int
}

// NewClient creates a new LLM client.
func NewClient(baseURL, model, apiKey, systemPrompt string) *Client {
	return &Client{
		BaseURL:      baseURL,
		Model:        model,
		APIKey:       apiKey,
		SystemPrompt: systemPrompt,
		HTTPClient:   &http.Client{},
	}
}

// ChatMessage represents a message in the chat history.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatMessageContent is a content part within a chat message, supporting both
// plain text and image_url types for multimodal models.
type ChatMessageContent struct {
	Type     string    `json:"type,omitempty"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL holds an image URL for multimodal messages.
type ImageURL struct {
	URL string `json:"url"`
}

// ChatRequest is the request body for the chat completions endpoint.
type ChatRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
}

// ChatRequestMultimodal is the request body for multimodal chat completions.
// Think and MaxTokens are ds4 extensions; both are omitted when unset.
type ChatRequestMultimodal struct {
	Model     string                  `json:"model"`
	Messages  []ChatMessageMultimodal `json:"messages"`
	Think     *bool                   `json:"think,omitempty"`
	MaxTokens int                     `json:"max_tokens,omitempty"`
}

// ChatMessageMultimodal represents a message that can contain both text and image content parts.
type ChatMessageMultimodal struct {
	Role    string               `json:"role"`
	Content []ChatMessageContent `json:"content"`
}

// ChatResponse is the response from the chat completions endpoint.
type ChatResponse struct {
	Choices []Choice  `json:"choices"`
	Usage   UsageInfo `json:"usage"`
}

// UsageInfo holds token usage information from the LLM.
type UsageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Choice represents a single choice in the response.
type Choice struct {
	Message Message `json:"message"`
}

// Message represents the message content in a choice.
type Message struct {
	Content string `json:"content"`
}

// CompleteMultimodal sends an image and text prompt to a multimodal model and
// returns the generated text plus token usage.
func (c *Client) CompleteMultimodal(ctx context.Context, imageBase64, mimeType, textPrompt string) (string, int, int, int, error) {
	if c.HTTPClient == nil {
		c.HTTPClient = &http.Client{}
	}

	url := fmt.Sprintf("%s/chat/completions", c.BaseURL)

	mimeTypePrefix := "image/jpeg"
	if mimeType != "" && strings.HasPrefix(mimeType, "image/") {
		mimeTypePrefix = mimeType
	}

	// The image prompt already states what is wanted, so a system message is
	// only sent when the caller configured one.
	messages := make([]ChatMessageMultimodal, 0, 2)
	if c.SystemPrompt != "" {
		messages = append(messages, ChatMessageMultimodal{
			Role:    "system",
			Content: []ChatMessageContent{{Type: "text", Text: c.SystemPrompt}},
		})
	}
	messages = append(messages, ChatMessageMultimodal{
		Role: "user",
		Content: []ChatMessageContent{
			{
				Type:     "image_url",
				ImageURL: &ImageURL{URL: "data:" + mimeTypePrefix + ";base64," + imageBase64},
			},
			{Type: "text", Text: textPrompt},
		},
	})

	req := ChatRequestMultimodal{
		Model:     c.Model,
		Messages:  messages,
		Think:     c.Think,
		MaxTokens: c.MaxTokens,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return "", 0, 0, 0, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "", 0, 0, 0, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	if c.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	}

	resp, err := c.HTTPClient.Do(httpReq)
	if err != nil {
		return "", 0, 0, 0, fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", 0, 0, 0, responseError(resp)
	}

	var chatResp ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return "", 0, 0, 0, fmt.Errorf("decode response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return "", 0, 0, 0, fmt.Errorf("no choices in response")
	}

	content := strings.TrimLeft(chatResp.Choices[0].Message.Content, " \t\n\r")
	return content, chatResp.Usage.PromptTokens, chatResp.Usage.CompletionTokens, chatResp.Usage.TotalTokens, nil
}

// Complete sends the conversation history to the LLM and returns the
// generated text plus token usage. The most recent history entry should be
// the current user message.
func (c *Client) Complete(ctx context.Context, history []ChatMessage) (string, int, int, int, error) {
	if c.HTTPClient == nil {
		c.HTTPClient = &http.Client{}
	}

	url := fmt.Sprintf("%s/chat/completions", c.BaseURL)

	messages := []ChatMessage{
		{Role: "system", Content: c.SystemPrompt},
	}

	// Add history (includes the current user message)
	messages = append(messages, history...)

	req := ChatRequest{
		Model:    c.Model,
		Messages: messages,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return "", 0, 0, 0, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "", 0, 0, 0, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	if c.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	}

	resp, err := c.HTTPClient.Do(httpReq)
	if err != nil {
		return "", 0, 0, 0, fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", 0, 0, 0, responseError(resp)
	}

	var chatResp ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return "", 0, 0, 0, fmt.Errorf("decode response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return "", 0, 0, 0, fmt.Errorf("no choices in response")
	}

	content := strings.TrimLeft(chatResp.Choices[0].Message.Content, " \t\n\r")
	return content, chatResp.Usage.PromptTokens, chatResp.Usage.CompletionTokens, chatResp.Usage.TotalTokens, nil
}
