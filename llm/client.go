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

// Tool is an OpenAI-style tool definition as accepted by llama.cpp's server.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction describes one callable tool.
type ToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// ToolCall is a function call requested by the model.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall carries the tool name and its JSON-encoded arguments.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ChatMessageTool is a chat message that may carry tool calls or a tool
// result. Content is a plain string because llama.cpp's tool templates do not
// accept structured content parts on the tool path.
// Content is always serialized, even when empty: llama.cpp rejects any
// non-assistant message that omits the field entirely, which happens for the
// tool-result turn when a description comes back blank.
type ChatMessageTool struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ChatRequestTools is the request body for the tool-calling chat endpoint.
// Tools and ToolChoice are omitted when unset: dropping Tools entirely is what
// makes a tool-calling model stop calling and answer, see CompleteTools.
type ChatRequestTools struct {
	Model       string            `json:"model"`
	Messages    []ChatMessageTool `json:"messages"`
	Tools       []Tool            `json:"tools,omitempty"`
	ToolChoice  any               `json:"tool_choice,omitempty"`
	MaxTokens   int               `json:"max_tokens,omitempty"`
	Temperature *float64          `json:"temperature,omitempty"`
}

// ToolMessage is the assistant or tool turn on the tool path.
type ToolMessage struct {
	Content      string     `json:"content"`
	ToolCalls    []ToolCall `json:"tool_calls"`
	FinishReason string     `json:"finish_reason"`
}

type toolsChoice struct {
	Message ToolMessage `json:"message"`
}

type toolsResponse struct {
	Choices []toolsChoice `json:"choices"`
	Usage   UsageInfo     `json:"usage"`
}

// CompleteTools sends history to the model with tool support. When tools is
// non-empty the model may answer with tool calls; when tools is empty and
// toolChoice is "none" the server is told to answer in plain text.
//
// llama.cpp's Llama 3.2 build has no native tool template, so the server
// bolts on a PEG parser. Observed behavior that shapes this API: with tools
// offered on a turn that already contains a tool result the model re-issues
// the same call indefinitely and never reads the result, so callers must drop
// Tools (and set tool_choice "none") once the tool budget is spent to get a
// terminating answer. Emitting a second tool call in one turn hard-fails the
// server with HTTP 500, so callers must treat tools as one-per-turn.
func (c *Client) CompleteTools(ctx context.Context, history []ChatMessageTool, tools []Tool, toolChoice any, temperature float64, maxTokens int) (ToolMessage, UsageInfo, error) {
	if c.HTTPClient == nil {
		c.HTTPClient = &http.Client{}
	}

	url := fmt.Sprintf("%s/chat/completions", c.BaseURL)

	messages := make([]ChatMessageTool, 0, len(history)+1)
	if c.SystemPrompt != "" {
		messages = append(messages, ChatMessageTool{Role: "system", Content: c.SystemPrompt})
	}
	messages = append(messages, history...)

	temp := temperature
	req := ChatRequestTools{
		Model:       c.Model,
		Messages:    messages,
		Tools:       tools,
		ToolChoice:  toolChoice,
		MaxTokens:   maxTokens,
		Temperature: &temp,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return ToolMessage{}, UsageInfo{}, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return ToolMessage{}, UsageInfo{}, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	}

	resp, err := c.HTTPClient.Do(httpReq)
	if err != nil {
		return ToolMessage{}, UsageInfo{}, fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return ToolMessage{}, UsageInfo{}, responseError(resp)
	}

	var parsed toolsResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return ToolMessage{}, UsageInfo{}, fmt.Errorf("decode response: %w", err)
	}
	if len(parsed.Choices) == 0 {
		return ToolMessage{}, UsageInfo{}, fmt.Errorf("no choices in response")
	}
	msg := parsed.Choices[0].Message
	msg.Content = strings.TrimLeft(msg.Content, " \t\n\r")
	return msg, parsed.Usage, nil
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
