package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestServer_InflightMiddleware(t *testing.T) {
	c := &inflightCounter{}
	mw := CreateInflightMiddleware(c)

	var duringRequest int64
	handler := mw(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		duringRequest = c.Current()
	}))

	handler.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil))

	if duringRequest != 1 {
		t.Errorf("counter during request = %d, want 1", duringRequest)
	}
	if got := c.Current(); got != 0 {
		t.Errorf("counter after request = %d, want 0", got)
	}
}

func TestServer_APIVersion(t *testing.T) {
	s := newTestServer(newStubRouter(nil, ""), newStubRouter(nil, ""))
	s.build = BuildInfo{Version: "1.2.3", Commit: "deadbeef", Date: "2026-05-19"}

	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/version", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d", w.Code)
	}
	var got map[string]string
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got["version"] != "1.2.3" || got["commit"] != "deadbeef" || got["build_date"] != "2026-05-19" {
		t.Errorf("body = %v", got)
	}
}

func TestServer_APIMetrics_Empty(t *testing.T) {
	s := newTestServer(newStubRouter(nil, ""), newStubRouter(nil, ""))

	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/metrics", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d", w.Code)
	}
	if body := strings.TrimSpace(w.Body.String()); body != "[]" {
		t.Errorf("body = %q, want []", body)
	}
}

func TestServer_APIPerformance_Unavailable(t *testing.T) {
	s := newTestServer(newStubRouter(nil, ""), newStubRouter(nil, ""))

	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/performance", nil))

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", w.Code)
	}
}

func TestServer_APIEvents_InitialPayload(t *testing.T) {
	s := newTestServer(newStubRouter(nil, ""), newStubRouter(nil, ""))

	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest(http.MethodGet, "/api/events", nil).WithContext(ctx)
	w := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		s.ServeHTTP(w, req)
		close(done)
	}()

	time.Sleep(100 * time.Millisecond)
	cancel()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("handler did not return after context cancel")
	}

	body := w.Body.String()
	for _, want := range []string{`"type":"modelStatus"`, `"type":"inflight"`, `"type":"logData"`} {
		if !strings.Contains(body, want) {
			t.Errorf("initial SSE payload missing %s; body=%q", want, body)
		}
	}
}

func TestServer_APIEvents_SendSkipsCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	sendBuffer := make(chan messageEnvelope, 1)
	sender := newAPIEventSender(ctx, sendBuffer)

	cancel()
	sender.send(messageEnvelope{Type: msgTypeInFlight, Data: `{"total":1}`})

	if got := len(sendBuffer); got != 0 {
		t.Errorf("sendBuffer length = %d, want 0", got)
	}
	if got := sender.droppedCount(); got != 0 {
		t.Errorf("droppedCount = %d, want 0", got)
	}
}

func TestServer_APIEvents_SendCountsFullBuffer(t *testing.T) {
	ctx := context.Background()
	sendBuffer := make(chan messageEnvelope, 1)
	sender := newAPIEventSender(ctx, sendBuffer)
	original := messageEnvelope{Type: msgTypeModelStatus, Data: "[]"}
	sendBuffer <- original

	sender.send(messageEnvelope{Type: msgTypeInFlight, Data: `{"total":1}`})

	if got := sender.droppedCount(); got != 1 {
		t.Errorf("droppedCount = %d, want 1", got)
	}
	select {
	case got := <-sendBuffer:
		if got != original {
			t.Errorf("buffer message = %+v, want %+v", got, original)
		}
	default:
		t.Fatal("sendBuffer is empty, want original message preserved")
	}
}
