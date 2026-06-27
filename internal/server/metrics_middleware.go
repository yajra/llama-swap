package server

import (
	"bytes"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/mostlygeek/llama-swap/internal/chain"
	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/shared"
	"github.com/tidwall/sjson"
)

var (
	specDecodingLogGrace        = 2500 * time.Millisecond
	specDecodingLogPollInterval = 50 * time.Millisecond
)

// CreateMetricsMiddleware returns middleware that records token metrics for
// model-dispatched POST requests. It resolves the model, tees the response into
// a buffer, and parses token usage once the upstream handler returns.
func CreateMetricsMiddleware(mm *metricsMonitor, cfg config.Config, modelLogHistory modelLogHistoryFunc) chain.Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if mm == nil || r.Method != http.MethodPost {
				next.ServeHTTP(w, r)
				return
			}

			// Determine the model-routed endpoint path. Regular routes are
			// already meterable; /upstream/<model>/<path> is metered only when
			// the remaining path matches a model-dispatched endpoint.
			checkPath := r.URL.Path
			if strings.HasPrefix(r.URL.Path, "/upstream/") {
				var found bool
				_, _, checkPath, found = shared.FindModelInPath(cfg, strings.TrimPrefix(r.URL.Path, "/upstream"))
				if !found {
					next.ServeHTTP(w, r)
					return
				}
			}

			if !isMetricsRecordPath(checkPath) {
				next.ServeHTTP(w, r)
				return
			}

			// Resolve the model now so downstream dispatch hits the context
			// fast path; FetchContext restores the request body for regular
			// routes and extracts the model from the URL for /upstream routes.
			data, err := shared.FetchContext(r, cfg)
			if err != nil {
				shared.SendError(w, r, shared.ErrNoModelInContext)
				return
			}

			if shouldRequestStreamingUsage(checkPath, data, r) {
				if err := requestStreamingUsage(r); err != nil && mm.logger != nil {
					mm.logger.Warnf("metrics: failed to request streaming usage: %v", err)
				}
			}

			// Buffer the request body/headers for capture before dispatch
			// consumes them.
			cf := captureFieldsFor(checkPath)
			var reqBody []byte
			var reqHeaders map[string]string
			if mm.enableCaptures {
				if cf&captureReqBody != 0 && r.Body != nil {
					if buffered, err := io.ReadAll(r.Body); err == nil {
						reqBody = buffered
						r.Body.Close()
						r.Body = io.NopCloser(bytes.NewReader(reqBody))
					}
				}
				if cf&captureReqHeaders != 0 {
					reqHeaders = headerMap(r.Header)
					redactHeaders(reqHeaders)
				}
			}

			// Restrict Accept-Encoding to encodings we can decompress so the
			// buffered response body stays parseable.
			if ae := r.Header.Get("Accept-Encoding"); ae != "" {
				r.Header.Set("Accept-Encoding", filterAcceptEncoding(ae))
			}

			requestStart := time.Now()
			logBefore := getModelLogHistory(modelLogHistory, data.ModelID)
			recorder := newBodyCopier(w)
			next.ServeHTTP(recorder, r)
			logAfter := getModelLogHistory(modelLogHistory, data.ModelID)
			specMetrics, _ := parseSpecDecodingMetrics(logHistoryDelta(logBefore, logAfter))
			if shouldWaitForSpecDecodingLogs(logAfter, specMetrics) {
				recordReq := cloneRequestForMetrics(r)
				recordRecorder := cloneResponseBodyCopier(recorder)
				go func() {
					specMetrics := collectSpecDecodingMetrics(modelLogHistory, data.ModelID, logBefore, specDecodingLogGrace)
					mm.record(data.ModelID, requestStart, recordReq, recordRecorder, cf, reqBody, reqHeaders, specMetrics)
				}()
				return
			}
			mm.record(data.ModelID, requestStart, r, recorder, cf, reqBody, reqHeaders, specMetrics)
		})
	}
}

type modelLogHistoryFunc func(modelID string) []byte

func getModelLogHistory(fn modelLogHistoryFunc, modelID string) []byte {
	if fn == nil {
		return nil
	}
	return fn(modelID)
}

func logHistoryDelta(before, after []byte) []byte {
	if len(after) == 0 {
		return nil
	}
	if len(before) == 0 {
		return after
	}
	if bytes.HasPrefix(after, before) {
		return after[len(before):]
	}
	maxOverlap := min(len(before), len(after))
	for n := maxOverlap; n > 0; n-- {
		if bytes.HasPrefix(after, before[len(before)-n:]) {
			return after[n:]
		}
	}
	return after
}

func shouldWaitForSpecDecodingLogs(logHistory []byte, metrics specDecodingMetrics) bool {
	if specDecodingLogGrace <= 0 || len(logHistory) == 0 {
		return false
	}
	return metrics.DraftedTokens > 0 ||
		bytes.Contains(logHistory, []byte("SpecDecoding metrics:")) ||
		bytes.Contains(logHistory, []byte("SpeculativeConfig")) ||
		bytes.Contains(logHistory, []byte("speculative_config"))
}

func collectSpecDecodingMetrics(modelLogHistory modelLogHistoryFunc, modelID string, logBefore []byte, grace time.Duration) specDecodingMetrics {
	if modelLogHistory == nil || grace <= 0 {
		metrics, _ := parseSpecDecodingMetrics(logHistoryDelta(logBefore, getModelLogHistory(modelLogHistory, modelID)))
		return metrics
	}

	var metrics specDecodingMetrics
	lastHistory := logBefore
	deadline := time.Now().Add(grace)
	for {
		currentHistory := getModelLogHistory(modelLogHistory, modelID)
		if parsed, ok := parseSpecDecodingMetrics(logHistoryDelta(lastHistory, currentHistory)); ok {
			metrics.AcceptedTokens += parsed.AcceptedTokens
			metrics.DraftedTokens += parsed.DraftedTokens
		}
		lastHistory = currentHistory

		remaining := time.Until(deadline)
		if remaining <= 0 {
			return metrics
		}
		sleepFor := specDecodingLogPollInterval
		if sleepFor <= 0 || sleepFor > remaining {
			sleepFor = remaining
		}
		time.Sleep(sleepFor)
	}
}

type metricsResponseWriter struct {
	header http.Header
}

func (w *metricsResponseWriter) Header() http.Header {
	return w.header
}

func (w *metricsResponseWriter) Write(b []byte) (int, error) {
	return len(b), nil
}

func (w *metricsResponseWriter) WriteHeader(statusCode int) {}

func cloneRequestForMetrics(r *http.Request) *http.Request {
	clone := r.Clone(r.Context())
	clone.Header = r.Header.Clone()
	if r.URL != nil {
		urlCopy := *r.URL
		clone.URL = &urlCopy
	}
	return clone
}

func cloneResponseBodyCopier(recorder *responseBodyCopier) *responseBodyCopier {
	return &responseBodyCopier{
		ResponseWriter: &metricsResponseWriter{header: recorder.Header().Clone()},
		body:           bytes.NewBuffer(append([]byte(nil), recorder.body.Bytes()...)),
		status:         recorder.status,
		wroteHeader:    recorder.wroteHeader,
		start:          recorder.start,
	}
}

func shouldRequestStreamingUsage(path string, data shared.ReqContextData, r *http.Request) bool {
	return data.Streaming &&
		path == "/v1/chat/completions" &&
		strings.Contains(r.Header.Get("Content-Type"), "application/json")
}

func requestStreamingUsage(r *http.Request) error {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return err
	}
	r.Body.Close()

	body, err = sjson.SetBytes(body, "stream_options.include_usage", true)
	if err != nil {
		r.Body = io.NopCloser(bytes.NewReader(body))
		return err
	}

	r.Body = io.NopCloser(bytes.NewReader(body))
	r.Header.Del("Transfer-Encoding")
	r.Header.Set("Content-Length", strconv.Itoa(len(body)))
	r.ContentLength = int64(len(body))
	return nil
}
