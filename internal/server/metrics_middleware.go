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
	return after
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
