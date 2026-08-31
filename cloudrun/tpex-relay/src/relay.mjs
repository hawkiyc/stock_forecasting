const UPSTREAM_ORIGIN = "https://www.tpex.org.tw";
const RELAY_VERSION = "1";
const AUTHENTICATION_HEADER = "X-TPEX-Relay-Token";
const MAX_UPSTREAM_REDIRECTS = 3;
const MAX_UPSTREAM_REDIRECT_COOKIES = 8;
const MAX_UPSTREAM_COOKIE_HEADER_BYTES = 4096;
const MAX_UPSTREAM_BODY_BYTES = 16 * 1024 * 1024;
const UPSTREAM_TIMEOUT_MS = 30_000;

const ROUTES = Object.freeze({
  "/www/zh-tw/afterTrading/dailyQuotes": Object.freeze({
    required: Object.freeze(["date", "id", "response"]),
    dateParameters: Object.freeze(["date"]),
  }),
  "/www/zh-tw/bulletin/exDailyQ": Object.freeze({
    required: Object.freeze(["startDate", "endDate", "response"]),
    dateParameters: Object.freeze(["startDate", "endDate"]),
  }),
  "/www/zh-tw/indexInfo/ROE": Object.freeze({
    required: Object.freeze(["date", "response"]),
    dateParameters: Object.freeze(["date"]),
  }),
  "/www/zh-tw/indexInfo/inx": Object.freeze({
    required: Object.freeze(["date", "response"]),
    dateParameters: Object.freeze(["date"]),
  }),
});

const DATE_PATTERN = /^[0-9]{4}\/[0-9]{2}\/[0-9]{2}$/;
const COOKIE_NAME_PATTERN = /^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/;
const METADATA_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$/;

function validCalendarDate(value) {
  if (!DATE_PATTERN.test(value)) {
    return false;
  }
  const [year, month, day] = value.split("/").map(Number);
  const parsed = new Date(Date.UTC(year, month - 1, day));
  return (
    parsed.getUTCFullYear() === year &&
    parsed.getUTCMonth() === month - 1 &&
    parsed.getUTCDate() === day
  );
}

function safeMetadata(value, fallback) {
  return typeof value === "string" && METADATA_PATTERN.test(value) ? value : fallback;
}

function relayMetadata(env) {
  return Object.freeze({
    version: RELAY_VERSION,
    region: safeMetadata(env.TPEX_RELAY_REGION, "unknown"),
    revision: safeMetadata(env.K_REVISION, "local"),
  });
}

function responseHeaders(env, contentType, upstreamResult = {}) {
  const metadata = relayMetadata(env);
  const headers = new Headers({
    "Cache-Control": "no-store",
    "Content-Type": contentType,
    "X-Content-Type-Options": "nosniff",
    "X-TPEX-Relay-Version": metadata.version,
    "X-TPEX-Relay-Region": metadata.region,
    "X-TPEX-Relay-Revision": metadata.revision,
  });
  if (Number.isInteger(upstreamResult.redirectCount)) {
    headers.set("X-TPEX-Upstream-Redirects", String(upstreamResult.redirectCount));
  }
  if (Number.isInteger(upstreamResult.redirectCookieCount)) {
    headers.set(
      "X-TPEX-Upstream-Redirect-Cookies",
      String(upstreamResult.redirectCookieCount),
    );
  }
  return headers;
}

function jsonResponse(env, status, payload, upstreamResult = {}) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: responseHeaders(
      env,
      "application/json; charset=utf-8",
      upstreamResult,
    ),
  });
}

function jsonError(env, status, code, upstreamResult = {}) {
  return jsonResponse(env, status, { error: code }, upstreamResult);
}

function constantTimeEqual(left, right) {
  const leftBytes = new TextEncoder().encode(left);
  const rightBytes = new TextEncoder().encode(right);
  const length = Math.max(leftBytes.length, rightBytes.length);
  let difference = leftBytes.length ^ rightBytes.length;
  for (let index = 0; index < length; index += 1) {
    difference |= (leftBytes[index] ?? 0) ^ (rightBytes[index] ?? 0);
  }
  return difference === 0;
}

function authorized(request, expectedToken) {
  if (
    typeof expectedToken !== "string" ||
    expectedToken.length < 32 ||
    expectedToken.length > 512
  ) {
    return false;
  }
  const suppliedToken = request.headers.get(AUTHENTICATION_HEADER) ?? "";
  if (suppliedToken.length > 512) {
    return false;
  }
  return constantTimeEqual(suppliedToken, expectedToken);
}

function validatedUpstreamUrl(requestUrl) {
  const route = ROUTES[requestUrl.pathname];
  if (route === undefined) {
    return null;
  }

  const suppliedNames = [...new Set(requestUrl.searchParams.keys())].sort();
  const requiredNames = [...route.required].sort();
  if (
    suppliedNames.length !== requiredNames.length ||
    suppliedNames.some((name, index) => name !== requiredNames[index])
  ) {
    return null;
  }
  if (requiredNames.some((name) => requestUrl.searchParams.getAll(name).length !== 1)) {
    return null;
  }
  if (requestUrl.searchParams.get("response") !== "json") {
    return null;
  }
  if (
    route.dateParameters.some(
      (name) => !validCalendarDate(requestUrl.searchParams.get(name) ?? ""),
    )
  ) {
    return null;
  }
  if (
    requestUrl.pathname === "/www/zh-tw/bulletin/exDailyQ" &&
    requestUrl.searchParams.get("startDate") > requestUrl.searchParams.get("endDate")
  ) {
    return null;
  }
  if (
    requestUrl.pathname === "/www/zh-tw/afterTrading/dailyQuotes" &&
    requestUrl.searchParams.get("id") !== ""
  ) {
    return null;
  }

  const upstream = new URL(requestUrl.pathname, UPSTREAM_ORIGIN);
  for (const name of route.required) {
    upstream.searchParams.set(name, requestUrl.searchParams.get(name));
  }
  return upstream;
}

function validatedRedirectUrl(currentUrl, location) {
  if (typeof location !== "string" || location.length === 0) {
    return null;
  }
  let redirected;
  try {
    redirected = new URL(location, currentUrl);
  } catch (_error) {
    return null;
  }
  if (
    redirected.origin !== UPSTREAM_ORIGIN ||
    redirected.username !== "" ||
    redirected.password !== "" ||
    redirected.hash !== ""
  ) {
    return null;
  }
  return redirected;
}

function redirectSetCookieValues(headers) {
  if (typeof headers.getSetCookie === "function") {
    return headers.getSetCookie();
  }
  const value = headers.get("Set-Cookie");
  return value === null ? [] : [value];
}

function parsedCookiePair(rawValue) {
  if (typeof rawValue !== "string") {
    return null;
  }
  const pair = rawValue.split(";", 1)[0].trim();
  const separatorIndex = pair.indexOf("=");
  if (separatorIndex <= 0) {
    return null;
  }
  const name = pair.slice(0, separatorIndex).trim();
  const value = pair.slice(separatorIndex + 1).trim();
  if (!COOKIE_NAME_PATTERN.test(name) || /[\u0000-\u001f\u007f;,]/.test(value)) {
    return null;
  }
  return { name, pair: `${name}=${value}` };
}

function captureRedirectCookies(response, cookieJar) {
  for (const rawValue of redirectSetCookieValues(response.headers)) {
    const parsed = parsedCookiePair(rawValue);
    if (parsed === null) {
      continue;
    }
    const candidate = new Map(cookieJar);
    candidate.set(parsed.name, parsed.pair);
    const headerValue = [...candidate.values()].join("; ");
    if (
      candidate.size > MAX_UPSTREAM_REDIRECT_COOKIES ||
      new TextEncoder().encode(headerValue).length > MAX_UPSTREAM_COOKIE_HEADER_BYTES
    ) {
      return false;
    }
    cookieJar.set(parsed.name, parsed.pair);
  }
  return true;
}

function upstreamRequestHeaders(cookieJar) {
  const headers = new Headers({
    Accept: "application/json,text/plain,*/*",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    Referer: "https://www.tpex.org.tw/",
    "User-Agent":
      "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 " +
      "(KHTML, like Gecko) Chrome/140.0 Safari/537.36",
  });
  if (cookieJar.size > 0) {
    headers.set("Cookie", [...cookieJar.values()].join("; "));
  }
  return headers;
}

function upstreamStateKey(url, cookieJar) {
  return `${url.toString()}\n${[...cookieJar.values()].join("; ")}`;
}

async function fetchUpstream(initialUrl, fetchImpl, signal) {
  let currentUrl = initialUrl;
  const cookieJar = new Map();
  const visitedStates = new Set([upstreamStateKey(currentUrl, cookieJar)]);
  for (let redirectCount = 0; redirectCount <= MAX_UPSTREAM_REDIRECTS; redirectCount += 1) {
    let response;
    try {
      response = await fetchImpl(currentUrl, {
        method: "GET",
        headers: upstreamRequestHeaders(cookieJar),
        cache: "no-store",
        redirect: "manual",
        signal,
      });
    } catch (_error) {
      return {
        error: "tpex_upstream_unreachable",
        redirectCount,
        redirectCookieCount: cookieJar.size,
      };
    }

    if (response.status < 300 || response.status >= 400) {
      return { response, redirectCount, redirectCookieCount: cookieJar.size };
    }
    if (redirectCount === MAX_UPSTREAM_REDIRECTS) {
      return {
        error: "tpex_upstream_redirect_limit_exceeded",
        redirectCount,
        redirectCookieCount: cookieJar.size,
      };
    }

    const redirected = validatedRedirectUrl(currentUrl, response.headers.get("Location"));
    if (redirected === null) {
      return {
        error: "tpex_upstream_redirect_rejected",
        redirectCount,
        redirectCookieCount: cookieJar.size,
      };
    }
    if (!captureRedirectCookies(response, cookieJar)) {
      return {
        error: "tpex_upstream_redirect_cookie_limit_exceeded",
        redirectCount,
        redirectCookieCount: cookieJar.size,
      };
    }
    currentUrl = redirected;
    const redirectedState = upstreamStateKey(currentUrl, cookieJar);
    if (visitedStates.has(redirectedState)) {
      return {
        error: "tpex_upstream_redirect_loop",
        redirectCount: redirectCount + 1,
        redirectCookieCount: cookieJar.size,
      };
    }
    visitedStates.add(redirectedState);
  }
  return {
    error: "tpex_upstream_redirect_limit_exceeded",
    redirectCount: MAX_UPSTREAM_REDIRECTS,
    redirectCookieCount: cookieJar.size,
  };
}

async function readBoundedBody(response) {
  const rawLength = response.headers.get("Content-Length");
  if (rawLength !== null && /^[0-9]+$/.test(rawLength)) {
    if (Number(rawLength) > MAX_UPSTREAM_BODY_BYTES) {
      try {
        await response.body?.cancel("upstream body exceeds relay limit");
      } catch (_error) {
        // The response is rejected even when stream cancellation has already occurred.
      }
      return null;
    }
  }
  if (response.body === null) {
    return new Uint8Array();
  }

  const reader = response.body.getReader();
  const chunks = [];
  let totalBytes = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    totalBytes += value.byteLength;
    if (totalBytes > MAX_UPSTREAM_BODY_BYTES) {
      await reader.cancel("upstream body exceeds relay limit");
      return null;
    }
    chunks.push(value);
  }

  const body = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return body;
}

function validJsonBody(body) {
  try {
    const parsed = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(body));
    return parsed !== null && typeof parsed === "object";
  } catch (_error) {
    return false;
  }
}

function safeContentType(value) {
  if (
    typeof value === "string" &&
    value.length <= 256 &&
    !/[\r\n\0]/.test(value)
  ) {
    return value;
  }
  return "application/json; charset=utf-8";
}

export async function handleRequest(
  request,
  env,
  fetchImpl = fetch,
  timeoutSignalFactory = () => AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
) {
  if (request.method !== "GET") {
    return jsonError(env, 405, "method_not_allowed");
  }
  if (!authorized(request, env.TPEX_PROXY_SHARED_SECRET)) {
    return jsonError(env, 401, "unauthorized");
  }

  const requestUrl = new URL(request.url);
  if (requestUrl.pathname === "/_internal/warmup") {
    if (requestUrl.search !== "") {
      return jsonError(env, 400, "unsupported_warmup_request");
    }
    return jsonResponse(env, 200, { ready: true, ...relayMetadata(env) });
  }

  const upstreamUrl = validatedUpstreamUrl(requestUrl);
  if (upstreamUrl === null) {
    return jsonError(env, 400, "unsupported_tpex_request");
  }

  const upstreamResult = await fetchUpstream(
    upstreamUrl,
    fetchImpl,
    timeoutSignalFactory(),
  );
  if (upstreamResult.error !== undefined) {
    return jsonError(env, 502, upstreamResult.error, upstreamResult);
  }

  let body;
  try {
    body = await readBoundedBody(upstreamResult.response);
  } catch (_error) {
    return jsonError(env, 502, "tpex_upstream_body_unreadable", upstreamResult);
  }
  if (body === null) {
    return jsonError(env, 502, "tpex_upstream_body_too_large", upstreamResult);
  }
  if (
    upstreamResult.response.status >= 200 &&
    upstreamResult.response.status < 300 &&
    !validJsonBody(body)
  ) {
    return jsonError(env, 502, "tpex_upstream_invalid_json", upstreamResult);
  }

  return new Response(body, {
    status: upstreamResult.response.status,
    headers: responseHeaders(
      env,
      safeContentType(upstreamResult.response.headers.get("Content-Type")),
      upstreamResult,
    ),
  });
}

export const relayContract = Object.freeze({
  authenticationHeader: AUTHENTICATION_HEADER,
  maxUpstreamBodyBytes: MAX_UPSTREAM_BODY_BYTES,
  maxUpstreamRedirects: MAX_UPSTREAM_REDIRECTS,
  upstreamOrigin: UPSTREAM_ORIGIN,
  upstreamTimeoutMilliseconds: UPSTREAM_TIMEOUT_MS,
  version: RELAY_VERSION,
});
