const UPSTREAM_ORIGIN = "https://www.tpex.org.tw";
const PROXY_VERSION = "1";

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

function jsonError(status, code) {
  return new Response(JSON.stringify({ error: code }), {
    status,
    headers: {
      "Cache-Control": "no-store",
      "Content-Type": "application/json; charset=utf-8",
      "X-Content-Type-Options": "nosniff",
      "X-TPEX-Proxy-Version": PROXY_VERSION,
    },
  });
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
  const suppliedToken = request.headers.get("X-TPEX-Proxy-Token") ?? "";
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

export async function handleRequest(request, env, fetchImpl = fetch) {
  if (request.method !== "GET") {
    return jsonError(405, "method_not_allowed");
  }
  if (!authorized(request, env.TPEX_PROXY_SHARED_SECRET)) {
    return jsonError(401, "unauthorized");
  }

  const upstreamUrl = validatedUpstreamUrl(new URL(request.url));
  if (upstreamUrl === null) {
    return jsonError(400, "unsupported_tpex_request");
  }

  let upstreamResponse;
  try {
    upstreamResponse = await fetchImpl(upstreamUrl, {
      method: "GET",
      headers: {
        Accept: "application/json,text/plain,*/*",
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        Referer: "https://www.tpex.org.tw/",
        "User-Agent":
          "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 " +
          "(KHTML, like Gecko) Chrome/140.0 Safari/537.36",
      },
      redirect: "manual",
    });
  } catch (_error) {
    return jsonError(502, "tpex_upstream_unreachable");
  }

  if (upstreamResponse.status >= 300 && upstreamResponse.status < 400) {
    return jsonError(502, "tpex_upstream_redirect_rejected");
  }

  const responseHeaders = new Headers({
    "Cache-Control": "no-store",
    "Content-Type":
      upstreamResponse.headers.get("Content-Type") ?? "application/json; charset=utf-8",
    "X-Content-Type-Options": "nosniff",
    "X-TPEX-Proxy-Version": PROXY_VERSION,
  });
  return new Response(upstreamResponse.body, {
    status: upstreamResponse.status,
    statusText: upstreamResponse.statusText,
    headers: responseHeaders,
  });
}

export default {
  fetch(request, env) {
    return handleRequest(request, env);
  },
};
