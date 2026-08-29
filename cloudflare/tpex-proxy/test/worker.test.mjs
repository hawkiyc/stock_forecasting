import assert from "node:assert/strict";
import test from "node:test";

import { handleRequest } from "../src/index.mjs";

const TOKEN = "fixture-token-that-is-longer-than-thirty-two-characters";
const ACTION_URL =
  "https://proxy.example/www/zh-tw/bulletin/exDailyQ" +
  "?startDate=2026%2F01%2F01&endDate=2026%2F01%2F31&response=json";

function request(url = ACTION_URL, token = TOKEN) {
  return new Request(url, { headers: { "X-TPEX-Proxy-Token": token } });
}

test("forwards only the fixed TPEx origin and approved query", async () => {
  let capturedUrl;
  let capturedOptions;
  const response = await handleRequest(request(), { TPEX_PROXY_SHARED_SECRET: TOKEN }, async (
    url,
    options,
  ) => {
    capturedUrl = url;
    capturedOptions = options;
    return new Response(JSON.stringify({ tables: [] }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  });

  assert.equal(response.status, 200);
  assert.equal(capturedUrl.origin, "https://www.tpex.org.tw");
  assert.equal(capturedUrl.pathname, "/www/zh-tw/bulletin/exDailyQ");
  assert.equal(capturedUrl.searchParams.get("startDate"), "2026/01/01");
  assert.equal(capturedOptions.redirect, "manual");
  assert.equal(response.headers.get("Cache-Control"), "no-store");
});

test("follows a bounded same-origin TPEx redirect", async () => {
  const fetchedUrls = [];
  const response = await handleRequest(
    request(),
    { TPEX_PROXY_SHARED_SECRET: TOKEN },
    async (url, options) => {
      fetchedUrls.push(url.toString());
      assert.equal(options.redirect, "manual");
      if (fetchedUrls.length === 1) {
        return new Response(null, {
          status: 302,
          headers: { Location: url.toString() },
        });
      }
      return new Response(JSON.stringify({ tables: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    },
  );

  assert.equal(response.status, 200);
  assert.equal(fetchedUrls.length, 2);
  assert.equal(fetchedUrls[1], fetchedUrls[0]);
});

test("rejects cross-origin and unbounded TPEx redirects", async () => {
  const crossOriginResponse = await handleRequest(
    request(),
    { TPEX_PROXY_SHARED_SECRET: TOKEN },
    async () =>
      new Response(null, {
        status: 302,
        headers: { Location: "https://example.com/not-tpex" },
      }),
  );
  assert.equal(crossOriginResponse.status, 502);
  assert.deepEqual(await crossOriginResponse.json(), {
    error: "tpex_upstream_redirect_rejected",
  });

  let redirectCalls = 0;
  const loopingResponse = await handleRequest(
    request(),
    { TPEX_PROXY_SHARED_SECRET: TOKEN },
    async (url) => {
      redirectCalls += 1;
      return new Response(null, {
        status: 302,
        headers: { Location: url.toString() },
      });
    },
  );
  assert.equal(loopingResponse.status, 502);
  assert.deepEqual(await loopingResponse.json(), {
    error: "tpex_upstream_redirect_limit_exceeded",
  });
  assert.equal(redirectCalls, 4);
});

test("rejects missing authentication without reaching TPEx", async () => {
  let fetched = false;
  const response = await handleRequest(
    request(ACTION_URL, "wrong-token-that-is-also-longer-than-thirty-two"),
    { TPEX_PROXY_SHARED_SECRET: TOKEN },
    async () => {
      fetched = true;
      return new Response("{}");
    },
  );

  assert.equal(response.status, 401);
  assert.equal(fetched, false);
});

test("rejects unapproved paths, parameters, duplicate values, and methods", async () => {
  const cases = [
    request("https://proxy.example/arbitrary?response=json"),
    request(`${ACTION_URL}&target=https%3A%2F%2Fevil.example`),
    request(`${ACTION_URL}&response=json`),
    request(
      "https://proxy.example/www/zh-tw/bulletin/exDailyQ" +
        "?startDate=2026%2F02%2F31&endDate=2026%2F03%2F01&response=json",
    ),
    new Request(ACTION_URL, {
      method: "POST",
      headers: { "X-TPEX-Proxy-Token": TOKEN },
    }),
  ];

  for (const candidate of cases) {
    const response = await handleRequest(candidate, { TPEX_PROXY_SHARED_SECRET: TOKEN });
    assert.ok([400, 405].includes(response.status));
  }
});
