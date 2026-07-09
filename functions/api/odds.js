// Cloudflare Pages Function — proxies The Odds API server-side.
// Deployed automatically by Cloudflare Pages when this file sits at
// /functions/api/odds.js in the repo root. No build step needed.
//
// The API key is never stored here — it's passed through on each request
// from the browser (?apiKey=...) and forwarded straight to The Odds API.
// This function only exists to make the outbound call from Cloudflare's
// edge (server-to-server, no CORS involved) instead of from the browser
// (which The Odds API blocks with a CORS error).

export async function onRequestGet(context) {
  const { request } = context;
  const url = new URL(request.url);

  const apiKey = url.searchParams.get('apiKey');
  if (!apiKey) {
    return new Response(JSON.stringify({ error: 'missing apiKey' }), {
      status: 400,
      headers: { 'content-type': 'application/json' },
    });
  }

  const sport = url.searchParams.get('sport') || 'soccer_fifa_world_cup';
  const regions = url.searchParams.get('regions') || 'us';
  const markets = url.searchParams.get('markets') || 'h2h,spreads,totals';
  const oddsFormat = url.searchParams.get('oddsFormat') || 'american';

  const target =
    `https://api.the-odds-api.com/v4/sports/${encodeURIComponent(sport)}/odds` +
    `?regions=${encodeURIComponent(regions)}` +
    `&markets=${encodeURIComponent(markets)}` +
    `&oddsFormat=${encodeURIComponent(oddsFormat)}` +
    `&apiKey=${encodeURIComponent(apiKey)}`;

  let upstream;
  try {
    upstream = await fetch(target);
  } catch (err) {
    return new Response(JSON.stringify({ error: 'upstream fetch failed', detail: String(err) }), {
      status: 502,
      headers: { 'content-type': 'application/json' },
    });
  }

  const body = await upstream.text();

  const headers = { 'content-type': 'application/json' };
  const remaining = upstream.headers.get('x-requests-remaining');
  const used = upstream.headers.get('x-requests-used');
  if (remaining !== null) headers['x-requests-remaining'] = remaining;
  if (used !== null) headers['x-requests-used'] = used;

  return new Response(body, { status: upstream.status, headers });
}
