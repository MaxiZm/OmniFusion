#!/usr/bin/env node
/**
 * OpenAI Node SDK compatibility smoke against a running OmniFusion instance.
 *
 * Opt-in / live only. Skips cleanly (exit 0) when the endpoint env vars are
 * unset, so it is safe to wire into a Makefile. Asserts wire compatibility — the
 * canonical OpenAI response shape and streaming — never any quality claim.
 *
 *   npm install openai
 *   export OMNIFUSION_BASE_URL=http://127.0.0.1:8000/v1
 *   export OMNIFUSION_API_KEY=your-omnifusion-client-key
 *   node scripts/compat/openai_node.mjs
 */

const baseURL = process.env.OMNIFUSION_BASE_URL;
const apiKey = process.env.OMNIFUSION_API_KEY;
const model = process.env.OMNIFUSION_MODEL || "fusion/general";

async function main() {
  if (!baseURL || !apiKey) {
    console.log(
      "[skip] OMNIFUSION_BASE_URL / OMNIFUSION_API_KEY not set; " +
        "this is an opt-in live smoke. Nothing to do."
    );
    return 0;
  }

  let OpenAI;
  try {
    ({ default: OpenAI } = await import("openai"));
  } catch {
    console.log("[skip] openai package not installed (npm install openai).");
    return 0;
  }

  const client = new OpenAI({ baseURL, apiKey });

  // 1. Non-streaming completion + run-id header via raw response.
  const { data: completion, response } = await client.chat.completions
    .create({
      model,
      messages: [{ role: "user", content: "Reply with the single word: pong." }],
      max_tokens: 64,
    })
    .withResponse();

  const runId = response.headers.get("x-omnifusion-run-id");
  if (!runId) throw new Error("missing X-OmniFusion-Run-Id header");
  if (!completion.choices || completion.choices.length === 0)
    throw new Error("no choices returned");
  const content = completion.choices[0].message.content ?? "";
  console.log(`[ok] non-stream  model=${completion.model} run_id=${runId}`);
  console.log(`     content: ${JSON.stringify(content.slice(0, 80))}`);

  // 2. Streaming completion.
  const stream = await client.chat.completions.create({
    model,
    messages: [{ role: "user", content: "Count: one two three." }],
    max_tokens: 64,
    stream: true,
    stream_options: { include_usage: true },
  });
  let chunks = 0;
  let streamed = "";
  for await (const event of stream) {
    chunks += 1;
    const delta = event.choices?.[0]?.delta?.content;
    if (delta) streamed += delta;
  }
  if (chunks === 0) throw new Error("stream produced no chunks");
  console.log(`[ok] streaming   chunks=${chunks} content=${JSON.stringify(streamed.slice(0, 80))}`);

  console.log("\nOpenAI Node SDK compatibility smoke passed.");
  return 0;
}

main()
  .then((code) => process.exit(code))
  .catch((err) => {
    console.error(`[FAIL] ${err.message}`);
    process.exit(1);
  });
