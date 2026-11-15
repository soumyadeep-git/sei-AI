# Wise "Where is my money?" voice agent

A phone support agent that answers inbound calls about Wise's
["Where is my money?"](https://wise.com/help/topics/5bVKT0uQdBrDp6T62keyfz/sending-money)
FAQs and hands everything else to a human (here: a handoff line, then hang up).

```
caller ── PSTN ── Twilio ──(media stream, websocket)── bot.py
                                                         │
          Silero VAD → Deepgram STT → Groq LLM → Deepgram TTS
                                         │
                       tools: transfer_to_human, end_call
```

Built on [Pipecat](https://github.com/pipecat-ai/pipecat). Every service runs on a free tier.

| Concern   | Choice                          | Why |
|-----------|---------------------------------|-----|
| Telephony | Twilio (trial)                  | Free number; Pipecat ships a Twilio serializer |
| STT / TTS | Deepgram Nova / Aura-2          | Low-latency streaming, one free account for both |
| LLM       | Groq, `openai/gpt-oss-120b`     | Fast first token, reliable tool calling, free tier |
| Hosting   | Render (Docker, free)           | Public HTTPS/WSS endpoint for Twilio |

```
bot.py                          prompt, tools and pipeline (the whole agent)
knowledge/where_is_my_money.md  the six Wise articles, the agent's only source of truth
evals/                          behavioural scenarios + judge LLM
tests/                          deterministic tests for call handling
```

## Design

**Grounding: the whole knowledge base goes in the prompt, not RAG.** The six
articles are ~2k tokens. At that size, retrieval only adds latency and a way
to miss the right chunk. The prompt tells the model to answer only from that
text. Retrieval is the next step once the corpus outgrows the context budget.

**Deflection is a tool call in the same LLM pass.** A separate intent
classifier would add a model round trip to every turn. Instead the model
decides in-scope vs out-of-scope with the full conversation in view, and
calls `transfer_to_human` when the knowledge base doesn't answer. The default
is to deflect: no partial answers. That covers other Wise topics, anything
needing the caller's account, unrelated questions, and attempts to change the
bot's role.

**Every call ends the same way.** Deflection, goodbye and silence all go
through `closing_frames`: a fixed line, marked uninterruptible so barge-in
can't cut it off, followed by an end frame. The pipeline flushes the speech
before shutting down, then the Twilio serializer hangs up through the REST
API. The lines are fixed text, so the bot can't promise anything a human
then has to honour. In production, `transfer_to_human` would redirect the call
to a `<Dial>` queue instead of hanging up.

**Silence is handled.** After 10 seconds of silence the bot checks once
whether the caller is still there. A second silence ends the call, so an open
line doesn't hold a session or burn quota.

**Only Twilio can start a call.** The media stream websocket is the only
thing that costs API quota, so the bot checks Twilio's `X-Twilio-Signature`
on the handshake and closes any connection without a valid one.

## Run locally

```bash
uv sync
cp .env.example .env    # add DEEPGRAM_API_KEY and GROQ_API_KEY
uv run bot.py           # talk to it in the browser at http://localhost:7860
```

## Quality checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest           # signature check, silence handling, how calls end
```

Behaviour is tested with Pipecat's eval harness. Scripted text scenarios run
headless against the live bot:

```bash
uv run bot.py                                  # terminal 1
PYTHONPATH=. uv run pipecat eval run evals/ -v # terminal 2
```

`in_scope.yaml` has one question per article, which the bot must answer
without deflecting, plus a multi-turn call that ends with `end_call`.
`out_of_scope.yaml` covers fees, exchange rates, cards, account actions,
off-topic requests and prompt injection, which must all go to
`transfer_to_human`. Tool calls are checked deterministically. Answer content
is checked by a judge LLM on Groq (`qwen/qwen3.8-27b`), a different model
family from the bot so the bot doesn't grade its own answers.

CI (`.github/workflows/ci.yml`) runs lint, type checks and tests on every push.
The evals use live APIs, so they run on manual dispatch, using repository
secrets.

## Deploy

1. Push the repo to GitHub, then in Render choose **New → Blueprint** and
   select it. `render.yaml` sets up a free Docker web service. Fill in the four
   secrets when prompted.
2. In the Twilio console, open your number → **Voice configuration** →
   "A call comes in": **Webhook**, `https://<service>.onrender.com/`, **HTTP POST**.
3. Call the number. (Trial accounts play a short Twilio notice first; press
   any key.)

To run anywhere else: `docker build -t wise-agent . && docker run -p 7860:7860
--env-file .env -e PUBLIC_HOST=<public-hostname> wise-agent`. For local
telephony testing, `ngrok http 7860` gives you a hostname.

## Known limitations

- **Cold starts.** Render's free tier sleeps after 15 idle minutes and takes
  about a minute to wake. Twilio stops waiting for the webhook after 15
  seconds, so the first call after a quiet spell can fail. Open `/status`
  before a demo, or keep the service awake with a free uptime pinger.
- **Caller transcripts in logs.** Pipecat's runner logs at debug level, which
  includes what the caller says. Production would need log redaction or a
  higher log level.
- **Trial telephony.** A Twilio trial number only takes domestic calls, and
  Twilio caps trial calls at 10 minutes.
- English only, and a single knowledge source.
