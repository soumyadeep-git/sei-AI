# Wise "Where is my money?" voice agent

A phone support agent for Wise. It answers questions from the six
["Where is my money?"](https://wise.com/help/topics/5bVKT0uQdBrDp6T62keyfz/sending-money)
help articles and hands everything else to a human, which here means a short
handoff line and then hanging up.

Try it at [wise-voice-agent.onrender.com](https://wise-voice-agent.onrender.com)
in the browser, or call +1 (424) 541-1576 from the US. The page can also have
the agent call your phone.

## How it works

```
caller -> Twilio -> media stream websocket -> bot.py
browser -> /ws-client websocket ------------> bot.py

bot.py: Silero VAD -> Deepgram STT -> Groq LLM -> Deepgram TTS
        tools: transfer_to_human, end_call
```

Built on [Pipecat](https://github.com/pipecat-ai/pipecat). Speech is Deepgram
(Nova-3 in, Aura-2 out), the LLM is `openai/gpt-oss-120b` on Groq, telephony is
a Twilio trial number and it's hosted on Render. Everything runs on free tiers.

```
bot.py                          prompt, tools and pipeline
knowledge/where_is_my_money.md  the six articles, the only source of truth
web/                            landing page and browser demo
evals/                          scenario evals with an LLM judge
tests/                          unit tests for call handling
```

## Design choices

The six articles are about 2k tokens, so they go straight into the system
prompt. Retrieval would add latency and a chance of missing the right chunk
without any benefit at this size.

Deciding what's in scope happens in the same LLM call that answers, through the
`transfer_to_human` tool. A separate classifier would add a round trip to every
turn. When the articles don't cover a question the bot hands off rather than
giving a partial answer. That includes other Wise topics, anything that needs
the caller's account, off-topic questions and prompt injection.

Handoff, goodbye and silence all end the call the same way: a fixed closing
line that can't be interrupted, then an end frame, after which Twilio hangs up.
Fixed lines mean the bot can't promise something a human then has to honour.
In production `transfer_to_human` would `<Dial>` a support queue instead.

After 10 seconds of silence the bot asks if the caller is still there. A second
silence ends the call.

The media stream handshake must carry a valid `X-Twilio-Signature`, so only
Twilio can open a phone session. The browser demo takes one caller at a time
for up to three minutes, and the "Call me" form allows one call a minute.

## Run locally

```bash
uv sync
cp .env.example .env    # add DEEPGRAM_API_KEY and GROQ_API_KEY
uv run bot.py           # http://localhost:7860
```

## Tests and evals

```bash
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest
```

The evals use Pipecat's eval harness against a running bot:

```bash
uv run bot.py
PYTHONPATH=. uv run pipecat eval run evals/ -v
```

`in_scope.yaml` asks one question per article and must get an answer, not a
handoff. It also checks that a call ends with `end_call` when the caller says
goodbye. `out_of_scope.yaml` covers fees, exchange rates, cards, account
actions, off-topic requests and prompt injection, all of which must go to
`transfer_to_human`. Answers are graded by `qwen/qwen3.8-27b` on Groq, a
different model family from the bot.

CI runs lint, type checks and unit tests on every push. The evals call live
APIs, so they run on manual dispatch.

## Deploy

1. In Render, create a Blueprint from this repo. `render.yaml` sets up a free
   Docker service and asks for the five secrets.
2. In the Twilio console, set the number's incoming call webhook to
   `https://<service>.onrender.com/` (HTTP POST).
3. Call the number. Trial accounts play a short notice first, then ask you to
   press a key.

## Known limitations

- Render's free tier sleeps after 15 idle minutes and takes about a minute to
  wake, longer than Twilio waits for the webhook. Open `/status` before a demo.
- A Twilio trial number only takes domestic calls and can only call verified
  numbers.
- Pipecat logs at debug level, which includes caller transcripts. Production
  would need redaction.
- Turn-taking isn't perfect. A pause mid-sentence can make the bot reply early,
  and a late transcript occasionally makes it answer twice.
- English only.
