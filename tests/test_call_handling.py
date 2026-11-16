"""Deterministic call handling: who may connect, silence, and how calls end.

Whether the LLM picks the right tool is covered by the evals in evals/.
"""

import asyncio
import base64
import hashlib
import hmac
from unittest.mock import AsyncMock, MagicMock

import pytest
from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.runner.types import WebSocketRunnerArguments
from pipecat.workers.runner import WorkerRunner

import bot


class FrameRecorder(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.frames = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        await self.push_frame(frame, direction)


def test_a_new_caller_message_resets_the_silence_count():
    question = {"role": "user", "content": "Where is my transfer?"}
    answer = {"role": "assistant", "content": "You can check it in the app."}

    strikes, heard = bot.silence_strikes(0, [], 0)
    assert strikes == 1
    strikes, heard = bot.silence_strikes(strikes, [question, answer], heard)
    assert strikes == 1
    strikes, heard = bot.silence_strikes(strikes, [question, answer], heard)
    assert strikes == 2
    strikes, heard = bot.silence_strikes(strikes, [question, answer, question], heard)
    assert strikes == 1


def _signature(token: str, url: str) -> str:
    return base64.b64encode(hmac.new(token.encode(), url.encode(), hashlib.sha1).digest()).decode()


@pytest.mark.parametrize(
    ("header", "accepted"),
    [
        (_signature("secret", "wss://agent.example/ws"), True),
        (_signature("secret", "wss://agent.example/ws/"), True),
        (_signature("wrong", "wss://agent.example/ws"), False),
        (None, False),
    ],
    ids=["signed", "signed_with_trailing_slash", "wrong_token", "unsigned"],
)
def test_only_twilio_signed_media_streams_are_accepted(header, accepted):
    websocket = MagicMock(headers={"x-twilio-signature": header} if header else {})
    assert bot.signed_by_twilio(websocket, "agent.example", "secret") is accepted


def test_a_second_browser_session_is_turned_away():
    websocket = MagicMock(close=AsyncMock())
    args = WebSocketRunnerArguments(websocket=websocket, transport_type="websocket")

    async def run():
        async with bot.browser_line:
            await bot.bot(args)

    asyncio.run(run())
    websocket.close.assert_awaited_once_with(code=4003)


def test_closing_line_is_spoken_before_the_pipeline_ends():
    recorder = FrameRecorder()
    worker = PipelineWorker(Pipeline([recorder]))

    async def run():
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        await worker.queue_frames(bot.closing_frames(bot.GOODBYE))
        await asyncio.wait_for(runner.run(), timeout=5)

    asyncio.run(run())

    speech, end = [f for f in recorder.frames if isinstance(f, TTSSpeakFrame | EndFrame)]
    assert isinstance(speech, TTSSpeakFrame)
    assert speech.text == bot.GOODBYE
    assert speech.interruptible is False
    assert isinstance(end, EndFrame)


@pytest.mark.parametrize(
    ("call_tool", "line"),
    [
        (lambda params: bot.transfer_to_human(params, reason="exchange rates"), bot.DEFLECTION),
        (bot.end_call, bot.GOODBYE),
    ],
    ids=["transfer_to_human", "end_call"],
)
def test_tools_close_the_call_without_an_llm_reply(call_tool, line):
    params = MagicMock(result_callback=AsyncMock())
    params.llm.push_frame = AsyncMock()

    asyncio.run(call_tool(params))

    assert params.result_callback.await_args.kwargs["properties"].run_llm is False
    pushed = [c.args[0] for c in params.llm.push_frame.await_args_list]
    assert [type(f) for f in pushed] == [type(f) for f in bot.closing_frames(line)]
    assert pushed[0].text == line
