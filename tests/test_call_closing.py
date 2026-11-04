"""How calls end: a fixed closing line, spoken in full, then hang-up.

Whether the LLM picks the right tool is covered by the evals in evals/.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
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
    count, since = bot.next_silence(1, None, None)
    assert (count, since) == (2, None)

    count, since = bot.next_silence(count, "How do I check the status of my transfer?", since)
    assert (count, since) == (1, "How do I check the status of my transfer?")

    count, since = bot.next_silence(count, since, since)
    assert count == 2


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
