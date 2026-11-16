import { PipecatClient } from "https://esm.sh/@pipecat-ai/client-js@1.16.0";
import { WebSocketTransport } from "https://esm.sh/@pipecat-ai/websocket-transport@1.8.2?deps=@pipecat-ai/client-js@1.16.0";

const SESSION_LIMIT_S = 180;

const button = document.querySelector("#call");
const status = document.querySelector("#status");
const timer = document.querySelector("#timer");
const transcript = document.querySelector("#transcript");

let client = null;
let live = false;
let countdown = null;

function show(state, label) {
  document.body.dataset.call = state;
  status.textContent = label;
  button.textContent = state === "idle" ? "Talk to the agent" : "End call";
  button.disabled = state === "connecting";
}

function add(role, text) {
  const last = transcript.lastElementChild;
  if (last?.dataset.role === role) {
    last.querySelector("p").textContent += ` ${text}`;
  } else {
    const line = document.createElement("li");
    line.dataset.role = role;
    line.append(document.createElement("span"), document.createElement("p"));
    line.querySelector("span").textContent = role === "bot" ? "Agent" : "You";
    line.querySelector("p").textContent = text;
    transcript.append(line);
  }
  transcript.scrollTop = transcript.scrollHeight;
}

function startCountdown() {
  const end = Date.now() + SESSION_LIMIT_S * 1000;
  const tick = () => {
    const left = Math.max(0, Math.round((end - Date.now()) / 1000));
    timer.textContent = `${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}`;
  };
  tick();
  countdown = setInterval(tick, 1000);
}

function finish(label) {
  clearInterval(countdown);
  timer.textContent = "";
  delete document.body.dataset.speaking;
  client = null;
  show("idle", label);
}

async function start() {
  transcript.replaceChildren();
  live = false;
  show("connecting", "Connecting");
  client = new PipecatClient({
    transport: new WebSocketTransport(),
    enableMic: true,
    enableCam: false,
    callbacks: {
      onBotReady: () => {
        live = true;
        show("live", "Live");
        startCountdown();
      },
      onUserInput: ({ input_type, final, text }) => {
        if (input_type === "transcription" && final) add("user", text);
      },
      onBotOutput: ({ spoken_status, spoken_progress, text }) => {
        if (spoken_status === "completed") add("bot", spoken_progress?.accumulated_text || text);
      },
      onBotStartedSpeaking: () => (document.body.dataset.speaking = ""),
      onBotStoppedSpeaking: () => delete document.body.dataset.speaking,
      onDisconnected: () =>
        finish(live ? "Call ended" : "Line busy or waking up, try again in a minute"),
    },
  });

  const scheme = location.protocol === "https:" ? "wss" : "ws";
  try {
    await client.connect({ wsUrl: `${scheme}://${location.host}/ws-client` });
  } catch {
    finish("Couldn't connect. Allow the microphone and try again");
  }
}

button.addEventListener("click", () => (client ? client.disconnect() : start()));
