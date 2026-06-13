# Settings you care about

The dashboard's Settings tab is where you set your preferences. Settings are stored locally on your machine in a plain settings file and are read every time the dashboard starts. Some settings only take effect after a restart, so when in doubt, restart the dashboard after changing one.

Device preference controls which compute device your trainers use. The choices are auto, cpu, mps, and cuda. Auto detects the best available device. On Apple Silicon, mps is the GPU; on machines with an NVIDIA GPU, cuda is available. This setting is read by trainers when a run starts.

The public AI key is handled specially. Veritate ships with a shared public Carpathian key so the chat works out of the box with no setup. That key lives in the source and is injected at runtime rather than saved into your settings file, which means it is never stranded as a stale copy and updates reach every install. If you want to use your own key instead, set your personal override; your override takes precedence over the public key.

Teacher configuration lives in Settings as well. You pick a provider, supply a key or base URL, and choose a model. Each provider's configuration is remembered separately, so switching back to a provider restores what you saved for it. See the teacher and synthetic data topic for what teachers are used for.

Mesh role sets how this machine participates in a Veritate mesh. The options are off, hub, node, and both. Leave it off if you are running a single standalone machine.

Heartbeat and telemetry are consent-based. The heartbeat sends a presence signal to the Carpathian dashboard and shows an editable device name (auto-generated on first run, capped at fifteen characters). Additional telemetry is opt-in through separate advanced toggles: one to include the full training payload, one to include error detail in presence pings, and one to send a diagnostics payload. These are off by default and you choose what, if anything, to share.
