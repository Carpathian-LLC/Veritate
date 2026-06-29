# The chat front door

Veritate has a chat page served at both / and /chat. It is a self-contained page where you type a message and get an answer back. It is the simplest way to talk to a model on the platform, and it works with no setup because it can fall back to a public hosted model.

You choose which model answers from a settings panel on the chat page. The model picker groups the options. Under "Local (trained)" are the models you have trained yourself on this machine. Under "Cloud" is the Carpathian AI public model, which is always available. Then there is one group per teacher provider you have configured in Settings: configured API providers contribute their saved model, and local providers such as Ollama, LM Studio, and llama.cpp are enumerated live so every installed sub-model on those servers is listed individually.

When you select a locally trained model, you also choose an engine: PyTorch or Veritate (C). PyTorch runs the model directly in Python. The Veritate C engine runs an exported, compiled version of the model and is faster, but it is only available for models that have a veritate.bin export. The chat page disables the C button for any model that does not have one. Selecting a remote model (cloud or a teacher provider) hides the engine choice, since that model runs on the provider's side.

The chat page also has a Knowledge section with a toggle to ground answers using your own documents, plus an uploader to add documents to the local knowledge base. The grounding toggle stays disabled until the knowledge base has at least one file. When grounding is on, the model is given relevant snippets from your uploaded documents as context before it answers, and the answer shows which sources it used.

Your model selection, engine choice, and grounding setting are remembered in the browser between visits. If you have not trained a model and have no teacher configured, sending a message simply uses the public Carpathian model, labelled "Carpathian AI (public)".
