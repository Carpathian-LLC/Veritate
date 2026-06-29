# Teacher models and synthetic data

A teacher model in Veritate is an external, more capable model that you point the platform at to help train or ground your own smaller models. You configure a teacher in the Settings tab by choosing a provider, supplying a key or a base URL as needed, and picking a model. Supported providers include hosted API services (Carpathian, OpenAI, Anthropic, Gemini, xAI, DeepSeek, Mistral, Groq, OpenRouter) and local servers (Ollama, LM Studio, llama.cpp). For local providers, the platform asks the server which models it has and lists them for you.

The platform remembers a separate configuration per provider, so switching providers in Settings restores the key, model, and base URL you saved for that provider. Your keys are stored locally and are shown back to you as a mask rather than in the clear.

One use of a teacher is synthetic data generation, the "synth" action in the Training tab. You pick one or more seed topics from a catalog (the catalog covers domains such as math and code), and the platform asks the teacher model to produce many training examples for those topics. Each generated sample passes through quality gates (length checks, valid JSON where required, and near-duplicate filtering) before it is accepted, and accepted samples are written to disk. The job tracks its own state so it can resume, and a circuit breaker aborts a run that is mostly failing instead of grinding away.

The teacher is also used by the RAG training action. There, the platform uses the teacher to generate a question-and-answer corpus anchored to context, then continue-trains your chosen base model on it so the model learns to answer from supplied context. Because both synth and RAG depend on a teacher, those actions are disabled until you configure one, and the dashboard shows a banner prompting you to set a teacher up.

In short, teacher models are a tool for bootstrapping your own models: they generate training data and grounded examples so a small local model can be taught capabilities it would otherwise need far more raw data to learn.
