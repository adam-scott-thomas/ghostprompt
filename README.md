# ghostprompt

Versioned prompt registry with semantic search, SHA-256 hashing, and audit trail.

Store prompts by name, pull with variable substitution, search by meaning. Every pull is hashed and logged — optionally sealed to Blackbox via ghostseal.

## Install

```bash
pip install ghostprompt
```

## Usage

```python
from ghostprompt import PromptRegistry

registry = PromptRegistry()

# Store prompts
registry.store("evaluate_code", "Evaluate this {{language}} code:\n{{code}}", tags=["eval", "code"])
registry.store("write_summary", "Summarize the following in {{length}} words:\n{{text}}", tags=["writing"])

# Pull with variables — returns rendered text + metadata
text, meta = registry.pull("evaluate_code", variables={"language": "Python", "code": "x = 1"})
# text = "Evaluate this Python code:\nx = 1"
# meta.hash = "sha256:a1b2c3..."  (hash of rendered text)
# meta.version = 1

# Search by meaning (requires Ollama with nomic-embed-text)
results = registry.search("code evaluation prompt")
# [("evaluate_code", 0.92), ...]

# Version tracking — templates auto-increment on change
registry.store("evaluate_code", "Review this {{language}} code:\n{{code}}")
info = registry.get("evaluate_code")
# info["version"] = 2
```

## With spine + ghostseal (full audit chain)

```python
from spine import Core
from ghostseal import SealClient
from ghostprompt import PromptRegistry

def setup(c):
    audit = SealClient(blackbox_url="https://blackbox:8443", api_key="...")
    prompts = PromptRegistry(audit=audit)
    prompts.store("evaluate_code", "Evaluate this {{language}} code:\n{{code}}")

    c.register("audit", audit)
    c.register("prompts", prompts)
    c.boot(env="prod")

Core.boot_once(setup)

# Any file:
prompts = Core.instance().get("prompts")
text, meta = prompts.pull("evaluate_code", variables={"language": "Python", "code": "..."})
# → ghostseal emits "ghostprompt.pull" with prompt_name, version, content_hash
# → Blackbox seals it into a hash-chained capsule
```

## Semantic search

Requires Ollama running locally with an embedding model:

```bash
ollama pull nomic-embed-text
```

```python
from ghostprompt import PromptRegistry
from ghostprompt.embedder import Embedder

registry = PromptRegistry(embedder=Embedder())
registry.store("assess_technical_v4", "Evaluate the {{domain}} work...", tags=["evaluation"])

# Search by meaning, not name
results = registry.search("code evaluation prompt")
# [("assess_technical_v4", 0.91)]
```

Falls back to text substring matching if Ollama isn't available.

## Part of the GhostLogic SDK

```
maelspine   → config registry
ghostseal   → audit backbone
ghostprompt → prompt management (this package)
ghostrouter → LLM routing
ghostserver → MCP tools
```

## License

Apache 2.0
