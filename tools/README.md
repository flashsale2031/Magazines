# Local article generator

`generate_articles.py` scans recent news leads for ten topics and creates one standalone Markdown article per topic.

## Topics

The generator covers **community events, education, nature, food, entertainment, games and consoles, automotive, technology, finance, and comics**. Searches default to `Anchorage Alaska`, matching the repository’s Anchorage View focus, but the location can be changed with `--location`.

## Requirements

The script uses only Python’s standard library. It reads recent Google News RSS search results for discovery and calls an OpenAI-compatible Chat Completions endpoint for article generation. Set `OPENAI_API_KEY` before a live run. The default model is `gpt-5-mini`; change it with `--model` or `ARTICLE_MODEL`. The endpoint defaults to `OPENAI_API_BASE` when set, otherwise `https://api.openai.com/v1`.

## Usage

From the repository root:

```bash
# Check feed discovery without calling an LLM or writing articles.
python3 tools/generate_articles.py --dry-run

# Generate one article per topic into generated_articles/.
export OPENAI_API_KEY="your-key"
python3 tools/generate_articles.py

# Replace existing topic files and use a different location.
python3 tools/generate_articles.py \
  --location "Anchorage Alaska" \
  --hours 72 \
  --overwrite
```

The command writes these files by default:

```text
generated_articles/
├── automotive.md
├── comics.md
├── community-events.md
├── education.md
├── entertainment.md
├── finance.md
├── food.md
├── games-and-consoles.md
├── nature.md
└── technology.md
```

Each article contains a headline, deck, body paragraphs, a “Why it matters” note, and a source list linking to the exact feed leads used. The script supplies only story metadata and feed snippets to the model, validates that generated source URLs came from the selected leads, and adds a disclosure that the copy is an original paraphrased article rather than a replacement for the cited publisher.

## Editorial and operational notes

The generator does not copy article text or generate unattributed claims by design, but every generated article should still be reviewed by an editor before publication. RSS availability and freshness vary by publisher. Use `--dry-run` to inspect the current leads before generating copy, and use `--hours` to control the freshness window.

The script continues through all ten topics if one feed or model request fails, then exits with a nonzero status and lists the failed topics. It will not overwrite existing files unless `--overwrite` is provided.
