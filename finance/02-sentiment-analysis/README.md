# 02 — Financial Sentiment Analysis

> Use AI to analyse financial news headlines and score them as bullish, bearish, or neutral.

---

## What This Does

Market sentiment is a key input for trading and investment decisions. Reading and scoring
hundreds of news articles manually is impractical.

This script feeds financial news headlines to an AI model. For each headline, the AI returns:
- A **sentiment label** (BULLISH / BEARISH / NEUTRAL)
- A **confidence score** (0–100 %)
- The **affected asset or sector**
- A brief **market impact note**

---

## How to Run

```bash
cd finance/02-sentiment-analysis
python sentiment_analysis.py
```

---

## Expected Input

File: `data/sample_news.json`

Each record has:
- `id` — unique article identifier
- `headline` — news headline text
- `source` — news source name
- `published_at` — publication date

---

## Sample Output

```
📰 Financial Sentiment Analyser
   AI Provider: OPENAI
============================================================

[NEWS-003]  2024-12-10
Headline: "Fed signals three rate cuts in 2025 amid cooling inflation"
  Sentiment  : 🟢 BULLISH  (confidence: 91%)
  Affects    : Equities, Bonds, REITs
  Impact     : Rate cut expectations typically boost equity valuations and bond prices.

[NEWS-005]  2024-12-11
Headline: "Major bank reports $2.1B Q3 loss on bad loans"
  Sentiment  : 🔴 BEARISH  (confidence: 96%)
  Affects    : Banking sector, Financial ETFs
  Impact     : Large unexpected losses signal credit quality deterioration.

✅ Analysis complete. Bullish: 3 | Bearish: 2 | Neutral: 1
```

---

## Extending This Use-Case

- Scrape live headlines using `requests` + `BeautifulSoup`
- Aggregate sentiment scores across articles to produce a daily market signal
- Feed sentiment into a trading strategy back-test
