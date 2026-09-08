# AEO scoring pipeline with AWS Lambda durable functions

A serverless pipeline that scores any web page for **Agent Engine Optimization (AEO)** — how discoverable, parsable, and usable your content is to autonomous AI agents (coding agents, AI crawlers). It runs as a single [AWS Lambda durable function](https://docs.aws.amazon.com/lambda/latest/dg/durable-functions.html) with five checkpointed steps.

Scoring is **fully deterministic** — every point maps to a measurable, threshold-based check, so the same page always produces the same score. Amazon Bedrock is used only in one step, to generate a prose explanation of the results and recommend fixes. It never influences the numeric score.

This is sample code that accompanies the AWS Compute Blog post *Building an AEO scoring pipeline with AWS Lambda durable functions*. It is intended for learning and experimentation, not production use as-is.

## How it works

The pipeline is one durable function that chains five steps. Each step's output is checkpointed before the next begins, so a failure resumes from the failed step rather than re-running the whole chain.

| # | Step | What it does | Uses an LLM? |
|---|------|--------------|--------------|
| 1 | `fetch_and_parse` | HTTP GET of the target page plus `robots.txt`, `llms.txt`, `AGENTS.md`, `skill.md`, and the Markdown twin. Extracts structured signals. | No |
| 2 | `score_signals` | Applies the deterministic rubric to the signals and produces a 0–100 score. | No |
| 3 | `analyze_with_ai` | Sends the scores and signals to Amazon Bedrock, which returns prose findings and fix recommendations (temperature 0). | Yes |
| 4 | `publish_report` | Renders the report as HTML and JSON and writes both to Amazon S3. | No |
| 5 | `record_outcome` | Persists the score and metadata to Amazon DynamoDB for trend tracking. | No |

Only step 3 calls Amazon Bedrock. The other four steps are pure Python.

The page is fetched with an AI-crawler user agent (`ClaudeBot/1.0`) over a plain HTTP GET — the same way non-rendering AI agents see pages today. Pages that depend on client-side JavaScript will score lower because the fetch does not execute JavaScript.

## The scoring rubric

Weights align with Addy Osmani's [`agentic-seo`](https://github.com/addyosmani/agentic-seo) framework (100 points total). Every check is a threshold or a boolean — no model judgment is involved.

| Signal | Max points | Sample checks |
|--------|-----------:|---------------|
| Discoverability | 25 | HTTP 200 + server-rendered content, AI crawlers not blocked in `robots.txt`, `llms.txt` present and concise, `AGENTS.md` present |
| Parsability | 25 | H1 present, valid heading hierarchy, code blocks, tables, Markdown twin, low HTML noise ratio |
| Token efficiency | 25 | Token count under the 15,000 quick-start target / 30,000 ceiling, not JavaScript-dependent |
| Capability signaling | 15 | `skill.md` present, `AGENTS.md` present, structured content (code + tables) |
| Access control | 10 | AI-specific rules in `robots.txt`, sitemap declared, intentional allow rules |

Thresholds come from the [Osmani AEO framework](https://addyosmani.com/blog/agentic-engine-optimization/), [RFC 9309 (robots.txt)](https://www.rfc-editor.org/rfc/rfc9309.html), and the [llms.txt proposed standard](https://llmstxt.org/).

## Architecture

```
                    ┌─────────────────────────────────────────────┐
   target URL ─────▶│  AWS Lambda durable function                 │
                    │                                              │
                    │  1. fetch_and_parse   (no LLM)               │
                    │  2. score_signals     (no LLM, deterministic)│
                    │  3. analyze_with_ai   ──────▶ Amazon Bedrock │
                    │  4. publish_report    ──────▶ Amazon S3      │
                    │  5. record_outcome    ──────▶ Amazon DynamoDB│
                    └─────────────────────────────────────────────┘
```

An optional human-review callback can be turned on between steps 3 and 4 (see [Optional: human review gate](#optional-human-review-gate)).

## Repository layout

```
.
├── src_aeo/
│   ├── app.py              # The durable function: all five steps
│   └── requirements.txt    # aws-durable-execution-sdk-python, boto3
├── template.aeo.yaml       # AWS SAM template (Lambda, S3, DynamoDB, IAM)
├── LICENSE                 # MIT-0
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
└── README.md
```

## Prerequisites

- An AWS account with Amazon Bedrock model access enabled
- [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html) v1.100 or later, installed and configured
- Python 3.12 or later
- IAM permissions to deploy AWS CloudFormation stacks (Lambda, DynamoDB, S3, and `bedrock:InvokeModel`)

## Deploy

```bash
sam build --template-file template.aeo.yaml
sam deploy --template-file .aws-sam/build/template.yaml \
  --stack-name aeo-scoring-pipeline \
  --capabilities CAPABILITY_IAM \
  --resolve-s3 \
  --guided
```

The template provisions the Lambda function, the DynamoDB table, the S3 report bucket, and the required IAM roles. By default it deploys the current Bedrock model referenced by the `BedrockModelId` parameter; override it at deploy time if you want a different model:

```bash
sam deploy ... --parameter-overrides BedrockModelId=<your-model-or-inference-profile-id>
```

## Run an audit

Invoke the function's published alias with a target URL:

```bash
echo '{"execution_name":"run-1","url":"https://docs.aws.amazon.com/lambda/latest/dg/durable-functions.html"}' > payload.json

aws lambda invoke \
  --function-name aeo-analyzer:prod \
  --invocation-type Event \
  --payload fileb://payload.json \
  --region <your-region> \
  /dev/null
```

The pipeline writes an HTML and a JSON report to the S3 bucket (`aeo-reports/<host><path>.html` / `.json`) and records the score in DynamoDB. Because scoring is deterministic, running the same URL twice yields the same score.

## Configuration

The function reads these environment variables (set in `template.aeo.yaml`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `ARTIFACTS_BUCKET` | (set by template) | S3 bucket for HTML/JSON reports |
| `STATE_TABLE` | (set by template) | DynamoDB table for score history |
| `BEDROCK_MODEL_ID` | (template parameter) | Bedrock model or inference profile ID used in step 3 |
| `APPROVAL_GATE` | `false` | Turn on the human-review callback between steps 3 and 4 |
| `BEDROCK_TEMPERATURE` | `0` | Temperature for the analyze step; 0 keeps prose stable across runs |

## Optional: human review gate

By default the pipeline runs all five steps without stopping. Set `APPROVAL_GATE=true` to insert a human-review callback after the analyze step. The function suspends at zero cost until an external system resumes it by calling the `SendDurableExecutionCallbackSuccess` Lambda API with the callback ID. Because that is an authenticated AWS API call, only principals with the appropriate IAM permissions can resume the execution.

## Clean up

To avoid ongoing charges, delete the stack when you are done:

```bash
sam delete --stack-name aeo-scoring-pipeline
```

This removes the Lambda function, DynamoDB table, S3 bucket (if empty), and associated IAM roles. If the S3 bucket contains reports you want to keep, download them first.

## Security

This sample validates that target URLs use `https://` before fetching. It does not add authentication to the Lambda function itself — invoke it only with credentials you control. For production use, review the [AWS Well-Architected Framework Security Pillar](https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/welcome.html), and if the Bedrock-generated analysis is ever surfaced to end users, add [Amazon Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html).

## References

- [Agent Engine Optimization](https://addyosmani.com/blog/agentic-engine-optimization/) — Addy Osmani
- [`agentic-seo`](https://github.com/addyosmani/agentic-seo) — the deterministic scoring framework these weights align with
- [AWS Lambda durable functions Developer Guide](https://docs.aws.amazon.com/lambda/latest/dg/durable-functions.html)
- [AWS Durable Execution SDK Developer Guide](https://docs.aws.amazon.com/durable-functions/)
- [RFC 9309: Robots Exclusion Protocol](https://www.rfc-editor.org/rfc/rfc9309.html)
- [llms.txt proposed standard](https://llmstxt.org/)

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
