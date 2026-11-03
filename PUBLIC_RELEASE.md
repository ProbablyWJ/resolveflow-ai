# Public release checklist

## Released

- Public repository: <https://github.com/ProbablyWJ/resolveflow-ai>
- Public demo: <https://resolveflow-ai-demo.onrender.com/>
- Release commit was scanned for secrets; `.env` is excluded from Git and Docker build context.
- The public demo uses Render Secrets, disables administrative/write and full-evaluation surfaces, and rate-limits chat requests.
- Backend 269 tests and frontend 26 tests passed; the Render image, health endpoint, homepage, and a real progressive-clarification chat were verified.
- The demo uses free ephemeral infrastructure and simulated business systems; it is not represented as a production enterprise integration.

## Ready

- Root README explains the product, architecture, demo path, evaluation evidence, and business boundary.
- `.env`, runtime data, logs, dependencies, IDE files, and local reference material are excluded from Git.
- The example environment file contains placeholders only.
- The project reports development-set and independent holdout results separately.
- User-facing replies and internal traces are separated.

## Checklist retained for future releases

1. Confirm that every inherited dependency and source file may be redistributed publicly. No open-source license is added automatically because distribution rights have not yet been confirmed.
2. Choose the GitHub account, repository name, and visibility.
3. Run the secret scan and both backend/frontend test suites on the exact commit to be published.
4. For a public demo, configure production authentication, random credentials, allowed origins/hosts, rate limits, and a deployment-platform secret manager.
5. Do not describe simulated business systems as real enterprise integrations.
