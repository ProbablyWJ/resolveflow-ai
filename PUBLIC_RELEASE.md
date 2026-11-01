# Public release checklist

## Ready

- Root README explains the product, architecture, demo path, evaluation evidence, and business boundary.
- `.env`, runtime data, logs, dependencies, IDE files, and local reference material are excluded from Git.
- The example environment file contains placeholders only.
- The project reports development-set and independent holdout results separately.
- User-facing replies and internal traces are separated.

## Required before making the repository public

1. Confirm that every inherited dependency and source file may be redistributed publicly. No open-source license is added automatically because distribution rights have not yet been confirmed.
2. Choose the GitHub account, repository name, and visibility.
3. Run the secret scan and both backend/frontend test suites on the exact commit to be published.
4. For a public demo, configure production authentication, random credentials, allowed origins/hosts, rate limits, and a deployment-platform secret manager.
5. Do not describe simulated business systems as real enterprise integrations.
