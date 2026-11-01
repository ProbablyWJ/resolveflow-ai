# Security

## Secrets

Never commit `后端/.env`, API keys, access tokens, customer data, or production traces. Start from `后端/.env.example` and keep real values only in the deployment platform's secret manager or the local ignored `.env` file.

If a key is exposed, revoke it at the provider first, then remove it from every commit and deployment environment. Deleting it only from the latest file is not sufficient.

## Demo boundary

ResolveFlow AI uses simulated orders, devices, warranties, tickets, and service policies. It is not connected to a real CRM, OMS, payment system, logistics system, or human-service queue.

Public deployments must enable API authentication, replace the Redis password, restrict allowed origins and hosts, and avoid collecting raw personal information.
