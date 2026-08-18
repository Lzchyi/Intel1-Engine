# Security Policy

Do not open public issues for credential exposure or sensitive vulnerabilities. Use GitHub private vulnerability reporting when available, or contact the maintainer privately.

Never commit `.env`, API keys, GitHub tokens, service-account files, signing material, private production credentials, or user data. Use GitHub Actions Secrets and environment variables.

If a secret is ever committed, rotate/revoke it immediately; deleting it from the latest revision is not sufficient.

Treat all retrieved external content as untrusted input.
