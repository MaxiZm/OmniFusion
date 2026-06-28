# Security

Report security issues privately to the project maintainer. Do not open a public
issue for suspected credential leakage, SSRF bypasses, authentication bypasses,
or budget-accounting vulnerabilities.

Security-sensitive defaults:

- Set a real `OMNIFUSION_SECRET_KEY` and `OMNIFUSION_ADMIN_PASSWORD` before
  serving traffic.
- Docker Compose binds `127.0.0.1:8000` by default. Put TLS and public ingress in
  a reverse proxy you operate.
- Keep `OMNIFUSION_ALLOW_PRIVATE_EGRESS=0` unless private-network provider egress
  is intentional and reviewed.
- Treat exported `omnifusion.yaml` files as secrets because they contain decrypted
  provider keys.

No benchmark advantage claim should be made from mock, local-only, or
unreproducible runs.

