# Migration notes

Backend migrated from AWS EC2 (dynamic IP, No-IP DDNS) to Oracle Cloud
Always-Free ARM (reserved IP, DuckDNS) in this branch.

Full write-up: [`docs/migration/oracle-arm.md`](docs/migration/oracle-arm.md).

## Cut-over cheat sheet

```bash
# after provisioning the Oracle VM + registering the DuckDNS name
ORACLE_HOST=<vm-ip> DUCKDNS_TOKEN=<token> GROQ_API_KEY=<key> \
  backend/deploy/oracle/first-deploy.sh

ORACLE_HOST=<vm-ip> \
  backend/deploy/oracle/rotate-secrets.sh
```

Details: [`backend/deploy/oracle/README.md`](backend/deploy/oracle/README.md).
