# Oracle Cloud Always-Free bootstrap

Idempotent bootstrap for the Helios backend on an Oracle Cloud A1.Flex
(ARM64) Ubuntu 22.04 VM.

```bash
# on your laptop
scp -i ~/.ssh/id_ed25519_helios_oracle backend/deploy/oracle/bootstrap.sh \
    ubuntu@$ORACLE_HOST:~/bootstrap.sh

# on the VM
ssh -i ~/.ssh/id_ed25519_helios_oracle ubuntu@$ORACLE_HOST
sudo bash ~/bootstrap.sh
```

The script is safe to re-run; each step guards on the desired state and no-ops
if already satisfied.

## What it does

1. `apt` update + install Docker CE, docker-compose plugin, certbot,
   netfilter-persistent, git.
2. Adds `ubuntu` to the `docker` group.
3. Opens ports 80 + 443 in iptables (Oracle Ubuntu default policy blocks them).
4. Clones `github.com/Hridambiswas/helios` into `/home/ubuntu/helios`.
5. Reminds you to populate `backend/.env` and `/etc/helios/duckdns.env`.
