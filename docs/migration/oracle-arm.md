# Migration: EC2 → Oracle Cloud Always-Free ARM

Tracks the move of the Helios backend from AWS EC2 (t2.micro, dynamic public IP,
No-IP DDNS) to Oracle Cloud Always-Free (VM.Standard.A1.Flex, reserved public
IP, DuckDNS).

## Why

- Old EC2 public IP was reassigned after an instance stop → backend unreachable.
- `helios-hridam.ddns.net` (No-IP free tier) expired and the hostname stopped
  resolving.
- Frontend surfaced this as `request failed` on every API call.

## Target state

| Piece      | Before                          | After                                    |
| ---------- | ------------------------------- | ---------------------------------------- |
| Host       | AWS EC2 t2.micro (x86_64)       | Oracle A1.Flex (ARM64, 4 OCPU / 24 GB)   |
| Public IP  | Dynamic (`3.110.161.146`)       | Reserved (never changes)                 |
| DNS        | No-IP (`*.ddns.net`, monthly)   | DuckDNS (`*.duckdns.org`, no expiry)     |
| SSL        | Let's Encrypt on `.ddns.net`    | Let's Encrypt on `.duckdns.org`          |
| SSH key    | `helios-key.pem` (AWS)          | `id_ed25519_helios_oracle`               |
| Deploy     | GitHub Actions → EC2 SSH        | GitHub Actions → Oracle SSH              |

## Provisioning checklist (Oracle side, done in browser)

1. Sign up at <https://cloud.oracle.com/free> — needs a credit card for identity
   verification; Always-Free tier will not be charged.
2. Region: pick one close to you where **Ampere A1 capacity is available**
   (Mumbai `ap-mumbai-1` was full at time of writing; Hyderabad and Singapore
   have had capacity — you may need to retry).
3. Create instance: shape `VM.Standard.A1.Flex`, image `Canonical Ubuntu 22.04`,
   4 OCPU, 24 GB RAM (all inside Always-Free).
4. Paste the ed25519 public key into "Add SSH keys → Paste public keys".
5. Under Networking, tick **Assign a public IPv4 address** and, after the
   instance is up, promote it to a **Reserved Public IP** so it never changes.
6. Open ports 22, 80, 443 in the VCN default security list (Ingress rules).

## Oracle firewall gotcha

Oracle's Ubuntu images ship with an iptables policy that drops inbound traffic
even after the VCN security list allows it. First-boot fix (baked into
`backend/deploy/oracle/bootstrap.sh`):

```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```
