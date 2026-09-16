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
