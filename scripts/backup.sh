#!/usr/bin/env bash
# scripts/backup.sh — Helios database + MinIO backup to local archive
# Usage: bash scripts/backup.sh [backup_dir]
# Requires: docker compose running, pg_dump, mc (MinIO client)
set -euo pipefail

BACKUP_DIR="${1:-./backups}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_PATH="$BACKUP_DIR/$TIMESTAMP"
mkdir -p "$BACKUP_PATH"

echo "[backup] Starting Helios backup — $TIMESTAMP"

# ── PostgreSQL dump ───────────────────────────────────────────────────────────
echo "[backup] Dumping PostgreSQL..."
docker compose exec -T postgres pg_dump \
  -U helios \
  -d helios \
  --no-password \
  --format=custom \
  --compress=9 \
  > "$BACKUP_PATH/helios_pg.dump"
echo "[backup] PostgreSQL dump: $(du -sh "$BACKUP_PATH/helios_pg.dump" | cut -f1)"

# ── MinIO object snapshot ─────────────────────────────────────────────────────
echo "[backup] Mirroring MinIO bucket..."
if command -v mc &>/dev/null; then
  mc alias set helios_local http://localhost:9000 minioadmin minioadmin 2>/dev/null || true
  mc mirror helios_local/helios-docs "$BACKUP_PATH/minio_objects/" --quiet || true
  echo "[backup] MinIO mirror: $(du -sh "$BACKUP_PATH/minio_objects/" 2>/dev/null | cut -f1 || echo 'empty')"
else
  echo "[backup] mc not found — skipping MinIO snapshot (install from https://min.io/docs/minio/linux/reference/minio-mc.html)"
fi

# ── Compress archive ──────────────────────────────────────────────────────────
echo "[backup] Compressing archive..."
tar -czf "$BACKUP_DIR/helios_backup_$TIMESTAMP.tar.gz" -C "$BACKUP_DIR" "$TIMESTAMP"
rm -rf "$BACKUP_PATH"
ARCHIVE="$BACKUP_DIR/helios_backup_$TIMESTAMP.tar.gz"
echo "[backup] Archive: $ARCHIVE ($(du -sh "$ARCHIVE" | cut -f1))"

# ── Prune old backups (keep last 7) ──────────────────────────────────────────
find "$BACKUP_DIR" -name "helios_backup_*.tar.gz" -type f \
  | sort -r | tail -n +8 | xargs -r rm -v

echo "[backup] Done. Stored: $ARCHIVE"
