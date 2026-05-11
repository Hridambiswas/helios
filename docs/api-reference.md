# Helios API Reference

Base URL: `https://helios-hridam.ddns.net`

All authenticated endpoints require `Authorization: Bearer <access_token>`.

---

## Authentication

### POST /api/v1/auth/register

Register a new user account.

**Body**
```json
{
  "username": "alice",
  "email": "alice@example.com",
  "password": "Str0ng!Pass"
}
```

**Response 201**
```json
{
  "access_token": "<jwt>",
  "refresh_token": "<jwt>",
  "token_type": "bearer"
}
```

**Errors**: `400` username/email taken · `422` validation failed

---

### POST /api/v1/auth/login

Exchange credentials for tokens.

**Body (form-encoded)**
```
username=alice&password=Str0ng!Pass
```

**Response 200**
```json
{
  "access_token": "<jwt>",
  "refresh_token": "<jwt>",
  "token_type": "bearer"
}
```

---

### POST /api/v1/auth/refresh

Rotate access token using a valid refresh token.

**Body**
```json
{ "refresh_token": "<jwt>" }
```

**Response 200** — same shape as login.

---

### POST /api/v1/auth/logout

Revoke the current refresh token.

**Body**
```json
{ "refresh_token": "<jwt>" }
```

**Response 204** — no body.

---

### GET /api/v1/auth/me

Return the authenticated user's profile.

**Response 200**
```json
{
  "id": "uuid",
  "username": "alice",
  "email": "alice@example.com",
  "is_active": true
}
```

---

## Query

### POST /api/v1/query

Run the full Helios agent pipeline against a natural-language query.

**Body**
```json
{
  "query": "What is retrieval-augmented generation?",
  "stream": false
}
```

**Response 200**
```json
{
  "query_id": "uuid",
  "answer": "RAG is ...",
  "cited_doc_ids": ["doc-1", "doc-2"],
  "web_sources": [
    { "title": "...", "url": "https://...", "snippet": "..." }
  ],
  "follow_up_questions": [
    "How does RAG compare to fine-tuning?",
    "What vector databases work well with RAG?"
  ],
  "critic_passed": true,
  "critic_scores": { "relevance": 0.92, "factuality": 0.88 },
  "latency_ms": 3241,
  "pipeline_version": "1.1.0"
}
```

**Rate limit**: 6 req/min per IP.

---

## Documents

### POST /api/v1/documents/upload

Upload a file for indexing into the knowledge base.

**Body**: `multipart/form-data` with field `file`.

Supported types: PDF, TXT, MD, DOCX (max 50 MB).

**Response 202**
```json
{
  "document_id": "uuid",
  "filename": "paper.pdf",
  "status": "queued"
}
```

### GET /api/v1/documents

List documents visible to the authenticated user.

**Query params**: `limit` (default 50), `offset` (default 0).

---

## Health

### GET /health

Unauthenticated. Returns service health.

**Response 200**
```json
{
  "status": "ok",
  "version": "1.1.0",
  "db": true,
  "redis": true
}
```
