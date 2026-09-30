#!/usr/bin/env node
// shots.mjs — Playwright screenshot pipeline for the redesign.
//
// Builds the frontend with VITE_DEMO_MODE=true, spins up `vite preview`
// on a free port, then captures:
//   docs/redesign/hero-desktop.png       (1440x900 hero, idle)
//   docs/redesign/mid-query-desktop.png  (1440x900, mid-query planning/retrieving)
//   docs/redesign/answer-desktop.png     (1440x900, answer view rendered)
//   docs/redesign/hero-mobile.png        (375x812 hero, idle)
//   docs/redesign/answer-mobile.png      (375x812, answer view)
//
// Launch args force ANGLE + SwiftShader so WebGL renders under
// headless Chromium (Playwright's headless-shell doesn't ship a real
// GPU driver). Without these flags the R3F canvas paints nothing.

import { spawn } from 'node:child_process'
import { chromium } from 'playwright'
import { mkdir } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const FRONTEND_DIR = path.resolve(__dirname, '..')
const REPO_ROOT   = path.resolve(FRONTEND_DIR, '..')
const OUT_DIR     = path.join(REPO_ROOT, 'docs', 'redesign')
const PORT        = 4173

const LAUNCH_ARGS = [
  '--use-gl=angle',
  '--use-angle=swiftshader',
  '--enable-webgl',
  '--ignore-gpu-blocklist',
]

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// ── Serve the built app ─────────────────────────────────────────────────────
function startPreview() {
  const proc = spawn(
    'npx',
    ['vite', 'preview', '--port', String(PORT), '--strictPort'],
    { cwd: FRONTEND_DIR, stdio: ['ignore', 'inherit', 'inherit'] },
  )
  return proc
}

// ── Wait for the server to accept a connection ──────────────────────────────
async function waitForServer(url, timeoutMs = 15000) {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    try {
      const r = await fetch(url)
      if (r.ok) return
    } catch (_) { /* server not up yet */ }
    await sleep(200)
  }
  throw new Error(`Server did not respond at ${url} within ${timeoutMs}ms`)
}

async function captureDesktop(browser) {
  const ctx = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1.5,
  })
  const page = await ctx.newPage()

  // 1) Hero, idle
  await page.goto(`http://localhost:${PORT}/`)
  await page.waitForLoadState('networkidle')
  // Wait for the R3F canvas to actually paint. The sun rise takes 1.2s;
  // give it 2s to be safe under headless.
  await sleep(2000)
  await page.screenshot({ path: path.join(OUT_DIR, 'hero-desktop.png') })

  // 2) Mid-query (planning/retrieving)
  await page.locator('#helios-query').fill('Compare BM25 and dense retrieval')
  await page.locator('button[type="submit"]').click()
  await sleep(900) // land between retrieving and executing
  await page.screenshot({ path: path.join(OUT_DIR, 'mid-query-desktop.png') })

  // 3) Answer view
  await sleep(3400) // full scripted stream is 3.95s
  // Scroll to the answer card
  await page.evaluate(() => {
    document.querySelector('article')?.scrollIntoView({ behavior: 'auto', block: 'center' })
  })
  await sleep(400)
  await page.screenshot({ path: path.join(OUT_DIR, 'answer-desktop.png') })

  await ctx.close()
}

async function captureMobile(browser) {
  const ctx = await browser.newContext({
    viewport: { width: 375, height: 812 },
    deviceScaleFactor: 2,
    isMobile: true,
    hasTouch: true,
  })
  const page = await ctx.newPage()

  // 1) Mobile hero
  await page.goto(`http://localhost:${PORT}/`)
  await page.waitForLoadState('networkidle')
  await sleep(2000)
  await page.screenshot({ path: path.join(OUT_DIR, 'hero-mobile.png') })

  // 2) Mobile after query
  await page.locator('#helios-query').fill('Explain RLHF in 5 lines')
  await page.locator('button[type="submit"]').click()
  await sleep(4300) // full stream + one extra frame
  await page.evaluate(() => {
    document.querySelector('article')?.scrollIntoView({ behavior: 'auto', block: 'center' })
  })
  await sleep(400)
  await page.screenshot({ path: path.join(OUT_DIR, 'answer-mobile.png') })

  await ctx.close()
}

// ── Main ────────────────────────────────────────────────────────────────────
async function main() {
  await mkdir(OUT_DIR, { recursive: true })
  const preview = startPreview()
  try {
    await waitForServer(`http://localhost:${PORT}/`)
    const browser = await chromium.launch({
      args: LAUNCH_ARGS,
      headless: true,
    })
    try {
      await captureDesktop(browser)
      await captureMobile(browser)
    } finally {
      await browser.close()
    }
  } finally {
    preview.kill('SIGTERM')
    // Give vite a moment to release the port so subsequent runs work.
    await sleep(200)
  }
  console.log(`Screenshots written to ${OUT_DIR}`)
}

main().catch((err) => {
  console.error(err)
  process.exitCode = 1
})
