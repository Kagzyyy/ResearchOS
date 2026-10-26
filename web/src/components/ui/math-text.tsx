"use client";

import katex from "katex";
import "katex/dist/katex.min.css";
import { Fragment, useMemo, type ReactNode } from "react";

/* Typeset the maths in plain text.
 *
 * Models and arXiv abstracts write equations as LaTeX in four different
 * delimiters - $$..$$, \[..\], \(..\) and $..$ - and left unrendered they show
 * up as slashes and braces. This finds them, renders each with KaTeX, and
 * leaves everything else untouched.
 *
 * Equations are pulled out BEFORE any splitting on newlines or markdown: a
 * display equation routinely spans several lines, and `**` or `_` inside one
 * must not be read as emphasis. They are swapped for placeholder characters
 * (private-use code points, which never occur in real text), the surrounding
 * text goes through whatever formatting the caller does, and withMath() puts
 * the typeset equations back.
 *
 * KaTeX runs with trust off, so a \href or \includegraphics in text we did not
 * write cannot link or load anything. A formula it cannot parse is shown as the
 * original source rather than throwing. */

export interface MathItem {
  tex: string;
  display: boolean;
}

/* Alternatives, in order: $$..$$ | \[..\] | \(..\) | $..$.
 * The single-dollar form is the dangerous one ("it costs $5 and $10"), so it
 * needs a non-space after the opening $, a non-space before the closing $, no
 * digit straight after it, and at least a letter or backslash inside. */
const MATH_RE =
  /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|(?<![\\$\w])\$(?![\s$])([^$\n]*?[A-Za-z\\][^$\n]*?)(?<![\s\\])\$(?![\d$])/g;

const OPEN = "";
const CLOSE = "";
const PLACEHOLDER = /(\d+)/g;

export function extractMath(text: string): { text: string; items: MathItem[] } {
  const items: MathItem[] = [];
  const out = text.replace(MATH_RE, (_m, dd, br, par, dollar) => {
    const display = dd !== undefined || br !== undefined;
    const tex = (dd ?? br ?? par ?? dollar ?? "").trim();
    if (!tex) return _m;
    items.push({ tex, display });
    return `${OPEN}${items.length - 1}${CLOSE}`;
  });
  return { text: out, items };
}

const cache = new Map<string, string>();

function render(tex: string, display: boolean): string {
  const key = (display ? "D" : "I") + tex;
  const hit = cache.get(key);
  if (hit !== undefined) return hit;
  let html: string;
  try {
    html = katex.renderToString(tex, {
      displayMode: display,
      throwOnError: true,
      strict: "ignore",
      trust: false,
      output: "htmlAndMathml",
      maxExpand: 1000,
    });
  } catch {
    // Unparseable: show the source, escaped, instead of a red error box.
    const esc = tex.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]!);
    html = `<code class="font-mono text-[0.9em] text-slate-300">${esc}</code>`;
  }
  if (cache.size > 800) cache.clear();
  cache.set(key, html);
  return html;
}

export function Equation({ tex, display }: MathItem) {
  return (
    <span
      className={display ? "math-display" : undefined}
      dangerouslySetInnerHTML={{ __html: render(tex, display) }}
    />
  );
}

/** Put typeset equations back into a piece of text that came out of extractMath(). */
export function withMath(part: string, items: MathItem[], keyPrefix = "m"): ReactNode[] {
  const nodes: ReactNode[] = [];
  let last = 0;
  let n = 0;
  for (const m of part.matchAll(PLACEHOLDER)) {
    const at = m.index ?? 0;
    if (at > last) nodes.push(part.slice(last, at));
    const item = items[Number(m[1])];
    nodes.push(item ? <Equation key={`${keyPrefix}${n++}`} {...item} /> : null);
    last = at + m[0].length;
  }
  if (last < part.length) nodes.push(part.slice(last));
  return nodes;
}

/** Plain text with its equations typeset. For abstracts, snippets and captions. */
export default function MathText({ text, className }: { text: string; className?: string }) {
  const parts = useMemo(() => extractMath(text), [text]);
  return (
    <span className={className}>
      {withMath(parts.text, parts.items).map((n, i) => (
        <Fragment key={i}>{n}</Fragment>
      ))}
    </span>
  );
}
