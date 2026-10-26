"use client";

import { ArrowUpRight, ImageIcon } from "lucide-react";
import { useEffect, useState } from "react";
import MathText from "@/components/ui/math-text";
import { getFigures, type PaperFigure } from "@/lib/api";

/* Figures from the papers behind an answer, with their captions.
 *
 * Loaded after the answer is already on screen, in one batched request, so they
 * can only ever add to a page and never delay one. Images are not copied: the
 * browser loads them from arXiv's own rendering of each paper, and a click opens
 * the original. Anything that goes wrong - no HTML rendering for a paper, arXiv
 * slow, an image that will not load - just means fewer figures; the component
 * renders nothing rather than an error. */

const ARXIV_ID = /^\d{4}\.\d{4,5}$/;

export function arxivIds(ids: string[], max = 3): string[] {
  return [...new Set(ids.filter((i) => ARXIV_ID.test(i)))].slice(0, max);
}

export default function PaperFigures({
  paperIds,
  question,
  titles = {},
  perPaper = 2,
}: {
  paperIds: string[];
  question: string;
  titles?: Record<string, string>;
  perPaper?: number;
}) {
  const ids = arxivIds(paperIds);
  const key = `${ids.join(",")}|${question}|${perPaper}`;
  const [loaded, setLoaded] = useState<{ key: string; figures: Record<string, PaperFigure[]> } | null>(null);

  useEffect(() => {
    if (ids.length === 0) return;
    let cancelled = false;
    getFigures(ids, question, perPaper)
      .then((figures) => !cancelled && setLoaded({ key, figures }))
      // Figures are decoration: a failed request is the same as "none found".
      .catch(() => !cancelled && setLoaded({ key, figures: {} }));
    return () => {
      cancelled = true;
    };
    // `key` already encodes ids, question and perPaper.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  if (ids.length === 0) return null;
  const ready = loaded?.key === key;
  const groups = ids
    .map((id) => ({ id, figures: (ready ? loaded.figures[id] : undefined) ?? [] }))
    .filter((g) => g.figures.length > 0);

  if (!ready) {
    return (
      <p className="flex items-center gap-2 text-xs text-slate-500">
        <ImageIcon size={13} className="animate-pulse" /> Looking for figures from these papers…
      </p>
    );
  }
  if (groups.length === 0) return null;

  return (
    <section aria-label="Figures from the cited papers" className="space-y-5">
      <h3 className="flex items-center gap-2 text-xs tracking-wider text-slate-500 uppercase">
        <ImageIcon size={13} /> Figures
      </h3>
      {groups.map(({ id, figures }) => (
        <div key={id} className="space-y-3">
          <a
            href={`https://arxiv.org/abs/${id}`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex max-w-full items-center gap-1 text-sm text-slate-300 hover:text-white"
          >
            <span className="truncate">{titles[id] ?? `arXiv:${id}`}</span>
            <ArrowUpRight size={13} className="shrink-0" />
          </a>
          <div className="grid gap-4 sm:grid-cols-2">
            {figures.map((f) => (
              <figure
                key={`${id}-${f.number}`}
                className="overflow-hidden rounded-xl border border-white/10 bg-white/[0.03]"
              >
                {/* Light tile: most paper figures are black on transparent and
                    would vanish on the dark page. */}
                <div className="flex flex-wrap gap-px bg-white p-2">
                  {f.images.map((src) => (
                    <a
                      key={src}
                      href={src}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex min-w-[40%] flex-1 items-center justify-center"
                      title="Open the full-size figure"
                    >
                      {/* eslint-disable-next-line @next/next/no-img-element -- remote, unknown size, never ours to optimise */}
                      <img
                        src={src}
                        alt={`${f.label}: ${f.caption.slice(0, 110)}`}
                        loading="lazy"
                        referrerPolicy="no-referrer"
                        className="max-h-60 w-auto max-w-full object-contain"
                        onError={(e) => {
                          e.currentTarget.style.display = "none";
                        }}
                      />
                    </a>
                  ))}
                </div>
                <figcaption className="space-y-1 p-3 text-xs leading-relaxed text-slate-400">
                  <span className="font-medium text-slate-200">{f.label}</span>
                  {f.caption && (
                    <MathText text={f.caption} className="line-clamp-4 block text-slate-400" />
                  )}
                </figcaption>
              </figure>
            ))}
          </div>
        </div>
      ))}
      <p className="text-[11px] text-slate-600">
        Shown from arXiv&apos;s HTML version of each paper; the figures belong to their authors.
      </p>
    </section>
  );
}
