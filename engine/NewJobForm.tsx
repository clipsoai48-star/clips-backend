"use client";

import { useState } from "react";
import { api, User } from "@/lib/api";

const CAPTION_STYLES = [
  { value: "none", label: "No captions", paid: false },
  { value: "basic", label: "Basic", paid: false },
  { value: "bold_yellow", label: "Bold Yellow", paid: true },
  { value: "minimal_white", label: "Minimal White", paid: true },
  { value: "boxed", label: "Boxed", paid: true },
  { value: "creator_pink", label: "Creator Pink", paid: true },
];

export default function NewJobForm({
  user,
  onCreated,
}: {
  user: User | null;
  onCreated: () => void;
}) {
  const [url, setUrl] = useState("");
  const [clipCount, setClipCount] = useState(3);
  const [clipLength, setClipLength] = useState(20);
  const [captionStyle, setCaptionStyle] = useState("basic");
  const [speakerColors, setSpeakerColors] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isPaid = user?.is_paid_tier ?? false;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await api.createJobFromUrl({
        source_url: url,
        target_clip_count: clipCount,
        clip_length_seconds: clipLength,
        caption_style: captionStyle,
        speaker_colors: speakerColors,
        use_llm_rerank: false,
      });
      setUrl("");
      onCreated();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't submit that job");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="rounded-2xl border border-line bg-surface p-6">
      <label htmlFor="url" className="text-sm text-mist">YouTube or Twitch URL</label>
      <input
        id="url"
        type="url"
        required
        placeholder="https://youtube.com/watch?v=..."
        value={url}
        onChange={(e) => setUrl(e.target.value)}
        className="mt-1.5 w-full rounded-lg border border-line bg-ink px-4 py-2.5 text-paper outline-none transition focus:border-violet"
      />

      <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <div>
          <label htmlFor="clipCount" className="text-xs text-mist">Clips</label>
          <input
            id="clipCount"
            type="number"
            min={1}
            max={10}
            value={clipCount}
            onChange={(e) => setClipCount(Number(e.target.value))}
            className="mt-1 w-full rounded-lg border border-line bg-ink px-3 py-2 text-sm text-paper outline-none focus:border-violet"
          />
        </div>
        <div>
          <label htmlFor="clipLength" className="text-xs text-mist">Length (sec)</label>
          <input
            id="clipLength"
            type="number"
            min={10}
            max={60}
            value={clipLength}
            onChange={(e) => setClipLength(Number(e.target.value))}
            className="mt-1 w-full rounded-lg border border-line bg-ink px-3 py-2 text-sm text-paper outline-none focus:border-violet"
          />
        </div>
        <div className="col-span-2">
          <label htmlFor="captionStyle" className="text-xs text-mist">Caption style</label>
          <select
            id="captionStyle"
            value={captionStyle}
            onChange={(e) => setCaptionStyle(e.target.value)}
            className="mt-1 w-full rounded-lg border border-line bg-ink px-3 py-2 text-sm text-paper outline-none focus:border-violet"
          >
            {CAPTION_STYLES.map((s) => (
              <option key={s.value} value={s.value} disabled={s.paid && !isPaid}>
                {s.label}{s.paid && !isPaid ? " (Pro)" : ""}
              </option>
            ))}
          </select>
        </div>
      </div>

      <label className="mt-4 flex items-center gap-2.5 text-sm text-mist">
        <input
          type="checkbox"
          checked={speakerColors}
          disabled={!isPaid}
          onChange={(e) => setSpeakerColors(e.target.checked)}
          className="h-4 w-4 rounded border-line accent-violet"
        />
        Speaker-colored captions {!isPaid && <span className="text-xs">(Pro)</span>}
      </label>

      {error && (
        <p className="mt-4 rounded-lg border border-ember/30 bg-ember/10 px-4 py-2.5 text-sm text-ember">
          {error}
        </p>
      )}

      <button
        type="submit"
        disabled={submitting}
        className="mt-5 rounded-full bg-brand-gradient px-6 py-2.5 text-sm font-medium text-white transition hover:opacity-90 disabled:opacity-50"
      >
        {submitting ? "Submitting…" : "Generate clips"}
      </button>
    </form>
  );
}
