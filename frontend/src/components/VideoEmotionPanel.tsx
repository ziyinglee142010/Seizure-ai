"use client";

import {
  EMOTION_LABELS,
  type EmotionLabel,
  type EmotionResult,
} from "@/lib/video-detection";

const EMOTION_NAMES: Record<EmotionLabel | "no_face", string> = {
  angry: "Angry",
  disgust: "Disgust",
  fear: "Fear",
  happy: "Happy",
  neutral: "Neutral",
  sad: "Sad",
  surprise: "Surprise",
  no_face: "No face found",
};

function emotionColor(label: EmotionLabel | "no_face") {
  return `var(--color-emotion-${label === "no_face" ? "none" : label})`;
}

function percent(value: number | null | undefined) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "–";
}

function clock(seconds: number) {
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}

/** Per-second facial expression for one analysed video; separate from VSViG. */
export function VideoEmotionPanel({ emotion }: { emotion?: EmotionResult }) {
  if (!emotion || emotion.status === "disabled") return null;

  return (
    <section
      className="panel mt-6 p-5 sm:p-6"
      aria-labelledby="video-emotion-heading"
    >
      <p className="text-xs font-semibold uppercase tracking-wide text-ink-faint">
        Facial expression · separate from VSViG
      </p>
      <h2 id="video-emotion-heading" className="mt-1 text-lg font-semibold">
        Visible facial expression over time
      </h2>

      {emotion.status === "failed" ? (
        <p
          className="mt-4 rounded-md border border-amber/40 bg-amber-soft px-4 py-3 text-sm leading-6"
          role="status"
        >
          Facial expression analysis did not complete for this video. The VSViG
          seizure scores above are unaffected.
        </p>
      ) : (
        <EmotionDetails emotion={emotion} />
      )}
    </section>
  );
}

function EmotionDetails({
  emotion,
}: {
  emotion: Extract<EmotionResult, { status: "complete" }>;
}) {
  const { timeline } = emotion;
  const duration = timeline.end_sec.at(-1) ?? 0;
  const ranked = [...EMOTION_LABELS].sort(
    (a, b) =>
      (emotion.dominant_emotion_share[b] ?? 0) -
      (emotion.dominant_emotion_share[a] ?? 0),
  );
  const present = new Set(timeline.dominant_emotion);

  return (
    <>
      <p className="mt-2 text-sm leading-6 text-ink-muted">
        The patient&apos;s face was found in{" "}
        <strong className="text-ink">{percent(emotion.face_coverage)}</strong>{" "}
        of analysed frames ({emotion.windows_with_face} of {emotion.windows}{" "}
        seconds have a face).
        {emotion.model.model ? ` Model: ${emotion.model.model}.` : ""}
      </p>

      <h3 className="mt-5 text-sm font-semibold">
        Main expression each second
      </h3>
      <div
        className="mt-2 flex h-8 w-full overflow-hidden rounded-md border border-rule"
        role="img"
        aria-label="Main facial expression for each second of the video; grey where no face was found"
      >
        {timeline.dominant_emotion.map((label, index) => (
          <div
            key={timeline.start_sec[index]}
            style={{
              flexGrow: Math.max(
                timeline.end_sec[index] - timeline.start_sec[index],
                0.001,
              ),
              background: emotionColor(label),
            }}
            title={`${clock(timeline.start_sec[index])}: ${EMOTION_NAMES[label]}${
              label === "no_face"
                ? ""
                : ` (${percent(timeline.dominant_probability[index])})`
            }`}
          />
        ))}
      </div>
      <div className="mt-1 flex justify-between text-xs text-ink-faint">
        <span>0:00</span>
        <span>{clock(duration)}</span>
      </div>
      <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-muted">
        {[...EMOTION_LABELS, "no_face" as const]
          .filter((label) => present.has(label))
          .map((label) => (
            <li key={label} className="flex items-center gap-1.5">
              <span
                className="size-2.5 rounded-sm"
                style={{ background: emotionColor(label) }}
                aria-hidden="true"
              />
              {EMOTION_NAMES[label]}
            </li>
          ))}
      </ul>

      <h3 className="mt-6 text-sm font-semibold">
        Share of seconds by main expression
      </h3>
      <ul className="mt-2 grid gap-2">
        {ranked.map((label) => (
          <li
            key={label}
            className="grid grid-cols-[6rem_minmax(0,1fr)_3rem] items-center gap-3 text-sm"
          >
            <span className="flex items-center gap-1.5">
              <span
                className="size-2.5 shrink-0 rounded-sm"
                style={{ background: emotionColor(label) }}
                aria-hidden="true"
              />
              {EMOTION_NAMES[label]}
            </span>
            <span className="h-2 rounded-full bg-surface-muted">
              <span
                className="block h-2 rounded-full"
                style={{
                  width: percent(emotion.dominant_emotion_share[label] ?? 0),
                  background: emotionColor(label),
                }}
              />
            </span>
            <span className="text-right tabular-nums text-ink-muted">
              {percent(emotion.dominant_emotion_share[label])}
            </span>
          </li>
        ))}
      </ul>

      <details className="mt-6 border-y border-rule py-4">
        <summary className="cursor-pointer text-sm font-semibold">
          All seconds
        </summary>
        <div className="mt-4 max-h-72 overflow-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr>
                <th className="py-2">Second</th>
                <th>Main expression</th>
                <th>Score</th>
                <th>Face found</th>
              </tr>
            </thead>
            <tbody>
              {timeline.start_sec.map((start, index) => (
                <tr key={start} className="border-t border-rule">
                  <td className="text-ink-muted tabular-nums">
                    {clock(start)}
                  </td>
                  <td>{EMOTION_NAMES[timeline.dominant_emotion[index]]}</td>
                  <td className="tabular-nums">
                    {percent(timeline.dominant_probability[index])}
                  </td>
                  <td className="tabular-nums">
                    {percent(timeline.face_coverage[index])}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>

      <p className="mt-4 border-l-2 border-amber pl-3 text-xs leading-5 text-ink-muted">
        {emotion.validation_note}
      </p>
    </>
  );
}
