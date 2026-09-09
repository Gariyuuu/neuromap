import { ImageResponse } from "next/og";

export const size = { width: 32, height: 32 };
export const contentType = "image/png";

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          background: "#0b0d10",
          borderRadius: 7,
        }}
      >
        <svg width="32" height="32" viewBox="0 0 32 32">
          <line x1="10" y1="21" x2="22" y2="11" stroke="#8b96a3" strokeWidth="2" />
          <circle cx="10" cy="21" r="6" fill="#3ecf8e" />
          <circle cx="22" cy="11" r="6" fill="#6ea8fe" />
        </svg>
      </div>
    ),
    { ...size }
  );
}
