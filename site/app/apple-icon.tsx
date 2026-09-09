import { ImageResponse } from "next/og";

export const size = { width: 180, height: 180 };
export const contentType = "image/png";

export default function AppleIcon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#0b0d10",
        }}
      >
        <svg width="130" height="130" viewBox="0 0 130 130">
          <line x1="30" y1="95" x2="65" y2="35" stroke="#5c6672" strokeWidth="4" />
          <line x1="65" y1="35" x2="100" y2="95" stroke="#5c6672" strokeWidth="4" />
          <line x1="30" y1="95" x2="100" y2="95" stroke="#5c6672" strokeWidth="4" />
          <circle cx="30" cy="95" r="20" fill="#3ecf8e" />
          <circle cx="100" cy="95" r="20" fill="#6ea8fe" />
          <circle cx="65" cy="35" r="16" fill="#e6ebf0" />
        </svg>
      </div>
    ),
    { ...size }
  );
}
