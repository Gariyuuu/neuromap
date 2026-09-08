"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function NavLinks({ routes }: { routes: { href: string; label: string }[] }) {
  const pathname = usePathname();
  return (
    <nav className="flex gap-1 overflow-x-auto text-sm">
      {routes.map((r) => {
        const active = pathname === r.href;
        return (
          <Link
            key={r.href}
            href={r.href}
            className={`px-3 py-1.5 rounded-md whitespace-nowrap transition-colors ${
              active ? "bg-[#171c22] text-fg" : "text-dim hover:text-fg"
            }`}
          >
            {r.label}
          </Link>
        );
      })}
    </nav>
  );
}
