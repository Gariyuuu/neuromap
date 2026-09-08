export function Panel({
  title,
  subtitle,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel p-5 ${className}`}>
      {title && (
        <div className="mb-3">
          <h2 className="text-sm font-medium">{title}</h2>
          {subtitle && <p className="text-xs text-dim mt-0.5">{subtitle}</p>}
        </div>
      )}
      {children}
    </section>
  );
}

export function StatTile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="panel panel-2 p-4">
      <div className="text-xs text-dim uppercase tracking-wide">{label}</div>
      <div className="text-2xl font-mono mt-1">{value}</div>
      {sub && <div className="text-xs text-dimmer mt-1">{sub}</div>}
    </div>
  );
}

export function EmptyState({ what }: { what: string }) {
  return (
    <div className="panel p-8 text-center text-dim text-sm">
      <p>
        {what} not yet generated. Run the corresponding pipeline script
        (see <code className="font-mono text-dimmer">Makefile</code>) then re-export site data with{" "}
        <code className="font-mono text-dimmer">make site</code>.
      </p>
    </div>
  );
}

/** Screen-reader-only text description accompanying a chart — charts never rely on
 * color alone (project accessibility requirement). */
export function ChartDescription({ children }: { children: React.ReactNode }) {
  return <p className="sr-only">{children}</p>;
}

export function Pill({ children, color }: { children: React.ReactNode; color?: string }) {
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-mono panel-2 border border-hairline"
    >
      {color && <span className="w-2 h-2 rounded-full inline-block" style={{ background: color }} />}
      {children}
    </span>
  );
}
